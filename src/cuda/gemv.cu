#include "zeno_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>

#pragma pack(push, 1)

// Q8_0: 34 bytes (32 weights per block)
struct BlockQ8_0 {
    half d;
    int8_t qs[32];
};

// Q4_K: 144 bytes (256 weights per super-block)
struct BlockQ4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

// Q6_K: 210 bytes (256 weights per super-block)
struct BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

// Q5_0: 22 bytes (32 weights per block)
struct BlockQ5_0 {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};

#pragma pack(pop)

static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 size mismatch (expected 34)");
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K size mismatch (expected 144)");
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K size mismatch (expected 210)");
static_assert(sizeof(BlockQ5_0) == 22, "BlockQ5_0 size mismatch (expected 22)");

static inline int set_cuda_device(int device_id, int *orig_device) {
    if (device_id < 0) return ZENO_CUDA_ERR_INVALID_DEVICE;
    int count = 0;
    cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err != cudaSuccess) {
        return (count_err == cudaErrorNoDevice) ? ZENO_CUDA_ERR_NO_DEVICE : ZENO_CUDA_ERR_DRIVER_ERROR;
    }
    if (count == 0) return ZENO_CUDA_ERR_NO_DEVICE;
    if (device_id >= count) return ZENO_CUDA_ERR_INVALID_DEVICE;

    if (orig_device) {
        if (cudaGetDevice(orig_device) != cudaSuccess) {
            return ZENO_CUDA_ERR_GET_DEVICE_FAILED;
        }
    }
    if (cudaSetDevice(device_id) != cudaSuccess) {
        return ZENO_CUDA_ERR_SET_DEVICE_FAILED;
    }
    return ZENO_CUDA_SUCCESS;
}

static inline int restore_cuda_device(int orig_device, int op_status) {
    cudaError_t err = cudaSetDevice(orig_device);
    if (op_status != ZENO_CUDA_SUCCESS) {
        return op_status;
    }
    if (err != cudaSuccess) {
        return ZENO_CUDA_ERR_RESTORE_DEVICE_FAILED;
    }
    return ZENO_CUDA_SUCCESS;
}

static inline bool ranges_overlap(const void *p1, size_t n1, const void *p2, size_t n2) {
    if (!p1 || !p2 || n1 == 0 || n2 == 0) return false;
    uintptr_t s1 = (uintptr_t)p1;
    uintptr_t e1 = s1 + n1;
    uintptr_t s2 = (uintptr_t)p2;
    uintptr_t e2 = s2 + n2;
    return (s1 < e2 && s2 < e1);
}

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// GEMV Kernels (1 Warp per Output Row)
// ---------------------------------------------------------------------------

__global__ void gemv_q8_0_kernel(
    const BlockQ8_0 *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    int batch_idx = blockIdx.y;

    if (row >= out_dim) return;

    const float *batch_x = x + (size_t)batch_idx * in_dim;
    float *batch_out = out + (size_t)batch_idx * out_dim;
    const BlockQ8_0 *row_w = W + (size_t)row * blocks_per_row;

    float thread_sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const BlockQ8_0 *blk = row_w + b;
        float d = __half2float(blk->d);
        float q = (float)blk->qs[lane_id];
        float xv = batch_x[b * 32 + lane_id];
        thread_sum += d * (q * xv);
    }

    float row_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        batch_out[row] = row_sum;
    }
}

__global__ void gemv_q4k_kernel(
    const BlockQ4_K *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    int batch_idx = blockIdx.y;

    if (row >= out_dim) return;

    const float *batch_x = x + (size_t)batch_idx * in_dim;
    float *batch_out = out + (size_t)batch_idx * out_dim;
    const BlockQ4_K *row_w = W + (size_t)row * blocks_per_row;

    float thread_sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const BlockQ4_K *blk = row_w + b;
        float d = __half2float(blk->d);
        float min = __half2float(blk->dmin);
        size_t base_x = (size_t)b * 256;

        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            int is = chunk * 2;
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk->scales, &sc1, &m1);
            get_scale_min_k4(is + 1, blk->scales, &sc2, &m2);

            float d1 = d * (float)sc1;
            float m_val1 = min * (float)m1;
            float d2 = d * (float)sc2;
            float m_val2 = min * (float)m2;

            const uint8_t *q_slice = blk->qs + chunk * 32;
            uint8_t q_byte = q_slice[lane_id];

            float q1 = (float)(q_byte & 0x0F);
            float q2 = (float)(q_byte >> 4);

            float x1 = batch_x[base_x + chunk * 64 + lane_id];
            float x2 = batch_x[base_x + chunk * 64 + 32 + lane_id];

            float w1 = d1 * q1 - m_val1;
            float w2 = d2 * q2 - m_val2;

            thread_sum += w1 * x1 + w2 * x2;
        }
    }

    float row_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        batch_out[row] = row_sum;
    }
}

__global__ void gemv_q6k_kernel(
    const BlockQ6_K *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    int batch_idx = blockIdx.y;

    if (row >= out_dim) return;

    const float *batch_x = x + (size_t)batch_idx * in_dim;
    float *batch_out = out + (size_t)batch_idx * out_dim;
    const BlockQ6_K *row_w = W + (size_t)row * blocks_per_row;

    float thread_sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const BlockQ6_K *blk = row_w + b;
        float d = __half2float(blk->d);
        size_t base_x = (size_t)b * 256;
        int is = lane_id / 16; // 0 or 1

        #pragma unroll
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql = blk->ql + n * 64;
            const uint8_t *qh = blk->qh + n * 32;
            const int8_t  *sc = blk->scales + n * 8;

            float sc0 = d * (float)sc[is + 0];
            float sc2 = d * (float)sc[is + 2];
            float sc4 = d * (float)sc[is + 4];
            float sc6 = d * (float)sc[is + 6];

            int q1_i = (int)((ql[lane_id + 0]  & 0x0F) | (((qh[lane_id] >> 0) & 3) << 4)) - 32;
            int q2_i = (int)((ql[lane_id + 32] & 0x0F) | (((qh[lane_id] >> 2) & 3) << 4)) - 32;
            int q3_i = (int)((ql[lane_id + 0]  >> 4)   | (((qh[lane_id] >> 4) & 3) << 4)) - 32;
            int q4_i = (int)((ql[lane_id + 32] >> 4)   | (((qh[lane_id] >> 6) & 3) << 4)) - 32;

            float x1 = batch_x[base_x + n * 128 + lane_id + 0];
            float x2 = batch_x[base_x + n * 128 + lane_id + 32];
            float x3 = batch_x[base_x + n * 128 + lane_id + 64];
            float x4 = batch_x[base_x + n * 128 + lane_id + 96];

            thread_sum += (sc0 * (float)q1_i) * x1 +
                          (sc2 * (float)q2_i) * x2 +
                          (sc4 * (float)q3_i) * x3 +
                          (sc6 * (float)q4_i) * x4;
        }
    }

    float row_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        batch_out[row] = row_sum;
    }
}

__global__ void gemv_q5_0_kernel(
    const BlockQ5_0 *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    int batch_idx = blockIdx.y;

    if (row >= out_dim) return;

    const float *batch_x = x + (size_t)batch_idx * in_dim;
    float *batch_out = out + (size_t)batch_idx * out_dim;
    const BlockQ5_0 *row_w = W + (size_t)row * blocks_per_row;

    float thread_sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const BlockQ5_0 *blk = row_w + b;
        float d = 0.0f;
        uint32_t qh = 0;
        if (lane_id == 0) {
            d = __half2float(blk->d);
            uint16_t qh_lo = *(const uint16_t*)(blk->qh + 0);
            uint16_t qh_hi = *(const uint16_t*)(blk->qh + 2);
            qh = (uint32_t)qh_lo | ((uint32_t)qh_hi << 16);
        }
        d = __shfl_sync(0xffffffff, d, 0);
        qh = __shfl_sync(0xffffffff, qh, 0);
        uint16_t qs_pair = 0;
        if (lane_id < 8) {
            qs_pair = *(const uint16_t*)(blk->qs + lane_id * 2);
        }
        int sub_lane = (lane_id < 16) ? lane_id : (lane_id - 16);
        uint16_t fetched_pair = __shfl_sync(0xffffffff, qs_pair, sub_lane >> 1);
        uint8_t q_byte = (uint8_t)((sub_lane & 1) == 0 ? (fetched_pair & 0xFF) : (fetched_pair >> 8));

        size_t base_x = (size_t)b * 32;

        if (lane_id < 16) {
            uint8_t bit = (uint8_t)((qh >> lane_id) & 1);
            int q = (int)((q_byte & 0x0F) | (bit << 4)) - 16;
            float w = d * (float)q;
            float xv = batch_x[base_x + lane_id];
            thread_sum += w * xv;
        } else {
            int j = lane_id - 16;
            uint8_t bit = (uint8_t)((qh >> (j + 16)) & 1);
            int q = (int)((q_byte >> 4) | (bit << 4)) - 16;
            float w = d * (float)q;
            float xv = batch_x[base_x + lane_id];
            thread_sum += w * xv;
        }
    }

    float row_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        batch_out[row] = row_sum;
    }
}

// ---------------------------------------------------------------------------
// C ABI Host Functions
// ---------------------------------------------------------------------------

extern "C" {

int zeno_cuda_gemv_q8_0(
    const void *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
) {
    if (!W || !x || !out) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (out_dim <= 0 || in_dim <= 0 || batch_size <= 0 || batch_size > 65535) return ZENO_CUDA_ERR_INVALID_PARAM;
    if ((in_dim % 32) != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int blocks_per_row = in_dim / 32;
    size_t w_bytes = (size_t)out_dim * (size_t)blocks_per_row * sizeof(BlockQ8_0);
    size_t x_bytes = (size_t)batch_size * (size_t)in_dim * sizeof(float);
    size_t out_bytes = (size_t)batch_size * (size_t)out_dim * sizeof(float);

    if (ranges_overlap(x, x_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(W, w_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256; // 8 warps per threadblock
    unsigned int grid_x = (unsigned int)((out_dim + 7) / 8);
    dim3 grid(grid_x, (unsigned int)batch_size);
    cudaStream_t custream = (cudaStream_t)stream;

    gemv_q8_0_kernel<<<grid, block_size, 0, custream>>>(
        (const BlockQ8_0*)W,
        x,
        out,
        out_dim,
        in_dim,
        blocks_per_row
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_gemv_q4k(
    const void *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
) {
    if (!W || !x || !out) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (out_dim <= 0 || in_dim <= 0 || batch_size <= 0 || batch_size > 65535) return ZENO_CUDA_ERR_INVALID_PARAM;
    if ((in_dim % 256) != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int blocks_per_row = in_dim / 256;
    size_t w_bytes = (size_t)out_dim * (size_t)blocks_per_row * sizeof(BlockQ4_K);
    size_t x_bytes = (size_t)batch_size * (size_t)in_dim * sizeof(float);
    size_t out_bytes = (size_t)batch_size * (size_t)out_dim * sizeof(float);

    if (ranges_overlap(x, x_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(W, w_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((out_dim + 7) / 8);
    dim3 grid(grid_x, (unsigned int)batch_size);
    cudaStream_t custream = (cudaStream_t)stream;

    gemv_q4k_kernel<<<grid, block_size, 0, custream>>>(
        (const BlockQ4_K*)W,
        x,
        out,
        out_dim,
        in_dim,
        blocks_per_row
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_gemv_q6k(
    const void *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
) {
    if (!W || !x || !out) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (out_dim <= 0 || in_dim <= 0 || batch_size <= 0 || batch_size > 65535) return ZENO_CUDA_ERR_INVALID_PARAM;
    if ((in_dim % 256) != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int blocks_per_row = in_dim / 256;
    size_t w_bytes = (size_t)out_dim * (size_t)blocks_per_row * sizeof(BlockQ6_K);
    size_t x_bytes = (size_t)batch_size * (size_t)in_dim * sizeof(float);
    size_t out_bytes = (size_t)batch_size * (size_t)out_dim * sizeof(float);

    if (ranges_overlap(x, x_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(W, w_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((out_dim + 7) / 8);
    dim3 grid(grid_x, (unsigned int)batch_size);
    cudaStream_t custream = (cudaStream_t)stream;

    gemv_q6k_kernel<<<grid, block_size, 0, custream>>>(
        (const BlockQ6_K*)W,
        x,
        out,
        out_dim,
        in_dim,
        blocks_per_row
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_gemv_q5_0(
    const void *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
) {
    if (!W || !x || !out) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (out_dim <= 0 || in_dim <= 0 || batch_size <= 0 || batch_size > 65535) return ZENO_CUDA_ERR_INVALID_PARAM;
    if ((in_dim % 32) != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int blocks_per_row = in_dim / 32;
    size_t w_bytes = (size_t)out_dim * (size_t)blocks_per_row * sizeof(BlockQ5_0);
    size_t x_bytes = (size_t)batch_size * (size_t)in_dim * sizeof(float);
    size_t out_bytes = (size_t)batch_size * (size_t)out_dim * sizeof(float);

    if (ranges_overlap(x, x_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(W, w_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((out_dim + 7) / 8);
    dim3 grid(grid_x, (unsigned int)batch_size);
    cudaStream_t custream = (cudaStream_t)stream;

    gemv_q5_0_kernel<<<grid, block_size, 0, custream>>>(
        (const BlockQ5_0*)W,
        x,
        out,
        out_dim,
        in_dim,
        blocks_per_row
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

} // extern "C"

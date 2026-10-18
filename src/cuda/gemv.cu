#include "zeno_cuda.h"
#include "quant_types.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>

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

__global__ void gemv_q8_0_batched_4x_kernel(
    const BlockQ8_0 *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    int t_base = blockIdx.y * 4;

    if (row >= out_dim) return;

    int t0 = t_base + 0;
    int t1 = t_base + 1;
    int t2 = t_base + 2;
    int t3 = t_base + 3;

    const float *x0 = (t0 < batch_size) ? (x + (size_t)t0 * in_dim) : NULL;
    const float *x1 = (t1 < batch_size) ? (x + (size_t)t1 * in_dim) : NULL;
    const float *x2 = (t2 < batch_size) ? (x + (size_t)t2 * in_dim) : NULL;
    const float *x3 = (t3 < batch_size) ? (x + (size_t)t3 * in_dim) : NULL;

    const BlockQ8_0 *row_w = W + (size_t)row * blocks_per_row;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const BlockQ8_0 *blk = row_w + b;
        float d = __half2float(blk->d);
        float w = d * (float)blk->qs[lane_id];
        size_t off = (size_t)b * 32 + lane_id;

        if (x0) sum0 += w * x0[off];
        if (x1) sum1 += w * x1[off];
        if (x2) sum2 += w * x2[off];
        if (x3) sum3 += w * x3[off];
    }

    if (x0) {
        float r0 = warp_reduce_sum(sum0);
        if (lane_id == 0) out[(size_t)t0 * out_dim + row] = r0;
    }
    if (x1) {
        float r1 = warp_reduce_sum(sum1);
        if (lane_id == 0) out[(size_t)t1 * out_dim + row] = r1;
    }
    if (x2) {
        float r2 = warp_reduce_sum(sum2);
        if (lane_id == 0) out[(size_t)t2 * out_dim + row] = r2;
    }
    if (x3) {
        float r3 = warp_reduce_sum(sum3);
        if (lane_id == 0) out[(size_t)t3 * out_dim + row] = r3;
    }
}

__global__ void gemv_q8_0_batched_16x_kernel(
    const BlockQ8_0 *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    __shared__ alignas(16) float s_x[16][32];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int row = blockIdx.x * 8 + warp_id;
    int t_base = blockIdx.y * 16;
    bool active_row = (row < out_dim);

    const float *tok_x[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        int tidx = t_base + t;
        tok_x[t] = (tidx < batch_size) ? (x + (size_t)tidx * in_dim) : NULL;
    }

    const BlockQ8_0 *row_w = active_row ? (W + (size_t)row * blocks_per_row) : NULL;

    float sum[16] = {0.0f};
    for (int b = 0; b < blocks_per_row; b++) {
        size_t base_x = (size_t)b * 32;

        if (tid < 128) {
            int t = tid >> 3;
            int f_idx = (tid & 7) * 4;
            if (tok_x[t]) {
                *(float4*)&s_x[t][f_idx] = *(const float4*)&tok_x[t][base_x + f_idx];
            } else {
                *(float4*)&s_x[t][f_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        if (active_row) {
            const BlockQ8_0 *blk = row_w + b;
            float d = __half2float(blk->d);
            float w = d * (float)blk->qs[lane_id];

            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                if (tok_x[t]) sum[t] += w * s_x[t][lane_id];
            }
        }

        if (b + 1 < blocks_per_row) {
            __syncthreads();
        }
    }

    if (active_row) {
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (tok_x[t]) {
                float r = warp_reduce_sum(sum[t]);
                if (lane_id == 0) out[(size_t)(t_base + t) * out_dim + row] = r;
            }
        }
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
        size_t base_x = (size_t)b * 256;

        float my_d, my_m;
        precompute_q4k_scales(blk, lane_id, &my_d, &my_m);

        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            float d1 = __shfl_sync(0xffffffff, my_d, chunk * 2 + 0);
            float m_val1 = __shfl_sync(0xffffffff, my_m, chunk * 2 + 0);
            float d2 = __shfl_sync(0xffffffff, my_d, chunk * 2 + 1);
            float m_val2 = __shfl_sync(0xffffffff, my_m, chunk * 2 + 1);

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

__global__ void gemv_q4k_batched_4x_kernel(
    const BlockQ4_K *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    int t_base = blockIdx.y * 4;

    if (row >= out_dim) return;

    int t0 = t_base + 0;
    int t1 = t_base + 1;
    int t2 = t_base + 2;
    int t3 = t_base + 3;

    const float *x0 = (t0 < batch_size) ? (x + (size_t)t0 * in_dim) : NULL;
    const float *x1 = (t1 < batch_size) ? (x + (size_t)t1 * in_dim) : NULL;
    const float *x2 = (t2 < batch_size) ? (x + (size_t)t2 * in_dim) : NULL;
    const float *x3 = (t3 < batch_size) ? (x + (size_t)t3 * in_dim) : NULL;

    const BlockQ4_K *row_w = W + (size_t)row * blocks_per_row;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        const BlockQ4_K *blk = row_w + b;
        size_t base_x = (size_t)b * 256;

        float my_d, my_m;
        precompute_q4k_scales(blk, lane_id, &my_d, &my_m);

        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            float d1 = __shfl_sync(0xffffffff, my_d, chunk * 2 + 0);
            float m_val1 = __shfl_sync(0xffffffff, my_m, chunk * 2 + 0);
            float d2 = __shfl_sync(0xffffffff, my_d, chunk * 2 + 1);
            float m_val2 = __shfl_sync(0xffffffff, my_m, chunk * 2 + 1);

            const uint8_t *q_slice = blk->qs + chunk * 32;
            uint8_t q_byte = q_slice[lane_id];

            float q1 = (float)(q_byte & 0x0F);
            float q2 = (float)(q_byte >> 4);

            float w1 = d1 * q1 - m_val1;
            float w2 = d2 * q2 - m_val2;

            size_t off1 = base_x + chunk * 64 + lane_id;
            size_t off2 = base_x + chunk * 64 + 32 + lane_id;

            if (x0) sum0 += w1 * x0[off1] + w2 * x0[off2];
            if (x1) sum1 += w1 * x1[off1] + w2 * x1[off2];
            if (x2) sum2 += w1 * x2[off1] + w2 * x2[off2];
            if (x3) sum3 += w1 * x3[off1] + w2 * x3[off2];
        }
    }

    if (x0) {
        float r0 = warp_reduce_sum(sum0);
        if (lane_id == 0) out[(size_t)t0 * out_dim + row] = r0;
    }
    if (x1) {
        float r1 = warp_reduce_sum(sum1);
        if (lane_id == 0) out[(size_t)t1 * out_dim + row] = r1;
    }
    if (x2) {
        float r2 = warp_reduce_sum(sum2);
        if (lane_id == 0) out[(size_t)t2 * out_dim + row] = r2;
    }
    if (x3) {
        float r3 = warp_reduce_sum(sum3);
        if (lane_id == 0) out[(size_t)t3 * out_dim + row] = r3;
    }
}

__global__ void gemv_q4k_batched_16x_kernel(
    const BlockQ4_K *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    __shared__ float s_x[16][256];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int row0 = blockIdx.x * 16 + warp_id * 2;
    int row1 = row0 + 1;
    int t_base = blockIdx.y * 16;
    bool active_row0 = (row0 < out_dim);
    bool active_row1 = (row1 < out_dim);

    const float *tok_x[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        int tidx = t_base + t;
        tok_x[t] = (tidx < batch_size) ? (x + (size_t)tidx * in_dim) : NULL;
    }

    const BlockQ4_K *row_w0 = active_row0 ? (W + (size_t)row0 * blocks_per_row) : NULL;
    const BlockQ4_K *row_w1 = active_row1 ? (W + (size_t)row1 * blocks_per_row) : NULL;
    float sum0[16] = {0.0f};
    float sum1[16] = {0.0f};

    float4 *s_x4 = (float4*)s_x;

    for (int b = 0; b < blocks_per_row; b++) {
        size_t base_x = (size_t)b * 256;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx4 = tid + i * 256;
            int t = idx4 >> 6;
            int f_idx = (idx4 & 63) * 4;
            if (tok_x[t]) {
                s_x4[idx4] = *(const float4*)&tok_x[t][base_x + f_idx];
            } else {
                s_x4[idx4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        const BlockQ4_K *blk0 = active_row0 ? (row_w0 + b) : NULL;
        const BlockQ4_K *blk1 = active_row1 ? (row_w1 + b) : NULL;
        float my_d0 = 0.0f, my_m0 = 0.0f;
        float my_d1 = 0.0f, my_m1 = 0.0f;
        if (active_row0) precompute_q4k_scales(blk0, lane_id, &my_d0, &my_m0);
        if (active_row1) precompute_q4k_scales(blk1, lane_id, &my_d1, &my_m1);

        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            float d1_0 = __shfl_sync(0xffffffff, my_d0, chunk * 2 + 0);
            float m1_0 = __shfl_sync(0xffffffff, my_m0, chunk * 2 + 0);
            float d2_0 = __shfl_sync(0xffffffff, my_d0, chunk * 2 + 1);
            float m2_0 = __shfl_sync(0xffffffff, my_m0, chunk * 2 + 1);

            float d1_1 = __shfl_sync(0xffffffff, my_d1, chunk * 2 + 0);
            float m1_1 = __shfl_sync(0xffffffff, my_m1, chunk * 2 + 0);
            float d2_1 = __shfl_sync(0xffffffff, my_d1, chunk * 2 + 1);
            float m2_1 = __shfl_sync(0xffffffff, my_m1, chunk * 2 + 1);

            float w1_0 = 0.0f, w2_0 = 0.0f;
            if (active_row0) {
                uint8_t q_byte = blk0->qs[chunk * 32 + lane_id];
                w1_0 = d1_0 * (float)(q_byte & 0x0F) - m1_0;
                w2_0 = d2_0 * (float)(q_byte >> 4) - m2_0;
            }

            float w1_1 = 0.0f, w2_1 = 0.0f;
            if (active_row1) {
                uint8_t q_byte = blk1->qs[chunk * 32 + lane_id];
                w1_1 = d1_1 * (float)(q_byte & 0x0F) - m1_1;
                w2_1 = d2_1 * (float)(q_byte >> 4) - m2_1;
            }

            int off1 = chunk * 64 + lane_id;
            int off2 = chunk * 64 + 32 + lane_id;

            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                if (tok_x[t]) {
                    float x1 = s_x[t][off1];
                    float x2 = s_x[t][off2];
                    if (active_row0) sum0[t] += w1_0 * x1 + w2_0 * x2;
                    if (active_row1) sum1[t] += w1_1 * x1 + w2_1 * x2;
                }
            }
        }

        if (b + 1 < blocks_per_row) {
            __syncthreads();
        }
    }

    if (active_row0) {
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (tok_x[t]) {
                float r = warp_reduce_sum(sum0[t]);
                if (lane_id == 0) out[(size_t)(t_base + t) * out_dim + row0] = r;
            }
        }
    }

    if (active_row1) {
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (tok_x[t]) {
                float r = warp_reduce_sum(sum1[t]);
                if (lane_id == 0) out[(size_t)(t_base + t) * out_dim + row1] = r;
            }
        }
    }
}

__global__ void __launch_bounds__(256, 3) gemv_q4k_fused_qk_16x_kernel(
    const BlockQ4_K *W_q,
    const BlockQ4_K *W_k,
    const float *x,
    float *out_q,
    float *out_k,
    float *out_v,
    int q_dim,
    int k_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    __shared__ float s_x[16][256];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int row0 = blockIdx.x * 16 + warp_id * 2;
    int row1 = row0 + 1;
    int t_base = blockIdx.y * 16;
    int total_rows = q_dim + k_dim;
    bool active_row0 = (row0 < total_rows);
    bool active_row1 = (row1 < total_rows);

    const float *tok_x[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        int tidx = t_base + t;
        tok_x[t] = (tidx < batch_size) ? (x + (size_t)tidx * in_dim) : NULL;
    }

    const BlockQ4_K *row_w0 = NULL;
    if (active_row0) {
        if (row0 < q_dim) {
            row_w0 = W_q + (size_t)row0 * blocks_per_row;
        } else {
            row_w0 = W_k + (size_t)(row0 - q_dim) * blocks_per_row;
        }
    }

    const BlockQ4_K *row_w1 = NULL;
    if (active_row1) {
        if (row1 < q_dim) {
            row_w1 = W_q + (size_t)row1 * blocks_per_row;
        } else {
            row_w1 = W_k + (size_t)(row1 - q_dim) * blocks_per_row;
        }
    }

    float sum0[16] = {0.0f};
    float sum1[16] = {0.0f};

    float4 *s_x4 = (float4*)s_x;

    for (int b = 0; b < blocks_per_row; b++) {
        size_t base_x = (size_t)b * 256;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx4 = tid + i * 256;
            int t = idx4 >> 6;
            int f_idx = (idx4 & 63) * 4;
            if (tok_x[t]) {
                s_x4[idx4] = *(const float4*)&tok_x[t][base_x + f_idx];
            } else {
                s_x4[idx4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        const BlockQ4_K *blk0 = active_row0 ? (row_w0 + b) : NULL;
        const BlockQ4_K *blk1 = active_row1 ? (row_w1 + b) : NULL;
        float my_d0 = 0.0f, my_m0 = 0.0f;
        float my_d1 = 0.0f, my_m1 = 0.0f;
        if (active_row0) precompute_q4k_scales(blk0, lane_id, &my_d0, &my_m0);
        if (active_row1) precompute_q4k_scales(blk1, lane_id, &my_d1, &my_m1);

        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            float d1_0 = __shfl_sync(0xffffffff, my_d0, chunk * 2 + 0);
            float m1_0 = __shfl_sync(0xffffffff, my_m0, chunk * 2 + 0);
            float d2_0 = __shfl_sync(0xffffffff, my_d0, chunk * 2 + 1);
            float m2_0 = __shfl_sync(0xffffffff, my_m0, chunk * 2 + 1);

            float d1_1 = __shfl_sync(0xffffffff, my_d1, chunk * 2 + 0);
            float m1_1 = __shfl_sync(0xffffffff, my_m1, chunk * 2 + 0);
            float d2_1 = __shfl_sync(0xffffffff, my_d1, chunk * 2 + 1);
            float m2_1 = __shfl_sync(0xffffffff, my_m1, chunk * 2 + 1);

            float w1_0 = 0.0f, w2_0 = 0.0f;
            if (active_row0) {
                uint8_t q_byte = blk0->qs[chunk * 32 + lane_id];
                w1_0 = d1_0 * (float)(q_byte & 0x0F) - m1_0;
                w2_0 = d2_0 * (float)(q_byte >> 4) - m2_0;
            }

            float w1_1 = 0.0f, w2_1 = 0.0f;
            if (active_row1) {
                uint8_t q_byte = blk1->qs[chunk * 32 + lane_id];
                w1_1 = d1_1 * (float)(q_byte & 0x0F) - m1_1;
                w2_1 = d2_1 * (float)(q_byte >> 4) - m2_1;
            }

            int off1 = chunk * 64 + lane_id;
            int off2 = chunk * 64 + 32 + lane_id;

            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                if (tok_x[t]) {
                    float x1 = s_x[t][off1];
                    float x2 = s_x[t][off2];
                    if (active_row0) sum0[t] += w1_0 * x1 + w2_0 * x2;
                    if (active_row1) sum1[t] += w1_1 * x1 + w2_1 * x2;
                }
            }
        }

        if (b + 1 < blocks_per_row) {
            __syncthreads();
        }
    }

    if (active_row0) {
        float *dst0 = (row0 < q_dim) ? out_q : out_k;
        int d_row0 = (row0 < q_dim) ? row0 : (row0 - q_dim);
        int d_dim0 = (row0 < q_dim) ? q_dim : k_dim;
        bool write_v0 = (row0 >= q_dim && out_v != NULL);

        for (int t = 0; t < 16; ++t) {
            if (tok_x[t]) {
                float r = warp_reduce_sum(sum0[t]);
                if (lane_id == 0) {
                    dst0[(size_t)(t_base + t) * d_dim0 + d_row0] = r;
                    if (write_v0) {
                        out_v[(size_t)(t_base + t) * k_dim + d_row0] = r;
                    }
                }
            }
        }
    }

    if (active_row1) {
        float *dst1 = (row1 < q_dim) ? out_q : out_k;
        int d_row1 = (row1 < q_dim) ? row1 : (row1 - q_dim);
        int d_dim1 = (row1 < q_dim) ? q_dim : k_dim;
        bool write_v1 = (row1 >= q_dim && out_v != NULL);

        for (int t = 0; t < 16; ++t) {
            if (tok_x[t]) {
                float r = warp_reduce_sum(sum1[t]);
                if (lane_id == 0) {
                    dst1[(size_t)(t_base + t) * d_dim1 + d_row1] = r;
                    if (write_v1) {
                        out_v[(size_t)(t_base + t) * k_dim + d_row1] = r;
                    }
                }
            }
        }
    }
}

__global__ void dense_gemv_gate_up_geglu_q4k_kernel(
    const BlockQ4_K *__restrict__ W_gate,
    const BlockQ4_K *__restrict__ W_up,
    const float *__restrict__ x,
    float *__restrict__ out_geglu,
    int out_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    __shared__ float s_x[16][256];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int row = blockIdx.x * 8 + warp_id;
    int t_base = blockIdx.y * 16;
    bool active_row = (row < out_dim);

    const float *tok_x[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        int tidx = t_base + t;
        tok_x[t] = (tidx < batch_size) ? (x + (size_t)tidx * in_dim) : NULL;
    }

    const BlockQ4_K *gate_row_w = active_row ? (W_gate + (size_t)row * blocks_per_row) : NULL;
    const BlockQ4_K *up_row_w   = active_row ? (W_up   + (size_t)row * blocks_per_row) : NULL;

    float gate_sum[16] = {0.0f};
    float up_sum[16]   = {0.0f};

    float4 *s_x4 = (float4*)s_x;

    for (int b = 0; b < blocks_per_row; b++) {
        size_t base_x = (size_t)b * 256;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx4 = tid + i * 256;
            int t = idx4 >> 6;
            int f_idx = (idx4 & 63) * 4;
            if (tok_x[t]) {
                s_x4[idx4] = *(const float4*)&tok_x[t][base_x + f_idx];
            } else {
                s_x4[idx4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        if (active_row) {
            const BlockQ4_K *gate_blk = gate_row_w + b;
            const BlockQ4_K *up_blk   = up_row_w   + b;

            float gate_my_d, gate_my_m;
            precompute_q4k_scales(gate_blk, lane_id, &gate_my_d, &gate_my_m);

            float up_my_d, up_my_m;
            precompute_q4k_scales(up_blk, lane_id, &up_my_d, &up_my_m);

            #pragma unroll
            for (int chunk = 0; chunk < 4; chunk++) {
                float g_d1 = __shfl_sync(0xffffffff, gate_my_d, chunk * 2 + 0);
                float g_m1 = __shfl_sync(0xffffffff, gate_my_m, chunk * 2 + 0);
                float g_d2 = __shfl_sync(0xffffffff, gate_my_d, chunk * 2 + 1);
                float g_m2 = __shfl_sync(0xffffffff, gate_my_m, chunk * 2 + 1);

                const uint8_t *g_q_slice = gate_blk->qs + chunk * 32;
                uint8_t g_q_byte = g_q_slice[lane_id];
                float g_w1 = g_d1 * (float)(g_q_byte & 0x0F) - g_m1;
                float g_w2 = g_d2 * (float)(g_q_byte >> 4)   - g_m2;

                float u_d1 = __shfl_sync(0xffffffff, up_my_d, chunk * 2 + 0);
                float u_m1 = __shfl_sync(0xffffffff, up_my_m, chunk * 2 + 0);
                float u_d2 = __shfl_sync(0xffffffff, up_my_d, chunk * 2 + 1);
                float u_m2 = __shfl_sync(0xffffffff, up_my_m, chunk * 2 + 1);

                const uint8_t *u_q_slice = up_blk->qs + chunk * 32;
                uint8_t u_q_byte = u_q_slice[lane_id];
                float u_w1 = u_d1 * (float)(u_q_byte & 0x0F) - u_m1;
                float u_w2 = u_d2 * (float)(u_q_byte >> 4)   - u_m2;

                int off1 = chunk * 64 + lane_id;
                int off2 = chunk * 64 + 32 + lane_id;

                #pragma unroll
                for (int t = 0; t < 16; ++t) {
                    if (tok_x[t]) {
                        float x1 = s_x[t][off1];
                        float x2 = s_x[t][off2];
                        gate_sum[t] += g_w1 * x1 + g_w2 * x2;
                        up_sum[t]   += u_w1 * x1 + u_w2 * x2;
                    }
                }
            }
        }

        if (b + 1 < blocks_per_row) {
            __syncthreads();
        }
    }

    if (active_row) {
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (tok_x[t]) {
                float g = warp_reduce_sum(gate_sum[t]);
                float u = warp_reduce_sum(up_sum[t]);
                if (lane_id == 0) {
                    out_geglu[(size_t)(t_base + t) * out_dim + row] = gelu_tanh_scalar(g) * u;
                }
            }
        }
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

__global__ void gemv_q6k_batched_4x_kernel(
    const BlockQ6_K *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    int t_base = blockIdx.y * 4;

    if (row >= out_dim) return;

    int t0 = t_base + 0;
    int t1 = t_base + 1;
    int t2 = t_base + 2;
    int t3 = t_base + 3;

    const float *x0 = (t0 < batch_size) ? (x + (size_t)t0 * in_dim) : NULL;
    const float *x1 = (t1 < batch_size) ? (x + (size_t)t1 * in_dim) : NULL;
    const float *x2 = (t2 < batch_size) ? (x + (size_t)t2 * in_dim) : NULL;
    const float *x3 = (t3 < batch_size) ? (x + (size_t)t3 * in_dim) : NULL;

    const BlockQ6_K *row_w = W + (size_t)row * blocks_per_row;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
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

            float w1 = sc0 * (float)q1_i;
            float w2 = sc2 * (float)q2_i;
            float w3 = sc4 * (float)q3_i;
            float w4 = sc6 * (float)q4_i;

            size_t off1 = base_x + n * 128 + lane_id + 0;
            size_t off2 = base_x + n * 128 + lane_id + 32;
            size_t off3 = base_x + n * 128 + lane_id + 64;
            size_t off4 = base_x + n * 128 + lane_id + 96;

            if (x0) sum0 += w1 * x0[off1] + w2 * x0[off2] + w3 * x0[off3] + w4 * x0[off4];
            if (x1) sum1 += w1 * x1[off1] + w2 * x1[off2] + w3 * x1[off3] + w4 * x1[off4];
            if (x2) sum2 += w1 * x2[off1] + w2 * x2[off2] + w3 * x2[off3] + w4 * x2[off4];
            if (x3) sum3 += w1 * x3[off1] + w2 * x3[off2] + w3 * x3[off3] + w4 * x3[off4];
        }
    }

    if (x0) {
        float r0 = warp_reduce_sum(sum0);
        if (lane_id == 0) out[(size_t)t0 * out_dim + row] = r0;
    }
    if (x1) {
        float r1 = warp_reduce_sum(sum1);
        if (lane_id == 0) out[(size_t)t1 * out_dim + row] = r1;
    }
    if (x2) {
        float r2 = warp_reduce_sum(sum2);
        if (lane_id == 0) out[(size_t)t2 * out_dim + row] = r2;
    }
    if (x3) {
        float r3 = warp_reduce_sum(sum3);
        if (lane_id == 0) out[(size_t)t3 * out_dim + row] = r3;
    }
}

__global__ void gemv_q6k_batched_16x_kernel(
    const BlockQ6_K *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    __shared__ float s_x[16][256];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int row = blockIdx.x * 8 + warp_id;
    int t_base = blockIdx.y * 16;
    bool active_row = (row < out_dim);

    const float *tok_x[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        int tidx = t_base + t;
        tok_x[t] = (tidx < batch_size) ? (x + (size_t)tidx * in_dim) : NULL;
    }

    const BlockQ6_K *row_w = active_row ? (W + (size_t)row * blocks_per_row) : NULL;

    float sum[16] = {0.0f};
    float4 *s_x4 = (float4*)s_x;

    for (int b = 0; b < blocks_per_row; b++) {
        size_t base_x = (size_t)b * 256;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx4 = tid + i * 256;
            int t = idx4 >> 6;
            int f_idx = (idx4 & 63) * 4;
            if (tok_x[t]) {
                s_x4[idx4] = *(const float4*)&tok_x[t][base_x + f_idx];
            } else {
                s_x4[idx4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        if (active_row) {
            const BlockQ6_K *blk = row_w + b;
            float d = __half2float(blk->d);
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

                float w1 = sc0 * (float)q1_i;
                float w2 = sc2 * (float)q2_i;
                float w3 = sc4 * (float)q3_i;
                float w4 = sc6 * (float)q4_i;

                int off1 = n * 128 + lane_id + 0;
                int off2 = n * 128 + lane_id + 32;
                int off3 = n * 128 + lane_id + 64;
                int off4 = n * 128 + lane_id + 96;

                #pragma unroll
                for (int t = 0; t < 16; ++t) {
                    if (tok_x[t]) {
                        sum[t] += w1 * s_x[t][off1] + w2 * s_x[t][off2] + w3 * s_x[t][off3] + w4 * s_x[t][off4];
                    }
                }
            }
        }

        if (b + 1 < blocks_per_row) {
            __syncthreads();
        }
    }

    if (active_row) {
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (tok_x[t]) {
                float r = warp_reduce_sum(sum[t]);
                if (lane_id == 0) out[(size_t)(t_base + t) * out_dim + row] = r;
            }
        }
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

__global__ void gemv_q5_0_batched_4x_kernel(
    const BlockQ5_0 *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    int t_base = blockIdx.y * 4;

    if (row >= out_dim) return;

    int t0 = t_base + 0;
    int t1 = t_base + 1;
    int t2 = t_base + 2;
    int t3 = t_base + 3;

    const float *x0 = (t0 < batch_size) ? (x + (size_t)t0 * in_dim) : NULL;
    const float *x1 = (t1 < batch_size) ? (x + (size_t)t1 * in_dim) : NULL;
    const float *x2 = (t2 < batch_size) ? (x + (size_t)t2 * in_dim) : NULL;
    const float *x3 = (t3 < batch_size) ? (x + (size_t)t3 * in_dim) : NULL;

    const BlockQ5_0 *row_w = W + (size_t)row * blocks_per_row;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
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

        float w;
        if (lane_id < 16) {
            uint8_t bit = (uint8_t)((qh >> lane_id) & 1);
            int q = (int)((q_byte & 0x0F) | (bit << 4)) - 16;
            w = d * (float)q;
        } else {
            int j = lane_id - 16;
            uint8_t bit = (uint8_t)((qh >> (j + 16)) & 1);
            int q = (int)((q_byte >> 4) | (bit << 4)) - 16;
            w = d * (float)q;
        }

        size_t off = base_x + lane_id;
        if (x0) sum0 += w * x0[off];
        if (x1) sum1 += w * x1[off];
        if (x2) sum2 += w * x2[off];
        if (x3) sum3 += w * x3[off];
    }

    if (x0) {
        float r0 = warp_reduce_sum(sum0);
        if (lane_id == 0) out[(size_t)t0 * out_dim + row] = r0;
    }
    if (x1) {
        float r1 = warp_reduce_sum(sum1);
        if (lane_id == 0) out[(size_t)t1 * out_dim + row] = r1;
    }
    if (x2) {
        float r2 = warp_reduce_sum(sum2);
        if (lane_id == 0) out[(size_t)t2 * out_dim + row] = r2;
    }
    if (x3) {
        float r3 = warp_reduce_sum(sum3);
        if (lane_id == 0) out[(size_t)t3 * out_dim + row] = r3;
    }
}

__global__ void gemv_q5_0_batched_16x_kernel(
    const BlockQ5_0 *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int blocks_per_row,
    int batch_size
) {
    __shared__ alignas(16) float s_x[16][32];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int row = blockIdx.x * 8 + warp_id;
    int t_base = blockIdx.y * 16;
    bool active_row = (row < out_dim);

    const float *tok_x[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        int tidx = t_base + t;
        tok_x[t] = (tidx < batch_size) ? (x + (size_t)tidx * in_dim) : NULL;
    }

    const BlockQ5_0 *row_w = active_row ? (W + (size_t)row * blocks_per_row) : NULL;
    float sum[16] = {0.0f};

    for (int b = 0; b < blocks_per_row; b++) {
        size_t base_x = (size_t)b * 32;

        if (tid < 128) {
            int t = tid >> 3;
            int f_idx = (tid & 7) * 4;
            if (tok_x[t]) {
                *(float4*)&s_x[t][f_idx] = *(const float4*)&tok_x[t][base_x + f_idx];
            } else {
                *(float4*)&s_x[t][f_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        if (active_row) {
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

            float w;
            if (lane_id < 16) {
                uint8_t bit = (uint8_t)((qh >> lane_id) & 1);
                int q = (int)((q_byte & 0x0F) | (bit << 4)) - 16;
                w = d * (float)q;
            } else {
                int j = lane_id - 16;
                uint8_t bit = (uint8_t)((qh >> (j + 16)) & 1);
                int q = (int)((q_byte >> 4) | (bit << 4)) - 16;
                w = d * (float)q;
            }

            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                if (tok_x[t]) {
                    float xv = s_x[t][lane_id];
                    sum[t] += w * xv;
                }
            }
        }

        if (b + 1 < blocks_per_row) {
            __syncthreads();
        }
    }

    if (active_row) {
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (tok_x[t]) {
                float r = warp_reduce_sum(sum[t]);
                if (lane_id == 0) out[(size_t)(t_base + t) * out_dim + row] = r;
            }
        }
    }
}

__global__ void gemv_f32_kernel(
    const float *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    int batch_idx = blockIdx.y;

    if (row >= out_dim) return;

    const float *batch_x = x + (size_t)batch_idx * in_dim;
    float *batch_out = out + (size_t)batch_idx * out_dim;
    const float *row_w = W + (size_t)row * in_dim;

    float thread_sum = 0.0f;
    if ((in_dim % 4 == 0) && (((uintptr_t)row_w & 0xF) == 0) && (((uintptr_t)batch_x & 0xF) == 0)) {
        const float4 *row_w4 = (const float4 *)row_w;
        const float4 *batch_x4 = (const float4 *)batch_x;
        int in_dim4 = in_dim / 4;
        for (int i = lane_id; i < in_dim4; i += 32) {
            float4 w4 = row_w4[i];
            float4 x4 = batch_x4[i];
            thread_sum += w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w;
        }
    } else {
        for (int col = lane_id; col < in_dim; col += 32) {
            thread_sum += row_w[col] * batch_x[col];
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
    cudaStream_t custream = (cudaStream_t)stream;

    if (batch_size == 1) {
        dim3 grid(grid_x, 1);
        gemv_q8_0_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ8_0*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row
        );
    } else if (batch_size >= 16 && (batch_size % 16 == 0)) {
        unsigned int grid_y = (unsigned int)(batch_size / 16);
        dim3 grid(grid_x, grid_y);
        gemv_q8_0_batched_16x_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ8_0*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row,
            batch_size
        );
    } else {
        unsigned int grid_y = (unsigned int)((batch_size + 3) / 4);
        dim3 grid(grid_x, grid_y);
        gemv_q8_0_batched_4x_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ8_0*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row,
            batch_size
        );
    }

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
    cudaStream_t custream = (cudaStream_t)stream;

    if (batch_size == 1) {
        dim3 grid(grid_x, 1);
        gemv_q4k_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ4_K*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row
        );
    } else if (batch_size >= 16 && (batch_size % 16 == 0)) {
        unsigned int grid_x_16x = (unsigned int)((out_dim + 15) / 16);
        unsigned int grid_y = (unsigned int)(batch_size / 16);
        dim3 grid(grid_x_16x, grid_y);
        gemv_q4k_batched_16x_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ4_K*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row,
            batch_size
        );
    } else {
        unsigned int grid_y = (unsigned int)((batch_size + 3) / 4);
        dim3 grid(grid_x, grid_y);
        gemv_q4k_batched_4x_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ4_K*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row,
            batch_size
        );
    }

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_gemv_q4k_fused_qk(
    const void *W_q,
    const void *W_k,
    const float *x,
    float *out_q,
    float *out_k,
    float *out_v,
    int q_dim,
    int k_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
) {
    if (!W_q || !W_k || !x || !out_q || !out_k) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (q_dim <= 0 || k_dim <= 0 || in_dim <= 0 || batch_size <= 0 || batch_size > 65535) return ZENO_CUDA_ERR_INVALID_PARAM;
    if ((in_dim % 256) != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int blocks_per_row = in_dim / 256;
    size_t w_q_bytes = (size_t)q_dim * (size_t)blocks_per_row * sizeof(BlockQ4_K);
    size_t w_k_bytes = (size_t)k_dim * (size_t)blocks_per_row * sizeof(BlockQ4_K);
    size_t x_bytes = (size_t)batch_size * (size_t)in_dim * sizeof(float);
    size_t out_q_bytes = (size_t)batch_size * (size_t)q_dim * sizeof(float);
    size_t out_k_bytes = (size_t)batch_size * (size_t)k_dim * sizeof(float);

    if (ranges_overlap(x, x_bytes, out_q, out_q_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(x, x_bytes, out_k, out_k_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(W_q, w_q_bytes, out_q, out_q_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(W_k, w_k_bytes, out_k, out_k_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(out_q, out_q_bytes, out_k, out_k_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    if (out_v) {
        size_t out_v_bytes = (size_t)batch_size * (size_t)k_dim * sizeof(float);
        if (ranges_overlap(x, x_bytes, out_v, out_v_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
        if (ranges_overlap(W_k, w_k_bytes, out_v, out_v_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
        if (ranges_overlap(out_q, out_q_bytes, out_v, out_v_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
        if (ranges_overlap(out_k, out_k_bytes, out_v, out_v_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    }

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaStream_t custream = (cudaStream_t)stream;

    if (batch_size >= 16 && (batch_size % 16 == 0)) {
        const int block_size = 256;
        int total_rows = q_dim + k_dim;
        unsigned int grid_x = (unsigned int)((total_rows + 15) / 16);
        unsigned int grid_y = (unsigned int)(batch_size / 16);
        dim3 grid(grid_x, grid_y);
        gemv_q4k_fused_qk_16x_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ4_K*)W_q,
            (const BlockQ4_K*)W_k,
            x,
            out_q,
            out_k,
            out_v,
            q_dim,
            k_dim,
            in_dim,
            blocks_per_row,
            batch_size
        );
    } else {
        int rc_q = zeno_cuda_gemv_q4k(W_q, x, out_q, q_dim, in_dim, batch_size, device_id, stream);
        if (rc_q != ZENO_CUDA_SUCCESS) return restore_cuda_device(orig, rc_q);
        int rc_k = zeno_cuda_gemv_q4k(W_k, x, out_k, k_dim, in_dim, batch_size, device_id, stream);
        if (rc_k != ZENO_CUDA_SUCCESS) return restore_cuda_device(orig, rc_k);
        if (out_v != NULL) {
            size_t kv_bytes = (size_t)batch_size * (size_t)k_dim * sizeof(float);
            cudaError_t cpy_err = cudaMemcpyAsync(out_v, out_k, kv_bytes, cudaMemcpyDeviceToDevice, custream);
            if (cpy_err != cudaSuccess) return restore_cuda_device(orig, ZENO_CUDA_ERR_COPY_FAILED);
        }
    }

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_dense_gate_up_geglu_q4k(
    const void *W_gate,
    const void *W_up,
    const float *x,
    float *out_geglu,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
) {
    if (!W_gate || !W_up || !x || !out_geglu) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (out_dim <= 0 || in_dim <= 0 || batch_size <= 0 || batch_size > 65535) return ZENO_CUDA_ERR_INVALID_PARAM;
    if ((in_dim % 256) != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int blocks_per_row = in_dim / 256;
    size_t w_bytes = (size_t)out_dim * (size_t)blocks_per_row * sizeof(BlockQ4_K);
    size_t x_bytes = (size_t)batch_size * (size_t)in_dim * sizeof(float);
    size_t out_bytes = (size_t)batch_size * (size_t)out_dim * sizeof(float);

    if (ranges_overlap(x, x_bytes, out_geglu, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(W_gate, w_bytes, out_geglu, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(W_up, w_bytes, out_geglu, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((out_dim + 7) / 8);
    unsigned int grid_y = (unsigned int)((batch_size + 15) / 16);
    dim3 grid(grid_x, grid_y);
    cudaStream_t custream = (cudaStream_t)stream;

    dense_gemv_gate_up_geglu_q4k_kernel<<<grid, block_size, 0, custream>>>(
        (const BlockQ4_K*)W_gate,
        (const BlockQ4_K*)W_up,
        x,
        out_geglu,
        out_dim,
        in_dim,
        blocks_per_row,
        batch_size
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
    cudaStream_t custream = (cudaStream_t)stream;

    if (batch_size == 1) {
        dim3 grid(grid_x, 1);
        gemv_q6k_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ6_K*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row
        );
    } else if (batch_size >= 16 && (batch_size % 16 == 0)) {
        unsigned int grid_y = (unsigned int)(batch_size / 16);
        dim3 grid(grid_x, grid_y);
        gemv_q6k_batched_16x_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ6_K*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row,
            batch_size
        );
    } else {
        unsigned int grid_y = (unsigned int)((batch_size + 3) / 4);
        dim3 grid(grid_x, grid_y);
        gemv_q6k_batched_4x_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ6_K*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row,
            batch_size
        );
    }

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
    cudaStream_t custream = (cudaStream_t)stream;

    if (batch_size == 1) {
        dim3 grid(grid_x, 1);
        gemv_q5_0_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ5_0*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row
        );
    } else if (batch_size >= 16 && (batch_size % 16 == 0)) {
        unsigned int grid_y = (unsigned int)(batch_size / 16);
        dim3 grid(grid_x, grid_y);
        gemv_q5_0_batched_16x_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ5_0*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row,
            batch_size
        );
    } else {
        unsigned int grid_y = (unsigned int)((batch_size + 3) / 4);
        dim3 grid(grid_x, grid_y);
        gemv_q5_0_batched_4x_kernel<<<grid, block_size, 0, custream>>>(
            (const BlockQ5_0*)W,
            x,
            out,
            out_dim,
            in_dim,
            blocks_per_row,
            batch_size
        );
    }

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_gemv_f32(
    const float *W,
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

    size_t w_bytes = (size_t)out_dim * (size_t)in_dim * sizeof(float);
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

    gemv_f32_kernel<<<grid, block_size, 0, custream>>>(
        W,
        x,
        out,
        out_dim,
        in_dim
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

} // extern "C"

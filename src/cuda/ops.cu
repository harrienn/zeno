#include "zeno_cuda.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

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
        // Operation failed: preserve primary error
        return op_status;
    }
    // Operation succeeded: check restoration error
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

static inline bool partial_overlap(const void *p1, size_t n1, const void *p2, size_t n2) {
    if (p1 == p2) return false; // Exact alias is permitted
    return ranges_overlap(p1, n1, p2, n2);
}

static inline int next_power_of_two(int n) {
    if (n <= 1) return 1;
    int p = 1;
    while (p < n && p < 1073741824) p <<= 1;
    return p;
}

static inline int calc_rms_norm_block_size(int dim) {
    int p = next_power_of_two(dim);
    if (p < 32) p = 32;
    if (p > 512) p = 512;
    return p;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void rms_norm_kernel(
    const float *x,
    const float *weight,
    float *out,
    int num_rows,
    int dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    const float *row_x = x + (size_t)row * (size_t)dim;
    float *row_out = out + (size_t)row * (size_t)dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = row_x[i];
        sum_sq += v * v;
    }

    // Intra-warp reduction with full mask 0xffffffff (inactive lanes contributed 0.0f)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_warp_sums[32];
    __shared__ float s_inv_rms;

    if (lane_id == 0) {
        s_warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane_id == 0) {
            s_inv_rms = rsqrtf(block_sum / (float)dim + eps);
        }
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = row_x[i] * inv_rms;
        if (weight) {
            val *= weight[i];
        }
        row_out[i] = val;
    }
}

__global__ void geglu_kernel(
    const float *gate,
    const float *up,
    float *out,
    size_t count
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float g = gate[idx];
        float u = up[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float inner = sqrt_2_over_pi * (g + 0.044715f * g * g * g);
        float gelu = 0.5f * g * (1.0f + tanhf(inner));
        out[idx] = gelu * u;
    }
}

__global__ void rope_kernel(
    float *vec,
    int num_heads,
    int head_dim,
    int pos_offset,
    float theta,
    const float *freq_factors
) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int i = threadIdx.x;
    int half = head_dim / 2;

    float *head_vec = vec + ((size_t)t * (size_t)num_heads + (size_t)h) * (size_t)head_dim;
    float pos_f = (float)(pos_offset + t);
    float exp_val = 2.0f * (float)i / (float)head_dim;
    float ff = freq_factors ? freq_factors[i] : 1.0f;
    float freq = (1.0f / powf(theta, exp_val)) / ff;
    float angle = pos_f * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float v0 = head_vec[i];
    float v1 = head_vec[half + i];
    head_vec[i] = v0 * cos_val - v1 * sin_val;
    head_vec[half + i] = v0 * sin_val + v1 * cos_val;
}

__global__ void add_residual_kernel(
    float *x,
    const float *residual,
    size_t count
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        x[idx] += residual[idx];
    }
}

__global__ void scale_kernel(
    float *x,
    float scale,
    size_t count
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        x[idx] *= scale;
    }
}

__global__ void layer_output_kernel(
    float *out,
    const float *x_mid,
    const float *post_ffn,
    int prompt_len,
    int num_tokens,
    int dim,
    float enc_scale,
    float layer_scale
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)num_tokens * dim;
    if (idx < total) {
        int token_idx = (int)(idx / dim);
        float s = (token_idx < prompt_len) ? enc_scale : layer_scale;
        out[idx] = (x_mid[idx] + post_ffn[idx]) * s;
    }
}

// ---------------------------------------------------------------------------
// C ABI Implementations
// ---------------------------------------------------------------------------

extern "C" {

int zeno_cuda_malloc(void **out_ptr, size_t bytes, int device_id) {
    if (!out_ptr) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (bytes == 0) {
        *out_ptr = NULL;
        return ZENO_CUDA_SUCCESS;
    }
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaMalloc(out_ptr, bytes);
    int op_rc = ZENO_CUDA_SUCCESS;
    if (err == cudaErrorMemoryAllocation) op_rc = ZENO_CUDA_ERR_OUT_OF_MEMORY;
    else if (err != cudaSuccess) op_rc = ZENO_CUDA_ERR_DRIVER_ERROR;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_free(void *ptr, int device_id) {
    if (!ptr) return ZENO_CUDA_SUCCESS;
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaFree(ptr);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_DRIVER_ERROR : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes, int device_id) {
    if (bytes == 0) return ZENO_CUDA_SUCCESS;
    if (!dst || !src) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_COPY_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, int device_id) {
    if (bytes == 0) return ZENO_CUDA_SUCCESS;
    if (!dst || !src) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_COPY_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_memcpy_d2d(void *dst, const void *src, size_t bytes, int device_id, void *stream) {
    if (bytes == 0) return ZENO_CUDA_SUCCESS;
    if (!dst || !src) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaStream_t custream = (cudaStream_t)stream;
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, custream);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_COPY_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_memset(void *dst, int value, size_t bytes, int device_id) {
    if (bytes == 0) return ZENO_CUDA_SUCCESS;
    if (!dst) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaMemset(dst, value, bytes);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_DRIVER_ERROR : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_stream_create(void **out_stream, int device_id) {
    if (!out_stream) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaStream_t s = NULL;
    cudaError_t err = cudaStreamCreate(&s);
    int op_rc = ZENO_CUDA_SUCCESS;
    if (err != cudaSuccess) {
        op_rc = ZENO_CUDA_ERR_STREAM_FAILED;
    } else {
        *out_stream = (void *)s;
    }

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_stream_destroy(void *stream, int device_id) {
    if (!stream) return ZENO_CUDA_SUCCESS;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_STREAM_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_stream_sync(void *stream, int device_id) {
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = stream ? cudaStreamSynchronize((cudaStream_t)stream) : cudaStreamSynchronize(0);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_STREAM_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_graph_begin_capture(void *stream, int device_id) {
    if (!stream) return ZENO_CUDA_ERR_INVALID_PARAM;
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaStreamBeginCapture((cudaStream_t)stream, cudaStreamCaptureModeGlobal);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_STREAM_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_graph_end_capture(void *stream, void **out_graph_exec, int device_id) {
    if (!stream || !out_graph_exec) return ZENO_CUDA_ERR_INVALID_PARAM;
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture((cudaStream_t)stream, &graph);
    if (err != cudaSuccess || !graph) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED);
    }

    cudaGraphExec_t graph_exec = NULL;
    err = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess || !graph_exec) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED);
    }

    *out_graph_exec = (void*)graph_exec;
    return restore_cuda_device(orig, ZENO_CUDA_SUCCESS);
}

int zeno_cuda_graph_launch(void *graph_exec, void *stream, int device_id) {
    if (!graph_exec || !stream) return ZENO_CUDA_ERR_INVALID_PARAM;
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaGraphLaunch((cudaGraphExec_t)graph_exec, (cudaStream_t)stream);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_graph_destroy(void *graph_exec, int device_id) {
    if (!graph_exec) return ZENO_CUDA_SUCCESS;
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaGraphExecDestroy((cudaGraphExec_t)graph_exec);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_DRIVER_ERROR : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_rms_norm(
    const float *x,
    const float *weight,
    float *out,
    int num_rows,
    int dim,
    float eps,
    int device_id,
    void *stream
) {
    if (!x || !out) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_rows < 0 || num_rows > 65535) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (dim < 1 || dim > 8192) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (eps <= 0.0f) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_rows == 0) return ZENO_CUDA_SUCCESS;

    size_t total_bytes = (size_t)num_rows * (size_t)dim * sizeof(float);
    if (partial_overlap(out, total_bytes, x, total_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (weight) {
        size_t dim_bytes = (size_t)dim * sizeof(float);
        if (partial_overlap(out, total_bytes, weight, dim_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
        if (partial_overlap(x, total_bytes, weight, dim_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    }

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    int block_size = calc_rms_norm_block_size(dim);
    dim3 grid(num_rows);
    dim3 block(block_size);
    cudaStream_t custream = (cudaStream_t)stream;

    rms_norm_kernel<<<grid, block, 0, custream>>>(x, weight, out, num_rows, dim, eps);
    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_geglu(
    const float *gate,
    const float *up,
    float *out,
    size_t count,
    int device_id,
    void *stream
) {
    if (!gate || !up || !out) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (count > (((size_t)1 << 31) - 1)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (count == 0) return ZENO_CUDA_SUCCESS;

    size_t bytes = count * sizeof(float);
    // Disallow partial overlap between any pair. Exact in-place aliases are permitted.
    if (partial_overlap(out, bytes, gate, bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (partial_overlap(out, bytes, up, bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (partial_overlap(gate, bytes, up, bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int num_blocks = (unsigned int)((count + block_size - 1) / block_size);
    cudaStream_t custream = (cudaStream_t)stream;

    geglu_kernel<<<num_blocks, block_size, 0, custream>>>(gate, up, out, count);
    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_rope(
    float *vec,
    int num_tokens,
    int num_heads,
    int head_dim,
    int pos_offset,
    float theta,
    const float *freq_factors,
    int device_id,
    void *stream
) {
    if (!vec) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens < 0 || num_tokens > 65535) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_heads < 1 || num_heads > 64) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (head_dim < 1 || head_dim > 1024 || (head_dim % 2) != 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (pos_offset < 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (theta <= 0.0f) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens == 0) return ZENO_CUDA_SUCCESS;

    if (freq_factors) {
        size_t vec_bytes = (size_t)num_tokens * (size_t)num_heads * (size_t)head_dim * sizeof(float);
        size_t ff_bytes = ((size_t)head_dim / 2) * sizeof(float);
        if (ranges_overlap(vec, vec_bytes, freq_factors, ff_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    }

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    dim3 grid(num_heads, num_tokens);
    dim3 block(head_dim / 2);
    cudaStream_t custream = (cudaStream_t)stream;

    rope_kernel<<<grid, block, 0, custream>>>(vec, num_heads, head_dim, pos_offset, theta, freq_factors);
    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_add_residual(
    float *x,
    const float *residual,
    size_t count,
    int device_id,
    void *stream
) {
    if (!x || !residual) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (x == residual) return ZENO_CUDA_ERR_INVALID_PARAM; // Residual source must be distinct
    if (count > (((size_t)1 << 31) - 1)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (count == 0) return ZENO_CUDA_SUCCESS;

    size_t bytes = count * sizeof(float);
    if (ranges_overlap(x, bytes, residual, bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int num_blocks = (unsigned int)((count + block_size - 1) / block_size);
    cudaStream_t custream = (cudaStream_t)stream;

    add_residual_kernel<<<num_blocks, block_size, 0, custream>>>(x, residual, count);
    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_scale(
    float *x,
    float scale,
    size_t count,
    int device_id,
    void *stream
) {
    if (!x) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (count > (((size_t)1 << 31) - 1)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (count == 0) return ZENO_CUDA_SUCCESS;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int num_blocks = (unsigned int)((count + block_size - 1) / block_size);
    cudaStream_t custream = (cudaStream_t)stream;

    scale_kernel<<<num_blocks, block_size, 0, custream>>>(x, scale, count);
    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_layer_output(
    float *out,
    const float *x_mid,
    const float *post_ffn,
    int prompt_len,
    int num_tokens,
    int dim,
    float enc_scale,
    float layer_scale,
    int device_id,
    void *stream
) {
    if (!out || !x_mid || !post_ffn) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens <= 0 || dim <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (prompt_len < 0 || prompt_len > num_tokens) return ZENO_CUDA_ERR_INVALID_PARAM;

    size_t count = (size_t)num_tokens * dim;
    if (count > (((size_t)1 << 31) - 1)) return ZENO_CUDA_ERR_INVALID_PARAM;

    size_t bytes = count * sizeof(float);
    if (partial_overlap(out, bytes, x_mid, bytes) || partial_overlap(out, bytes, post_ffn, bytes)) {
        return ZENO_CUDA_ERR_INVALID_PARAM;
    }

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int num_blocks = (unsigned int)((count + block_size - 1) / block_size);
    cudaStream_t custream = (cudaStream_t)stream;

    layer_output_kernel<<<num_blocks, block_size, 0, custream>>>(
        out, x_mid, post_ffn, prompt_len, num_tokens, dim, enc_scale, layer_scale
    );
    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;

    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_memcpy_peer(
    void *dst,
    int dst_device_id,
    const void *src,
    int src_device_id,
    size_t bytes
) {
    if (bytes == 0) return ZENO_CUDA_SUCCESS;
    if (!dst || !src) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (dst_device_id < 0 || src_device_id < 0) return ZENO_CUDA_ERR_INVALID_DEVICE;

    if (dst_device_id == src_device_id) {
        return zeno_cuda_memcpy_d2d(dst, src, bytes, dst_device_id, NULL);
    }

    cudaError_t err = cudaMemcpyPeer(dst, dst_device_id, src, src_device_id, bytes);
    return (err != cudaSuccess) ? ZENO_CUDA_ERR_COPY_FAILED : ZENO_CUDA_SUCCESS;
}

int zeno_cuda_enable_peer_access(int current_dev, int peer_dev) {
    if (current_dev < 0 || peer_dev < 0 || current_dev == peer_dev) return ZENO_CUDA_SUCCESS;
    int orig = 0;
    int rc = set_cuda_device(current_dev, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    int can_access = 0;
    cudaError_t can_err = cudaDeviceCanAccessPeer(&can_access, current_dev, peer_dev);
    if (can_err != cudaSuccess) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_PEER_QUERY_FAILED);
    }
    if (!can_access) {
        return restore_cuda_device(orig, ZENO_CUDA_PEER_UNSUPPORTED);
    }

    cudaError_t err = cudaDeviceEnablePeerAccess(peer_dev, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_PEER_ENABLE_FAILED);
    }
    return restore_cuda_device(orig, ZENO_CUDA_SUCCESS);
}

int zeno_cuda_memcpy_peer_async(
    void *dst,
    int dst_device_id,
    const void *src,
    int src_device_id,
    size_t bytes,
    void *stream
) {
    if (bytes == 0) return ZENO_CUDA_SUCCESS;
    if (!dst || !src) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (dst_device_id < 0 || src_device_id < 0) return ZENO_CUDA_ERR_INVALID_DEVICE;

    int orig = 0;
    int rc = set_cuda_device(src_device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaStream_t custream = (cudaStream_t)stream;
    if (dst_device_id == src_device_id) {
        int d2d_rc = zeno_cuda_memcpy_d2d(dst, src, bytes, dst_device_id, custream);
        return restore_cuda_device(orig, d2d_rc);
    }

    int can_access = 0;
    cudaError_t can_err = cudaDeviceCanAccessPeer(&can_access, dst_device_id, src_device_id);
    if (can_err != cudaSuccess || !can_access) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_COPY_FAILED);
    }

    cudaError_t err = cudaMemcpyPeerAsync(dst, dst_device_id, src, src_device_id, bytes, custream);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_COPY_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_event_create(void **out_event, int device_id) {
    if (!out_event) return ZENO_CUDA_ERR_INVALID_PARAM;
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaEvent_t ev;
    cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_DRIVER_ERROR);
    }
    *out_event = (void *)ev;
    return restore_cuda_device(orig, ZENO_CUDA_SUCCESS);
}

int zeno_cuda_event_destroy(void *event, int device_id) {
    if (!event) return ZENO_CUDA_SUCCESS;
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaEventDestroy((cudaEvent_t)event);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_DRIVER_ERROR : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_event_record(void *event, void *stream, int device_id) {
    if (!event) return ZENO_CUDA_ERR_INVALID_PARAM;
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_DRIVER_ERROR : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_stream_wait_event(void *stream, void *event, int device_id) {
    if (!event) return ZENO_CUDA_ERR_INVALID_PARAM;
    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaError_t err = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0);
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_DRIVER_ERROR : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

} // extern "C"

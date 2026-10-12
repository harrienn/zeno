#include "zeno_cuda.h"
#include <cuda_runtime.h>
#include <math.h>
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
// 1. Router Input Preparation Kernel
// r_in[t, i] *= root_size * router_scale[i]
// ---------------------------------------------------------------------------

__global__ void moe_router_prep_kernel(
    float *__restrict__ r_in,
    const float *__restrict__ router_scale,
    float root_size,
    int num_tokens,
    int dim
) {
    int t = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_tokens && idx < dim) {
        r_in[(size_t)t * dim + idx] *= root_size * router_scale[idx];
    }
}

// ---------------------------------------------------------------------------
// 2. Router Top-8 Selection & Softmax Renormalization Kernel
// Given logits [num_tokens, 128], selects top-8 indices, computes softmax,
// and scales by exp_down_scale.
// ---------------------------------------------------------------------------

__global__ void moe_router_topk_kernel(
    const float *__restrict__ router_logits,
    const float *__restrict__ exp_down_scale,
    int *__restrict__ out_indices,
    float *__restrict__ out_weights,
    int num_tokens
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    const float *tok_logits = router_logits + (size_t)t * 128;

    // Single-thread per token for selection of top-8 from 128 values
    // to strictly preserve CPU reference deterministic index selection ordering
    if (threadIdx.x == 0) {
        int top_indices[8];
        float top_vals[8];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            top_indices[i] = i;
            top_vals[i] = tok_logits[i];
        }

        for (int cand_idx = 8; cand_idx < 128; ++cand_idx) {
            float cand_val = tok_logits[cand_idx];
            int min_pos = 0;
            float min_val = top_vals[0];

            #pragma unroll
            for (int j = 1; j < 8; ++j) {
                if (top_vals[j] < min_val) {
                    min_val = top_vals[j];
                    min_pos = j;
                }
            }

            if (cand_val > min_val) {
                top_indices[min_pos] = cand_idx;
                top_vals[min_pos] = cand_val;
            }
        }

        // Softmax over the 8 top logits
        float max_l = top_vals[0];
        #pragma unroll
        for (int i = 1; i < 8; ++i) {
            if (top_vals[i] > max_l) max_l = top_vals[i];
        }

        float exp_vals[8];
        float sum_exp = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            exp_vals[i] = expf(top_vals[i] - max_l);
            sum_exp += exp_vals[i];
        }

        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float w = exp_vals[i] * inv_sum;
            if (exp_down_scale) {
                w *= exp_down_scale[top_indices[i]];
            }
            out_indices[(size_t)t * 8 + i] = top_indices[i];
            out_weights[(size_t)t * 8 + i] = w;
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Vector Accumulation with Scalar Scale
// accum[i] += x[i] * scale
// ---------------------------------------------------------------------------

__global__ void accumulate_scaled_kernel(
    float *__restrict__ accum,
    const float *__restrict__ x,
    float scale,
    size_t count
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        accum[idx] += x[idx] * scale;
    }
}

// ---------------------------------------------------------------------------
// C ABI Host Entrypoints
// ---------------------------------------------------------------------------

extern "C" {

int zeno_cuda_moe_router_prep(
    float *r_in,
    const float *router_scale,
    float root_size,
    int num_tokens,
    int dim,
    int device_id,
    void *stream
) {
    if (!r_in || !router_scale) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens <= 0 || dim <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    dim3 grid((unsigned int)((dim + block_size - 1) / block_size), (unsigned int)num_tokens);
    cudaStream_t custream = (cudaStream_t)stream;

    moe_router_prep_kernel<<<grid, block_size, 0, custream>>>(
        r_in,
        router_scale,
        root_size,
        num_tokens,
        dim
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_router_topk(
    const float *router_logits,
    const float *exp_down_scale,
    int *out_indices,
    float *out_weights,
    int num_tokens,
    int device_id,
    void *stream
) {
    if (!router_logits || !out_indices || !out_weights) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    size_t logits_bytes = (size_t)num_tokens * 128 * sizeof(float);
    size_t indices_bytes = (size_t)num_tokens * 8 * sizeof(int);
    size_t weights_bytes = (size_t)num_tokens * 8 * sizeof(float);

    if (ranges_overlap(router_logits, logits_bytes, out_weights, weights_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(router_logits, logits_bytes, out_indices, indices_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    dim3 grid((unsigned int)num_tokens);
    const int block_size = 32;
    cudaStream_t custream = (cudaStream_t)stream;

    moe_router_topk_kernel<<<grid, block_size, 0, custream>>>(
        router_logits,
        exp_down_scale,
        out_indices,
        out_weights,
        num_tokens
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_accumulate_scaled(
    float *accum,
    const float *x,
    float scale,
    size_t count,
    int device_id,
    void *stream
) {
    if (!accum || !x) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (count == 0) return ZENO_CUDA_SUCCESS;

    size_t bytes = count * sizeof(float);
    if (ranges_overlap(accum, bytes, x, bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((count + block_size - 1) / block_size);
    cudaStream_t custream = (cudaStream_t)stream;

    accumulate_scaled_kernel<<<grid_x, block_size, 0, custream>>>(
        accum,
        x,
        scale,
        count
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

} // extern "C"

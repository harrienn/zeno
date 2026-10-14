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

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ---------------------------------------------------------------------------
// Unified Multi-Head Attention Kernel
// Grid: (num_tokens, num_q_heads)
// Block: 256 threads (8 warps)
// ---------------------------------------------------------------------------

__global__ void attention_kernel(
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    float *__restrict__ out,
    int num_tokens,
    int prompt_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int stride_q,
    int stride_k,
    int stride_v,
    int stride_out
) {
    int t = blockIdx.x; // Token index [0 .. num_tokens - 1]
    int h = blockIdx.y; // Query head index [0 .. num_q_heads - 1]
    int tid = threadIdx.x;

    __shared__ float s_q[512];
    __shared__ float s_scores[512];
    __shared__ float s_red[32]; // Reductions across 8 warps

    int gqa_ratio = num_q_heads / num_kv_heads;
    int kv = h / gqa_ratio;
    int visible_len = (t < prompt_len) ? (t + 1) : num_tokens;

    int lane = tid & 31;
    int warp_id = tid >> 5;

    // 1. Load Query vector Q[t, h] into shared memory
    const float *q_head_ptr = q + (size_t)t * stride_q + (size_t)h * head_dim;
    s_q[tid] = q_head_ptr[tid];
    if (head_dim == 512) {
        s_q[tid + 256] = q_head_ptr[tid + 256];
    }
    __syncthreads();

    // 2. Compute dot product Q[t, h] . K[s, kv] for each visible key s
    for (int s = 0; s < visible_len; ++s) {
        const float *k_head_ptr = k + (size_t)s * stride_k + (size_t)kv * head_dim;
        float prod = s_q[tid] * k_head_ptr[tid];
        if (head_dim == 512) {
            prod += s_q[tid + 256] * k_head_ptr[tid + 256];
        }

        prod = warp_reduce_sum(prod);
        if (lane == 0) {
            s_red[warp_id] = prod;
        }
        __syncthreads();

        if (tid == 0) {
            float dot_sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                dot_sum += s_red[w];
            }
            // Gemma 4 attention scale is 1.0f
            s_scores[s] = dot_sum;
        }
        __syncthreads();
    }

    // Initialize scores for masked positions
    for (int s = visible_len + tid; s < num_tokens; s += 256) {
        s_scores[s] = -1e30f;
    }
    __syncthreads();

    // 3. Fused Softmax across s in [0 .. num_tokens - 1]
    // 3a. Find max score
    float thread_max = -1e30f;
    for (int s = tid; s < num_tokens; s += 256) {
        thread_max = fmaxf(thread_max, s_scores[s]);
    }
    thread_max = warp_reduce_max(thread_max);
    if (lane == 0) {
        s_red[warp_id] = thread_max;
    }
    __syncthreads();

    if (tid == 0) {
        float global_max = -1e30f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            global_max = fmaxf(global_max, s_red[w]);
        }
        s_red[0] = global_max;
    }
    __syncthreads();
    float max_val = s_red[0];

    // 3b. Exponentiate and accumulate sum
    float thread_sum = 0.0f;
    for (int s = tid; s < num_tokens; s += 256) {
        float exp_val = (s < visible_len) ? expf(s_scores[s] - max_val) : 0.0f;
        s_scores[s] = exp_val;
        thread_sum += exp_val;
    }
    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        s_red[warp_id] = thread_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float global_sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            global_sum += s_red[w];
        }
        s_red[0] = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    }
    __syncthreads();
    float inv_sum = s_red[0];

    // 3c. Normalize probabilities
    for (int s = tid; s < visible_len; s += 256) {
        s_scores[s] *= inv_sum;
    }
    __syncthreads();

    // 4. Value accumulation: O[t, h, d] = sum_{s=0}^{visible_len-1} (s_scores[s] * V[s, kv, d])
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int s = 0; s < visible_len; ++s) {
        float w = s_scores[s];
        if (w == 0.0f) continue;
        const float *v_head_ptr = v + (size_t)s * stride_v + (size_t)kv * head_dim;
        acc0 += w * v_head_ptr[tid];
        if (head_dim == 512) {
            acc1 += w * v_head_ptr[tid + 256];
        }
    }

    float *out_head_ptr = out + (size_t)t * stride_out + (size_t)h * head_dim;
    out_head_ptr[tid] = acc0;
    if (head_dim == 512) {
        out_head_ptr[tid + 256] = acc1;
    }
}

// ---------------------------------------------------------------------------
// Unified Multi-Head Attention Kernel with Cached Prompt KV (Phase 7B)
// Grid: (canvas_len, num_q_heads)
// Block: 256 threads (8 warps)
// ---------------------------------------------------------------------------

__global__ void attention_cached_kernel(
    const float *__restrict__ q,
    const float *__restrict__ prompt_k,
    const float *__restrict__ prompt_v,
    const float *__restrict__ canvas_k,
    const float *__restrict__ canvas_v,
    float *__restrict__ out,
    int prompt_len,
    int canvas_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int stride_q,
    int stride_k,
    int stride_v,
    int stride_out
) {
    int c = blockIdx.x; // Canvas token index [0 .. canvas_len - 1]
    int h = blockIdx.y; // Query head index [0 .. num_q_heads - 1]
    int tid = threadIdx.x;

    __shared__ float s_q[512];
    __shared__ float s_scores[512];
    __shared__ float s_red[32]; // Reductions across 8 warps

    int gqa_ratio = num_q_heads / num_kv_heads;
    int kv = h / gqa_ratio;
    int total_tokens = prompt_len + canvas_len;

    int lane = tid & 31;
    int warp_id = tid >> 5;

    // 1. Load Query vector Q[c, h] into shared memory
    const float *q_head_ptr = q + (size_t)c * stride_q + (size_t)h * head_dim;
    s_q[tid] = q_head_ptr[tid];
    if (head_dim == 512) {
        s_q[tid + 256] = q_head_ptr[tid + 256];
    }
    __syncthreads();

    // 2. Compute dot product Q[c, h] . K[s, kv] for all visible keys s in [0 .. total_tokens - 1]
    for (int s = 0; s < total_tokens; ++s) {
        const float *k_head_ptr;
        if (s < prompt_len) {
            k_head_ptr = prompt_k + (size_t)s * stride_k + (size_t)kv * head_dim;
        } else {
            k_head_ptr = canvas_k + (size_t)(s - prompt_len) * stride_k + (size_t)kv * head_dim;
        }

        float prod = s_q[tid] * k_head_ptr[tid];
        if (head_dim == 512) {
            prod += s_q[tid + 256] * k_head_ptr[tid + 256];
        }

        prod = warp_reduce_sum(prod);
        if (lane == 0) {
            s_red[warp_id] = prod;
        }
        __syncthreads();

        if (tid == 0) {
            float dot_sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                dot_sum += s_red[w];
            }
            s_scores[s] = dot_sum;
        }
        __syncthreads();
    }

    // Initialize scores for positions beyond total_tokens (if any)
    for (int s = total_tokens + tid; s < 512; s += 256) {
        s_scores[s] = -1e30f;
    }
    __syncthreads();

    // 3. Fused Softmax across s in [0 .. total_tokens - 1]
    // 3a. Find max score
    float thread_max = -1e30f;
    for (int s = tid; s < total_tokens; s += 256) {
        thread_max = fmaxf(thread_max, s_scores[s]);
    }
    thread_max = warp_reduce_max(thread_max);
    if (lane == 0) {
        s_red[warp_id] = thread_max;
    }
    __syncthreads();

    if (tid == 0) {
        float global_max = -1e30f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            global_max = fmaxf(global_max, s_red[w]);
        }
        s_red[0] = global_max;
    }
    __syncthreads();
    float max_val = s_red[0];

    // 3b. Exponentiate and accumulate sum
    float thread_sum = 0.0f;
    for (int s = tid; s < total_tokens; s += 256) {
        float exp_val = expf(s_scores[s] - max_val);
        s_scores[s] = exp_val;
        thread_sum += exp_val;
    }
    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        s_red[warp_id] = thread_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float global_sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            global_sum += s_red[w];
        }
        s_red[0] = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    }
    __syncthreads();
    float inv_sum = s_red[0];

    // 3c. Normalize probabilities
    for (int s = tid; s < total_tokens; s += 256) {
        s_scores[s] *= inv_sum;
    }
    __syncthreads();

    // 4. Value accumulation: O[c, h, d] = sum_{s=0}^{total_tokens-1} (s_scores[s] * V[s, kv, d])
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int s = 0; s < total_tokens; ++s) {
        float w = s_scores[s];
        if (w == 0.0f) continue;
        const float *v_head_ptr;
        if (s < prompt_len) {
            v_head_ptr = prompt_v + (size_t)s * stride_v + (size_t)kv * head_dim;
        } else {
            v_head_ptr = canvas_v + (size_t)(s - prompt_len) * stride_v + (size_t)kv * head_dim;
        }

        acc0 += w * v_head_ptr[tid];
        if (head_dim == 512) {
            acc1 += w * v_head_ptr[tid + 256];
        }
    }

    float *out_head_ptr = out + (size_t)c * stride_out + (size_t)h * head_dim;
    out_head_ptr[tid] = acc0;
    if (head_dim == 512) {
        out_head_ptr[tid + 256] = acc1;
    }
}

// ---------------------------------------------------------------------------
// C ABI Host Entrypoints
// ---------------------------------------------------------------------------

extern "C" {

int zeno_cuda_attention(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    int num_tokens,
    int prompt_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int stride_q,
    int stride_k,
    int stride_v,
    int stride_out,
    int device_id,
    void *stream
) {
    if (!q || !k || !v || !out) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens <= 0 || num_tokens > 512) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (prompt_len < 0 || prompt_len > num_tokens) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_q_heads <= 0 || num_kv_heads <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if ((num_q_heads % num_kv_heads) != 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (head_dim != 256 && head_dim != 512) return ZENO_CUDA_ERR_INVALID_PARAM;

    if (stride_q < num_q_heads * head_dim) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (stride_k < num_kv_heads * head_dim) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (stride_v < num_kv_heads * head_dim) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (stride_out < num_q_heads * head_dim) return ZENO_CUDA_ERR_INVALID_PARAM;

    size_t q_bytes = (size_t)num_tokens * (size_t)stride_q * sizeof(float);
    size_t k_bytes = (size_t)num_tokens * (size_t)stride_k * sizeof(float);
    size_t v_bytes = (size_t)num_tokens * (size_t)stride_v * sizeof(float);
    size_t out_bytes = (size_t)num_tokens * (size_t)stride_out * sizeof(float);

    if (ranges_overlap(q, q_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(k, k_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(v, v_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    dim3 grid((unsigned int)num_tokens, (unsigned int)num_q_heads);
    const int block_size = 256;
    cudaStream_t custream = (cudaStream_t)stream;

    attention_kernel<<<grid, block_size, 0, custream>>>(
        q,
        k,
        v,
        out,
        num_tokens,
        prompt_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        stride_q,
        stride_k,
        stride_v,
        stride_out
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_attention_cached(
    const float *q,
    const float *prompt_k,
    const float *prompt_v,
    const float *canvas_k,
    const float *canvas_v,
    float *out,
    int prompt_len,
    int canvas_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int stride_q,
    int stride_k,
    int stride_v,
    int stride_out,
    int device_id,
    void *stream
) {
    if (!q || !canvas_k || !canvas_v || !out) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (prompt_len < 0 || canvas_len <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (prompt_len + canvas_len > 512) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (prompt_len > 0 && (!prompt_k || !prompt_v)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_q_heads <= 0 || num_kv_heads <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if ((num_q_heads % num_kv_heads) != 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (head_dim != 256 && head_dim != 512) return ZENO_CUDA_ERR_INVALID_PARAM;

    if (stride_q < num_q_heads * head_dim) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (stride_k < num_kv_heads * head_dim) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (stride_v < num_kv_heads * head_dim) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (stride_out < num_q_heads * head_dim) return ZENO_CUDA_ERR_INVALID_PARAM;

    size_t q_bytes = (size_t)canvas_len * (size_t)stride_q * sizeof(float);
    size_t canvas_k_bytes = (size_t)canvas_len * (size_t)stride_k * sizeof(float);
    size_t canvas_v_bytes = (size_t)canvas_len * (size_t)stride_v * sizeof(float);
    size_t out_bytes = (size_t)canvas_len * (size_t)stride_out * sizeof(float);

    if (ranges_overlap(q, q_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(canvas_k, canvas_k_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (ranges_overlap(canvas_v, canvas_v_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (prompt_len > 0) {
        size_t prompt_k_bytes = (size_t)prompt_len * (size_t)stride_k * sizeof(float);
        size_t prompt_v_bytes = (size_t)prompt_len * (size_t)stride_v * sizeof(float);
        if (ranges_overlap(prompt_k, prompt_k_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
        if (ranges_overlap(prompt_v, prompt_v_bytes, out, out_bytes)) return ZENO_CUDA_ERR_INVALID_PARAM;
    }

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    dim3 grid((unsigned int)canvas_len, (unsigned int)num_q_heads);
    const int block_size = 256;
    cudaStream_t custream = (cudaStream_t)stream;

    attention_cached_kernel<<<grid, block_size, 0, custream>>>(
        q,
        prompt_k,
        prompt_v,
        canvas_k,
        canvas_v,
        out,
        prompt_len,
        canvas_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        stride_q,
        stride_k,
        stride_v,
        stride_out
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

} // extern "C"

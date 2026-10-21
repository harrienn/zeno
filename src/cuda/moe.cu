#include "zeno_cuda.h"
#include "quant_types.cuh"
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
// 4. Batched Device-Side MoE Kernels (R22-01)
// ---------------------------------------------------------------------------

__global__ void moe_gemv_gate_up_batched_kernel(
    const uint8_t *__restrict__ all_expert_weights,
    const float *__restrict__ tokens_in,
    const int *__restrict__ top_indices,
    float *__restrict__ out_gate_up,
    int num_tokens,
    int in_dim,
    int out_dim,
    int blocks_per_row,
    size_t expert_bytes
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row_base = (blockIdx.x * 8 + warp_id) * 4;
    int k = blockIdx.y;

    float *k_out = out_gate_up + (size_t)k * out_dim;
    int exp_id = top_indices[k];
    if (exp_id < 0) {
        if (lane_id == 0) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (row_base + r < out_dim) k_out[row_base + r] = 0.0f;
            }
        }
        return;
    }

    int token_idx = k / 8;
    const BlockQ4_K *exp_W = (const BlockQ4_K *)(all_expert_weights + (size_t)exp_id * expert_bytes);
    const float *token_x = tokens_in + (size_t)token_idx * in_dim;

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int row = row_base + r;
        if (row >= out_dim) continue;

        const BlockQ4_K *row_w = exp_W + (size_t)row * blocks_per_row;
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

                float x1 = token_x[base_x + chunk * 64 + lane_id];
                float x2 = token_x[base_x + chunk * 64 + 32 + lane_id];

                float w1 = d1 * q1 - m_val1;
                float w2 = d2 * q2 - m_val2;

                thread_sum += w1 * x1 + w2 * x2;
            }
        }

        float row_sum = warp_reduce_sum(thread_sum);
        if (lane_id == 0) {
            k_out[row] = row_sum;
        }
    }
}

__global__ void moe_geglu_batched_kernel(
    const float *__restrict__ gate_up,
    float *__restrict__ act_out,
    int num_instances,
    int half_dim
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)num_instances * half_dim;
    if (idx < total) {
        size_t inst = idx / half_dim;
        size_t col = idx % half_dim;
        float g = gate_up[inst * (half_dim * 2) + col];
        float u = gate_up[inst * (half_dim * 2) + half_dim + col];
        act_out[idx] = gelu_tanh_scalar(g) * u;
    }
}

__global__ void moe_gemv_down_batched_kernel(
    const uint8_t *__restrict__ all_expert_weights,
    int expert_type,
    const float *__restrict__ acts_in,
    const int *__restrict__ top_indices,
    float *__restrict__ out_down,
    int num_tokens,
    int in_dim,
    int out_dim,
    int blocks_per_row,
    size_t expert_bytes
) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row_base = (blockIdx.x * 8 + warp_id) * 4;
    int k = blockIdx.y;

    float *k_out = out_down + (size_t)k * out_dim;
    int exp_id = top_indices[k];
    if (exp_id < 0) {
        if (lane_id == 0) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (row_base + r < out_dim) k_out[row_base + r] = 0.0f;
            }
        }
        return;
    }

    const float *inst_act = acts_in + (size_t)k * in_dim;

    if (expert_type == 8) { // Q8_0
        const BlockQ8_0 *exp_W = (const BlockQ8_0 *)(all_expert_weights + (size_t)exp_id * expert_bytes);
        int r0 = row_base + 0;
        int r1 = row_base + 1;
        int r2 = row_base + 2;
        int r3 = row_base + 3;

        const BlockQ8_0 *row_w0 = (r0 < out_dim) ? (exp_W + (size_t)r0 * blocks_per_row) : NULL;
        const BlockQ8_0 *row_w1 = (r1 < out_dim) ? (exp_W + (size_t)r1 * blocks_per_row) : NULL;
        const BlockQ8_0 *row_w2 = (r2 < out_dim) ? (exp_W + (size_t)r2 * blocks_per_row) : NULL;
        const BlockQ8_0 *row_w3 = (r3 < out_dim) ? (exp_W + (size_t)r3 * blocks_per_row) : NULL;

        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            float xv = inst_act[b * 32 + lane_id];
            if (row_w0) {
                const BlockQ8_0 *blk0 = row_w0 + b;
                sum0 += __half2float(blk0->d) * ((float)blk0->qs[lane_id] * xv);
            }
            if (row_w1) {
                const BlockQ8_0 *blk1 = row_w1 + b;
                sum1 += __half2float(blk1->d) * ((float)blk1->qs[lane_id] * xv);
            }
            if (row_w2) {
                const BlockQ8_0 *blk2 = row_w2 + b;
                sum2 += __half2float(blk2->d) * ((float)blk2->qs[lane_id] * xv);
            }
            if (row_w3) {
                const BlockQ8_0 *blk3 = row_w3 + b;
                sum3 += __half2float(blk3->d) * ((float)blk3->qs[lane_id] * xv);
            }
        }

        if (row_w0) {
            float rsum0 = warp_reduce_sum(sum0);
            if (lane_id == 0) k_out[r0] = rsum0;
        }
        if (row_w1) {
            float rsum1 = warp_reduce_sum(sum1);
            if (lane_id == 0) k_out[r1] = rsum1;
        }
        if (row_w2) {
            float rsum2 = warp_reduce_sum(sum2);
            if (lane_id == 0) k_out[r2] = rsum2;
        }
        if (row_w3) {
            float rsum3 = warp_reduce_sum(sum3);
            if (lane_id == 0) k_out[r3] = rsum3;
        }
    } else { // Q5_0
        const BlockQ5_0 *exp_W = (const BlockQ5_0 *)(all_expert_weights + (size_t)exp_id * expert_bytes);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int row = row_base + r;
            if (row >= out_dim) continue;
            const BlockQ5_0 *row_w = exp_W + (size_t)row * blocks_per_row;
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
                    float xv = inst_act[base_x + lane_id];
                    thread_sum += w * xv;
                } else {
                    int j = lane_id - 16;
                    uint8_t bit = (uint8_t)((qh >> (j + 16)) & 1);
                    int q = (int)((q_byte >> 4) | (bit << 4)) - 16;
                    float w = d * (float)q;
                    float xv = inst_act[base_x + lane_id];
                    thread_sum += w * xv;
                }
            }
            float row_sum = warp_reduce_sum(thread_sum);
            if (lane_id == 0) {
                k_out[row] = row_sum;
            }
        }
    }
}

__global__ void moe_accumulate_scaled_batched_kernel(
    float *__restrict__ moe_accum,
    const float *__restrict__ down_out,
    const float *__restrict__ top_weights,
    int num_tokens,
    int dim
) {
    int token_idx = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx < num_tokens && d < dim) {
        float sum = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            int k = token_idx * 8 + e;
            float w = top_weights[k];
            sum += down_out[(size_t)k * dim + d] * w;
        }
        moe_accum[(size_t)token_idx * dim + d] = sum;
    }
}

// ---------------------------------------------------------------------------
// 5. Stage 7 Token-Sorted Expert MoE Kernels (Phase 7A)
// ---------------------------------------------------------------------------

__global__ void moe_build_expert_bins_kernel(
    const int *__restrict__ top_indices,
    int *__restrict__ expert_counts,
    int *__restrict__ expert_offsets,
    int *__restrict__ sorted_instances,
    int *__restrict__ instance_to_sorted,
    int *__restrict__ active_experts,
    int *__restrict__ num_active_experts,
    int num_tokens
) {
    __shared__ int s_counts[128];
    __shared__ int s_offsets[128];
    __shared__ int s_write_ptrs[128];

    int tid = threadIdx.x;
    int num_instances = num_tokens * 8;

    if (tid < 128) {
        s_counts[tid] = 0;
    }
    __syncthreads();

    // 1. Histogram of expert assignments
    for (int i = tid; i < num_instances; i += blockDim.x) {
        int exp_id = top_indices[i];
        if (exp_id >= 0 && exp_id < 128) {
            atomicAdd(&s_counts[exp_id], 1);
        }
    }
    __syncthreads();

    // 2. Prefix scan over 128 experts & active-expert compaction (performed deterministically by thread 0)
    if (tid == 0) {
        int offset = 0;
        int active_count = 0;
        for (int e = 0; e < 128; ++e) {
            int cnt = s_counts[e];
            s_offsets[e] = offset;
            s_write_ptrs[e] = offset;
            offset += cnt;
            if (cnt > 0) {
                if (active_experts) {
                    active_experts[active_count] = e;
                }
                active_count++;
            }
        }
        if (num_active_experts) {
            *num_active_experts = active_count;
        }
    }
    __syncthreads();

    // 3. Deterministic scatter of instances into sorted buckets (thread 0)
    if (tid == 0) {
        for (int i = 0; i < num_instances; ++i) {
            int exp_id = top_indices[i];
            if (exp_id >= 0 && exp_id < 128) {
                int pos = s_write_ptrs[exp_id]++;
                sorted_instances[pos] = i;
                instance_to_sorted[i] = pos;
            } else {
                instance_to_sorted[i] = -1;
            }
        }
    }
    __syncthreads();

    // 4. Write back counts and offsets to global memory
    if (tid < 128) {
        expert_counts[tid] = s_counts[tid];
        expert_offsets[tid] = s_offsets[tid];
    }
}

__global__ void moe_gemv_gate_up_sorted_kernel(
    const uint8_t *__restrict__ all_expert_weights,
    const float *__restrict__ tokens_in,
    const int *__restrict__ expert_counts,
    const int *__restrict__ expert_offsets,
    const int *__restrict__ sorted_instances,
    float *__restrict__ out_gate_up,
    int in_dim,
    int out_dim,
    int blocks_per_row,
    size_t expert_bytes
) {
    int exp_id = blockIdx.y;
    int count = expert_counts[exp_id];
    if (count <= 0) return;

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row_base = (blockIdx.x * 8 + warp_id) * 8;
    int offset = expert_offsets[exp_id];

    const BlockQ4_K *exp_W = (const BlockQ4_K *)(all_expert_weights + (size_t)exp_id * expert_bytes);

    for (int m_base = 0; m_base < count; m_base += 4) {
        int batch = count - m_base;
        if (batch > 4) batch = 4;

        const float *tok_ptrs[4];
        float *out_ptrs[4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m < batch) {
                int sorted_slot = offset + m_base + m;
                int inst = sorted_instances[sorted_slot];
                int tok_idx = inst / 8;
                tok_ptrs[m] = tokens_in + (size_t)tok_idx * in_dim;
                out_ptrs[m] = out_gate_up + (size_t)sorted_slot * out_dim;
            } else {
                tok_ptrs[m] = NULL;
                out_ptrs[m] = NULL;
            }
        }

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int row = row_base + r;
            if (row >= out_dim) continue;

            const BlockQ4_K *row_w = exp_W + (size_t)row * blocks_per_row;
            float thread_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

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

                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) {
                            float x1 = tok_ptrs[m][base_x + chunk * 64 + lane_id];
                            float x2 = tok_ptrs[m][base_x + chunk * 64 + 32 + lane_id];
                            thread_sum[m] += w1 * x1 + w2 * x2;
                        }
                    }
                }
            }

            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < batch) {
                    float row_sum = warp_reduce_sum(thread_sum[m]);
                    if (lane_id == 0) {
                        out_ptrs[m][row] = row_sum;
                    }
                }
            }
        }
    }
}

__launch_bounds__(256, 4)
__global__ void moe_gemv_gate_up_geglu_sorted_kernel(
    const uint8_t *__restrict__ all_expert_weights,
    const float *__restrict__ tokens_in,
    const int *__restrict__ expert_counts,
    const int *__restrict__ expert_offsets,
    const int *__restrict__ sorted_instances,
    float *__restrict__ out_act,
    int in_dim,
    int half_dim,
    int blocks_per_row,
    size_t expert_bytes
) {
    __shared__ alignas(16) float s_x[4][256];

    int exp_id = blockIdx.y;
    int count = expert_counts[exp_id];
    if (count <= 0) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int row0 = (blockIdx.x * 8 + warp_id) * 2;
    int row1 = row0 + 1;
    bool active_row0 = (row0 < half_dim);
    bool active_row1 = (row1 < half_dim);
    int offset = expert_offsets[exp_id];

    const BlockQ4_K *exp_W = (const BlockQ4_K *)(all_expert_weights + (size_t)exp_id * expert_bytes);

    for (int m_base = 0; m_base < count; m_base += 4) {
        int batch = count - m_base;
        if (batch > 4) batch = 4;

        const BlockQ4_K *gate_row_w0 = active_row0 ? (exp_W + (size_t)row0 * blocks_per_row) : NULL;
        const BlockQ4_K *up_row_w0   = active_row0 ? (exp_W + (size_t)(row0 + half_dim) * blocks_per_row) : NULL;
        const BlockQ4_K *gate_row_w1 = active_row1 ? (exp_W + (size_t)row1 * blocks_per_row) : NULL;
        const BlockQ4_K *up_row_w1   = active_row1 ? (exp_W + (size_t)(row1 + half_dim) * blocks_per_row) : NULL;

        float gate_thread_sum0[4] = {0.0f}, up_thread_sum0[4] = {0.0f};
        float gate_thread_sum1[4] = {0.0f}, up_thread_sum1[4] = {0.0f};

        for (int b = 0; b < blocks_per_row; b++) {
            size_t base_x = (size_t)b * 256;

            // Collaborative 128-bit float4 loading of s_x[4][256] across 256 threads (WP3 stackless)
            int m_load = tid >> 6;
            int elem = (tid & 63) * 4;
            float4 loaded_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (m_base + m_load < count) {
                int sorted_slot = offset + m_base + m_load;
                int inst = sorted_instances[sorted_slot];
                int tok_idx = inst / 8;
                loaded_val = *(const float4*)&tokens_in[(size_t)tok_idx * in_dim + base_x + elem];
            }
            *(float4*)&s_x[m_load][elem] = loaded_val;
            __syncthreads();

            const BlockQ4_K *blk_gate0 = active_row0 ? (gate_row_w0 + b) : NULL;
            const BlockQ4_K *blk_up0   = active_row0 ? (up_row_w0 + b)   : NULL;
            const BlockQ4_K *blk_gate1 = active_row1 ? (gate_row_w1 + b) : NULL;
            const BlockQ4_K *blk_up1   = active_row1 ? (up_row_w1 + b)   : NULL;

            float my_d_g0 = 0.0f, my_m_g0 = 0.0f;
            float my_d_u0 = 0.0f, my_m_u0 = 0.0f;
            float my_d_g1 = 0.0f, my_m_g1 = 0.0f;
            float my_d_u1 = 0.0f, my_m_u1 = 0.0f;

            if (active_row0) {
                precompute_q4k_scales(blk_gate0, lane_id, &my_d_g0, &my_m_g0);
                precompute_q4k_scales(blk_up0, lane_id, &my_d_u0, &my_m_u0);
            }
            if (active_row1) {
                precompute_q4k_scales(blk_gate1, lane_id, &my_d_g1, &my_m_g1);
                precompute_q4k_scales(blk_up1, lane_id, &my_d_u1, &my_m_u1);
            }

            #pragma unroll
            for (int chunk = 0; chunk < 4; chunk++) {
                int off1 = chunk * 64 + lane_id;
                int off2 = chunk * 64 + 32 + lane_id;

                float x1[4], x2[4];
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        x1[m] = s_x[m][off1];
                        x2[m] = s_x[m][off2];
                    }
                }

                if (active_row0) {
                    float d1_g0 = __shfl_sync(0xffffffff, my_d_g0, chunk * 2 + 0);
                    float m1_g0 = __shfl_sync(0xffffffff, my_m_g0, chunk * 2 + 0);
                    float d2_g0 = __shfl_sync(0xffffffff, my_d_g0, chunk * 2 + 1);
                    float m2_g0 = __shfl_sync(0xffffffff, my_m_g0, chunk * 2 + 1);

                    float d1_u0 = __shfl_sync(0xffffffff, my_d_u0, chunk * 2 + 0);
                    float m1_u0 = __shfl_sync(0xffffffff, my_m_u0, chunk * 2 + 0);
                    float d2_u0 = __shfl_sync(0xffffffff, my_d_u0, chunk * 2 + 1);
                    float m2_u0 = __shfl_sync(0xffffffff, my_m_u0, chunk * 2 + 1);

                    uint8_t q_byte_g0 = blk_gate0->qs[chunk * 32 + lane_id];
                    float w1_g0 = d1_g0 * (float)(q_byte_g0 & 0x0F) - m1_g0;
                    float w2_g0 = d2_g0 * (float)(q_byte_g0 >> 4) - m2_g0;

                    uint8_t q_byte_u0 = blk_up0->qs[chunk * 32 + lane_id];
                    float w1_u0 = d1_u0 * (float)(q_byte_u0 & 0x0F) - m1_u0;
                    float w2_u0 = d2_u0 * (float)(q_byte_u0 >> 4) - m2_u0;

                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) {
                            gate_thread_sum0[m] += w1_g0 * x1[m] + w2_g0 * x2[m];
                            up_thread_sum0[m]   += w1_u0 * x1[m] + w2_u0 * x2[m];
                        }
                    }
                }

                if (active_row1) {
                    float d1_g1 = __shfl_sync(0xffffffff, my_d_g1, chunk * 2 + 0);
                    float m1_g1 = __shfl_sync(0xffffffff, my_m_g1, chunk * 2 + 0);
                    float d2_g1 = __shfl_sync(0xffffffff, my_d_g1, chunk * 2 + 1);
                    float m2_g1 = __shfl_sync(0xffffffff, my_m_g1, chunk * 2 + 1);

                    float d1_u1 = __shfl_sync(0xffffffff, my_d_u1, chunk * 2 + 0);
                    float m1_u1 = __shfl_sync(0xffffffff, my_m_u1, chunk * 2 + 0);
                    float d2_u1 = __shfl_sync(0xffffffff, my_d_u1, chunk * 2 + 1);
                    float m2_u1 = __shfl_sync(0xffffffff, my_m_u1, chunk * 2 + 1);

                    uint8_t q_byte_g1 = blk_gate1->qs[chunk * 32 + lane_id];
                    float w1_g1 = d1_g1 * (float)(q_byte_g1 & 0x0F) - m1_g1;
                    float w2_g1 = d2_g1 * (float)(q_byte_g1 >> 4) - m2_g1;

                    uint8_t q_byte_u1 = blk_up1->qs[chunk * 32 + lane_id];
                    float w1_u1 = d1_u1 * (float)(q_byte_u1 & 0x0F) - m1_u1;
                    float w2_u1 = d2_u1 * (float)(q_byte_u1 >> 4) - m2_u1;

                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) {
                            gate_thread_sum1[m] += w1_g1 * x1[m] + w2_g1 * x2[m];
                            up_thread_sum1[m]   += w1_u1 * x1[m] + w2_u1 * x2[m];
                        }
                    }
                }
            }

            if (b + 1 < blocks_per_row) {
                __syncthreads();
            }
        }

        if (active_row0) {
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < batch) {
                    float g0 = warp_reduce_sum(gate_thread_sum0[m]);
                    float u0 = warp_reduce_sum(up_thread_sum0[m]);
                    if (lane_id == 0) {
                        int sorted_slot = offset + m_base + m;
                        out_act[(size_t)sorted_slot * half_dim + row0] = gelu_tanh_scalar(g0) * u0;
                    }
                }
            }
        }

        if (active_row1) {
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < batch) {
                    float g1 = warp_reduce_sum(gate_thread_sum1[m]);
                    float u1 = warp_reduce_sum(up_thread_sum1[m]);
                    if (lane_id == 0) {
                        int sorted_slot = offset + m_base + m;
                        out_act[(size_t)sorted_slot * half_dim + row1] = gelu_tanh_scalar(g1) * u1;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Phase 13 WP2: Active-Expert Persistent MoE Gate+Up+GeGLU Kernel
// Launches a static hardware-matched grid (84 CTAs on Ampere SM 8.6) pulling active expert row tiles dynamically
// from work_counter via atomicAdd. Eliminates ~60,000 empty Gate+Up CTA launches per canvas step.
__launch_bounds__(256, 2)
__global__ void moe_gemv_gate_up_geglu_persistent_kernel(
    const uint8_t *__restrict__ all_expert_weights,
    const float *__restrict__ tokens_in,
    const int *__restrict__ expert_counts,
    const int *__restrict__ expert_offsets,
    const int *__restrict__ sorted_instances,
    const int *__restrict__ active_experts,
    const int *__restrict__ num_active_experts_ptr,
    int *__restrict__ work_counter,
    float *__restrict__ out_act,
    int in_dim,
    int half_dim,
    int blocks_per_row,
    size_t expert_bytes
) {
    __shared__ int s_num_active;
    __shared__ int s_task_id;

    if (threadIdx.x == 0) {
        s_num_active = *num_active_experts_ptr;
    }
    __syncthreads();

    int num_active = s_num_active;
    if (num_active <= 0) return;

    int num_row_tiles = (half_dim + 31) / 32;
    int total_tasks = num_active * num_row_tiles;

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    while (true) {
        if (threadIdx.x == 0) {
            s_task_id = atomicAdd(work_counter, 1);
        }
        __syncthreads();

        int task_id = s_task_id;
        if (task_id >= total_tasks) break;

        int active_idx = task_id / num_row_tiles;
        int row_tile = task_id % num_row_tiles;
        int exp_id = active_experts[active_idx];
        int count = expert_counts[exp_id];
        int offset = expert_offsets[exp_id];

        int row_base = (row_tile * 8 + warp_id) * 4;
        const BlockQ4_K *exp_W = (const BlockQ4_K *)(all_expert_weights + (size_t)exp_id * expert_bytes);

        for (int m_base = 0; m_base < count; m_base += 4) {
            int batch = count - m_base;
            if (batch > 4) batch = 4;

            const float *tok_ptrs[4];
            float *out_ptrs[4];
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < batch) {
                    int sorted_slot = offset + m_base + m;
                    int inst = sorted_instances[sorted_slot];
                    int tok_idx = inst / 8;
                    tok_ptrs[m] = tokens_in + (size_t)tok_idx * in_dim;
                    out_ptrs[m] = out_act + (size_t)sorted_slot * half_dim;
                } else {
                    tok_ptrs[m] = NULL;
                    out_ptrs[m] = NULL;
                }
            }

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int row = row_base + r;
                if (row >= half_dim) continue;

                const BlockQ4_K *gate_row_w = exp_W + (size_t)row * blocks_per_row;
                const BlockQ4_K *up_row_w = exp_W + (size_t)(row + half_dim) * blocks_per_row;
                float gate_thread_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float up_thread_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int b = 0; b < blocks_per_row; b++) {
                    const BlockQ4_K *blk_gate = gate_row_w + b;
                    const BlockQ4_K *blk_up = up_row_w + b;
                    size_t base_x = (size_t)b * 256;

                    float my_d_g, my_m_g;
                    precompute_q4k_scales(blk_gate, lane_id, &my_d_g, &my_m_g);

                    float my_d_u, my_m_u;
                    precompute_q4k_scales(blk_up, lane_id, &my_d_u, &my_m_u);

                    #pragma unroll
                    for (int chunk = 0; chunk < 4; chunk++) {
                        float d1_g = __shfl_sync(0xffffffff, my_d_g, chunk * 2 + 0);
                        float m1_g = __shfl_sync(0xffffffff, my_m_g, chunk * 2 + 0);
                        float d2_g = __shfl_sync(0xffffffff, my_d_g, chunk * 2 + 1);
                        float m2_g = __shfl_sync(0xffffffff, my_m_g, chunk * 2 + 1);

                        float d1_u = __shfl_sync(0xffffffff, my_d_u, chunk * 2 + 0);
                        float m1_u = __shfl_sync(0xffffffff, my_m_u, chunk * 2 + 0);
                        float d2_u = __shfl_sync(0xffffffff, my_d_u, chunk * 2 + 1);
                        float m2_u = __shfl_sync(0xffffffff, my_m_u, chunk * 2 + 1);

                        const uint8_t *q_slice_g = blk_gate->qs + chunk * 32;
                        uint8_t q_byte_g = q_slice_g[lane_id];
                        float q1_g = (float)(q_byte_g & 0x0F);
                        float q2_g = (float)(q_byte_g >> 4);
                        float w1_g = d1_g * q1_g - m1_g;
                        float w2_g = d2_g * q2_g - m2_g;

                        const uint8_t *q_slice_u = blk_up->qs + chunk * 32;
                        uint8_t q_byte_u = q_slice_u[lane_id];
                        float q1_u = (float)(q_byte_u & 0x0F);
                        float q2_u = (float)(q_byte_u >> 4);
                        float w1_u = d1_u * q1_u - m1_u;
                        float w2_u = d2_u * q2_u - m2_u;

                        int off1 = chunk * 64 + lane_id;
                        int off2 = chunk * 64 + 32 + lane_id;

                        #pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            if (m < batch) {
                                float x1 = tok_ptrs[m][base_x + off1];
                                float x2 = tok_ptrs[m][base_x + off2];
                                gate_thread_sum[m] += w1_g * x1 + w2_g * x2;
                                up_thread_sum[m]   += w1_u * x1 + w2_u * x2;
                            }
                        }
                    }
                }

                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float g = warp_reduce_sum(gate_thread_sum[m]);
                        float u = warp_reduce_sum(up_thread_sum[m]);
                        if (lane_id == 0) {
                            out_ptrs[m][row] = gelu_tanh_scalar(g) * u;
                        }
                    }
                }
            }
        }
    }
}

__global__ void moe_gemv_down_sorted_kernel(
    const uint8_t *__restrict__ all_expert_weights,
    int expert_type,
    const float *__restrict__ acts_in,
    const int *__restrict__ expert_counts,
    const int *__restrict__ expert_offsets,
    const int *__restrict__ sorted_instances,
    float *__restrict__ out_down,
    int in_dim,
    int out_dim,
    int blocks_per_row,
    size_t expert_bytes
) {
    __shared__ alignas(16) float s_acts[4][704];

    int exp_id = blockIdx.y;
    int count = expert_counts[exp_id];
    if (count <= 0) return;

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row_base = (blockIdx.x * 8 + warp_id) * 4;
    int offset = expert_offsets[exp_id];

    for (int m_base = 0; m_base < count; m_base += 4) {
        int batch = count - m_base;
        if (batch > 4) batch = 4;

        float *tok_outs[4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m < batch) {
                int sorted_slot = offset + m_base + m;
                tok_outs[m] = out_down + (size_t)sorted_slot * out_dim;
                const float4 *src_tok = (const float4 *)(acts_in + (size_t)sorted_slot * in_dim);
                float4 *dst_tok = (float4 *)s_acts[m];
                for (int i = threadIdx.x; i < 176; i += blockDim.x) {
                    dst_tok[i] = src_tok[i];
                }
            } else {
                tok_outs[m] = NULL;
            }
        }
        __syncthreads();

        if (expert_type == 8) { // Q8_0
            const BlockQ8_0 *exp_W = (const BlockQ8_0 *)(all_expert_weights + (size_t)exp_id * expert_bytes);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int row = row_base + r;
                if (row >= out_dim) continue;

                const BlockQ8_0 *row_w = exp_W + (size_t)row * blocks_per_row;
                float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                int b = 0;
                for (; b + 1 < blocks_per_row; b += 2) {
                    const BlockQ8_0 *blk0 = row_w + b;
                    const BlockQ8_0 *blk1 = row_w + b + 1;
                    float w0 = __half2float(blk0->d) * (float)blk0->qs[lane_id];
                    float w1 = __half2float(blk1->d) * (float)blk1->qs[lane_id];

                    int act_off0 = b * 32 + lane_id;
                    int act_off1 = (b + 1) * 32 + lane_id;

                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) {
                            float xv0 = s_acts[m][act_off0];
                            float xv1 = s_acts[m][act_off1];
                            sum[m] += w0 * xv0 + w1 * xv1;
                        }
                    }
                }
                for (; b < blocks_per_row; b++) {
                    const BlockQ8_0 *blk = row_w + b;
                    float w = __half2float(blk->d) * (float)blk->qs[lane_id];
                    int act_off = b * 32 + lane_id;
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) {
                            sum[m] += w * s_acts[m][act_off];
                        }
                    }
                }

                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float rsum = warp_reduce_sum(sum[m]);
                        if (lane_id == 0) tok_outs[m][row] = rsum;
                    }
                }
            }
        } else { // Q5_0
            const BlockQ5_0 *exp_W = (const BlockQ5_0 *)(all_expert_weights + (size_t)exp_id * expert_bytes);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int row = row_base + r;
                if (row >= out_dim) continue;
                const BlockQ5_0 *row_w = exp_W + (size_t)row * blocks_per_row;
                float thread_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

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

                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) {
                            float xv = s_acts[m][base_x + lane_id];
                            thread_sum[m] += w * xv;
                        }
                    }
                }

                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float row_sum = warp_reduce_sum(thread_sum[m]);
                        if (lane_id == 0) {
                            tok_outs[m][row] = row_sum;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ void moe_accumulate_scaled_sorted_kernel(
    float *__restrict__ moe_accum,
    const float *__restrict__ sorted_down_out,
    const int *__restrict__ instance_to_sorted,
    const float *__restrict__ top_weights,
    int num_tokens,
    int dim
) {
    int token_idx = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx < num_tokens && d < dim) {
        float sum = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            int k = token_idx * 8 + e;
            int sorted_k = instance_to_sorted[k];
            if (sorted_k >= 0) {
                float w = top_weights[k];
                sum += sorted_down_out[(size_t)sorted_k * dim + d] * w;
            }
        }
        moe_accum[(size_t)token_idx * dim + d] = sum;
    }
}

// Phase 13 WP1: Fused MoE Down-Projection & Accumulation
// Directly accumulates down-projected activations multiplied by routing top_weights into moe_accum
// using hardware atomicAdd. Eliminates intermediate exp_down_buf DRAM roundtrip and saves 30 kernel dispatches per step.
__launch_bounds__(256, 4)
__global__ void moe_gemv_down_accum_sorted_kernel(
    const uint8_t *__restrict__ all_expert_weights,
    int expert_type,
    const float *__restrict__ acts_in,
    const int *__restrict__ expert_counts,
    const int *__restrict__ expert_offsets,
    const int *__restrict__ sorted_instances,
    const float *__restrict__ top_weights,
    float *__restrict__ moe_accum,
    int in_dim,
    int out_dim,
    int blocks_per_row,
    size_t expert_bytes
) {
    __shared__ alignas(16) float s_acts[4][704];

    int exp_id = blockIdx.y;
    int count = expert_counts[exp_id];
    if (count <= 0) return;

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row_base = (blockIdx.x * 8 + warp_id) * 4;
    int offset = expert_offsets[exp_id];

    for (int m_base = 0; m_base < count; m_base += 4) {
        int batch = count - m_base;
        if (batch > 4) batch = 4;

        int tok_indices[4];
        float weights[4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m < batch) {
                int sorted_slot = offset + m_base + m;
                int inst = sorted_instances[sorted_slot];
                tok_indices[m] = inst / 8;
                weights[m] = top_weights[inst];
                const float4 *src_tok = (const float4 *)(acts_in + (size_t)sorted_slot * in_dim);
                float4 *dst_tok = (float4 *)s_acts[m];
                for (int i = threadIdx.x; i < 176; i += blockDim.x) {
                    dst_tok[i] = src_tok[i];
                }
            } else {
                tok_indices[m] = -1;
                weights[m] = 0.0f;
            }
        }
        __syncthreads();

        if (expert_type == 8) { // Q8_0
            const BlockQ8_0 *exp_W = (const BlockQ8_0 *)(all_expert_weights + (size_t)exp_id * expert_bytes);

            int row0 = row_base + 0;
            int row1 = row_base + 1;
            int row2 = row_base + 2;
            int row3 = row_base + 3;
            bool active_r0 = (row0 < out_dim);
            bool active_r1 = (row1 < out_dim);
            bool active_r2 = (row2 < out_dim);
            bool active_r3 = (row3 < out_dim);

            const BlockQ8_0 *row_w0 = active_r0 ? (exp_W + (size_t)row0 * blocks_per_row) : NULL;
            const BlockQ8_0 *row_w1 = active_r1 ? (exp_W + (size_t)row1 * blocks_per_row) : NULL;
            const BlockQ8_0 *row_w2 = active_r2 ? (exp_W + (size_t)row2 * blocks_per_row) : NULL;
            const BlockQ8_0 *row_w3 = active_r3 ? (exp_W + (size_t)row3 * blocks_per_row) : NULL;

            float sum0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sum1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sum2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sum3[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            int b = 0;
            for (; b + 1 < blocks_per_row; b += 2) {
                int act_off0 = b * 32 + lane_id;
                int act_off1 = (b + 1) * 32 + lane_id;

                float xv0[4], xv1[4];
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        xv0[m] = s_acts[m][act_off0];
                        xv1[m] = s_acts[m][act_off1];
                    }
                }

                if (active_r0) {
                    const BlockQ8_0 *blk0 = row_w0 + b;
                    const BlockQ8_0 *blk1 = row_w0 + b + 1;
                    float w0 = __half2float(blk0->d) * (float)blk0->qs[lane_id];
                    float w1 = __half2float(blk1->d) * (float)blk1->qs[lane_id];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum0[m] += w0 * xv0[m] + w1 * xv1[m];
                    }
                }

                if (active_r1) {
                    const BlockQ8_0 *blk0 = row_w1 + b;
                    const BlockQ8_0 *blk1 = row_w1 + b + 1;
                    float w0 = __half2float(blk0->d) * (float)blk0->qs[lane_id];
                    float w1 = __half2float(blk1->d) * (float)blk1->qs[lane_id];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum1[m] += w0 * xv0[m] + w1 * xv1[m];
                    }
                }

                if (active_r2) {
                    const BlockQ8_0 *blk0 = row_w2 + b;
                    const BlockQ8_0 *blk1 = row_w2 + b + 1;
                    float w0 = __half2float(blk0->d) * (float)blk0->qs[lane_id];
                    float w1 = __half2float(blk1->d) * (float)blk1->qs[lane_id];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum2[m] += w0 * xv0[m] + w1 * xv1[m];
                    }
                }

                if (active_r3) {
                    const BlockQ8_0 *blk0 = row_w3 + b;
                    const BlockQ8_0 *blk1 = row_w3 + b + 1;
                    float w0 = __half2float(blk0->d) * (float)blk0->qs[lane_id];
                    float w1 = __half2float(blk1->d) * (float)blk1->qs[lane_id];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum3[m] += w0 * xv0[m] + w1 * xv1[m];
                    }
                }
            }

            for (; b < blocks_per_row; b++) {
                int act_off = b * 32 + lane_id;
                float xv[4];
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        xv[m] = s_acts[m][act_off];
                    }
                }

                if (active_r0) {
                    const BlockQ8_0 *blk = row_w0 + b;
                    float w = __half2float(blk->d) * (float)blk->qs[lane_id];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum0[m] += w * xv[m];
                    }
                }

                if (active_r1) {
                    const BlockQ8_0 *blk = row_w1 + b;
                    float w = __half2float(blk->d) * (float)blk->qs[lane_id];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum1[m] += w * xv[m];
                    }
                }

                if (active_r2) {
                    const BlockQ8_0 *blk = row_w2 + b;
                    float w = __half2float(blk->d) * (float)blk->qs[lane_id];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum2[m] += w * xv[m];
                    }
                }

                if (active_r3) {
                    const BlockQ8_0 *blk = row_w3 + b;
                    float w = __half2float(blk->d) * (float)blk->qs[lane_id];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum3[m] += w * xv[m];
                    }
                }
            }

            if (active_r0) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float rsum = warp_reduce_sum(sum0[m]);
                        if (lane_id == 0) {
                            atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row0], rsum * weights[m]);
                        }
                    }
                }
            }

            if (active_r1) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float rsum = warp_reduce_sum(sum1[m]);
                        if (lane_id == 0) {
                            atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row1], rsum * weights[m]);
                        }
                    }
                }
            }

            if (active_r2) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float rsum = warp_reduce_sum(sum2[m]);
                        if (lane_id == 0) {
                            atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row2], rsum * weights[m]);
                        }
                    }
                }
            }

            if (active_r3) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float rsum = warp_reduce_sum(sum3[m]);
                        if (lane_id == 0) {
                            atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row3], rsum * weights[m]);
                        }
                    }
                }
            }
        } else { // Q5_0
            const BlockQ5_0 *exp_W = (const BlockQ5_0 *)(all_expert_weights + (size_t)exp_id * expert_bytes);

            int row0 = row_base + 0;
            int row1 = row_base + 1;
            int row2 = row_base + 2;
            int row3 = row_base + 3;
            bool active_r0 = (row0 < out_dim);
            bool active_r1 = (row1 < out_dim);
            bool active_r2 = (row2 < out_dim);
            bool active_r3 = (row3 < out_dim);

            const BlockQ5_0 *row_w0 = active_r0 ? (exp_W + (size_t)row0 * blocks_per_row) : NULL;
            const BlockQ5_0 *row_w1 = active_r1 ? (exp_W + (size_t)row1 * blocks_per_row) : NULL;
            const BlockQ5_0 *row_w2 = active_r2 ? (exp_W + (size_t)row2 * blocks_per_row) : NULL;
            const BlockQ5_0 *row_w3 = active_r3 ? (exp_W + (size_t)row3 * blocks_per_row) : NULL;

            float sum0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sum1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sum2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sum3[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            int byte_idx = (lane_id < 16) ? lane_id : (lane_id - 16);

            for (int b = 0; b < blocks_per_row; b++) {
                int act_off = b * 32 + lane_id;
                float xv[4];
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        xv[m] = s_acts[m][act_off];
                    }
                }

                if (active_r0) {
                    const BlockQ5_0 *blk = row_w0 + b;
                    float d = __half2float(blk->d);
                    uint16_t qh_lo = *(const uint16_t*)(blk->qh + 0);
                    uint16_t qh_hi = *(const uint16_t*)(blk->qh + 2);
                    uint32_t qh = (uint32_t)qh_lo | ((uint32_t)qh_hi << 16);
                    uint8_t q_byte = blk->qs[byte_idx];
                    int q_nibble = (lane_id < 16) ? (q_byte & 0x0F) : (q_byte >> 4);
                    uint8_t bit = (uint8_t)((qh >> lane_id) & 1);
                    float w = d * (float)((q_nibble | (bit << 4)) - 16);
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum0[m] += w * xv[m];
                    }
                }

                if (active_r1) {
                    const BlockQ5_0 *blk = row_w1 + b;
                    float d = __half2float(blk->d);
                    uint16_t qh_lo = *(const uint16_t*)(blk->qh + 0);
                    uint16_t qh_hi = *(const uint16_t*)(blk->qh + 2);
                    uint32_t qh = (uint32_t)qh_lo | ((uint32_t)qh_hi << 16);
                    uint8_t q_byte = blk->qs[byte_idx];
                    int q_nibble = (lane_id < 16) ? (q_byte & 0x0F) : (q_byte >> 4);
                    uint8_t bit = (uint8_t)((qh >> lane_id) & 1);
                    float w = d * (float)((q_nibble | (bit << 4)) - 16);
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum1[m] += w * xv[m];
                    }
                }

                if (active_r2) {
                    const BlockQ5_0 *blk = row_w2 + b;
                    float d = __half2float(blk->d);
                    uint16_t qh_lo = *(const uint16_t*)(blk->qh + 0);
                    uint16_t qh_hi = *(const uint16_t*)(blk->qh + 2);
                    uint32_t qh = (uint32_t)qh_lo | ((uint32_t)qh_hi << 16);
                    uint8_t q_byte = blk->qs[byte_idx];
                    int q_nibble = (lane_id < 16) ? (q_byte & 0x0F) : (q_byte >> 4);
                    uint8_t bit = (uint8_t)((qh >> lane_id) & 1);
                    float w = d * (float)((q_nibble | (bit << 4)) - 16);
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum2[m] += w * xv[m];
                    }
                }

                if (active_r3) {
                    const BlockQ5_0 *blk = row_w3 + b;
                    float d = __half2float(blk->d);
                    uint16_t qh_lo = *(const uint16_t*)(blk->qh + 0);
                    uint16_t qh_hi = *(const uint16_t*)(blk->qh + 2);
                    uint32_t qh = (uint32_t)qh_lo | ((uint32_t)qh_hi << 16);
                    uint8_t q_byte = blk->qs[byte_idx];
                    int q_nibble = (lane_id < 16) ? (q_byte & 0x0F) : (q_byte >> 4);
                    uint8_t bit = (uint8_t)((qh >> lane_id) & 1);
                    float w = d * (float)((q_nibble | (bit << 4)) - 16);
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) sum3[m] += w * xv[m];
                    }
                }
            }

            if (active_r0) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float rsum = warp_reduce_sum(sum0[m]);
                        if (lane_id == 0) {
                            atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row0], rsum * weights[m]);
                        }
                    }
                }
            }

            if (active_r1) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float rsum = warp_reduce_sum(sum1[m]);
                        if (lane_id == 0) {
                            atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row1], rsum * weights[m]);
                        }
                    }
                }
            }

            if (active_r2) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float rsum = warp_reduce_sum(sum2[m]);
                        if (lane_id == 0) {
                            atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row2], rsum * weights[m]);
                        }
                    }
                }
            }

            if (active_r3) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < batch) {
                        float rsum = warp_reduce_sum(sum3[m]);
                        if (lane_id == 0) {
                            atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row3], rsum * weights[m]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Phase 13 WP2: Active-Expert Persistent MoE Down-Projection & Accumulation Kernel
// Launches a static hardware-matched grid (84 CTAs on Ampere SM 8.6) pulling active expert row tiles dynamically
// from work_counter via atomicAdd. Eliminates ~72,000 empty Down-Projection CTA launches per canvas step.
__launch_bounds__(256, 3)
__global__ void moe_gemv_down_accum_persistent_kernel(
    const uint8_t *__restrict__ all_expert_weights,
    int expert_type,
    const float *__restrict__ acts_in,
    const int *__restrict__ expert_counts,
    const int *__restrict__ expert_offsets,
    const int *__restrict__ sorted_instances,
    const float *__restrict__ top_weights,
    float *__restrict__ moe_accum,
    const int *__restrict__ active_experts,
    const int *__restrict__ num_active_experts_ptr,
    int *__restrict__ work_counter,
    int in_dim,
    int out_dim,
    int blocks_per_row,
    size_t expert_bytes
) {
    __shared__ alignas(16) float s_acts[4][704];
    __shared__ int s_num_active;
    __shared__ int s_task_id;

    if (threadIdx.x == 0) {
        s_num_active = *num_active_experts_ptr;
    }
    __syncthreads();

    int num_active = s_num_active;
    if (num_active <= 0) return;

    int num_row_tiles = (out_dim + 63) / 64; // 32
    int total_tasks = num_active * num_row_tiles;

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    while (true) {
        if (threadIdx.x == 0) {
            s_task_id = atomicAdd(work_counter, 1);
        }
        __syncthreads();

        int task_id = s_task_id;
        if (task_id >= total_tasks) break;

        int active_idx = task_id / num_row_tiles;
        int row_tile = task_id % num_row_tiles;
        int exp_id = active_experts[active_idx];
        int count = expert_counts[exp_id];
        int offset = expert_offsets[exp_id];

        int row_base = (row_tile * 8 + warp_id) * 8;

        for (int m_base = 0; m_base < count; m_base += 4) {
            int batch = count - m_base;
            if (batch > 4) batch = 4;

            int tok_indices[4];
            float weights[4];
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < batch) {
                    int sorted_slot = offset + m_base + m;
                    int inst = sorted_instances[sorted_slot];
                    tok_indices[m] = inst / 8;
                    weights[m] = top_weights[inst];
                    const float4 *src_tok = (const float4 *)(acts_in + (size_t)sorted_slot * in_dim);
                    float4 *dst_tok = (float4 *)s_acts[m];
                    for (int i = threadIdx.x; i < 176; i += blockDim.x) {
                        dst_tok[i] = src_tok[i];
                    }
                } else {
                    tok_indices[m] = -1;
                    weights[m] = 0.0f;
                }
            }
            __syncthreads();

            if (expert_type == 8) { // Q8_0
                const BlockQ8_0 *exp_W = (const BlockQ8_0 *)(all_expert_weights + (size_t)exp_id * expert_bytes);
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int row = row_base + r;
                    if (row >= out_dim) continue;

                    const BlockQ8_0 *row_w = exp_W + (size_t)row * blocks_per_row;
                    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                    for (int b = 0; b < blocks_per_row; b++) {
                        const BlockQ8_0 *blk = row_w + b;
                        float w = __half2float(blk->d) * (float)blk->qs[lane_id];

                        #pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            if (m < batch) {
                                float xv = s_acts[m][b * 32 + lane_id];
                                sum[m] += w * xv;
                            }
                        }
                    }

                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) {
                            float rsum = warp_reduce_sum(sum[m]);
                            if (lane_id == 0) {
                                atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row], rsum * weights[m]);
                            }
                        }
                    }
                }
            } else { // Q5_0
                const BlockQ5_0 *exp_W = (const BlockQ5_0 *)(all_expert_weights + (size_t)exp_id * expert_bytes);
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int row = row_base + r;
                    if (row >= out_dim) continue;
                    const BlockQ5_0 *row_w = exp_W + (size_t)row * blocks_per_row;
                    float thread_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

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

                        #pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            if (m < batch) {
                                float xv = s_acts[m][base_x + lane_id];
                                thread_sum[m] += w * xv;
                            }
                        }
                    }

                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < batch) {
                            float row_sum = warp_reduce_sum(thread_sum[m]);
                            if (lane_id == 0) {
                                atomicAdd(&moe_accum[(size_t)tok_indices[m] * out_dim + row], row_sum * weights[m]);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
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

int zeno_cuda_moe_gemv_gate_up_batched(
    const uint8_t *all_expert_weights,
    const float *tokens_in,
    const int *top_indices,
    float *out_gate_up,
    int num_tokens,
    int in_dim,
    int out_dim,
    int device_id,
    void *stream
) {
    if (!all_expert_weights || !tokens_in || !top_indices || !out_gate_up) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens <= 0 || in_dim <= 0 || out_dim <= 0 || in_dim % 256 != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    int blocks_per_row = in_dim / 256;
    size_t expert_bytes = (size_t)out_dim * (size_t)blocks_per_row * sizeof(BlockQ4_K);

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((out_dim + 31) / 32);
    unsigned int grid_y = (unsigned int)(num_tokens * 8);
    dim3 grid(grid_x, grid_y);
    cudaStream_t custream = (cudaStream_t)stream;

    moe_gemv_gate_up_batched_kernel<<<grid, block_size, 0, custream>>>(
        all_expert_weights,
        tokens_in,
        top_indices,
        out_gate_up,
        num_tokens,
        in_dim,
        out_dim,
        blocks_per_row,
        expert_bytes
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_geglu_batched(
    const float *gate_up,
    float *act_out,
    int num_instances,
    int half_dim,
    int device_id,
    void *stream
) {
    if (!gate_up || !act_out) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_instances <= 0 || half_dim <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    size_t total = (size_t)num_instances * half_dim;
    const int block_size = 256;
    unsigned int num_blocks = (unsigned int)((total + block_size - 1) / block_size);
    cudaStream_t custream = (cudaStream_t)stream;

    moe_geglu_batched_kernel<<<num_blocks, block_size, 0, custream>>>(
        gate_up,
        act_out,
        num_instances,
        half_dim
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_gemv_down_batched(
    const uint8_t *all_expert_weights,
    int expert_type,
    const float *acts_in,
    const int *top_indices,
    float *out_down,
    int num_tokens,
    int in_dim,
    int out_dim,
    int device_id,
    void *stream
) {
    if (!all_expert_weights || !acts_in || !top_indices || !out_down) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens <= 0 || in_dim <= 0 || out_dim <= 0 || in_dim % 32 != 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (expert_type != 6 && expert_type != 8) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    int blocks_per_row = in_dim / 32;
    size_t block_size_bytes = (expert_type == 6) ? sizeof(BlockQ5_0) : sizeof(BlockQ8_0);
    size_t expert_bytes = (size_t)out_dim * (size_t)blocks_per_row * block_size_bytes;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((out_dim + 31) / 32);
    unsigned int grid_y = (unsigned int)(num_tokens * 8);
    dim3 grid(grid_x, grid_y);
    cudaStream_t custream = (cudaStream_t)stream;

    moe_gemv_down_batched_kernel<<<grid, block_size, 0, custream>>>(
        all_expert_weights,
        expert_type,
        acts_in,
        top_indices,
        out_down,
        num_tokens,
        in_dim,
        out_dim,
        blocks_per_row,
        expert_bytes
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_accumulate_scaled_batched(
    float *moe_accum,
    const float *down_out,
    const float *top_weights,
    int num_tokens,
    int dim,
    int device_id,
    void *stream
) {
    if (!moe_accum || !down_out || !top_weights) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens <= 0 || dim <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((dim + block_size - 1) / block_size);
    dim3 grid(grid_x, (unsigned int)num_tokens);
    cudaStream_t custream = (cudaStream_t)stream;

    moe_accumulate_scaled_batched_kernel<<<grid, block_size, 0, custream>>>(
        moe_accum,
        down_out,
        top_weights,
        num_tokens,
        dim
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_build_expert_bins(
    const int *top_indices,
    int *expert_counts,
    int *expert_offsets,
    int *sorted_instances,
    int *instance_to_sorted,
    int *active_experts,
    int *num_active_experts,
    int num_tokens,
    int device_id,
    void *stream
) {
    if (!top_indices || !expert_counts || !expert_offsets || !sorted_instances || !instance_to_sorted) {
        return ZENO_CUDA_ERR_INVALID_PARAM;
    }
    if (num_tokens <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaStream_t custream = (cudaStream_t)stream;
    moe_build_expert_bins_kernel<<<1, 128, 0, custream>>>(
        top_indices,
        expert_counts,
        expert_offsets,
        sorted_instances,
        instance_to_sorted,
        active_experts,
        num_active_experts,
        num_tokens
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_gemv_gate_up_sorted(
    const uint8_t *all_expert_weights,
    const float *tokens_in,
    const int *expert_counts,
    const int *expert_offsets,
    const int *sorted_instances,
    float *out_gate_up,
    int num_tokens,
    int in_dim,
    int out_dim,
    int device_id,
    void *stream
) {
    if (!all_expert_weights || !tokens_in || !expert_counts || !expert_offsets || !sorted_instances || !out_gate_up) {
        return ZENO_CUDA_ERR_INVALID_PARAM;
    }
    if (num_tokens <= 0 || in_dim <= 0 || out_dim <= 0 || in_dim % 256 != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    int blocks_per_row = in_dim / 256;
    size_t expert_bytes = (size_t)out_dim * (size_t)blocks_per_row * sizeof(BlockQ4_K);

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((out_dim + 63) / 64);
    unsigned int grid_y = 128;
    dim3 grid(grid_x, grid_y);
    cudaStream_t custream = (cudaStream_t)stream;

    moe_gemv_gate_up_sorted_kernel<<<grid, block_size, 0, custream>>>(
        all_expert_weights,
        tokens_in,
        expert_counts,
        expert_offsets,
        sorted_instances,
        out_gate_up,
        in_dim,
        out_dim,
        blocks_per_row,
        expert_bytes
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_gemv_gate_up_geglu_sorted(
    const uint8_t *all_expert_weights,
    const float *tokens_in,
    const int *expert_counts,
    const int *expert_offsets,
    const int *sorted_instances,
    float *out_act,
    int num_tokens,
    int in_dim,
    int half_dim,
    int device_id,
    void *stream
) {
    if (!all_expert_weights || !tokens_in || !expert_counts || !expert_offsets || !sorted_instances || !out_act) {
        return ZENO_CUDA_ERR_INVALID_PARAM;
    }
    if (num_tokens <= 0 || in_dim <= 0 || half_dim <= 0 || in_dim % 256 != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    int blocks_per_row = in_dim / 256;
    size_t expert_bytes = (size_t)(half_dim * 2) * (size_t)blocks_per_row * sizeof(BlockQ4_K);

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((half_dim + 15) / 16);
    unsigned int grid_y = 128;
    dim3 grid(grid_x, grid_y);
    cudaStream_t custream = (cudaStream_t)stream;

    moe_gemv_gate_up_geglu_sorted_kernel<<<grid, block_size, 0, custream>>>(
        all_expert_weights,
        tokens_in,
        expert_counts,
        expert_offsets,
        sorted_instances,
        out_act,
        in_dim,
        half_dim,
        blocks_per_row,
        expert_bytes
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_gemv_down_sorted(
    const uint8_t *all_expert_weights,
    int expert_type,
    const float *acts_in,
    const int *expert_counts,
    const int *expert_offsets,
    const int *sorted_instances,
    float *out_down,
    int num_tokens,
    int in_dim,
    int out_dim,
    int device_id,
    void *stream
) {
    if (!all_expert_weights || !acts_in || !expert_counts || !expert_offsets || !sorted_instances || !out_down) {
        return ZENO_CUDA_ERR_INVALID_PARAM;
    }
    if (num_tokens <= 0 || in_dim <= 0 || out_dim <= 0 || in_dim % 32 != 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (expert_type != 6 && expert_type != 8) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    int blocks_per_row = in_dim / 32;
    size_t block_size_bytes = (expert_type == 6) ? sizeof(BlockQ5_0) : sizeof(BlockQ8_0);
    size_t expert_bytes = (size_t)out_dim * (size_t)blocks_per_row * block_size_bytes;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((out_dim + 31) / 32);
    unsigned int grid_y = 128;
    dim3 grid(grid_x, grid_y);
    cudaStream_t custream = (cudaStream_t)stream;

    moe_gemv_down_sorted_kernel<<<grid, block_size, 0, custream>>>(
        all_expert_weights,
        expert_type,
        acts_in,
        expert_counts,
        expert_offsets,
        sorted_instances,
        out_down,
        in_dim,
        out_dim,
        blocks_per_row,
        expert_bytes
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_accumulate_scaled_sorted(
    float *moe_accum,
    const float *sorted_down_out,
    const int *instance_to_sorted,
    const float *top_weights,
    int num_tokens,
    int dim,
    int device_id,
    void *stream
) {
    if (!moe_accum || !sorted_down_out || !instance_to_sorted || !top_weights) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (num_tokens <= 0 || dim <= 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((dim + block_size - 1) / block_size);
    dim3 grid(grid_x, (unsigned int)num_tokens);
    cudaStream_t custream = (cudaStream_t)stream;

    moe_accumulate_scaled_sorted_kernel<<<grid, block_size, 0, custream>>>(
        moe_accum,
        sorted_down_out,
        instance_to_sorted,
        top_weights,
        num_tokens,
        dim
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_gemv_down_accum_sorted(
    const uint8_t *all_expert_weights,
    int expert_type,
    const float *acts_in,
    const int *expert_counts,
    const int *expert_offsets,
    const int *sorted_instances,
    const float *top_weights,
    float *moe_accum,
    int num_tokens,
    int in_dim,
    int out_dim,
    int device_id,
    void *stream
) {
    if (!all_expert_weights || !acts_in || !expert_counts || !expert_offsets || !sorted_instances || !top_weights || !moe_accum) {
        return ZENO_CUDA_ERR_INVALID_PARAM;
    }
    if (num_tokens <= 0 || in_dim <= 0 || out_dim <= 0 || in_dim % 32 != 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (expert_type != 6 && expert_type != 8) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaStream_t custream = (cudaStream_t)stream;
    size_t accum_bytes = (size_t)num_tokens * (size_t)out_dim * sizeof(float);
    cudaError_t mem_err = cudaMemsetAsync(moe_accum, 0, accum_bytes, custream);
    if (mem_err != cudaSuccess) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_DRIVER_ERROR);
    }

    int blocks_per_row = in_dim / 32;
    size_t block_size_bytes = (expert_type == 6) ? sizeof(BlockQ5_0) : sizeof(BlockQ8_0);
    size_t expert_bytes = (size_t)out_dim * (size_t)blocks_per_row * block_size_bytes;

    const int block_size = 256;
    unsigned int grid_x = (unsigned int)((out_dim + 31) / 32);
    unsigned int grid_y = 128;
    dim3 grid(grid_x, grid_y);

    moe_gemv_down_accum_sorted_kernel<<<grid, block_size, 0, custream>>>(
        all_expert_weights,
        expert_type,
        acts_in,
        expert_counts,
        expert_offsets,
        sorted_instances,
        top_weights,
        moe_accum,
        in_dim,
        out_dim,
        blocks_per_row,
        expert_bytes
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_gemv_gate_up_geglu_persistent(
    const uint8_t *all_expert_weights,
    const float *tokens_in,
    const int *expert_counts,
    const int *expert_offsets,
    const int *sorted_instances,
    const int *active_experts,
    const int *num_active_experts,
    int *work_counter,
    float *out_act,
    int num_tokens,
    int in_dim,
    int half_dim,
    int device_id,
    void *stream
) {
    if (!all_expert_weights || !tokens_in || !expert_counts || !expert_offsets ||
        !sorted_instances || !active_experts || !num_active_experts || !work_counter || !out_act) {
        return ZENO_CUDA_ERR_INVALID_PARAM;
    }
    if (num_tokens <= 0 || in_dim <= 0 || half_dim <= 0 || in_dim % 256 != 0) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaStream_t custream = (cudaStream_t)stream;
    cudaError_t mem_err = cudaMemsetAsync(work_counter, 0, sizeof(int), custream);
    if (mem_err != cudaSuccess) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_DRIVER_ERROR);
    }

    int blocks_per_row = in_dim / 256;
    size_t expert_bytes = (size_t)(half_dim * 2) * (size_t)blocks_per_row * sizeof(BlockQ4_K);

    const int block_size = 256;
    int num_sms = 28;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
    dim3 grid((unsigned int)(num_sms * 3));

    moe_gemv_gate_up_geglu_persistent_kernel<<<grid, block_size, 0, custream>>>(
        all_expert_weights,
        tokens_in,
        expert_counts,
        expert_offsets,
        sorted_instances,
        active_experts,
        num_active_experts,
        work_counter,
        out_act,
        in_dim,
        half_dim,
        blocks_per_row,
        expert_bytes
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

int zeno_cuda_moe_gemv_down_accum_persistent(
    const uint8_t *all_expert_weights,
    int expert_type,
    const float *acts_in,
    const int *expert_counts,
    const int *expert_offsets,
    const int *sorted_instances,
    const float *top_weights,
    const int *active_experts,
    const int *num_active_experts,
    int *work_counter,
    float *moe_accum,
    int num_tokens,
    int in_dim,
    int out_dim,
    int device_id,
    void *stream
) {
    if (!all_expert_weights || !acts_in || !expert_counts || !expert_offsets ||
        !sorted_instances || !top_weights || !active_experts || !num_active_experts || !work_counter || !moe_accum) {
        return ZENO_CUDA_ERR_INVALID_PARAM;
    }
    if (num_tokens <= 0 || in_dim <= 0 || out_dim <= 0 || in_dim % 32 != 0) return ZENO_CUDA_ERR_INVALID_PARAM;
    if (expert_type != 6 && expert_type != 8) return ZENO_CUDA_ERR_INVALID_PARAM;

    int orig = 0;
    int rc = set_cuda_device(device_id, &orig);
    if (rc != ZENO_CUDA_SUCCESS) return rc;

    cudaStream_t custream = (cudaStream_t)stream;
    size_t accum_bytes = (size_t)num_tokens * (size_t)out_dim * sizeof(float);
    cudaError_t mem_err = cudaMemsetAsync(moe_accum, 0, accum_bytes, custream);
    if (mem_err != cudaSuccess) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_DRIVER_ERROR);
    }
    mem_err = cudaMemsetAsync(work_counter, 0, sizeof(int), custream);
    if (mem_err != cudaSuccess) {
        return restore_cuda_device(orig, ZENO_CUDA_ERR_DRIVER_ERROR);
    }

    int blocks_per_row = in_dim / 32;
    size_t block_size_bytes = (expert_type == 6) ? sizeof(BlockQ5_0) : sizeof(BlockQ8_0);
    size_t expert_bytes = (size_t)out_dim * (size_t)blocks_per_row * block_size_bytes;

    const int block_size = 256;
    int num_sms = 28;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
    dim3 grid((unsigned int)(num_sms * 3));

    moe_gemv_down_accum_persistent_kernel<<<grid, block_size, 0, custream>>>(
        all_expert_weights,
        expert_type,
        acts_in,
        expert_counts,
        expert_offsets,
        sorted_instances,
        top_weights,
        moe_accum,
        active_experts,
        num_active_experts,
        work_counter,
        in_dim,
        out_dim,
        blocks_per_row,
        expert_bytes
    );

    cudaError_t err = cudaGetLastError();
    int op_rc = (err != cudaSuccess) ? ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED : ZENO_CUDA_SUCCESS;
    return restore_cuda_device(orig, op_rc);
}

} // extern "C"

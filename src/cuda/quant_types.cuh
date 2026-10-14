#ifndef ZENO_QUANT_TYPES_CUH
#define ZENO_QUANT_TYPES_CUH

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

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
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

__device__ inline void precompute_q4k_scales(
    const BlockQ4_K *blk,
    int lane_id,
    float *my_d,
    float *my_m
) {
    float d = __half2float(blk->d);
    float min = __half2float(blk->dmin);
    if (lane_id < 8) {
        uint8_t sc, m;
        get_scale_min_k4(lane_id, blk->scales, &sc, &m);
        *my_d = d * (float)sc;
        *my_m = min * (float)m;
    } else {
        *my_d = 0.0f;
        *my_m = 0.0f;
    }
}

__device__ inline float gelu_tanh_scalar(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

#endif // ZENO_QUANT_TYPES_CUH

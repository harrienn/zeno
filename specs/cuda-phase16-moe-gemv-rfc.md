# RFC: CUDA Phase 16 — MoE GEMV Shared-Memory Hoisting, Scale Broadcast & Memory Alignment Optimization for Sub-100ms Canvas Execution

**Date**: October 2026  
**Status**: Proposed / Under Review  
**Target Goal**: Reduce $C=16$ canvas execution latency from **138.64 ms** to **$< 100 \text{ ms}$** by addressing the dominant bottleneck: MoE GEMV projections (accounting for **80 ms / 58%** of total forward time).

---

## 1. Context & Micro-Profiling Analysis

In Phase 15, intra-layer kernel fusion successfully collapsed elementwise overhead from 8.5% down to $< 1.2\%$ and eliminated 270 dispatches per canvas pass, bringing $C=16$ latency down to **138.64 ms** (spread 0.08 ms).

Nsight Systems profiling of the post-Phase 15 runtime reveals the exact GPU execution profile:
```text
Kernel                                    Count   Avg Latency   Total Time   % GPU Time
---------------------------------------------------------------------------------------
moe_gemv_gate_up_geglu_sorted_kernel         30      4.08 ms      122.3 ms        35.8%
gemv_q4k_batched_4x_kernel                  102      0.83 ms       84.5 ms        24.7%
moe_gemv_down_accum_sorted_kernel            30      2.57 ms       77.1 ms        22.5%
dense_gemv_gate_up_geglu_q4k_kernel          30      1.00 ms       30.0 ms         8.8%
gemv_q5_0 / q8_0 / q6k                       43      0.52 ms       22.5 ms         6.6%
attention_kernel                             30      0.06 ms        1.7 ms         0.5%
fused_head_norm_rope_kernel                  60      0.005 ms       0.3 ms         0.1%
fused_post_ffn_combine_kernel                30      0.008 ms       0.25 ms        0.1%
fused_pre_ffn_norm_kernel                    30      0.005 ms       0.15 ms        0.05%
```

### Key Observation:
- `moe_gemv_gate_up_geglu_sorted_kernel` (35.8%, ~49 ms on dual GPU) + `moe_gemv_down_accum_sorted_kernel` (22.5%, ~31 ms on dual GPU) together account for **~80 ms** of the remaining 138 ms!
- The computational work for these kernels is $< 0.1 \text{ ms}$, while memory reads at 360 GB/s peak bandwidth should theoretically take $\sim 0.35 \text{ ms}$. The kernels currently exhibit a **7× to 10× memory efficiency gap**.

---

## 2. Root Cause Analysis

### 2.1 MoE Down-Projection Redundant Shared-Memory Traffic
In [`src/cuda/moe.cu`](file:///hdd/Projects/zeno/src/cuda/moe.cu#L1120-L1145):
```cuda
for (int r = 0; r < 8; ++r) {
    int row = row_base + r;
    for (int b = 0; b < blocks_per_row; b++) {
        const BlockQ8_0 *blk = row_w + b;
        float w = __half2float(blk->d) * (float)blk->qs[lane_id];
        for (int m = 0; m < 4; ++m) {
            float xv = s_acts[m][b * 32 + lane_id]; // <-- READ FROM SHARED MEMORY 8 TIMES!
            sum[m] += w * xv;
        }
    }
}
```
1. **8× Redundant Shared-Memory Reads**: For each of the 8 rows $r \in [0..7]$ assigned to a warp, `s_acts[m][b * 32 + lane_id]` is read from shared memory repeatedly. Total reads: $8 \times 22 \times \text{batch} = 176 \times \text{batch}$ reads per warp.
2. **32-Way Redundant Half Loads**: In every block $b$, all 32 lanes independently read `blk->d` (a 2-byte half) from unaligned DRAM addresses, causing 32-way broadcast conflicts in L1 cache.

### 2.2 MoE Gate/Up Projection Redundant Shared-Memory Traffic
In [`src/cuda/moe.cu`](file:///hdd/Projects/zeno/src/cuda/moe.cu#L668-L718):
```cuda
for (int r = 0; r < 4; ++r) {
    for (int chunk = 0; chunk < 4; chunk++) {
        // ... scale shuffles ...
        for (int m = 0; m < 4; ++m) {
            float x1 = s_x[m][off1]; // <-- READ FROM SHARED MEMORY 4 TIMES!
            float x2 = s_x[m][off2]; // <-- READ FROM SHARED MEMORY 4 TIMES!
            gate_thread_sum[r][m] += w1_g * x1 + w2_g * x2;
            up_thread_sum[r][m]   += w1_u * x1 + w2_u * x2;
        }
    }
}
```
1. **4× Redundant Shared-Memory Reads**: For each of the 4 rows $r \in [0..3]$, `s_x[m][off1]` and `s_x[m][off2]` are re-read from shared memory. Total reads: $4 \times 4 \times 11 \times 2 \times \text{batch} = 352 \times \text{batch}$ reads per warp.
2. **Redundant Scale Precomputations**: `precompute_q4k_scales` and subsequent shuffle networks run redundantly inside row loops.

---

## 3. Work Packages

### Work Package 1 (WP1): MoE Down-Accum Shared-Memory Hoisting & Warp Broadcast
- Hoist activation loading out of the row loop:
  Loop over blocks $b \in [0 .. \text{blocks\_per\_row} - 1]$ on the outside, loading `xv[m] = s_acts[m][b * 32 + lane_id]` into registers once.
- Broadcast `blk->d`:
  Have `lane_id == 0` load `__half2float(blk->d)` and broadcast across the warp using `__shfl_sync(0xffffffff, d, 0)`.
- Accumulate across all 8 rows in registers:
  `sum[8][4]` in registers, reducing shared memory load traffic by **87.5%** (from 176 down to 22 loads per instance).
- Perform `warp_reduce_sum` across the 8 rows after block iteration completes.

### Work Package 2 (WP2): MoE Gate/Up Shared-Memory Hoisting & Scale Shuffling
- Invert the chunk/row inner loops or hoist `s_x` loading:
  For each chunk, load `x1[m] = s_x[m][off1]` and `x2[m] = s_x[m][off2]` into registers once before computing across rows $r \in [0..3]$.
- Hoist `s_x` reads from $4 \times 4 = 16$ reads down to $4$ reads per chunk (a **75% reduction** in shared-memory load transactions).
- Keep `BlockQ4_K` weights resident in registers and optimize warp shuffle paths.

### Work Package 3 (WP3): Inactive Expert CTA Culling / Compact Grid Mapping
- In `moe_gemv_down_accum_sorted_kernel` and `moe_gemv_gate_up_geglu_sorted_kernel`, evaluate packing active experts to eliminate launching 3,000+ empty CTAs whose `expert_counts[exp_id] == 0`.
- Maintain static CTA grid ordering to preserve L2 cache locality discovered in Phase 13.

---

## 4. Verification & Invariants

1. **Exact Numerical Parity**:
   - Max difference vs CPU reference $\le 10^{-4}$ in standalone unit parity tests.
   - Exact discrete decision match ($5/5$, $100\%$) across all golden oracle fixtures (`./tools/verify_all_oracles.sh --cuda`).
2. **CPU Isolation**:
   - `zig build test -Dcuda=false` passes without dependencies.
3. **HTTP Server Parity**:
   - 15/15 integration tests pass (`python3 tools/test_server.py --cuda`).
4. **Latency Target**:
   - Measure $C=16$ median latency reduction toward $< 100 \text{ ms}$.

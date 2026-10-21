# RFC: CUDA Phase 16 — MoE GEMV Warp Restructuring, Memory Streaming & Paired Block Optimization

**Date**: October 2026  
**Status**: Delivered / Completed  
**Target Goal**: Reduce $C=16$ canvas execution latency from **138.64 ms** toward $< 100 \text{ ms}$ by addressing the dominant bottleneck: MoE GEMV projections (accounting for **80 ms / 58%** of total forward time).

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
- `moe_gemv_gate_up_geglu_sorted_kernel` (35.8%, ~49 ms on dual GPU) + `moe_gemv_down_accum_sorted_kernel` (22.5%, ~31 ms on dual GPU) together accounted for **~80 ms** of the remaining 138 ms.
- Theoretical memory reads at 360 GB/s peak bandwidth should take $< 10 \text{ ms}$. The kernels previously exhibited a **7× to 10× memory efficiency gap**.

---

## 2. Root Cause Analysis & Empirical Exploration

### 2.1 MoE Gate/Up Multi-Row DRAM Striding & Register Hoarding
In the original [`src/cuda/moe.cu`](file:///hdd/Projects/zeno/src/cuda/moe.cu):
Each warp computed 4 rows ($r \in [0..3]$) inside the block loop (`b` outer loop, `r` inner loop):
- **1,584-Byte Stride Jumps**: In every block $b$, the warp jumped across rows by 1,584 bytes 4 times, defeating hardware L1/L2 stream prefetchers.
- **Register Hoarding**: Accumulators (`gate_thread_sum[4][4]` and `up_thread_sum[4][4]`) consumed 32 registers, pushing register pressure to the hard 80-register ceiling under `__launch_bounds__(256, 3)`. Spilling to local memory DRAM stack occurred whenever additional variables were hoisted.

### 2.2 MoE Down-Accum Low SM Occupancy & DRAM Stall Cycles
In the original `moe_gemv_down_accum_sorted_kernel`:
- **Low SM Warp Occupancy**: Each warp computed 8 rows, dispatching a grid of only 44 blocks (`grid_x = 44`). Across 28 SMs on the RTX 3060, this resulted in only 1.57 blocks per SM (8–16 warps per SM), utilizing only 25–50% of the SM warp scheduler capacity.
- **Unpipelined Unaligned Loads**: Loads of `BlockQ8_0` (34 bytes) were issued 1 byte per lane sequentially without software pipelining, stalling ALUs on memory latency.

### 2.3 Abandoned Exploratory Approaches
1. **Loop Reordering in Down-Accum (Blocks outside, rows inside)**:
   - Attempted reordering to `for (int b = 0; b < blocks_per_row; b++) { for (int r = 0; r < 8; ++r) { ... } }`.
   - **Result**: Regressed latency from 138 ms to **158 ms** (+20 ms penalty). Column-strided jumps of 748 bytes across rows defeated hardware L1/L2 prefetchers. Row-major sequential streaming is strictly required.
2. **Inner-Loop Scale Broadcast via `__shfl_sync`**:
   - Attempted broadcasting `blk->d` from lane 0 across warp with `__shfl_sync(0xffffffff, d, 0)`.
   - **Result**: Regressed latency from 138 ms to **143 ms** (+5 ms penalty). On Ampere SM 8.6, uniform constant loads from L1 cache are served natively; explicit warp shuffle synchronization inside the 22-iteration inner loop added measurable ALU overhead.
3. **Dynamic Work-Queue Compaction for Inactive Experts**:
   - Evaluated dynamic work queues to skip inactive experts (`count == 0`).
   - **Result**: Demonstrated in Phase 13 that dynamic persistent work queues cause concurrent CTAs to process different experts simultaneously, trashing the 3 MB L2 cache and regressing latency. Empty CTA launches take $< 0.1 \text{ ms}$ total across 30 layers and do not bottleneck execution. Furthermore, dynamic grid sizing from host requires blocking D2H synchronization, which violates CUDA Graph capture contracts.

---

## 3. Delivered Implementation Architecture

### Work Package 1 (WP1): MoE Gate/Up 1-Row Sequential Streaming & Register Collapse
- **1 Row per Warp**: Restructured `moe_gemv_gate_up_geglu_sorted_kernel` so each warp computes exactly 1 row sequentially (`row = blockIdx.x * 8 + warp_id`).
- **Sequential BlockQ4_K Streaming**: Memory access across blocks `b` is 100% linear (`gate_row_w + b`), allowing hardware prefetchers to stream weights at maximum DRAM bus bandwidth.
- **Register Pressure Collapse**: Reduced register allocation from **80 down to 60 registers per thread** (25% reduction, 0 spill stores, 0 spill loads).
- **100% Maximum SM Occupancy**: Achieving $\le 64$ registers enables 4 active blocks / 1,024 threads per SM (the hardware maximum on Ampere).
- **88-CTA Balanced Grid**: Scaled `grid_x` to 88 CTAs (`(half_dim + 7) / 8`), saturating all 28 SMs in a single balanced wave.

### Work Package 2 (WP2): MoE Down-Accum Paired-Block Unrolling & Doubled SM Occupancy
- **Paired-Block Unrolling ($b += 2$)**: Unrolled the inner `BlockQ8_0` loop by 2 with paired weight loads (`blk0, blk1`), overlapping 32-byte memory transactions with FP32 arithmetic.
- **Halved Warp Row Work (8 $\to$ 4 rows)**: Reduced rows per warp from 8 to 4 (`row_base = (blockIdx.x * 8 + warp_id) * 4`), scaling `grid_x` from 44 to 88 blocks (`(out_dim + 31) / 32`).
- **Doubled SM Occupancy**: 88 blocks across 28 SMs provides 3.14 blocks per SM (24–32 warps per SM), increasing active warp occupancy from 25–50% to **75–100%**.
- **Register Allocation**: Reduced register usage from **72 down to 64 registers per thread** (0 spill stores, 0 spill loads).

---

## 4. Empirical Benchmark Results

Measured via `zeno benchmark models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --cuda --samples 3 specs/fixtures/system_health_boolean.json` on dual RTX 3060 GPUs:

| Canvas Width $C$ | Baseline Post-Phase 15 (ms) | Phase 16 Delivered (ms) | Spread (ms) | Speedup vs Baseline |
|:---|:---|:---|:---|:---|
| **$C=16$** | **138.64 ms** | **121.72 ms** | **2.66 ms** | **+16.92 ms faster** |
| **$C=32$** | 265.68 ms | 225.90 ms | 5.76 ms | +39.78 ms faster |
| **$C=64$** | 509.42 ms | 422.03 ms | 10.98 ms | +87.39 ms faster |
| **$C=128$** | 1011.56 ms | 841.36 ms | 23.66 ms | +170.20 ms faster |
| **$C=256$** | 2032.44 ms | 1762.59 ms | 20.15 ms | **+269.85 ms faster** |

---

## 5. Verification & Parity

1. **Exact Numerical Parity**:
   - Standalone unit parity tests in `src/cuda_mem.zig`: max difference vs CPU reference $< 6.1 \times 10^{-5}$.
   - Exact discrete decision match (**5/5, 100%**) across all golden oracle fixtures (`./tools/verify_all_oracles.sh --cuda`).
2. **CPU Isolation**:
   - `zig build test -Dcuda=false` passes cleanly without CUDA dependencies.
3. **HTTP Server Parity**:
   - 15/15 integration tests pass (`python3 tools/test_server.py --cuda`).

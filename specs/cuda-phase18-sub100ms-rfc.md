# CUDA Phase 18 RFC — Full Attention QKV Fusion, Post-Attention Norm-Residual Fusion, and MoE Tiling for Sub-100ms Execution

**Author:** Antigravity  
**Status:** Delivered / Implemented / Verified  
**Date:** 2026-10-21  
**Target Sequence:** $C = 16$ single-step structured read canvas execution  
**Target Hardware:** Dual Ampere GPUs (e.g. RTX 3060 12GB x 2, `sm_86`)  
**Objective:** Reduce single-step canvas latency from **108.73 ms** down to **$< 100\text{ ms}$**.

---

## 1. Executive Summary & Problem Formulation

Following the completion of CUDA Phase 17 ([`specs/cuda-phase17-dense-gemv-rfc.md`](cuda-phase17-dense-gemv-rfc.md)), single-step structured read canvas execution ($C=16$) latency reached **108.73 ms** (and **110.52 ms** total end-to-end HTTP request latency over loopback).

To break through the historic 100 ms latency barrier on dual Ampere GPUs, Phase 18 systematically resolved the remaining kernel dispatch, memory movement, and warp occupancy bottlenecks across all 30 transformer layers.

---

## 2. Delivered Work Packages

### WP1: Full Attention Q-K-V Projection Fusion (`gemv_q4k_fused_qkv_16x_kernel`)
- **Location**: [`src/cuda/gemv.cu`](file:///hdd/Projects/zeno/src/cuda/gemv.cu#L532), [`src/cuda_mem.zig`](file:///hdd/Projects/zeno/src/cuda_mem.zig#L916), [`src/infer_cuda.zig`](file:///hdd/Projects/zeno/src/infer_cuda.zig#L1023, #L1248).
- **Implementation**:
  - Unified Q, K, and V projections into `gemv_q4k_fused_qkv_16x_kernel` for all 17 layers where Q, K, and V are Q4_K (or tied Global).
  - Tiled 16 output rows per CTA (2 rows/warp) across the unified rows ($8192$ rows in SWA, $9216$ rows in Global).
  - Shared memory staging `s_x[16][256]` is populated once per CTA and reused across all rows in the block.
  - Eliminated 17 separate V-projection kernel launches per canvas forward pass.

### WP2: Post-Attention Norm & Residual Fusion (`fused_post_attn_norm_residual_kernel`)
- **Location**: [`src/cuda/ops.cu`](file:///hdd/Projects/zeno/src/cuda/ops.cu#L134, #L885), [`src/cuda_mem.zig`](file:///hdd/Projects/zeno/src/cuda_mem.zig#L537), [`src/infer_cuda.zig`](file:///hdd/Projects/zeno/src/infer_cuda.zig#L1310).
- **Implementation**:
  - Fused post-attention RMSNorm and residual addition into registers:
    $$X_{\text{mid}, i} = \text{RMSNorm}(A, \gamma)_i + R_i = \left(\frac{A_i}{\sqrt{\frac{1}{d} \sum_j A_j^2 + \epsilon}} \cdot \gamma_i\right) + R_i$$
  - Launch with $16$ CTAs (1 CTA per token, 256 threads per CTA).
  - Eliminated 30 kernel dispatches per canvas pass and eliminated intermediate VRAM read/write roundtrips of the $16 \times 2816$ activation tensor.

### WP3: MoE Gate/Up Sorted GEMV 16-Row Tiling & Activation Read Deduplication
- **Location**: [`src/cuda/moe.cu`](file:///hdd/Projects/zeno/src/cuda/moe.cu#L593).
- **Implementation**:
  - Restructured `moe_gemv_gate_up_geglu_sorted_kernel` to tile 16 output rows per CTA (2 rows per warp: `row0`, `row1`), halving CTAs from 88 to 44 per active expert.
  - Eliminated local pointer arrays `tok_ptrs[4]` and `out_ptrs[4]`; staged `s_x` directly with collaborative float4 loads.
  - Hoisted activation reads `x1[m] = s_x[m][off1]` and `x2[m] = s_x[m][off2]` to outer chunk scope, sharing activation registers across `row0` and `row1` and eliminating 50% of shared memory load instructions.
  - Enforced `__launch_bounds__(256, 4)` for 66.7% SM warp occupancy (32 / 48 warps, 100% register-file saturation at 64 registers/thread; **0 bytes stack frame, 0 spill stores, 0 spill loads**).

### WP4: MoE Down Accum Sorted GEMV Activation Hoisting for Q8_0
- **Location**: [`src/cuda/moe.cu`](file:///hdd/Projects/zeno/src/cuda/moe.cu#L1110).
- **Implementation**:
  - Inverted the loop nesting in `moe_gemv_down_accum_sorted_kernel` for Q8_0: hoisted the block pair loop ($b$) to the outside of the 4 rows ($r \in [0, 4)$).
  - Hoisted shared activations `s_acts` into registers once per block pair, cutting shared memory load transactions by 75%.
  - Enforced `__launch_bounds__(256, 4)` (**0 bytes stack frame, 0 spill stores, 64 registers**).

### WP5: MoE Down Accum Q5_0 Loop Inversion, Direct Bit Extraction, & Activation Hoisting
- **Location**: [`src/cuda/moe.cu`](file:///hdd/Projects/zeno/src/cuda/moe.cu#L1342).
- **Implementation**:
  - DiffusionGemma-26B uses Q5_0 expert down weights in 16 out of 30 layers.
  - Inverted the nested row loop ($r \in [0, 4)$) to the inside of the block loop ($b \in [0, 22)$), maintaining 4 row accumulators in registers (`sum0[4]`, `sum1[4]`, `sum2[4]`, `sum3[4]`).
  - Eliminated slow warp-shuffle extraction of `qs_pair` and `sub_lane`: replaced with direct thread-lane indexing `uint8_t bit = (uint8_t)((qh >> lane_id) & 1)` and `byte_idx = (lane_id < 16) ? lane_id : (lane_id - 16)`.
  - Hoisted `xv[m] = s_acts[m][b*32 + lane_id]` once per block across all 4 rows.
  - Enforced `__launch_bounds__(256, 4)` (**0 bytes stack frame, 0 spill stores, 64 registers**). Reduced Q5_0 down-projection time by over 2 ms.

### WP6: 16-Row CTA Tiling for Dense GEMV Q5_0 & Attention V-Projection Q6_K
- **Location**: [`src/cuda/gemv.cu`](file:///hdd/Projects/zeno/src/cuda/gemv.cu#L973, #L1221).
- **Implementation**:
  - `gemv_q5_0_batched_16x_kernel`: Upgraded from 8 rows/CTA to 16 rows/CTA (2 rows/warp). Processes block pairs with collaborative float4 loads into `s_x[16][64]`. Halved CTAs from 352 to 176 per layer (**0 stack frame, 0 spill stores, 60 registers**).
  - `gemv_q6k_batched_16x_kernel`: Upgraded from 8 rows/CTA to 16 rows/CTA (2 rows/warp). Halved CTAs from 256 to 128 for all 13 layers where V is Q6_K (**0 stack frame, 0 spill stores, 64 registers**).

### WP7: Attention QK Projection Fusion on Q6_K Layers
- **Location**: [`src/infer_cuda.zig`](file:///hdd/Projects/zeno/src/infer_cuda.zig#L1040, #L1283).
- **Implementation**:
  - In the 13 layers where V is Q6_K, Q (4096 rows) and K (2048 rows) are both Q4_K.
  - Rewired `runLayerCuda` and `runLayerCanvasCuda` to invoke `gemvQ4_K_FusedQK` for Q and K together, staging `nx_slice` in shared memory and eliminating 13 separate Q/K kernel launches.

---

## 3. Empirical Latency Milestones

Evaluated on dual NVIDIA GeForce RTX 3060 12GB GPUs (`sm_86`, `dual_gpu_split` topology) on $C=16$ single-step structured decision execution (`/v1/systemone`):

| Phase / Optimization Stage | Canvas Step 1 Latency | HTTP Request Round-Trip | Prefill Latency ($P=29$) |
| :--- | :--- | :--- | :--- |
| **Phase 16 Baseline** | 114.28 ms | 115.12 ms | ~400 ms |
| **Phase 17 Baseline** | 108.73 ms | 110.52 ms | ~400 ms |
| **Phase 18 (WP1–WP4)** | 106.19 ms | 106.86 ms | 212.77 ms |
| **Phase 18 (WP5: MoE Q5_0 Inversion)** | 101.16 ms | 101.94 ms | 194.47 ms |
| **Phase 18 (WP6–WP7: 16-Row Q5_0/Q6_K & Fused QK)** | **99.87 ms (median)**<br>**99.80 ms (min)** | **100.55 ms (median)**<br>**100.46 ms (min)** | **192.21 ms** |

---

## 4. Verification Suite Results

1. **Unit Test Parity**:
   - `zig build test -Dcuda=false`: **PASS** (Zero-dependency CPU build isolation).
   - `zig build test -Dcuda=true --summary all`: **PASS** (98/98 tests passing, 7 skipped).
2. **Oracle Parity (5/5 Fixtures)**:
   - `./tools/verify_all_oracles.sh --cuda`: **5/5 PASSED (100% discrete decision parity, 100% top-5 vocab match)** against pinned `llama.cpp` reference oracle.
3. **HTTP Server Integration**:
   - `python3 tools/test_server.py --cuda`: **15/15 PASSED**.

---

## 5. ARC-Challenge Benchmark Results (N=150, C=256)

Phase 18 was evaluated on the 150-question ARC-Challenge evaluation slice (`N=150`, full canvas width `C=256`) across both base canonical order and cyclic option rotation (`rot1`), verifying that kernel fusions and GEMV restructuring preserved numerical accuracy and calibration while achieving a **~42–43% end-to-end dataset latency reduction**:

| Evaluation Run | Result File | Accuracy | Mean Latency (Phase 8 $\to$ Phase 18) | Median Latency | Mean $H_{\text{norm}}$ | Step Split |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base $T=1$** | [`results_t1_phase18.json`](file:///hdd/Projects/zeno/specs/fixtures/arc_results/results_t1_phase18.json) | **75.33%** (113/150) | $1046.3\text{ ms} \to \mathbf{598.41\text{ ms}}$ (`-42.8%`) | $576.49\text{ ms}$ | $0.3528$ | $150\text{ at }T=1$ |
| **Rot-1 $T=1$** | [`results_t1_rot1_phase18.json`](file:///hdd/Projects/zeno/specs/fixtures/arc_results/results_t1_rot1_phase18.json) | **65.33%** (98/150) | $1017.8\text{ ms} \to \mathbf{596.64\text{ ms}}$ (`-41.4%`) | $574.17\text{ ms}$ | $0.3709$ | $150\text{ at }T=1$ |
| **Base $T=2$ ($\tau=0.10$)** | [`results_t2_adapt_phase18.json`](file:///hdd/Projects/zeno/specs/fixtures/arc_results/results_t2_adapt_phase18.json) | **86.00%** (129/150) | $1109.1\text{ ms} \to \mathbf{634.63\text{ ms}}$ (`-42.8%`) | $616.28\text{ ms}$ | $0.2193$ | $100\text{ at }T=1,\ 50\text{ at }T=2$ |
| **Rot-1 $T=2$ ($\tau=0.10$)** | [`results_t2_adapt_rot1_phase18.json`](file:///hdd/Projects/zeno/specs/fixtures/arc_results/results_t2_adapt_rot1_phase18.json) | **84.67%** (127/150) | $1084.9\text{ ms} \to \mathbf{637.47\text{ ms}}$ (`-41.2%`) | $0.2283$ | $95\text{ at }T=1,\ 55\text{ at }T=2$ |

*Summary*: Across all four runs, decision accuracy, confidence calibration, and adaptive step distributions match Phase 8 and Phase 10 exactly, while end-to-end evaluation time dropped from ~1,050–1,100 ms to ~595–635 ms per prompt-canvas evaluation.

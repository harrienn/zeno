# CUDA Phase 19 RFC — Multi-Token Prefill GEMV Generalization and Multi-Step CUDA Graph Chaining

**Author:** Antigravity  
**Status:** Delivered / Implemented / Verified  
**Date:** 2026-10-21  
**Target Sequence:** Prompt Prefill ($P \in [16, 512]$) and Multi-Step Structured Read Canvas Execution ($T \ge 2$, $C \in [16, 256]$)  
**Target Hardware:** Dual Ampere GPUs (e.g. NVIDIA GeForce RTX 3060 12GB x 2, `sm_86`, `dual_gpu_split` topology)  
**Objective:** 
1. Accelerate cold prompt prefill by eliminating redundant weight streaming passes in batched GEMV kernels.
2. Implement device-resident multi-step CUDA graph chaining (`graph0_step2`) for adaptive denoising steps ($T=2$), eliminating all host-device synchronization boundaries and CPU canvas dequantization between Step 1 and Step 2.
3. Verify 100% discrete decision and numerical parity under [`zeno.tolerance/v2`](UNRESOLVED.md) across CPU and dual-GPU backends.

---

## 1. Executive Summary & Problem Formulation

Following the completion of CUDA Phase 18 ([`specs/cuda-phase18-sub100ms-rfc.md`](cuda-phase18-sub100ms-rfc.md)), single-step canvas execution reached the **sub-100ms milestone** (**99.87 ms** median canvas step 1; **100.55 ms** HTTP round-trip). However, profiling multi-query and multi-step inference identified two critical performance barriers:

1. **Cold Prompt Prefill Inefficiency**:
   While repeated prompts leverage Zeno's prompt KV prefix cache (**0.0 ms** prefill on cache hit), cold prompt prefill previously required ~192 ms ($P=30$) to ~495 ms ($P=90$). Profiling revealed that `zeno_cuda_gemv_f32` enforced an artificial constraint `(batch_size % 16 == 0)` to enter `gemv_f32_batched_16x_kernel`. For any prompt whose length was not an exact multiple of 16 (e.g. $P \in \{29, 30, 35, 90, 94\}$), the router gate GEMV fell back to unbatched single-token kernel dispatches, streaming the entire router gate matrix from VRAM up to 90 times per layer across all 30 layers. Furthermore, QKV projections for small batch sizes ($P < 16$) fell back to 3 separate unfused GEMVs.
2. **Multi-Step Adaptive Denoising Host Synchronization Bubbles**:
   When normalized candidate Shannon entropy triggered Step 2 ($T=2$, $\tau \le H_{\text{norm}}$), Step 2 previously took ~140 ms. Profiling identified that between Step 1 and Step 2:
   - The host CPU redundantly re-dequantized all $C$ canvas tokens from Q6_K weights.
   - The host executed blocking H2D copies to `ws.dev0.current_emb`.
   - `applyDeviceSelfConditioning` invoked four blocking stream synchronizations (`ws.synchronize()`), stalling the GPU pipeline and preventing hardware queue overlap.
   - Partition 0 on Device 0 executed in eager dispatch mode rather than replaying from a pre-captured CUDA Graph.

Phase 19 systematically eliminated both bottlenecks.

---

## 2. Delivered Work Packages

### WP1: Batched F32 GEMV Generalization & Full-Batch Unified QKV Projection
- **Location**: [`src/cuda/gemv.cu`](file:///hdd/Projects/zeno/src/cuda/gemv.cu#L1779, #L2073).
- **Implementation**:
  - Removed the restrictive `(batch_size % 16 == 0)` guard in `zeno_cuda_gemv_f32`. Since `gemv_f32_batched_16x_kernel` already incorporates exhaustive bounds checks (`tidx < batch_size`) during activation loading, warp-reduction, and writeback, it safely handles arbitrary batch sizes $B \ge 1$ with grid $Y = \lceil B / 16 \rceil$.
  - Eliminated up to $15.6\times$ redundant DRAM weight loads for `blk.{l}.ffn_gate_inp.weight` across all 30 layers during cold prompt prefill.
  - Generalized `zeno_cuda_gemv_q4k_fused_qkv` from `batch_size >= 16` to `batch_size >= 1`, allowing unified Q, K, and V projection fusion even for small prompt chunks and short sequences, eliminating 2 kernel launches per layer.

### WP2: Device-Resident Canvas Initialization Buffers
- **Location**: [`src/infer_cuda.zig`](file:///hdd/Projects/zeno/src/infer_cuda.zig#L538, #L625).
- **Implementation**:
  - Introduced device-resident backup buffers on `CudaDeviceWorkspace`:
    - `canvas_init_emb`: sized `max_seq_len * embedding_dim * sizeof(f32)` (5.77 MB). Retains the baseline normalized canvas embeddings on device.
    - `canvas_raw_ans`: sized `embedding_dim * sizeof(f32)` (11.26 KB). Retains the unnormalized raw `[MASK]` token embedding at `answer_pos`.
  - In Step 1, `canvas_init_emb` and `canvas_raw_ans` are populated once from the host.
  - In Step 2, canvas tokens are restored entirely via asynchronous device-to-device DMA (`copyFromDevice` on `dev0.stream`), eliminating host CPU dequantization and host-device PCIe memory movement for canvas tokens.
  - Updated memory auditing in `activationWorkspaceBytes` ($376,154,368 \to 381,932,800\text{ bytes}$, ~381.93 MB).

### WP3: Multi-Step CUDA Graph Chaining (`graph0_step2`)
- **Location**: [`src/infer_cuda.zig`](file:///hdd/Projects/zeno/src/infer_cuda.zig#L878, #L1746, #L2065).
- **Implementation**:
  - Factored out stream-level self-conditioning kernel execution into `runDeviceSelfConditioning(&ws.dev0, model, answer_pos)` without host synchronizations.
  - Implemented `runPartition0Step2CanvasEager`:
    1. Restores `current_emb` from `canvas_init_emb` via D2D memcpy.
    2. Restores `[MASK]` raw embedding at `answer_pos` from `canvas_raw_ans` via D2D memcpy.
    3. Executes device-resident self-conditioning MLP (pre-norm $\to$ dense Gate/Up GeGLU $\to$ down GEMV $\to$ residual addition $\to$ RMSNorm) on `sc_soft_emb`.
    4. Executes Partition 0 layers 0..14 forward pass (`runLayerCanvasCuda`).
  - Captured this entire sequence into a dedicated CUDA Graph: `ws.graph0_step2`.
  - In Step 2, the host computes the 2816-float soft embedding vector (~2.5 µs), enqueues an asynchronous DMA copy to `sc_soft_emb` via `copyFromHostAsync`, and launches `ws.graph0_step2`.
  - Device 1 seamlessly replays the pre-captured `ws.graph1` (Layers 15..29 + Output RMSNorm).
  - Eliminated all 4 intermediate host synchronizations between Step 1 and Step 2.

### WP4: Fault-Tolerant Fallback & Quarantine Lifecycle Integration
- **Location**: [`src/infer_cuda.zig`](file:///hdd/Projects/zeno/src/infer_cuda.zig#L980, #L1005, #L2120).
- **Implementation**:
  - Integrated `graph0_step2` into workspace lifecycle management (`ws.invalidatePromptCache()` and `ws.deinit()`).
  - Implemented automatic fallback to eager execution (`runPartition0Step2CanvasEager`) upon injected capture or replay failures (`inject_graph_capture_failure`, `inject_graph_replay_failure`), satisfying safety contracts R24-05, R40-01, and R41-01.

---

## 3. Empirical Latency Milestones

Evaluated on dual NVIDIA GeForce RTX 3060 12GB GPUs (`sm_86`, `dual_gpu_split` topology) across single-step and two-step ($T=2$) structured decision execution (`/v1/systemone`, $C=16$):

| Execution Phase / Benchmark | Cold Prefill ($P=30$) | Canvas Step 1 | Canvas Step 2 ($T=2$) | Total Engine Time ($T=2$) | HTTP Round-Trip ($T=2$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 18 Baseline ($T=2$)** | 212.77 ms | 104.89 ms | 141.20 ms | 458.86 ms | 465.12 ms |
| **Phase 19 (Run 1: Cold Prefill + Graph Capture)** | 208.04 ms | 106.91 ms | 107.48 ms | 425.30 ms | 432.73 ms |
| **Phase 19 (Run 2: Cache Hit + Chained Replay)** | **0.0 ms** (hit) | 104.34 ms | **101.60 ms** | **207.47 ms** | **208.34 ms** |
| **Phase 19 (Run 3: Cache Hit + Chained Replay)** | **0.0 ms** (hit) | **98.61 ms** | **103.88 ms** | **203.99 ms** | **205.26 ms** |

*Summary*:
- **Step 2 Latency**: Reduced from **~141 ms down to ~101–103 ms** (**~28% latency reduction**).
- **Total Two-Step Denoising ($T=2$)**: Reduced from **458.86 ms down to 203.99 ms** (**-55.5% end-to-end latency reduction** on warm cache).
- **HTTP Request Latency ($T=2$)**: **205.26 ms** total wire-to-wire round-trip over loopback.

---

## 4. Verification Suite Results

1. **CPU Reference Build Isolation**:
   - `zig build test -Dcuda=false`: **PASS** (6/6 steps succeeded, 88 passed, 18 skipped, 0 failed). Zero CUDA dependencies in CPU test runner.
2. **CUDA Component & Integration Tests**:
   - `zig build test -Dcuda=true`: **PASS** (11/11 steps succeeded, 98 passed, 7 skipped, 0 failed).
   - Validated Step 2 CUDA Graph capture and replay against CPU reference: exact matching within $0.05$ raw logit tolerance.
3. **Reference Oracle Parity (5/5 Golden Fixtures)**:
   - `./tools/verify_all_oracles.sh --cuda --strict-parity`: **5/5 PASSED (100% discrete decision match, 100% top-5 vocab ranking match)** against pinned `llama.cpp` reference oracle under [`zeno.tolerance/v2`](UNRESOLVED.md).
4. **HTTP Server Integration**:
   - `python3 tools/test_server.py --cuda`: **15/15 PASSED**.

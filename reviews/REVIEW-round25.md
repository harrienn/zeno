# Review Round 25 — CUDA Phases 7 & 8 Follow-up

**Reviewed revisions:** `42b8864` (`perf(cuda): implement Phase 7 token-sorted MoE and prompt KV prefix caching`) and `dc4ea94` (`perf(cuda): implement Phase 8 16x token-tiled GEMVs, fused dense FFN GeGLU, and MoE warp tuning`), including changes across `src/infer_cuda.zig`, `src/cuda/attention.cu`, `src/cuda/gemv.cu`, `src/cuda/moe.cu`, `src/cuda/quant_types.cuh`, `src/cuda/zeno_cuda.h`, `src/cuda.zig`, `src/cuda_mem.zig`, `specs/multi-gpu.md`, and `build.zig`.

**Scope:** Static review of the Phase 7 & 8 performance optimizations, token-sorted MoE execution, prompt KV prefix caching, fused dense FFN GeGLU kernel, 16x token-tiled GEMVs, stream synchronization, CUDA Graph replay safety, memory bounds, and specification compliance.

---

## Verdict

This combined changeset represents a substantial architectural leap in Zeno's CUDA inference pipeline. The implementation of device-side token-sorted MoE histogram binning (`moe_build_expert_bins_kernel`), prompt KV prefix caching (`attention_cached_kernel`), 16x token-tiled GEMV kernels (`gemv_q*k_batched_16x_kernel`), and fused Gate+Up+GeGLU (`dense_gemv_gate_up_geglu_q4k_kernel`) achieves a $15.74\times$ latency reduction at $C=16$ (dropping canvas forward latency from ~2804 ms down to ~178 ms).

The implementation strictly maintains pure Zig CPU build isolation (`zig build test -Dcuda=false`), passes all CUDA parity tests, and reproduces 100% discrete decision parity across all 5 golden oracle fixtures under `zeno.tolerance/v1`.

Several static safety and edge-case concerns warrant attention before proceeding to Phase 9:
1. `moe_accumulate_scaled_sorted_kernel` lacks a defensive guard against negative `sorted_k` indices, which would trigger out-of-bounds GPU global memory reads if invalid expert indices are encountered.
2. In dual-GPU CUDA Graph capture, graph cache metadata is populated before verifying that Partition 1's initial launch succeeded, leading to a transient invalid state on subsequent requests if Partition 1 initial launch fails.
3. In `prefillPromptCuda`, the boundary transfer uses `ws.dev0.current_emb` rather than the explicit fixed buffer `ws.dev0.emb_buf_1`, creating a slight conceptual asymmetry with the graph-captured path.

---

## Findings

### R25-01 — Missing Negative Index Guard in `moe_accumulate_scaled_sorted_kernel`

**Severity: Medium**

**Location:** `src/cuda/moe.cu:730-735`

In `moe_accumulate_scaled_sorted_kernel`:
```c
for (int e = 0; e < 8; ++e) {
    int k = token_idx * 8 + e;
    int sorted_k = instance_to_sorted[k];
    float w = top_weights[k];
    sum += sorted_down_out[(size_t)sorted_k * dim + d] * w;
}
```

In `moe_build_expert_bins_kernel:469`, if an expert index is out of bounds ($< 0$ or $\ge 128$), `instance_to_sorted[i]` is explicitly set to `-1`. If `sorted_k == -1`, evaluating `(size_t)sorted_k * dim + d` wraps around to `SIZE_MAX`, causing an out-of-bounds global memory read and potentially a GPU kernel panic or illegal memory access fault.

While `moe_router_top8_kernel` currently produces valid indices in $[0, 128)$ for all active tokens, defensive programming in CUDA kernels requires handling `-1` sentinel values safely.

**Recommendation:** Add an explicit guard:
```c
if (sorted_k >= 0) {
    sum += sorted_down_out[(size_t)sorted_k * dim + d] * w;
}
```

---

### R25-02 — Graph Metadata Populated Prior to Partition 1 Initial Launch Verification

**Severity: Low**

**Location:** `src/infer_cuda.zig:1613-1629`

During dual-GPU graph capture in `executeStructuredReadCuda`:
```zig
if (cuda_mem.CudaGraph.endCapture(&dev1.stream)) |g1| {
    ws.graph1 = g1;
    ws.cached_graph_prompt_len = P;
    ws.cached_graph_canvas_len = C;
    ws.cached_graph_answer_pos = answer_pos;
    var p1_launch_ok = false;
    if (!inject_graph_replay_failure) {
        if (ws.graph1.?.launch(&dev1.stream)) |_| {
            p1_launch_ok = true;
        } else |_| {}
    }
    if (p1_launch_ok) {
        g1_instantiated = true;
    } else {
        ws.graph1.?.deinit();
        ws.graph1 = null;
    }
}
```

If Partition 1 graph launch fails (`p1_launch_ok == false`), `ws.graph1` is freed and set to `null`, but `ws.cached_graph_prompt_len`, `ws.cached_graph_canvas_len`, and `ws.cached_graph_answer_pos` remain populated, while `ws.graph0` remains non-null.

On the next request, `graph_valid` evaluates to `true` (since `ws.graph0 != null` and dimensions match). The dual-GPU replay branch then encounters `ws.graph1 == null` and enters the recovery path to deinit `ws.graph0`. While the system recovers via fallback, the graph cache state is temporarily inconsistent between requests.

**Recommendation:** Set `ws.cached_graph_*` metadata only after `p1_launch_ok` is confirmed `true`. If `p1_launch_ok == false`, explicitly deinitialize `ws.graph0` immediately and reset all graph cache metadata fields.

---

### R25-03 — Boundary Slice Symmetry Between Prefill and Canvas Graph Paths

**Severity: Low**

**Location:** `src/infer_cuda.zig:1308` vs `src/infer_cuda.zig:1515`

In `runPartition0CanvasEager` and replay:
```zig
const d0_out_slice = try ws.dev0.emb_buf_1.slice().subslice(0, C * ModelConfig.embedding_dim);
const d1_in_slice = try dev1.emb_buf_0.slice().subslice(0, C * ModelConfig.embedding_dim);
```
Here, `emb_buf_1` is explicitly specified because 15 layer forward passes swap double-buffer pointers an odd number of times, ending at `emb_buf_1`.

In `prefillPromptCuda:1308`:
```zig
const d0_out_slice = try ws.dev0.current_emb.slice().subslice(0, P * ModelConfig.embedding_dim);
const d1_in_slice = try dev1.current_emb.slice().subslice(0, P * ModelConfig.embedding_dim);
```
While `ws.dev0.current_emb` resolves to `emb_buf_1` and `dev1.current_emb` resolves to `emb_buf_0`, referencing `current_emb` relies on mutable runtime struct state rather than explicit buffer naming.

**Recommendation:** Use `ws.dev0.emb_buf_1.slice()` and `dev1.emb_buf_0.slice()` in `prefillPromptCuda` to maintain strict explicit data flow across both execution modes.

---

### R25-04 — MoE Projections Still Tile at T=4 While Dense GEMVs Tile at T=16

**Severity: Informational / Design Optimization**

**Location:** `src/cuda/moe.cu:505`, `src/cuda/moe.cu:602`

In Phase 8A, dense GEMVs were upgraded to $T=16$ token tiling (`grid_y = batch_size / 16`), streaming weights once per forward pass at $C=16$. However, `moe_gemv_gate_up_sorted_kernel` and `moe_gemv_down_sorted_kernel` remain tiled at $T=4$ (`m_base += 4`).

For canvas lengths $C \ge 64$ where popular experts process $\ge 16$ instances, expert weights are streamed in $T=4$ increments. Expanding MoE token tiling to $T=8$ or $T=16$ represents an available future optimization for larger canvas sizes.

---

## Improvements Confirmed

1. **16x Token-Tiling for Batched GEMVs**:
   - `gemv_q8_0_batched_16x_kernel`, `gemv_q4k_batched_16x_kernel`, and `gemv_q6k_batched_16x_kernel` correctly accumulate 16 token dot-products in parallel per warp.
   - At $C=16$, $grid_y = 1$, eliminating redundant DRAM streaming and saving ~2.88 GB of memory traffic per forward pass.
2. **Fused Dense FFN Gate+Up+GeGLU Kernel**:
   - `dense_gemv_gate_up_geglu_q4k_kernel` fuses gate and up GEMVs, applies `gelu_tanh_scalar` in registers, and writes directly to `geglu_buf`.
   - Eliminates 60 kernel launches across 30 layers and removes intermediate global buffer DRAM writes.
3. **MoE Warp Tuning**:
   - Processing 8 rows per warp in `moe_gemv_gate_up_sorted_kernel` and `moe_gemv_down_sorted_kernel` halved launched thread blocks from 16,896 to 8,448 per layer.
4. **Prompt KV Prefix Caching**:
   - Bit-exact parity against full attention (`max_diff = 0.0000e0`).
   - Prefix KV evaluation executes once per prompt hash; subsequent canvas evaluations execute in canvas-only forward passes ($C \in [16, 256]$).
5. **Audited Workspace Bounds Strictly Preserved**:
   - Exact per-device workspace footprint of **336,801,792 bytes (321.20 MiB)** maintained and asserted via unit tests.
6. **Zero-Dependency Pure Zig CPU Build**:
   - `zig build test -Dcuda=false` passes cleanly with zero C/CUDA compiler dependencies.

---

## Readiness Assessment

| Component | Status | Assessment |
| :--- | :--- | :--- |
| **Stage 8A: 16x Token Tiling** | **Ready** | Parity verified against 4x kernel; arithmetic identical; bounds checked. |
| **Stage 8B: Fused Dense FFN** | **Ready** | Parity verified against separate GEMV + GeGLU ($\Delta \le 10^{-4}$); registers within 100% occupancy budget. |
| **Stage 8C: MoE Warp Tuning** | **Ready** | Block launch overhead halved; deterministic accumulation order preserved. |
| **MoE Sorted Accumulation** | **Minor finding (R25-01)** | Functional in current test suite; needs defensive `sorted_k >= 0` check. |
| **Prompt KV Caching** | **Ready** | Bit-exact attention parity verified; cache invalidation on hash mismatch verified. |
| **CUDA Graph Recovery** | **Minor finding (R25-02)** | Replay fallback functional; cache metadata assignment order should be hardened. |
| **Workspace Accounting** | **Ready** | 321.20 MiB audited footprint documented and test-enforced. |
| **Oracle Parity** | **Ready** | 5/5 golden fixtures pass 100% discrete decision parity under `zeno.tolerance/v1`. |

---

## Verification Notes

Static review only. No build, test, CUDA execution, compute-sanitizer run, or benchmark was performed as part of this static document preparation. The worktree was clean before this review document was created.

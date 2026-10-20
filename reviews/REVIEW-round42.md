# Review Round 42 — CUDA Phases 15–16 and Quarantine Remediation

**Reviewed revisions:** `468527a` (Round 41 synchronization and quarantine remediation), `60aa788` (CUDA Phase 15 kernel fusion), and `50c34f7` (CUDA Phase 16 MoE GEMV restructuring).

**Scope:** Follow-up review of pinned host-buffer quarantine and drainage, fused head normalization, MoE GEMV launch geometry, and correspondence between the Phase 16 RFC and the shipped kernels. The review is based on source and documentation inspection; no implementation changes were made.

## Findings

### R42-01 — Quarantined pinned buffers are released without confirming device quiescence

**Severity: High — do not free a quarantined host buffer until work that may reference it is known to have completed.**

`drainQuarantinedPinnedBuffers()` calls `zeno_cuda_host_free()` for each stored pointer and clears its slot without synchronizing the associated device/streams first (`src/infer_cuda.zig`, around lines 784–790). This leaves the original lifetime question unresolved: if a failed synchronization left an asynchronous transfer using that pinned allocation, draining the quarantine can free memory while DMA may still be in flight. The drain also clears the slot regardless of the free result, losing the pointer if release fails.

The added test injects a Device 1 synchronization failure, deinitializes the workspace, then drains the quarantine (`src/infer_cuda.zig`, around lines 3040–3051). It does not queue a transfer before the injected failure, so it does not exercise the case where the quarantined allocation is still actively referenced by DMA.

**Recommendation:** Define and enforce the condition that makes a quarantined allocation safe to free. Drain only after the relevant device activity is confirmed quiescent, and retain the slot if the free operation fails.

### R42-02 — Quarantine capacity is consumed cumulatively and is not reused

**Severity: Medium — repeated synchronization failures can strand pinned memory.**

Quarantine insertion uses a monotonically increasing count and stores pointers only while the count is below the 16-slot capacity (`src/infer_cuda.zig`, around lines 767–780). Draining clears the array entries but does not reset or decrement the count. After 16 lifetime insertions, later buffers are no longer recorded even if earlier slots have been drained. Those allocations have left their workspace owner but cannot be released through this table, so repeated rare failures can leak pinned host memory.

**Recommendation:** Track occupied slots and reuse them after successful release. Define an explicit fallback for the case where all slots are simultaneously occupied so a buffer is never silently left untracked.

### R42-03 — Fused head normalization can index past its shared reduction array

**Severity: Medium — the wrapper accepts dimensions that exceed the kernel's shared-array capacity.**

The fused head-normalization kernel declares `s_warp_sums[8]` and indexes it by `warp_id` (`src/cuda/ops.cu`, around lines 426 and 434). The wrapper accepts even `head_dim` values up to 1024 (`src/cuda_mem.zig`, around line 1512), while the launch uses `head_dim / 2` threads. At dimensions above 512, the block has more than eight warps, so those warps index beyond the shared array. The current DiffusionGemma target dimension of 256 does not reach the failure, but the accepted wrapper contract does.

**Recommendation:** Size the shared array for the maximum supported warp count, or narrow and document the wrapper's accepted dimension range so it matches the kernel.

### R42-04 — Phase 16 over-launches CTAs for the sorted gate/up GEMV kernel

**Severity: Low — unnecessary empty CTA dispatch overhead.**

`moe_gemv_gate_up_sorted_kernel` assigns eight warps to eight rows each, for 64 output rows per CTA (`src/cuda/moe.cu`, around line 512). Its launch grid now uses `ceil(out_dim / 32)` (`src/cuda/moe.cu`, around lines 1759–1768), although the kernel's row mapping requires `ceil(out_dim / 64)`. The extra CTAs are guarded out of range and do not appear to overlap valid output rows, but they add launch work without contributing output.

**Recommendation:** Derive the grid from the kernel's current 64-row tile, or adjust the kernel mapping and grid together if a different tile is intended.

### R42-05 — Phase 16 RFC optimization claims do not match the current Q8 down kernel

**Severity: Medium — reconcile the RFC and implementation before treating the planned optimization as delivered.**

The Phase 16 RFC describes hoisting activation loads out of the row loop, broadcasting block scales from lane 0 with shuffles, and accumulating eight rows in registers for the Q8 path (`specs/cuda-phase16-moe-gemv-rfc.md`, around lines 76–89). The current `moe_gemv_down_accum_sorted_kernel` still loops over four rows, loads shared activations inside that loop, and has each lane load the block scales (`src/cuda/moe.cu`, around lines 1128–1167). The implementation does combine two quantization blocks in the accumulation path, but it does not implement the stated activation-hoisting and scale-broadcast mechanism or the associated 87.5% shared-load reduction claim.

The RFC is marked proposed/under review, so this is a contract and status mismatch rather than evidence that the shipped kernel is incorrect. **Recommendation:** Update the RFC and phase completion notes to describe the implemented kernel and measured results, or implement the specified mechanism before claiming that work package complete.

## Positive Changes

- The synchronization remediation attempts both device synchronizations independently, avoiding the earlier short-circuit behavior.
- Phase 15 introduces the intended intra-layer fusion and elementwise streamlining, with corresponding numerical coverage in the implementation.
- Phase 16 groups Q8 down-projection work across paired quantization blocks and revises the warp mapping for the GeGLU sorted path.

## Review Verdict

The Phase 15/16 work advances the CUDA path, but the quarantine drain still does not establish that it is safe to release pinned memory after synchronization failure. Resolve R42-01 and R42-02 before relying on quarantine for safe ownership recovery. R42-03 should be corrected or the wrapper contract narrowed before using dimensions above the current model target. R42-04 is a small launch-efficiency correction. R42-05 needs the RFC and implementation status brought into agreement.

## Verification Limits

The review inspected the three revisions, relevant CUDA kernels and wrappers, quarantine lifecycle code, tests, and the Phase 16 RFC. No CUDA compilation, build, tests, profiler run, or benchmark was performed.

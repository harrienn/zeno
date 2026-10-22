# Review Round 49 — Retained Workspace Pool Exhaustion

**Reviewed revision:** `f29eaa1` (`fix: remediate Review Round 48 finding on retained workspace allocator lifetime`), compared with Round 48 review commit `2a5ca67`.

**Scope:** Follow-up review of the R48-01 fix, focusing on ownership when the new process-lifetime retained-workspace pool is full. No implementation changes were made.

## Finding

### R49-01 — Exhausting the retained pool leaves a pinned buffer untracked

**Severity: Medium — the bounded terminal fallback can still lose track of an allocation that cannot yet be safely freed.**

`retained_workspace_pool` has a fixed capacity of 16 records (`src/infer_cuda.zig`, `MAX_RETAINED_WORKSPACES`). If stream synchronization fails, the ordinary quarantine tiers and emergency reserve are full, and all 16 retained records are also occupied, `CudaWorkspace.deinit` reaches the `!placed_retained` branch and only logs that the buffer cannot be preserved (`src/infer_cuda.zig`, `CudaWorkspace.deinit`). The pinned-buffer pointer remains in `self.host_staging_pinned`, but teardown then continues to destroy the workspace and its allocator-owned state. The pinned CUDA allocation is not freed, which avoids freeing memory while device work may still be active, but it is no longer tracked for a later safe drain and can leak for the remainder of the process.

The added regression test retains one workspace from a short-lived arena and drains it after arena destruction. It does not fill the 16-record pool or exercise the next teardown, when the final fallback fails.

**Recommendation:** Define a safe and bounded behavior when all retained records are occupied. For example, apply backpressure before accepting more workspaces than the system can retain, or make teardown establish device quiescence before releasing ownership. Do not continue destroying the only owner while the pinned allocation remains unsafe to free. Add a saturation test that fills the retained pool, exercises one more failed-sync teardown, and verifies that the selected behavior keeps the allocation tracked and can eventually release it.

## Review Verdict

The Round 48 change fixes the caller-allocator lifetime issue for buffers successfully transferred into the static pool. The pool's exhaustion branch still has no ownership-preserving outcome, so the terminal teardown guarantee is limited to at most 16 simultaneously retained workspaces. Resolve or explicitly enforce that capacity bound before treating the fallback as safe under repeated synchronization failures.

## Verification Limits

The review inspected the retained pool definition, drainage loop, teardown fallback, and updated saturation test in `src/infer_cuda.zig`. No build, test, CUDA runtime check, reference comparison, or benchmark was run for this review.

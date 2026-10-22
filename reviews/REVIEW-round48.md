# Review Round 48 — Retained Workspace Allocator Lifetime

**Reviewed revision:** `b4028a5` (`fix: remediate Review Round 47 finding on quarantine reserve exhaustion`), compared with the Round 47 review commit `b5bb691`.

**Scope:** Follow-up review of the retained-workspace fallback added for quarantine exhaustion. The review focuses on whether globally retained workspace state remains valid until a later drain. No implementation changes were made.

## Finding

### R48-01 — Retained workspace can outlive its allocator

**Severity: Medium — a later quarantine drain can dereference and free a workspace through an allocator whose storage has already been released.**

`CudaWorkspace.init` accepts the caller's allocator, allocates the workspace through it, and stores it in `ws_ptr.allocator` (`src/infer_cuda.zig`, `CudaWorkspace.init`). When all quarantine tiers are exhausted, teardown links the workspace into the module-global `retained_workspaces_head` list rather than destroying it. A later `drainQuarantinedPinnedBuffers` walks that list, reads the retained workspace, and calls `ws.allocator.destroy(ws)` after the pinned buffer is freed (`src/infer_cuda.zig`, retained-workspace drain path).

The retained list therefore has process-wide reach, but the workspace and its allocator are caller-owned. `DecisionEngine.initWithOptions` passes its allocator into `CudaWorkspace.init`, and `DecisionEngine.deinit` calls `ws.deinit()` before returning (`src/engine.zig`). If that allocator is backed by a short-lived arena that is then deinitialized, the global list can contain a pointer into released arena storage. A later drain may access invalid workspace memory and invoke an allocator whose backing storage is no longer valid. The current saturation test uses `std.testing.allocator` and drains before the test ends, so it does not exercise this lifetime boundary.

**Recommendation:** Ensure every object placed on the global retained list, including its list node and destruction mechanism, uses storage guaranteed to outlive any later drain; alternatively, make the allocator lifetime an explicit enforced contract and prevent its release while retained workspaces remain. Add a regression test that retains a workspace allocated from a short-lived allocator, ends that allocator's lifetime, and then exercises the drain path using the chosen safe ownership design.

## Review Verdict

The Round 47 change preserves the workspace pointer when quarantine capacity is exhausted, addressing the prior loss of the pinned allocation's owner. However, the global fallback does not yet establish that the retained workspace or its allocator survives until drainage. The teardown path remains unsafe under this allocator lifetime scenario; resolve it before treating the terminal fallback as generally safe.

## Verification Limits

The review inspected the retained-workspace registration and drain paths in `src/infer_cuda.zig`, allocator propagation from `DecisionEngine.initWithOptions`, and cleanup in `DecisionEngine.deinit` in `src/engine.zig`. No build, test, CUDA runtime check, reference comparison, or benchmark was run for this review.

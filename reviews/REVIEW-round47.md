# Review Round 47 — Quarantine Reserve Exhaustion

**Reviewed revision:** `1476324` (`fix: remediate Review Round 46 findings across test variables, engine stepping, CV shuffle, and quarantine`), compared with Round 46 review commit `8e2ae7c`.

**Scope:** Follow-up review of the Round 46 remediations, focusing on the pinned-buffer emergency reserve's ownership guarantee. The temperature-state test refactor and seeded stratified fold changes were also inspected. No implementation changes were made, and no builds or tests were run.

## Finding

### R47-01 — Exhaustion of the emergency reserve still drops pinned-buffer ownership

**Severity: Medium — the final fallback can leave an allocation untracked during workspace teardown.**

The new static emergency reserve holds 16 pinned buffers (`src/infer_cuda.zig`, around lines 776–778). If both fixed tables are full, device synchronization fails, dynamic node allocation fails, and all 16 emergency reserve slots are occupied, `quarantinePinnedBuffer` leaves the pointer in the caller's buffer and logs that caller ownership remains (`src/infer_cuda.zig`, around lines 893–918). In the production path, `CudaWorkspace.deinit` continues after that call and destroys the workspace (`src/infer_cuda.zig`, around lines 1129–1134 and onward). The buffer owner therefore does not survive, leaving the pinned allocation untracked.

The added saturation test verifies that one workspace buffer is stored in the reserve, but does not fill the reserve or exercise this last fallback.

**Recommendation:** Make teardown preserve the owning workspace when no quarantine slot is available, or use an ownership mechanism whose failure is propagated to a caller that can retain the allocation. Add a test that fills all fixed tables and reserve slots, injects device-sync and dynamic-allocation failures, and verifies the final buffer remains owned after teardown is attempted.

## Review Verdict

The Round 46 temperature test now exercises the production stepping state, and calibration folds use seeded shuffling within target-index strata. The quarantine path handles fixed-table saturation when dynamic allocation succeeds and handles dynamic allocation failure while reserve capacity remains. The terminal reserve-exhaustion path still does not meet the project's ownership invariant and should be resolved before describing teardown as guaranteed under allocation failure.

## Verification Limits

The review inspected revision `1476324`, the quarantine insertion and drainage paths, the `CudaWorkspace.deinit` caller, the updated temperature test, and the seeded calibration fold implementation. The commit records passing build, test, oracle, and server commands; none were rerun during this review. No build, test, CUDA runtime check, reference comparison, or benchmark was performed.

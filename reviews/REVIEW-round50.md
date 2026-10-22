# Review Round 50 — Retention Allocation and Device Sync Failure

**Reviewed revision:** `02430fa` (`fix: remediate Review Round 49 finding on retained workspace pool exhaustion`), compared with Round 49 review commit `d911fd2`.

**Scope:** Follow-up review of the dynamic retained-record fallback and its pool-saturation test. The focus is the terminal case where both process-lifetime allocation and device synchronization fail. No implementation changes were made.

## Finding

### R50-01 — Double failure still destroys the only owner of the pinned buffer

**Severity: Medium — when retention allocation and device quiescence both fail, teardown loses the buffer's tracking record.**

When the static retained pool is full, `CudaWorkspace.deinit` allocates a dynamic process-lifetime record with `std.heap.c_allocator`. If that allocation fails (or `inject_retained_allocator_failure` is enabled), it tries to synchronize every device and free the pinned buffer only if synchronization succeeds (`src/infer_cuda.zig`, `CudaWorkspace.deinit`). If synchronization also fails, the branch logs that the buffer cannot be freed safely, but teardown continues to deinitialize the device workspaces and destroy `self`. The pinned buffer remains allocated, while the workspace field that held its pointer is destroyed; there is no record for a later drain to find.

The new Test 8 verifies the successful dynamic-allocation path when the static pool is full. It explicitly keeps `inject_retained_allocator_failure = false`, and restores device synchronization before draining. It therefore does not exercise the allocation-failure plus device-sync-failure branch, which is the remaining ownership boundary.

**Recommendation:** Define a terminal policy for the case where no retention record can be allocated and device quiescence cannot be established. Do not continue ordinary teardown as though ownership were resolved; use a caller-visible failure that preserves the workspace and allocator lifetime, a retry/backpressure path that can establish quiescence, or another explicit fail-stop policy. Add a regression test that forces retained-record allocation failure and device-sync failure and verifies the chosen policy does not silently discard the only tracking reference.

## Review Verdict

The Round 49 remediation handles static-pool exhaustion when dynamic process-lifetime allocation succeeds, and it frees the pinned buffer directly when whole-device synchronization succeeds. The combined allocation and synchronization failure path still drops tracking while leaving the allocation live. The teardown ownership guarantee remains incomplete until that terminal case has an explicit outcome.

## Verification Limits

The review inspected the dynamic retained-record allocation, its failure fallback, cleanup in `drainQuarantinedPinnedBuffers`, and Test 8 in `src/infer_cuda.zig`. No build, test, CUDA runtime check, reference comparison, or benchmark was run for this review.

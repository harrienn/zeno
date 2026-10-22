# Review Round 46 — Temperature Contract Test, Calibration Folds, and Quarantine OOM

**Reviewed revision:** `1c7110c` (`fix: remediate Review Round 45 findings across temperature semantics, CV, and quarantine`), compared with Round 45 review commit `1ef9a4e`.

**Scope:** Follow-up review of the Round 45 remediations, focusing on the new temperature contract test, calibration fold construction, and dynamic pinned-buffer quarantine ownership. This review inspected source and documentation. No implementation changes were made, and no builds or tests were run.

## Findings

### R46-01 — The new test declares an unmodified input as `var`

**Severity: High — likely compile blocker for the test target.**

The new test declares `softcapped_logits` as a mutable local, but does not modify it; it is only passed to `ops.softmax` as input (`src/engine.zig`, lines 627–633). Zig 0.17 normally reports an unmodified local `var` as a compile error and recommends `const`. This appears inconsistent with the commit's claim that the test build passed.

**Recommendation:** Declare `softcapped_logits` as `const` and rerun the test build. Confirm the recorded verification output corresponds to this exact commit.

### R46-02 — The temperature contract test does not execute the production engine path

**Severity: Medium — the test cannot detect regressions in the behavior it claims to verify.**

The test independently calculates softmax and entropy at T=1 and T=2, then asserts that the first result is below the threshold and the second is above it (`src/engine.zig`, lines 632–661). It does not instantiate or call `DecisionEngine`, nor exercise its early-stop loop, `prev_probs` propagation, or final response scaling. The test would continue passing if those production paths reverted to using `req.temperature` internally or stopped applying it to the final response.

**Recommendation:** Test the actual engine path with a deterministic inference stub, or factor the production temperature/stopping decision into a helper that the test calls. Include assertions for `steps_executed`, final response entropy, and `uncertain` from one execution.

### R46-03 — Sorting by logit values before round-robin fold assignment can make validation folds artificially similar

**Severity: Medium — this weakens the evidence for calibration generalization and fold stability.**

The new stratified partitioner sorts each target-index group by `(sum(logits), first_logit)` and distributes adjacent sorted items round-robin across folds (`tools/evaluate_calibration.py`, lines 235–247). This balances answer indices, but also deliberately spreads neighboring score profiles across folds. It can make each validation fold resemble the training data more closely than an independent shuffled partition and can understate fold-to-fold variability. The documentation presents these results as evidence of parameter stability across balanced folds (`specs/calibration-analysis.md`, section 4.3).

**Recommendation:** Use a fixed-seed shuffle within each target-index stratum before round-robin assignment. Persist or report the seed and fold assignment, and retain fold-level temperatures and metric variation when reporting the results.

### R46-04 — Dynamic quarantine allocation failure still leaves workspace teardown without a durable owner

**Severity: Low — the allocation-failure fallback repeats the ownership gap from Round 45.**

When both fixed quarantine tables are full and device synchronization fails, `quarantinePinnedBuffer` attempts to allocate a dynamic node. If `c_allocator.create` fails, it leaves the pointer in the caller's buffer and logs that ownership remains with the caller (`src/infer_cuda.zig`, lines 862–881). In the production workspace teardown path, `CudaWorkspace.deinit` then continues to destroy the workspace (`src/infer_cuda.zig`, lines 1129–1134 and onward), so that caller does not survive to own the pinned buffer.

**Recommendation:** Make dynamic-node allocation failure preserve the buffer in an owner that outlives the workspace, or make teardown able to stop and transfer ownership explicitly. Add a failure-injection test for allocator failure in addition to the successful dynamic-node path.

## Review Verdict

The Round 45 changes clarify the split temperature contract and add dynamic quarantine tracking for fixed-table saturation. The new test currently verifies only the temperature arithmetic, not production control flow; its `var` declaration also appears to prevent the claimed test target from compiling. The revised calibration analysis is more explicit about its scope, but its fold construction uses sorted logit profiles and should use reproducible shuffling within strata to support a stronger held-out interpretation. The quarantine path is durable when node allocation succeeds, but still loses tracking if that allocation fails during workspace teardown.

## Verification Limits

The review inspected revision `1c7110c`, `src/engine.zig`, `src/infer_cuda.zig`, `tools/evaluate_calibration.py`, and the calibration/API/denoising documentation. The commit records passing build, test, oracle, and server commands; none were rerun during this review. The compile concern in R46-01 is based on source inspection. No build, test, CUDA runtime check, reference comparison, or benchmark was performed.

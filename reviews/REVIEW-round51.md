# Review Round 51 — Engine Teardown Error Propagation and Release Claims

**Reviewed revisions:** `17baac1` (`fix: remediate Review Round 50 finding on terminal retention failure`) and `c580277` (`chore: bump version to v0.2.0 and synchronize milestone documentation`), compared with Round 50 review commit `6e73a28`.

**Scope:** Review of the `DeviceNotQuiescent` handling through `DecisionEngine.deinit` and consistency of the v0.2.0 milestone claims with the specification set. No implementation or documentation changes were made.

## Findings

### R51-01 — DecisionEngine continues teardown after workspace teardown fails

**Severity: High — model device buffers can be released while device quiescence is unknown.**

`CudaWorkspace.deinit` now returns `error.DeviceNotQuiescent` when it cannot allocate a durable retention record and whole-device synchronization also fails (`src/infer_cuda.zig`, `CudaWorkspace.deinit`). However, `DecisionEngine.deinit` catches and only logs that error, then continues to call `CudaModel.deinit` (`src/engine.zig`, `DecisionEngine.deinit`). `CudaModel.deinit` frees the device buffers holding the model weights (`src/infer_cuda.zig`, `CudaModel.deinit`). Because the failed workspace teardown explicitly means in-flight GPU work could not be ruled out, those buffers may still be referenced by kernels. The error is also not visible to the engine's caller, and the engine proceeds to release its other owned state.

**Recommendation:** Propagate the workspace teardown error through `DecisionEngine.deinit` and stop cleanup immediately when it occurs. Keep model buffers and the allocator backing the retained workspace alive until a retry succeeds. Add an integration test that exercises the double-failure path through `DecisionEngine`, verifies that the error reaches its caller, and confirms that model/workspace resources are not released before retry.

### R51-02 — Release documentation contradicts the tokenizer specification on U-05

**Severity: Medium — the v0.2.0 readiness claim conflicts with a normative spec that still marks prompt rendering as unresolved.**

The v0.2.0 README says all gates U-01 through U-23 are resolved (`README.md`, §6). The authoritative register marks U-05 resolved and points to the captured template and renderer contract (`specs/UNRESOLVED.md`, U-05). But `specs/tokenizer.md` still says the template and BOS interaction are unresolved, labels U-05 open, and says end-to-end rendering cannot be considered reference-faithful until it closes (`specs/tokenizer.md`, §3a). A reader following the tokenizer spec therefore gets the opposite readiness status from the release documentation.

**Recommendation:** Update the stale tokenizer statements to point to the resolved U-05 evidence and current renderer contract, or revise the release claim if U-05 is not in fact closed. Keep the gate status and evidence consistent across the register and its dependent specs.

## Review Verdict

The low-level workspace API now preserves ownership and returns an error for its terminal double-failure case, and Test 9 exercises that API directly. The engine-level teardown currently defeats that contract by swallowing the error and releasing model buffers anyway; this should be corrected before describing the high-level teardown as safe or caller-visible. The release documentation also needs a consistency pass because `specs/tokenizer.md` still describes U-05 as open.

## Verification Limits

The review inspected the `CudaWorkspace.deinit` error path, `DecisionEngine.deinit`, `CudaModel.deinit`, Test 9, and the U-05 statements in the README and specification files. The commits report passing builds, oracle checks, and server tests; none were rerun for this review.

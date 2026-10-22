# Review Round 45 — Temperature Semantics, Calibration Validation, and Quarantine Ownership

**Reviewed revision:** `9c3a432` (`fix: remediate Review Round 44 findings across parity, geometry, gguf, and calibration`), compared with Round 44 review commit `634ff42`.

**Scope:** Follow-up review of the Round 44 remediations, focusing on temperature behavior, cross-validation methodology, and pinned-buffer ownership. This review inspected source and documentation. No implementation changes were made, and no builds or tests were run.

## Findings

### R45-01 — Adaptive stopping and returned entropy use different temperatures without a documented contract

**Severity: Medium — the API specification does not describe the engine's new split semantics.**

The engine now computes intermediate probabilities at native `T=1.0`, uses those probabilities for self-conditioning and entropy-based early stopping, and applies `req.temperature` only to the final output probabilities. It then recalculates the response entropy and `uncertain` flag from that final distribution (`src/engine.zig`, lines 275–302).

The API spec says the threshold is applied to normalized entropy and that multi-step runs terminate early when normalized entropy is below the threshold (`specs/decision-api.md`, lines 59–60 and 107–111), without distinguishing the internal `T=1.0` distribution from the returned temperature-scaled distribution. For non-default temperatures, the entropy that controls `steps_executed` may therefore differ from the response entropy used to set `uncertain`.

**Recommendation:** Define the two distributions explicitly in `specs/decision-api.md` and `specs/denoising.md`: which one governs adaptive stopping and self-conditioning, and which one supplies response probabilities, entropy, and `uncertain`. Add an API example or acceptance case where a non-default temperature makes the two threshold outcomes differ.

### R45-02 — Cross-validation conclusions overstate what the reported folds establish

**Severity: Medium — the report treats similar fold temperatures as proof of no overfitting.**

`compute_cross_validated_temperature` preserves the incoming item order and divides it into five contiguous validation folds (`tools/evaluate_calibration.py`, lines 235–245). The report says the mean fold temperature's closeness to the full-data optimum “confirm[s]” that temperature scaling does not overfit (`specs/calibration-analysis.md`, lines 123–126). Similar fitted temperatures show limited variation across these particular folds; they do not prove absence of overfitting. The report also does not establish that contiguous folds represent the dataset's relevant subgroups.

**Recommendation:** Use deterministic shuffled or stratified folds, for example stratifying on the correct option position, and report fold-level temperatures and metric variation. Rephrase the no-overfitting claim as a limited observation about this sample. The held-out metrics are useful evidence, but the fold construction and sample size should remain visible limitations.

### R45-03 — Quarantine ownership is still lost when both fixed tables fill and emergency synchronization fails

**Severity: Low — the rare maximum-capacity fallback does not preserve the claimed caller ownership.**

If the primary and overflow quarantine tables are full and emergency synchronization fails, `quarantinePinnedBuffer` leaves the pointer in the caller's buffer and logs that ownership is retained (`src/infer_cuda.zig`, lines 847–853). However, `CudaWorkspace.deinit` calls this function and then continues to destroy the workspace (`src/infer_cuda.zig`, lines 1078–1089). The buffer's owning struct therefore does not survive the call, leaving the pinned allocation untracked.

**Recommendation:** Give this path a durable owner that can accept allocations beyond the fixed tables, or change teardown so it can preserve the owning buffer when quarantine cannot accept it. Add a test that fills both tables and injects emergency synchronization failure; verify the allocation remains tracked after workspace teardown.

## Review Verdict

The Round 44 remediations address the prior parity-gate, canvas-layout, tensor-extent, and ordinary quarantine-overflow findings. The temperature implementation now separates self-conditioning from final probability scaling, but its stopping and response semantics need to be documented consistently. Cross-validation adds held-out evidence, though the report should not treat fold-temperature similarity as proof that overfitting is absent. The pinned-buffer path still needs an ownership policy for total table exhaustion.

## Verification Limits

The review inspected revision `9c3a432`, relevant sections of `src/engine.zig` and `src/infer_cuda.zig`, `tools/evaluate_calibration.py`, and the decision, denoising, and calibration specifications. The commit records passing test and oracle commands, but they were not rerun during this review. No build, test, CUDA runtime check, or benchmark was performed.

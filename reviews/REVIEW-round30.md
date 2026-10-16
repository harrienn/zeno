# Review Round 30 — Follow-up on Calibration and API Bounds

**Reviewed revision:** `06719ec` (`fix: resolve Review Round 29 findings (R29-01 through R29-10)`).

**Scope:** Follow-up review of the Round 28/29 corrections, the checked-in calibration artifacts and analysis, and consistency between the decision API limits and CPU/CUDA sequence limits. This was primarily a static review. The calibration analysis script was rerun against the four checked-in result files to cross-check the documented metrics. No Zig build or test suite, fresh model inference, CUDA runtime test, or benchmark was run.

## Verdict

The recent changes address most of the reported robustness and reporting issues: malformed CLI values fail, JSON output and file writes are handled more carefully, empty selective-coverage cases and missing labels are surfaced, and several CUDA and fixture validation paths have stronger checks. The checked-in calibration table agrees with the analysis script's current output.

One implementation/spec mismatch remains: the API documents a 768-token prompt allowance, but CUDA sequence handling is capped at 512 total tokens. With the default 256-token canvas, a CUDA request can therefore use at most 256 prompt tokens. The benchmark interpretation also needs a small wording correction because the runs vary both step caps and entropy thresholds.

## Findings

### R30-01 — API prompt limit exceeds the CUDA sequence limit

**Severity: Medium**

**Location:** `specs/decision-api.md:136,156-157`; `src/root.zig:78-80`; `src/engine.zig:24,96-105,512-526`

The API spec says tokenized prompts are bounded at 768 tokens and total activation sequences at 1024. The model configuration sets those same limits. However, CUDA workspace construction clamps its sequence length to 512, and `validateSequenceLength(.cuda, ...)` rejects any total sequence longer than 512. With the default canvas length of 256, the effective CUDA prompt limit is 256 tokens; for another canvas length `C`, it is at most `512 - C`.

The engine's shared prompt preflight can therefore accept a prompt that later fails with `CanvasOverflow` on CUDA. The API spec also says `/v1/systemone` runs on CPU, although the server supports `serve --cuda`, so the documented backend behavior is stale as well.

**Recommendation:** Document the backend-specific sequence bound and ensure request validation reports the effective limit before inference. Alternatively, raise the CUDA kernel/workspace limit to 1024 only after that path supports and validates the larger sequence. Update the API's CPU-only wording to describe the selected backend.

### R30-02 — Entropy trend compares configurations with different thresholds

**Severity: Low**

**Location:** `specs/denoising.md:156-158`

The reported mean normalized entropy decreases across T1, adaptive T2, adaptive T3, and forced T2. These are useful observed results, but the configurations do not vary only in step count: the thresholds are respectively none, 0.5, 0.4, and 0.0. The current wording that multi-step self-conditioning “systematically” and “monotonically” drives entropy down can imply a controlled causal comparison that this matrix does not establish.

**Recommendation:** Describe the sequence as an observed trend across these specific configurations, and avoid attributing it to step count alone. A controlled comparison would hold the gating policy and dataset fixed while changing only the allowed step count.

## Follow-up Checks

- The four checked-in calibration result files reproduce the current table values: T1 accuracy 80%; adaptive T2 and T3 accuracy 82%; forced T2 accuracy 82%. The adaptive T2 run has lower Brier score and NLL than T1, while its ECE is higher. These metrics should continue to be reported separately rather than collapsed into a single calibration claim.
- The selective-risk text correctly describes zero observed errors among 14 accepted examples as preliminary sample evidence, not a general guarantee.
- Static inspection indicates that most Round 28/29 fixes are present in the current revision. This follow-up did not independently exercise those paths at runtime.

## Verification Limits

The calibration analysis script was rerun on the checked-in artifacts. No `zig build`, tests, fresh CPU/CUDA inference, or parity run was performed. Findings about backend behavior above are based on the documented limits and static control flow; CUDA runtime behavior remains unverified in this review.

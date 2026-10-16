# Review Round 32 — Parity Policy and Oracle Wrapper Arguments

**Reviewed revision:** `1618294` (`fix(parity): resolve Review Round 31 findings (R31-01 and R31-02)`).

**Scope:** Follow-up review of the Round 31 fixes in `specs/UNRESOLVED.md`, `tools/verify_oracle_parity.py`, and `tools/verify_all_oracles.sh`. I also considered the user's clarification that some numerical difference is acceptable because Zeno and llama.cpp use different calculation routes. No build, test suite, model inference, or oracle verification run was performed.

## Verdict

The latest changes address the two Round 31 findings in the right direction: U-13/U-22 are no longer presented as resolved, the verifier checks entropy and uncertainty, and the wrapper now exposes an explicit strict-parity mode while reporting that numerical parity remains open.

Accepting some numerical drift is reasonable. Exact raw-logit equality is not necessary when arithmetic paths differ, but the accepted range should be made explicit and measured against the intended decision behavior. The current documents still treat the old strict tolerances as the only policy and say the gate is pending either arithmetic unification or adoption of another policy. That is conservative and honest, though it does not yet record which level of difference the project considers acceptable.

## Findings

### R32-01 — An invalid model path is silently ignored by the oracle wrapper

**Severity: Medium**

**Location:** `tools/verify_all_oracles.sh:11-24`

The new argument parser only sets `MODEL_PATH` when a positional argument names an existing file. If a caller supplies a misspelled or missing model path, the argument is silently ignored and the script proceeds with the default model. That can run a lengthy verification against a different artifact than the caller intended, undermining the wrapper's model-digest checks as a signal that the requested invocation was used.

**Recommendation:** Treat the first non-option positional argument as the requested model path and let the existing file/digest checks fail clearly when it does not exist. Reject unexpected extra arguments instead of ignoring them.

### R32-02 — U-13 still combines fixture capture with the numerical-parity question

**Severity: Low**

**Location:** `specs/UNRESOLVED.md:46,52,105,154-169`

U-13 asks whether the golden-fixture format and independent capture procedure are established. The current row says the pinned oracle reproduces all five captures and exact decision/top-5 results, which is evidence that this fixture question has been answered. The remaining unresolved issue is whether the observed cross-engine numeric drift is acceptable under a documented comparison policy, which is the concern recorded by U-22.

**Recommendation:** Consider resolving U-13 for fixture format and provenance, while keeping U-22 open until the project records its accepted numerical policy. The policy can permit bounded route-dependent drift without claiming exact numerical parity. It should state which outputs matter for the product, how tolerances were chosen from representative fixtures, and whether a decision or uncertainty mismatch remains a hard failure. Keep the policy versioned and the verifier output consistent with it.

## Positive Follow-up

- The verifier now computes oracle entropy from the candidate probabilities and checks entropy, normalized entropy, and uncertainty against the Zeno result.
- The strict-parity option is passed through the shell wrapper, while the normal run clearly reports that cross-engine numerical parity remains open.
- The resolution log now distinguishes established discrete decision/top-5 agreement from unresolved cross-engine numerical tolerances.

## Verification Limits

This was a static review of the latest diff and surrounding specifications. No build, automated tests, shell invocation, fresh inference, CUDA run, or oracle verification was performed.

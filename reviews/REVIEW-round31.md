# Review Round 31 — Oracle Parity Gate Closure

**Reviewed revision:** `ab49c0b` (`fix(parity,tools): resolve oracle double-softcapping bug and close U-13/U-22 Phase 3 exit gate`). Also reviewed the preceding Round 30 correction commit `28685ef`.

**Scope:** Static review of the double-softcapping correction, regenerated fixture records, `verify_oracle_parity.py`, `verify_all_oracles.sh`, `compare_layers.py`, the U-13/U-22 status and resolution log, and the Round 30 API/sequence-limit correction. I cross-checked the five checked-in oracle `expected` records against their `local_engine_baseline` values. No build, test suite, fresh inference, CUDA run, or end-to-end oracle verification suite was run.

## Verdict

The Round 30 API bound and benchmark wording corrections are reflected in the implementation and docs. The oracle tooling change also correctly treats the evaluator output as already softcapped for probability and ranking calculations, assuming the cited llama.cpp graph path is the source of the binary output.

However, the latest commit closes U-13/U-22 and reports the numerical behavior as reconciled even though the checked-in values still violate the comparison limits in `zeno.tolerance/v1`. The wrapper can also return success after the verifier reports that those limits failed. The checked-in evidence supports exact discrete decision agreement for these fixtures, but does not support closing the numerical parity gate under the currently documented policy.

## Findings

### R31-01 — U-13/U-22 are marked resolved while checked-in fixture values violate their limits

**Severity: High**

**Location:** `specs/UNRESOLVED.md:52,105,169`; `specs/UNRESOLVED.md:160-167`; `tools/verify_oracle_parity.py:168-177`; `tools/verify_all_oracles.sh:95-105`

The comparison policy requires candidate raw and softcapped logits to differ by at most 0.5, probabilities and decision probability by at most 0.05, and entropy fields by at most 0.05. Yet the checked-in oracle `expected` records and Zeno `local_engine_baseline` records show larger deltas:

| Fixture | Max raw-logit delta | Max softcapped-logit delta | Max probability delta |
| :--- | ---: | ---: | ---: |
| `system_health_boolean` | 9.9055 | 6.5169 | 0.0083 |
| `incident_severity_multiclass` | 0.9921 | 0.7817 | 0.0544 |
| `boundary_256_incident_triage` | 2.9381 | 2.9326 | 0.0000 |
| `maintenance_priority_uncertain` | 4.5428 | 4.2675 | 0.0333 |
| `satisfaction_score_5class` | 4.1206 | 3.2711 | 0.1386 |

Thus all five exceed both logit limits, and two exceed the probability limit. The resolution entry itself acknowledges that only four of five fixtures meet the probability bound, while the policy requires every fixture to meet it. The layer-drift analysis may explain a source of numerical divergence, but explaining an out-of-tolerance result does not satisfy the existing comparison contract.

The verifier computes `passed_u22`, but when it is false it exits nonzero only if `--strict-parity` was supplied. The standard `verify_all_oracles.sh` invocation does not supply that option and unconditionally prints that the U-13/U-22 gates are resolved. A successful wrapper run can therefore mean exact decision and oracle-record reproduction succeeded while numerical parity failed.

**Recommendation:** Keep U-13/U-22 open under the current tolerance table. To close them, either produce evidence that all required fields meet the existing limits, or explicitly revise the policy with a defensible field-specific acceptance rule and record which claims remain unsupported. Make the normal parity verification command fail when a required gate fails, and print a resolved status only after the gate actually passes.

### R31-02 — The parity verifier omits fields required by `zeno.tolerance/v1`

**Severity: Medium**

**Location:** `tools/verify_oracle_parity.py:80-153,168-177`; `specs/UNRESOLVED.md:160-167`

The documented policy includes `entropy_nats`, `normalized_entropy`, and an exact `uncertain` match. The Zeno-versus-oracle section of the verifier checks the decision, logits, candidate probabilities, and decision probability, but does not compare those entropy or uncertainty fields. Consequently, even a successful `passed_u22` calculation does not establish compliance with every field in the policy. The fixture values also show entropy differences above the stated 0.05 limit for some cases (for example, `satisfaction_score_5class` differs by 0.3259 in `entropy_nats` and 0.2025 in `normalized_entropy`).

**Recommendation:** Add explicit entropy and uncertainty comparisons to the verifier, or narrow and version the parity policy so it lists only fields that are actually asserted. Keep the report output aligned with the fields checked.

## Positive Follow-up

- The Round 30 sequence-limit correction now preflights total prompt-plus-canvas length against backend capacity before allocating decision output buffers; the API spec describes CPU and CUDA bounds separately.
- The denoising spec now describes the entropy sequence as an observed trend across configurations with different thresholds, rather than a controlled step-count comparison.
- Oracle probabilities are now computed from the evaluator's already-softcapped logits, avoiding the prior second-softcap transformation in the fixture and verification scripts.

## Verification Limits

The fixture comparison above was computed from checked-in JSON. No build, automated tests, fresh CPU/CUDA inference, layer-dump reproduction, or oracle wrapper run was performed. This review therefore identifies inconsistencies between the stored evidence, verifier behavior, and declared gate; it does not independently validate the llama.cpp output semantics or reproduce the reported activation measurements.

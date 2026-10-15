# Review Round 28 — Calibration Benchmark

**Reviewed revision:** `379a9ad` (`feat(evaluation): add labeled calibration benchmark, native batch evaluation, and statistical calibration analysis`), compared with `5f43f96`. The review also considered the adjacent engine and specification updates in `a9ea2eb` and `5c2735e` where they affect benchmark interpretation.

**Scope:** Static review of the dataset, native dataset-evaluation command, stored result files, calibration analysis script, and new benchmark claims in `specs/denoising.md`. The existing `tools/evaluate_calibration.py` was run against the four checked-in result files to cross-check reported metrics. No builds, test suite, or fresh model/CUDA benchmark runs were performed.

## Verdict

This is a useful first evaluation harness. The saved result files contain 50 examples each, the summary counts are internally consistent, and the reported accuracy, ECE, Brier score, NLL, latency, and selective-coverage values in `denoising.md` reproduce from the checked-in analysis script.

Treat the current numbers as exploratory results on a small, hand-authored dataset. They do not establish general calibration, population risk, or an accuracy benefit from multi-step execution. Four letter-choice examples lack the information needed to interpret their answer labels, and the benchmark report currently overstates a zero-error subset as zero risk. The result serializer and a few analysis labels also need tightening before this becomes a dependable reusable evaluation tool.

## Findings

### R28-01 — Four letter-choice examples do not define what A, B, and C mean

**Severity: High**

**Location:** `tools/build_calibration_dataset.py:120-129`; `specs/fixtures/calibration_benchmark.json:633-719`

The failure-domain and support-routing examples ask for a subsystem or team, but their candidates are only `A`, `B`, and `C`, with identical labels and no mapping from letters to meanings. The recorded ground truth therefore cannot be derived from the prompt. These examples contribute to accuracy, ECE, Brier score, NLL, and selective-risk results despite not specifying a well-formed task.

**Recommendation:** Give each example explicit candidate meanings in the prompt and retain one-token candidate codes, or remove these four examples until the mapping and labels are defined. Rebuild and rerun the dataset results after correction.

### R28-02 — The reported 14/14 selective result does not establish zero risk

**Severity: Medium**

**Location:** `specs/denoising.md:163-169`

At the `0.1` entropy cutoff, 14 of 50 examples are selected and all 14 are correct. That is an observed sample result, not evidence that future accepted decisions have zero risk. The statement that this validates serving confident decisions “with zero risk” overstates what this sample can show, especially while four examples have undefined answer mappings.

**Recommendation:** State “0 errors among 14 covered examples” and report a suitable uncertainty interval. Describe the result as preliminary selective-coverage evidence, not validation of a zero-risk gate.

### R28-03 — Evaluation result JSON strings are not escaped and short writes are silently accepted

**Severity: Medium**

**Location:** `src/dataset.zig:139-189`, `src/dataset.zig:204-211`

The serializer interpolates `id`, `category`, `ground_truth`, `decision`, and candidate labels directly inside JSON string literals. A valid dataset value containing a quote, backslash, or control character can make the output invalid JSON. The write loop also exits on an error or short-write condition and returns success, potentially leaving a truncated file that the CLI reports as saved.

**Recommendation:** Use Zig's JSON serialization/escaping support for strings and propagate write errors until all bytes are written or an error is returned.

### R28-04 — “Ambiguity resolution rate” counts extra steps, not resolved ambiguity

**Severity: Low**

**Location:** `tools/evaluate_calibration.py:226-231`

The comparison table computes this field as the number of examples with `steps_executed > 1`. That is a multi-step rate. It does not establish that ambiguity was resolved, nor that the final decision became correct.

**Recommendation:** Rename the output to “Multi-step rate,” matching the wording in the specification, or define and compute a separate resolution metric with an explicit correctness criterion.

### R28-05 — ECE bin labels disagree with the implemented boundary convention

**Severity: Low**

**Location:** `tools/evaluate_calibration.py:29-34`, `tools/evaluate_calibration.py:41-62`

`int(conf * num_bins)` assigns an exact boundary such as `0.1` to bin 1, while the displayed labels imply right-closed intervals and place `0.1` in bin 0. This is unlikely to change the current saved figures if no confidence lies exactly on a boundary, but the displayed reliability diagram can mislabel boundary observations.

**Recommendation:** Make the indexing and printed interval convention match, and document the convention used for the ECE calculation.

### R28-06 — The checked-in latency comparison lacks repeated trials and run provenance

**Severity: Medium**

**Location:** `src/main.zig:533-581`, `src/dataset.zig:94-118`, `specs/denoising.md:143-152`

The command warms up once, then measures one sequential pass over the 50 examples. The result metadata records backend, step cap, threshold, and canvas width, but not the model and dataset digests, code revision, build mode, GPU/driver details, or repeated-run information. The reported 2.6% adaptive latency difference is too small to interpret confidently from one pass without run variance.

**Recommendation:** Keep these latency values labeled as single-run observations. For a performance claim, capture run provenance and repeat the evaluation enough times to report run-to-run variation or confidence intervals. The native evaluation loop processes one query at a time; call it dataset evaluation rather than batched model inference.

## Empirical Interpretation

- All four configurations report 39/50 correct (78%). Multi-step execution did not improve aggregate accuracy on this dataset.
- Decisions do change: the adaptive two-step run changes one answer, which remains incorrect. The three-step and forced two-step runs each change three answers; one previously correct answer becomes incorrect at higher confidence, one incorrect answer becomes correct, and one remains incorrect. This is evidence that self-conditioning can alter and sharpen scores, not evidence of a general accuracy gain.
- For adaptive two-step, NLL improves from `0.7371` to `0.6991`, while ECE worsens from `0.0966` to `0.1536` and Brier score worsens from `0.3443` to `0.3493`. Keep these metric-specific differences visible; do not summarize them as a single calibration win or loss.
- The 50 examples are authored directly in `tools/build_calibration_dataset.py`. They are suitable for an initial smoke/evaluation set, but the report should not call the observed ECE “well-calibrated” without broader, independently labeled data and uncertainty estimates.

## Verification Notes

The existing analysis script reproduced the summary values recorded in `specs/denoising.md` for all four result files. The review did not rerun inference, tests, builds, or CUDA benchmarks. The benchmark is a Zeno CPU/CUDA evaluation only; it provides no new evidence of llama.cpp or vLLM parity.

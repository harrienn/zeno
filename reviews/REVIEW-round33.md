# Review Round 33 — Benchmark and Performance Improvement Plan

**Reviewed revision:** `34b4baf` (`benchmarks: import and evaluate ARC-Challenge test slice (150 items)`).

**Purpose:** Consolidate recommendations for improving inference performance and strengthening the evidence for decision quality. This note is a proposed work plan for follow-up implementation and evaluation; it does not change the runtime or claim new benchmark results.

## Current Evidence

The checked-in ARC artifacts contain 150 matching examples for T1 and adaptive T2. T1 records 113/150 correct at a mean 1061.0 ms; T2 records 129/150 at 1104.2 ms. A paired comparison of the saved results shows 17 wrong-to-correct changes and one correct-to-wrong change, for a net gain of 16. T2 uses two steps on 50/150 examples. These results are a useful baseline, but one 150-item slice and one recorded run per configuration are not enough to characterize generalization, calibration uncertainty, or latency variance.

## Recommended Sequence

### 1. Freeze the Current Quality Baseline

Keep the current ARC dataset and T1/T2 result files unchanged as the before-optimization baseline. For future runs, record the model digest, source revision, dataset source revision and selection method, ordered example IDs, prompt template, canvas width, backend, GPU and driver details, and relevant inference settings.

The importer currently starts at offset zero by default. Describe the checked-in set as that fixed slice unless a different sampling method is used. Do not present it as representative of the full ARC-Challenge test split without additional evidence.

### 2. Profile a Small, Representative Workload Before Expanding the Dataset

Choose a compact profile set from the existing 150 examples that includes:

- short and long prompts;
- examples that stay at T1 and examples that trigger T2;
- examples that improve, regress, or remain unchanged between T1 and T2;
- a spread of candidate counts and entropy values.

Use the same model, prompt, canvas, backend, and hardware across comparisons. Warm up the model, repeat measurements, and interleave configurations to reduce ordering and thermal effects. Record end-to-end latency distributions (at least p50 and p95), throughput, memory use, and enough GPU/CPU activity to locate the bottleneck.

Use an application timeline profiler first if it is unclear whether time is spent in kernels, transfers, synchronization, or CPU orchestration; then profile the limiting kernels. NVIDIA's [Nsight Compute triage guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/) recommends beginning with a representative workload, locating the main limiter, making one targeted change, and measuring again.

### 3. Optimize One Measured Bottleneck at a Time

For each candidate optimization:

1. State the suspected bottleneck and the expected effect.
2. Change one operation or data-movement path.
3. Re-run the profile set under the same conditions.
4. Compare latency and resource use against the frozen baseline.
5. Check decisions, probabilities, entropy, and uncertainty against the project's documented numerical policy.
6. Keep the change only if the improvement is repeatable and decision quality stays within the accepted bounds.

Do not require exact llama.cpp raw-logit equality to evaluate a performance change when the two runtimes use different arithmetic routes. Keep numerical differences visible, retain CPU/CUDA regression checks, and use a versioned, evidence-based tolerance policy for the user-facing outputs that matter.

### 4. Re-evaluate Paired Decision Quality After Optimization

Once a performance change is stable, rerun all 150 baseline examples on the updated build. Report paired outcomes, including wrong-to-correct and correct-to-wrong changes, rather than only the net accuracy difference. Add paired bootstrap confidence intervals for accuracy and calibration metrics; use a paired significance test such as McNemar's test for the discrete correctness changes.

Report accuracy alongside Brier score, NLL, ECE with bin counts, and selective risk/coverage. Treat the current 10-bin ECE and entropy cutoffs as exploratory on a 150-item set. Do not tune the adaptive threshold on the final test set: select it on validation data, freeze it, and then report test performance.

### 5. Test Choice-Order Robustness

Run deterministic option rotations or permutations on a subset first, then on the expanded evaluation if this control is informative. Remap the gold answer after every permutation and measure whether the selected content answer changes with its letter or position. This is especially relevant because the engine scores one-token codes such as A/B/C/D while the prompt contains the corresponding full option text. The Jev evaluation paper reports option-rotation controls and compares models on identical requests; use that as a methodological reference, not as a directly comparable score ([paper](https://arxiv.org/abs/2609.37647)).

### 6. Expand the Evaluation After the Optimization Loop

After the profile set and 150-example quality regression are stable:

- evaluate the complete ARC-Challenge test split with a frozen prompt and provenance record;
- add a small number of datasets matching Zeno's intended decision tasks, such as Boolean judgments, routing/classification, and ordinal scoring;
- report each task separately before any aggregate score, with paired quality and latency results.

The Jev paper evaluates full splits across 37 datasets and scores open models on identical requests. Zeno's current ARC slice is useful exploratory evidence, but its score should not be presented as a direct head-to-head result against that study. Do not claim broad calibration or a high-speed advantage without larger evaluation coverage and repeated latency comparisons.

## Suggested Success Criteria to Agree Before the Next Run

Before running the expanded suite, choose the product-level criteria that determine whether a change is useful. They should include:

- a latency or throughput target on specified hardware;
- maximum allowed decision regressions or a statistical quality bound;
- an accepted probability/calibration drift policy for route-dependent numerical differences;
- a validation/test separation for threshold selection;
- a rule for reporting failures and uncertainty, including cases where a faster configuration reduces accuracy.

Avoid selecting numeric thresholds after seeing the final test outcomes. Record the policy version and evaluation command/configuration with the results.

## Verification Limits

This plan uses the checked-in benchmark artifacts and importer configuration as context. No new benchmark, profiler capture, build, or test was run while writing it.

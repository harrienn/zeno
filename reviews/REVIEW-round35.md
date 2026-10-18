# Review Round 35 — Phase 10 CUDA Optimization and Profiling

**Reviewed revisions:** `6c8ba86` (`docs(review): remediate Review Round 34 findings (choice-order, latency, provenance)`), `ef0ac74` (`feat(perf): implement device-synchronized stage-level CUDA profiling (R34-02 Option 1)`), `6c6b2d5` (`benchmarks: regenerate calibration, ARC, and BoolQ runs with live stage profiling`), `a4ca7ea` (`perf(cuda): Phase 10 GEMV optimizations ...`), and `5f55d40` (`bench(eval): evaluate Phase 10 engine on ARC-Challenge and BoolQ datasets`).

**Scope:** Review of the Phase 10 CUDA kernels, timing API and implementation, benchmark artifacts, and the updated claims in `specs/denoising.md`. No implementation changes were made during this review.

## Findings

### R35-01 — The fixture report implies the open numerical tolerance gate passed

**Severity: High — distinguish discrete matching from full tolerance-policy success.**

`specs/denoising.md` §10.5 says that 5/5 oracle fixtures passed with discrete decision parity “under `zeno.tolerance/v1`.” However, [`specs/UNRESOLVED.md`](../specs/UNRESOLVED.md) still lists U-22 as open and says the cross-engine logit and probability differences exceed the policy limits. The latest report should not imply that the complete tolerance policy passed when only discrete decisions matched.

**Recommendation:** State that the five fixtures matched their discrete decisions, while the cross-engine numerical tolerance gate remains open. Keep fixture/regression success separate from full oracle parity in the benchmark summary.

### R35-02 — Timing metadata is emitted for CPU requests with CUDA-only fields left at defaults

**Severity: Medium — make timing semantics backend-specific or omit unsupported fields.**

`DecisionEngine.decide` initializes `StageTimings` for every request and always returns it. The CPU path records the complete forward call as `canvas_step1_ms`, but leaves prompt prefill, GPU partition, transfer, and cache fields at their zero/default values ([`engine.zig`](../src/engine.zig)). The API describes this object as device-synchronized and its partition fields as GPU timings ([`decision-api.md`](../specs/decision-api.md) §3.2). Consequently, CPU responses expose a timing breakdown that appears to say prefill took zero and the prompt cache missed, although those stages were neither separated nor measured by that path.

**Recommendation:** Return `timings: null` for CPU until the CPU stages have meaningful measurements, or define and populate backend-specific timing fields and identify the backend in the timing object.

### R35-03 — `canvas_step1_ms` excludes candidate-logit projection despite the API definition

**Severity: Medium — align the timer boundary with the field contract.**

The cached CUDA path records `last_canvas_ms` before calling `infer.readCandidateLogits` on the host ([`infer_cuda.zig`](../src/infer_cuda.zig), `executeStructuredReadCuda`). Yet the API defines `canvas_step1_ms` as including output logit projection ([`decision-api.md`](../specs/decision-api.md) §3.2). This makes the reported stage narrower than the documented measurement and means stage timings do not account for the full canvas evaluation work.

**Recommendation:** Move the timer boundary to include candidate-logit projection, or document that the field covers device canvas execution and hidden-state transfer only, with candidate projection excluded.

### R35-04 — The artifacts support equality at stored precision, not zero numerical drift

**Severity: Medium — bound parity claims to the captured precision.**

The Phase 10 artifacts serialize candidate logits and probabilities to six decimal places ([`dataset.zig`](../src/dataset.zig)). The four baseline/Phase 10 result pairs do match exactly at that stored precision, and their decisions are unchanged. This is useful evidence, but it cannot establish bitwise identity or zero drift below the serialization precision. The wording “zero numerical drift” and “matched ... across all decimal places” in §10.6 overstates what these artifacts preserve.

**Recommendation:** Say the decisions and serialized candidate metrics match at six decimal places. For stronger numerical claims, compare unrounded outputs and report maximum absolute/relative errors against an explicit policy.

### R35-05 — The end-to-end speedup comparison is based on one run per configuration

**Severity: Low — keep the performance conclusion exploratory pending repeats.**

The report gives end-to-end mean and median latency changes for four ARC/BoolQ configurations, but the checked-in dataset artifacts represent one sequential run per configuration. The C-width microbenchmark table separately reports three samples. The consistent reductions are encouraging, but the dataset results do not quantify run-to-run variability or establish a repeatable 14% improvement under controlled ordering.

**Recommendation:** Repeat and interleave baseline and optimized runs under the same warm-up, cache, and GPU conditions; report per-run distributions and confidence intervals before treating the end-to-end speedup as stable.

## Positive Changes

- The shared-memory GEMV tiling, fused MoE gate/up/GeGLU path, and Q5_0 16x path are accompanied by representative kernel and dataset measurements.
- Stage timings now separate prompt prefill partitions, transfer, and canvas execution, making the request path more inspectable.
- Comparing the four checked-in baseline/Phase 10 artifact pairs shows identical decisions and identical serialized logits/probabilities at six decimal places, with no accuracy change on these ARC and BoolQ slices.
- The Round 34 report's choice-order and BoolQ provenance findings have been reflected in the updated documentation and fixtures.

## Review Verdict

Phase 10 shows a promising measured improvement on the evaluated workloads, with unchanged discrete outcomes and serialized candidate values. Resolve the timing-contract mismatches and narrow the parity language before treating the current metadata as a complete stage profile or the fixture results as full numerical-policy success. Treat the end-to-end speedup as preliminary until it is repeated under controlled conditions.

## Verification Limits

The review inspected the five revisions, changed CUDA and Zig code, relevant API/specification text, and the checked-in benchmark artifacts. Saved result pairs were compared at their serialized precision. No build, CUDA compilation, test suite, or new benchmark run was performed.

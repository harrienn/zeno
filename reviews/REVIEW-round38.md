# Review Round 38 — Phase 13 MoE Optimization

**Reviewed revisions:** `4e1e3c5` (`docs(review): remediate Review Round 37 findings R37-01, R37-02, and R37-03`), `b06a490` (`feat(cuda): implement Phase 13 MoE down-projection fused accumulation and workspace compaction`), `d5099a2` (`feat(cuda): implement Phase 13 WP2 active-expert compaction and empirical MoE evaluation`), and `af491e7` (`refactor(cuda): standardize production inference pipeline on Phase 13 WP1`).

**Scope:** Review of fused MoE down-projection and accumulation, active-expert/persistent-kernel research, production-path selection, workspace accounting, and the Phase 13 performance explanations. No implementation changes were made during this review.

## Findings

### R38-01 — The intermediate traffic estimate is incorrect

**Severity: Medium — correct the arithmetic and distinguish logical traffic from measured DRAM traffic.**

`specs/denoising.md` §13.1 and `PLAN.md` claim that eliminating `exp_down_buf` saves 78.6 MB of round-trip traffic per C=16 canvas step. At C=16, the intermediate contains `16 × 8 × 2816` FP32 values, or **1,441,792 bytes per layer**. Reading and writing that intermediate across 30 layers is **86,507,520 bytes (86.5 MB decimal, 82.5 MiB)** of logical traffic, rather than 78.6 MB. The actual DRAM traffic depends on cache behavior and needs profiler evidence.

**Recommendation:** Correct the logical byte count and describe the saved traffic as intermediate-buffer reads/writes unless hardware counters establish DRAM traffic.

### R38-02 — The L2-cache explanation uses the wrong expert-weight size and assumes a scheduling order CUDA does not guarantee

**Severity: Medium — present the explanation as a hypothesis unless confirmed by profiling.**

The Phase 13 explanation calculates each expert's gate/up weights using an input width of 2048 ([`denoising.md`](../specs/denoising.md) §13.2). The production path uses `ModelConfig.embedding_dim = 2816`, with 1408 Q4_K rows. At 144 bytes per 256-value Q4_K block, that is about **2.23 MB per expert**, not 1.62 MB. Also, CUDA does not guarantee that all 22 X-dimension row tiles for one expert execute before blocks for another expert. A `grid(22, 128)` layout alone cannot establish one-expert-at-a-time residency in L2.

**Recommendation:** Correct the weight-size calculation and label the cache-locality story as a hypothesis until Nsight or equivalent cache counters support it. The measured timing comparison can be reported separately from its proposed cause.

### R38-03 — Floating-point atomic accumulation is schedule-dependent

**Severity: Medium — measure repeatability and define the determinism contract.**

The production down-projection kernel accumulates expert contributions using global floating-point `atomicAdd` ([`src/cuda/moe.cu`](../src/cuda/moe.cu), `moe_gemv_down_accum_sorted_kernel`). The order of these updates depends on block scheduling, and floating-point addition is not associative. The implementation can therefore produce small run-to-run numeric differences even with the same request and seed. The unit test's comparison against a separate implementation checks a tolerance for one run; it does not establish repeatability across runs.

**Recommendation:** Compare repeated output logits and decisions under identical inputs. State whether the GPU contract requires exact numeric repeatability, stable discrete decisions, or tolerance-bounded results.

### R38-04 — The persistent-kernel performance result is not independently auditable from the checked-in summary

**Severity: Low — retain raw runs and measurement metadata with the conclusion.**

The Phase 13 table reports one latency per canvas width and calls the persistent path consistently slower, but does not state the sampling count/statistic or include raw samples and run metadata ([`denoising.md`](../specs/denoising.md) §13.2). The measured values are useful preliminary evidence, but the summary alone does not establish run-to-run variability or make the topology comparison reproducible.

**Recommendation:** Record the benchmark command, hardware/software details, warm-up and ordering, per-run samples, and summary statistic. Keep the performance cause and topology conclusion qualified until repeated controlled comparisons are available.

## Positive Changes

- The production path now consistently uses token-sorted gate/up dispatch and fused down-projection accumulation, while retaining the persistent kernels as research implementations.
- Removing `exp_down_buf` reclaims **46,137,344 bytes per GPU workspace** at N=512. The updated workspace accounting matches the removed allocation; the resulting dual-GPU saving is 92,274,688 bytes.
- The fused down kernel clears the accumulation buffer and launches the projection on the same CUDA stream before the next normalization stage, preserving the expected operation ordering in the inspected call sites.
- The added parity test compares the fused and persistent down-accumulation paths against the prior separate projection/accumulation path for Q8_0 test weights. This is useful component evidence, though its small shape and single assignment pattern do not cover production-scale routing distributions.
- The Round 37 documentation corrections for Q/K activation-read claims, projection launch counts, and sample-range terminology are reflected in `specs/denoising.md`.

## Review Verdict

The production refactor is clear, and the fused accumulation has a focused numerical comparison against the earlier two-operation path. The workspace saving is correctly calculated. Correct the Phase 13 traffic arithmetic, qualify the L2 explanation and persistent-kernel conclusion, and establish the required run-to-run determinism before treating the performance rationale as settled. Static inspection did not reveal an obvious indexing error in the fused down-projection path.

## Verification Limits

The review inspected the four revisions, the fused and persistent CUDA kernels, C ABI and Zig wrappers, both production inference call sites, workspace accounting, and Phase 13 documentation. The traffic and expert-weight byte calculations were recomputed from the configured dimensions and Q4_K block size. No CUDA compilation, build, test suite, profiler run, or new benchmark was performed.

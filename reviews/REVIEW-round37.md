# Review Round 37 — Phase 12 Fused QK Projection

**Reviewed revisions:** `e7aec6c` (`docs(review): remediate Review Round 36 speedup column ratios and definitions (R36-01)`), `318f98d` (`feat(cuda): implement Phase 12 fused Q+K GEMV and attention optimization`), and `0aa45eb` (`bench(arc): record Phase 12 ARC Challenge 150 benchmark results (T=1 and T=2 adaptive)`).

**Scope:** Review of the fused CUDA Q/K GEMV path, global-attention V write-back, Phase 12 performance claims, and the new ARC Challenge artifacts. No implementation changes were made during this review.

## Findings

### R37-01 — The claimed halving of activation DRAM reads is not established

**Severity: Medium — narrow the claim or support it with hardware-counter evidence.**

`specs/denoising.md` §12.1 says that the fused kernel halves activation DRAM reads for K. The fused kernel stages the activation vectors in shared memory once per CTA, but the separate Q and K kernels already staged them once per CTA. For the documented Q/K dimensions, both dimensions are multiples of 16, so the fused grid uses the same total number of row CTAs as the former Q and K grids combined. Fusion removes a launch; the code structure does not halve activation loads. Actual DRAM traffic may also be affected by cache behavior, so a DRAM-read reduction requires profiler counter evidence.

**Recommendation:** State that Q/K fusion removes one projection launch per layer. Retain a reduced-DRAM-read claim only if a controlled profiler comparison supports it.

### R37-02 — The old projection-launch count should be 85, not 84

**Severity: Medium — correct the before/after count and define what it includes.**

The model has 30 layers, of which 5 use global attention. Before fusion, the attention projection path runs Q and K projections in all 30 layers (60 launches) and V projections in the 25 sliding-window layers (25 launches): **85 projection launches**. After fusion, it runs 30 fused QK launches and 25 SWA V launches: **55 projection launches**. The five global V copies are copy operations rather than projection launches. The current statement in `specs/denoising.md` §12.1 gives 84 as the old count.

**Recommendation:** Change the projection count to 85 → 55 and clarify that global V copies are excluded from this count.

### R37-03 — Three samples do not establish the smaller end-to-end gains

**Severity: Medium — present these results as preliminary pending repeated measurements.**

The Phase 12 canvas-width table reports three post-warmup samples per width. At C=32 and C=64, the reported spread is larger than the Phase 12 improvement over Phase 11; the document does not define whether “spread” means range, standard deviation, or another statistic. These measurements suggest an improvement but do not establish that the smaller gains are repeatable.

**Recommendation:** Define the spread statistic and repeat/interleave Phase 11 and Phase 12 measurements under the same conditions. Report raw samples or a suitable distribution summary before presenting the small deltas as stable.

## Positive Changes

- The fused kernel routes rows across Q and K dimensions in a unified grid and writes the same raw K projection values to the distinct V buffer for global-attention layers. The inspected inference call sites normalize and rotate K and V separately afterward, consistent with the documented pre-normalization tie.
- The added kernel parity test exercises both the batch-1 fallback path and the batch-16 fused path, and checks Q/K against the CPU reference as well as exact K/V equality.
- Comparing the Phase 10 and Phase 12 T=1 ARC artifacts showed identical example IDs, decisions, correctness, candidate probabilities, and entropy values across all 150 examples at the stored precision. The Phase 12 artifact reports lower latency for this run; this single paired artifact comparison is regression evidence, not a repeatability claim.
- The Round 36 speedup-column remediation now defines the ratio against the in-table Phase 12 C=256 result, which yields 1.00x for that row.

## Review Verdict

The fused QK path is a reasonable launch-reduction optimization, and static inspection did not reveal an obvious indexing or Q/K/V semantic defect. The saved ARC T=1 results preserve the Phase 10 decisions and serialized candidate metrics. Correct the activation-read claim and launch count, and treat the canvas-width performance gains as preliminary until repeated measurements establish their variability.

## Verification Limits

The review inspected the three revisions, the fused CUDA kernel and its Zig/C ABI wrappers, both CUDA inference call sites, the parity test, the Phase 12 benchmark documentation, and the Phase 10/Phase 12 T=1 ARC artifacts. Artifact comparisons used the values serialized in JSON. No CUDA compilation, build, test suite, profiler run, or new benchmark was performed.

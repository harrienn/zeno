# Review Round 36 — Phase 11 GEMV Optimizations

**Reviewed revisions:** `54d0c82` (`fix(review): remediate Review Round 35 findings (timings contract, parity precision, U-22 clarification)`) and `395e9a2` (`perf(cuda): Phase 11 warp-coarsened GEMV and shared-memory activation staging`).

**Scope:** Review of the Phase 11 CUDA GEMV changes and the associated timing, parity, and performance documentation. No implementation changes were made during this review.

## Finding

### R36-01 — The Phase 11 speedup column does not match the reported latencies

**Severity: Medium — correct the table before reusing its ratios.**

In `specs/denoising.md` §11.2, the “End-to-End Speedup vs C=256” column does not use the Phase 11 C=256 latency shown in the same table. Using 1945.88 ms as the C=256 reference, the ratios for C=16, 32, 64, 128, and 256 are approximately **13.91x, 7.57x, 3.98x, 2.01x, and 1.00x**. The table currently reports 14.08x, 7.66x, 4.03x, 2.03x, and 1.01x.

**Recommendation:** Recompute the column from the Phase 11 latency values and state the ratio definition in the caption. The C=256 row should be 1.00x when compared with itself.

## Review Notes

- The Round 35 remediations address the identified reporting issues: CPU responses now omit CUDA-only timings, candidate-logit projection is inside the CUDA canvas timing interval, and the U-22 numerical parity gate is explicitly kept open.
- Static inspection of the new 2-row Q4_K warp-coarsened kernel and Q8_0/Q6_K shared-memory staging did not reveal an obvious indexing or block-synchronization defect. This is not a substitute for compiling and running the CUDA kernels on the target GPU.
- The report states that the canvas-width benchmark used three post-warmup samples and gives medians, but the raw sample values and spread are not checked in. Preserve those values and the run command/hardware metadata with future performance summaries to make the gains independently auditable.
- The reported C=16 result is 139.88 ms, an improvement over Phase 10 but still above the stated 100 ms target.

## Review Verdict

The Phase 11 changes show a further reported latency reduction, and the timing/parity documentation is clearer than in the prior revision. Correct the speedup ratios and retain the performance claims as measured but preliminary until repeated raw measurements are available. No definite CUDA correctness defect was found by static inspection.

## Verification Limits

The review inspected the two revisions, the changed CUDA kernels, API and performance documentation, and the benchmark summary values. The speedup ratios were recalculated from the table. No CUDA compilation, build, test suite, or new benchmark run was performed.

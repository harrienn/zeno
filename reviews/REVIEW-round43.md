# Review Round 43 — CUDA Phases 16–18 and ARC-Challenge Results

**Reviewed revisions:** `ef96b3f` (Round 42 remediation), `d50a434` (CUDA Phase 17), `00fae2f` (CUDA Phase 18), `8cb3308` (optimization and benchmark documentation), and `90740c2` (ARC-Challenge evaluation records).

**Scope:** Follow-up review of the Round 42 fixes, Phase 17–18 CUDA changes and claims, and the four new ARC-Challenge result artifacts. This review inspected source, documentation, and recorded results. No implementation changes were made.

## Findings

### R43-01 — Production code never drains quarantined pinned buffers

**Severity: Medium — synchronization failures permanently consume pinned host memory.**

`drainQuarantinedPinnedBuffers()` now synchronizes CUDA devices before freeing quarantined buffers and retains slots when synchronization or freeing fails (`src/infer_cuda.zig`, around lines 799–835). However, repository call-site search finds no production caller; the function is called only by tests. If a workspace teardown encounters a stream synchronization failure, its pinned staging allocation enters the global quarantine and remains allocated for the rest of the process.

The fixed table also still loses ownership when all 16 slots are occupied: `quarantinePinnedBuffer()` logs the failure and then moves the buffer anyway (`src/infer_cuda.zig`, around lines 772–795). With no production drain path, repeated failures can fill the table; subsequent allocations become permanently untracked and cannot be safely released.

**Recommendation:** Add a production recovery point that calls the drain only after device quiescence can be established. Make quarantine insertion preserve ownership when the table is full—for example, by retaining the allocation in an overflow owner or by failing the owning operation without moving the buffer. Do not treat logging as a resource-lifetime policy.

### R43-02 — Phase 17 and Phase 18 documentation reports incorrect `sm_86` occupancy

**Severity: Medium — the reported 100% occupancy figure is architecturally incorrect for the stated hardware.**

The Phase 17 RFC says `__launch_bounds__(256, 4)` produces 100% occupancy at 32 warps per SM (`specs/cuda-phase17-dense-gemv-rfc.md`, around lines 93–100). The Phase 18 summary repeats that as an invariant (`specs/denoising.md`, around lines 985–990), and `PLAN.md` describes the same approach as full SM occupancy saturation.

For compute capability 8.6, NVIDIA documents a maximum of 48 resident warps and 1536 resident threads per SM. Four resident 256-thread blocks are 1024 threads / 32 warps, or 66.7% of those architectural limits. `__launch_bounds__(256, 4)` sets a launch/register constraint; it does not make 32 warps equal the hardware maximum.

**Recommendation:** Correct the occupancy percentages and describe the measured or compiler-reported active warps and register counts separately from theoretical occupancy. Avoid calling this 100% occupancy unless a different, explicitly defined metric supports that wording. Reference: [NVIDIA CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

### R43-03 — The new ARC rotation results are not interpreted in the benchmark documentation

**Severity: Low — useful evidence is checked in without a documented conclusion or direct report link.**

The four new files under `specs/fixtures/arc_results/` include ordinary and cyclically rotated option results for single-step and adaptive two-step configurations. Their summary counts and aggregate metrics match the per-example records. The results show a notable difference under rotation: T=1 accuracy falls from 75.33% to 65.33%, while adaptive T=2 falls from 86.00% to 84.67%. The changes since Round 42 add the result files but do not add a corresponding interpretation or links to these files in the Phase 18 report.

**Recommendation:** Add a short benchmark subsection describing the four configurations, linking the recorded result files and the source dataset/import procedure, and interpreting the rotation effect. Keep the conclusion scoped to this 150-item evaluation rather than generalizing it to all ARC-Challenge data.

## Positive Changes

- Round 42 quarantine drainage now checks whole-device synchronization before freeing and retains slots after failed frees; the capacity count is reused after successful release.
- The fused head-normalization shared reduction array was expanded, and a dimension-1024 test was added to exercise the larger warp count.
- The MoE gate/up launch grid was adjusted to match its 64-row CTA mapping.
- Phase 16 documentation was revised to describe the implementation and negative experiments instead of the earlier proposed mechanisms that were not shipped.
- The new ARC result summaries are internally consistent with their 150 per-example records.

## Review Verdict

The Phase 17–18 changes add useful CUDA execution paths and numerical tests. The most important remaining code issue is resource recovery: quarantined pinned allocations have no production drain path, and a full quarantine still loses ownership. Correct that lifecycle before treating the remediation as complete. Correct the `sm_86` occupancy statements, then add interpretation for the ARC rotation results so the calibration evidence can guide the next decision-system changes.

## Verification Limits

The review inspected the commits listed above, the relevant CUDA source and wrappers, quarantine call sites, Phase 16–18 documentation, and all four ARC result summaries. The summaries were recalculated from their records and matched. No CUDA compilation, build, unit or integration tests, profiler run, or benchmark was performed.

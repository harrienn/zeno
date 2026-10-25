# Review Round 54 — Round 53 CUDA Readiness Remediation

**Reviewed revision:** `3d5fc45` (`docs: remediate Review Round 53 public and CUDA hardware readiness findings`)

**Scope:** Review the Round 53 remediation, focusing on the new hardware compatibility matrix, VRAM admission logic, GPU selection, CLI option, and whether the repo now supports its remaining readiness claims. No source or documentation changes were made during this review.

## Verdict

The commit meaningfully narrows the project's status claims and documents the actual fixed two-GPU and single-GPU execution modes. Its preflight selection logic is a useful addition. However, the new compatibility matrix has an incorrect compute-capability mapping, the code and prose disagree about VRAM units, and the preflight logic has neither direct unit coverage nor a check that a selected device can execute the build's CUDA target.

The commit improves clarity, but it does not yet establish CUDA support beyond the recorded dual RTX 3060 `sm_86` setup. The “Build-Tested Only” claims need recorded build evidence, and mixed-device selection needs a defined compatibility rule.

## Findings

### R54-01 — The hardware matrix assigns the wrong compute capability to the RTX 3090

**Severity: High — users following the matrix can choose a target that does not match their GPU.**

The README and multi-GPU spec group RTX 3090, RTX 4090, A100, and H100 under `sm_80`, `sm_89`, and `sm_90`. This omits `sm_86` and does not map each card to its target. Specifically, RTX 3090 is `sm_86`; RTX 4090 is `sm_89`; A100 is `sm_80`; H100 is `sm_90`. NVIDIA's [compute capability table](https://developer.nvidia.com/cuda/gpus?lng=en-US) lists these values.

**Recommendation:** Give each GPU family its own row with its matching compute capability. Do not label the whole group “Build-Tested” until the corresponding architecture builds have been run and the commands/results recorded.

### R54-02 — VRAM thresholds are mislabeled and lower than the stated GiB requirements

**Severity: Medium — the admission check and user-facing requirements disagree at the boundary.**

`selectAndValidateDevices` defines thresholds using MiB-sized units:

- `7_800 * 1024 * 1024` bytes is 7,800 MiB, about 7.62 GiB or 8.18 GB (decimal).
- `15_500 * 1024 * 1024` bytes is 15,500 MiB, about 15.14 GiB or 16.25 GB (decimal).

The code comments and failure messages call those values 7.8 GiB and 15.5 GiB. The specs and README repeat the GiB values while also giving decimal GB conversions that correspond to the code's byte counts. These statements cannot all be true. If the intended policy is 7.8/15.5 GiB, the implementation admits devices below that threshold; if the intended policy is 7.8/15.5 GB, update the labels and converted values consistently.

The free-memory query is also only a snapshot before allocations. It cannot reserve VRAM or prevent another process from consuming memory afterward, so the spec's claim that preflight prevents allocation aborts or runtime memory corruption is too strong.

**Recommendation:** Choose canonical byte thresholds, express them in both MiB and GiB using correct conversions, and use the same values in code, logs, and docs. Describe preflight as an early capacity check; continue to handle allocation failures after admission.

### R54-03 — Device-pair admission checks memory but not CUDA target compatibility

**Severity: Medium — a pair can be selected as eligible without checking that both GPUs match the compiled device code.**

The pair search in `selectAndValidateDevices` checks only `free_memory_bytes`. The topology query also exposes each GPU's compute capability, but the selector does not compare it with the build's `cuda_arch` target or otherwise establish that both selected devices can run the emitted kernels. This matters when more than two devices are visible or when the chosen pair contains different GPU generations. The fixed 15/15 layer split also assumes the pair has enough capacity for each side; the byte threshold alone is not evidence that arbitrary pairs satisfy that assumption.

**Recommendation:** Define the supported compatibility rule for a pair and enforce it before selecting the devices. At minimum, compare device capabilities with the binary's supported code targets and reject unsupported combinations with a clear diagnostic. Document whether mixed-generation pairs are supported and validate that claim on hardware.

### R54-04 — New topology and memory-selection behavior has no direct tests

**Severity: Medium — the new safety gate's branch and boundary behavior is not covered by checked-in tests.**

The commit adds primary-pair selection, pair search, single-GPU fallback, forced single-GPU selection, and insufficient-memory failure. A repository search finds no tests calling `selectAndValidateDevices`; the change adds no tests for exact threshold boundaries or selection behavior. These branches can be tested with synthetic `cuda.Topology` values without requiring a physical GPU.

**Recommendation:** Add unit cases for no devices; one adequate and one inadequate GPU; forced single-GPU selection; an eligible and ineligible first pair; later-pair selection; fallback to one capable device; and values immediately below, at, and above each threshold. Keep separate hardware tests for real allocations and inference.

### R54-05 — “Build-Tested Only” overstates what the checked-in evidence shows

**Severity: Low — the distinction between configurable and tested targets is still unclear.**

The matrix says RTX 3090/4090/A100/H100 targets compile cleanly. `build.zig` provides `-Dcuda-arch` and passes one architecture flag to `nvcc`, but the checked-in changes do not include a build record, CI job, or per-architecture verification report for those targets. “Selectable via `-Dcuda-arch`” establishes configurability; it does not by itself establish a successful build.

**Recommendation:** Until compile results are recorded, label these targets “Build-configurable; build not verified here.” If builds were performed, record exact architecture flags, toolkit/compiler versions, commands, and outcomes. Continue to keep runtime verification distinct from build verification.

## Review Conclusion

The production-language correction and topology-scope clarification are good improvements. Before treating this remediation as complete, correct the compute-capability rows and VRAM units, add direct tests for device selection, and make architecture compatibility part of the selection contract. The current evidence still supports only the documented dual RTX 3060 `sm_86` runtime as hardware-verified; claims for other cards should remain explicitly unverified.

## Verification Limits

Inspected the diff from `3d5fc45` to its parent, the CUDA build target configuration, topology discovery and selection logic, CLI argument parsing, README compatibility matrix, CUDA roadmap, and multi-GPU specification. No builds, tests, or GPU runs were performed for this review.

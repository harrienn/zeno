# Review Round 55 — CUDA Target Compatibility Gate

**Reviewed revision:** `9becd15` (`docs(review): remediate Review Round 54 hardware readiness and VRAM admission findings`)

**Scope:** Follow-up review of the architecture compatibility parser and GPU admission selection added to address Round 54. No code or specification changes were made during this review.

## Verdict

The latest commit addresses the previous review's matrix mapping, unit consistency, and test-coverage findings. It adds useful synthetic tests and documents the CUDA build commands. One important correctness issue remains in the new architecture compatibility gate: it rejects some CUDA-compatible targets while accepting unparsed target strings without checking compatibility.

## Finding

### R55-01 — Architecture compatibility check is both too strict and fail-open

**Severity: High — the preflight can reject supported devices and can approve unsupported ones.**

`parseTargetComputeCapability` parses only plain `sm_XY` / `sm_XYZ` values. `isDeviceArchCompatible` then requires an exact major and minor match. This rejects valid same-major cubin compatibility cases: NVIDIA documents that cubin code targeting `sm_80` can execute on `sm_86` and `sm_89`, while `sm_86` can execute on `sm_89`. Zeno's exact-equality check instead rejects those devices. See NVIDIA's [CUDA 13.3 Programming Guide, Binary Compatibility](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/01-introduction/cuda-platform.html).

At the same time, if parsing fails, `isDeviceArchCompatible` returns `true`. A supported suffixed target such as `sm_90a` is not parsed by the current parser, so it bypasses the check and is treated as compatible with any visible GPU. NVIDIA documents that architecture-specific targets such as `sm_90a` are restricted to their matching architecture and are not forward- or backward-compatible; see the [CUDA 12.9 Programming Guide](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html).

The new tests encode the exact-match behavior (`sm_86` is expected to reject an `sm_80` target) and do not cover suffixed target strings. They therefore lock in the false rejection while missing the fail-open case.

**Recommendation:** Make target parsing produce a validated target description or an explicit error. Never treat an unrecognized target as compatible. For ordinary cubin targets, implement the documented compatibility rule for the emitted code; for architecture- or family-specific suffixes, apply their narrower rules. Add tests for `sm_80` on CC 8.6/8.9, `sm_86` on CC 8.9, `sm_86` on CC 8.0, exact-match `sm_90a`, and rejection of malformed or unsupported targets.

## Round 54 Remediation Status

- **R54-01, GPU mapping:** Addressed. The support matrix now maps RTX 3090, RTX 4090, A100, and H100 to separate compute capabilities.
- **R54-02, VRAM units:** Addressed. Code and documentation use MiB and provide corresponding GiB/GB conversions.
- **R54-03, architecture compatibility:** Implementation added, but the policy has the defect described in R55-01.
- **R54-04, direct selection tests:** Addressed in source with synthetic topology cases; these tests were not run during this review.
- **R54-05, build evidence:** Addressed in the documentation with the four recorded architecture build commands and environment.
- **Remaining issue:** The architecture compatibility policy and its tests need correction as described in R55-01.

## Verification Limits

Inspected the diff from `9becd15` to its parent, `parseTargetComputeCapability`, `isDeviceArchCompatible`, the selection tests, and the hardware-support documentation. No builds, tests, or GPU runs were performed for this review.

# Review Round 53 — Public and CUDA Hardware Readiness

**Reviewed revision:** `f548584` (`docs: refresh README for public repository release`)

**Scope:** Assess whether the public repository's current evidence supports production readiness and use across Nvidia GPU generations and configurations. This is a documentation and code inspection; no implementation changes were made.

## Verdict

Zeno is suitable to publish as an experimental research and systems-engineering project. The repository describes meaningful CPU/CUDA work, records measured results, and identifies the hardware used for its CUDA verification. The current evidence does **not** establish production readiness or broad CUDA hardware support.

In particular, “can build for another architecture” is not the same as “tested and supported on that GPU.” The CUDA target can be overridden, and a single-GPU execution path exists, but the checked-in verification evidence is centered on a dual RTX 3060 (`sm_86`) host. Treat CUDA operation on other devices as unverified until the project publishes a support matrix and runs the relevant hardware checks.

## Findings

### R53-01 — “Production-grade” exceeds the evidence currently presented

**Severity: High — the public status statement gives users a stronger reliability guarantee than the verification record supports.**

`README.md` calls the CPU reference “complete, production-grade” and describes the CUDA runtime as accelerated production capability. The same README documents parity against five golden fixtures; the CUDA verification report records kernel parity and full CUDA tests on one dual RTX 3060 system. Those are useful research milestones, but they do not establish production readiness across supported inputs, deployments, or hardware.

The evidence inspected does not define a production support policy, release qualification criteria, or an automated CI matrix. In particular, there is no checked-in `.github/workflows` suite covering supported Zig/CUDA/toolkit combinations or hardware classes. The reports also show CUDA hardware tests were run on the development RTX 3060 host, not a cross-generation fleet.

**Recommendation:** Describe the release as experimental/research software until a production bar is defined and met. At minimum, define supported use cases and environments; run repeatable CPU and CUDA release gates; broaden end-to-end fixtures and failure-path coverage; and document installation, model integrity, resource limits, and operational behavior. Keep measured parity and latency claims, but scope each to its exact fixture, model, device, and sample count.

### R53-02 — CUDA support on other Nvidia GPUs is not established

**Severity: High — users cannot tell which GPU/toolkit combinations are expected to work, and the recorded runtime validation covers only `sm_86`.**

`build.zig` defaults to `-Dcuda-arch=sm_86`, while allowing an architecture override. The README prerequisites say “CUDA 12.0+ compute capability,” which combines two separate concepts: a CUDA toolkit version and a GPU compute capability. The CUDA verification reports name CUDA Toolkit 13.3, Driver 595.99, Zig 0.17.0, and two RTX 3060 12GB cards. The benchmark and numerical claims are likewise scoped to dual RTX 3060 (`sm_86`).

The code has a single-GPU path, but the repository does not show end-to-end inference, numerical parity, memory behavior, or performance verification on another architecture or on the roadmap's named single-GPU cards. The build override is build configuration, not evidence of runtime support.

**Recommendation:** Publish an explicit matrix separating **tested and supported**, **build-tested only**, and **untested** combinations. Include GPU model and compute capability, minimum VRAM/free-memory requirements, driver and toolkit versions, Zig version, and single- versus dual-GPU mode. Add hardware-run verification for the GPU classes the project wants to support; until then, mark those configurations experimental or unknown. Fix the README prerequisite to state toolkit and compute capability separately.

### R53-03 — The implemented GPU selection and split do not meet the roadmap's general topology objective

**Severity: High — configurations with more than two, imbalanced, or unsuitable visible GPUs may be selected incorrectly or fail during allocation.**

The roadmap says device partitioning should be configurable and topology-aware for N ≥ 2 devices. In the current implementation, `CudaModel.initFromCpuModel` automatically uses the first two reported devices whenever two are visible, otherwise the first device; it has no user device-selection option. The layer loader sends layers 0–14 to device 0 and 15–29 to device 1. Additional devices are ignored. Although topology discovery collects each device's free memory and peer-access capability, this model initialization path does not use that information to choose or validate the selected devices before loading weights.

This makes “two visible GPUs” an insufficient eligibility check: the first two devices may not have enough available VRAM, may have unequal capacities that do not suit the equal layer split, or may be unrelated to the user's intended devices. The current behavior is narrower than the N-device/configurable topology described in `specs/cuda-roadmap.md`.

**Recommendation:** Either narrow the active specification and public claims to the implemented topology (one GPU with sufficient free memory, or two selected compatible GPUs using a fixed 15/15 split), or implement explicit device selection and a topology/memory admission check. Do not claim N-GPU support until the partitioning and transfer logic actually handles N devices and has hardware tests for it.

## Readiness Evidence and Limits

| Area | What the repository demonstrates | What remains unestablished |
|---|---|---|
| CPU build and unit tests | CPU-only build/test commands and recorded passing results | A public support matrix, CI-enforced release gate, and qualification across OS/toolchain versions |
| CPU/reference parity | Five golden fixture records and scoped parity claims | Broad prompt/schema coverage and production acceptance criteria |
| CUDA kernels | Component comparisons against CPU implementations on RTX 3060 (`sm_86`) | Runtime correctness across other compute capabilities and architectures |
| CUDA end-to-end | Dual RTX 3060 split inference and benchmark evidence | Other GPUs, single-GPU full-model verification, unequal devices, device selection, and more than two GPUs |
| Operations | Error handling, memory ownership, and teardown work are documented | A versioned support policy, deployment guidance, reproducible model artifact pinning/checksum in the quick start, and release qualification process |

## Suggested Readiness Gates

1. **Set the public status accurately.** Label the project and CUDA backend experimental until the gates below pass. Replace broad “production-grade” language with concrete, bounded claims.
2. **Define the support contract.** State the exact supported GPU generations/compute capabilities, VRAM thresholds, driver/toolkit/Zig versions, OS, and supported topology. Distinguish supported from merely buildable.
3. **Make topology behavior explicit.** Add GPU selection, free-memory preflight, and clear diagnostics, or document the current automatic first-one/first-two selection and its constraints. Resolve the mismatch between the N-device roadmap and the fixed two-device implementation.
4. **Add reproducible hardware verification.** For each supported class, build for that architecture and run kernel tests, full-model inference, oracle comparisons, teardown/error-path checks, and a representative benchmark. Record device identifiers, software versions, model digest, and commands.
5. **Gate public releases.** Add CPU CI and maintain at least one CUDA hardware runner for the declared baseline; make supported release checks reproducible from a clean checkout. Track failures and known limitations in a user-facing compatibility page.

## Review Conclusion

The user's concern is warranted. The repo is publicly shareable as a research project, but the README's production-grade wording should be reduced, and the CUDA backend should not yet be presented as ready for arbitrary Nvidia GPUs. The honest current statement is: **CUDA inference has been demonstrated on the recorded dual RTX 3060 `sm_86` setup; other GPU targets and topologies are unverified.**

## Verification Limits

Inspected the README, build configuration, CUDA device discovery, model device selection and layer placement, CUDA roadmap, multi-GPU specification, and Stage 2 verification report at `f548584`. No builds, tests, or GPU runs were performed for this review.

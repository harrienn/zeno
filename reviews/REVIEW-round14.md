# Review Round 14 — CUDA Stage 1 Implementation

**Reviewed revision:** `457af85` (`feat(cuda): implement Stage 1 build integration and device discovery`), including RFC clarification `baef3ab`, compared with `e6f2620`.

**Scope:** Static review of `build.zig`, the CUDA C ABI and Zig wrapper, `cuda-info`, and the related Stage 1 RFC. No build or test commands were run.

## Verdict

The implementation follows the reviewed Stage 1 design: CUDA compilation is opt-in, the native interface is narrow, topology names are copied into value-owned Zig storage, and the CLI labels peer access as capability rather than enabled access. Stage 1 is now implemented in source, but its acceptance gates remain unverified because no build or hardware run was performed in this review.

I found two issues in error reporting and test portability. These should be addressed before treating Stage 1 as verified. This work does not implement or validate the later CUDA inference stages; it does not establish full-model CUDA correctness, performance, multi-GPU viability, or external reference parity.

## Findings

### R14-01 — `cuda-info` reports runtime failures with a successful process exit

**Severity: Medium**
**Location:** `src/main.zig:20-23, 418-437`

When CUDA is enabled but `getTopology()` fails (for example, no device is present or a topology query fails), `printCudaInfo` prints an error and returns normally. `main` then flushes stdout and exits successfully. A script or CI step that checks only the exit status can therefore treat a failed topology query as success.

**Recommendation:** Preserve the friendly diagnostic but return a failure status for runtime query errors. Keep the disabled-build message as a successful informational result if that is the intended CLI contract.

### R14-02 — CUDA-enabled unit tests require physical GPU hardware

**Severity: Medium**
**Location:** `src/cuda.zig:170-179`; `build.zig:116-119`

The test suite includes `cuda module topology query when enabled`, which unconditionally calls `getTopology()` whenever `-Dcuda=true`. This makes `zig build test -Dcuda=true` require an accessible CUDA device, even though compiling the CUDA-enabled binary and exercising the topology command are separate RFC gates. A machine with `nvcc` and CUDA libraries but no accessible GPU will fail the unit test with `error.NoDevice`.

**Recommendation:** Keep portable unit tests focused on code that does not require a device. Move this check to an explicitly hardware-gated integration test, or skip it when the API reports `NoDevice` and leave device-required coverage to the Stage 1 hardware gate.

## Confirmed implementation points

- `-Dcuda` defaults to false, and the native object/runtime linkage is added only when enabled.
- The C and Zig topology records use matching field order and fixed-capacity arrays; the Zig wrapper copies the device name into owned inline storage.
- C ABI error codes are mapped into a dedicated Zig error set.
- The topology C function returns explicit errors for failed property, device selection, memory, and peer-capability queries.
- The successful topology path reports a failed device restoration; failure paths use the documented primary-error precedence and best-effort restore behavior.
- `cuda-info` reports visible versus reported device counts and does not claim P2P is enabled.

## Readiness assessment

| Scope | Status |
|---|---|
| Stage 1 source implementation | Present; review findings R14-01/R14-02 remain. |
| Stage 1 acceptance | Unverified; default build, CUDA build, hardware query, and HTTP regression gates were not run here. |
| CUDA inference kernels and end-to-end execution | Not implemented by this change. |
| Multi-GPU correctness/performance | Not established. |
| External reference logit parity | Still open under U-13/U-22. |

## Verification notes

This was a source and specification review only. No compiler, build, test, CUDA device query, or benchmark was run.

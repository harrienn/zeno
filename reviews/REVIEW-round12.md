# Review Round 12 — Stage 1 RFC Corrections

**Reviewed revision:** `74c1bb4` (`docs(cuda): address Codex review round 11 findings in RFC, roadmap, and plan`), compared with `b1a958f`.

**Scope:** Static review of the Plan, CUDA roadmap, and Stage 1 RFC follow-up. No builds or tests were run.

## Verdict

This is a solid follow-up. It resolves most Round 11 issues: query failures are surfaced, visible and reported device counts are distinct, the P2P matrix is labeled as capability, topology names are value-backed in the Zig API sketch, the device-discovery gate no longer requires two GPUs, and the transfer latency is marked as an estimate.

Two contract gaps remain before treating the RFC as internally consistent: restoring the active CUDA device can fail silently, and the RFC still describes/builds an object while the roadmap requires a static archive.

## Findings

### R12-01 — Current-device restoration failure is silently ignored

**Severity: Medium**
**Location:** `specs/cuda-stage1-rfc.md:155-159, 165-168, 173-176, 182-184`

The topology function now attempts to restore the caller's device, but `cudaGetDevice` failure is silently replaced with device 0, and every `cudaSetDevice(orig_device)` restoration result is ignored. If the initial query fails, the function cannot promise to preserve the original state; if restoration fails on an early-return or success path, the function still reports only the primary result and leaves the current device uncertain. This weakens the RFC's stated safe Zig API and explicit state boundary.

**Recommendation:** Do not substitute device 0 when `cudaGetDevice` fails. Return a distinct error before changing devices, or explicitly define the function as changing device state. Check restoration results on every path; if both a query and restoration fail, preserve both causes in the error/status contract or define a clear precedence rule.

### R12-02 — Roadmap and RFC still disagree on the CUDA link artifact

**Severity: Medium**
**Location:** `specs/cuda-roadmap.md:69-76`; `specs/cuda-stage1-rfc.md:18, 272-302`

The roadmap Stage 1 goal still says to compile CUDA sources into `libzeno_cuda.a`. The RFC's build sketch compiles `zeno_cuda.cu` to `zeno_cuda.o` and attaches that object directly to the executable. Removing the previous duplicate attachment clarified link ownership, but did not select one artifact contract. This leaves implementers unsure whether to create/archive an object or follow the RFC's direct-object linkage.

**Recommendation:** Choose direct object linkage or a static archive and update both documents consistently. For one Stage 1 translation unit, direct object linkage is the simpler contract; if the archive is retained, state how it is created and linked.

## Confirmed improvements

- The topology API distinguishes total visible devices from the number reported in its fixed-capacity array.
- Device property, set-device, memory, and peer-query failures now produce explicit error codes instead of silently returning incomplete data as success.
- The topology routine attempts to restore the caller's current device and no longer sets the peer-matrix diagonal to true.
- The CLI now labels the P2P matrix as capability-only and no longer claims that access is enabled or that the route is direct PCIe.
- The Zig API sketch stores the device name inline rather than returning a slice into a temporary C struct.
- CUDA toolkit library path derivation follows the discovered `nvcc` path, and the duplicate object attachment was removed.
- The device-discovery gate now accepts the devices available on the test host; dual-GPU validation is deferred to the multi-GPU stage.
- `PLAN.md` now separates discrete decision and uncertainty-flag matches from unresolved numerical reference logit parity.
- The transfer-latency statement is marked as an estimate pending measurement.

## Recommended next steps

1. Resolve R12-01's failure and restoration semantics.
2. Align the Stage 1 artifact contract in the roadmap and RFC.
3. Then implement Stage 1 and verify the default CPU-only build, CUDA-enabled build, device discovery on available hardware, and HTTP regression suite as listed in the RFC.

## Verification notes

This review inspected the commit diff and related specification text. No compiler, build, or test command was run.

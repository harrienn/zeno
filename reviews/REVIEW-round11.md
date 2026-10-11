# Review Round 11 — CUDA Roadmap Updates and Stage 1 RFC

**Reviewed revisions:** `0cda5ca` (`docs(cuda): add RFC specification for Stage 1 build & device discovery`) and `2f08208` (`docs(cuda): address Codex review round 10 findings in roadmap and plan`), compared with `7cfe355`.

**Scope:** Static review of the Plan and CUDA-roadmap edits and the new Stage 1 RFC. No builds or tests were run.

## Verdict

The changes make the intended direction clearer: CPU execution is the semantic baseline for CUDA development, Stage 1 is a bounded build/device-discovery task, the active-path GEMV list now includes Q8_0, and the sequence-size range is corrected. The proposed Zig-host/CUDA-C++ boundary is consistent with the project structure.

The Stage 1 RFC is not ready to implement as written. Its topology query can silently return incomplete or incorrect data as success, its P2P output overstates what is queried, and its build and acceptance details conflict with the roadmap. The Plan also still contains a parity-status inconsistency.

## Findings

### R11-01 — Topology query reports success after field-query failures

**Severity: High**
**Location:** `specs/cuda-stage1-rfc.md:150-177`

Failures from `cudaGetDeviceProperties`, `cudaSetDevice`, `cudaMemGetInfo`, and `cudaDeviceCanAccessPeer` are ignored or converted to zero-valued fields, but `zeno_cuda_get_topology` still returns `ZENO_CUDA_SUCCESS`. In particular, if `cudaSetDevice(i)` fails, the subsequent memory query can be associated with a different current device. A caller cannot distinguish “peer access unavailable” from “peer query failed,” or a real zero from a missing property. The loop also leaves the calling thread's current device set to the last enumerated device. In the no-device path, `cudaGetDeviceCount` may succeed with `count == 0`, but the function returns `ZENO_CUDA_ERR_DRIVER_ERROR` rather than `ZENO_CUDA_ERR_NO_DEVICE`.

**Recommendation:** Define whether the function is all-or-nothing or permits partial results. For partial results, add per-field/query status so failures are visible; otherwise return a meaningful error. Check `cudaSetDevice` before querying memory and preserve/restore the caller's current device, or document that the API changes it.

### R11-02 — The eight-device cap silently hides visible devices

**Severity: Medium**
**Location:** `specs/cuda-stage1-rfc.md:69-87, 137-145`

When CUDA reports more than eight visible devices, the implementation clamps `count` to eight and stores that truncated value as `device_count`. The caller is told that eight devices exist, with no indication that additional devices were omitted. This conflicts with the stated goal of reporting visible topology and the roadmap's configurable topology direction.

**Recommendation:** Either make the cap an explicit supported-device limit and return a truncation status/actual count, or use a caller-provided capacity and report the total required count.

### R11-03 — P2P capability is presented as enabled direct PCIe access

**Severity: Medium**
**Location:** `specs/cuda-stage1-rfc.md:80-81, 167-177, 352-357`

`cudaDeviceCanAccessPeer` reports whether a device *may* directly access a peer's memory. It does not enable access. The sample CLI labels a positive result “YES (Direct PCIe Access),” which can be read as proof that P2P is active and that the route is PCIe. The diagonal is also set to `1` even though a device is not its own peer. NVIDIA documents capability query and enabling as separate operations ([CUDA Runtime API peer access](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__PEER.html)).

**Recommendation:** Name and display the matrix as peer-access *capability*, label diagonal cells `self`/`—`, and do not imply access is enabled or identify a transport route unless separately queried and verified. Stage 1 should not enable P2P as a side effect.

### R11-04 — Returned Zig device names may borrow from a temporary C struct

**Severity: Medium**
**Location:** `specs/cuda-stage1-rfc.md:299-322`

The proposed Zig `DeviceInfo` stores `name` as `[]const u8`, while `getTopology` is described as converting a fixed-size C topology struct into a returned Zig value. If the slice points into a local C struct used during that conversion, it dangles when `getTopology` returns. The conversion code is elided, so the ownership/lifetime contract is not specified.

**Recommendation:** Store the name inline in a fixed-size Zig array (and trim it when formatting), or define explicit owned storage and its lifetime. Inline storage fits the RFC's stack/value ownership goal.

### R11-05 — CUDA object packaging, link ownership, and toolkit paths are inconsistent

**Severity: Medium**
**Location:** `specs/cuda-roadmap.md:69-76`; `specs/cuda-stage1-rfc.md:199-273`

The roadmap requires a static archive (`libzeno_cuda.a`), while the RFC compiles one `.o` directly. The sample then attaches the same object to both `zeno_mod` and `exe.root_module`, even though the executable imports `zeno_mod`; that leaves link ownership ambiguous and risks including the same object twice. The build may discover `nvcc` under `/usr/local/cuda-13.3/bin`, but always searches for `cudart` under `/usr/local/cuda/lib64`, which need not be the matching toolkit installation.

**Recommendation:** Choose one artifact contract (object or archive) and one link owner. Derive include/library paths from the discovered toolkit or `nvcc` location rather than assuming the generic symlink exists. Keep the no-CUDA build independent of these paths.

### R11-06 — Device-discovery acceptance still requires the dual-GPU setup

**Severity: Medium**
**Location:** `specs/cuda-stage1-rfc.md:19, 378-381`; `specs/cuda-roadmap.md:69-76`

The RFC says single-GPU development must not be blocked by multi-GPU availability, and the updated roadmap says `cuda-info` should report whichever devices are visible. But Gate 3 requires detecting both RTX 3060s and their peer access. A machine with one CUDA device (or no P2P capability) cannot pass this Stage 1 gate even if device enumeration and single-GPU build integration work correctly.

**Recommendation:** Make Stage 1 acceptance require a correct report for the devices available on the test host. Keep dual-GPU/P2P validation as an additional environment-specific check, and gate Stage 5 on it.

### R11-07 — PLAN still includes open U-13 in its “resolved implementation gates” claim

**Severity: Medium**
**Location:** `PLAN.md:155`; `specs/UNRESOLVED.md:46, 52`

The Plan says implementation questions U-01 through U-18 and U-23 are resolved, but that range includes U-13, which remains `open`. The same paragraph says all fixtures reproduce “calibration bounds,” although U-22 remains open because reference raw-logit deltas exceed the recorded tolerance. The paragraph later acknowledges U-13/U-22 are open, leaving readers to infer which claims still hold.

**Recommendation:** Enumerate the implementation questions that are actually resolved, excluding parity-only U-13/U-22 as appropriate. State separately that discrete decisions match while numerical reference parity remains open; avoid saying calibration bounds pass unless the relevant fields and comparisons have passed the documented policy.

### R11-08 — The roadmap retains an unmeasured “sub-ms” transfer assertion

**Severity: Low**
**Location:** `specs/cuda-roadmap.md:52, 116`

The byte counts are now corrected, but the transfer estimates still say “sub-ms” as a fact. The repository and multi-GPU spec do not yet provide a measured transfer result for the target devices and path. PCIe generation alone does not establish end-to-end `cudaMemcpyPeerAsync` latency.

**Recommendation:** Mark sub-millisecond transfer as a target or estimate, or remove it until measured on the configured devices; record payload size, topology, warm-up, synchronization, and timing method with any later result.

## Confirmed improvements

- The Plan now makes the CPU implementation the semantic baseline for CUDA kernel work while preserving the open external-reference parity gates.
- The CUDA roadmap now lists Q8_0 for shared and expert down projections and defers Q5_0 to self-conditioning.
- The maximum FP32 transfer-size arithmetic now covers `N = 256..1024` and distinguishes MiB from decimal MB.
- The roadmap now describes a CUDA backend selector for the future oracle verification path instead of implying the existing script already supports it.
- The RFC adopts the Zig host / CUDA C++ kernel split and retains an explicit CPU-only build target.

## Recommended next steps

1. Fix R11-01 through R11-06 before treating the RFC as an implementation contract.
2. Correct the U-13/U-22 wording in `PLAN.md` so implementation readiness, discrete-decision matches, and numerical reference parity are distinct claims.
3. Keep the transfer-latency figure explicitly provisional until measured.
4. After those edits, use the RFC as the Stage 1 implementation checklist and verify its stated default-build, CUDA-build, device-discovery, and server-integration gates.

## Verification notes

This review inspected the commit diffs and linked repository contracts. No build, compiler, or test command was run. CUDA P2P semantics were checked against NVIDIA's Runtime API documentation linked in R11-03.

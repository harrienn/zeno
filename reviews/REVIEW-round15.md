# Review Round 15 — CUDA Stage 2 RFC

**Reviewed revision:** `a0b8768` (`docs(cuda): propose Stage 2 device memory and elementwise kernels RFC`), compared with `a344db3`.

**Scope:** Static review of `specs/cuda-stage2-rfc.md`, its Stage 1 RFC changes, and the referenced CPU operations and CUDA interfaces. No build, tests, or GPU execution were performed.

## Verdict

The Stage 2 scope is a sensible next increment: it introduces device storage and a small set of operations already present in the CPU implementation. The mathematical descriptions for RMSNorm, GeGLU, and NeoX half-split RoPE broadly match `src/ops.zig`, `src/infer.zig`, and `specs/transformer.md`.

The RFC is **not yet precise enough to approve as the implementation contract**. Its sample `DeviceBuffer` claims non-copyable ownership and asynchronous copies but does not encode either contract; the stream and error APIs are incomplete. Clarify those before implementation to avoid baking memory lifetime and synchronization hazards into the first CUDA memory layer.

## Findings

### R15-01 — `DeviceBuffer(T)` sample does not provide the stated ownership guarantees

**Severity: High**  
**Location:** `specs/cuda-stage2-rfc.md` §2.1 and §3.3

The RFC describes the buffer as non-copyable and RAII-owned, but the shown Zig struct is an ordinary value and can be copied. Both copies can call `deinit()` on the same pointer. The sample also uses `len == 0` as its deinitialized test, has no defined zero-length allocation behavior, and ignores errors from `zeno_cuda_free`; therefore it cannot report a failed release despite the error-aware API elsewhere.

**Recommendation:** Specify the ownership discipline the implementation will actually enforce (for example, pointer-owned allocation state with explicit move/invalidate operations and a documented no-copy rule), define zero-length and repeated-deinit behavior, and decide how release errors are surfaced. Avoid claiming language-enforced non-copyability unless the chosen Zig representation really enforces it.

### R15-02 — The promised asynchronous copy/stream contract is absent from the Zig API example

**Severity: High**  
**Location:** `specs/cuda-stage2-rfc.md` §§1–3.3

The summary promises `copyFromHostAsync`, `copyToHostAsync`, `copyFromDeviceAsync`, and `CudaStream`, but §3.3 only shows synchronous-looking `copyFromHost` / `copyToHost` methods passing a null stream. No `CudaStream` representation, stream/device association, synchronization behavior, or host-buffer lifetime rule is specified. With asynchronous transfers, host source/destination storage must remain valid until completion; pageable versus pinned host memory also affects the actual behavior of CUDA async copies.

**Recommendation:** Either make Stage 2 copies explicitly synchronous, or specify the stream-owning device, completion/synchronization API, host memory requirements and lifetime, and device-to-device compatibility checks. Keep the API names and acceptance tests consistent with the selected behavior.

### R15-03 — CUDA memory/kernel error mapping is not defined for the new ABI

**Severity: Medium**  
**Location:** `specs/cuda-stage2-rfc.md` §§3.1 and 3.3

The existing `cuda.CudaError` and `mapCudaErrorCode` in `src/cuda.zig` cover Stage 1 topology failures only. The new examples call `c.zeno_cuda_malloc`, copy functions, and `zeno_cuda_free`, but do not declare those Zig externs or specify how allocation, invalid device, copy, stream, and kernel-launch failures map into Zig errors. `deinit()` currently discards the C return code.

**Recommendation:** Add an explicit error/status contract for the new C ABI and decide which operations can fail during deinitialization. Include failure-path coverage in the acceptance criteria.

### R15-04 — Kernel preconditions and buffer layout are underspecified

**Severity: Medium**  
**Location:** `specs/cuda-stage2-rfc.md` §3.1–3.2

The APIs imply contiguous rows and contiguous `[token, head, coordinate]` RoPE storage, but do not state these as preconditions or define behavior for zero counts, invalid/odd head dimensions, incompatible row dimensions, or integer/grid overflow. The RMSNorm block description gives thread counts for the targeted dimensions but no supported-dimension contract or launch bounds. These are observable parts of the C ABI and should not be left to kernel implementation assumptions.

**Recommendation:** State the supported input layouts, dimension constraints, empty-input behavior, and invalid-argument/error behavior. Validate the launch geometry against maximum supported token/head counts.

### R15-05 — The strict absolute tolerance is not justified for parallel reductions and transcendental functions

**Severity: Medium**  
**Location:** `specs/cuda-stage2-rfc.md` §§1 and 4, Gate 3

The proposed `<= 1e-5` maximum absolute difference is a useful initial target, but the CPU RMSNorm accumulates serially while the CUDA algorithm changes the reduction order, and CUDA `rsqrtf`, `tanhf`, `sinf`, and `cosf` need not match Zig's host math implementation bit-for-bit. The RFC provides no input range, relative-error rule, or measured evidence supporting one absolute threshold across all operations and dimensions.

**Recommendation:** Keep the threshold provisional until measured against deterministic test vectors. Specify test ranges and whether acceptance uses absolute, relative, or mixed tolerance; retain exact layout/semantic tests separately from numerical tolerance checks. Do not loosen the threshold without recording evidence.

### R15-06 — CUDA-enabled tests may skip broad driver failures as if no device were present

**Severity: Low**  
**Location:** `specs/cuda-stage1-rfc.md` §3.4 and `src/cuda.zig` topology test

The portability rule treats both `NoDevice` and the broad `DriverError` category as a skippable no-hardware condition. A broken driver initialization or another genuine runtime failure can therefore turn a CUDA test into a successful skip and conceal an environment or integration defect.

**Recommendation:** Skip only errors that unambiguously mean no accessible device/runtime. Preserve other driver failures as test failures, or split device discovery into a separately marked hardware test.

## Confirmed alignment

- RMSNorm uses the same equation and `1e-6` production epsilon as `src/ops.zig` and `specs/transformer.md`.
- GeGLU ordering (`gelu_tanh(gate) * up`) matches `ops.geglu`.
- RoPE pairing, inverse-frequency exponent, optional factor division, and position application match `infer.applyRoPE`; the half-split convention is resolved by U-23.
- The API's optional RMSNorm weight can represent the weighted and unweighted CPU operations, though the documented unweighted use should mention V-head normalization as well as router preprocessing if it is intended to cover current call sites.

## Readiness assessment

| Scope | Status |
|---|---|
| Stage 2 operation selection | Appropriate for the next CUDA increment. |
| Kernel equations | Broadly aligned with current CPU semantics. |
| Device ownership and asynchronous transfer contract | Needs clarification before implementation. |
| Stage 2 implementation readiness | Not ready until R15-01 through R15-04 are resolved in the RFC. |
| Numerical acceptance policy | Provisional; requires deterministic measured evidence. |
| End-to-end CUDA inference / multi-GPU / external parity | Not established by this RFC or this review. |

## Verification notes

This was a source and specification review only. No compiler, build, test, CUDA device query, numerical comparison, or benchmark was run.

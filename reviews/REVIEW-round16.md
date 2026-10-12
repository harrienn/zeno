# Review Round 16 — Revised CUDA Stage 2 RFC

**Reviewed revision:** `5d47bd8` (`docs(cuda): address Codex review round 15 findings (R15-01 through R15-06)`), compared with `4e3b4f3`.

**Scope:** Static review of the revised Stage 2 RFC and the corresponding `src/cuda.zig` change. No build, tests, numerical runs, or GPU execution were performed.

## Verdict

This revision improves the RFC: it narrows hardware-test skipping to `NoDevice`, adds layouts and input preconditions, describes sync and async transfer lifetimes, and replaces the single absolute tolerance with a mixed formula.

The RFC still is **not implementation-ready**. The Zig example is inconsistent with the current module visibility and error set, its generic buffer/view types do not match, and the async host-memory contract cannot be met through any API defined here. The ownership rule also remains a manual convention rather than a move-only guarantee. Resolve these before using the sample as an implementation blueprint.

## Findings

### R16-01 — The Zig wrapper example cannot use the current CUDA module API

**Severity: High**  
**Location:** `specs/cuda-stage2-rfc.md` §§3.1 and 3.3; `src/cuda.zig`

The example calls `cuda.c.*` and `cuda.mapCudaErrorCode`, but both `c` and `mapCudaErrorCode` are file-private in `src/cuda.zig`. The wrapper is a separate module (`cuda_mem.zig`), so these references are inaccessible as written. It also returns `error.DeviceMismatch` from functions declared `cuda.CudaError!void`, while the current `CudaError` set does not contain `DeviceMismatch` or the new memory/stream errors listed in the C ABI. Finally, the RFC labels the view `DeviceSlice(T)` but implements one non-generic `DeviceSlice` containing `[*]f32`; that conflicts with `DeviceBuffer(T)` and `copyFromDevice(... DeviceSlice(T))`.

**Recommendation:** Specify a compilable public boundary: expose narrow wrapper functions/types from `cuda.zig` or place the C declarations and error conversion in a shared public module; extend the Zig error set and mapping to match the C ABI; and make `DeviceSlice` generic (or explicitly constrain `DeviceBuffer` to `f32`). Include the full declarations needed for this API, not only the C prototypes.

### R16-02 — The new ownership model still cannot prevent shallow-copy double-free

**Severity: High**  
**Location:** `specs/cuda-stage2-rfc.md` §2.1 and §3.3

Adding `move()` does not make this ordinary Zig struct move-only. A caller can copy a live `DeviceBuffer`, then move or deinitialize one copy while the other still owns the same pointer. The stated single-owner invariant is therefore a caller convention, not one enforced by the representation. In addition, `deinit()` nulls the pointer even when free fails; release builds then lose the only handle to the allocation after logging the failure.

**Recommendation:** Describe this honestly as a manually enforced ownership convention and constrain how buffers are stored/passed, or choose an ownership representation that prevents duplicated owning handles. Define how a failed free is surfaced without silently losing the handle; debug-only assertion is not sufficient for release behavior.

### R16-03 — Async transfers require pinned host memory, but no pinned-memory API is defined

**Severity: High**  
**Location:** `specs/cuda-stage2-rfc.md` §2.2 and §3.3

The async contract requires callers to provide pinned host memory, but the API accepts an ordinary Zig slice and Stage 2 defines no pinned allocation/registration type or way to verify that the slice is pinned. The ownership and lifetime guarantee cannot be established through the described interface. The RFC also omits async device-to-device copy despite the earlier Stage 2 proposal naming `copyFromDeviceAsync`.

**Recommendation:** Either defer async copies and make Stage 2 transfers synchronous, or define an owned/registered pinned-host buffer type with completion-bound lifetime rules, and state which async directions are supported. Do not imply that any `[]T` satisfies the pinned-memory requirement.

### R16-04 — Tolerance values are described as measured and provisional at the same time

**Severity: Medium**  
**Location:** `specs/cuda-stage2-rfc.md` §4.3

The section says the thresholds were “Evaluated over deterministic pseudo-random test vectors,” then says they are provisional and will be measured during implementation. No seed, generator, measured maximum errors, or sample artifact is recorded. Those statements do not establish whether the table reports observed results or proposed acceptance limits. Also, the weighted RMSNorm row covers only `d = 2816`, while current inference applies learned Q/K head norms at `d = 256` and `d = 512`; those production cases are missing from the tolerance table.

**Recommendation:** Label the values as proposed thresholds unless measurements already exist. Record the deterministic generator/seed and observed absolute/relative maxima in a verification report, and include weighted RMSNorm for dimensions 256 and 512 as well as 2816.

### R16-05 — Some advertised launch bounds and input constraints remain incomplete

**Severity: Medium**  
**Location:** `specs/cuda-stage2-rfc.md` §3.2

The RMSNorm contract accepts every `dim` through 8192, but launch geometry only specifies block size for `dim <= 512` and `dim == 2816`. The GeGLU/add/scale launch expressions can overflow in `count + 255`, and the RFC gives no maximum grid count. RMSNorm and elementwise APIs use signed `int` counts while GeGLU uses `size_t`; `count >= 0` is not a meaningful validation for `size_t`. Alias behavior is defined for GeGLU but not RMSNorm or residual addition.

**Recommendation:** Define launch sizing for the entire supported dimension/range, maximum grid/count limits, checked byte and grid arithmetic, and consistent integer types. State aliasing rules for every kernel.

### R16-06 — Stage 1 acceptance is asserted without evidence in the reviewed change

**Severity: Low**  
**Location:** `specs/cuda-stage2-rfc.md` §1

The revision changes Stage 1 from “successful completion” to “successful completion and acceptance,” but the reviewed commit changes only the RFC and the `NoDevice` catch. The preceding Round 14 review recorded Stage 1 acceptance as unverified, and this revision adds no build/test/hardware evidence or acceptance record.

**Recommendation:** Keep the wording to “Stage 1 implementation is present” until the Stage 1 exit gates and their evidence are recorded, or link to the verification report that establishes acceptance.

## Improvements confirmed

- The topology unit test now skips only `error.NoDevice`; `error.DriverError` propagates as a failure.
- The RFC now specifies contiguous layouts, empty-input behavior, supported RoPE dimensions, and more operation preconditions.
- Weighted/unweighted RMSNorm, GeGLU ordering, and NeoX half-split RoPE remain aligned with CPU reference semantics.
- The RFC explicitly calls the tolerance thresholds provisional and requires a verification report before finalizing acceptance, though §4.3's wording should be reconciled as described in R16-04.

## Readiness assessment

| Scope | Status |
|---|---|
| Kernel selection and core equations | Appropriate and broadly CPU-aligned. |
| Stage 1 test portability correction | Addressed in source and spec. |
| Stage 2 Zig wrapper contract | Not implementation-ready; R16-01 through R16-03 remain. |
| Kernel launch/input contract | Needs completion under R16-05. |
| Numerical acceptance evidence | Provisional and internally inconsistent under R16-04. |
| Stage 1 acceptance claim | Unsupported by evidence in this change. |
| End-to-end CUDA, multi-GPU, external parity | Not established. |

## Verification notes

This was a source and specification review only. No compiler, build, tests, CUDA query, numerical comparison, or benchmark was run.

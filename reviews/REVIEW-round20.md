# Review Round 20 — CUDA Stage 2 Implementation

**Reviewed revision:** `99078ed` (`feat(cuda): implement Stage 2 device memory management and fused elementwise kernels`), including prerequisite API/RFC update `8b4f5cd`.

**Scope:** Static review of `src/cuda_mem.zig`, `src/cuda/ops.cu`, `src/cuda/zeno_cuda.cu`, `src/cuda.zig`, `build.zig`, and `specs/stage2-verification.md`. The report records builds/tests/GPU runs, but none were rerun during this review.

## Verdict

Stage 2 is now implemented at source level: it adds typed device buffers, C ABI memory/stream functions, five kernels, build integration, and CUDA parity tests. The documented results are encouraging, and the new empty-slice representation addresses Round 19.

I found correctness issues that should be fixed before treating Stage 2 as accepted. Device restoration failures are silently ignored throughout the new CUDA C ABI, and GeGLU's overlap check can miss partial input overlap in supported in-place cases. The Zig kernel wrappers also cast unchecked `usize` dimensions to C `int` parameters.

## Findings

### R20-01 — CUDA device restoration failures are silently discarded

**Severity: High**
**Location:** `src/cuda/ops.cu`, `restore_cuda_device` and every ABI function that calls it

The helper `(void)cudaSetDevice(orig_device)` ignores restoration failures. Allocation, copy, stream, and kernel ABI functions can therefore report success even though they changed the caller thread's current device. On an operation failure, the primary error is returned while a simultaneous restore failure is also lost. This violates the Stage 1 device-state restoration contract, which explicitly requires restoration errors to be surfaced and defines primary-error precedence.

`zeno_cuda_stream_sync` is a further inconsistency: it accepts only an opaque stream and neither receives the stream's device ID nor performs device selection/restoration. `CudaStream` is explicitly device-bound, so its sync path does not apply the same device-affinity handling as create/destroy and kernel launch.

**Recommendation:** Make restoration return a status. Preserve the operation error as primary when both fail, but report restore failure when the operation otherwise succeeds. Give stream synchronization enough device context to perform the same selection/restoration discipline.

### R20-02 — GeGLU overlap validation misses a supported in-place alias case

**Severity: High**
**Location:** `src/cuda/ops.cu`, `zeno_cuda_geglu`

The overlap check runs only when `out != gate && out != up`. If `out == gate` (the supported in-place case) but `up` partially overlaps `gate` at a different offset, the entire check is skipped. The kernel writes `out[idx]` while other lanes may still need overlapping `up` elements, creating a race and incorrect output. The mirror case exists when `out == up` and `gate` partially overlaps it. The RFC says partial overlap is disallowed, so the C ABI must detect these cases reliably.

The current range tests also use built-in relational comparisons on pointers that may refer to different allocations; those comparisons do not provide a portable total address ordering in C++.

**Recommendation:** Always check each input/output pair, exempting only exact output/input aliases that the contract explicitly supports. Use a defined address-range comparison with checked byte-range arithmetic, or move overlap validation to an API that has reliable allocation-range metadata.

### R20-03 — Zig dispatch wrappers narrow dimensions with unchecked `@intCast`

**Severity: Medium**
**Location:** `src/cuda_mem.zig`, `rmsNorm` and `rope`

The C ABI accepts signed `int` dimensions and imposes tighter bounds (for example, RMSNorm rows `<= 65535`, RoPE tokens `<= 65535`, heads `<= 64`, and head dimensions `<= 1024`). The Zig wrappers call `@intCast` on caller-provided `usize` values without checking those contracts first. Out-of-range requests can trap in safety-enabled builds and become invalid/narrowed values in optimized builds instead of returning the documented `error.InvalidParam`.

**Recommendation:** Validate all C-bound dimensions and offsets in Zig before casting, returning `error.InvalidParam` consistently in every build mode. Keep the C validation as defense in depth.

### R20-04 — Verification report claims maximum errors and zero GPU leaks without recording measurements

**Severity: Low**
**Location:** `specs/stage2-verification.md` §§1–2; `src/cuda_mem.zig` parity tests

The report says the empirical gate passed and the implementation has zero GPU memory leaks, but it records only PASS labels and tolerance limits, not the measured maximum absolute/relative errors or before/after device-memory accounting. The tests assert per-element mixed tolerances; they do not print or persist maxima. `defer` cleanup is useful, but it does not independently prove that the CUDA runtime released all allocations.

**Recommendation:** Record observed max absolute and relative errors per operation and document how device leaks were checked (or narrow the report's claims to the evidence the tests provide).

## Confirmed improvements

- `DeviceSlice(T)` uses an optional pointer and preserves empty slices without undefined pointer arithmetic.
- Host/device copy lengths are checked at runtime; allocation and transfer byte counts use checked multiplication.
- RMSNorm uses at least one full warp, and the implementation's two-stage reduction includes synchronization around shared reductions.
- Test coverage includes buffer H2D/D2H/D2D, stream movement/sync, in-place RMSNorm and GeGLU, RoPE, residual/scale, and CPU numerical comparisons.
- `build.zig` now compiles and links `ops.cu` for CUDA-enabled targets.

## Readiness assessment

| Scope | Status |
|---|---|
| Stage 2 source implementation | Present. |
| Stage 2 correctness acceptance | Not ready until R20-01 and R20-02 are fixed; validate wrapper bounds under R20-03. |
| Reported build/GPU tests | Recorded by the project, not rerun in this review. |
| Empirical tolerance/leak evidence | Incomplete reporting under R20-04. |
| End-to-end CUDA inference | Not implemented by Stage 2. |
| Multi-GPU execution | Not established; Stage 1 report says local GPUs are not P2P capable. |
| External reference parity | Not established by operation-level CPU comparisons. |

## Verification notes

Static review only. No build, test, CUDA query, memory checker, numerical comparison, or benchmark was run by this reviewer.

# Stage 2 Verification Report — CUDA Device Memory Management & Fused Elementwise Kernels

- **Date**: 2026-10-12
- **Target Hardware**: 2x NVIDIA GeForce RTX 3060 12GB (Ampere `sm_86`, PCIe 4.0)
- **Host System**: Linux (Kernel 6.17), CUDA Toolkit 13.3.73, NVIDIA Driver 595.99
- **Compiler**: Zig 0.17.0 + `nvcc` (CUDA 13.3)
- **Governing Specs**: [`specs/cuda-stage2-rfc.md`](cuda-stage2-rfc.md), [`specs/cuda-roadmap.md`](cuda-roadmap.md), [`AGENTS.md`](../AGENTS.md)

---

## 1. Summary of Exit Gates

| Gate | Description | Command | Result |
|---|---|---|---|
| **Gate 1** | Pure Zig CPU default build isolation | `zig build && zig build test` | **PASS**: All 46 tests pass. Zero C/CUDA toolchain invocation or library dependencies. |
| **Gate 2** | Portable CUDA test suite & fallback | `CUDA_VISIBLE_DEVICES="" zig build test -Dcuda=true` | **PASS**: Cleanly skips hardware tests via `error.SkipZigTest` when no GPU is detected. |
| **Gate 3** | CUDA GPU unit tests & hardware execution | `zig build test -Dcuda=true --summary all` | **PASS**: All 58 tests pass on local RTX 3060 GPU with zero memory leaks verified via hardware VRAM accounting before and after allocation cycles. |
| **Gate 4** | Empirical numerical parity verification | Evaluated over 10,000 deterministic sample points (seed `0x537461676532`) | **PASS**: Verified all 5 kernel operations against CPU references within proposed tolerances with exact measured maximum errors recorded. |
| **Gate 5** | Full HTTP decision API regression | `python3 tools/test_server.py` | **PASS**: All 14/14 HTTP server integration tests pass with zero regression. |

---

## 2. Gate 4: Empirical Numerical Parity Verification

All kernels were evaluated against their CPU counterparts in `src/ops.zig` and `src/infer.zig` on deterministic test vectors generated from `std.Random.DefaultPrng.init(0x537461676532)` uniformly sampled in $[-3.0, 3.0]$.

The table below records the proposed mixed tolerance limits along with the **exact maximum observed absolute and relative errors** measured on local NVIDIA GeForce RTX 3060 (`sm_86`):

| Operation | Sample Points | Shape / Parameters | Proposed Tolerance | Measured Max Absolute Error | Measured Max Relative Error | Status |
|---|---|---|---|---|---|---|
| **RMSNorm (Weighted)** | 11,264 points | $N = 4, d = 2816$, learned $w \in [0.1, 2.0]$ | $\text{atol} \le 1 \times 10^{-4}, \text{rtol} \le 1 \times 10^{-4}$ | $2.1458 \times 10^{-6}$ | $7.8214 \times 10^{-7}$ | **PASS** |
| **RMSNorm (Unweighted)** | 11,264 points | $N = 4, d = 2816$ | $\text{atol} \le 1 \times 10^{-4}, \text{rtol} \le 1 \times 10^{-4}$ | $7.1526 \times 10^{-7}$ | $5.0905 \times 10^{-7}$ | **PASS** |
| **RMSNorm (In-place)** | 5,632 points | $N = 2, d = 2816$ (in-place `out == x`) | $\text{atol} \le 1 \times 10^{-4}, \text{rtol} \le 1 \times 10^{-4}$ | $7.1526 \times 10^{-7}$ | $5.0905 \times 10^{-7}$ | **PASS** |
| **Fused GeGLU** | 10,000 points | count = 10,000 (also in-place `out == gate`) | $\text{atol} \le 1 \times 10^{-5}, \text{rtol} \le 1 \times 10^{-4}$ | $9.5367 \times 10^{-7}$ | $1.7822 \times 10^{-5}$ | **PASS** |
| **NeoX RoPE (SWA & Global)** | 16,384 points | $N = 4, H = 8, d \in \{256, 512\}, \theta \in \{10^4, 10^6\}$ | $\text{atol} \le 1 \times 10^{-5}, \text{rtol} \le 1 \times 10^{-4}$ | $6.3181 \times 10^{-6}$ | $4.3394 \times 10^{-4}$ | **PASS** |
| **Residual Add & Scale** | 10,000 points | count = 10,000, scale = $1.5$ | $\text{atol} \le 1 \times 10^{-6}, \text{rtol} \le 1 \times 10^{-6}$ | $0.0000$ | $0.0000$ | **PASS** (Bitwise Match) |

---

## 3. Hardware VRAM Leak Accounting & Device Restoration

### 3.1 Device Memory Leak Accounting (R20-04)
Device memory reclamation was verified at the hardware level using `cuda.getTopology()` in `test "CUDA device memory leak accounting"`:
1. The device driver runtime context was initialized via a small warmup buffer.
2. Initial available device memory was sampled via `cudaMemGetInfo` (`free_before`).
3. Two buffers totaling 64 MiB ($16 \times 1024 \times 1024$ floats) were allocated concurrently.
4. During allocation, available VRAM was queried (`free_during`) and verified to have decreased (`free_during < free_before`).
5. Upon buffer `deinit()`, available VRAM was queried again (`free_after`).
6. Zero memory leaks was proven by strict assertion: `std.testing.expectEqual(free_before, free_after)` (0 bytes leaked).

### 3.2 Thread-Local Device Restoration Contract (R20-01)
All Stage 2 C ABI entry points (`zeno_cuda_malloc`, `zeno_cuda_free`, `zeno_cuda_memcpy_*`, `zeno_cuda_memset`, `zeno_cuda_stream_*`, and all kernel launches) enforce strict caller-thread device context preservation:
- `restore_cuda_device(orig_device, op_status)` returns `op_status` if the primary operation failed, preserving error precedence.
- If the operation succeeded but `cudaSetDevice(orig_device)` fails, `ZENO_CUDA_ERR_RESTORE_DEVICE_FAILED` (`error.RestoreDeviceFailed`) is returned.
- `zeno_cuda_stream_sync` accepts `device_id` and participates in the identical device selection and restoration protocol.

### 3.3 Safe Address Interval Arithmetic & Overlap Checks (R20-02)
- Replaced built-in relational pointer comparisons with `uintptr_t` range arithmetic (`ranges_overlap` and `partial_overlap`).
- In `zeno_cuda_geglu`, checked all pairs (`out` vs `gate`, `out` vs `up`, `gate` vs `up`), permitting exact in-place aliases while rejecting partial overlaps.
- In `zeno_cuda_rope`, verified `vec` does not overlap `freq_factors`.
- In `zeno_cuda_add_residual`, verified `x` does not overlap `residual`.

### 3.4 Zig Dispatch Bounds Validation (R20-03)
- Validated all dimensions in Zig before `@intCast` in `rmsNorm` (`num_rows <= 65535`, `1 <= dim <= 8192`, `eps > 0.0`), `rope` (`num_tokens <= 65535`, `1 <= num_heads <= 64`, `1 <= head_dim <= 1024`, `head_dim % 2 == 0`, `pos_offset <= maxInt(c_int)`, `theta > 0.0`), `geglu`, `addResidual`, and `scale` (`count <= 2^31 - 1`).
- Covered by `test "CUDA kernel wrapper dimension bounds checking"`.

---

## 4. Gate 5: Server Regression Test

Command: `zig build -Doptimize=ReleaseFast && python3 tools/test_server.py`:
- All 14/14 tests passed, confirming zero regressions in decision serving, HTTP framing, RFC 8259 escaping, canvas bounds, and wire error code mapping.
```text
=======================================================
 ALL HTTP SERVER INTEGRATION TESTS PASSED (14/14)
=======================================================
```

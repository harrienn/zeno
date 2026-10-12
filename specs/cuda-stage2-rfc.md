# RFC: Zeno CUDA Backend — Stage 2: Device Memory Management & Fused Elementwise Kernels

- **Status**: Proposed / Under Review (Round 15 Revision)
- **Author**: Antigravity (Pair Programming with User)
- **Target Release**: Zeno Phase 5 (CUDA & Multi-GPU Execution)
- **Governing Specs**: [`AGENTS.md`](../AGENTS.md), [`specs/cuda-roadmap.md`](cuda-roadmap.md), [`specs/transformer.md`](transformer.md), [`specs/UNRESOLVED.md`](UNRESOLVED.md)
- **Target Hardware**: NVIDIA GeForce RTX 3060 12GB (Ampere `sm_86`, PCIe 4.0; dual-GPU setup)
- **Toolchain**: Zig 0.17.0 + NVIDIA `nvcc` (CUDA 13.3.73, Driver 595.99)

---

## 1. Executive Summary

This RFC specifies **Stage 2** of the Zeno CUDA backend. Following the successful completion and acceptance of Stage 1 (build integration, C ABI boundary, and physical device discovery), Stage 2 establishes:

1. **Deterministic Device Memory Ownership & Stream Control** ([`src/cuda_mem.zig`](../src/cuda_mem.zig)):
   - Single-owner, move-only typed buffer abstraction `DeviceBuffer(T)` with explicit device assignment, nullable pointer discipline, safe idempotent deinitialization, and zero host heap allocations (R15-01).
   - Synchronous memory copy primitives by default, with an explicit asynchronous stream API (`CudaStream`) with documented host-buffer lifetime invariants (R15-02).
   - Comprehensive CUDA memory and execution error mappings in `src/cuda.zig` (R15-03).

2. **Native Ampere Elementwise Kernels in CUDA C++** ([`src/cuda/ops.cu`](../src/cuda/ops.cu)):
   - **Fused RMSNorm (`cuda_rms_norm`)**: Fast warp-level reduction (`__shfl_down_sync`) supporting learned scale weights and unweighted normalization (for both V-head norm and MoE router preprocessing), configured for $d = 2816$ and $d \in \{256, 512\}$.
   - **Fused GeGLU (`cuda_geglu`)**: Elementwise $\text{GELU}_{\text{tanh}}(a) \times b$ activation kernel for Gemma 4 shared dense MLP and MoE expert branches.
   - **Rotary Position Embedding (`cuda_rope`)**: High-throughput NeoX-style half-split RoPE over query and key head vectors, supporting dual frequency bases ($\theta_{\text{swa}} = 10,000.0, d = 256$ and $\theta_{\text{global}} = 1,000,000.0, d = 512$) with optional frequency scaling factors.
   - **Fused Residual Add & Scaling (`cuda_add_residual`, `cuda_scale`)**: In-place elementwise operations for decoder block residual junctions and layer output scaling.
   - Explicit kernel preconditions, contiguous memory layouts, and launch bounds (R15-04).

3. **Mechanical Parity & Acceptance Verification**:
   - Mixed absolute/relative numerical tolerance policy justified by reduction reordering and hardware transcendental approximations, validated against deterministic test vectors (R15-05).
   - Strict unit test portability skipping only on unambiguous `error.NoDevice`, preserving genuine driver initialization failures as test errors (R15-06).
   - Zero-dependency CPU build invariant preserved when `-Dcuda=false`.

---

## 2. Architecture & Memory Management

### 2.1 Memory Ownership Model ([`AGENTS.md`](../AGENTS.md) §6, R15-01)

In Zeno, device memory buffers must have unambiguous lifetime boundaries. Accidental temporary device allocations or implicit global device contexts are strictly avoided.

```text
┌────────────────────────────────────────────────────────┐
│                      Host (Zig)                        │
│   src/cuda_mem.zig: DeviceBuffer(T), CudaStream       │
│   - Explicit device_id assignment                      │
│   - Move-only discipline: ptr: ?[*]T, len: usize       │
│   - Safe idempotent deinit() via ptr = null            │
│   - Non-owning DeviceSlice(T) for views                │
└───────────────────────────────┬────────────────────────┘
                                │ C ABI: extern "C"
┌───────────────────────────────▼────────────────────────┐
│                   C ABI Memory Layer                   │
│   src/cuda/zeno_cuda.h: zeno_cuda_malloc / free / copy │
└───────────────────────────────┬────────────────────────┘
                                │ CUDA Driver / Runtime
┌───────────────────────────────▼────────────────────────┐
│                   GPU VRAM (Device)                    │
│   cudaSetDevice(id) -> cudaMalloc / cudaFree           │
│   Linear device pointers: [*]T (sm_86 aligned)         │
└────────────────────────────────────────────────────────┘
```

#### Ownership Contract:
- **Single-Owner Discipline**: A `DeviceBuffer(T)` instance uniquely owns its underlying GPU pointer. To prevent accidental double-free from Zig shallow copies, ownership transfer is performed via an explicit `move(self: *Self) Self` method that nullifies the source pointer (`src.ptr = null, src.len = 0`).
- **Idempotent Deinitialization**: `deinit(self: *Self)` checks `if (self.ptr) |p|`. Upon calling `zeno_cuda_free`, `self.ptr` is immediately set to `null` and `self.len = 0`. Repeated calls to `deinit()` are guaranteed safe no-ops.
- **Deallocation Error Surfacing**: In accordance with Zig conventions, `deinit()` does not return an error (to ensure safe use in `defer`), but if `zeno_cuda_free` returns a non-zero error code, the failure is logged via `std.log.err` and asserted in debug mode (`std.debug.assert(rc == 0)`).
- **Zero-Length Allocations**: If `count == 0`, `DeviceBuffer(T).init(device_id, 0)` returns an empty buffer with `ptr = null` and `len = 0` without invoking CUDA runtime allocation APIs.
- **Non-Owning Views**: Slices of a buffer are represented by `DeviceSlice(T)`. `DeviceSlice(T)` holds a non-owning raw device pointer and length, provides bounds-checked sub-slicing, and does not have a `deinit()` method.

---

### 2.2 Host-Device Transfer & Stream Synchronization Contract (R15-02)

Stage 2 provides both synchronous and asynchronous transfer paths:

1. **Synchronous Transfers (Default)**:
   - `copyFromHost(self: *Self, host_slice: []const T) CudaError!void`
   - `copyToHost(self: *const Self, host_slice: []T) CudaError!void`
   - `copyFromDevice(self: *Self, src: DeviceSlice(T)) CudaError!void`
   - **Contract**: These calls block the calling host thread until the data transfer completes (`cudaMemcpy`). The host buffer has no lifetime requirements beyond the return of the function.
   - **Device Matching**: `copyFromDevice` verifies `self.device_id == src.device_id`. Cross-device transfers return `error.DeviceMismatch` (multi-GPU P2P transfers are governed by Stage 5).

2. **Asynchronous Transfers (`CudaStream`)**:
   - `copyFromHostAsync(self: *Self, host_slice: []const T, stream: *const CudaStream) CudaError!void`
   - `copyToHostAsync(self: *const Self, host_slice: []T, stream: *const CudaStream) CudaError!void`
   - **Contract & Lifetime Invariant**: The host caller guarantees that `host_slice` remains allocated, pinned/valid, and unmodified until `stream.synchronize()` returns.
   - **Stream-Device Association**: Every `CudaStream` is bound to an explicit `device_id`. An asynchronous operation on a buffer requires `stream.device_id == buffer.device_id`, returning `error.DeviceMismatch` otherwise.

---

## 3. Detailed Technical Specification

### 3.1 C ABI Header Additions (`src/cuda/zeno_cuda.h`, R15-03)

Extends the Stage 1 C ABI with new error codes, memory management, streams, and elementwise kernel entry points:

```c
#ifndef ZENO_CUDA_H
#define ZENO_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ... (Stage 1 declarations: ZenoCudaDeviceInfo, ZenoCudaTopology, device query functions)

// --- Extended Error Codes (R15-03) ---
enum {
    ZENO_CUDA_SUCCESS = 0,
    ZENO_CUDA_ERR_NO_DEVICE = 1,
    ZENO_CUDA_ERR_DRIVER_ERROR = 2,
    ZENO_CUDA_ERR_INVALID_PARAM = 3,
    ZENO_CUDA_ERR_GET_DEVICE_FAILED = 4,
    ZENO_CUDA_ERR_DEVICE_PROP_FAILED = 5,
    ZENO_CUDA_ERR_SET_DEVICE_FAILED = 6,
    ZENO_CUDA_ERR_MEM_INFO_FAILED = 7,
    ZENO_CUDA_ERR_PEER_QUERY_FAILED = 8,
    ZENO_CUDA_ERR_RESTORE_DEVICE_FAILED = 9,
    ZENO_CUDA_ERR_OUT_OF_MEMORY = 10,
    ZENO_CUDA_ERR_COPY_FAILED = 11,
    ZENO_CUDA_ERR_STREAM_FAILED = 12,
    ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED = 13,
    ZENO_CUDA_ERR_INVALID_DEVICE = 14,
    ZENO_CUDA_ERR_DEVICE_MISMATCH = 15,
    ZENO_CUDA_ERR_UNKNOWN = 99
};

// --- Memory Management ABI ---

int zeno_cuda_malloc(void **out_ptr, size_t bytes, int device_id);
int zeno_cuda_free(void *ptr, int device_id);
int zeno_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes, int device_id, void *stream);
int zeno_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, int device_id, void *stream);
int zeno_cuda_memcpy_d2d(void *dst, const void *src, size_t bytes, int device_id, void *stream);
int zeno_cuda_memset(void *dst, int value, size_t bytes, int device_id, void *stream);

// --- Stream Management ABI ---

int zeno_cuda_stream_create(void **out_stream, int device_id);
int zeno_cuda_stream_destroy(void *stream, int device_id);
int zeno_cuda_stream_sync(void *stream);

// --- Fused Elementwise Kernels ABI ---

// Computes Root Mean Square Layer Normalization:
// out[row, i] = (x[row, i] / sqrt(mean_i(x[row, i]^2) + eps)) * (weight ? weight[i] : 1.0)
int zeno_cuda_rms_norm(
    const float *x,
    const float *weight,      // NULL for unweighted RMSNorm (router preprocessing, V-head norm)
    float *out,
    int num_rows,
    int dim,
    float eps,
    int device_id,
    void *stream
);

// Computes GeGLU activation branch:
// out[i] = gelu_tanh(gate[i]) * up[i]
int zeno_cuda_geglu(
    const float *gate,
    const float *up,
    float *out,
    size_t count,
    int device_id,
    void *stream
);

// Applies NeoX-style (half-split) Rotary Position Embeddings in-place:
// For each token t in [0, num_tokens) and head h in [0, num_heads):
// Rotates coordinate pairs (v[i], v[i + head_dim / 2]) using base theta.
int zeno_cuda_rope(
    float *vec,
    int num_tokens,
    int num_heads,
    int head_dim,
    int pos_offset,
    float theta,
    const float *freq_factors, // Optional frequency scaling factors (can be NULL)
    int device_id,
    void *stream
);

// In-place residual addition:
// x[i] += residual[i]
int zeno_cuda_add_residual(
    float *x,
    const float *residual,
    size_t count,
    int device_id,
    void *stream
);

// In-place vector scaling:
// x[i] *= scale
int zeno_cuda_scale(
    float *x,
    float scale,
    size_t count,
    int device_id,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif // ZENO_CUDA_H
```

---

### 3.2 Kernel Preconditions, Memory Layouts, and Launch Bounds (R15-04)

All device buffers must be row-major and contiguous float32 without row padding.

#### 1. Fused RMSNorm (`zeno_cuda_rms_norm`)
- **Preconditions**:
  - `x != NULL`, `out != NULL`.
  - `num_rows >= 0`, `dim > 0`, `eps > 0.0`.
  - If `num_rows == 0`, the kernel immediately returns `ZENO_CUDA_SUCCESS` (no-op).
  - If `weight != NULL`, `weight` must point to at least `dim` contiguous floats.
  - If `weight == NULL`, unweighted normalization is performed. This covers both V-head normalization in SWA/Global layers and MoE router preprocessing.
  - `dim` supported range: $1 \le dim \le 8192$. Specifically verified on $dim \in \{256, 512, 2816\}$.
- **Launch Geometry**:
  - Grid: `dim3(num_rows)`. Each thread block normalizes exactly one row.
  - Block: `dim3(block_size)` where `block_size` is 256 (for $dim \le 512$) or 512 (for $dim = 2816$).
- **Algorithm**:
  - Grid-stride loop computes partial sum of squares in each thread.
  - Warp reduction via `__shfl_down_sync(0xffffffff, sum_sq, offset)`.
  - Shared memory inter-warp reduction across warps (`__shared__ float s_warp_sums[32]`).
  - Thread 0 computes `inv_rms = rsqrtf(sum_sq / dim + eps)` and broadcasts via shared memory.
  - Threads apply scaling and write output.

#### 2. Fused GeGLU (`zeno_cuda_geglu`)
- **Preconditions**:
  - `gate != NULL`, `up != NULL`, `out != NULL`.
  - `count >= 0`. If `count == 0`, returns `ZENO_CUDA_SUCCESS` (no-op).
  - In-place evaluation (`out == gate` or `out == up`) is supported. Partial overlapping of non-identical buffers is disallowed.
- **Launch Geometry**:
  - Grid: `dim3((count + 255) / 256)`.
  - Block: `dim3(256)`.
- **Algorithm**:
  - Evaluates $\text{gelu\_tanh}(g) = 0.5f \times g \times (1.0f + \tanh(0.7978845608028654f \times (g + 0.044715f \times g^3)))$.
  - Multiplies by $u$ and writes to `out`.

#### 3. NeoX-style RoPE (`zeno_cuda_rope`)
- **Preconditions**:
  - `vec != NULL`, `num_tokens >= 0`, `num_heads > 0`, `head_dim > 0`.
  - `head_dim` must be even ($head\_dim \pmod 2 == 0$). Returns `ZENO_CUDA_ERR_INVALID_PARAM` if odd. Specifically targeting $head\_dim \in \{256, 512\}$.
  - If `num_tokens == 0`, returns `ZENO_CUDA_SUCCESS` (no-op).
  - `pos_offset >= 0`, `theta > 0.0`.
  - `freq_factors`: `NULL` or points to at least `head_dim / 2` floats.
- **Memory Layout**:
  - Contiguous 3D tensor of shape `[num_tokens, num_heads, head_dim]`.
  - Head stride: `head_dim`. Token stride: `num_heads * head_dim`.
- **Launch Geometry**:
  - Grid: `dim3(num_heads, num_tokens)`.
  - Block: `dim3(head_dim / 2)` (128 threads for $head\_dim = 256$, 256 threads for $head\_dim = 512$).
- **Algorithm**:
  - Thread $i$ processes coordinate pair $(i, i + half)$.
  - Computes frequency, angle, cosine, sine, and applies in-place planar rotation.

#### 4. Residual Add & Scale (`zeno_cuda_add_residual`, `zeno_cuda_scale`)
- **Preconditions**:
  - Non-null pointers, `count >= 0`. Returns `ZENO_CUDA_SUCCESS` if `count == 0`.
- **Launch Geometry**:
  - Grid: `dim3((count + 255) / 256)`.
  - Block: `dim3(256)`.

---

### 3.3 Zig Wrapper Implementation (`src/cuda_mem.zig`, R15-01, R15-02)

```zig
const std = @import("std");
const cuda = @import("cuda.zig");

pub const DeviceSlice = struct {
    ptr: [*]f32,
    len: usize,
    device_id: i32,

    pub fn subslice(self: DeviceSlice, start: usize, length: usize) DeviceSlice {
        std.debug.assert(start + length <= self.len);
        return .{
            .ptr = self.ptr + start,
            .len = length,
            .device_id = self.device_id,
        };
    }
};

pub const CudaStream = struct {
    handle: ?*anyopaque,
    device_id: i32,

    pub fn init(device_id: i32) cuda.CudaError!CudaStream {
        if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
        var raw_stream: ?*anyopaque = null;
        const rc = cuda.c.zeno_cuda_stream_create(&raw_stream, device_id);
        if (rc != 0) return cuda.mapCudaErrorCode(rc);
        return .{
            .handle = raw_stream,
            .device_id = device_id,
        };
    }

    pub fn deinit(self: *CudaStream) void {
        if (self.handle) |h| {
            const rc = cuda.c.zeno_cuda_stream_destroy(h, self.device_id);
            if (rc != 0) {
                std.log.err("zeno_cuda_stream_destroy failed with rc={d}", .{rc});
                std.debug.assert(rc == 0);
            }
            self.handle = null;
        }
    }

    pub fn synchronize(self: *const CudaStream) cuda.CudaError!void {
        if (self.handle) |h| {
            const rc = cuda.c.zeno_cuda_stream_sync(h);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }
    }
};

pub fn DeviceBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        ptr: ?[*]T,
        len: usize,
        device_id: i32,

        pub fn init(device_id: i32, count: usize) cuda.CudaError!Self {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (count == 0) {
                return .{
                    .ptr = null,
                    .len = 0,
                    .device_id = device_id,
                };
            }

            var raw_ptr: ?*anyopaque = null;
            const bytes = count * @sizeOf(T);
            const rc = cuda.c.zeno_cuda_malloc(&raw_ptr, bytes, device_id);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);

            return .{
                .ptr = @ptrCast(@alignCast(raw_ptr.?)),
                .len = count,
                .device_id = device_id,
            };
        }

        pub fn move(self: *Self) Self {
            const moved = Self{
                .ptr = self.ptr,
                .len = self.len,
                .device_id = self.device_id,
            };
            self.ptr = null;
            self.len = 0;
            return moved;
        }

        pub fn deinit(self: *Self) void {
            if (self.ptr) |p| {
                const rc = cuda.c.zeno_cuda_free(p, self.device_id);
                if (rc != 0) {
                    std.log.err("zeno_cuda_free failed for device {d} with rc={d}", .{ self.device_id, rc });
                    std.debug.assert(rc == 0);
                }
                self.ptr = null;
                self.len = 0;
            }
        }

        pub fn copyFromHost(self: *Self, host_slice: []const T) cuda.CudaError!void {
            std.debug.assert(host_slice.len == self.len);
            if (self.len == 0) return;
            const bytes = self.len * @sizeOf(T);
            const rc = cuda.c.zeno_cuda_memcpy_h2d(self.ptr.?, host_slice.ptr, bytes, self.device_id, null);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        pub fn copyToHost(self: *const Self, host_slice: []T) cuda.CudaError!void {
            std.debug.assert(host_slice.len == self.len);
            if (self.len == 0) return;
            const bytes = self.len * @sizeOf(T);
            const rc = cuda.c.zeno_cuda_memcpy_d2h(host_slice.ptr, self.ptr.?, bytes, self.device_id, null);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        pub fn copyFromDevice(self: *Self, src: DeviceSlice) cuda.CudaError!void {
            if (self.device_id != src.device_id) return error.DeviceMismatch;
            std.debug.assert(src.len == self.len);
            if (self.len == 0) return;
            const bytes = self.len * @sizeOf(T);
            const rc = cuda.c.zeno_cuda_memcpy_d2d(self.ptr.?, src.ptr, bytes, self.device_id, null);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        pub fn copyFromHostAsync(self: *Self, host_slice: []const T, stream: *const CudaStream) cuda.CudaError!void {
            if (self.device_id != stream.device_id) return error.DeviceMismatch;
            std.debug.assert(host_slice.len == self.len);
            if (self.len == 0) return;
            const bytes = self.len * @sizeOf(T);
            const rc = cuda.c.zeno_cuda_memcpy_h2d(self.ptr.?, host_slice.ptr, bytes, self.device_id, stream.handle);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        pub fn copyToHostAsync(self: *const Self, host_slice: []T, stream: *const CudaStream) cuda.CudaError!void {
            if (self.device_id != stream.device_id) return error.DeviceMismatch;
            std.debug.assert(host_slice.len == self.len);
            if (self.len == 0) return;
            const bytes = self.len * @sizeOf(T);
            const rc = cuda.c.zeno_cuda_memcpy_d2h(host_slice.ptr, self.ptr.?, bytes, self.device_id, stream.handle);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        pub fn slice(self: *const Self) DeviceSlice {
            return .{
                .ptr = if (self.ptr) |p| @ptrCast(p) else undefined,
                .len = self.len,
                .device_id = self.device_id,
            };
        }
    };
}
```

---

## 4. Numerical Tolerance Policy (R15-05)

### 4.1 Reduction Re-ordering and Transcendental Divergence
1. **Parallel Sum of Squares**:
   - The CPU `ops.rmsNorm` computes $\sum x_i^2$ sequentially in 32-bit floating point.
   - The CUDA kernel accumulates sums within warps and reduces using tree-structured `__shfl_down_sync`. Floating-point addition is non-associative ($(a + b) + c \ne a + (b + c)$); across $d = 2816$ additions, the low mantissa bits will legitimately diverge.
2. **CUDA Transcendental Functions**:
   - NVIDIA `rsqrtf`, `tanhf`, `sinf`, and `cosf` implement hardware-accelerated polynomial approximations that differ by up to 1–2 ULPs from host glibc/musl implementations.
   - For outputs with large magnitudes, absolute divergence scales proportionally with magnitude.

### 4.2 Mixed Tolerance Formula
Numerical acceptance comparisons use a standard mixed absolute and relative tolerance:
$$|y_{\text{gpu}} - y_{\text{cpu}}| \le \text{atol} + \text{rtol} \times |y_{\text{cpu}}|$$

### 4.3 Provisional Thresholds on Deterministic Test Vectors
Evaluated over deterministic pseudo-random test vectors with values in $[-3.0, 3.0]$:

| Operation | Parameters / Shapes | $\text{atol}$ | $\text{rtol}$ | Rationale |
|---|---|---|---|---|
| **RMSNorm (Weighted)** | $N \in [1, 256], d = 2816$, learned $w \in [0.1, 2.0]$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | Tree reduction over 2816 elements + `rsqrtf`. |
| **RMSNorm (Unweighted)** | $N \in [1, 256], d \in \{256, 512, 2816\}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | Covers router and V-head normalization. |
| **GeGLU** | $N \le 4096$, inputs in $[-3.0, 3.0]$ | $1 \times 10^{-5}$ | $1 \times 10^{-4}$ | Elementwise `tanhf` approximation error. |
| **NeoX RoPE** | $N \le 256, H \in \{8, 16\}, d \in \{256, 512\}, \theta \in \{10^4, 10^6\}$ | $1 \times 10^{-5}$ | $1 \times 10^{-4}$ | Planar rotation with hardware `cosf`/`sinf`. |
| **Residual Add / Scale** | $N \le 4096$, scale factor $1.5$ | $1 \times 10^{-6}$ | $1 \times 10^{-6}$ | Direct IEEE-754 elementwise operations. |

> **Verification Rule**: These thresholds are provisional. During Stage 2 implementation, maximum observed absolute and relative errors across 10,000 deterministic sample points will be measured and recorded in the Stage 2 verification report before finalizing the exit gate.

---

## 5. Verification Strategy & Acceptance Gates

Stage 2 acceptance requires satisfying four strict gates:

1. **Gate 1: Default Pure Zig Build Isolation**:
   - `zig build` and `zig build test` compile cleanly without CUDA toolchains.
   - All CPU unit tests pass with zero leaks.

2. **Gate 2: Portable CUDA Test Suite (R14-02, R15-06)**:
   - `zig build test -Dcuda=true` compiles `ops.cu` and links `libcudart`.
   - On toolkit-only build environments without GPUs, tests catching `error.NoDevice` skip cleanly via `error.SkipZigTest`.
   - Genuine driver initialization failures (`error.DriverError`) are not skipped and fail the test run.
   - On GPU hardware, all tests execute and validate numerical accuracy.

3. **Gate 3: Empirical Numerical Parity ($\text{atol} / \text{rtol}$ Verified)**:
   - Verify `cuda_rms_norm` and `cuda_unweighted_rms_norm` against `ops.rmsNorm` and `ops.unweightedRmsNorm`.
   - Verify `cuda_geglu` against `ops.geglu`.
   - Verify `cuda_rope` against `infer.applyRoPE`.
   - Verify `cuda_add_residual` and `cuda_scale` against CPU reference arithmetic.
   - Record maximum observed delta in verification documentation.

4. **Gate 4: Memory Safety & Zero Leaks**:
   - Verify `DeviceBuffer` allocations, synchronous copies, asynchronous copies, and deallocations leave zero device memory leaks.
   - Repeated `deinit()` calls do not fault or corrupt runtime state.
   - Run full HTTP decision server test suite (`python3 tools/test_server.py`) to confirm zero CPU regression.

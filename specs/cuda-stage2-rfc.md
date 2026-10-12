# RFC: Zeno CUDA Backend — Stage 2: Device Memory Management & Fused Elementwise Kernels

- **Status**: Proposed / Under Review (Round 19 Revision)
- **Author**: Antigravity (Pair Programming with User)
- **Target Release**: Zeno Phase 5 (CUDA & Multi-GPU Execution)
- **Governing Specs**: [`AGENTS.md`](../AGENTS.md), [`specs/cuda-roadmap.md`](cuda-roadmap.md), [`specs/transformer.md`](transformer.md), [`specs/stage1-verification.md`](stage1-verification.md), [`specs/UNRESOLVED.md`](UNRESOLVED.md)
- **Target Hardware**: NVIDIA GeForce RTX 3060 12GB (Ampere `sm_86`, PCIe 4.0; dual-GPU setup)
- **Toolchain**: Zig 0.17.0 + NVIDIA `nvcc` (CUDA 13.3.73, Driver 595.99)

---

## 1. Executive Summary

This RFC specifies **Stage 2** of the Zeno CUDA backend. Following the Stage 1 implementation and physical hardware verification on local RTX 3060 dual-GPU host (documented in [`specs/stage1-verification.md`](stage1-verification.md), commits `457af85` and `ee31f5c`, R16-06, R17-04), Stage 2 establishes:

1. **Deterministic Device Memory Management in Zig** ([`src/cuda_mem.zig`](../src/cuda_mem.zig)):
   - A single-owner typed buffer abstraction `DeviceBuffer(T)` constrained to non-zero POD numeric types (`f32, f16, i32, u32, u8`), with checked allocation byte bounds (16 GiB guard cap), manual move semantics via `move()`, nullable pointer discipline, safe idempotent deinitialization, and zero host heap allocations (R15-01, R16-02, R18-02).
   - Generic, non-owning view abstraction `DeviceSlice(T)` with safe nullable pointer representation (`?[*]T`), explicit empty slice constructor, and checked arithmetic in `subslice` returning `error.InvalidParam` on overflow or out-of-bounds while preserving empty representations without pointer arithmetic (R16-01, R18-01, R19-01).
   - Explicitly synchronous host-device transfers (`copyFromHost`, `copyToHost`, `copyFromDevice`) with release-mode runtime length validation returning `error.InvalidParam` on mismatch across all build modes, eliminating pinned-memory and stream-lifetime hazards from Stage 2 (R16-03, R18-01).
   - Execution stream wrapper `CudaStream` with explicit single-owner move semantics (`move()`), release-mode handle retention on failed destruction, and strict kernel scheduling scoping, while deferring asynchronous host streaming and pinned host memory (`cudaHostAlloc`) to Stage 5 pipelining (R16-03, R18-03).
   - Complete public C ABI boundary and error mappings in `src/cuda.zig` and `src/cuda/zeno_cuda.h` (R15-03, R16-01, R17-01).

2. **Native Ampere Elementwise Kernels in CUDA C++** ([`src/cuda/ops.cu`](../src/cuda/ops.cu)):
   - **Fused RMSNorm (`cuda_rms_norm`)**: Fast warp-level reduction (`__shfl_down_sync`) supporting learned scale weights and unweighted normalization (for both V-head norm and MoE router preprocessing), covering $d = 2816$ (residual streams) and $d \in \{256, 512\}$ (per-head Q/K/V norms).
   - **Fused GeGLU (`cuda_geglu`)**: Elementwise $\text{GELU}_{\text{tanh}}(a) \times b$ activation kernel for Gemma 4 shared dense MLP and MoE expert branches.
   - **Rotary Position Embedding (`cuda_rope`)**: High-throughput NeoX-style half-split RoPE over query and key head vectors, supporting dual frequency bases ($\theta_{\text{swa}} = 10,000.0, d = 256$ and $\theta_{\text{global}} = 1,000,000.0, d = 512$) with optional frequency scaling factors.
   - **Fused Residual Add & Scaling (`cuda_add_residual`, `cuda_scale`)**: In-place elementwise operations for decoder block residual junctions and layer output scaling.
   - Explicit kernel preconditions, launch bounds, integer types, aliasing rules, and a minimum 32-thread block ensuring warp shuffle mask validity across all supported dimensions (R15-04, R16-05, R17-02).

3. **Mechanical Parity & Acceptance Verification**:
   - A justified mixed absolute/relative numerical tolerance policy ($|y_{\text{gpu}} - y_{\text{cpu}}| \le \text{atol} + \text{rtol} \times |y_{\text{cpu}}|$) with proposed acceptance thresholds across all production dimensions, to be empirically measured against deterministic pseudo-random test vectors during Stage 2 implementation (R15-05, R16-04, R17-03).
   - Strict unit test portability skipping only on unambiguous `error.NoDevice`, preserving genuine driver initialization failures as test errors (R15-06).
   - Zero-dependency CPU build invariant preserved when `-Dcuda=false`.

---

## 2. Architecture & Memory Management

### 2.1 Memory Ownership Model ([`AGENTS.md`](../AGENTS.md) §6, R15-01, R16-02)

In Zeno, device memory buffers must have unambiguous lifetime boundaries. Accidental temporary device allocations or implicit global device contexts are strictly avoided.

```text
┌────────────────────────────────────────────────────────┐
│                      Host (Zig)                        │
│   src/cuda_mem.zig: DeviceBuffer(T), DeviceSlice(T)   │
│   - Explicit device_id assignment                      │
│   - Manual single-owner discipline via move()          │
│   - Nullable pointer: ptr: ?[*]T, len: usize           │
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

#### Ownership Discipline:
- **Manually Enforced Single Ownership (R16-02, R18-03)**: Like standard Zig allocators and file handles, Zig does not have linear/affine types or compiler-enforced move semantics. Both `DeviceBuffer(T)` and `CudaStream` are explicitly single-owner structs governed by programmer discipline. Duplicating owning handles (e.g. assigning `var b2 = b1` and calling `deinit` on both) is a contract violation.
- **Explicit Move Semantics (R16-02, R18-03)**: To transfer ownership across scopes or structs, callers must invoke `var b2 = b1.move()` (for buffers) or `var s2 = s1.move()` (for streams), which transfers the active handle and zeroes/nulls the source (`b1.ptr = null, b1.len = 0` or `s1.handle = null`).
- **Non-Owning Views & Safe Empty Slices (R16-01, R18-01, R19-01)**: Slices of a buffer are represented by `DeviceSlice(T)`. `DeviceSlice(T)` holds a nullable non-owning raw device pointer (`ptr: ?[*]T`) and length (`len: usize`). When representing an empty view (`len == 0`), `ptr` is safely represented as `null` (via `DeviceSlice(T).empty(device_id)`), completely eliminating `undefined` pointers. Sub-slicing via `subslice(start, length)` performs checked arithmetic; when `length == 0` it returns an empty slice without executing pointer arithmetic on null bases. When `length > 0`, it validates that `self.ptr` is non-null before performing offset arithmetic.
- **Idempotent Deinitialization**: `deinit(self: *Self)` checks `if (self.ptr) |p|` (or `if (self.handle) |h|`). If destruction succeeds, the pointer/handle is immediately set to `null` (and `len = 0` for buffers). Repeated calls to `deinit()` are guaranteed safe no-ops.
- **Failed Deallocation & Destruction Behavior (R16-02, R18-03)**: If `zeno_cuda_free` or `zeno_cuda_stream_destroy` fails, `deinit()` logs the error via `std.log.err` and asserts in debug mode. In release builds, `self.ptr` and `self.handle` are **retained** rather than zeroed, ensuring the failure does not silently drop the handle or leak the runtime resource from tracking.
- **Zero-Length Allocations**: If `count == 0`, `DeviceBuffer(T).init(device_id, 0)` returns an empty buffer with `ptr = null` and `len = 0` without calling CUDA runtime allocation.
- **POD Type Constraints & Allocation Bounds (R18-02)**: `DeviceBuffer(T)` is constrained at compile time to POD numeric types (`f32, f16, i32, u32, u8`) with `@sizeOf(T) > 0`. Byte calculations use checked arithmetic (`std.math.mul(usize, count, @sizeOf(T)) catch return error.InvalidParam`). Maximum allocation size is capped at 16 GiB (`MAX_ALLOC_BYTES = 16 * 1024 * 1024 * 1024`) to prevent integer overflow and unserviceable device requests.

---

### 2.2 Host-Device Transfer Contract (R15-02, R16-03, R18-01)

To ensure memory safety and avoid complex pinned-memory hazards in Stage 2:

1. **Synchronous Transfers by Default**:
   - `copyFromHost(self: *Self, host_slice: []const T) CudaError!void`
   - `copyToHost(self: *const Self, host_slice: []T) CudaError!void`
   - `copyFromDevice(self: *Self, src: DeviceSlice(T)) CudaError!void`
   - **Contract**: Transfers are executed synchronously via `cudaMemcpy(..., cudaMemcpyHostToDevice / cudaMemcpyDeviceToHost / cudaMemcpyDeviceToDevice)`. The host thread blocks until transfer completion. Host buffers require no lifetime guarantees beyond function return, and ordinary pageable Zig slices are safely supported.
   - **Runtime Length Validation (R18-01)**: Every copy method validates `host_slice.len == self.len` (or `src.len == self.len`) at runtime in all build modes (`ReleaseFast`, `ReleaseSafe`, `ReleaseSmall`, `Debug`), returning `error.InvalidParam` on mismatch before issuing any copy. Never rely on debug assertions for API bounds checking.
   - **Checked Range Arithmetic (R18-01)**: `DeviceSlice(T).subslice(start, length)` performs checked arithmetic (`std.math.add(usize, start, length) catch return error.InvalidParam; if (end > self.len) return error.InvalidParam;`) and returns `CudaError!DeviceSlice(T)`.
   - **Device Matching**: `copyFromDevice` verifies `self.device_id == src.device_id`. Cross-device transfers return `error.DeviceMismatch`.

2. **Stream Execution vs. Asynchronous Transfers (R16-03)**:
   - Stage 2 provides `CudaStream` strictly for **kernel execution ordering**: kernels accept an optional `stream: ?*const CudaStream` (passed as `cudaStream_t` or `NULL` for default stream).
   - Asynchronous memory streaming and pinned host memory allocation (`PinnedHostBuffer` via `cudaHostAlloc` / `cudaHostRegister`) are explicitly deferred to Stage 5, where multi-stream pipelining and P2P transfers are designed. Stage 2 avoids exposing incomplete async transfer primitives that lack pinned-memory guarantees.

---

## 3. Detailed Technical Specification

### 3.1 C ABI Header Additions (`src/cuda/zeno_cuda.h`, R15-03, R16-01, R17-01)

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

// --- Extended Error Codes (R15-03, R16-01) ---
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
int zeno_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes, int device_id);
int zeno_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, int device_id);
int zeno_cuda_memcpy_d2d(void *dst, const void *src, size_t bytes, int device_id);
int zeno_cuda_memset(void *dst, int value, size_t bytes, int device_id);

// --- Stream Management ABI ---

int zeno_cuda_stream_create(void **out_stream, int device_id);
int zeno_cuda_stream_destroy(void *stream, int device_id);
int zeno_cuda_stream_sync(void *stream, int device_id);

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
    void *stream              // Optional cudaStream_t (can be NULL)
);

// Computes GeGLU activation branch:
// out[i] = gelu_tanh(gate[i]) * up[i]
int zeno_cuda_geglu(
    const float *gate,
    const float *up,
    float *out,
    size_t count,
    int device_id,
    void *stream              // Optional cudaStream_t (can be NULL)
);

// Applies NeoX-style (half-split) Rotary Position Embeddings in-place:
// For each token t in [0, num_tokens) and head h in [0, num_heads):
// Rotates coordinate pairs (vec[i], vec[i + head_dim / 2]) using base theta.
int zeno_cuda_rope(
    float *vec,
    int num_tokens,
    int num_heads,
    int head_dim,
    int pos_offset,
    float theta,
    const float *freq_factors, // Optional frequency scaling factors (can be NULL)
    int device_id,
    void *stream              // Optional cudaStream_t (can be NULL)
);

// In-place residual addition:
// x[i] += residual[i]
int zeno_cuda_add_residual(
    float *x,
    const float *residual,
    size_t count,
    int device_id,
    void *stream              // Optional cudaStream_t (can be NULL)
);

// In-place vector scaling:
// x[i] *= scale
int zeno_cuda_scale(
    float *x,
    float scale,
    size_t count,
    int device_id,
    void *stream              // Optional cudaStream_t (can be NULL)
);

#ifdef __cplusplus
}
#endif

#endif // ZENO_CUDA_H
```

---

### 3.2 Kernel Preconditions, Launch Bounds, and Aliasing Rules (R15-04, R16-05, R17-02)

All device buffers must be row-major and contiguous float32 without padding.

#### 1. Fused RMSNorm (`zeno_cuda_rms_norm`)
- **Preconditions**:
  - `x != NULL`, `out != NULL`.
  - `0 <= num_rows <= 65535`, `1 <= dim <= 8192`, `eps > 0.0`. Returns `ZENO_CUDA_ERR_INVALID_PARAM` if out of bounds.
  - If `num_rows == 0`, immediately returns `ZENO_CUDA_SUCCESS` (no-op).
  - If `weight != NULL`, `weight` must point to at least `dim` contiguous floats.
  - If `weight == NULL`, unweighted normalization is performed. This covers both V-head normalization in SWA/Global layers and MoE router preprocessing.
- **Aliasing**: In-place normalization (`out == x`) is explicitly supported. Partial overlapping of non-identical buffers returns `ZENO_CUDA_ERR_INVALID_PARAM`.
- **Launch Geometry & Warp Participation (R17-02)**:
  - Grid: `dim3(num_rows)`. Each block normalizes exactly one row.
  - Block: `dim3(block_size)` where `block_size = max(32, min(512, next_power_of_two(dim)))`.
  - For $dim < 32$, a full 32-thread warp is launched; threads with $tid \ge dim$ contribute $0.0f$ to the reduction sum of squares. This guarantees that all 32 lanes participate in `__shfl_down_sync(0xffffffff, ...)` without undefined shuffle behavior.
  - For $dim \le 512$: `block_size` matches the next power of two ($256$ for $d=256$, $512$ for $d=512$).
  - For $dim > 512$ (e.g. $dim = 2816$): `block_size = 512` threads using a grid-stride accumulation loop.
- **Algorithm**:
  - Grid-stride loop computes partial sum of squares: `float sum_sq = 0.0f; for (int i = tid; i < dim; i += blockDim.x) sum_sq += x[row_offset + i] * x[row_offset + i];`
  - Intra-warp reduction with `__shfl_down_sync(0xffffffff, sum_sq, offset)`.
  - Inter-warp reduction via `__shared__ float s_warp_sums[32]`.
  - Thread 0 computes `inv_rms = rsqrtf(sum_sq / dim + eps)` and broadcasts via shared memory.
  - Threads apply scaling and write to `out`.

#### 2. Fused GeGLU (`zeno_cuda_geglu`)
- **Preconditions**:
  - `gate != NULL`, `up != NULL`, `out != NULL`.
  - `0 <= count <= ((size_t)1 << 31) - 1`.
  - If `count == 0`, immediately returns `ZENO_CUDA_SUCCESS` (no-op).
- **Aliasing**: In-place evaluation (`out == gate` or `out == up`) is supported. Partial overlapping of non-identical buffers returns `ZENO_CUDA_ERR_INVALID_PARAM`.
- **Launch Geometry**:
  - Block: `dim3(256)`.
  - Grid: `dim3((int)((count + 255) / 256))`.
- **Algorithm**:
  - Evaluates $\text{gelu\_tanh}(g) = 0.5f \times g \times (1.0f + \tanh(0.7978845608028654f \times (g + 0.044715f \times g^3)))$.
  - Multiplies by $u$ and writes to `out`.

#### 3. NeoX-style RoPE (`zeno_cuda_rope`)
- **Preconditions**:
  - `vec != NULL`, `0 <= num_tokens <= 65535`, `1 <= num_heads <= 64`.
  - `1 <= head_dim <= 1024`, and `head_dim % 2 == 0`. Returns `ZENO_CUDA_ERR_INVALID_PARAM` if odd or out of bounds. Specifically validated on $head\_dim \in \{256, 512\}$.
  - If `num_tokens == 0`, immediately returns `ZENO_CUDA_SUCCESS` (no-op).
  - `pos_offset >= 0`, `theta > 0.0`.
  - `freq_factors`: `NULL` or points to at least `head_dim / 2` floats.
- **Aliasing**: In-place on `vec` by definition.
- **Memory Layout**: Contiguous 3D tensor of shape `[num_tokens, num_heads, head_dim]`.
- **Launch Geometry**:
  - Grid: `dim3(num_heads, num_tokens)`.
  - Block: `dim3(head_dim / 2)` (128 threads for $d_{\text{head}}=256$, 256 threads for $d_{\text{head}}=512$).
- **Algorithm**:
  - Thread $i$ computes coordinate pair $(i, i + half)$.
  - Computes $\text{freq} = (\theta^{-2i / head\_dim}) / (\text{factors} ? \text{factors}[i] : 1.0f)$, $\alpha = (\text{blockIdx.y} + pos\_offset) \times \text{freq}$, applies planar rotation.

#### 4. Residual Add & Scale (`zeno_cuda_add_residual`, `zeno_cuda_scale`)
- **Preconditions**:
  - Non-null pointers, `0 <= count <= ((size_t)1 << 31) - 1`. If `count == 0`, returns `ZENO_CUDA_SUCCESS`.
- **Aliasing**:
  - `add_residual`: In-place on `x` (`x += residual`). If `x == residual`, returns `ZENO_CUDA_ERR_INVALID_PARAM` (residual source must be distinct).
  - `scale`: In-place on `x` (`x *= scale`).
- **Launch Geometry**:
  - Block: `dim3(256)`.
  - Grid: `dim3((int)((count + 255) / 256))`.

---

### 3.3 Zig Wrapper Implementation (`src/cuda_mem.zig`, R16-01, R16-02, R16-03, R17-01, R18-01, R18-02, R18-03, R19-01)

The implementation in `src/cuda_mem.zig` directly binds to the public `cuda.c` declarations:

```zig
const std = @import("std");
const cuda = @import("cuda.zig");

pub fn DeviceSlice(comptime T: type) type {
    return struct {
        const Self = @This();

        ptr: ?[*]T,
        len: usize,
        device_id: i32,

        /// Returns an empty non-owning slice for a given device with a safe null pointer (R19-01).
        pub fn empty(device_id: i32) Self {
            return .{
                .ptr = null,
                .len = 0,
                .device_id = device_id,
            };
        }

        /// Returns a subslice [start, start + length).
        /// Performs checked addition and bounds checking in all build modes (R18-01).
        /// Preserves the safe null empty representation without pointer arithmetic (R19-01).
        pub fn subslice(self: Self, start: usize, length: usize) cuda.CudaError!Self {
            const end = std.math.add(usize, start, length) catch return error.InvalidParam;
            if (end > self.len) return error.InvalidParam;
            if (length == 0) {
                return empty(self.device_id);
            }
            const base = self.ptr orelse return error.InvalidParam;
            return .{
                .ptr = base + start,
                .len = length,
                .device_id = self.device_id,
            };
        }
    };
}

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

    /// Explicitly transfers ownership to the caller, leaving source stream empty (R18-03).
    pub fn move(self: *CudaStream) CudaStream {
        const moved = CudaStream{
            .handle = self.handle,
            .device_id = self.device_id,
        };
        self.handle = null;
        return moved;
    }

    /// Idempotent stream destruction. If stream destruction fails, logs error
    /// and retains handle in release mode so the resource failure is not masked (R18-03).
    pub fn deinit(self: *CudaStream) void {
        if (self.handle) |h| {
            const rc = cuda.c.zeno_cuda_stream_destroy(h, self.device_id);
            if (rc != 0) {
                std.log.err("zeno_cuda_stream_destroy failed for device {d} with rc={d}", .{ self.device_id, rc });
                std.debug.assert(rc == 0);
                // Retain handle on failure in release mode matching DeviceBuffer
                return;
            }
            self.handle = null;
        }
    }

    pub fn synchronize(self: *const CudaStream) cuda.CudaError!void {
        if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
        const rc = cuda.c.zeno_cuda_stream_sync(self.handle, self.device_id);
        if (rc != 0) return cuda.mapCudaErrorCode(rc);
    }
};

pub fn DeviceBuffer(comptime T: type) type {
    comptime {
        // Enforce POD numeric types suitable for tensor and parameter buffers (R18-02)
        const valid = switch (T) {
            f32, f16, i32, u32, u8 => true,
            else => false,
        };
        if (!valid or @sizeOf(T) == 0) {
            @compileError("DeviceBuffer only supports non-zero-sized POD numeric types (f32, f16, i32, u32, u8)");
        }
    }

    return struct {
        const Self = @This();

        /// Guard cap preventing integer overflow and unserviceable device requests (16 GiB)
        pub const MAX_ALLOC_BYTES: usize = 16 * 1024 * 1024 * 1024;

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

            // Checked multiplication prevents overflow before issuing CUDA allocation (R18-02)
            const bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.InvalidParam;
            if (bytes > MAX_ALLOC_BYTES) return error.InvalidParam;

            var raw_ptr: ?*anyopaque = null;
            const rc = cuda.c.zeno_cuda_malloc(&raw_ptr, bytes, device_id);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);

            return .{
                .ptr = @ptrCast(@alignCast(raw_ptr.?)),
                .len = count,
                .device_id = device_id,
            };
        }

        /// Explicitly transfers ownership to the caller, leaving source buffer empty (R16-02).
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

        /// Idempotent device buffer deallocation. If free fails, logs error
        /// and retains pointer in release mode to preserve trackable state (R16-02).
        pub fn deinit(self: *Self) void {
            if (self.ptr) |p| {
                const rc = cuda.c.zeno_cuda_free(p, self.device_id);
                if (rc != 0) {
                    std.log.err("zeno_cuda_free failed for device {d} with rc={d}", .{ self.device_id, rc });
                    std.debug.assert(rc == 0);
                    // Retain pointer in release mode so failure is not masked
                    return;
                }
                self.ptr = null;
                self.len = 0;
            }
        }

        /// Synchronous copy from host slice to device buffer.
        /// Validates slice lengths at runtime in all build modes (R18-01).
        pub fn copyFromHost(self: *Self, host_slice: []const T) cuda.CudaError!void {
            if (host_slice.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const dst = self.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memcpy_h2d(dst, host_slice.ptr, bytes, self.device_id);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Synchronous copy from device buffer to host slice.
        /// Validates slice lengths at runtime in all build modes (R18-01).
        pub fn copyToHost(self: *const Self, host_slice: []T) cuda.CudaError!void {
            if (host_slice.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const src = self.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memcpy_d2h(host_slice.ptr, src, bytes, self.device_id);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Synchronous copy from device slice to device buffer.
        /// Validates device ID matching and slice lengths at runtime in all build modes (R18-01).
        pub fn copyFromDevice(self: *Self, src: DeviceSlice(T)) cuda.CudaError!void {
            if (self.device_id != src.device_id) return error.DeviceMismatch;
            if (src.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const dst = self.ptr orelse return error.InvalidParam;
            const src_ptr = src.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memcpy_d2d(dst, src_ptr, bytes, self.device_id);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Returns a non-owning device slice. If the buffer is empty, returns an
        /// explicit safe empty slice with a null pointer rather than undefined (R19-01).
        pub fn slice(self: *const Self) DeviceSlice(T) {
            return .{
                .ptr = self.ptr,
                .len = self.len,
                .device_id = self.device_id,
            };
        }
    };
}
```

---

## 4. Numerical Tolerance Policy (R15-05, R16-04, R17-03)

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

### 4.3 Proposed Acceptance Thresholds (Pending Verification Measurement, R16-04, R17-03)
To be evaluated over deterministic pseudo-random test vectors generated via `std.Random.DefaultPrng.init(0x537461676532)` (ASCII `"Stage2"`) with values uniformly distributed in $[-3.0, 3.0]$:

| Operation | Parameters / Shapes | Proposed $\text{atol}$ | Proposed $\text{rtol}$ | Rationale |
|---|---|---|---|---|
| **RMSNorm (Weighted)** | $N \in [1, 256], d = 2816$, learned $w \in [0.1, 2.0]$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | Tree reduction over 2816 elements + `rsqrtf`. |
| **RMSNorm (Weighted Head)** | $N \in [1, 256], d \in \{256, 512\}$, learned $w \in [0.1, 2.0]$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | Learned Q/K per-head norm in SWA and Global layers. |
| **RMSNorm (Unweighted)** | $N \in [1, 256], d \in \{256, 512, 2816\}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | Covers router and V-head normalization. |
| **GeGLU** | $N \le 4096$, inputs in $[-3.0, 3.0]$ | $1 \times 10^{-5}$ | $1 \times 10^{-4}$ | Elementwise `tanhf` polynomial approximation error. |
| **NeoX RoPE** | $N \le 256, H \in \{8, 16\}, d \in \{256, 512\}, \theta \in \{10^4, 10^6\}$ | $1 \times 10^{-5}$ | $1 \times 10^{-4}$ | Planar rotation with hardware `cosf`/`sinf`. |
| **Residual Add / Scale** | $N \le 4096$, scale factor $1.5$ | $1 \times 10^{-6}$ | $1 \times 10^{-6}$ | Direct IEEE-754 elementwise operations. |

> **Verification Rule (R16-04, R17-03)**: These thresholds are proposed targets. During Stage 2 implementation, maximum observed absolute and relative errors across 10,000 deterministic sample points generated from seed `0x537461676532` will be measured and recorded in the Stage 2 verification report before finalizing the exit gate.

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
   - Verify `cuda_rms_norm` and `cuda_unweighted_rms_norm` against `ops.rmsNorm` and `ops.unweightedRmsNorm` across $d \in \{256, 512, 2816\}$.
   - Verify `cuda_geglu` against `ops.geglu`.
   - Verify `cuda_rope` against `infer.applyRoPE` across SWA and Global dimensions.
   - Verify `cuda_add_residual` and `cuda_scale` against CPU reference arithmetic.
   - Record maximum observed delta in verification documentation.

4. **Gate 4: Memory Safety & Zero Leaks**:
   - Verify `DeviceBuffer` allocations, synchronous copies, and deallocations leave zero device memory leaks.
   - Repeated `deinit()` calls do not fault or corrupt runtime state.
   - Run full HTTP decision server test suite (`python3 tools/test_server.py`) to confirm zero CPU regression.

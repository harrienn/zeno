# RFC: Zeno CUDA Backend — Stage 2: Device Memory Management & Fused Elementwise Kernels

- **Status**: Proposed / Under Review
- **Author**: Antigravity (Pair Programming with User)
- **Target Release**: Zeno Phase 5 (CUDA & Multi-GPU Execution)
- **Governing Specs**: [`AGENTS.md`](../AGENTS.md), [`specs/cuda-roadmap.md`](cuda-roadmap.md), [`specs/transformer.md`](transformer.md), [`specs/UNRESOLVED.md`](UNRESOLVED.md)
- **Target Hardware**: NVIDIA GeForce RTX 3060 12GB (Ampere `sm_86`, PCIe 4.0; dual-GPU setup)
- **Toolchain**: Zig 0.17.0 + NVIDIA `nvcc` (CUDA 13.3.73, Driver 595.99)

---

## 1. Executive Summary

This RFC specifies **Stage 2** of the Zeno CUDA backend. Following the successful completion of Stage 1 (build pipeline, C ABI boundary, and physical device discovery), Stage 2 establishes:

1. **Deterministic Device Memory Ownership in Zig** ([`src/cuda_mem.zig`](../src/cuda_mem.zig)):
   - RAII typed buffer abstraction `DeviceBuffer(T)` with explicit device assignment, bounds checking, and zero dynamic heap dependencies on the host.
   - Non-blocking asynchronous copy primitives (`copyFromHostAsync`, `copyToHostAsync`, `copyFromDeviceAsync`) and explicit stream synchronization (`CudaStream`).
   - Strict conformity with [`AGENTS.md`](../AGENTS.md) §6 (explicit allocator, owner, lifetime, and release).

2. **Native Ampere Elementwise Kernels in CUDA C++** ([`src/cuda/ops.cu`](../src/cuda/ops.cu)):
   - **Fused RMSNorm (`cuda_rms_norm`)**: Fast warp-level reduction (`__shfl_down_sync`) supporting both learned scale weights and unweighted normalization, configured for $d = 2816$ (residual streams) and $d \in \{256, 512\}$ (per-head Q/K/V norms).
   - **Fused GeGLU (`cuda_geglu`)**: Elementwise $\text{GELU}_{\text{tanh}}(a) \times b$ activation kernel for Gemma 4 shared dense MLP and MoE expert branches.
   - **Rotary Position Embedding (`cuda_rope`)**: High-throughput NeoX-style half-split RoPE over query and key head vectors, supporting dual frequency bases ($\theta_{\text{swa}} = 10,000.0, d = 256$ and $\theta_{\text{global}} = 1,000,000.0, d = 512$) with optional frequency scaling factors.
   - **Fused Residual Add & Scaling (`cuda_add_residual`, `cuda_scale`)**: In-place elementwise operations for decoder block residual junctions and layer output scaling.

3. **Mechanical Parity & Acceptance Verification**:
   - Zero tolerance inflation: all GPU kernel outputs must match CPU reference implementations in [`src/ops.zig`](../src/ops.zig) and [`src/infer.zig`](../src/infer.zig) within $\le 10^{-5}$ absolute numerical difference.
   - Zero-dependency CPU build invariant preserved when `-Dcuda=false`.
   - Portable test execution skipping cleanly (`error.SkipZigTest`) on toolkit-only build environments without GPUs.

---

## 2. Architecture & Memory Management

### 2.1 Memory Ownership Model ([`AGENTS.md`](../AGENTS.md) §6)

In Zeno, device memory buffers must have unambiguous lifetime boundaries. Accidental temporary device allocations or implicit global device contexts are strictly avoided.

```text
┌────────────────────────────────────────────────────────┐
│                      Host (Zig)                        │
│   src/cuda_mem.zig: DeviceBuffer(T), CudaStream       │
│   - Explicit device_id assignment                      │
│   - Value-type handle with RAII deinit()               │
│   - Non-copyable semantics                             │
└───────────────────────────────┬────────────────────────┘
                                │ C ABI: extern "C"
┌───────────────────────────────▼────────────────────────┐
│                   C ABI Memory Layer                   │
│   src/cuda/zeno_cuda.h: zeno_cuda_malloc / free / copy │
└───────────────────────────────┬────────────────────────┘
                                │ CUDA Driver / Runtime
┌───────────────────────────────▼────────────────────────┐
│                   GPU VRAM (Device)                    │
│   cudaSetDevice(id) -> cudaMalloc / cudaFreeAsync      │
│   Linear device pointers: [*]T (sm_86 aligned)         │
└────────────────────────────────────────────────────────┘
```

#### Ownership Questions Answered:
- **Who allocates it?** The host engine creates `DeviceBuffer(T).init(device_id, count)`.
- **Who owns it?** The struct holding the `DeviceBuffer(T)` instance.
- **How long does it live?** Bound to the owning struct's lifecycle (e.g. `InferenceWorkspaceCuda` for scratch activations, `ModelCuda` for long-lived weights).
- **Who releases or resets it?** Explicit `buf.deinit()` calling `zeno_cuda_free`. The buffer pointer is nulled out immediately to prevent use-after-free.
- **Can it move between CPU and GPU?** Explicit transfers only via `copyFromHost` / `copyToHost`.
- **Can another component retain a reference to it?** Only via non-owning `DeviceSlice(T)` handles that cannot perform deallocation.

---

## 3. Detailed Technical Specification

### 3.1 C ABI Header Additions (`src/cuda/zeno_cuda.h`)

Extends the Stage 1 C ABI with memory management, streams, and elementwise kernel entry points:

```c
#ifndef ZENO_CUDA_H
#define ZENO_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ... (Stage 1 declarations: ZenoCudaDeviceInfo, ZenoCudaTopology, device query functions)

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
    const float *weight,      // NULL for unweighted RMSNorm (router preprocessing)
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

### 3.2 CUDA Kernel Implementations (`src/cuda/ops.cu`)

All kernels are compiled by `nvcc -arch=sm_86 -O3 -Xcompiler -fPIC`.

#### 1. Fused RMSNorm (`zeno_cuda_rms_norm`)
- **Block Layout**: 1 row per threadblock. Block size = 256 threads (for $d \in \{256, 512\}$) or 512 threads (for $d = 2816$).
- **Algorithm**:
  1. Each thread accumulates partial sum of squares: `float sum_sq = 0.0f; for (int i = tid; i < dim; i += blockDim.x) sum_sq += x[row_offset + i] * x[row_offset + i];`
  2. Intra-warp reduction using `__shfl_down_sync(0xffffffff, sum_sq, offset)`.
  3. Inter-warp reduction using `__shared__ float s_warp_sums[32]`.
  4. Thread 0 computes `inv_rms = rsqrtf(sum_sq / dim + eps)` and stores in shared memory.
  5. All threads read `inv_rms` and apply scaling:
     `out[row_offset + i] = x[row_offset + i] * inv_rms * (weight ? weight[i] : 1.0f);`
- **Numerical Behavior**: Uses `rsqrtf` with float32 accumulation to match CPU `1.0 / @sqrt(mean_sq + eps)`.

#### 2. Fused GeGLU (`zeno_cuda_geglu`)
- **Grid Layout**: 1D grid, block size = 256 threads. Grid size = $\lceil \text{count} / 256 \rceil$.
- **Algorithm**:
  $$\text{gelu\_tanh}(x) = 0.5f \times x \times (1.0f + \tanh(0.7978845608028654f \times (x + 0.044715f \times x^3)))$$
  $$\text{out}_i = \text{gelu\_tanh}(\text{gate}_i) \times \text{up}_i$$
- Fully coalesced 32-bit reads and writes.

#### 3. NeoX-style RoPE (`zeno_cuda_rope`)
- **Grid Layout**:
  - `blockDim.x = head_dim / 2` (128 for $d_{\text{head}}=256$, 256 for $d_{\text{head}}=512$).
  - `gridDim.x = num_heads`.
  - `gridDim.y = num_tokens`.
- **Algorithm**:
  - Token position $p = \text{blockIdx.y} + \text{pos\_offset}$.
  - Coordinate index $i = \text{threadIdx.x} < \text{half\_dim}$.
  - Frequency $\text{freq} = (\theta^{-2i / d_{\text{head}}}) / (\text{freq\_factors} ? \text{freq\_factors}[i] : 1.0f)$.
  - Angle $\alpha = p \times \text{freq}$.
  - Vector offsets: $v_0 = \text{vec}[\text{offset} + i]$, $v_1 = \text{vec}[\text{offset} + \text{half\_dim} + i]$.
  - Rotation:
    $$\text{vec}[\text{offset} + i] = v_0 \cos(\alpha) - v_1 \sin(\alpha)$$
    $$\text{vec}[\text{offset} + \text{half\_dim} + i] = v_0 \sin(\alpha) + v_1 \cos(\alpha)$$
  - Zero shared memory overhead; executes in a single non-blocking pass.

---

### 3.3 Zig Device Memory Abstraction (`src/cuda_mem.zig`)

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

pub fn DeviceBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        ptr: [*]T,
        len: usize,
        device_id: i32,

        pub fn init(device_id: i32, count: usize) cuda.CudaError!Self {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            var raw_ptr: ?*anyopaque = null;
            const bytes = count * @sizeOf(T);
            const rc = c.zeno_cuda_malloc(&raw_ptr, bytes, device_id);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);

            return .{
                .ptr = @ptrCast(@alignCast(raw_ptr.?)),
                .len = count,
                .device_id = device_id,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.len > 0) {
                _ = c.zeno_cuda_free(self.ptr, self.device_id);
                self.ptr = undefined;
                self.len = 0;
            }
        }

        pub fn copyFromHost(self: *Self, host_slice: []const T) cuda.CudaError!void {
            std.debug.assert(host_slice.len == self.len);
            const bytes = self.len * @sizeOf(T);
            const rc = c.zeno_cuda_memcpy_h2d(self.ptr, host_slice.ptr, bytes, self.device_id, null);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        pub fn copyToHost(self: *const Self, host_slice: []T) cuda.CudaError!void {
            std.debug.assert(host_slice.len == self.len);
            const bytes = self.len * @sizeOf(T);
            const rc = c.zeno_cuda_memcpy_d2h(host_slice.ptr, self.ptr, bytes, self.device_id, null);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        pub fn slice(self: *const Self) DeviceSlice {
            return .{
                .ptr = @ptrCast(self.ptr),
                .len = self.len,
                .device_id = self.device_id,
            };
        }
    };
}
```

---

## 4. Verification Strategy & Acceptance Gates

Stage 2 acceptance requires satisfying four strict gates:

1. **Gate 1: Default Pure Zig Build Isolation**:
   - `zig build` and `zig build test` compile cleanly without CUDA toolchains.
   - All CPU unit tests pass with zero leaks.

2. **Gate 2: Portable CUDA Test Suite (R14-02 Conformant)**:
   - `zig build test -Dcuda=true` compiles `ops.cu` and links `libcudart`.
   - On toolkit-only build environments without GPUs, tests detecting `error.NoDevice` or `error.DriverError` cleanly skip via `error.SkipZigTest`.
   - On GPU hardware, all tests execute and validate numerical accuracy.

3. **Gate 3: Numerical Parity with CPU Reference ($\le 10^{-5}$)**:
   - **RMSNorm Test**: Compare GPU `cuda_rms_norm` and `cuda_unweighted_rms_norm` against [`ops.rmsNorm`](../src/ops.zig#L10) and [`ops.unweightedRmsNorm`](../src/ops.zig#L28) on random vectors across dimensions $d \in \{256, 512, 2816\}$. Maximum absolute difference $\le 10^{-5}$.
   - **GeGLU Test**: Compare GPU `cuda_geglu` against [`ops.geglu`](../src/ops.zig#L120) across random inputs. Maximum absolute difference $\le 10^{-5}$.
   - **RoPE Test**: Compare GPU `cuda_rope` against [`infer.applyRoPE`](../src/infer.zig#L115) across SWA ($\theta = 10000, d = 256$) and Global ($\theta = 1000000, d = 512$) layers across sequence length $N = 256$. Maximum absolute difference $\le 10^{-5}$.
   - **Residual & Scale Test**: Compare GPU in-place additions and scalings against CPU slice arithmetic.

4. **Gate 4: Integration Safety & Memory Lifetime**:
   - Verify `DeviceBuffer` allocations, copies, and deallocations leave zero device memory leaks.
   - Run full HTTP decision server test suite (`python3 tools/test_server.py`) to confirm zero CPU regression.

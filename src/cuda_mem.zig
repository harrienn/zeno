//! Zeno — CUDA Backend Device Memory Management & Fused Elementwise Kernels (Stage 2).
//!
//! Provides type-safe abstractions (DeviceBuffer, DeviceSlice, CudaStream) and
//! typed kernel launch interfaces over the native CUDA C ABI runtime layer.
//! All host allocations are 0-byte (pure handle ownership); device operations are
//! explicitly checked for bounds, alignments, single-ownership, and device affinity.

const std = @import("std");
const cuda = @import("cuda.zig");
const ops = @import("ops.zig");
const infer = @import("infer.zig");
const quant = @import("quant.zig");

/// Non-owning view of contiguous device memory on a specific CUDA device.
pub fn DeviceSlice(comptime T: type) type {
    comptime {
        const valid = switch (T) {
            f32, f16, i32, u32, u8 => true,
            else => false,
        };
        if (!valid or @sizeOf(T) == 0) {
            @compileError("DeviceSlice only supports non-zero-sized POD numeric types (f32, f16, i32, u32, u8)");
        }
    }

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

/// Explicit single-owner CUDA execution stream wrapper for kernel ordering.
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

/// Typed owning device buffer with explicit device assignment and single ownership.
pub fn DeviceBuffer(comptime T: type) type {
    comptime {
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
                    return;
                }
                self.ptr = null;
                self.len = 0;
            }
        }

        /// Synchronous copy from host slice to device buffer.
        /// Validates slice lengths at runtime in all build modes (R18-01).
        pub fn copyFromHost(self: *Self, host_slice: []const T) cuda.CudaError!void {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
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
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
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
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
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

// ---------------------------------------------------------------------------
// Typed Kernel Dispatch Functions
// ---------------------------------------------------------------------------

/// Computes Root Mean Square Layer Normalization:
/// out[row, i] = (x[row, i] / sqrt(mean_i(x[row, i]^2) + eps)) * (weight ? weight[i] : 1.0)
pub fn rmsNorm(
    x: DeviceSlice(f32),
    weight: ?DeviceSlice(f32),
    out: DeviceSlice(f32),
    num_rows: usize,
    dim: usize,
    eps: f32,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    // R20-03: Validate bounds before casting to c_int
    if (num_rows > 65535 or dim < 1 or dim > 8192 or eps <= 0.0) return error.InvalidParam;
    const total = std.math.mul(usize, num_rows, dim) catch return error.InvalidParam;
    if (x.len != total or out.len != total) return error.InvalidParam;
    if (x.device_id != out.device_id) return error.DeviceMismatch;
    if (weight) |w| {
        if (w.len != dim) return error.InvalidParam;
        if (w.device_id != x.device_id) return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != x.device_id) return error.DeviceMismatch;
    }
    if (num_rows == 0) return;

    const x_ptr = x.ptr orelse return error.InvalidParam;
    const out_ptr = out.ptr orelse return error.InvalidParam;
    const w_ptr = if (weight) |w| (w.ptr orelse return error.InvalidParam) else null;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_rms_norm(
        x_ptr,
        w_ptr,
        out_ptr,
        @intCast(num_rows),
        @intCast(dim),
        eps,
        x.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Computes GeGLU activation branch:
/// out[i] = gelu_tanh(gate[i]) * up[i]
pub fn geglu(
    gate: DeviceSlice(f32),
    up: DeviceSlice(f32),
    out: DeviceSlice(f32),
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    // R20-03: Validate count bound before C ABI call
    if (gate.len > (((@as(usize, 1) << 31) - 1))) return error.InvalidParam;
    if (gate.len != up.len or gate.len != out.len) return error.InvalidParam;
    if (gate.device_id != up.device_id or gate.device_id != out.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != gate.device_id) return error.DeviceMismatch;
    }
    if (gate.len == 0) return;

    const g_ptr = gate.ptr orelse return error.InvalidParam;
    const u_ptr = up.ptr orelse return error.InvalidParam;
    const o_ptr = out.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_geglu(
        g_ptr,
        u_ptr,
        o_ptr,
        gate.len,
        gate.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Applies NeoX-style (half-split) Rotary Position Embeddings in-place:
/// Rotates coordinate pairs (vec[i], vec[i + head_dim / 2]) using base theta.
pub fn rope(
    vec: DeviceSlice(f32),
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    pos_offset: usize,
    theta: f32,
    freq_factors: ?DeviceSlice(f32),
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    // R20-03: Validate all dimension bounds before casting to c_int
    if (num_tokens > 65535 or num_heads < 1 or num_heads > 64) return error.InvalidParam;
    if (head_dim < 1 or head_dim > 1024 or (head_dim % 2) != 0) return error.InvalidParam;
    if (pos_offset > std.math.maxInt(c_int)) return error.InvalidParam;
    if (theta <= 0.0) return error.InvalidParam;

    const total_heads = std.math.mul(usize, num_tokens, num_heads) catch return error.InvalidParam;
    const total_elems = std.math.mul(usize, total_heads, head_dim) catch return error.InvalidParam;
    if (vec.len != total_elems) return error.InvalidParam;
    if (freq_factors) |ff| {
        if (ff.len < head_dim / 2) return error.InvalidParam;
        if (ff.device_id != vec.device_id) return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != vec.device_id) return error.DeviceMismatch;
    }
    if (num_tokens == 0) return;

    const v_ptr = vec.ptr orelse return error.InvalidParam;
    const ff_ptr = if (freq_factors) |ff| (ff.ptr orelse return error.InvalidParam) else null;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_rope(
        v_ptr,
        @intCast(num_tokens),
        @intCast(num_heads),
        @intCast(head_dim),
        @intCast(pos_offset),
        theta,
        ff_ptr,
        vec.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// In-place residual addition: x[i] += residual[i]
pub fn addResidual(
    x: DeviceSlice(f32),
    residual: DeviceSlice(f32),
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    // R20-03: Validate count bound before C ABI call
    if (x.len > (((@as(usize, 1) << 31) - 1))) return error.InvalidParam;
    if (x.len != residual.len) return error.InvalidParam;
    if (x.device_id != residual.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != x.device_id) return error.DeviceMismatch;
    }
    if (x.len == 0) return;

    const x_ptr = x.ptr orelse return error.InvalidParam;
    const r_ptr = residual.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_add_residual(
        x_ptr,
        r_ptr,
        x.len,
        x.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// In-place vector scaling: x[i] *= scale_val
pub fn scale(
    x: DeviceSlice(f32),
    scale_val: f32,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    // R20-03: Validate count bound before C ABI call
    if (x.len > (((@as(usize, 1) << 31) - 1))) return error.InvalidParam;
    if (stream) |s| {
        if (s.device_id != x.device_id) return error.DeviceMismatch;
    }
    if (x.len == 0) return;

    const x_ptr = x.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_scale(
        x_ptr,
        scale_val,
        x.len,
        x.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Matrix-vector multiplication for Q8_0 quantized weights: out = W * x
/// w: DeviceSlice(u8) representing row-major BlockQ8_0 blocks
/// x: DeviceSlice(f32) of length batch_size * in_dim
/// out: DeviceSlice(f32) of length batch_size * out_dim
pub fn gemvQ8_0(
    w: DeviceSlice(u8),
    x: DeviceSlice(f32),
    out: DeviceSlice(f32),
    out_dim: usize,
    in_dim: usize,
    batch_size: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (out_dim == 0 or in_dim == 0 or batch_size == 0) return error.InvalidParam;
    if (in_dim % quant.QK8_0 != 0) return error.InvalidParam;
    if (out_dim > std.math.maxInt(c_int) or in_dim > std.math.maxInt(c_int) or batch_size > 65535) {
        return error.InvalidParam;
    }

    const blocks_per_row = in_dim / quant.QK8_0;
    const bytes_per_row = blocks_per_row * @sizeOf(quant.BlockQ8_0);
    const expected_w_bytes = std.math.mul(usize, out_dim, bytes_per_row) catch return error.InvalidParam;
    const expected_x_len = std.math.mul(usize, batch_size, in_dim) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, batch_size, out_dim) catch return error.InvalidParam;

    if (w.len != expected_w_bytes or x.len != expected_x_len or out.len != expected_out_len) {
        return error.InvalidParam;
    }
    if (w.device_id != x.device_id or w.device_id != out.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != w.device_id) return error.DeviceMismatch;
    }

    const w_ptr = w.ptr orelse return error.InvalidParam;
    const x_ptr = x.ptr orelse return error.InvalidParam;
    const out_ptr = out.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_gemv_q8_0(
        w_ptr,
        x_ptr,
        out_ptr,
        @intCast(out_dim),
        @intCast(in_dim),
        @intCast(batch_size),
        w.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Matrix-vector multiplication for Q4_K quantized weights: out = W * x
/// w: DeviceSlice(u8) representing row-major BlockQ4_K blocks
/// x: DeviceSlice(f32) of length batch_size * in_dim
/// out: DeviceSlice(f32) of length batch_size * out_dim
pub fn gemvQ4_K(
    w: DeviceSlice(u8),
    x: DeviceSlice(f32),
    out: DeviceSlice(f32),
    out_dim: usize,
    in_dim: usize,
    batch_size: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (out_dim == 0 or in_dim == 0 or batch_size == 0) return error.InvalidParam;
    if (in_dim % quant.QK_K != 0) return error.InvalidParam;
    if (out_dim > std.math.maxInt(c_int) or in_dim > std.math.maxInt(c_int) or batch_size > 65535) {
        return error.InvalidParam;
    }

    const blocks_per_row = in_dim / quant.QK_K;
    const bytes_per_row = blocks_per_row * @sizeOf(quant.BlockQ4_K);
    const expected_w_bytes = std.math.mul(usize, out_dim, bytes_per_row) catch return error.InvalidParam;
    const expected_x_len = std.math.mul(usize, batch_size, in_dim) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, batch_size, out_dim) catch return error.InvalidParam;

    if (w.len != expected_w_bytes or x.len != expected_x_len or out.len != expected_out_len) {
        return error.InvalidParam;
    }
    if (w.device_id != x.device_id or w.device_id != out.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != w.device_id) return error.DeviceMismatch;
    }

    const w_ptr = w.ptr orelse return error.InvalidParam;
    const x_ptr = x.ptr orelse return error.InvalidParam;
    const out_ptr = out.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_gemv_q4k(
        w_ptr,
        x_ptr,
        out_ptr,
        @intCast(out_dim),
        @intCast(in_dim),
        @intCast(batch_size),
        w.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Matrix-vector multiplication for Q6_K quantized weights: out = W * x
/// w: DeviceSlice(u8) representing row-major BlockQ6_K blocks
/// x: DeviceSlice(f32) of length batch_size * in_dim
/// out: DeviceSlice(f32) of length batch_size * out_dim
pub fn gemvQ6_K(
    w: DeviceSlice(u8),
    x: DeviceSlice(f32),
    out: DeviceSlice(f32),
    out_dim: usize,
    in_dim: usize,
    batch_size: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (out_dim == 0 or in_dim == 0 or batch_size == 0) return error.InvalidParam;
    if (in_dim % quant.QK_K != 0) return error.InvalidParam;
    if (out_dim > std.math.maxInt(c_int) or in_dim > std.math.maxInt(c_int) or batch_size > 65535) {
        return error.InvalidParam;
    }

    const blocks_per_row = in_dim / quant.QK_K;
    const bytes_per_row = blocks_per_row * @sizeOf(quant.BlockQ6_K);
    const expected_w_bytes = std.math.mul(usize, out_dim, bytes_per_row) catch return error.InvalidParam;
    const expected_x_len = std.math.mul(usize, batch_size, in_dim) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, batch_size, out_dim) catch return error.InvalidParam;

    if (w.len != expected_w_bytes or x.len != expected_x_len or out.len != expected_out_len) {
        return error.InvalidParam;
    }
    if (w.device_id != x.device_id or w.device_id != out.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != w.device_id) return error.DeviceMismatch;
    }

    const w_ptr = w.ptr orelse return error.InvalidParam;
    const x_ptr = x.ptr orelse return error.InvalidParam;
    const out_ptr = out.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_gemv_q6k(
        w_ptr,
        x_ptr,
        out_ptr,
        @intCast(out_dim),
        @intCast(in_dim),
        @intCast(batch_size),
        w.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Matrix-vector multiplication for Q5_0 quantized weights: out = W * x
/// w: DeviceSlice(u8) representing row-major BlockQ5_0 blocks
/// x: DeviceSlice(f32) of length batch_size * in_dim
/// out: DeviceSlice(f32) of length batch_size * out_dim
pub fn gemvQ5_0(
    w: DeviceSlice(u8),
    x: DeviceSlice(f32),
    out: DeviceSlice(f32),
    out_dim: usize,
    in_dim: usize,
    batch_size: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (out_dim == 0 or in_dim == 0 or batch_size == 0) return error.InvalidParam;
    if (in_dim % quant.QK5_0 != 0) return error.InvalidParam;
    if (out_dim > std.math.maxInt(c_int) or in_dim > std.math.maxInt(c_int) or batch_size > 65535) {
        return error.InvalidParam;
    }

    const blocks_per_row = in_dim / quant.QK5_0;
    const bytes_per_row = blocks_per_row * @sizeOf(quant.BlockQ5_0);
    const expected_w_bytes = std.math.mul(usize, out_dim, bytes_per_row) catch return error.InvalidParam;
    const expected_x_len = std.math.mul(usize, batch_size, in_dim) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, batch_size, out_dim) catch return error.InvalidParam;

    if (w.len != expected_w_bytes or x.len != expected_x_len or out.len != expected_out_len) {
        return error.InvalidParam;
    }
    if (w.device_id != x.device_id or w.device_id != out.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != w.device_id) return error.DeviceMismatch;
    }

    const w_ptr = w.ptr orelse return error.InvalidParam;
    const x_ptr = x.ptr orelse return error.InvalidParam;
    const out_ptr = out.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_gemv_q5_0(
        w_ptr,
        x_ptr,
        out_ptr,
        @intCast(out_dim),
        @intCast(in_dim),
        @intCast(batch_size),
        w.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

// ---------------------------------------------------------------------------
// Stage 4 Unified Attention & MoE Expert Execution Wrappers
// ---------------------------------------------------------------------------

/// Unified multi-head attention with causal prompt masking and bidirectional canvas visibility:
/// out[t, h, d] = sum_{s < visible} softmax(dot(q[t,h], k[s,kv])) * v[s,kv,d]
pub fn attention(
    q: DeviceSlice(f32),
    k: DeviceSlice(f32),
    v: DeviceSlice(f32),
    out: DeviceSlice(f32),
    num_tokens: usize,
    prompt_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    stride_q: usize,
    stride_k: usize,
    stride_v: usize,
    stride_out: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or num_tokens > 512 or prompt_len > num_tokens) return error.InvalidParam;
    if (num_q_heads == 0 or num_kv_heads == 0 or (num_q_heads % num_kv_heads) != 0) return error.InvalidParam;
    if (head_dim != 256 and head_dim != 512) return error.InvalidParam;
    if (stride_q < num_q_heads * head_dim or stride_k < num_kv_heads * head_dim or stride_v < num_kv_heads * head_dim or stride_out < num_q_heads * head_dim) {
        return error.InvalidParam;
    }

    const min_q_len = std.math.mul(usize, num_tokens, stride_q) catch return error.InvalidParam;
    const min_k_len = std.math.mul(usize, num_tokens, stride_k) catch return error.InvalidParam;
    const min_v_len = std.math.mul(usize, num_tokens, stride_v) catch return error.InvalidParam;
    const min_out_len = std.math.mul(usize, num_tokens, stride_out) catch return error.InvalidParam;

    if (q.len < min_q_len or k.len < min_k_len or v.len < min_v_len or out.len < min_out_len) {
        return error.InvalidParam;
    }

    if (q.device_id != k.device_id or q.device_id != v.device_id or q.device_id != out.device_id) {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != q.device_id) return error.DeviceMismatch;
    }

    const q_ptr = q.ptr orelse return error.InvalidParam;
    const k_ptr = k.ptr orelse return error.InvalidParam;
    const v_ptr = v.ptr orelse return error.InvalidParam;
    const out_ptr = out.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_attention(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        @intCast(num_tokens),
        @intCast(prompt_len),
        @intCast(num_q_heads),
        @intCast(num_kv_heads),
        @intCast(head_dim),
        @intCast(stride_q),
        @intCast(stride_k),
        @intCast(stride_v),
        @intCast(stride_out),
        q.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Prepares MoE router input: r_in[t, dim] *= root_size * router_scale[dim]
pub fn moeRouterPrep(
    r_in: DeviceSlice(f32),
    router_scale: DeviceSlice(f32),
    root_size: f32,
    num_tokens: usize,
    dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or dim == 0) return error.InvalidParam;
    const min_len = std.math.mul(usize, num_tokens, dim) catch return error.InvalidParam;
    if (r_in.len < min_len or router_scale.len < dim) return error.InvalidParam;

    if (r_in.device_id != router_scale.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != r_in.device_id) return error.DeviceMismatch;
    }

    const r_ptr = r_in.ptr orelse return error.InvalidParam;
    const s_ptr = router_scale.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_router_prep(
        r_ptr,
        s_ptr,
        root_size,
        @intCast(num_tokens),
        @intCast(dim),
        r_in.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Selects top-8 experts from 128 router logits, computes softmax renormalization, and applies per-expert output scaling
pub fn moeRouterTop8(
    router_logits: DeviceSlice(f32),
    exp_down_scale: ?DeviceSlice(f32),
    out_indices: DeviceSlice(i32),
    out_weights: DeviceSlice(f32),
    num_tokens: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0) return error.InvalidParam;

    const min_logits_len = std.math.mul(usize, num_tokens, 128) catch return error.InvalidParam;
    const min_out_len = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;

    if (router_logits.len < min_logits_len or out_indices.len < min_out_len or out_weights.len < min_out_len) {
        return error.InvalidParam;
    }

    if (router_logits.device_id != out_indices.device_id or router_logits.device_id != out_weights.device_id) {
        return error.DeviceMismatch;
    }
    if (exp_down_scale) |eds| {
        if (eds.device_id != router_logits.device_id or eds.len < 128) return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != router_logits.device_id) return error.DeviceMismatch;
    }

    const logits_ptr = router_logits.ptr orelse return error.InvalidParam;
    const eds_ptr = if (exp_down_scale) |eds| (eds.ptr orelse return error.InvalidParam) else null;
    const indices_ptr = out_indices.ptr orelse return error.InvalidParam;
    const weights_ptr = out_weights.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_router_topk(
        logits_ptr,
        eds_ptr,
        @ptrCast(indices_ptr),
        weights_ptr,
        @intCast(num_tokens),
        router_logits.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Accumulates scaled vector: accum[i] += x[i] * scale_factor
pub fn accumulateScaled(
    accum: DeviceSlice(f32),
    x: DeviceSlice(f32),
    scale_factor: f32,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (accum.len != x.len) return error.InvalidParam;
    if (accum.device_id != x.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != accum.device_id) return error.DeviceMismatch;
    }
    if (accum.len == 0) return;

    const a_ptr = accum.ptr orelse return error.InvalidParam;
    const x_ptr = x.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_accumulate_scaled(
        a_ptr,
        x_ptr,
        scale_factor,
        accum.len,
        accum.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

// ---------------------------------------------------------------------------
// Unit Tests & Numerical Parity Verification
// ---------------------------------------------------------------------------

fn requireHardwareDevice() !i32 {
    if (!cuda.is_enabled) return error.SkipZigTest;
    const count = cuda.getDeviceCount() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (count == 0) return error.SkipZigTest;
    return 0; // Test on device 0
}

test "DeviceSlice and subslice checked arithmetic" {
    const dev: i32 = 0;
    const empty_slice = DeviceSlice(f32).empty(dev);
    try std.testing.expectEqual(@as(?[*]f32, null), empty_slice.ptr);
    try std.testing.expectEqual(@as(usize, 0), empty_slice.len);
    try std.testing.expectEqual(dev, empty_slice.device_id);

    // subslice(0, 0) on empty slice returns empty slice without error
    const sub_empty = try empty_slice.subslice(0, 0);
    try std.testing.expectEqual(@as(?[*]f32, null), sub_empty.ptr);
    try std.testing.expectEqual(@as(usize, 0), sub_empty.len);

    // subslice with start > 0 or len > 0 on empty slice returns InvalidParam
    try std.testing.expectError(error.InvalidParam, empty_slice.subslice(1, 0));
    try std.testing.expectError(error.InvalidParam, empty_slice.subslice(0, 1));

    // subslice arithmetic overflow check
    try std.testing.expectError(error.InvalidParam, empty_slice.subslice(std.math.maxInt(usize), 1));
}

test "DeviceBuffer empty initialization and move" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    var buf = try DeviceBuffer(f32).init(dev, 0);
    defer buf.deinit();

    try std.testing.expectEqual(@as(?[*]f32, null), buf.ptr);
    try std.testing.expectEqual(@as(usize, 0), buf.len);
    try std.testing.expectEqual(dev, buf.device_id);

    const s = buf.slice();
    try std.testing.expectEqual(@as(?[*]f32, null), s.ptr);
    try std.testing.expectEqual(@as(usize, 0), s.len);

    var moved = buf.move();
    defer moved.deinit();
    try std.testing.expectEqual(@as(?[*]f32, null), buf.ptr);
    try std.testing.expectEqual(@as(usize, 0), buf.len);
    try std.testing.expectEqual(@as(usize, 0), moved.len);

    // Repeated deinit is safe no-op
    buf.deinit();
    moved.deinit();
}

test "DeviceBuffer allocation, H2D and D2H transfer" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const count: usize = 1024;
    var host_src: [1024]f32 = undefined;
    for (&host_src, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 1.5;
    }

    var dev_buf = try DeviceBuffer(f32).init(dev, count);
    defer dev_buf.deinit();

    try std.testing.expect(dev_buf.ptr != null);
    try std.testing.expectEqual(count, dev_buf.len);

    // Transfer H2D
    try dev_buf.copyFromHost(&host_src);

    // Transfer D2H
    var host_dst: [1024]f32 = undefined;
    @memset(&host_dst, 0.0);
    try dev_buf.copyToHost(&host_dst);

    for (host_src, host_dst) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }

    // Length mismatch returns InvalidParam in all build modes
    var small_host: [512]f32 = undefined;
    try std.testing.expectError(error.InvalidParam, dev_buf.copyFromHost(&small_host));
    try std.testing.expectError(error.InvalidParam, dev_buf.copyToHost(&small_host));
}

test "DeviceBuffer D2D transfer and device matching" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const count: usize = 256;
    var b1 = try DeviceBuffer(f32).init(dev, count);
    defer b1.deinit();
    var b2 = try DeviceBuffer(f32).init(dev, count);
    defer b2.deinit();

    var host_data: [256]f32 = undefined;
    for (&host_data, 0..) |*v, i| {
        v.* = @floatFromInt(i + 1);
    }
    try b1.copyFromHost(&host_data);

    // Copy D2D
    try b2.copyFromDevice(b1.slice());

    var host_result: [256]f32 = undefined;
    try b2.copyToHost(&host_result);

    for (host_data, host_result) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }

    // Cross-device mismatch check
    const fake_slice = DeviceSlice(f32){
        .ptr = b1.ptr,
        .len = count,
        .device_id = dev + 99,
    };
    try std.testing.expectError(error.DeviceMismatch, b2.copyFromDevice(fake_slice));
}

test "CudaStream lifecycle, move, and sync" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    var stream = try CudaStream.init(dev);
    defer stream.deinit();
    try std.testing.expect(stream.handle != null);

    try stream.synchronize();

    var moved = stream.move();
    defer moved.deinit();
    try std.testing.expectEqual(@as(?*anyopaque, null), stream.handle);
    try std.testing.expect(moved.handle != null);

    try moved.synchronize();
}

test "CUDA kernel wrapper dimension bounds checking" {
    if (!cuda.is_enabled) return;
    const dummy = DeviceSlice(f32).empty(0);

    // RMSNorm bounds validation
    try std.testing.expectError(error.InvalidParam, rmsNorm(dummy, null, dummy, 70000, 256, 1e-5, null));
    try std.testing.expectError(error.InvalidParam, rmsNorm(dummy, null, dummy, 1, 0, 1e-5, null));
    try std.testing.expectError(error.InvalidParam, rmsNorm(dummy, null, dummy, 1, 10000, 1e-5, null));
    try std.testing.expectError(error.InvalidParam, rmsNorm(dummy, null, dummy, 1, 256, -1.0, null));
    try std.testing.expectError(error.InvalidParam, rmsNorm(dummy, null, dummy, 1, 256, 0.0, null));

    // RoPE bounds validation
    try std.testing.expectError(error.InvalidParam, rope(dummy, 70000, 8, 256, 0, 10000.0, null, null));
    try std.testing.expectError(error.InvalidParam, rope(dummy, 1, 0, 256, 0, 10000.0, null, null));
    try std.testing.expectError(error.InvalidParam, rope(dummy, 1, 100, 256, 0, 10000.0, null, null));
    try std.testing.expectError(error.InvalidParam, rope(dummy, 1, 8, 0, 0, 10000.0, null, null));
    try std.testing.expectError(error.InvalidParam, rope(dummy, 1, 8, 1025, 0, 10000.0, null, null));
    try std.testing.expectError(error.InvalidParam, rope(dummy, 1, 8, 255, 0, 10000.0, null, null));
    try std.testing.expectError(error.InvalidParam, rope(dummy, 1, 8, 256, @as(usize, std.math.maxInt(c_int)) + 1, 10000.0, null, null));
    try std.testing.expectError(error.InvalidParam, rope(dummy, 1, 8, 256, 0, 0.0, null, null));
    try std.testing.expectError(error.InvalidParam, rope(dummy, 1, 8, 256, 0, -1.0, null, null));

    // GeGLU, AddResidual, Scale count bound validation
    const huge_slice = DeviceSlice(f32){ .ptr = null, .len = (@as(usize, 1) << 31), .device_id = 0 };
    try std.testing.expectError(error.InvalidParam, geglu(huge_slice, huge_slice, huge_slice, null));
    try std.testing.expectError(error.InvalidParam, addResidual(huge_slice, huge_slice, null));
    try std.testing.expectError(error.InvalidParam, scale(huge_slice, 1.0, null));
}

test "CUDA device memory leak accounting" {
    if (!cuda.is_enabled) return;
    const count = cuda.getDeviceCount() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (count == 0) return error.SkipZigTest;
    // Prefer idle secondary device (e.g. dev 1) to avoid desktop display manager allocation races
    const dev: i32 = if (count > 1) 1 else 0;

    // Warm up CUDA runtime context on this device so driver context initialization allocations are complete
    {
        var warmup = try DeviceBuffer(f32).init(dev, 1024);
        warmup.deinit();
    }

    const topo_before = try cuda.getTopology();
    const free_before = topo_before.devices[@intCast(dev)].free_memory_bytes;

    // Allocate 64 MiB (16M floats * 4 bytes)
    const alloc_floats: usize = 16 * 1024 * 1024;
    {
        var buf1 = try DeviceBuffer(f32).init(dev, alloc_floats / 2);
        defer buf1.deinit();
        var buf2 = try DeviceBuffer(f32).init(dev, alloc_floats / 2);
        defer buf2.deinit();

        const topo_during = try cuda.getTopology();
        const free_during = topo_during.devices[@intCast(dev)].free_memory_bytes;

        // Verify that free memory decreased substantially during 64 MiB allocation (> 32 MiB decrease)
        try std.testing.expect(free_before > free_during);
        try std.testing.expect((free_before - free_during) >= 32 * 1024 * 1024);
    }

    // After buf1 and buf2 are deinit'd, query topology again
    const topo_after = try cuda.getTopology();
    const free_after = topo_after.devices[@intCast(dev)].free_memory_bytes;

    // Verify all allocated VRAM has been reclaimed within 8 MiB tolerance (absorbs window manager activity on primary dev)
    const delta = if (free_before > free_after) (free_before - free_after) else (free_after - free_before);
    try std.testing.expect(delta < 8 * 1024 * 1024);
}

test "CUDA RMSNorm numerical parity (weighted)" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x537461676532);
    const rand = prng.random();

    const dims = [_]usize{ 256, 512, 2816 };
    const eps: f32 = 1e-6;

    for (dims) |d| {
        const num_rows: usize = 4;
        const total = num_rows * d;

        const host_x = try allocator.alloc(f32, total);
        defer allocator.free(host_x);
        const host_w = try allocator.alloc(f32, d);
        defer allocator.free(host_w);
        const cpu_out = try allocator.alloc(f32, total);
        defer allocator.free(cpu_out);
        const gpu_out = try allocator.alloc(f32, total);
        defer allocator.free(gpu_out);

        for (host_x) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
        for (host_w) |*v| v.* = (rand.float(f32) * 1.9) + 0.1;

        // CPU reference
        for (0..num_rows) |r| {
            const rx = host_x[r * d .. (r + 1) * d];
            const ro = cpu_out[r * d .. (r + 1) * d];
            ops.rmsNorm(rx, host_w, eps, ro);
        }

        // CUDA execution
        var d_x = try DeviceBuffer(f32).init(dev, total);
        defer d_x.deinit();
        var d_w = try DeviceBuffer(f32).init(dev, d);
        defer d_w.deinit();
        var d_out = try DeviceBuffer(f32).init(dev, total);
        defer d_out.deinit();

        try d_x.copyFromHost(host_x);
        try d_w.copyFromHost(host_w);

        try rmsNorm(d_x.slice(), d_w.slice(), d_out.slice(), num_rows, d, eps, null);
        try d_out.copyToHost(gpu_out);

        // Tolerance check: atol = 1e-4, rtol = 1e-4
        const atol: f32 = 1e-4;
        const rtol: f32 = 1e-4;
        var max_diff: f32 = 0.0;
        for (cpu_out, gpu_out) |c, g| {
            const diff = @abs(c - g);
            if (diff > max_diff) max_diff = diff;
            const tol = atol + rtol * @abs(c);
            try std.testing.expect(diff <= tol);
        }
    }
}

test "CUDA RMSNorm numerical parity (unweighted) and in-place" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x537461676532);
    const rand = prng.random();

    const d: usize = 2816;
    const num_rows: usize = 2;
    const total = num_rows * d;
    const eps: f32 = 1e-6;

    const host_x = try allocator.alloc(f32, total);
    defer allocator.free(host_x);
    const cpu_out = try allocator.alloc(f32, total);
    defer allocator.free(cpu_out);
    const gpu_out = try allocator.alloc(f32, total);
    defer allocator.free(gpu_out);

    for (host_x) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;

    for (0..num_rows) |r| {
        const rx = host_x[r * d .. (r + 1) * d];
        const ro = cpu_out[r * d .. (r + 1) * d];
        ops.unweightedRmsNorm(rx, eps, ro);
    }

    // CUDA in-place execution (out == x, weight == null)
    var d_x = try DeviceBuffer(f32).init(dev, total);
    defer d_x.deinit();
    try d_x.copyFromHost(host_x);

    try rmsNorm(d_x.slice(), null, d_x.slice(), num_rows, d, eps, null);
    try d_x.copyToHost(gpu_out);

    const atol: f32 = 1e-4;
    const rtol: f32 = 1e-4;
    for (cpu_out, gpu_out) |c, g| {
        const diff = @abs(c - g);
        const tol = atol + rtol * @abs(c);
        try std.testing.expect(diff <= tol);
    }
}

test "CUDA GeGLU numerical parity and in-place" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x537461676532);
    const rand = prng.random();

    const count: usize = 4096;
    const host_gate = try allocator.alloc(f32, count);
    defer allocator.free(host_gate);
    const host_up = try allocator.alloc(f32, count);
    defer allocator.free(host_up);
    const cpu_out = try allocator.alloc(f32, count);
    defer allocator.free(cpu_out);
    const gpu_out = try allocator.alloc(f32, count);
    defer allocator.free(gpu_out);

    for (host_gate) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
    for (host_up) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;

    ops.geglu(host_gate, host_up, cpu_out);

    // In-place on gate (out == gate)
    var d_gate = try DeviceBuffer(f32).init(dev, count);
    defer d_gate.deinit();
    var d_up = try DeviceBuffer(f32).init(dev, count);
    defer d_up.deinit();

    try d_gate.copyFromHost(host_gate);
    try d_up.copyFromHost(host_up);

    try geglu(d_gate.slice(), d_up.slice(), d_gate.slice(), null);
    try d_gate.copyToHost(gpu_out);

    // Tolerance: atol = 1e-5, rtol = 1e-4
    const atol: f32 = 1e-5;
    const rtol: f32 = 1e-4;
    for (cpu_out, gpu_out) |c, g| {
        const diff = @abs(c - g);
        const tol = atol + rtol * @abs(c);
        try std.testing.expect(diff <= tol);
    }
}

test "CUDA RoPE numerical parity across SWA and Global heads" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x537461676532);
    const rand = prng.random();

    const num_tokens: usize = 4;
    const num_heads: usize = 8;
    const pos_offset: usize = 16;

    // Test SWA head_dim = 256, theta = 10000.0 and Global head_dim = 512, theta = 1000000.0
    const configs = [_]struct { head_dim: usize, theta: f32 }{
        .{ .head_dim = 256, .theta = 10000.0 },
        .{ .head_dim = 512, .theta = 1000000.0 },
    };

    for (configs) |cfg| {
        for ([_]bool{ false, true }) |with_factors| {
            const total = num_tokens * num_heads * cfg.head_dim;
            const host_vec = try allocator.alloc(f32, total);
            defer allocator.free(host_vec);
            const cpu_vec = try allocator.alloc(f32, total);
            defer allocator.free(cpu_vec);
            const gpu_vec = try allocator.alloc(f32, total);
            defer allocator.free(gpu_vec);

            for (host_vec) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
            @memcpy(cpu_vec, host_vec);

            const half_dim = cfg.head_dim / 2;
            const host_ff = try allocator.alloc(f32, half_dim);
            defer allocator.free(host_ff);
            for (host_ff) |*f| f.* = 1.0 + rand.float(f32) * 2.0;

            const cpu_ff: ?[*]const f32 = if (with_factors) host_ff.ptr else null;

            // CPU reference
            for (0..num_tokens) |t| {
                for (0..num_heads) |h| {
                    const off = (t * num_heads + h) * cfg.head_dim;
                    const head_slice = cpu_vec[off .. off + cfg.head_dim];
                    infer.applyRoPE(head_slice, pos_offset + t, cfg.head_dim, cfg.theta, cpu_ff);
                }
            }

            // CUDA execution
            var d_vec = try DeviceBuffer(f32).init(dev, total);
            defer d_vec.deinit();
            try d_vec.copyFromHost(host_vec);

            var d_ff: ?DeviceBuffer(f32) = if (with_factors) try DeviceBuffer(f32).init(dev, half_dim) else null;
            defer if (d_ff) |*ff_buf| ff_buf.deinit();
            if (d_ff) |*ff_buf| {
                try ff_buf.copyFromHost(host_ff);
            }

            const ff_slice: ?DeviceSlice(f32) = if (d_ff) |*ff_buf| ff_buf.slice() else null;

            try rope(d_vec.slice(), num_tokens, num_heads, cfg.head_dim, pos_offset, cfg.theta, ff_slice, null);
            try d_vec.copyToHost(gpu_vec);

            const atol: f32 = 1e-5;
            const rtol: f32 = 1e-4;
            for (cpu_vec, gpu_vec) |c, g| {
                const diff = @abs(c - g);
                const tol = atol + rtol * @abs(c);
                try std.testing.expect(diff <= tol);
            }
        }
    }
}

test "CUDA Residual Add and Scale numerical parity" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x537461676532);
    const rand = prng.random();

    const count: usize = 2048;
    const host_x = try allocator.alloc(f32, count);
    defer allocator.free(host_x);
    const host_res = try allocator.alloc(f32, count);
    defer allocator.free(host_res);
    const cpu_x = try allocator.alloc(f32, count);
    defer allocator.free(cpu_x);
    const gpu_x = try allocator.alloc(f32, count);
    defer allocator.free(gpu_x);

    for (host_x) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
    for (host_res) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
    @memcpy(cpu_x, host_x);

    // CPU operations: Add residual, then scale
    const scale_val: f32 = 1.5;
    for (cpu_x, host_res) |*x_val, res_val| {
        x_val.* += res_val;
        x_val.* *= scale_val;
    }

    // CUDA operations
    var d_x = try DeviceBuffer(f32).init(dev, count);
    defer d_x.deinit();
    var d_res = try DeviceBuffer(f32).init(dev, count);
    defer d_res.deinit();

    try d_x.copyFromHost(host_x);
    try d_res.copyFromHost(host_res);

    try addResidual(d_x.slice(), d_res.slice(), null);
    try scale(d_x.slice(), scale_val, null);

    try d_x.copyToHost(gpu_x);

    const atol: f32 = 1e-6;
    const rtol: f32 = 1e-6;
    for (cpu_x, gpu_x) |c, g| {
        const diff = @abs(c - g);
        const tol = atol + rtol * @abs(c);
        try std.testing.expect(diff <= tol);
    }
}

test "Stage 2 Numerical Parity 10000 points empirical measurement" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x537461676532);
    const rand = prng.random();

    // 1. RMSNorm (Weighted): 4 rows x 2816 = 11,264 points (> 10,000 points)
    {
        const num_rows: usize = 4;
        const d: usize = 2816;
        const total = num_rows * d;
        const eps: f32 = 1e-6;

        const host_x = try allocator.alloc(f32, total);
        defer allocator.free(host_x);
        const host_w = try allocator.alloc(f32, d);
        defer allocator.free(host_w);
        const cpu_out = try allocator.alloc(f32, total);
        defer allocator.free(cpu_out);
        const gpu_out = try allocator.alloc(f32, total);
        defer allocator.free(gpu_out);

        for (host_x) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
        for (host_w) |*v| v.* = (rand.float(f32) * 1.9) + 0.1;

        for (0..num_rows) |r| {
            ops.rmsNorm(host_x[r * d .. (r + 1) * d], host_w, eps, cpu_out[r * d .. (r + 1) * d]);
        }

        var d_x = try DeviceBuffer(f32).init(dev, total);
        defer d_x.deinit();
        var d_w = try DeviceBuffer(f32).init(dev, d);
        defer d_w.deinit();
        var d_out = try DeviceBuffer(f32).init(dev, total);
        defer d_out.deinit();

        try d_x.copyFromHost(host_x);
        try d_w.copyFromHost(host_w);
        try rmsNorm(d_x.slice(), d_w.slice(), d_out.slice(), num_rows, d, eps, null);
        try d_out.copyToHost(gpu_out);

        var max_abs: f32 = 0.0;
        var max_rel: f32 = 0.0;
        for (cpu_out, gpu_out) |c, g| {
            const diff = @abs(c - g);
            if (diff > max_abs) max_abs = diff;
            const rel = if (@abs(c) > 1e-7) diff / @abs(c) else diff;
            if (rel > max_rel) max_rel = rel;
            try std.testing.expect(diff <= 1e-4 + 1e-4 * @abs(c));
        }
        std.debug.print("\n[parity] RMSNorm (Weighted): max_abs = {e:.4}, max_rel = {e:.4}\n", .{ max_abs, max_rel });
    }

    // 2. RMSNorm (Unweighted): 4 rows x 2816 = 11,264 points
    {
        const num_rows: usize = 4;
        const d: usize = 2816;
        const total = num_rows * d;
        const eps: f32 = 1e-6;

        const host_x = try allocator.alloc(f32, total);
        defer allocator.free(host_x);
        const cpu_out = try allocator.alloc(f32, total);
        defer allocator.free(cpu_out);
        const gpu_out = try allocator.alloc(f32, total);
        defer allocator.free(gpu_out);

        for (host_x) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;

        for (0..num_rows) |r| {
            ops.unweightedRmsNorm(host_x[r * d .. (r + 1) * d], eps, cpu_out[r * d .. (r + 1) * d]);
        }

        var d_x = try DeviceBuffer(f32).init(dev, total);
        defer d_x.deinit();
        var d_out = try DeviceBuffer(f32).init(dev, total);
        defer d_out.deinit();

        try d_x.copyFromHost(host_x);
        try rmsNorm(d_x.slice(), null, d_out.slice(), num_rows, d, eps, null);
        try d_out.copyToHost(gpu_out);

        var max_abs: f32 = 0.0;
        var max_rel: f32 = 0.0;
        for (cpu_out, gpu_out) |c, g| {
            const diff = @abs(c - g);
            if (diff > max_abs) max_abs = diff;
            const rel = if (@abs(c) > 1e-7) diff / @abs(c) else diff;
            if (rel > max_rel) max_rel = rel;
            try std.testing.expect(diff <= 1e-4 + 1e-4 * @abs(c));
        }
        std.debug.print("[parity] RMSNorm (Unweighted): max_abs = {e:.4}, max_rel = {e:.4}\n", .{ max_abs, max_rel });
    }

    // 3. GeGLU: 10,000 points
    {
        const count: usize = 10000;
        const host_gate = try allocator.alloc(f32, count);
        defer allocator.free(host_gate);
        const host_up = try allocator.alloc(f32, count);
        defer allocator.free(host_up);
        const cpu_out = try allocator.alloc(f32, count);
        defer allocator.free(cpu_out);
        const gpu_out = try allocator.alloc(f32, count);
        defer allocator.free(gpu_out);

        for (host_gate) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
        for (host_up) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;

        ops.geglu(host_gate, host_up, cpu_out);

        var d_gate = try DeviceBuffer(f32).init(dev, count);
        defer d_gate.deinit();
        var d_up = try DeviceBuffer(f32).init(dev, count);
        defer d_up.deinit();
        var d_out = try DeviceBuffer(f32).init(dev, count);
        defer d_out.deinit();

        try d_gate.copyFromHost(host_gate);
        try d_up.copyFromHost(host_up);

        try geglu(d_gate.slice(), d_up.slice(), d_out.slice(), null);
        try d_out.copyToHost(gpu_out);

        var max_abs: f32 = 0.0;
        var max_rel: f32 = 0.0;
        for (cpu_out, gpu_out) |c, g| {
            const diff = @abs(c - g);
            if (diff > max_abs) max_abs = diff;
            const rel = if (@abs(c) > 1e-7) diff / @abs(c) else diff;
            if (rel > max_rel) max_rel = rel;
            try std.testing.expect(diff <= 1e-5 + 1e-4 * @abs(c));
        }
        std.debug.print("[parity] GeGLU: max_abs = {e:.4}, max_rel = {e:.4}\n", .{ max_abs, max_rel });
    }

    // 4. RoPE: 4 tokens x 8 heads x 512 dim = 16,384 points (> 10,000 points)
    {
        const num_tokens: usize = 4;
        const num_heads: usize = 8;
        const head_dim: usize = 512;
        const total = num_tokens * num_heads * head_dim;
        const pos_offset: usize = 32;
        const theta: f32 = 1000000.0;

        const host_vec = try allocator.alloc(f32, total);
        defer allocator.free(host_vec);
        const cpu_vec = try allocator.alloc(f32, total);
        defer allocator.free(cpu_vec);
        const gpu_vec = try allocator.alloc(f32, total);
        defer allocator.free(gpu_vec);

        for (host_vec) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
        @memcpy(cpu_vec, host_vec);

        for (0..num_tokens) |t| {
            for (0..num_heads) |h| {
                const off = (t * num_heads + h) * head_dim;
                infer.applyRoPE(cpu_vec[off .. off + head_dim], pos_offset + t, head_dim, theta, null);
            }
        }

        var d_vec = try DeviceBuffer(f32).init(dev, total);
        defer d_vec.deinit();
        try d_vec.copyFromHost(host_vec);

        try rope(d_vec.slice(), num_tokens, num_heads, head_dim, pos_offset, theta, null, null);
        try d_vec.copyToHost(gpu_vec);

        var max_abs: f32 = 0.0;
        var max_rel: f32 = 0.0;
        for (cpu_vec, gpu_vec) |c, g| {
            const diff = @abs(c - g);
            if (diff > max_abs) max_abs = diff;
            const rel = if (@abs(c) > 1e-7) diff / @abs(c) else diff;
            if (rel > max_rel) max_rel = rel;
            try std.testing.expect(diff <= 1e-5 + 1e-4 * @abs(c));
        }
        std.debug.print("[parity] RoPE: max_abs = {e:.4}, max_rel = {e:.4}\n", .{ max_abs, max_rel });
    }

    // 5. Residual Add & Scale: 10,000 points
    {
        const count: usize = 10000;
        const host_x = try allocator.alloc(f32, count);
        defer allocator.free(host_x);
        const host_res = try allocator.alloc(f32, count);
        defer allocator.free(host_res);
        const cpu_x = try allocator.alloc(f32, count);
        defer allocator.free(cpu_x);
        const gpu_x = try allocator.alloc(f32, count);
        defer allocator.free(gpu_x);

        for (host_x) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
        for (host_res) |*v| v.* = (rand.float(f32) * 6.0) - 3.0;
        @memcpy(cpu_x, host_x);

        const scale_val: f32 = 1.5;
        for (cpu_x, host_res) |*x_val, res_val| {
            x_val.* += res_val;
            x_val.* *= scale_val;
        }

        var d_x = try DeviceBuffer(f32).init(dev, count);
        defer d_x.deinit();
        var d_res = try DeviceBuffer(f32).init(dev, count);
        defer d_res.deinit();

        try d_x.copyFromHost(host_x);
        try d_res.copyFromHost(host_res);

        try addResidual(d_x.slice(), d_res.slice(), null);
        try scale(d_x.slice(), scale_val, null);

        try d_x.copyToHost(gpu_x);

        var max_abs: f32 = 0.0;
        var max_rel: f32 = 0.0;
        for (cpu_x, gpu_x) |c, g| {
            const diff = @abs(c - g);
            if (diff > max_abs) max_abs = diff;
            const rel = if (@abs(c) > 1e-7) diff / @abs(c) else diff;
            if (rel > max_rel) max_rel = rel;
            try std.testing.expect(diff <= 1e-6 + 1e-6 * @abs(c));
        }
        std.debug.print("[parity] Residual Add & Scale: max_abs = {e:.4}, max_rel = {e:.4}\n", .{ max_abs, max_rel });
    }
}

test "CUDA gemvQ8_0 numerical parity with CPU reference" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x513830);
    const rand = prng.random();

    const out_dim: usize = 32;
    const in_dim: usize = 256;
    const blocks_per_row = in_dim / quant.QK8_0; // 8
    const total_blocks = out_dim * blocks_per_row;

    const host_w = try allocator.alloc(quant.BlockQ8_0, total_blocks);
    defer allocator.free(host_w);
    for (host_w) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.4) + 0.1);
        for (&blk.qs) |*q| {
            q.* = rand.intRangeAtMost(i8, -120, 120);
        }
    }

    const host_x = try allocator.alloc(f32, in_dim);
    defer allocator.free(host_x);
    for (host_x) |*x| {
        x.* = (rand.float(f32) * 4.0) - 2.0;
    }

    const cpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(cpu_out);
    infer.gemvQ8_0(host_w.ptr, host_x, cpu_out, out_dim, in_dim);

    const raw_w_bytes = std.mem.sliceAsBytes(host_w);
    var d_w = try DeviceBuffer(u8).init(dev, raw_w_bytes.len);
    defer d_w.deinit();
    try d_w.copyFromHost(raw_w_bytes);

    var d_x = try DeviceBuffer(f32).init(dev, in_dim);
    defer d_x.deinit();
    try d_x.copyFromHost(host_x);

    var d_out = try DeviceBuffer(f32).init(dev, out_dim);
    defer d_out.deinit();

    try gemvQ8_0(d_w.slice(), d_x.slice(), d_out.slice(), out_dim, in_dim, 1, null);

    const gpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(gpu_out);
    try d_out.copyToHost(gpu_out);

    var max_diff: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_diff) max_diff = diff;
        try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
    }
    std.debug.print("\n[parity] gemvQ8_0 (32x256): max_diff = {e:.4}\n", .{max_diff});
}

test "CUDA gemvQ4_K numerical parity with CPU reference" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51344B);
    const rand = prng.random();

    const out_dim: usize = 16;
    const in_dim: usize = 512;
    const blocks_per_row = in_dim / quant.QK_K; // 2
    const total_blocks = out_dim * blocks_per_row;

    const host_w = try allocator.alloc(quant.BlockQ4_K, total_blocks);
    defer allocator.free(host_w);
    for (host_w) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.3) + 0.1);
        blk.dmin = @floatCast((rand.float(f32) * 0.15) + 0.05);
        for (&blk.scales) |*sc| sc.* = rand.int(u8);
        for (&blk.qs) |*q| q.* = rand.int(u8);
    }

    const host_x = try allocator.alloc(f32, in_dim);
    defer allocator.free(host_x);
    for (host_x) |*x| {
        x.* = (rand.float(f32) * 4.0) - 2.0;
    }

    const cpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(cpu_out);
    infer.gemvQ4_K(host_w.ptr, host_x, cpu_out, out_dim, in_dim);

    const raw_w_bytes = std.mem.sliceAsBytes(host_w);
    var d_w = try DeviceBuffer(u8).init(dev, raw_w_bytes.len);
    defer d_w.deinit();
    try d_w.copyFromHost(raw_w_bytes);

    var d_x = try DeviceBuffer(f32).init(dev, in_dim);
    defer d_x.deinit();
    try d_x.copyFromHost(host_x);

    var d_out = try DeviceBuffer(f32).init(dev, out_dim);
    defer d_out.deinit();

    try gemvQ4_K(d_w.slice(), d_x.slice(), d_out.slice(), out_dim, in_dim, 1, null);

    const gpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(gpu_out);
    try d_out.copyToHost(gpu_out);

    var max_diff: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_diff) max_diff = diff;
        try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
    }
    std.debug.print("[parity] gemvQ4_K (16x512): max_diff = {e:.4}\n", .{max_diff});
}

test "CUDA gemvQ6_K numerical parity with CPU reference" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51364B);
    const rand = prng.random();

    const out_dim: usize = 16;
    const in_dim: usize = 512;
    const blocks_per_row = in_dim / quant.QK_K; // 2
    const total_blocks = out_dim * blocks_per_row;

    const host_w = try allocator.alloc(quant.BlockQ6_K, total_blocks);
    defer allocator.free(host_w);
    for (host_w) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.25) + 0.1);
        for (&blk.scales) |*sc| sc.* = rand.intRangeAtMost(i8, -16, 16);
        for (&blk.ql) |*q| q.* = rand.int(u8);
        for (&blk.qh) |*q| q.* = rand.int(u8);
    }

    const host_x = try allocator.alloc(f32, in_dim);
    defer allocator.free(host_x);
    for (host_x) |*x| {
        x.* = (rand.float(f32) * 4.0) - 2.0;
    }

    const cpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(cpu_out);
    infer.gemvQ6_K(host_w.ptr, host_x, cpu_out, out_dim, in_dim);

    const raw_w_bytes = std.mem.sliceAsBytes(host_w);
    var d_w = try DeviceBuffer(u8).init(dev, raw_w_bytes.len);
    defer d_w.deinit();
    try d_w.copyFromHost(raw_w_bytes);

    var d_x = try DeviceBuffer(f32).init(dev, in_dim);
    defer d_x.deinit();
    try d_x.copyFromHost(host_x);

    var d_out = try DeviceBuffer(f32).init(dev, out_dim);
    defer d_out.deinit();

    try gemvQ6_K(d_w.slice(), d_x.slice(), d_out.slice(), out_dim, in_dim, 1, null);

    const gpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(gpu_out);
    try d_out.copyToHost(gpu_out);

    var max_diff: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_diff) max_diff = diff;
        try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
    }
    std.debug.print("[parity] gemvQ6_K (16x512): max_diff = {e:.4}\n", .{max_diff});
}

test "CUDA gemvQ5_0 numerical parity with CPU reference" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x513530);
    const rand = prng.random();

    const out_dim: usize = 32;
    const in_dim: usize = 256;
    const blocks_per_row = in_dim / quant.QK5_0; // 8
    const total_blocks = out_dim * blocks_per_row;

    const host_w = try allocator.alloc(quant.BlockQ5_0, total_blocks);
    defer allocator.free(host_w);
    for (host_w) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.3) + 0.1);
        for (&blk.qh) |*q| q.* = rand.int(u8);
        for (&blk.qs) |*q| q.* = rand.int(u8);
    }

    const host_x = try allocator.alloc(f32, in_dim);
    defer allocator.free(host_x);
    for (host_x) |*x| {
        x.* = (rand.float(f32) * 4.0) - 2.0;
    }

    const cpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(cpu_out);
    infer.gemvQ5_0(host_w.ptr, host_x, cpu_out, out_dim, in_dim);

    const raw_w_bytes = std.mem.sliceAsBytes(host_w);
    var d_w = try DeviceBuffer(u8).init(dev, raw_w_bytes.len);
    defer d_w.deinit();
    try d_w.copyFromHost(raw_w_bytes);

    var d_x = try DeviceBuffer(f32).init(dev, in_dim);
    defer d_x.deinit();
    try d_x.copyFromHost(host_x);

    var d_out = try DeviceBuffer(f32).init(dev, out_dim);
    defer d_out.deinit();

    try gemvQ5_0(d_w.slice(), d_x.slice(), d_out.slice(), out_dim, in_dim, 1, null);

    const gpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(gpu_out);
    try d_out.copyToHost(gpu_out);

    var max_diff: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_diff) max_diff = diff;
        try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
    }
    std.debug.print("[parity] gemvQ5_0 (32x256): max_diff = {e:.4}\n", .{max_diff});
}

test "CUDA GEMV batched execution parity" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x424154);
    const rand = prng.random();

    const out_dim: usize = 16;
    const in_dim: usize = 128;
    const batch_size: usize = 3;
    const blocks_per_row = in_dim / quant.QK8_0; // 4
    const total_blocks = out_dim * blocks_per_row;

    const host_w = try allocator.alloc(quant.BlockQ8_0, total_blocks);
    defer allocator.free(host_w);
    for (host_w) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.3) + 0.1);
        for (&blk.qs) |*q| q.* = rand.intRangeAtMost(i8, -100, 100);
    }

    const host_x = try allocator.alloc(f32, batch_size * in_dim);
    defer allocator.free(host_x);
    for (host_x) |*x| {
        x.* = (rand.float(f32) * 3.0) - 1.5;
    }

    const raw_w_bytes = std.mem.sliceAsBytes(host_w);
    var d_w = try DeviceBuffer(u8).init(dev, raw_w_bytes.len);
    defer d_w.deinit();
    try d_w.copyFromHost(raw_w_bytes);

    var d_x = try DeviceBuffer(f32).init(dev, host_x.len);
    defer d_x.deinit();
    try d_x.copyFromHost(host_x);

    var d_out = try DeviceBuffer(f32).init(dev, batch_size * out_dim);
    defer d_out.deinit();

    // Batched call
    try gemvQ8_0(d_w.slice(), d_x.slice(), d_out.slice(), out_dim, in_dim, batch_size, null);

    const gpu_batched_out = try allocator.alloc(f32, batch_size * out_dim);
    defer allocator.free(gpu_batched_out);
    try d_out.copyToHost(gpu_batched_out);

    // Verify each batch against single-vector CPU GEMV
    for (0..batch_size) |b| {
        const x_sub = host_x[b * in_dim .. (b + 1) * in_dim];
        var single_cpu_out: [16]f32 = undefined;
        infer.gemvQ8_0(host_w.ptr, x_sub, &single_cpu_out, out_dim, in_dim);

        const gpu_sub = gpu_batched_out[b * out_dim .. (b + 1) * out_dim];
        for (single_cpu_out, gpu_sub) |c, g| {
            const diff = @abs(c - g);
            try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
        }
    }
}

test "CUDA GEMV dimension validation and error conditions" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    var d_w = try DeviceBuffer(u8).init(dev, 256 * 34);
    defer d_w.deinit();
    var d_x = try DeviceBuffer(f32).init(dev, 256);
    defer d_x.deinit();
    var d_out = try DeviceBuffer(f32).init(dev, 32);
    defer d_out.deinit();

    // Q4_K requires in_dim % 256 == 0; 128 should error InvalidParam
    try std.testing.expectError(error.InvalidParam, gemvQ4_K(d_w.slice(), d_x.slice(), d_out.slice(), 32, 128, 1, null));

    // Q8_0 requires in_dim % 32 == 0; 30 should error InvalidParam
    try std.testing.expectError(error.InvalidParam, gemvQ8_0(d_w.slice(), d_x.slice(), d_out.slice(), 32, 30, 1, null));

    // Zero dimensions error
    try std.testing.expectError(error.InvalidParam, gemvQ8_0(d_w.slice(), d_x.slice(), d_out.slice(), 0, 32, 1, null));
    try std.testing.expectError(error.InvalidParam, gemvQ8_0(d_w.slice(), d_x.slice(), d_out.slice(), 32, 0, 1, null));
    try std.testing.expectError(error.InvalidParam, gemvQ8_0(d_w.slice(), d_x.slice(), d_out.slice(), 32, 32, 0, null));

    // Size mismatch (d_x.len is 256, but in_dim 32 * batch 1 expects len 32)
    try std.testing.expectError(error.InvalidParam, gemvQ8_0(d_w.slice(), d_x.slice(), d_out.slice(), 32, 32, 1, null));

    // R21-03: batch_size exceeding 65535 grid-Y limit should error InvalidParam
    try std.testing.expectError(error.InvalidParam, gemvQ8_0(d_w.slice(), d_x.slice(), d_out.slice(), 32, 32, 65536, null));
}

fn cpuReferenceAttention(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []f32,
    num_tokens: usize,
    prompt_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) void {
    const gqa_ratio = num_q_heads / num_kv_heads;
    const q_dim = num_q_heads * head_dim;
    const k_dim = num_kv_heads * head_dim;

    for (0..num_tokens) |t| {
        const is_prompt_query = (t < prompt_len);
        const visible_len = if (is_prompt_query) t + 1 else num_tokens;

        for (0..num_q_heads) |h| {
            const kv = h / gqa_ratio;
            const q_head = q[t * q_dim + h * head_dim .. t * q_dim + (h + 1) * head_dim];

            var scores: [512]f32 = undefined;
            for (0..visible_len) |s| {
                const k_head = k[s * k_dim + kv * head_dim .. s * k_dim + (kv + 1) * head_dim];
                var dot: f32 = 0.0;
                for (q_head, k_head) |qh, kh| {
                    dot += qh * kh;
                }
                scores[s] = dot;
            }
            if (visible_len < num_tokens) {
                @memset(scores[visible_len..num_tokens], -1e30);
            }

            var weights: [512]f32 = undefined;
            ops.softmax(scores[0..num_tokens], 1.0, weights[0..num_tokens]);

            const out_head = out[t * q_dim + h * head_dim .. t * q_dim + (h + 1) * head_dim];
            @memset(out_head, 0.0);
            for (0..visible_len) |s| {
                const w = weights[s];
                if (w == 0.0) continue;
                const v_head = v[s * k_dim + kv * head_dim .. s * k_dim + (kv + 1) * head_dim];
                for (out_head, v_head) |*oh, vh| {
                    oh.* += w * vh;
                }
            }
        }
    }
}

test "CUDA attention parity across SWA and Global heads" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1337_5432);
    const rand = prng.random();

    // 1. SWA configuration: num_q_heads=16, num_kv_heads=8, head_dim=256
    {
        const num_tokens: usize = 4;
        const prompt_len: usize = 2;
        const num_q_heads: usize = 16;
        const num_kv_heads: usize = 8;
        const head_dim: usize = 256;

        const q_dim = num_q_heads * head_dim;
        const k_dim = num_kv_heads * head_dim;

        const host_q = try allocator.alloc(f32, num_tokens * q_dim);
        defer allocator.free(host_q);
        const host_k = try allocator.alloc(f32, num_tokens * k_dim);
        defer allocator.free(host_k);
        const host_v = try allocator.alloc(f32, num_tokens * k_dim);
        defer allocator.free(host_v);

        for (host_q) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        for (host_k) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        for (host_v) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

        const cpu_out = try allocator.alloc(f32, num_tokens * q_dim);
        defer allocator.free(cpu_out);
        cpuReferenceAttention(host_q, host_k, host_v, cpu_out, num_tokens, prompt_len, num_q_heads, num_kv_heads, head_dim);

        var d_q = try DeviceBuffer(f32).init(dev, host_q.len);
        defer d_q.deinit();
        try d_q.copyFromHost(host_q);

        var d_k = try DeviceBuffer(f32).init(dev, host_k.len);
        defer d_k.deinit();
        try d_k.copyFromHost(host_k);

        var d_v = try DeviceBuffer(f32).init(dev, host_v.len);
        defer d_v.deinit();
        try d_v.copyFromHost(host_v);

        var d_out = try DeviceBuffer(f32).init(dev, num_tokens * q_dim);
        defer d_out.deinit();

        try attention(
            d_q.slice(),
            d_k.slice(),
            d_v.slice(),
            d_out.slice(),
            num_tokens,
            prompt_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_dim,
            k_dim,
            k_dim,
            q_dim,
            null,
        );

        const gpu_out = try allocator.alloc(f32, num_tokens * q_dim);
        defer allocator.free(gpu_out);
        try d_out.copyToHost(gpu_out);

        var max_diff: f32 = 0.0;
        for (cpu_out, gpu_out) |c, g| {
            const diff = @abs(c - g);
            if (diff > max_diff) max_diff = diff;
            try std.testing.expect(diff <= 5e-4 + 5e-4 * @abs(c));
        }
        std.debug.print("\n[parity] SWA Attention (16x8, d=256): max_diff = {e:.4}\n", .{max_diff});
    }

    // 2. Global configuration: num_q_heads=16, num_kv_heads=2, head_dim=512
    {
        const num_tokens: usize = 3;
        const prompt_len: usize = 1;
        const num_q_heads: usize = 16;
        const num_kv_heads: usize = 2;
        const head_dim: usize = 512;

        const q_dim = num_q_heads * head_dim;
        const k_dim = num_kv_heads * head_dim;

        const host_q = try allocator.alloc(f32, num_tokens * q_dim);
        defer allocator.free(host_q);
        const host_k = try allocator.alloc(f32, num_tokens * k_dim);
        defer allocator.free(host_k);
        const host_v = try allocator.alloc(f32, num_tokens * k_dim);
        defer allocator.free(host_v);

        for (host_q) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        for (host_k) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        for (host_v) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

        const cpu_out = try allocator.alloc(f32, num_tokens * q_dim);
        defer allocator.free(cpu_out);
        cpuReferenceAttention(host_q, host_k, host_v, cpu_out, num_tokens, prompt_len, num_q_heads, num_kv_heads, head_dim);

        var d_q = try DeviceBuffer(f32).init(dev, host_q.len);
        defer d_q.deinit();
        try d_q.copyFromHost(host_q);

        var d_k = try DeviceBuffer(f32).init(dev, host_k.len);
        defer d_k.deinit();
        try d_k.copyFromHost(host_k);

        var d_v = try DeviceBuffer(f32).init(dev, host_v.len);
        defer d_v.deinit();
        try d_v.copyFromHost(host_v);

        var d_out = try DeviceBuffer(f32).init(dev, num_tokens * q_dim);
        defer d_out.deinit();

        try attention(
            d_q.slice(),
            d_k.slice(),
            d_v.slice(),
            d_out.slice(),
            num_tokens,
            prompt_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_dim,
            k_dim,
            k_dim,
            q_dim,
            null,
        );

        const gpu_out = try allocator.alloc(f32, num_tokens * q_dim);
        defer allocator.free(gpu_out);
        try d_out.copyToHost(gpu_out);

        var max_diff: f32 = 0.0;
        for (cpu_out, gpu_out) |c, g| {
            const diff = @abs(c - g);
            if (diff > max_diff) max_diff = diff;
            try std.testing.expect(diff <= 5e-4 + 5e-4 * @abs(c));
        }
        std.debug.print("[parity] Global Attention (16x2, d=512): max_diff = {e:.4}\n", .{max_diff});
    }
}

test "CUDA MoE router top-8 selection and softmax parity" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9988_7766);
    const rand = prng.random();

    const num_tokens: usize = 4;
    const host_logits = try allocator.alloc(f32, num_tokens * 128);
    defer allocator.free(host_logits);
    for (host_logits) |*x| x.* = (rand.float(f32) * 20.0) - 10.0;

    const host_down_scales = try allocator.alloc(f32, 128);
    defer allocator.free(host_down_scales);
    for (host_down_scales) |*s| s.* = rand.float(f32) * 1.5 + 0.5;

    // CPU Reference top-8 and softmax
    const cpu_indices = try allocator.alloc(usize, num_tokens * 8);
    defer allocator.free(cpu_indices);
    const cpu_weights = try allocator.alloc(f32, num_tokens * 8);
    defer allocator.free(cpu_weights);

    for (0..num_tokens) |t| {
        const tok_logits = host_logits[t * 128 .. (t + 1) * 128];
        var top_idx: [8]usize = undefined;
        var top_w: [8]f32 = undefined;
        infer.topKExperts(tok_logits, 8, &top_idx, &top_w);

        for (0..8) |e| {
            top_w[e] *= host_down_scales[top_idx[e]];
            cpu_indices[t * 8 + e] = top_idx[e];
            cpu_weights[t * 8 + e] = top_w[e];
        }
    }

    var d_logits = try DeviceBuffer(f32).init(dev, host_logits.len);
    defer d_logits.deinit();
    try d_logits.copyFromHost(host_logits);

    var d_down_scales = try DeviceBuffer(f32).init(dev, host_down_scales.len);
    defer d_down_scales.deinit();
    try d_down_scales.copyFromHost(host_down_scales);

    var d_indices = try DeviceBuffer(i32).init(dev, num_tokens * 8);
    defer d_indices.deinit();

    var d_weights = try DeviceBuffer(f32).init(dev, num_tokens * 8);
    defer d_weights.deinit();

    try moeRouterTop8(
        d_logits.slice(),
        d_down_scales.slice(),
        d_indices.slice(),
        d_weights.slice(),
        num_tokens,
        null,
    );

    const gpu_indices = try allocator.alloc(i32, num_tokens * 8);
    defer allocator.free(gpu_indices);
    try d_indices.copyToHost(gpu_indices);

    const gpu_weights = try allocator.alloc(f32, num_tokens * 8);
    defer allocator.free(gpu_weights);
    try d_weights.copyToHost(gpu_weights);

    var max_weight_diff: f32 = 0.0;
    for (0..num_tokens * 8) |i| {
        // Exact top-8 index agreement
        try std.testing.expectEqual(@as(i32, @intCast(cpu_indices[i])), gpu_indices[i]);
        const w_diff = @abs(cpu_weights[i] - gpu_weights[i]);
        if (w_diff > max_weight_diff) max_weight_diff = w_diff;
        try std.testing.expect(w_diff <= 1e-5);
    }
    std.debug.print("[parity] MoE Router Top-8 Selection & Softmax: max_diff = {e:.4}\n", .{max_weight_diff});
}

test "CUDA MoE router prep and vector accumulation parity" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1122_3344);
    const rand = prng.random();

    const num_tokens: usize = 2;
    const dim: usize = 2816;
    const root_size: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));

    const host_r_in = try allocator.alloc(f32, num_tokens * dim);
    defer allocator.free(host_r_in);
    for (host_r_in) |*x| x.* = (rand.float(f32) * 4.0) - 2.0;

    const host_scale = try allocator.alloc(f32, dim);
    defer allocator.free(host_scale);
    for (host_scale) |*x| x.* = rand.float(f32) * 1.5 + 0.2;

    const cpu_prep = try allocator.alloc(f32, num_tokens * dim);
    defer allocator.free(cpu_prep);
    @memcpy(cpu_prep, host_r_in);
    for (0..num_tokens) |t| {
        for (0..dim) |d| {
            cpu_prep[t * dim + d] *= root_size * host_scale[d];
        }
    }

    var d_r_in = try DeviceBuffer(f32).init(dev, host_r_in.len);
    defer d_r_in.deinit();
    try d_r_in.copyFromHost(host_r_in);

    var d_scale = try DeviceBuffer(f32).init(dev, host_scale.len);
    defer d_scale.deinit();
    try d_scale.copyFromHost(host_scale);

    try moeRouterPrep(d_r_in.slice(), d_scale.slice(), root_size, num_tokens, dim, null);

    const gpu_prep = try allocator.alloc(f32, num_tokens * dim);
    defer allocator.free(gpu_prep);
    try d_r_in.copyToHost(gpu_prep);

    var max_prep_diff: f32 = 0.0;
    for (cpu_prep, gpu_prep) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_prep_diff) max_prep_diff = diff;
        try std.testing.expect(diff <= 1e-6);
    }
    std.debug.print("[parity] MoE Router Prep: max_diff = {e:.4}\n", .{max_prep_diff});

    // Test vector accumulate
    const host_accum = try allocator.alloc(f32, dim);
    defer allocator.free(host_accum);
    for (host_accum) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

    const host_x = try allocator.alloc(f32, dim);
    defer allocator.free(host_x);
    for (host_x) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

    const factor: f32 = 0.42;

    const cpu_accum = try allocator.alloc(f32, dim);
    defer allocator.free(cpu_accum);
    for (cpu_accum, host_accum, host_x) |*ca, ha, hx| {
        ca.* = ha + hx * factor;
    }

    var d_accum = try DeviceBuffer(f32).init(dev, dim);
    defer d_accum.deinit();
    try d_accum.copyFromHost(host_accum);

    var d_x = try DeviceBuffer(f32).init(dev, dim);
    defer d_x.deinit();
    try d_x.copyFromHost(host_x);

    try accumulateScaled(d_accum.slice(), d_x.slice(), factor, null);

    const gpu_accum = try allocator.alloc(f32, dim);
    defer allocator.free(gpu_accum);
    try d_accum.copyToHost(gpu_accum);

    var max_acc_diff: f32 = 0.0;
    for (cpu_accum, gpu_accum) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_acc_diff) max_acc_diff = diff;
        try std.testing.expect(diff <= 1e-6);
    }
    std.debug.print("[parity] Accumulate Scaled: max_diff = {e:.4}\n", .{max_acc_diff});
}

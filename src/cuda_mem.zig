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

        /// Synchronous copy from host slice to device slice.
        pub fn copyFromHost(self: Self, host_slice: []const T) cuda.CudaError!void {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (host_slice.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const dst = self.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memcpy_h2d(dst, host_slice.ptr, bytes, self.device_id);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Synchronous copy from device slice to host slice.
        pub fn copyToHost(self: Self, host_slice: []T) cuda.CudaError!void {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (host_slice.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const src = self.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memcpy_d2h(host_slice.ptr, src, bytes, self.device_id);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Phase 14 WP2: Asynchronous copy from pinned host slice to device slice.
        pub fn copyFromHostAsync(self: Self, host_slice: []const T, stream: *const CudaStream) cuda.CudaError!void {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (host_slice.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const dst = self.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memcpy_h2d_async(dst, host_slice.ptr, bytes, self.device_id, stream.handle);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Phase 14 WP2: Asynchronous copy from device slice to pinned host slice.
        pub fn copyToHostAsync(self: Self, host_slice: []T, stream: *const CudaStream) cuda.CudaError!void {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (host_slice.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const src = self.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memcpy_d2h_async(host_slice.ptr, src, bytes, self.device_id, stream.handle);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Synchronous or stream-ordered copy from device slice to device slice.
        pub fn copyFromDevice(self: Self, src: DeviceSlice(T), stream: ?*const CudaStream) cuda.CudaError!void {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (self.device_id != src.device_id) return error.DeviceMismatch;
            if (src.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const dst = self.ptr orelse return error.InvalidParam;
            const src_ptr = src.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const s_handle = if (stream) |s| s.handle else null;
            const rc = cuda.c.zeno_cuda_memcpy_d2d(dst, src_ptr, bytes, self.device_id, s_handle);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Asynchronous copy from peer device memory slice using P2P transfer if supported.
        pub fn copyFromPeerAsync(self: Self, src: DeviceSlice(T), stream: *const CudaStream) cuda.CudaError!void {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (src.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const dst = self.ptr orelse return error.InvalidParam;
            const src_ptr = src.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memcpy_peer_async(dst, self.device_id, src_ptr, src.device_id, bytes, stream.handle);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
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

    /// Idempotent stream destruction. Clears handle unconditionally and logs/asserts if destruction fails (R29-06).
    pub fn deinit(self: *CudaStream) void {
        if (!cuda.is_enabled) return;
        if (self.handle) |h| {
            self.handle = null;
            const rc = cuda.c.zeno_cuda_stream_destroy(h, self.device_id);
            if (rc != 0) {
                std.log.err("zeno_cuda_stream_destroy failed for device {d} with rc={d}", .{ self.device_id, rc });
                std.debug.assert(rc == 0);
            }
        }
    }

    pub fn synchronize(self: *const CudaStream) cuda.CudaError!void {
        if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
        const rc = cuda.c.zeno_cuda_stream_sync(self.handle, self.device_id);
        if (rc != 0) return cuda.mapCudaErrorCode(rc);
    }

    pub fn waitEvent(self: *const CudaStream, event: *const CudaEvent) cuda.CudaError!void {
        if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
        const sh = self.handle orelse return error.InvalidParam;
        const eh = event.handle orelse return error.InvalidParam;
        const rc = cuda.c.zeno_cuda_stream_wait_event(sh, eh, self.device_id);
        if (rc != 0) return cuda.mapCudaErrorCode(rc);
    }
};

/// Explicit single-owner CUDA event wrapper for cross-stream coordination.
pub const CudaEvent = struct {
    handle: ?*anyopaque,
    device_id: i32,

    pub fn init(device_id: i32) cuda.CudaError!CudaEvent {
        if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
        var raw_event: ?*anyopaque = null;
        const rc = cuda.c.zeno_cuda_event_create(&raw_event, device_id);
        if (rc != 0) return cuda.mapCudaErrorCode(rc);
        return .{
            .handle = raw_event,
            .device_id = device_id,
        };
    }

    pub fn deinit(self: *CudaEvent) void {
        if (!cuda.is_enabled) return;
        if (self.handle) |h| {
            _ = cuda.c.zeno_cuda_event_destroy(h, self.device_id);
            self.handle = null;
        }
    }

    pub fn record(self: *const CudaEvent, stream: *const CudaStream) cuda.CudaError!void {
        if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
        const eh = self.handle orelse return error.InvalidParam;
        const sh = stream.handle orelse return error.InvalidParam;
        const rc = cuda.c.zeno_cuda_event_record(eh, sh, self.device_id);
        if (rc != 0) return cuda.mapCudaErrorCode(rc);
    }
};

/// Explicit single-owner CUDA Graph wrapper for capture, instantiation, and zero-overhead replay.
pub const CudaGraph = struct {
    exec_handle: ?*anyopaque,
    device_id: i32,

    pub fn beginCapture(stream: *const CudaStream) cuda.CudaError!void {
        if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
        const sh = stream.handle orelse return error.InvalidParam;
        const rc = cuda.c.zeno_cuda_graph_begin_capture(sh, stream.device_id);
        if (rc != 0) return cuda.mapCudaErrorCode(rc);
    }

    pub fn endCapture(stream: *const CudaStream) cuda.CudaError!CudaGraph {
        if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
        const sh = stream.handle orelse return error.InvalidParam;
        var exec_handle: ?*anyopaque = null;
        const rc = cuda.c.zeno_cuda_graph_end_capture(sh, &exec_handle, stream.device_id);
        if (rc != 0) return cuda.mapCudaErrorCode(rc);
        return CudaGraph{
            .exec_handle = exec_handle,
            .device_id = stream.device_id,
        };
    }

    pub fn launch(self: *const CudaGraph, stream: *const CudaStream) cuda.CudaError!void {
        if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
        const eh = self.exec_handle orelse return error.InvalidParam;
        const sh = stream.handle orelse return error.InvalidParam;
        const rc = cuda.c.zeno_cuda_graph_launch(eh, sh, self.device_id);
        if (rc != 0) return cuda.mapCudaErrorCode(rc);
    }

    pub fn deinit(self: *CudaGraph) void {
        if (!cuda.is_enabled) return;
        if (self.exec_handle) |h| {
            _ = cuda.c.zeno_cuda_graph_destroy(h, self.device_id);
            self.exec_handle = null;
        }
    }

    /// Safely terminates an in-flight stream capture and releases any created graph handle,
    /// restoring the stream to an uncaptured, operational state.
    pub fn cancelCapture(stream: *const CudaStream) void {
        if (!cuda.is_enabled) return;
        if (endCapture(stream)) |mut_g| {
            var g = mut_g;
            g.deinit();
        } else |_| {}
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

        /// Idempotent device buffer deallocation. Clears pointer unconditionally and logs/asserts if free fails (R29-06).
        pub fn deinit(self: *Self) void {
            if (!cuda.is_enabled) return;
            if (self.ptr) |p| {
                self.ptr = null;
                self.len = 0;
                const rc = cuda.c.zeno_cuda_free(p, self.device_id);
                if (rc != 0) {
                    std.log.err("zeno_cuda_free failed for device {d} with rc={d}", .{ self.device_id, rc });
                    std.debug.assert(rc == 0);
                }
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
            const rc = cuda.c.zeno_cuda_memcpy_d2d(dst, src_ptr, bytes, self.device_id, null);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Copies from peer device memory slice using P2P transfer if supported.
        pub fn copyFromPeer(self: *Self, src: DeviceSlice(T)) cuda.CudaError!void {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (src.len != self.len) return error.InvalidParam;
            if (self.len == 0) return;
            const dst = self.ptr orelse return error.InvalidParam;
            const src_ptr = src.ptr orelse return error.InvalidParam;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memcpy_peer(dst, self.device_id, src_ptr, src.device_id, bytes);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);
        }

        /// Sets all elements of the device buffer to zero.
        pub fn setZero(self: *const Self) cuda.CudaError!void {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (self.len == 0) return;
            const p = self.ptr orelse return;
            const bytes = std.math.mul(usize, self.len, @sizeOf(T)) catch return error.InvalidParam;
            const rc = cuda.c.zeno_cuda_memset(p, 0, bytes, self.device_id);
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

/// Phase 14 WP2: Page-locked (pinned) host memory buffer allocated via cudaHostAllocPortable.
/// Enables zero-copy hardware DMA engine transfers over PCIe Gen4 without CPU bounce buffer overhead.
pub fn PinnedBuffer(comptime T: type) type {
    comptime {
        const valid = switch (T) {
            f32, f16, i32, u32, u8 => true,
            else => false,
        };
        if (!valid or @sizeOf(T) == 0) {
            @compileError("PinnedBuffer only supports non-zero-sized POD numeric types (f32, f16, i32, u32, u8)");
        }
    }

    return struct {
        const Self = @This();

        pub const MAX_ALLOC_BYTES: usize = 16 * 1024 * 1024 * 1024;

        ptr: ?[*]T,
        len: usize,

        pub fn init(count: usize) cuda.CudaError!Self {
            if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
            if (count == 0) {
                return .{
                    .ptr = null,
                    .len = 0,
                };
            }

            const bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.InvalidParam;
            if (bytes > MAX_ALLOC_BYTES) return error.InvalidParam;

            var raw_ptr: ?*anyopaque = null;
            const rc = cuda.c.zeno_cuda_host_alloc(&raw_ptr, bytes);
            if (rc != 0) return cuda.mapCudaErrorCode(rc);

            return .{
                .ptr = @ptrCast(@alignCast(raw_ptr.?)),
                .len = count,
            };
        }

        pub fn move(self: *Self) Self {
            const moved = Self{
                .ptr = self.ptr,
                .len = self.len,
            };
            self.ptr = null;
            self.len = 0;
            return moved;
        }

        pub fn deinit(self: *Self) void {
            if (!cuda.is_enabled) return;
            if (self.ptr) |p| {
                self.ptr = null;
                self.len = 0;
                const rc = cuda.c.zeno_cuda_host_free(p);
                if (rc != 0) {
                    std.log.err("zeno_cuda_host_free failed with rc={d}", .{rc});
                    std.debug.assert(rc == 0);
                }
            }
        }

        pub fn slice(self: Self) []T {
            if (self.len == 0 or self.ptr == null) return &[_]T{};
            return self.ptr.?[0..self.len];
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

/// Fused Attention Q+K (and optional tied V) GEMV Projection (WP1 / WP3):
/// Computes out_q = W_q * x and out_k = W_k * x in a single unified grid pass.
/// If out_v is provided (for Global attention layers where V is tied to K prior to norm/RoPE),
/// out_v also receives the unnormalized K values directly from registers.
pub fn gemvQ4_K_FusedQK(
    w_q: DeviceSlice(u8),
    w_k: DeviceSlice(u8),
    x: DeviceSlice(f32),
    out_q: DeviceSlice(f32),
    out_k: DeviceSlice(f32),
    out_v: ?DeviceSlice(f32),
    q_dim: usize,
    k_dim: usize,
    in_dim: usize,
    batch_size: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (q_dim == 0 or k_dim == 0 or in_dim == 0 or batch_size == 0) return error.InvalidParam;
    if (in_dim % quant.QK_K != 0) return error.InvalidParam;
    if (q_dim > std.math.maxInt(c_int) or k_dim > std.math.maxInt(c_int) or in_dim > std.math.maxInt(c_int) or batch_size > 65535) {
        return error.InvalidParam;
    }

    const blocks_per_row = in_dim / quant.QK_K;
    const bytes_per_row = blocks_per_row * @sizeOf(quant.BlockQ4_K);
    const expected_wq_bytes = std.math.mul(usize, q_dim, bytes_per_row) catch return error.InvalidParam;
    const expected_wk_bytes = std.math.mul(usize, k_dim, bytes_per_row) catch return error.InvalidParam;
    const expected_x_len = std.math.mul(usize, batch_size, in_dim) catch return error.InvalidParam;
    const expected_outq_len = std.math.mul(usize, batch_size, q_dim) catch return error.InvalidParam;
    const expected_outk_len = std.math.mul(usize, batch_size, k_dim) catch return error.InvalidParam;

    if (w_q.len != expected_wq_bytes or w_k.len != expected_wk_bytes or x.len != expected_x_len or out_q.len != expected_outq_len or out_k.len != expected_outk_len) {
        return error.InvalidParam;
    }
    if (w_q.device_id != w_k.device_id or w_q.device_id != x.device_id or w_q.device_id != out_q.device_id or w_q.device_id != out_k.device_id) {
        return error.DeviceMismatch;
    }
    if (out_v) |ov| {
        if (ov.len != expected_outk_len) return error.InvalidParam;
        if (ov.device_id != w_q.device_id) return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != w_q.device_id) return error.DeviceMismatch;
    }

    const w_q_ptr = w_q.ptr orelse return error.InvalidParam;
    const w_k_ptr = w_k.ptr orelse return error.InvalidParam;
    const x_ptr = x.ptr orelse return error.InvalidParam;
    const out_q_ptr = out_q.ptr orelse return error.InvalidParam;
    const out_k_ptr = out_k.ptr orelse return error.InvalidParam;
    const out_v_ptr = if (out_v) |ov| (ov.ptr orelse return error.InvalidParam) else null;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_gemv_q4k_fused_qk(
        w_q_ptr,
        w_k_ptr,
        x_ptr,
        out_q_ptr,
        out_k_ptr,
        out_v_ptr,
        @intCast(q_dim),
        @intCast(k_dim),
        @intCast(in_dim),
        @intCast(batch_size),
        w_q.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Fused dense FFN Gate + Up projections and GeGLU activation:
/// out_geglu = GeGLU(W_gate * x, W_up * x)
/// W_gate, W_up: DeviceSlice(u8) representing row-major BlockQ4_K blocks of shape [out_dim, in_dim]
/// x: DeviceSlice(f32) of length batch_size * in_dim
/// out_geglu: DeviceSlice(f32) of length batch_size * out_dim
pub fn denseGateUpGegluQ4_K(
    w_gate: DeviceSlice(u8),
    w_up: DeviceSlice(u8),
    x: DeviceSlice(f32),
    out_geglu: DeviceSlice(f32),
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

    if (w_gate.len != expected_w_bytes or w_up.len != expected_w_bytes or x.len != expected_x_len or out_geglu.len != expected_out_len) {
        return error.InvalidParam;
    }
    if (w_gate.device_id != x.device_id or w_gate.device_id != w_up.device_id or w_gate.device_id != out_geglu.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != w_gate.device_id) return error.DeviceMismatch;
    }

    const w_gate_ptr = w_gate.ptr orelse return error.InvalidParam;
    const w_up_ptr = w_up.ptr orelse return error.InvalidParam;
    const x_ptr = x.ptr orelse return error.InvalidParam;
    const out_ptr = out_geglu.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_dense_gate_up_geglu_q4k(
        w_gate_ptr,
        w_up_ptr,
        x_ptr,
        out_ptr,
        @intCast(out_dim),
        @intCast(in_dim),
        @intCast(batch_size),
        w_gate.device_id,
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

/// Matrix-vector multiplication for F32 weights: out = W * x
/// w: DeviceSlice(f32) of length out_dim * in_dim (row-major)
/// x: DeviceSlice(f32) of length batch_size * in_dim
/// out: DeviceSlice(f32) of length batch_size * out_dim
pub fn gemvF32(
    w: DeviceSlice(f32),
    x: DeviceSlice(f32),
    out: DeviceSlice(f32),
    out_dim: usize,
    in_dim: usize,
    batch_size: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (out_dim == 0 or in_dim == 0 or batch_size == 0) return error.InvalidParam;
    if (out_dim > std.math.maxInt(c_int) or in_dim > std.math.maxInt(c_int) or batch_size > 65535) {
        return error.InvalidParam;
    }

    const expected_w_len = std.math.mul(usize, out_dim, in_dim) catch return error.InvalidParam;
    const expected_x_len = std.math.mul(usize, batch_size, in_dim) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, batch_size, out_dim) catch return error.InvalidParam;

    if (w.len != expected_w_len or x.len != expected_x_len or out.len != expected_out_len) {
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

    const rc = cuda.c.zeno_cuda_gemv_f32(
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

/// Unified multi-head attention on GPU with cached prompt KV (Phase 7B):
/// Q: [canvas_len, stride_q]
/// prompt_k: ?DeviceSlice(f32) [prompt_len * stride_k] (optional if prompt_len == 0)
/// prompt_v: ?DeviceSlice(f32) [prompt_len * stride_v] (optional if prompt_len == 0)
/// canvas_k: [canvas_len, stride_k]
/// canvas_v: [canvas_len, stride_v]
/// out: [canvas_len, stride_out]
pub fn attentionCached(
    q: DeviceSlice(f32),
    prompt_k: ?DeviceSlice(f32),
    prompt_v: ?DeviceSlice(f32),
    canvas_k: DeviceSlice(f32),
    canvas_v: DeviceSlice(f32),
    out: DeviceSlice(f32),
    prompt_len: usize,
    canvas_len: usize,
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
    if (canvas_len == 0) return error.InvalidParam;
    if (prompt_len + canvas_len > 512) return error.InvalidParam;
    if (num_q_heads == 0 or num_kv_heads == 0) return error.InvalidParam;
    if (num_q_heads % num_kv_heads != 0) return error.InvalidParam;
    if (head_dim != 256 and head_dim != 512) return error.InvalidParam;
    if (stride_q < num_q_heads * head_dim or
        stride_k < num_kv_heads * head_dim or
        stride_v < num_kv_heads * head_dim or
        stride_out < num_q_heads * head_dim)
    {
        return error.InvalidParam;
    }

    const dev = q.device_id;
    if (canvas_k.device_id != dev or canvas_v.device_id != dev or out.device_id != dev) {
        return error.DeviceMismatch;
    }
    if (prompt_len > 0) {
        const pk = prompt_k orelse return error.InvalidParam;
        const pv = prompt_v orelse return error.InvalidParam;
        if (pk.device_id != dev or pv.device_id != dev) return error.DeviceMismatch;
        const min_pk_len = std.math.mul(usize, prompt_len, stride_k) catch return error.InvalidParam;
        const min_pv_len = std.math.mul(usize, prompt_len, stride_v) catch return error.InvalidParam;
        if (pk.len < min_pk_len or pv.len < min_pv_len) return error.InvalidParam;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }

    const min_q_len = std.math.mul(usize, canvas_len, stride_q) catch return error.InvalidParam;
    const min_ck_len = std.math.mul(usize, canvas_len, stride_k) catch return error.InvalidParam;
    const min_cv_len = std.math.mul(usize, canvas_len, stride_v) catch return error.InvalidParam;
    const min_out_len = std.math.mul(usize, canvas_len, stride_out) catch return error.InvalidParam;

    if (q.len < min_q_len or canvas_k.len < min_ck_len or canvas_v.len < min_cv_len or out.len < min_out_len) {
        return error.InvalidParam;
    }

    const q_ptr = q.ptr orelse return error.InvalidParam;
    const pk_ptr = if (prompt_k) |pk| pk.ptr else null;
    const pv_ptr = if (prompt_v) |pv| pv.ptr else null;
    const ck_ptr = canvas_k.ptr orelse return error.InvalidParam;
    const cv_ptr = canvas_v.ptr orelse return error.InvalidParam;
    const out_ptr = out.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_attention_cached(
        q_ptr,
        pk_ptr,
        pv_ptr,
        ck_ptr,
        cv_ptr,
        out_ptr,
        @intCast(prompt_len),
        @intCast(canvas_len),
        @intCast(num_q_heads),
        @intCast(num_kv_heads),
        @intCast(head_dim),
        @intCast(stride_q),
        @intCast(stride_k),
        @intCast(stride_v),
        @intCast(stride_out),
        dev,
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

/// Dual-region transformer layer output scaling: out[t, d] = (x_mid[t, d] + post_ffn[t, d]) * (t < prompt_len ? enc_scale : layer_scale)
pub fn layerOutput(
    out: DeviceSlice(f32),
    x_mid: DeviceSlice(f32),
    post_ffn: DeviceSlice(f32),
    prompt_len: usize,
    num_tokens: usize,
    dim: usize,
    enc_scale: f32,
    layer_scale: f32,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (out.device_id != x_mid.device_id or out.device_id != post_ffn.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != out.device_id) return error.DeviceMismatch;
    }
    const expected_len = std.math.mul(usize, num_tokens, dim) catch return error.InvalidParam;
    if (out.len != expected_len or x_mid.len != expected_len or post_ffn.len != expected_len) return error.InvalidParam;
    if (prompt_len > num_tokens) return error.InvalidParam;
    if (expected_len == 0) return;

    const out_ptr = out.ptr orelse return error.InvalidParam;
    const x_mid_ptr = x_mid.ptr orelse return error.InvalidParam;
    const post_ffn_ptr = post_ffn.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_layer_output(
        out_ptr,
        x_mid_ptr,
        post_ffn_ptr,
        @intCast(prompt_len),
        @intCast(num_tokens),
        @intCast(dim),
        enc_scale,
        layer_scale,
        out.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

// ---------------------------------------------------------------------------
// Phase 15: Intra-Layer Kernel Fusion Wrappers
// ---------------------------------------------------------------------------

pub fn fusedPreFfnNorm(
    x_mid: DeviceSlice(f32),
    ffn_norm: DeviceSlice(f32),
    pre_ffw_norm_2: DeviceSlice(f32),
    router_scale: DeviceSlice(f32),
    dense_in: DeviceSlice(f32),
    moe_in: DeviceSlice(f32),
    r_in: DeviceSlice(f32),
    root_size: f32,
    num_tokens: usize,
    dim: usize,
    eps: f32,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens > 65535 or dim < 1 or dim > 8192 or eps <= 0.0) return error.InvalidParam;
    const total = std.math.mul(usize, num_tokens, dim) catch return error.InvalidParam;
    if (x_mid.len != total or dense_in.len != total or moe_in.len != total or r_in.len != total) return error.InvalidParam;
    if (ffn_norm.len != dim or pre_ffw_norm_2.len != dim or router_scale.len != dim) return error.InvalidParam;
    const dev = x_mid.device_id;
    if (dense_in.device_id != dev or moe_in.device_id != dev or r_in.device_id != dev or
        ffn_norm.device_id != dev or pre_ffw_norm_2.device_id != dev or router_scale.device_id != dev)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    if (num_tokens == 0) return;

    const x_ptr = x_mid.ptr orelse return error.InvalidParam;
    const fn_ptr = ffn_norm.ptr orelse return error.InvalidParam;
    const p2_ptr = pre_ffw_norm_2.ptr orelse return error.InvalidParam;
    const rs_ptr = router_scale.ptr orelse return error.InvalidParam;
    const di_ptr = dense_in.ptr orelse return error.InvalidParam;
    const mi_ptr = moe_in.ptr orelse return error.InvalidParam;
    const ri_ptr = r_in.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_fused_pre_ffn_norm(
        x_ptr,
        fn_ptr,
        p2_ptr,
        rs_ptr,
        di_ptr,
        mi_ptr,
        ri_ptr,
        root_size,
        @intCast(num_tokens),
        @intCast(dim),
        eps,
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

pub fn fusedPostFfnCombine(
    moe_accum: DeviceSlice(f32),
    post_ffw_norm_2: DeviceSlice(f32),
    h1: DeviceSlice(f32),
    post_ffw_norm: DeviceSlice(f32),
    x_mid: DeviceSlice(f32),
    out: DeviceSlice(f32),
    prompt_len: usize,
    num_tokens: usize,
    dim: usize,
    enc_scale: f32,
    layer_scale: f32,
    eps: f32,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens > 65535 or dim < 1 or dim > 8192 or eps <= 0.0 or prompt_len > num_tokens) return error.InvalidParam;
    const total = std.math.mul(usize, num_tokens, dim) catch return error.InvalidParam;
    if (moe_accum.len != total or h1.len != total or x_mid.len != total or out.len != total) return error.InvalidParam;
    if (post_ffw_norm_2.len != dim or post_ffw_norm.len != dim) return error.InvalidParam;
    const dev = moe_accum.device_id;
    if (h1.device_id != dev or x_mid.device_id != dev or out.device_id != dev or
        post_ffw_norm_2.device_id != dev or post_ffw_norm.device_id != dev)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    if (num_tokens == 0) return;

    const ma_ptr = moe_accum.ptr orelse return error.InvalidParam;
    const p2_ptr = post_ffw_norm_2.ptr orelse return error.InvalidParam;
    const h1_ptr = h1.ptr orelse return error.InvalidParam;
    const pn_ptr = post_ffw_norm.ptr orelse return error.InvalidParam;
    const xm_ptr = x_mid.ptr orelse return error.InvalidParam;
    const out_ptr = out.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_fused_post_ffn_combine(
        ma_ptr,
        p2_ptr,
        h1_ptr,
        pn_ptr,
        xm_ptr,
        out_ptr,
        @intCast(prompt_len),
        @intCast(num_tokens),
        @intCast(dim),
        enc_scale,
        layer_scale,
        eps,
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

pub fn fusedHeadNormRope(
    vec: DeviceSlice(f32),
    weight: ?DeviceSlice(f32),
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    pos_offset: usize,
    theta: f32,
    freq_factors: ?DeviceSlice(f32),
    eps: f32,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens > 65535 or num_heads < 1 or num_heads > 64) return error.InvalidParam;
    if (head_dim < 1 or head_dim > 1024 or (head_dim % 2) != 0) return error.InvalidParam;
    if (pos_offset > std.math.maxInt(c_int) or theta <= 0.0 or eps <= 0.0) return error.InvalidParam;

    const total_heads = std.math.mul(usize, num_tokens, num_heads) catch return error.InvalidParam;
    const total_elems = std.math.mul(usize, total_heads, head_dim) catch return error.InvalidParam;
    if (vec.len != total_elems) return error.InvalidParam;
    const dev = vec.device_id;
    if (weight) |w| {
        if (w.len != head_dim) return error.InvalidParam;
        if (w.device_id != dev) return error.DeviceMismatch;
    }
    if (freq_factors) |ff| {
        if (ff.len < head_dim / 2) return error.InvalidParam;
        if (ff.device_id != dev) return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    if (total_elems == 0) return;

    const vec_ptr = vec.ptr orelse return error.InvalidParam;
    const w_ptr = if (weight) |w| (w.ptr orelse return error.InvalidParam) else null;
    const ff_ptr = if (freq_factors) |ff| (ff.ptr orelse return error.InvalidParam) else null;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_fused_head_norm_rope(
        vec_ptr,
        w_ptr,
        @intCast(num_tokens),
        @intCast(num_heads),
        @intCast(head_dim),
        @intCast(pos_offset),
        theta,
        ff_ptr,
        eps,
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Batched MoE expert gate/up projection on GPU: out_gate_up = all_experts[top_indices[k]] * tokens_in[k/8]
pub fn moeGemvGateUpBatched(
    all_expert_weights: DeviceSlice(u8),
    tokens_in: DeviceSlice(f32),
    top_indices: DeviceSlice(i32),
    out_gate_up: DeviceSlice(f32),
    num_tokens: usize,
    in_dim: usize,
    out_dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or in_dim == 0 or out_dim == 0) return error.InvalidParam;
    if (all_expert_weights.device_id != tokens_in.device_id or
        all_expert_weights.device_id != top_indices.device_id or
        all_expert_weights.device_id != out_gate_up.device_id)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != all_expert_weights.device_id) return error.DeviceMismatch;
    }
    const expected_in_len = std.math.mul(usize, num_tokens, in_dim) catch return error.InvalidParam;
    const expected_indices_len = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, expected_indices_len, out_dim) catch return error.InvalidParam;
    if (tokens_in.len < expected_in_len or top_indices.len < expected_indices_len or out_gate_up.len < expected_out_len) {
        return error.InvalidParam;
    }

    const w_ptr = all_expert_weights.ptr orelse return error.InvalidParam;
    const in_ptr = tokens_in.ptr orelse return error.InvalidParam;
    const idx_ptr = top_indices.ptr orelse return error.InvalidParam;
    const out_ptr = out_gate_up.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_gemv_gate_up_batched(
        w_ptr,
        in_ptr,
        @ptrCast(idx_ptr),
        out_ptr,
        @intCast(num_tokens),
        @intCast(in_dim),
        @intCast(out_dim),
        all_expert_weights.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Batched MoE GeGLU activation on GPU: act_out[k, i] = GELU(gate_up[k, i]) * gate_up[k, half_dim + i]
pub fn moeGegluBatched(
    gate_up: DeviceSlice(f32),
    act_out: DeviceSlice(f32),
    num_instances: usize,
    half_dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_instances == 0 or half_dim == 0) return error.InvalidParam;
    if (gate_up.device_id != act_out.device_id) return error.DeviceMismatch;
    if (stream) |s| {
        if (s.device_id != gate_up.device_id) return error.DeviceMismatch;
    }
    const expected_in_len = std.math.mul(usize, num_instances, half_dim * 2) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, num_instances, half_dim) catch return error.InvalidParam;
    if (gate_up.len < expected_in_len or act_out.len < expected_out_len) return error.InvalidParam;

    const in_ptr = gate_up.ptr orelse return error.InvalidParam;
    const out_ptr = act_out.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_geglu_batched(
        in_ptr,
        out_ptr,
        @intCast(num_instances),
        @intCast(half_dim),
        gate_up.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Batched MoE expert down projection on GPU: out_down = all_experts[top_indices[k]] * acts_in[k]
pub fn moeGemvDownBatched(
    all_expert_weights: DeviceSlice(u8),
    expert_type: u32,
    acts_in: DeviceSlice(f32),
    top_indices: DeviceSlice(i32),
    out_down: DeviceSlice(f32),
    num_tokens: usize,
    in_dim: usize,
    out_dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or in_dim == 0 or out_dim == 0) return error.InvalidParam;
    if (all_expert_weights.device_id != acts_in.device_id or
        all_expert_weights.device_id != top_indices.device_id or
        all_expert_weights.device_id != out_down.device_id)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != all_expert_weights.device_id) return error.DeviceMismatch;
    }
    const expected_indices_len = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_in_len = std.math.mul(usize, expected_indices_len, in_dim) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, expected_indices_len, out_dim) catch return error.InvalidParam;
    if (acts_in.len < expected_in_len or top_indices.len < expected_indices_len or out_down.len < expected_out_len) {
        return error.InvalidParam;
    }

    const w_ptr = all_expert_weights.ptr orelse return error.InvalidParam;
    const in_ptr = acts_in.ptr orelse return error.InvalidParam;
    const idx_ptr = top_indices.ptr orelse return error.InvalidParam;
    const out_ptr = out_down.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_gemv_down_batched(
        w_ptr,
        @intCast(expert_type),
        in_ptr,
        @ptrCast(idx_ptr),
        out_ptr,
        @intCast(num_tokens),
        @intCast(in_dim),
        @intCast(out_dim),
        all_expert_weights.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Batched MoE expert accumulation and scaling: moe_accum[t, d] = sum_{e=0..7} down_out[t*8+e, d] * top_weights[t*8+e]
pub fn moeAccumulateScaledBatched(
    moe_accum: DeviceSlice(f32),
    down_out: DeviceSlice(f32),
    top_weights: DeviceSlice(f32),
    num_tokens: usize,
    dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or dim == 0) return error.InvalidParam;
    if (moe_accum.device_id != down_out.device_id or moe_accum.device_id != top_weights.device_id) {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != moe_accum.device_id) return error.DeviceMismatch;
    }
    const expected_accum_len = std.math.mul(usize, num_tokens, dim) catch return error.InvalidParam;
    const expected_weights_len = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_down_len = std.math.mul(usize, expected_weights_len, dim) catch return error.InvalidParam;
    if (moe_accum.len < expected_accum_len or top_weights.len < expected_weights_len or down_out.len < expected_down_len) {
        return error.InvalidParam;
    }

    const accum_ptr = moe_accum.ptr orelse return error.InvalidParam;
    const down_ptr = down_out.ptr orelse return error.InvalidParam;
    const w_ptr = top_weights.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_accumulate_scaled_batched(
        accum_ptr,
        down_ptr,
        w_ptr,
        @intCast(num_tokens),
        @intCast(dim),
        moe_accum.device_id,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Stage 7A: Builds expert bins from top_indices:
/// - expert_counts: [128]i32 count of assigned instances per expert
/// - expert_offsets: [128]i32 start offset per expert in sorted_instances
/// - sorted_instances: [num_tokens * 8]i32 sorted mapping (sorted_idx -> orig_instance)
/// - instance_to_sorted: [num_tokens * 8]i32 reverse mapping (orig_instance -> sorted_idx)
pub fn moeBuildExpertBins(
    top_indices: DeviceSlice(i32),
    expert_counts: DeviceSlice(i32),
    expert_offsets: DeviceSlice(i32),
    sorted_instances: DeviceSlice(i32),
    instance_to_sorted: DeviceSlice(i32),
    active_experts: ?DeviceSlice(i32),
    num_active_experts: ?DeviceSlice(i32),
    num_tokens: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0) return error.InvalidParam;
    const num_instances = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    if (top_indices.len < num_instances or sorted_instances.len < num_instances or instance_to_sorted.len < num_instances) {
        return error.InvalidParam;
    }
    if (expert_counts.len < 128 or expert_offsets.len < 128) {
        return error.InvalidParam;
    }
    const dev = top_indices.device_id;
    if (expert_counts.device_id != dev or expert_offsets.device_id != dev or
        sorted_instances.device_id != dev or instance_to_sorted.device_id != dev)
    {
        return error.DeviceMismatch;
    }
    if (active_experts) |ae| {
        if (ae.device_id != dev or ae.len < 128) return error.InvalidParam;
    }
    if (num_active_experts) |nae| {
        if (nae.device_id != dev or nae.len < 1) return error.InvalidParam;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }

    const idx_ptr = top_indices.ptr orelse return error.InvalidParam;
    const cnt_ptr = expert_counts.ptr orelse return error.InvalidParam;
    const off_ptr = expert_offsets.ptr orelse return error.InvalidParam;
    const sort_ptr = sorted_instances.ptr orelse return error.InvalidParam;
    const rev_ptr = instance_to_sorted.ptr orelse return error.InvalidParam;
    const act_ptr = if (active_experts) |ae| ae.ptr else null;
    const nae_ptr = if (num_active_experts) |nae| nae.ptr else null;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_build_expert_bins(
        @ptrCast(idx_ptr),
        @ptrCast(cnt_ptr),
        @ptrCast(off_ptr),
        @ptrCast(sort_ptr),
        @ptrCast(rev_ptr),
        @ptrCast(act_ptr),
        @ptrCast(nae_ptr),
        @intCast(num_tokens),
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Stage 7A: Token-sorted expert gate/up projection on GPU:
/// For each active expert, streams weights once and projects against all assigned tokens in parallel.
pub fn moeGemvGateUpSorted(
    all_expert_weights: DeviceSlice(u8),
    tokens_in: DeviceSlice(f32),
    expert_counts: DeviceSlice(i32),
    expert_offsets: DeviceSlice(i32),
    sorted_instances: DeviceSlice(i32),
    out_gate_up: DeviceSlice(f32),
    num_tokens: usize,
    in_dim: usize,
    out_dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or in_dim == 0 or out_dim == 0) return error.InvalidParam;
    const dev = all_expert_weights.device_id;
    if (tokens_in.device_id != dev or expert_counts.device_id != dev or
        expert_offsets.device_id != dev or sorted_instances.device_id != dev or
        out_gate_up.device_id != dev)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    const num_instances = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_in_len = std.math.mul(usize, num_tokens, in_dim) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, num_instances, out_dim) catch return error.InvalidParam;
    if (tokens_in.len < expected_in_len or sorted_instances.len < num_instances or out_gate_up.len < expected_out_len) {
        return error.InvalidParam;
    }
    if (expert_counts.len < 128 or expert_offsets.len < 128) {
        return error.InvalidParam;
    }

    const w_ptr = all_expert_weights.ptr orelse return error.InvalidParam;
    const in_ptr = tokens_in.ptr orelse return error.InvalidParam;
    const cnt_ptr = expert_counts.ptr orelse return error.InvalidParam;
    const off_ptr = expert_offsets.ptr orelse return error.InvalidParam;
    const sort_ptr = sorted_instances.ptr orelse return error.InvalidParam;
    const out_ptr = out_gate_up.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_gemv_gate_up_sorted(
        w_ptr,
        in_ptr,
        @ptrCast(cnt_ptr),
        @ptrCast(off_ptr),
        @ptrCast(sort_ptr),
        out_ptr,
        @intCast(num_tokens),
        @intCast(in_dim),
        @intCast(out_dim),
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Phase 10: Fused Token-sorted expert gate/up projection + GeGLU activation on GPU (WP2):
/// Streams weights once, evaluates gate and up simultaneously, applies gelu_tanh(gate) * up in registers,
/// and writes directly to out_act (exp_act_buf), bypassing intermediate gate_up buffer and kernel launch.
pub fn moeGemvGateUpGegluSorted(
    all_expert_weights: DeviceSlice(u8),
    tokens_in: DeviceSlice(f32),
    expert_counts: DeviceSlice(i32),
    expert_offsets: DeviceSlice(i32),
    sorted_instances: DeviceSlice(i32),
    out_act: DeviceSlice(f32),
    num_tokens: usize,
    in_dim: usize,
    half_dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or in_dim == 0 or half_dim == 0) return error.InvalidParam;
    const dev = all_expert_weights.device_id;
    if (tokens_in.device_id != dev or expert_counts.device_id != dev or
        expert_offsets.device_id != dev or sorted_instances.device_id != dev or
        out_act.device_id != dev)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    const num_instances = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_in_len = std.math.mul(usize, num_tokens, in_dim) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, num_instances, half_dim) catch return error.InvalidParam;
    if (tokens_in.len < expected_in_len or sorted_instances.len < num_instances or out_act.len < expected_out_len) {
        return error.InvalidParam;
    }
    if (expert_counts.len < 128 or expert_offsets.len < 128) {
        return error.InvalidParam;
    }

    const w_ptr = all_expert_weights.ptr orelse return error.InvalidParam;
    const in_ptr = tokens_in.ptr orelse return error.InvalidParam;
    const cnt_ptr = expert_counts.ptr orelse return error.InvalidParam;
    const off_ptr = expert_offsets.ptr orelse return error.InvalidParam;
    const sort_ptr = sorted_instances.ptr orelse return error.InvalidParam;
    const out_ptr = out_act.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_gemv_gate_up_geglu_sorted(
        w_ptr,
        in_ptr,
        @ptrCast(cnt_ptr),
        @ptrCast(off_ptr),
        @ptrCast(sort_ptr),
        out_ptr,
        @intCast(num_tokens),
        @intCast(in_dim),
        @intCast(half_dim),
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Stage 7A: Token-sorted expert down projection on GPU:
/// For each active expert, streams weights once and down-projects all assigned tokens in parallel.
pub fn moeGemvDownSorted(
    all_expert_weights: DeviceSlice(u8),
    expert_type: u32,
    acts_in: DeviceSlice(f32),
    expert_counts: DeviceSlice(i32),
    expert_offsets: DeviceSlice(i32),
    sorted_instances: DeviceSlice(i32),
    out_down: DeviceSlice(f32),
    num_tokens: usize,
    in_dim: usize,
    out_dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or in_dim == 0 or out_dim == 0) return error.InvalidParam;
    const dev = all_expert_weights.device_id;
    if (acts_in.device_id != dev or expert_counts.device_id != dev or
        expert_offsets.device_id != dev or sorted_instances.device_id != dev or
        out_down.device_id != dev)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    const num_instances = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_in_len = std.math.mul(usize, num_instances, in_dim) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, num_instances, out_dim) catch return error.InvalidParam;
    if (acts_in.len < expected_in_len or sorted_instances.len < num_instances or out_down.len < expected_out_len) {
        return error.InvalidParam;
    }
    if (expert_counts.len < 128 or expert_offsets.len < 128) {
        return error.InvalidParam;
    }

    const w_ptr = all_expert_weights.ptr orelse return error.InvalidParam;
    const in_ptr = acts_in.ptr orelse return error.InvalidParam;
    const cnt_ptr = expert_counts.ptr orelse return error.InvalidParam;
    const off_ptr = expert_offsets.ptr orelse return error.InvalidParam;
    const sort_ptr = sorted_instances.ptr orelse return error.InvalidParam;
    const out_ptr = out_down.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_gemv_down_sorted(
        w_ptr,
        @intCast(expert_type),
        in_ptr,
        @ptrCast(cnt_ptr),
        @ptrCast(off_ptr),
        @ptrCast(sort_ptr),
        out_ptr,
        @intCast(num_tokens),
        @intCast(in_dim),
        @intCast(out_dim),
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Stage 7A: Token-sorted MoE expert accumulation and scaling:
/// moe_accum[t, d] = sum_{e=0..7} sorted_down_out[instance_to_sorted[t*8+e], d] * top_weights[t*8+e]
pub fn moeAccumulateScaledSorted(
    moe_accum: DeviceSlice(f32),
    sorted_down_out: DeviceSlice(f32),
    instance_to_sorted: DeviceSlice(i32),
    top_weights: DeviceSlice(f32),
    num_tokens: usize,
    dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or dim == 0) return error.InvalidParam;
    const dev = moe_accum.device_id;
    if (sorted_down_out.device_id != dev or instance_to_sorted.device_id != dev or top_weights.device_id != dev) {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    const num_instances = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_accum_len = std.math.mul(usize, num_tokens, dim) catch return error.InvalidParam;
    const expected_down_len = std.math.mul(usize, num_instances, dim) catch return error.InvalidParam;
    if (moe_accum.len < expected_accum_len or top_weights.len < num_instances or
        instance_to_sorted.len < num_instances or sorted_down_out.len < expected_down_len)
    {
        return error.InvalidParam;
    }

    const accum_ptr = moe_accum.ptr orelse return error.InvalidParam;
    const down_ptr = sorted_down_out.ptr orelse return error.InvalidParam;
    const rev_ptr = instance_to_sorted.ptr orelse return error.InvalidParam;
    const w_ptr = top_weights.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_accumulate_scaled_sorted(
        accum_ptr,
        down_ptr,
        @ptrCast(rev_ptr),
        w_ptr,
        @intCast(num_tokens),
        @intCast(dim),
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Phase 13: Fused MoE down-projection and scaled accumulation directly into moe_accum.
/// Eliminates intermediate exp_down_buf (saving ~78.6 MB DRAM writes/reads per canvas step)
/// and saves 30 kernel launch dispatches per step.
pub fn moeGemvDownAccumSorted(
    all_expert_weights: DeviceSlice(u8),
    expert_type: u32,
    acts_in: DeviceSlice(f32),
    expert_counts: DeviceSlice(i32),
    expert_offsets: DeviceSlice(i32),
    sorted_instances: DeviceSlice(i32),
    top_weights: DeviceSlice(f32),
    moe_accum: DeviceSlice(f32),
    num_tokens: usize,
    in_dim: usize,
    out_dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or in_dim == 0 or out_dim == 0) return error.InvalidParam;
    const dev = all_expert_weights.device_id;
    if (acts_in.device_id != dev or expert_counts.device_id != dev or
        expert_offsets.device_id != dev or sorted_instances.device_id != dev or
        top_weights.device_id != dev or moe_accum.device_id != dev)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    const num_instances = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_in_len = std.math.mul(usize, num_instances, in_dim) catch return error.InvalidParam;
    const expected_accum_len = std.math.mul(usize, num_tokens, out_dim) catch return error.InvalidParam;
    if (acts_in.len < expected_in_len or sorted_instances.len < num_instances or
        top_weights.len < num_instances or moe_accum.len < expected_accum_len)
    {
        return error.InvalidParam;
    }
    if (expert_counts.len < 128 or expert_offsets.len < 128) {
        return error.InvalidParam;
    }

    const w_ptr = all_expert_weights.ptr orelse return error.InvalidParam;
    const in_ptr = acts_in.ptr orelse return error.InvalidParam;
    const cnt_ptr = expert_counts.ptr orelse return error.InvalidParam;
    const off_ptr = expert_offsets.ptr orelse return error.InvalidParam;
    const sort_ptr = sorted_instances.ptr orelse return error.InvalidParam;
    const w_top_ptr = top_weights.ptr orelse return error.InvalidParam;
    const accum_ptr = moe_accum.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_gemv_down_accum_sorted(
        w_ptr,
        @intCast(expert_type),
        in_ptr,
        @ptrCast(cnt_ptr),
        @ptrCast(off_ptr),
        @ptrCast(sort_ptr),
        w_top_ptr,
        accum_ptr,
        @intCast(num_tokens),
        @intCast(in_dim),
        @intCast(out_dim),
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Phase 13 WP2: Active-Expert Persistent MoE Gate+Up+GeGLU on GPU:
/// Launches 84 persistent worker CTAs (matching 28 SMs x 3 CTAs/SM) pulling active expert row tiles dynamically
/// from work_counter via atomicAdd. Eliminates ~60,000 empty Gate+Up CTA launches per canvas step.
pub fn moeGemvGateUpGegluPersistent(
    all_expert_weights: DeviceSlice(u8),
    tokens_in: DeviceSlice(f32),
    expert_counts: DeviceSlice(i32),
    expert_offsets: DeviceSlice(i32),
    sorted_instances: DeviceSlice(i32),
    active_experts: DeviceSlice(i32),
    num_active_experts: DeviceSlice(i32),
    work_counter: DeviceSlice(i32),
    out_act: DeviceSlice(f32),
    num_tokens: usize,
    in_dim: usize,
    half_dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or in_dim == 0 or half_dim == 0 or in_dim % 256 != 0) return error.InvalidParam;
    const dev = all_expert_weights.device_id;
    if (tokens_in.device_id != dev or expert_counts.device_id != dev or
        expert_offsets.device_id != dev or sorted_instances.device_id != dev or
        active_experts.device_id != dev or num_active_experts.device_id != dev or
        work_counter.device_id != dev or out_act.device_id != dev)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    const num_instances = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_out_len = std.math.mul(usize, num_instances, half_dim) catch return error.InvalidParam;
    if (sorted_instances.len < num_instances or out_act.len < expected_out_len) {
        return error.InvalidParam;
    }
    if (expert_counts.len < 128 or expert_offsets.len < 128 or active_experts.len < 128 or num_active_experts.len < 1 or work_counter.len < 1) {
        return error.InvalidParam;
    }

    const w_ptr = all_expert_weights.ptr orelse return error.InvalidParam;
    const tok_ptr = tokens_in.ptr orelse return error.InvalidParam;
    const cnt_ptr = expert_counts.ptr orelse return error.InvalidParam;
    const off_ptr = expert_offsets.ptr orelse return error.InvalidParam;
    const sort_ptr = sorted_instances.ptr orelse return error.InvalidParam;
    const act_ptr = active_experts.ptr orelse return error.InvalidParam;
    const nae_ptr = num_active_experts.ptr orelse return error.InvalidParam;
    const wc_ptr = work_counter.ptr orelse return error.InvalidParam;
    const out_ptr = out_act.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_gemv_gate_up_geglu_persistent(
        w_ptr,
        tok_ptr,
        @ptrCast(cnt_ptr),
        @ptrCast(off_ptr),
        @ptrCast(sort_ptr),
        @ptrCast(act_ptr),
        @ptrCast(nae_ptr),
        @ptrCast(wc_ptr),
        out_ptr,
        @intCast(num_tokens),
        @intCast(in_dim),
        @intCast(half_dim),
        dev,
        s_handle,
    );
    if (rc != 0) return cuda.mapCudaErrorCode(rc);
}

/// Phase 13 WP2: Active-Expert Persistent MoE Down-Projection & Accumulation on GPU:
/// Launches 84 persistent worker CTAs (matching 28 SMs x 3 CTAs/SM) pulling active expert row tiles dynamically
/// from work_counter via atomicAdd. Eliminates ~72,000 empty Down-Projection CTA launches per canvas step.
pub fn moeGemvDownAccumPersistent(
    all_expert_weights: DeviceSlice(u8),
    expert_type: u32,
    acts_in: DeviceSlice(f32),
    expert_counts: DeviceSlice(i32),
    expert_offsets: DeviceSlice(i32),
    sorted_instances: DeviceSlice(i32),
    top_weights: DeviceSlice(f32),
    active_experts: DeviceSlice(i32),
    num_active_experts: DeviceSlice(i32),
    work_counter: DeviceSlice(i32),
    moe_accum: DeviceSlice(f32),
    num_tokens: usize,
    in_dim: usize,
    out_dim: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    if (!cuda.is_enabled) return error.CudaDisabledAtBuild;
    if (num_tokens == 0 or in_dim == 0 or out_dim == 0) return error.InvalidParam;
    const dev = all_expert_weights.device_id;
    if (acts_in.device_id != dev or expert_counts.device_id != dev or
        expert_offsets.device_id != dev or sorted_instances.device_id != dev or
        top_weights.device_id != dev or active_experts.device_id != dev or
        num_active_experts.device_id != dev or work_counter.device_id != dev or
        moe_accum.device_id != dev)
    {
        return error.DeviceMismatch;
    }
    if (stream) |s| {
        if (s.device_id != dev) return error.DeviceMismatch;
    }
    const num_instances = std.math.mul(usize, num_tokens, 8) catch return error.InvalidParam;
    const expected_in_len = std.math.mul(usize, num_instances, in_dim) catch return error.InvalidParam;
    const expected_accum_len = std.math.mul(usize, num_tokens, out_dim) catch return error.InvalidParam;
    if (acts_in.len < expected_in_len or sorted_instances.len < num_instances or
        top_weights.len < num_instances or moe_accum.len < expected_accum_len)
    {
        return error.InvalidParam;
    }
    if (expert_counts.len < 128 or expert_offsets.len < 128 or active_experts.len < 128 or
        num_active_experts.len < 1 or work_counter.len < 1)
    {
        return error.InvalidParam;
    }

    const w_ptr = all_expert_weights.ptr orelse return error.InvalidParam;
    const in_ptr = acts_in.ptr orelse return error.InvalidParam;
    const cnt_ptr = expert_counts.ptr orelse return error.InvalidParam;
    const off_ptr = expert_offsets.ptr orelse return error.InvalidParam;
    const sort_ptr = sorted_instances.ptr orelse return error.InvalidParam;
    const w_top_ptr = top_weights.ptr orelse return error.InvalidParam;
    const act_ptr = active_experts.ptr orelse return error.InvalidParam;
    const nae_ptr = num_active_experts.ptr orelse return error.InvalidParam;
    const wc_ptr = work_counter.ptr orelse return error.InvalidParam;
    const accum_ptr = moe_accum.ptr orelse return error.InvalidParam;
    const s_handle = if (stream) |s| s.handle else null;

    const rc = cuda.c.zeno_cuda_moe_gemv_down_accum_persistent(
        w_ptr,
        @intCast(expert_type),
        in_ptr,
        @ptrCast(cnt_ptr),
        @ptrCast(off_ptr),
        @ptrCast(sort_ptr),
        w_top_ptr,
        @ptrCast(act_ptr),
        @ptrCast(nae_ptr),
        @ptrCast(wc_ptr),
        accum_ptr,
        @intCast(num_tokens),
        @intCast(in_dim),
        @intCast(out_dim),
        dev,
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

test "CUDA Stage 7B cached attention parity against uncached full attention" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9988_7766);
    const rand = prng.random();

    const prompt_len: usize = 4;
    const canvas_len: usize = 2;
    const total_tokens = prompt_len + canvas_len;
    const num_q_heads: usize = 8;
    const num_kv_heads: usize = 2;
    const head_dim: usize = 256;

    const q_dim = num_q_heads * head_dim;
    const k_dim = num_kv_heads * head_dim;

    const host_q = try allocator.alloc(f32, total_tokens * q_dim);
    defer allocator.free(host_q);
    const host_k = try allocator.alloc(f32, total_tokens * k_dim);
    defer allocator.free(host_k);
    const host_v = try allocator.alloc(f32, total_tokens * k_dim);
    defer allocator.free(host_v);

    for (host_q) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
    for (host_k) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
    for (host_v) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

    // 1. Full uncached attention across all 6 tokens
    var d_full_q = try DeviceBuffer(f32).init(dev, host_q.len);
    defer d_full_q.deinit();
    try d_full_q.copyFromHost(host_q);

    var d_full_k = try DeviceBuffer(f32).init(dev, host_k.len);
    defer d_full_k.deinit();
    try d_full_k.copyFromHost(host_k);

    var d_full_v = try DeviceBuffer(f32).init(dev, host_v.len);
    defer d_full_v.deinit();
    try d_full_v.copyFromHost(host_v);

    var d_full_out = try DeviceBuffer(f32).init(dev, total_tokens * q_dim);
    defer d_full_out.deinit();

    try attention(
        d_full_q.slice(),
        d_full_k.slice(),
        d_full_v.slice(),
        d_full_out.slice(),
        total_tokens,
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

    const gpu_full_out = try allocator.alloc(f32, total_tokens * q_dim);
    defer allocator.free(gpu_full_out);
    try d_full_out.copyToHost(gpu_full_out);

    // 2. Cached attention: prompt KV cached, only canvas Q, K, V evaluated
    const prompt_k_slice = try d_full_k.slice().subslice(0, prompt_len * k_dim);
    const prompt_v_slice = try d_full_v.slice().subslice(0, prompt_len * k_dim);
    const canvas_q_slice = try d_full_q.slice().subslice(prompt_len * q_dim, canvas_len * q_dim);
    const canvas_k_slice = try d_full_k.slice().subslice(prompt_len * k_dim, canvas_len * k_dim);
    const canvas_v_slice = try d_full_v.slice().subslice(prompt_len * k_dim, canvas_len * k_dim);

    var d_cached_out = try DeviceBuffer(f32).init(dev, canvas_len * q_dim);
    defer d_cached_out.deinit();

    try attentionCached(
        canvas_q_slice,
        prompt_k_slice,
        prompt_v_slice,
        canvas_k_slice,
        canvas_v_slice,
        d_cached_out.slice(),
        prompt_len,
        canvas_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        q_dim,
        k_dim,
        k_dim,
        q_dim,
        null,
    );

    const gpu_cached_out = try allocator.alloc(f32, canvas_len * q_dim);
    defer allocator.free(gpu_cached_out);
    try d_cached_out.copyToHost(gpu_cached_out);

    // 3. Compare canvas tokens output from cached attention vs full attention
    const expected_canvas_out = gpu_full_out[prompt_len * q_dim .. total_tokens * q_dim];
    var max_cached_diff: f32 = 0.0;
    for (expected_canvas_out, gpu_cached_out) |exp, act| {
        const diff = @abs(exp - act);
        if (diff > max_cached_diff) max_cached_diff = diff;
        try std.testing.expect(diff <= 1e-6);
    }
    std.debug.print("[parity] Cached Attention vs Full Attention: max_diff = {e:.4}\n", .{max_cached_diff});
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

test "CUDA Stage 7A token-sorted MoE binning and accumulation parity" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    const num_tokens: usize = 4;
    const num_instances: usize = num_tokens * 8;
    const dim: usize = 64;

    // Synthetic top_indices with repeated experts across tokens
    const host_top_indices = try allocator.alloc(i32, num_instances);
    defer allocator.free(host_top_indices);
    const sample_experts = [_]i32{ 5, 12, 12, 42, 42, 42, 88, 103 };
    for (0..num_tokens) |t| {
        for (0..8) |s| {
            // Shift slightly per token to test overlap and multi-token active experts
            host_top_indices[t * 8 + s] = @mod(sample_experts[s] + @as(i32, @intCast(t * 3)), 128);
        }
    }

    // Expected CPU histogram
    var cpu_counts: [128]i32 = @splat(0);
    for (host_top_indices) |exp| {
        cpu_counts[@intCast(exp)] += 1;
    }

    var d_top_indices = try DeviceBuffer(i32).init(dev, num_instances);
    defer d_top_indices.deinit();
    try d_top_indices.copyFromHost(host_top_indices);

    var d_expert_counts = try DeviceBuffer(i32).init(dev, 128);
    defer d_expert_counts.deinit();
    var d_expert_offsets = try DeviceBuffer(i32).init(dev, 128);
    defer d_expert_offsets.deinit();
    var d_sorted_instances = try DeviceBuffer(i32).init(dev, num_instances);
    defer d_sorted_instances.deinit();
    var d_instance_to_sorted = try DeviceBuffer(i32).init(dev, num_instances);
    defer d_instance_to_sorted.deinit();
    var d_active_experts = try DeviceBuffer(i32).init(dev, 128);
    defer d_active_experts.deinit();
    var d_num_active = try DeviceBuffer(i32).init(dev, 1);
    defer d_num_active.deinit();

    try moeBuildExpertBins(
        d_top_indices.slice(),
        d_expert_counts.slice(),
        d_expert_offsets.slice(),
        d_sorted_instances.slice(),
        d_instance_to_sorted.slice(),
        d_active_experts.slice(),
        d_num_active.slice(),
        num_tokens,
        null,
    );

    const gpu_counts = try allocator.alloc(i32, 128);
    defer allocator.free(gpu_counts);
    try d_expert_counts.copyToHost(gpu_counts);

    const gpu_offsets = try allocator.alloc(i32, 128);
    defer allocator.free(gpu_offsets);
    try d_expert_offsets.copyToHost(gpu_offsets);

    const gpu_sorted = try allocator.alloc(i32, num_instances);
    defer allocator.free(gpu_sorted);
    try d_sorted_instances.copyToHost(gpu_sorted);

    const gpu_inst_to_sort = try allocator.alloc(i32, num_instances);
    defer allocator.free(gpu_inst_to_sort);
    try d_instance_to_sorted.copyToHost(gpu_inst_to_sort);

    const gpu_active_experts = try allocator.alloc(i32, 128);
    defer allocator.free(gpu_active_experts);
    try d_active_experts.copyToHost(gpu_active_experts);

    const gpu_num_active = try allocator.alloc(i32, 1);
    defer allocator.free(gpu_num_active);
    try d_num_active.copyToHost(gpu_num_active);

    // 1. Verify histogram counts & active-expert compaction (Phase 13 WP2)
    var expected_active: usize = 0;
    for (0..128) |e| {
        try std.testing.expectEqual(cpu_counts[e], gpu_counts[e]);
        if (cpu_counts[e] > 0) {
            try std.testing.expectEqual(@as(i32, @intCast(e)), gpu_active_experts[expected_active]);
            expected_active += 1;
        }
    }
    try std.testing.expectEqual(@as(i32, @intCast(expected_active)), gpu_num_active[0]);

    // 2. Verify bidirectional mapping consistency and expert groupings
    for (0..num_instances) |orig_i| {
        const sorted_idx = gpu_inst_to_sort[orig_i];
        try std.testing.expect(sorted_idx >= 0 and sorted_idx < num_instances);
        try std.testing.expectEqual(@as(i32, @intCast(orig_i)), gpu_sorted[@intCast(sorted_idx)]);

        const exp_id = host_top_indices[orig_i];
        const off = gpu_offsets[@intCast(exp_id)];
        const cnt = gpu_counts[@intCast(exp_id)];
        try std.testing.expect(sorted_idx >= off and sorted_idx < off + cnt);
    }

    // 3. Verify sorted accumulation parity
    var prng = std.Random.DefaultPrng.init(0x5566_7788);
    const rand = prng.random();

    const host_down_out = try allocator.alloc(f32, num_instances * dim);
    defer allocator.free(host_down_out);
    for (host_down_out) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

    const host_top_weights = try allocator.alloc(f32, num_instances);
    defer allocator.free(host_top_weights);
    for (host_top_weights) |*w| w.* = rand.float(f32) * 0.25;

    // CPU reference accumulation
    const cpu_accum = try allocator.alloc(f32, num_tokens * dim);
    defer allocator.free(cpu_accum);
    for (0..num_tokens) |t| {
        for (0..dim) |d| {
            var sum: f32 = 0.0;
            for (0..8) |s| {
                const k = t * 8 + s;
                sum += host_down_out[k * dim + d] * host_top_weights[k];
            }
            cpu_accum[t * dim + d] = sum;
        }
    }

    // Reorder host_down_out into sorted order
    const sorted_down_out = try allocator.alloc(f32, num_instances * dim);
    defer allocator.free(sorted_down_out);
    for (0..num_instances) |orig_k| {
        const sorted_k: usize = @intCast(gpu_inst_to_sort[orig_k]);
        @memcpy(sorted_down_out[sorted_k * dim .. (sorted_k + 1) * dim], host_down_out[orig_k * dim .. (orig_k + 1) * dim]);
    }

    var d_sorted_down = try DeviceBuffer(f32).init(dev, num_instances * dim);
    defer d_sorted_down.deinit();
    try d_sorted_down.copyFromHost(sorted_down_out);

    var d_top_weights = try DeviceBuffer(f32).init(dev, num_instances);
    defer d_top_weights.deinit();
    try d_top_weights.copyFromHost(host_top_weights);

    var d_moe_accum = try DeviceBuffer(f32).init(dev, num_tokens * dim);
    defer d_moe_accum.deinit();

    try moeAccumulateScaledSorted(
        d_moe_accum.slice(),
        d_sorted_down.slice(),
        d_instance_to_sorted.slice(),
        d_top_weights.slice(),
        num_tokens,
        dim,
        null,
    );

    const gpu_accum = try allocator.alloc(f32, num_tokens * dim);
    defer allocator.free(gpu_accum);
    try d_moe_accum.copyToHost(gpu_accum);

    var max_accum_diff: f32 = 0.0;
    for (cpu_accum, gpu_accum) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_accum_diff) max_accum_diff = diff;
        try std.testing.expect(diff <= 1e-6);
    }
    std.debug.print("[parity] Token-Sorted MoE Accumulate: max_diff = {e:.4}\n", .{max_accum_diff});
}

test "CUDA layerOutput dual-region scaling parity" {
    const dev = try requireHardwareDevice();
    const allocator = std.testing.allocator;

    const num_tokens: usize = 8;
    const prompt_len: usize = 3;
    const dim: usize = 2816;
    const total_len = num_tokens * dim;
    const enc_scale: f32 = 0.75;
    const layer_scale: f32 = 1.25;

    const host_xmid = try allocator.alloc(f32, total_len);
    defer allocator.free(host_xmid);
    const host_postffn = try allocator.alloc(f32, total_len);
    defer allocator.free(host_postffn);
    const cpu_out = try allocator.alloc(f32, total_len);
    defer allocator.free(cpu_out);

    for (0..total_len) |i| {
        host_xmid[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
        host_postffn[i] = @as(f32, @floatFromInt((i + 37) % 100)) * 0.01;
        const t = i / dim;
        const s = if (t < prompt_len) enc_scale else layer_scale;
        cpu_out[i] = (host_xmid[i] + host_postffn[i]) * s;
    }

    var d_out = try DeviceBuffer(f32).init(dev, total_len);
    defer d_out.deinit();
    var d_xmid = try DeviceBuffer(f32).init(dev, total_len);
    defer d_xmid.deinit();
    var d_postffn = try DeviceBuffer(f32).init(dev, total_len);
    defer d_postffn.deinit();

    try d_xmid.copyFromHost(host_xmid);
    try d_postffn.copyFromHost(host_postffn);

    try layerOutput(d_out.slice(), d_xmid.slice(), d_postffn.slice(), prompt_len, num_tokens, dim, enc_scale, layer_scale, null);

    const gpu_out = try allocator.alloc(f32, total_len);
    defer allocator.free(gpu_out);
    try d_out.copyToHost(gpu_out);

    var max_diff: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_diff) max_diff = diff;
        try std.testing.expect(diff <= 1e-6);
    }
    std.debug.print("[parity] Layer Output: max_diff = {e:.4}\n", .{max_diff});
}

test "CudaGraph capture, instantiation, and execution parity" {
    if (!cuda.is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const topo = cuda.getTopology() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (topo.reported_devices == 0) return error.SkipZigTest;
    const dev = topo.devices[0].device_id;

    var stream = try CudaStream.init(dev);
    defer stream.deinit();

    const count: usize = 1024;
    var d_x = try DeviceBuffer(f32).init(dev, count);
    defer d_x.deinit();
    var d_w = try DeviceBuffer(f32).init(dev, count);
    defer d_w.deinit();
    var d_out = try DeviceBuffer(f32).init(dev, count);
    defer d_out.deinit();

    const host_x = try allocator.alloc(f32, count);
    defer allocator.free(host_x);
    const host_w = try allocator.alloc(f32, count);
    defer allocator.free(host_w);

    for (0..count) |i| {
        host_x[i] = @as(f32, @floatFromInt(i % 50)) * 0.02 + 0.1;
        host_w[i] = 1.0;
    }

    try d_x.copyFromHost(host_x);
    try d_w.copyFromHost(host_w);
    try stream.synchronize();

    // 1. Capture graph: RMSNorm + residual add
    try CudaGraph.beginCapture(&stream);
    try rmsNorm(d_x.slice(), d_w.slice(), d_out.slice(), 1, @intCast(count), 1e-6, &stream);
    try addResidual(d_out.slice(), d_x.slice(), &stream);
    var graph = try CudaGraph.endCapture(&stream);
    defer graph.deinit();

    // 2. Launch graph
    try graph.launch(&stream);
    try stream.synchronize();

    const gpu_out = try allocator.alloc(f32, count);
    defer allocator.free(gpu_out);
    try d_out.copyToHost(gpu_out);

    // 3. Compute reference on CPU
    const cpu_norm = try allocator.alloc(f32, count);
    defer allocator.free(cpu_norm);
    ops.rmsNorm(host_x, host_w, 1e-6, cpu_norm);
    for (0..count) |i| {
        cpu_norm[i] += host_x[i];
    }

    var max_diff: f32 = 0.0;
    for (cpu_norm, gpu_out) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_diff) max_diff = diff;
        try std.testing.expect(diff <= 1e-5);
    }
    std.debug.print("[parity] CudaGraph Replay: max_diff = {e:.4}\n", .{max_diff});
}

test "CudaGraph capture failure eager fallback parity (R23-01)" {
    if (!cuda.is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const topo = cuda.getTopology() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (topo.reported_devices == 0) return error.SkipZigTest;
    const dev = topo.devices[0].device_id;

    var stream = try CudaStream.init(dev);
    defer stream.deinit();

    const count: usize = 1024;
    var d_x = try DeviceBuffer(f32).init(dev, count);
    defer d_x.deinit();
    var d_w = try DeviceBuffer(f32).init(dev, count);
    defer d_w.deinit();
    var d_out = try DeviceBuffer(f32).init(dev, count);
    defer d_out.deinit();

    const host_x = try allocator.alloc(f32, count);
    defer allocator.free(host_x);
    const host_w = try allocator.alloc(f32, count);
    defer allocator.free(host_w);

    for (0..count) |i| {
        host_x[i] = @as(f32, @floatFromInt(i % 50)) * 0.02 + 0.1;
        host_w[i] = 1.0;
    }

    try d_x.copyFromHost(host_x);
    try d_w.copyFromHost(host_w);
    try stream.synchronize();

    // 1. Begin capture, capture kernels, then end capture but simulate launch failure
    try CudaGraph.beginCapture(&stream);
    try rmsNorm(d_x.slice(), d_w.slice(), d_out.slice(), 1, @intCast(count), 1e-6, &stream);
    try addResidual(d_out.slice(), d_x.slice(), &stream);
    var graph = try CudaGraph.endCapture(&stream);
    // Discard graph without launching (simulating graph instantiation/launch failure)
    graph.deinit();

    // 2. Rerun eagerly outside capture (R23-01 fallback contract)
    try rmsNorm(d_x.slice(), d_w.slice(), d_out.slice(), 1, @intCast(count), 1e-6, &stream);
    try addResidual(d_out.slice(), d_x.slice(), &stream);
    try stream.synchronize();

    const gpu_out = try allocator.alloc(f32, count);
    defer allocator.free(gpu_out);
    try d_out.copyToHost(gpu_out);

    // 3. Compute CPU reference
    const cpu_norm = try allocator.alloc(f32, count);
    defer allocator.free(cpu_norm);
    ops.rmsNorm(host_x, host_w, 1e-6, cpu_norm);
    for (0..count) |i| {
        cpu_norm[i] += host_x[i];
    }

    var max_diff: f32 = 0.0;
    for (cpu_norm, gpu_out) |c, g| {
        const diff = @abs(c - g);
        if (diff > max_diff) max_diff = diff;
        try std.testing.expect(diff <= 1e-5);
    }
    std.debug.print("[parity] CudaGraph Eager Fallback: max_diff = {e:.4}\n", .{max_diff});
}

test "CudaGraph cancelCapture resets stream capture state (R24-02)" {
    if (!cuda.is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const topo = cuda.getTopology() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (topo.reported_devices == 0) return error.SkipZigTest;
    const dev = topo.devices[0].device_id;

    var stream = try CudaStream.init(dev);
    defer stream.deinit();

    const count: usize = 512;
    var d_x = try DeviceBuffer(f32).init(dev, count);
    defer d_x.deinit();
    var d_out = try DeviceBuffer(f32).init(dev, count);
    defer d_out.deinit();

    const host_x = try allocator.alloc(f32, count);
    defer allocator.free(host_x);
    for (0..count) |i| {
        host_x[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }
    try d_x.copyFromHost(host_x);
    try stream.synchronize();

    // 1. Begin capture and enqueue kernel
    try CudaGraph.beginCapture(&stream);
    try addResidual(d_out.slice(), d_x.slice(), &stream);

    // 2. Abort capture via cancelCapture
    CudaGraph.cancelCapture(&stream);

    // 3. Verify the stream is immediately operational outside capture
    try addResidual(d_out.slice(), d_x.slice(), &stream);
    try stream.synchronize();

    const gpu_out = try allocator.alloc(f32, count);
    defer allocator.free(gpu_out);
    try d_out.copyToHost(gpu_out);

    for (host_x, gpu_out) |h, g| {
        try std.testing.expectEqual(h, g);
    }
}

test "CUDA Stage 8A: 16x token-tiled batched GEMVs numerical parity" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x313658);
    const rand = prng.random();

    const out_dim: usize = 16;
    const in_dim: usize = 256;
    const batch_size: usize = 16;

    // 1. Test Q8_0 16x Batched GEMV
    {
        const blocks_per_row = in_dim / quant.QK8_0; // 8
        const total_blocks = out_dim * blocks_per_row;
        const host_w = try allocator.alloc(quant.BlockQ8_0, total_blocks);
        defer allocator.free(host_w);
        for (host_w) |*blk| {
            blk.d = @floatCast((rand.float(f32) * 0.3) + 0.1);
            for (&blk.qs) |*q| q.* = rand.intRangeAtMost(i8, -100, 100);
        }

        const host_x = try allocator.alloc(f32, batch_size * in_dim);
        defer allocator.free(host_x);
        for (host_x) |*x| x.* = (rand.float(f32) * 3.0) - 1.5;

        var d_w = try DeviceBuffer(u8).init(dev, std.mem.sliceAsBytes(host_w).len);
        defer d_w.deinit();
        try d_w.copyFromHost(std.mem.sliceAsBytes(host_w));

        var d_x = try DeviceBuffer(f32).init(dev, host_x.len);
        defer d_x.deinit();
        try d_x.copyFromHost(host_x);

        var d_out = try DeviceBuffer(f32).init(dev, batch_size * out_dim);
        defer d_out.deinit();

        try gemvQ8_0(d_w.slice(), d_x.slice(), d_out.slice(), out_dim, in_dim, batch_size, null);

        const gpu_out = try allocator.alloc(f32, batch_size * out_dim);
        defer allocator.free(gpu_out);
        try d_out.copyToHost(gpu_out);

        for (0..batch_size) |b| {
            const x_sub = host_x[b * in_dim .. (b + 1) * in_dim];
            var cpu_out: [16]f32 = undefined;
            infer.gemvQ8_0(host_w.ptr, x_sub, &cpu_out, out_dim, in_dim);
            const gpu_sub = gpu_out[b * out_dim .. (b + 1) * out_dim];
            for (cpu_out, gpu_sub) |c, g| {
                const diff = @abs(c - g);
                try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
            }
        }
    }

    // 2. Test Q4_K 16x Batched GEMV
    {
        const blocks_per_row = in_dim / quant.QK_K; // 1
        const total_blocks = out_dim * blocks_per_row;
        const host_w = try allocator.alloc(quant.BlockQ4_K, total_blocks);
        defer allocator.free(host_w);
        for (host_w) |*blk| {
            blk.d = @floatCast((rand.float(f32) * 0.2) + 0.05);
            blk.dmin = @floatCast((rand.float(f32) * 0.1) + 0.01);
            for (&blk.scales) |*s| s.* = rand.intRangeAtMost(u8, 1, 63);
            for (&blk.qs) |*q| q.* = rand.intRangeAtMost(u8, 0, 255);
        }

        const host_x = try allocator.alloc(f32, batch_size * in_dim);
        defer allocator.free(host_x);
        for (host_x) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

        var d_w = try DeviceBuffer(u8).init(dev, std.mem.sliceAsBytes(host_w).len);
        defer d_w.deinit();
        try d_w.copyFromHost(std.mem.sliceAsBytes(host_w));

        var d_x = try DeviceBuffer(f32).init(dev, host_x.len);
        defer d_x.deinit();
        try d_x.copyFromHost(host_x);

        var d_out = try DeviceBuffer(f32).init(dev, batch_size * out_dim);
        defer d_out.deinit();

        try gemvQ4_K(d_w.slice(), d_x.slice(), d_out.slice(), out_dim, in_dim, batch_size, null);

        const gpu_out = try allocator.alloc(f32, batch_size * out_dim);
        defer allocator.free(gpu_out);
        try d_out.copyToHost(gpu_out);

        for (0..batch_size) |b| {
            const x_sub = host_x[b * in_dim .. (b + 1) * in_dim];
            var cpu_out: [16]f32 = undefined;
            infer.gemvQ4_K(host_w.ptr, x_sub, &cpu_out, out_dim, in_dim);
            const gpu_sub = gpu_out[b * out_dim .. (b + 1) * out_dim];
            for (cpu_out, gpu_sub) |c, g| {
                const diff = @abs(c - g);
                try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
            }
        }
    }

    // 3. Test Q6_K 16x Batched GEMV
    {
        const blocks_per_row = in_dim / quant.QK_K; // 1
        const total_blocks = out_dim * blocks_per_row;
        const host_w = try allocator.alloc(quant.BlockQ6_K, total_blocks);
        defer allocator.free(host_w);
        for (host_w) |*blk| {
            blk.d = @floatCast((rand.float(f32) * 0.2) + 0.05);
            for (&blk.scales) |*s| s.* = rand.intRangeAtMost(i8, -32, 31);
            for (&blk.ql) |*q| q.* = rand.intRangeAtMost(u8, 0, 255);
            for (&blk.qh) |*q| q.* = rand.intRangeAtMost(u8, 0, 255);
        }

        const host_x = try allocator.alloc(f32, batch_size * in_dim);
        defer allocator.free(host_x);
        for (host_x) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

        var d_w = try DeviceBuffer(u8).init(dev, std.mem.sliceAsBytes(host_w).len);
        defer d_w.deinit();
        try d_w.copyFromHost(std.mem.sliceAsBytes(host_w));

        var d_x = try DeviceBuffer(f32).init(dev, host_x.len);
        defer d_x.deinit();
        try d_x.copyFromHost(host_x);

        var d_out = try DeviceBuffer(f32).init(dev, batch_size * out_dim);
        defer d_out.deinit();

        try gemvQ6_K(d_w.slice(), d_x.slice(), d_out.slice(), out_dim, in_dim, batch_size, null);

        const gpu_out = try allocator.alloc(f32, batch_size * out_dim);
        defer allocator.free(gpu_out);
        try d_out.copyToHost(gpu_out);

        for (0..batch_size) |b| {
            const x_sub = host_x[b * in_dim .. (b + 1) * in_dim];
            var cpu_out: [16]f32 = undefined;
            infer.gemvQ6_K(host_w.ptr, x_sub, &cpu_out, out_dim, in_dim);
            const gpu_sub = gpu_out[b * out_dim .. (b + 1) * out_dim];
            for (cpu_out, gpu_sub) |c, g| {
                const diff = @abs(c - g);
                try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
            }
        }
    }
}

test "CUDA Stage 8B: Fused Dense Gate+Up+GeGLU Q4_K parity against separate GEMV + GeGLU" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x465553);
    const rand = prng.random();

    const out_dim: usize = 32;
    const in_dim: usize = 256;
    const batch_size: usize = 16;
    const blocks_per_row = in_dim / quant.QK_K; // 1
    const total_blocks = out_dim * blocks_per_row;

    const host_w_gate = try allocator.alloc(quant.BlockQ4_K, total_blocks);
    defer allocator.free(host_w_gate);
    for (host_w_gate) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.2) + 0.05);
        blk.dmin = @floatCast((rand.float(f32) * 0.1) + 0.01);
        for (&blk.scales) |*s| s.* = rand.intRangeAtMost(u8, 1, 63);
        for (&blk.qs) |*q| q.* = rand.intRangeAtMost(u8, 0, 255);
    }

    const host_w_up = try allocator.alloc(quant.BlockQ4_K, total_blocks);
    defer allocator.free(host_w_up);
    for (host_w_up) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.2) + 0.05);
        blk.dmin = @floatCast((rand.float(f32) * 0.1) + 0.01);
        for (&blk.scales) |*s| s.* = rand.intRangeAtMost(u8, 1, 63);
        for (&blk.qs) |*q| q.* = rand.intRangeAtMost(u8, 0, 255);
    }

    const host_x = try allocator.alloc(f32, batch_size * in_dim);
    defer allocator.free(host_x);
    for (host_x) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

    var d_w_gate = try DeviceBuffer(u8).init(dev, std.mem.sliceAsBytes(host_w_gate).len);
    defer d_w_gate.deinit();
    try d_w_gate.copyFromHost(std.mem.sliceAsBytes(host_w_gate));

    var d_w_up = try DeviceBuffer(u8).init(dev, std.mem.sliceAsBytes(host_w_up).len);
    defer d_w_up.deinit();
    try d_w_up.copyFromHost(std.mem.sliceAsBytes(host_w_up));

    var d_x = try DeviceBuffer(f32).init(dev, host_x.len);
    defer d_x.deinit();
    try d_x.copyFromHost(host_x);

    // 1. Fused execution
    var d_fused_out = try DeviceBuffer(f32).init(dev, batch_size * out_dim);
    defer d_fused_out.deinit();

    try denseGateUpGegluQ4_K(
        d_w_gate.slice(),
        d_w_up.slice(),
        d_x.slice(),
        d_fused_out.slice(),
        out_dim,
        in_dim,
        batch_size,
        null,
    );

    // 2. Separate reference execution (gemv gate -> gemv up -> geglu)
    var d_gate_out = try DeviceBuffer(f32).init(dev, batch_size * out_dim);
    defer d_gate_out.deinit();
    var d_up_out = try DeviceBuffer(f32).init(dev, batch_size * out_dim);
    defer d_up_out.deinit();
    var d_ref_out = try DeviceBuffer(f32).init(dev, batch_size * out_dim);
    defer d_ref_out.deinit();

    try gemvQ4_K(d_w_gate.slice(), d_x.slice(), d_gate_out.slice(), out_dim, in_dim, batch_size, null);
    try gemvQ4_K(d_w_up.slice(), d_x.slice(), d_up_out.slice(), out_dim, in_dim, batch_size, null);
    try geglu(d_gate_out.slice(), d_up_out.slice(), d_ref_out.slice(), null);

    // 3. Compare outputs
    const fused_host = try allocator.alloc(f32, batch_size * out_dim);
    defer allocator.free(fused_host);
    try d_fused_out.copyToHost(fused_host);

    const ref_host = try allocator.alloc(f32, batch_size * out_dim);
    defer allocator.free(ref_host);
    try d_ref_out.copyToHost(ref_host);

    var max_diff: f32 = 0.0;
    for (fused_host, ref_host) |f, r| {
        const diff = @abs(f - r);
        if (diff > max_diff) max_diff = diff;
        try std.testing.expect(diff <= 1e-4 + 1e-4 * @abs(r));
    }
}

test "CUDA Review 25: moeAccumulateScaledSorted negative index guard (R25-01)" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    const num_tokens: usize = 2;
    const dim: usize = 64;
    const num_instances = num_tokens * 8; // 16

    // Half of instances are valid sorted slots [0..7], half are inactive (-1)
    const host_inst_to_sorted = try allocator.alloc(i32, num_instances);
    defer allocator.free(host_inst_to_sorted);
    for (host_inst_to_sorted, 0..) |*slot, i| {
        slot.* = if (i % 2 == 0) @intCast(i / 2) else -1;
    }

    const host_top_weights = try allocator.alloc(f32, num_instances);
    defer allocator.free(host_top_weights);
    for (host_top_weights) |*w| w.* = 0.25;

    // sorted_down_out has num_instances buffer capacity
    const host_sorted_down = try allocator.alloc(f32, num_instances * dim);
    defer allocator.free(host_sorted_down);
    for (host_sorted_down, 0..) |*v, idx| {
        v.* = @floatFromInt(idx + 1);
    }

    var d_accum = try DeviceBuffer(f32).init(dev, num_tokens * dim);
    defer d_accum.deinit();

    var d_sorted_down = try DeviceBuffer(f32).init(dev, host_sorted_down.len);
    defer d_sorted_down.deinit();
    try d_sorted_down.copyFromHost(host_sorted_down);

    var d_inst_to_sorted = try DeviceBuffer(i32).init(dev, host_inst_to_sorted.len);
    defer d_inst_to_sorted.deinit();
    try d_inst_to_sorted.copyFromHost(host_inst_to_sorted);

    var d_top_weights = try DeviceBuffer(f32).init(dev, host_top_weights.len);
    defer d_top_weights.deinit();
    try d_top_weights.copyFromHost(host_top_weights);

    // Should execute safely without segfault or illegal memory access
    try moeAccumulateScaledSorted(
        d_accum.slice(),
        d_sorted_down.slice(),
        d_inst_to_sorted.slice(),
        d_top_weights.slice(),
        num_tokens,
        dim,
        null,
    );

    const host_out = try allocator.alloc(f32, num_tokens * dim);
    defer allocator.free(host_out);
    try d_accum.copyToHost(host_out);

    // Verify non-zero accumulated values
    for (host_out) |v| {
        try std.testing.expect(v > 0.0);
    }
}

test "CUDA gemvQ4_K_FusedQK numerical parity with CPU reference" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x514655);
    const rand = prng.random();

    const q_dim: usize = 32;
    const k_dim: usize = 16;
    const in_dim: usize = 512;
    const blocks_per_row = in_dim / quant.QK_K; // 2

    // 1. Generate Q and K weights
    const q_blocks = q_dim * blocks_per_row;
    const host_wq = try allocator.alloc(quant.BlockQ4_K, q_blocks);
    defer allocator.free(host_wq);
    for (host_wq) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.3) + 0.1);
        blk.dmin = @floatCast((rand.float(f32) * 0.15) + 0.05);
        for (&blk.scales) |*sc| sc.* = rand.int(u8);
        for (&blk.qs) |*q| q.* = rand.int(u8);
    }

    const k_blocks = k_dim * blocks_per_row;
    const host_wk = try allocator.alloc(quant.BlockQ4_K, k_blocks);
    defer allocator.free(host_wk);
    for (host_wk) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.3) + 0.1);
        blk.dmin = @floatCast((rand.float(f32) * 0.15) + 0.05);
        for (&blk.scales) |*sc| sc.* = rand.int(u8);
        for (&blk.qs) |*q| q.* = rand.int(u8);
    }

    // 2. Upload weights to GPU
    var d_wq = try DeviceBuffer(u8).init(dev, std.mem.sliceAsBytes(host_wq).len);
    defer d_wq.deinit();
    try d_wq.copyFromHost(std.mem.sliceAsBytes(host_wq));

    var d_wk = try DeviceBuffer(u8).init(dev, std.mem.sliceAsBytes(host_wk).len);
    defer d_wk.deinit();
    try d_wk.copyFromHost(std.mem.sliceAsBytes(host_wk));

    // Test across batch_size 1 and 16
    const batch_sizes = [_]usize{ 1, 16 };
    for (batch_sizes) |batch_size| {
        const host_x = try allocator.alloc(f32, batch_size * in_dim);
        defer allocator.free(host_x);
        for (host_x) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

        var d_x = try DeviceBuffer(f32).init(dev, host_x.len);
        defer d_x.deinit();
        try d_x.copyFromHost(host_x);

        var d_out_q = try DeviceBuffer(f32).init(dev, batch_size * q_dim);
        defer d_out_q.deinit();
        var d_out_k = try DeviceBuffer(f32).init(dev, batch_size * k_dim);
        defer d_out_k.deinit();
        var d_out_v = try DeviceBuffer(f32).init(dev, batch_size * k_dim);
        defer d_out_v.deinit();

        // Run fused Q+K GEMV with out_v populated
        try gemvQ4_K_FusedQK(
            d_wq.slice(),
            d_wk.slice(),
            d_x.slice(),
            d_out_q.slice(),
            d_out_k.slice(),
            d_out_v.slice(),
            q_dim,
            k_dim,
            in_dim,
            batch_size,
            null,
        );

        const gpu_out_q = try allocator.alloc(f32, batch_size * q_dim);
        defer allocator.free(gpu_out_q);
        try d_out_q.copyToHost(gpu_out_q);

        const gpu_out_k = try allocator.alloc(f32, batch_size * k_dim);
        defer allocator.free(gpu_out_k);
        try d_out_k.copyToHost(gpu_out_k);

        const gpu_out_v = try allocator.alloc(f32, batch_size * k_dim);
        defer allocator.free(gpu_out_v);
        try d_out_v.copyToHost(gpu_out_v);

        var max_diff_q: f32 = 0.0;
        var max_diff_k: f32 = 0.0;
        // Verify each token against CPU reference
        for (0..batch_size) |b| {
            const x_sub = host_x[b * in_dim .. (b + 1) * in_dim];

            // Verify Q
            var cpu_q: [32]f32 = undefined;
            infer.gemvQ4_K(host_wq.ptr, x_sub, &cpu_q, q_dim, in_dim);
            const gpu_q_sub = gpu_out_q[b * q_dim .. (b + 1) * q_dim];
            for (cpu_q, gpu_q_sub) |c, g| {
                const diff = @abs(c - g);
                if (diff > max_diff_q) max_diff_q = diff;
                try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
            }

            // Verify K
            var cpu_k: [16]f32 = undefined;
            infer.gemvQ4_K(host_wk.ptr, x_sub, &cpu_k, k_dim, in_dim);
            const gpu_k_sub = gpu_out_k[b * k_dim .. (b + 1) * k_dim];
            for (cpu_k, gpu_k_sub) |c, g| {
                const diff = @abs(c - g);
                if (diff > max_diff_k) max_diff_k = diff;
                try std.testing.expect(diff <= 1e-3 + 1e-4 * @abs(c));
            }

            // Verify tied V matches K exactly
            const gpu_v_sub = gpu_out_v[b * k_dim .. (b + 1) * k_dim];
            for (gpu_k_sub, gpu_v_sub) |k, v| {
                try std.testing.expectEqual(k, v);
            }
        }
        std.debug.print("[parity] gemvQ4_K_FusedQK (batch={d}): max_diff_q = {e:.4}, max_diff_k = {e:.4}\n", .{ batch_size, max_diff_q, max_diff_k });
    }
}

test "CUDA Phase 13: moeGemvDownAccumSorted numerical parity with separate down + accumulate" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.SkipZigTest => return,
        else => return err,
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x131313);
    const rand = prng.random();

    const num_tokens: usize = 2;
    const in_dim: usize = 704;
    const out_dim: usize = 64;
    const blocks_per_row = in_dim / 32; // 22
    const num_instances = num_tokens * 8; // 16
    const num_experts: usize = 128;

    // 1. Allocate Q8_0 weights for 128 experts (out_dim rows x 22 blocks)
    const blocks_per_expert = out_dim * blocks_per_row;
    const total_blocks = num_experts * blocks_per_expert;
    const host_w = try allocator.alloc(quant.BlockQ8_0, total_blocks);
    defer allocator.free(host_w);
    for (host_w) |*blk| {
        blk.d = @floatCast((rand.float(f32) * 0.1) + 0.01);
        for (&blk.qs) |*q| q.* = rand.intRangeAtMost(i8, -127, 127);
    }

    var d_w = try DeviceBuffer(u8).init(dev, std.mem.sliceAsBytes(host_w).len);
    defer d_w.deinit();
    try d_w.copyFromHost(std.mem.sliceAsBytes(host_w));

    // 2. Setup expert assignments and routing
    // Assign instances 0..15 to experts 0..15
    const host_counts = try allocator.alloc(i32, num_experts);
    defer allocator.free(host_counts);
    @memset(host_counts, 0);

    const host_offsets = try allocator.alloc(i32, num_experts);
    defer allocator.free(host_offsets);

    const host_sorted_inst = try allocator.alloc(i32, num_instances);
    defer allocator.free(host_sorted_inst);

    const host_inst_to_sorted = try allocator.alloc(i32, num_instances);
    defer allocator.free(host_inst_to_sorted);

    for (0..num_instances) |i| {
        host_counts[i] = 1;
        host_offsets[i] = @intCast(i);
        host_sorted_inst[i] = @intCast(i);
        host_inst_to_sorted[i] = @intCast(i);
    }
    for (num_instances..num_experts) |i| {
        host_offsets[i] = @intCast(num_instances);
    }

    const host_top_weights = try allocator.alloc(f32, num_instances);
    defer allocator.free(host_top_weights);
    for (host_top_weights, 0..) |*w, i| {
        w.* = @as(f32, @floatFromInt((i % 8) + 1)) * 0.125;
    }

    const host_acts = try allocator.alloc(f32, num_instances * in_dim);
    defer allocator.free(host_acts);
    for (host_acts) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;

    var d_counts = try DeviceBuffer(i32).init(dev, num_experts);
    defer d_counts.deinit();
    try d_counts.copyFromHost(host_counts);

    var d_offsets = try DeviceBuffer(i32).init(dev, num_experts);
    defer d_offsets.deinit();
    try d_offsets.copyFromHost(host_offsets);

    var d_sorted_inst = try DeviceBuffer(i32).init(dev, num_instances);
    defer d_sorted_inst.deinit();
    try d_sorted_inst.copyFromHost(host_sorted_inst);

    var d_inst_to_sorted = try DeviceBuffer(i32).init(dev, num_instances);
    defer d_inst_to_sorted.deinit();
    try d_inst_to_sorted.copyFromHost(host_inst_to_sorted);

    var d_top_weights = try DeviceBuffer(f32).init(dev, num_instances);
    defer d_top_weights.deinit();
    try d_top_weights.copyFromHost(host_top_weights);

    var d_acts = try DeviceBuffer(f32).init(dev, host_acts.len);
    defer d_acts.deinit();
    try d_acts.copyFromHost(host_acts);

    // 3. Run reference path: moeGemvDownSorted + moeAccumulateScaledSorted
    var d_down_buf = try DeviceBuffer(f32).init(dev, num_instances * out_dim);
    defer d_down_buf.deinit();

    try moeGemvDownSorted(
        d_w.slice(),
        8, // Q8_0
        d_acts.slice(),
        d_counts.slice(),
        d_offsets.slice(),
        d_sorted_inst.slice(),
        d_down_buf.slice(),
        num_tokens,
        in_dim,
        out_dim,
        null,
    );

    var d_accum_ref = try DeviceBuffer(f32).init(dev, num_tokens * out_dim);
    defer d_accum_ref.deinit();

    try moeAccumulateScaledSorted(
        d_accum_ref.slice(),
        d_down_buf.slice(),
        d_inst_to_sorted.slice(),
        d_top_weights.slice(),
        num_tokens,
        out_dim,
        null,
    );

    // 4. Run fused path: moeGemvDownAccumSorted
    var d_accum_fused = try DeviceBuffer(f32).init(dev, num_tokens * out_dim);
    defer d_accum_fused.deinit();

    try moeGemvDownAccumSorted(
        d_w.slice(),
        8, // Q8_0
        d_acts.slice(),
        d_counts.slice(),
        d_offsets.slice(),
        d_sorted_inst.slice(),
        d_top_weights.slice(),
        d_accum_fused.slice(),
        num_tokens,
        in_dim,
        out_dim,
        null,
    );

    // 5. Run persistent path: moeGemvDownAccumPersistent (Phase 13 WP2)
    var d_active_exp = try DeviceBuffer(i32).init(dev, num_experts);
    defer d_active_exp.deinit();
    var host_active_exp: [num_experts]i32 = @splat(0);
    @memcpy(host_active_exp[0..num_instances], host_sorted_inst[0..num_instances]);
    try d_active_exp.copyFromHost(&host_active_exp);

    var d_num_active = try DeviceBuffer(i32).init(dev, 1);
    defer d_num_active.deinit();
    const init_active = [_]i32{@intCast(num_instances)};
    try d_num_active.copyFromHost(&init_active);

    var d_work_counter = try DeviceBuffer(i32).init(dev, 1);
    defer d_work_counter.deinit();

    var d_accum_persistent = try DeviceBuffer(f32).init(dev, num_tokens * out_dim);
    defer d_accum_persistent.deinit();

    try moeGemvDownAccumPersistent(
        d_w.slice(),
        8, // Q8_0
        d_acts.slice(),
        d_counts.slice(),
        d_offsets.slice(),
        d_sorted_inst.slice(),
        d_top_weights.slice(),
        d_active_exp.slice(),
        d_num_active.slice(),
        d_work_counter.slice(),
        d_accum_persistent.slice(),
        num_tokens,
        in_dim,
        out_dim,
        null,
    );

    // 6. Compare outputs
    const ref_host = try allocator.alloc(f32, num_tokens * out_dim);
    defer allocator.free(ref_host);
    try d_accum_ref.copyToHost(ref_host);

    const fused_host = try allocator.alloc(f32, num_tokens * out_dim);
    defer allocator.free(fused_host);
    try d_accum_fused.copyToHost(fused_host);

    const pers_host = try allocator.alloc(f32, num_tokens * out_dim);
    defer allocator.free(pers_host);
    try d_accum_persistent.copyToHost(pers_host);

    var max_diff: f32 = 0.0;
    var max_diff_pers: f32 = 0.0;
    for (ref_host, fused_host, pers_host) |r, f, p| {
        const diff = @abs(r - f);
        if (diff > max_diff) max_diff = diff;
        try std.testing.expect(diff <= 1e-4 + 1e-4 * @abs(r));

        const diff_p = @abs(r - p);
        if (diff_p > max_diff_pers) max_diff_pers = diff_p;
        try std.testing.expect(diff_p <= 1e-4 + 1e-4 * @abs(r));
    }
    std.debug.print("[parity] moeGemvDownAccumSorted: max_diff = {e:.4} | moeGemvDownAccumPersistent: max_diff = {e:.4}\n", .{ max_diff, max_diff_pers });
}

test "CUDA Phase 15: fused intra-layer kernels numerical parity with CPU reference" {
    if (!cuda.is_enabled) return;
    const dev = requireHardwareDevice() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(0x1515);
    const rand = prng.random();

    const num_tokens: usize = 4;
    const dim: usize = 2816;
    const eps: f32 = 1e-6;

    // 1. Test fusedPreFfnNorm parity
    {
        var x_mid_buf = try DeviceBuffer(f32).init(dev, num_tokens * dim);
        defer x_mid_buf.deinit();
        var ffn_norm_buf = try DeviceBuffer(f32).init(dev, dim);
        defer ffn_norm_buf.deinit();
        var pre_ffw_norm_2_buf = try DeviceBuffer(f32).init(dev, dim);
        defer pre_ffw_norm_2_buf.deinit();
        var router_scale_buf = try DeviceBuffer(f32).init(dev, dim);
        defer router_scale_buf.deinit();

        var dense_in_buf = try DeviceBuffer(f32).init(dev, num_tokens * dim);
        defer dense_in_buf.deinit();
        var moe_in_buf = try DeviceBuffer(f32).init(dev, num_tokens * dim);
        defer moe_in_buf.deinit();
        var r_in_buf = try DeviceBuffer(f32).init(dev, num_tokens * dim);
        defer r_in_buf.deinit();

        const host_x_mid = try allocator.alloc(f32, num_tokens * dim);
        defer allocator.free(host_x_mid);
        for (host_x_mid) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        try x_mid_buf.copyFromHost(host_x_mid);

        const host_ffn_norm = try allocator.alloc(f32, dim);
        defer allocator.free(host_ffn_norm);
        for (host_ffn_norm) |*x| x.* = 1.0 + (rand.float(f32) * 0.1);
        try ffn_norm_buf.copyFromHost(host_ffn_norm);

        const host_pre_ffw_norm_2 = try allocator.alloc(f32, dim);
        defer allocator.free(host_pre_ffw_norm_2);
        for (host_pre_ffw_norm_2) |*x| x.* = 1.0 + (rand.float(f32) * 0.1);
        try pre_ffw_norm_2_buf.copyFromHost(host_pre_ffw_norm_2);

        const host_router_scale = try allocator.alloc(f32, dim);
        defer allocator.free(host_router_scale);
        for (host_router_scale) |*x| x.* = 1.0 + (rand.float(f32) * 0.1);
        try router_scale_buf.copyFromHost(host_router_scale);

        const root_size: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));

        try fusedPreFfnNorm(
            x_mid_buf.slice(),
            ffn_norm_buf.slice(),
            pre_ffw_norm_2_buf.slice(),
            router_scale_buf.slice(),
            dense_in_buf.slice(),
            moe_in_buf.slice(),
            r_in_buf.slice(),
            root_size,
            num_tokens,
            dim,
            eps,
            null,
        );

        const host_dense_in = try allocator.alloc(f32, num_tokens * dim);
        defer allocator.free(host_dense_in);
        try dense_in_buf.copyToHost(host_dense_in);

        const host_moe_in = try allocator.alloc(f32, num_tokens * dim);
        defer allocator.free(host_moe_in);
        try moe_in_buf.copyToHost(host_moe_in);

        const host_r_in = try allocator.alloc(f32, num_tokens * dim);
        defer allocator.free(host_r_in);
        try r_in_buf.copyToHost(host_r_in);

        // CPU reference computation
        var max_diff_dense: f32 = 0.0;
        var max_diff_moe: f32 = 0.0;
        var max_diff_r: f32 = 0.0;

        for (0..num_tokens) |t| {
            const row = host_x_mid[t * dim .. (t + 1) * dim];
            var sum_sq: f32 = 0.0;
            for (row) |v| sum_sq += v * v;
            const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(dim)) + eps);

            for (0..dim) |i| {
                const unnorm = row[i] * inv_rms;
                const expected_dense = unnorm * host_ffn_norm[i];
                const expected_moe = unnorm * host_pre_ffw_norm_2[i];
                const expected_r = unnorm * (host_router_scale[i] * root_size);

                const diff_d = @abs(expected_dense - host_dense_in[t * dim + i]);
                if (diff_d > max_diff_dense) max_diff_dense = diff_d;
                try std.testing.expect(diff_d <= 1e-5);

                const diff_m = @abs(expected_moe - host_moe_in[t * dim + i]);
                if (diff_m > max_diff_moe) max_diff_moe = diff_m;
                try std.testing.expect(diff_m <= 1e-5);

                const diff_r = @abs(expected_r - host_r_in[t * dim + i]);
                if (diff_r > max_diff_r) max_diff_r = diff_r;
                try std.testing.expect(diff_r <= 1e-5);
            }
        }
        std.debug.print("\n[parity] Fused Pre-FFN Norm: max_diff_dense = {e:.4}, max_diff_moe = {e:.4}, max_diff_r = {e:.4}\n", .{
            max_diff_dense, max_diff_moe, max_diff_r,
        });
    }

    // 2. Test fusedPostFfnCombine parity
    {
        var moe_accum_buf = try DeviceBuffer(f32).init(dev, num_tokens * dim);
        defer moe_accum_buf.deinit();
        var post_ffw_norm_2_buf = try DeviceBuffer(f32).init(dev, dim);
        defer post_ffw_norm_2_buf.deinit();
        var h1_buf = try DeviceBuffer(f32).init(dev, num_tokens * dim);
        defer h1_buf.deinit();
        var post_ffw_norm_buf = try DeviceBuffer(f32).init(dev, dim);
        defer post_ffw_norm_buf.deinit();
        var x_mid_buf = try DeviceBuffer(f32).init(dev, num_tokens * dim);
        defer x_mid_buf.deinit();
        var out_buf = try DeviceBuffer(f32).init(dev, num_tokens * dim);
        defer out_buf.deinit();

        const host_moe_accum = try allocator.alloc(f32, num_tokens * dim);
        defer allocator.free(host_moe_accum);
        for (host_moe_accum) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        try moe_accum_buf.copyFromHost(host_moe_accum);

        const host_p2 = try allocator.alloc(f32, dim);
        defer allocator.free(host_p2);
        for (host_p2) |*x| x.* = 1.0 + (rand.float(f32) * 0.1);
        try post_ffw_norm_2_buf.copyFromHost(host_p2);

        const host_h1 = try allocator.alloc(f32, num_tokens * dim);
        defer allocator.free(host_h1);
        for (host_h1) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        try h1_buf.copyFromHost(host_h1);

        const host_pn = try allocator.alloc(f32, dim);
        defer allocator.free(host_pn);
        for (host_pn) |*x| x.* = 1.0 + (rand.float(f32) * 0.1);
        try post_ffw_norm_buf.copyFromHost(host_pn);

        const host_x_mid = try allocator.alloc(f32, num_tokens * dim);
        defer allocator.free(host_x_mid);
        for (host_x_mid) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        try x_mid_buf.copyFromHost(host_x_mid);

        try fusedPostFfnCombine(
            moe_accum_buf.slice(),
            post_ffw_norm_2_buf.slice(),
            h1_buf.slice(),
            post_ffw_norm_buf.slice(),
            x_mid_buf.slice(),
            out_buf.slice(),
            0,
            num_tokens,
            dim,
            1.0,
            1.0,
            eps,
            null,
        );

        const host_out = try allocator.alloc(f32, num_tokens * dim);
        defer allocator.free(host_out);
        try out_buf.copyToHost(host_out);

        var max_diff_post: f32 = 0.0;
        for (0..num_tokens) |t| {
            const row_moe = host_moe_accum[t * dim .. (t + 1) * dim];
            var sum_sq_m: f32 = 0.0;
            for (row_moe) |v| sum_sq_m += v * v;
            const inv_rms_m = 1.0 / @sqrt(sum_sq_m / @as(f32, @floatFromInt(dim)) + eps);

            const row_h1 = host_h1[t * dim .. (t + 1) * dim];
            var comb = try allocator.alloc(f32, dim);
            defer allocator.free(comb);
            var sum_sq_c: f32 = 0.0;
            for (0..dim) |i| {
                const h2 = row_moe[i] * inv_rms_m * host_p2[i];
                comb[i] = row_h1[i] + h2;
                sum_sq_c += comb[i] * comb[i];
            }
            const inv_rms_c = 1.0 / @sqrt(sum_sq_c / @as(f32, @floatFromInt(dim)) + eps);

            const row_xm = host_x_mid[t * dim .. (t + 1) * dim];
            for (0..dim) |i| {
                const post_ffn = comb[i] * inv_rms_c * host_pn[i];
                const expected = row_xm[i] + post_ffn;
                const diff = @abs(expected - host_out[t * dim + i]);
                if (diff > max_diff_post) max_diff_post = diff;
                try std.testing.expect(diff <= 1e-5);
            }
        }
        std.debug.print("[parity] Fused Post-FFN Combine: max_diff = {e:.4}\n", .{max_diff_post});
    }

    // 3. Test fusedHeadNormRope parity
    {
        const num_heads: usize = 16;
        const head_dim: usize = 256;
        const total_elems = num_tokens * num_heads * head_dim;

        var vec_buf = try DeviceBuffer(f32).init(dev, total_elems);
        defer vec_buf.deinit();
        var weight_buf = try DeviceBuffer(f32).init(dev, head_dim);
        defer weight_buf.deinit();

        const host_vec = try allocator.alloc(f32, total_elems);
        defer allocator.free(host_vec);
        for (host_vec) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        try vec_buf.copyFromHost(host_vec);

        const host_weight = try allocator.alloc(f32, head_dim);
        defer allocator.free(host_weight);
        for (host_weight) |*x| x.* = 1.0 + (rand.float(f32) * 0.1);
        try weight_buf.copyFromHost(host_weight);

        const theta: f32 = 10000.0;

        try fusedHeadNormRope(
            vec_buf.slice(),
            weight_buf.slice(),
            num_tokens,
            num_heads,
            head_dim,
            0,
            theta,
            null,
            eps,
            null,
        );

        const host_out = try allocator.alloc(f32, total_elems);
        defer allocator.free(host_out);
        try vec_buf.copyToHost(host_out);

        var max_diff_rope: f32 = 0.0;
        const half = head_dim / 2;

        for (0..num_tokens) |t| {
            for (0..num_heads) |h| {
                const head_idx = (t * num_heads + h) * head_dim;
                const head_slice = host_vec[head_idx .. head_idx + head_dim];

                var sum_sq: f32 = 0.0;
                for (head_slice) |v| sum_sq += v * v;
                const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(head_dim)) + eps);

                for (0..half) |i| {
                    const v0 = (head_slice[i] * inv_rms) * host_weight[i];
                    const v1 = (head_slice[half + i] * inv_rms) * host_weight[half + i];

                    const exp_val = 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim));
                    const freq = 1.0 / std.math.pow(f32, theta, exp_val);
                    const angle = @as(f32, @floatFromInt(t)) * freq;
                    const cos_val = @cos(angle);
                    const sin_val = @sin(angle);

                    const exp_v0 = v0 * cos_val - v1 * sin_val;
                    const exp_v1 = v0 * sin_val + v1 * cos_val;

                    const diff0 = @abs(exp_v0 - host_out[head_idx + i]);
                    if (diff0 > max_diff_rope) max_diff_rope = diff0;
                    try std.testing.expect(diff0 <= 1e-5);

                    const diff1 = @abs(exp_v1 - host_out[head_idx + half + i]);
                    if (diff1 > max_diff_rope) max_diff_rope = diff1;
                    try std.testing.expect(diff1 <= 1e-5);
                }
            }
        }
        std.debug.print("[parity] Fused Head Norm RoPE (head_dim=256): max_diff = {e:.4}\n", .{max_diff_rope});
    }

    // R42-03: Exercise wide head_dim = 1024 (16 warps) to verify shared reduction array bounds (s_warp_sums[32])
    {
        const num_heads: usize = 4;
        const head_dim: usize = 1024;
        const total_elems = num_tokens * num_heads * head_dim;

        var vec_buf = try DeviceBuffer(f32).init(dev, total_elems);
        defer vec_buf.deinit();
        var weight_buf = try DeviceBuffer(f32).init(dev, head_dim);
        defer weight_buf.deinit();

        const host_vec = try allocator.alloc(f32, total_elems);
        defer allocator.free(host_vec);
        for (host_vec) |*x| x.* = (rand.float(f32) * 2.0) - 1.0;
        try vec_buf.copyFromHost(host_vec);

        const host_weight = try allocator.alloc(f32, head_dim);
        defer allocator.free(host_weight);
        for (host_weight) |*x| x.* = 1.0 + (rand.float(f32) * 0.1);
        try weight_buf.copyFromHost(host_weight);

        const theta: f32 = 10000.0;

        try fusedHeadNormRope(
            vec_buf.slice(),
            weight_buf.slice(),
            num_tokens,
            num_heads,
            head_dim,
            0,
            theta,
            null,
            eps,
            null,
        );

        const host_out = try allocator.alloc(f32, total_elems);
        defer allocator.free(host_out);
        try vec_buf.copyToHost(host_out);

        var max_diff_rope: f32 = 0.0;
        const half = head_dim / 2;

        for (0..num_tokens) |t| {
            for (0..num_heads) |h| {
                const head_idx = (t * num_heads + h) * head_dim;
                const head_slice = host_vec[head_idx .. head_idx + head_dim];

                var sum_sq: f32 = 0.0;
                for (head_slice) |v| sum_sq += v * v;
                const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(head_dim)) + eps);

                for (0..half) |i| {
                    const v0 = (head_slice[i] * inv_rms) * host_weight[i];
                    const v1 = (head_slice[half + i] * inv_rms) * host_weight[half + i];

                    const exp_val = 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim));
                    const freq = 1.0 / std.math.pow(f32, theta, exp_val);
                    const angle = @as(f32, @floatFromInt(t)) * freq;
                    const cos_val = @cos(angle);
                    const sin_val = @sin(angle);

                    const exp_v0 = v0 * cos_val - v1 * sin_val;
                    const exp_v1 = v0 * sin_val + v1 * cos_val;

                    const diff0 = @abs(exp_v0 - host_out[head_idx + i]);
                    if (diff0 > max_diff_rope) max_diff_rope = diff0;
                    try std.testing.expect(diff0 <= 1e-5);

                    const diff1 = @abs(exp_v1 - host_out[head_idx + half + i]);
                    if (diff1 > max_diff_rope) max_diff_rope = diff1;
                    try std.testing.expect(diff1 <= 1e-5);
                }
            }
        }
        std.debug.print("[parity] Fused Head Norm RoPE (head_dim=1024): max_diff = {e:.4}\n", .{max_diff_rope});
    }
}


//! Zeno — CUDA Backend Host Interface & Device Topology Discovery (Stage 1).
//!
//! Provides idiomatic Zig abstractions over the native CUDA C ABI runtime layer.
//! When built with `-Dcuda=false` (default), all functions return error.CudaDisabledAtBuild
//! without requiring any CUDA libraries or toolchains.

const std = @import("std");
const build_options = @import("build_options");

/// Indicates whether CUDA backend support was compiled into this binary.
pub const is_enabled = if (@hasDecl(build_options, "enable_cuda")) build_options.enable_cuda else false;

/// Target compute architecture configured at build time (e.g. "sm_86").
pub const target_arch = if (@hasDecl(build_options, "cuda_arch")) build_options.cuda_arch else "none";

/// Explicit error set mapping CUDA runtime and C ABI return codes.
pub const CudaError = error{
    NoDevice,
    DriverError,
    InvalidParam,
    GetDeviceFailed,
    DevicePropFailed,
    SetDeviceFailed,
    MemInfoFailed,
    PeerQueryFailed,
    RestoreDeviceFailed,
    OutOfMemory,
    CopyFailed,
    StreamFailed,
    KernelLaunchFailed,
    InvalidDevice,
    DeviceMismatch,
    UnknownError,
    CudaDisabledAtBuild,
};

/// Hardware characteristics and memory statistics for a single CUDA GPU.
pub const DeviceInfo = struct {
    device_id: i32,
    name_buf: [256]u8,
    name_len: usize,
    compute_major: i32,
    compute_minor: i32,
    total_memory_bytes: usize,
    free_memory_bytes: usize,
    multi_processor_count: i32,
    warp_size: i32,
    can_access_peer: [8]bool,

    /// Returns the device model name as a string slice.
    pub fn name(self: *const DeviceInfo) []const u8 {
        return self.name_buf[0..self.name_len];
    }
};

/// Host topology representation capturing all visible and reported CUDA devices.
pub const Topology = struct {
    total_visible_devices: usize,
    reported_devices: usize,
    devices: [8]DeviceInfo,
};

// C ABI extern declarations (compiled only when CUDA is enabled)
pub const c = if (is_enabled) struct {
    pub const ZENO_CUDA_MAX_DEVICES = 8;

    pub const ZenoCudaDeviceInfo = extern struct {
        device_id: c_int,
        name: [256]u8,
        major: c_int,
        minor: c_int,
        total_memory_bytes: usize,
        free_memory_bytes: usize,
        multi_processor_count: c_int,
        warp_size: c_int,
        can_access_peer: [ZENO_CUDA_MAX_DEVICES]c_int,
    };

    pub const ZenoCudaTopology = extern struct {
        total_visible_devices: c_int,
        reported_devices: c_int,
        devices: [ZENO_CUDA_MAX_DEVICES]ZenoCudaDeviceInfo,
    };

    pub extern "c" fn zeno_cuda_get_device_count(out_count: *c_int) c_int;
    pub extern "c" fn zeno_cuda_get_topology(out_topo: *ZenoCudaTopology) c_int;
    pub extern "c" fn zeno_cuda_get_error_string(error_code: c_int) [*:0]const u8;

    // Memory & Stream ABI
    pub extern "c" fn zeno_cuda_malloc(out_ptr: *?*anyopaque, bytes: usize, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_free(ptr: ?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_memcpy_h2d(dst: ?*anyopaque, src: ?*const anyopaque, bytes: usize, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_memcpy_d2h(dst: ?*anyopaque, src: ?*const anyopaque, bytes: usize, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_memcpy_d2d(dst: ?*anyopaque, src: ?*const anyopaque, bytes: usize, device_id: c_int, stream: ?*anyopaque) c_int;
    pub extern "c" fn zeno_cuda_memset(dst: ?*anyopaque, value: c_int, bytes: usize, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_stream_create(out_stream: *?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_stream_destroy(stream: ?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_stream_sync(stream: ?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_graph_begin_capture(stream: ?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_graph_end_capture(stream: ?*anyopaque, out_graph_exec: *?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_graph_launch(graph_exec: ?*anyopaque, stream: ?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_graph_destroy(graph_exec: ?*anyopaque, device_id: c_int) c_int;

    // Fused Elementwise Kernels ABI
    pub extern "c" fn zeno_cuda_rms_norm(
        x: [*]const f32,
        weight: ?[*]const f32,
        out: [*]f32,
        num_rows: c_int,
        dim: c_int,
        eps: f32,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;
    pub extern "c" fn zeno_cuda_geglu(
        gate: [*]const f32,
        up: [*]const f32,
        out: [*]f32,
        count: usize,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;
    pub extern "c" fn zeno_cuda_rope(
        vec: [*]f32,
        num_tokens: c_int,
        num_heads: c_int,
        head_dim: c_int,
        pos_offset: c_int,
        theta: f32,
        freq_factors: ?[*]const f32,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;
    pub extern "c" fn zeno_cuda_add_residual(
        x: [*]f32,
        residual: [*]const f32,
        count: usize,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;
    pub extern "c" fn zeno_cuda_scale(
        x: [*]f32,
        scale: f32,
        count: usize,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    // Stage 3 Quantized GEMV ABI
    pub extern "c" fn zeno_cuda_gemv_q8_0(
        W: *const anyopaque,
        x: [*]const f32,
        out: [*]f32,
        out_dim: c_int,
        in_dim: c_int,
        batch_size: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;
    pub extern "c" fn zeno_cuda_gemv_q4k(
        W: *const anyopaque,
        x: [*]const f32,
        out: [*]f32,
        out_dim: c_int,
        in_dim: c_int,
        batch_size: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;
    pub extern "c" fn zeno_cuda_gemv_q6k(
        W: *const anyopaque,
        x: [*]const f32,
        out: [*]f32,
        out_dim: c_int,
        in_dim: c_int,
        batch_size: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;
    pub extern "c" fn zeno_cuda_gemv_q5_0(
        W: *const anyopaque,
        x: [*]const f32,
        out: [*]f32,
        out_dim: c_int,
        in_dim: c_int,
        batch_size: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;
    pub extern "c" fn zeno_cuda_gemv_f32(
        W: [*]const f32,
        x: [*]const f32,
        out: [*]f32,
        out_dim: c_int,
        in_dim: c_int,
        batch_size: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    // Stage 4 Unified Attention & MoE Execution ABI
    pub extern "c" fn zeno_cuda_attention(
        q: [*]const f32,
        k: [*]const f32,
        v: [*]const f32,
        out: [*]f32,
        num_tokens: c_int,
        prompt_len: c_int,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        stride_q: c_int,
        stride_k: c_int,
        stride_v: c_int,
        stride_out: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    pub extern "c" fn zeno_cuda_moe_router_prep(
        r_in: [*]f32,
        router_scale: [*]const f32,
        root_size: f32,
        num_tokens: c_int,
        dim: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    pub extern "c" fn zeno_cuda_moe_router_topk(
        router_logits: [*]const f32,
        exp_down_scale: ?[*]const f32,
        out_indices: [*]c_int,
        out_weights: [*]f32,
        num_tokens: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    pub extern "c" fn zeno_cuda_accumulate_scaled(
        accum: [*]f32,
        x: [*]const f32,
        scale: f32,
        count: usize,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    pub extern "c" fn zeno_cuda_layer_output(
        out: [*]f32,
        x_mid: [*]const f32,
        post_ffn: [*]const f32,
        prompt_len: c_int,
        num_tokens: c_int,
        dim: c_int,
        enc_scale: f32,
        layer_scale: f32,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    pub extern "c" fn zeno_cuda_memcpy_peer(
        dst: ?*anyopaque,
        dst_device_id: c_int,
        src: ?*const anyopaque,
        src_device_id: c_int,
        bytes: usize,
    ) c_int;

    pub extern "c" fn zeno_cuda_memcpy_peer_async(
        dst: ?*anyopaque,
        dst_device_id: c_int,
        src: ?*const anyopaque,
        src_device_id: c_int,
        bytes: usize,
        stream: ?*anyopaque,
    ) c_int;

    pub extern "c" fn zeno_cuda_event_create(out_event: *?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_event_destroy(event: ?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_event_record(event: ?*anyopaque, stream: ?*anyopaque, device_id: c_int) c_int;
    pub extern "c" fn zeno_cuda_stream_wait_event(stream: ?*anyopaque, event: ?*anyopaque, device_id: c_int) c_int;

    pub extern "c" fn zeno_cuda_moe_gemv_gate_up_batched(
        all_expert_weights: [*]const u8,
        tokens_in: [*]const f32,
        top_indices: [*]const c_int,
        out_gate_up: [*]f32,
        num_tokens: c_int,
        in_dim: c_int,
        out_dim: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    pub extern "c" fn zeno_cuda_moe_geglu_batched(
        gate_up: [*]const f32,
        act_out: [*]f32,
        num_instances: c_int,
        half_dim: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    pub extern "c" fn zeno_cuda_moe_gemv_down_batched(
        all_expert_weights: [*]const u8,
        expert_type: c_int,
        acts_in: [*]const f32,
        top_indices: [*]const c_int,
        out_down: [*]f32,
        num_tokens: c_int,
        in_dim: c_int,
        out_dim: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;

    pub extern "c" fn zeno_cuda_moe_accumulate_scaled_batched(
        moe_accum: [*]f32,
        down_out: [*]const f32,
        top_weights: [*]const f32,
        num_tokens: c_int,
        dim: c_int,
        device_id: c_int,
        stream: ?*anyopaque,
    ) c_int;
} else struct {};

pub fn mapCudaErrorCode(rc: c_int) CudaError {
    return switch (rc) {
        0 => unreachable, // Success
        1 => error.NoDevice,
        2 => error.DriverError,
        3 => error.InvalidParam,
        4 => error.GetDeviceFailed,
        5 => error.DevicePropFailed,
        6 => error.SetDeviceFailed,
        7 => error.MemInfoFailed,
        8 => error.PeerQueryFailed,
        9 => error.RestoreDeviceFailed,
        10 => error.OutOfMemory,
        11 => error.CopyFailed,
        12 => error.StreamFailed,
        13 => error.KernelLaunchFailed,
        14 => error.InvalidDevice,
        15 => error.DeviceMismatch,
        else => error.UnknownError,
    };
}

/// Returns the number of visible CUDA-capable devices on the host.
pub fn getDeviceCount() CudaError!usize {
    if (!is_enabled) {
        return error.CudaDisabledAtBuild;
    }
    var count: c_int = 0;
    const rc = c.zeno_cuda_get_device_count(&count);
    if (rc != 0) {
        return mapCudaErrorCode(rc);
    }
    return @intCast(count);
}

/// Queries and returns full GPU topology and inter-device P2P access capabilities.
pub fn getTopology() CudaError!Topology {
    if (!is_enabled) {
        return error.CudaDisabledAtBuild;
    }

    var raw_topo: c.ZenoCudaTopology = undefined;
    const rc = c.zeno_cuda_get_topology(&raw_topo);
    if (rc != 0) {
        return mapCudaErrorCode(rc);
    }

    var topo: Topology = .{
        .total_visible_devices = @intCast(raw_topo.total_visible_devices),
        .reported_devices = @intCast(raw_topo.reported_devices),
        .devices = undefined,
    };

    for (0..topo.reported_devices) |i| {
        const raw_dev = &raw_topo.devices[i];
        var dev: DeviceInfo = .{
            .device_id = raw_dev.device_id,
            .name_buf = undefined,
            .name_len = 0,
            .compute_major = raw_dev.major,
            .compute_minor = raw_dev.minor,
            .total_memory_bytes = raw_dev.total_memory_bytes,
            .free_memory_bytes = raw_dev.free_memory_bytes,
            .multi_processor_count = raw_dev.multi_processor_count,
            .warp_size = raw_dev.warp_size,
            .can_access_peer = [_]bool{ false, false, false, false, false, false, false, false },
        };

        // Find null terminator for name length safely
        const raw_name = &raw_dev.name;
        var name_end: usize = 0;
        while (name_end < raw_name.len and raw_name[name_end] != 0) : (name_end += 1) {}
        @memcpy(dev.name_buf[0..name_end], raw_name[0..name_end]);
        dev.name_len = name_end;

        for (0..topo.reported_devices) |j| {
            dev.can_access_peer[j] = (raw_dev.can_access_peer[j] != 0);
        }

        topo.devices[i] = dev;
    }

    return topo;
}

test "cuda module build disabled check" {
    if (!is_enabled) {
        try std.testing.expectError(error.CudaDisabledAtBuild, getDeviceCount());
        try std.testing.expectError(error.CudaDisabledAtBuild, getTopology());
    }
}

test "cuda module topology query when enabled" {
    if (is_enabled) {
        const topo = getTopology() catch |err| switch (err) {
            error.NoDevice => return error.SkipZigTest,
            else => return err,
        };
        try std.testing.expect(topo.total_visible_devices >= 1);
        try std.testing.expect(topo.reported_devices >= 1);
        const dev0 = &topo.devices[0];
        try std.testing.expect(dev0.name().len > 0);
        try std.testing.expect(dev0.total_memory_bytes > 0);
        try std.testing.expect(dev0.compute_major >= 1);
    }
}

//! Zeno — CUDA Execution Pipeline & Multi-GPU Layer Splitting (Stage 5).
//!
//! Orchestrates device-resident model weights, sequence activation workspaces,
//! multi-GPU layer partitioning (Strategy A from specs/multi-gpu.md), and complete
//! structured read forward evaluation matching the CPU reference contract.

const std = @import("std");
const root = @import("root.zig");
const cuda = @import("cuda.zig");
const cuda_mem = @import("cuda_mem.zig");
const model_mod = @import("model.zig");
const canvas_mod = @import("canvas.zig");
const quant = @import("quant.zig");
const ops = @import("ops.zig");
const infer = @import("infer.zig");
const gguf = @import("gguf.zig");
const decision = @import("decision.zig");

pub const is_enabled = cuda.is_enabled;
pub const ModelConfig = root.ModelConfig;
pub const DiffusionGemmaModel = model_mod.DiffusionGemmaModel;
pub const TransformerLayer = model_mod.TransformerLayer;
pub const Canvas = canvas_mod.Canvas;
pub const DeviceBuffer = cuda_mem.DeviceBuffer;
pub const DeviceSlice = cuda_mem.DeviceSlice;
pub const CudaStream = cuda_mem.CudaStream;
pub const StageTimings = decision.StageTimings;

pub fn getTensorByteSize(tensor: gguf.TensorInfo) usize {
    return gguf.getTensorByteSize(tensor) catch |err| {
        std.debug.panic("Failed to compute tensor byte size for '{s}': {s}", .{ tensor.name, @errorName(err) });
    };
}

pub fn getExactTensorSlice(cpu_model: DiffusionGemmaModel, tensor: gguf.TensorInfo) []const u8 {
    const raw = cpu_model.gguf_file.getTensorSlice(tensor);
    const size = getTensorByteSize(tensor);
    if (raw.len < size) {
        std.debug.panic("Truncated tensor slice for '{s}': available={d}, required={d}", .{ tensor.name, raw.len, size });
    }
    return raw[0..size];
}

pub const GGML_TYPE_Q4_K: u32 = 12;

pub fn gemvTensorCuda(
    ggml_type: u32,
    w: DeviceSlice(u8),
    x: DeviceSlice(f32),
    out: DeviceSlice(f32),
    out_dim: usize,
    in_dim: usize,
    batch_size: usize,
    stream: ?*const CudaStream,
) cuda.CudaError!void {
    return switch (ggml_type) {
        0 => {
            const w_f32 = DeviceSlice(f32){
                .ptr = @ptrCast(@alignCast(w.ptr)),
                .len = w.len / @sizeOf(f32),
                .device_id = w.device_id,
            };
            return cuda_mem.gemvF32(w_f32, x, out, out_dim, in_dim, batch_size, stream);
        },
        6 => cuda_mem.gemvQ5_0(w, x, out, out_dim, in_dim, batch_size, stream),
        8 => cuda_mem.gemvQ8_0(w, x, out, out_dim, in_dim, batch_size, stream),
        12 => cuda_mem.gemvQ4_K(w, x, out, out_dim, in_dim, batch_size, stream),
        14 => cuda_mem.gemvQ6_K(w, x, out, out_dim, in_dim, batch_size, stream),
        else => error.InvalidParam,
    };
}

pub const ExecutionTopology = enum {
    single_gpu,
    dual_gpu_split,
};

/// Device-resident weights for a single transformer layer.
pub const CudaLayerWeights = struct {
    device_id: i32,
    layer_idx: usize,
    is_global: bool,
    q_dim: usize,
    k_dim: usize,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,

    // Attention sub-layer
    attn_norm: DeviceBuffer(f32),
    q_norm: DeviceBuffer(f32),
    k_norm: DeviceBuffer(f32),
    q_weight: DeviceBuffer(u8),
    q_weight_type: u32,
    k_weight: DeviceBuffer(u8),
    k_weight_type: u32,
    v_weight: ?DeviceBuffer(u8),
    v_weight_type: u32,
    o_weight: DeviceBuffer(u8),
    o_weight_type: u32,
    post_attention_norm: DeviceBuffer(f32),

    // Dense FFN sub-layer
    ffn_norm: DeviceBuffer(f32),
    ffn_gate: DeviceBuffer(u8),
    ffn_gate_type: u32,
    ffn_up: DeviceBuffer(u8),
    ffn_up_type: u32,
    ffn_down: DeviceBuffer(u8),
    ffn_down_type: u32,
    post_ffw_norm_1: DeviceBuffer(f32),

    // MoE sub-layer
    pre_ffw_norm_2: DeviceBuffer(f32),
    router_gate: DeviceBuffer(u8),
    router_gate_type: u32,
    router_scale: DeviceBuffer(f32),
    expert_gate_up: DeviceBuffer(u8),
    expert_gate_up_type: u32,
    expert_down: DeviceBuffer(u8),
    expert_down_type: u32,
    expert_down_scale: DeviceBuffer(f32),
    post_ffw_norm_2: DeviceBuffer(f32),

    // Block output
    post_ffw_norm: DeviceBuffer(f32),
    layer_output_scale: f32,
    enc_layer_output_scale: f32,

    pub fn initFromModelLayer(cpu_model: DiffusionGemmaModel, layer_idx: usize, device_id: i32) !CudaLayerWeights {
        if (!is_enabled) return error.CudaDisabledAtBuild;
        const layer = cpu_model.layers[layer_idx];
        const is_global = layer.is_global;

        // Attention weights
        const attn_norm_slice = cpu_model.gguf_file.getTensorSlice(layer.attn_norm);
        var attn_norm = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer attn_norm.deinit();
        try attn_norm.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(attn_norm_slice.ptr)))[0..ModelConfig.embedding_dim]);

        const head_dim = layer.q_norm.dimensions[0];
        const q_norm_slice = cpu_model.gguf_file.getTensorSlice(layer.q_norm);
        var q_norm = try DeviceBuffer(f32).init(device_id, head_dim);
        errdefer q_norm.deinit();
        try q_norm.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(q_norm_slice.ptr)))[0..head_dim]);

        const k_norm_slice = cpu_model.gguf_file.getTensorSlice(layer.k_norm);
        var k_norm = try DeviceBuffer(f32).init(device_id, head_dim);
        errdefer k_norm.deinit();
        try k_norm.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(k_norm_slice.ptr)))[0..head_dim]);

        const q_w_slice = getExactTensorSlice(cpu_model, layer.q_weight);
        var q_weight = try DeviceBuffer(u8).init(device_id, q_w_slice.len);
        errdefer q_weight.deinit();
        try q_weight.copyFromHost(q_w_slice);

        const k_w_slice = getExactTensorSlice(cpu_model, layer.k_weight);
        var k_weight = try DeviceBuffer(u8).init(device_id, k_w_slice.len);
        errdefer k_weight.deinit();
        try k_weight.copyFromHost(k_w_slice);

        var v_weight: ?DeviceBuffer(u8) = null;
        if (!is_global) {
            const v_w_slice = getExactTensorSlice(cpu_model, layer.v_weight.?);
            var vw = try DeviceBuffer(u8).init(device_id, v_w_slice.len);
            errdefer vw.deinit();
            try vw.copyFromHost(v_w_slice);
            v_weight = vw;
        }

        const o_w_slice = getExactTensorSlice(cpu_model, layer.o_weight);
        var o_weight = try DeviceBuffer(u8).init(device_id, o_w_slice.len);
        errdefer o_weight.deinit();
        try o_weight.copyFromHost(o_w_slice);

        const post_attn_slice = cpu_model.gguf_file.getTensorSlice(layer.post_attention_norm);
        var post_attention_norm = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer post_attention_norm.deinit();
        try post_attention_norm.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(post_attn_slice.ptr)))[0..ModelConfig.embedding_dim]);

        // Dense FFN weights
        const ffn_norm_slice = cpu_model.gguf_file.getTensorSlice(layer.ffn_norm);
        var ffn_norm = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer ffn_norm.deinit();
        try ffn_norm.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(ffn_norm_slice.ptr)))[0..ModelConfig.embedding_dim]);

        const ffn_gate_slice = getExactTensorSlice(cpu_model, layer.ffn_gate);
        var ffn_gate = try DeviceBuffer(u8).init(device_id, ffn_gate_slice.len);
        errdefer ffn_gate.deinit();
        try ffn_gate.copyFromHost(ffn_gate_slice);

        const ffn_up_slice = getExactTensorSlice(cpu_model, layer.ffn_up);
        var ffn_up = try DeviceBuffer(u8).init(device_id, ffn_up_slice.len);
        errdefer ffn_up.deinit();
        try ffn_up.copyFromHost(ffn_up_slice);

        const ffn_down_slice = getExactTensorSlice(cpu_model, layer.ffn_down);
        var ffn_down = try DeviceBuffer(u8).init(device_id, ffn_down_slice.len);
        errdefer ffn_down.deinit();
        try ffn_down.copyFromHost(ffn_down_slice);

        const post_ffw_1_slice = cpu_model.gguf_file.getTensorSlice(layer.post_ffw_norm_1);
        var post_ffw_norm_1 = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer post_ffw_norm_1.deinit();
        try post_ffw_norm_1.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(post_ffw_1_slice.ptr)))[0..ModelConfig.embedding_dim]);

        // MoE weights
        const pre_ffw_2_slice = cpu_model.gguf_file.getTensorSlice(layer.pre_ffw_norm_2);
        var pre_ffw_norm_2 = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer pre_ffw_norm_2.deinit();
        try pre_ffw_norm_2.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(pre_ffw_2_slice.ptr)))[0..ModelConfig.embedding_dim]);

        const router_gate_slice = getExactTensorSlice(cpu_model, layer.router_gate);
        var router_gate = try DeviceBuffer(u8).init(device_id, router_gate_slice.len);
        errdefer router_gate.deinit();
        try router_gate.copyFromHost(router_gate_slice);

        const router_scale_slice = cpu_model.gguf_file.getTensorSlice(layer.router_scale);
        var router_scale = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer router_scale.deinit();
        try router_scale.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(router_scale_slice.ptr)))[0..ModelConfig.embedding_dim]);

        const exp_gate_up_slice = getExactTensorSlice(cpu_model, layer.expert_gate_up);
        var expert_gate_up = try DeviceBuffer(u8).init(device_id, exp_gate_up_slice.len);
        errdefer expert_gate_up.deinit();
        try expert_gate_up.copyFromHost(exp_gate_up_slice);

        const exp_down_slice = getExactTensorSlice(cpu_model, layer.expert_down);
        var expert_down = try DeviceBuffer(u8).init(device_id, exp_down_slice.len);
        errdefer expert_down.deinit();
        try expert_down.copyFromHost(exp_down_slice);

        const exp_down_scale_slice = cpu_model.gguf_file.getTensorSlice(layer.expert_down_scale);
        var expert_down_scale = try DeviceBuffer(f32).init(device_id, 128);
        errdefer expert_down_scale.deinit();
        try expert_down_scale.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(exp_down_scale_slice.ptr)))[0..128]);

        const post_ffw_2_slice = cpu_model.gguf_file.getTensorSlice(layer.post_ffw_norm_2);
        var post_ffw_norm_2 = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer post_ffw_norm_2.deinit();
        try post_ffw_norm_2.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(post_ffw_2_slice.ptr)))[0..ModelConfig.embedding_dim]);

        // Output block norm and scales
        const post_ffw_slice = cpu_model.gguf_file.getTensorSlice(layer.post_ffw_norm);
        var post_ffw_norm = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer post_ffw_norm.deinit();
        try post_ffw_norm.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(post_ffw_slice.ptr)))[0..ModelConfig.embedding_dim]);

        const layer_out_scale_slice = cpu_model.gguf_file.getTensorSlice(layer.layer_output_scale);
        const layer_out_scale = @as([*]const f32, @ptrCast(@alignCast(layer_out_scale_slice.ptr)))[0];

        const enc_layer_out_scale_slice = cpu_model.gguf_file.getTensorSlice(layer.enc_layer_output_scale);
        const enc_layer_out_scale = @as([*]const f32, @ptrCast(@alignCast(enc_layer_out_scale_slice.ptr)))[0];

        return CudaLayerWeights{
            .device_id = device_id,
            .layer_idx = layer_idx,
            .is_global = is_global,
            .q_dim = layer.q_weight.dimensions[1],
            .k_dim = layer.k_weight.dimensions[1],
            .head_dim = head_dim,
            .num_q_heads = layer.q_weight.dimensions[1] / head_dim,
            .num_kv_heads = layer.k_weight.dimensions[1] / head_dim,
            .attn_norm = attn_norm,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .q_weight = q_weight,
            .q_weight_type = layer.q_weight.ggml_type,
            .k_weight = k_weight,
            .k_weight_type = layer.k_weight.ggml_type,
            .v_weight = v_weight,
            .v_weight_type = if (layer.v_weight) |vw| vw.ggml_type else 0,
            .o_weight = o_weight,
            .o_weight_type = layer.o_weight.ggml_type,
            .post_attention_norm = post_attention_norm,
            .ffn_norm = ffn_norm,
            .ffn_gate = ffn_gate,
            .ffn_gate_type = layer.ffn_gate.ggml_type,
            .ffn_up = ffn_up,
            .ffn_up_type = layer.ffn_up.ggml_type,
            .ffn_down = ffn_down,
            .ffn_down_type = layer.ffn_down.ggml_type,
            .post_ffw_norm_1 = post_ffw_norm_1,
            .pre_ffw_norm_2 = pre_ffw_norm_2,
            .router_gate = router_gate,
            .router_gate_type = layer.router_gate.ggml_type,
            .router_scale = router_scale,
            .expert_gate_up = expert_gate_up,
            .expert_gate_up_type = layer.expert_gate_up.ggml_type,
            .expert_down = expert_down,
            .expert_down_type = layer.expert_down.ggml_type,
            .expert_down_scale = expert_down_scale,
            .post_ffw_norm_2 = post_ffw_norm_2,
            .post_ffw_norm = post_ffw_norm,
            .layer_output_scale = layer_out_scale,
            .enc_layer_output_scale = enc_layer_out_scale,
        };
    }

    pub fn deinit(self: *CudaLayerWeights) void {
        self.attn_norm.deinit();
        self.q_norm.deinit();
        self.k_norm.deinit();
        self.q_weight.deinit();
        self.k_weight.deinit();
        if (self.v_weight) |*vw| vw.deinit();
        self.o_weight.deinit();
        self.post_attention_norm.deinit();
        self.ffn_norm.deinit();
        self.ffn_gate.deinit();
        self.ffn_up.deinit();
        self.ffn_down.deinit();
        self.post_ffw_norm_1.deinit();
        self.pre_ffw_norm_2.deinit();
        self.router_gate.deinit();
        self.router_scale.deinit();
        self.expert_gate_up.deinit();
        self.expert_down.deinit();
        self.expert_down_scale.deinit();
        self.post_ffw_norm_2.deinit();
        self.post_ffw_norm.deinit();
    }
};

/// Partitioned device-resident model holding all 30 layers across 1 or 2 GPUs.
pub const CudaModel = struct {
    allocator: std.mem.Allocator,
    topology: ExecutionTopology,
    devices: [2]i32,
    layers: [ModelConfig.num_layers]CudaLayerWeights,
    output_norm: DeviceBuffer(f32),
    rope_freqs_dev0: ?DeviceBuffer(f32),
    rope_freqs_dev1: ?DeviceBuffer(f32),

    // Phase 9: Device-resident self-conditioning weights on Device 0
    sc_pre_norm: ?DeviceBuffer(f32),
    sc_gate: ?DeviceBuffer(u8),
    sc_up: ?DeviceBuffer(u8),
    sc_down: ?DeviceBuffer(u8),

    /// Returns true if all four self-conditioning tensor buffers are resident on Device 0.
    pub fn hasSelfConditioning(self: CudaModel) bool {
        return self.sc_pre_norm != null and
            self.sc_gate != null and
            self.sc_up != null and
            self.sc_down != null;
    }

    pub fn initFromCpuModel(allocator: std.mem.Allocator, cpu_model: DiffusionGemmaModel, force_single_gpu: bool) !*CudaModel {
        if (!is_enabled) return error.CudaDisabledAtBuild;
        const topo = try cuda.getTopology();
        if (topo.reported_devices == 0) return error.NoDevice;

        const use_multi_gpu = (topo.reported_devices >= 2 and !force_single_gpu);
        const topology: ExecutionTopology = if (use_multi_gpu) .dual_gpu_split else .single_gpu;
        const dev0 = topo.devices[0].device_id;
        const dev1 = if (use_multi_gpu) topo.devices[1].device_id else dev0;

        const model_ptr = try allocator.create(CudaModel);
        errdefer allocator.destroy(model_ptr);

        model_ptr.allocator = allocator;
        model_ptr.topology = topology;
        model_ptr.devices = [_]i32{ dev0, dev1 };
        model_ptr.rope_freqs_dev0 = null;
        model_ptr.rope_freqs_dev1 = null;
        model_ptr.sc_pre_norm = null;
        model_ptr.sc_gate = null;
        model_ptr.sc_up = null;
        model_ptr.sc_down = null;

        var loaded_layers: usize = 0;
        errdefer {
            for (0..loaded_layers) |l| {
                model_ptr.layers[l].deinit();
            }
            if (model_ptr.sc_down) |*b| b.deinit();
            if (model_ptr.sc_up) |*b| b.deinit();
            if (model_ptr.sc_gate) |*b| b.deinit();
            if (model_ptr.sc_pre_norm) |*b| b.deinit();
            if (model_ptr.rope_freqs_dev0) |*rf0| rf0.deinit();
            if (model_ptr.rope_freqs_dev1) |*rf1| rf1.deinit();
        }

        // Load rope_freqs if present
        if (cpu_model.rope_freqs) |rf| {
            const rf_slice = getExactTensorSlice(cpu_model, rf);
            const rf_count = rf_slice.len / @sizeOf(f32);
            var rf0 = try DeviceBuffer(f32).init(dev0, rf_count);
            {
                errdefer rf0.deinit();
                try rf0.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(rf_slice.ptr)))[0..rf_count]);
            }
            model_ptr.rope_freqs_dev0 = rf0;

            if (use_multi_gpu) {
                var rf1 = try DeviceBuffer(f32).init(dev1, rf_count);
                {
                    errdefer rf1.deinit();
                    try rf1.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(rf_slice.ptr)))[0..rf_count]);
                }
                model_ptr.rope_freqs_dev1 = rf1;
            }
        }

        // Load self-conditioning tensors onto Device 0 if present (Phase 9)
        if (cpu_model.self_cond_pre_norm != null and
            cpu_model.self_cond_gate != null and
            cpu_model.self_cond_up != null and
            cpu_model.self_cond_down != null)
        {
            const pre_norm_info = cpu_model.self_cond_pre_norm.?;
            const pre_norm_slice = getExactTensorSlice(cpu_model, pre_norm_info);
            var sc_pre_norm = try DeviceBuffer(f32).init(dev0, ModelConfig.embedding_dim);
            {
                errdefer sc_pre_norm.deinit();
                try sc_pre_norm.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(pre_norm_slice.ptr)))[0..ModelConfig.embedding_dim]);
            }
            model_ptr.sc_pre_norm = sc_pre_norm;

            const gate_info = cpu_model.self_cond_gate.?;
            const gate_slice = getExactTensorSlice(cpu_model, gate_info);
            var sc_gate = try DeviceBuffer(u8).init(dev0, gate_slice.len);
            {
                errdefer sc_gate.deinit();
                try sc_gate.copyFromHost(gate_slice);
            }
            model_ptr.sc_gate = sc_gate;

            const up_info = cpu_model.self_cond_up.?;
            const up_slice = getExactTensorSlice(cpu_model, up_info);
            var sc_up = try DeviceBuffer(u8).init(dev0, up_slice.len);
            {
                errdefer sc_up.deinit();
                try sc_up.copyFromHost(up_slice);
            }
            model_ptr.sc_up = sc_up;

            const down_info = cpu_model.self_cond_down.?;
            const down_slice = getExactTensorSlice(cpu_model, down_info);
            var sc_down = try DeviceBuffer(u8).init(dev0, down_slice.len);
            {
                errdefer sc_down.deinit();
                try sc_down.copyFromHost(down_slice);
            }
            model_ptr.sc_down = sc_down;
        }

        // Load all 30 layers with layer-splitting
        for (0..ModelConfig.num_layers) |l| {
            const target_dev = if (topology == .dual_gpu_split and l >= 15) dev1 else dev0;
            model_ptr.layers[l] = try CudaLayerWeights.initFromModelLayer(cpu_model, l, target_dev);
            loaded_layers += 1;
        }

        // Final output norm on the last execution device (dev1 for multi-gpu, dev0 for single)
        const final_dev = if (topology == .dual_gpu_split) dev1 else dev0;
        const out_norm_slice = getExactTensorSlice(cpu_model, cpu_model.output_norm);
        var out_norm = try DeviceBuffer(f32).init(final_dev, ModelConfig.embedding_dim);
        errdefer out_norm.deinit();
        try out_norm.copyFromHost(@as([*]const f32, @ptrCast(@alignCast(out_norm_slice.ptr)))[0..ModelConfig.embedding_dim]);
        model_ptr.output_norm = out_norm;

        return model_ptr;
    }

    pub fn deinit(self: *CudaModel) void {
        self.output_norm.deinit();
        for (&self.layers) |*l| {
            l.deinit();
        }
        if (self.sc_down) |*b| b.deinit();
        if (self.sc_up) |*b| b.deinit();
        if (self.sc_gate) |*b| b.deinit();
        if (self.sc_pre_norm) |*b| b.deinit();
        if (self.rope_freqs_dev0) |*rf0| rf0.deinit();
        if (self.rope_freqs_dev1) |*rf1| rf1.deinit();
        self.allocator.destroy(self);
    }
};

/// Per-device activation workspace pre-allocated to maximum sequence length.
pub const CudaDeviceWorkspace = struct {
    device_id: i32,
    stream: CudaStream,
    max_seq_len: usize,

    // Hidden activations (double-buffered)
    emb_buf_0: DeviceBuffer(f32),
    emb_buf_1: DeviceBuffer(f32),
    current_emb: DeviceBuffer(f32),
    next_emb: DeviceBuffer(f32),
    norm_x: DeviceBuffer(f32),

    // Attention buffers
    q_buf: DeviceBuffer(f32),
    k_buf: DeviceBuffer(f32),
    v_buf: DeviceBuffer(f32),
    head_out: DeviceBuffer(f32),
    attn_out: DeviceBuffer(f32),

    // Dense FFN buffers
    dense_in: DeviceBuffer(f32),
    gate_buf: DeviceBuffer(f32),
    up_buf: DeviceBuffer(f32),
    geglu_buf: DeviceBuffer(f32),
    dense_out: DeviceBuffer(f32),
    h1: DeviceBuffer(f32),

    // MoE buffers
    r_in: DeviceBuffer(f32),
    router_logits: DeviceBuffer(f32),
    top_indices: DeviceBuffer(i32),
    top_weights: DeviceBuffer(f32),
    moe_in: DeviceBuffer(f32),
    exp_act_buf: DeviceBuffer(f32),
    moe_accum: DeviceBuffer(f32),
    expert_counts: DeviceBuffer(i32),
    expert_offsets: DeviceBuffer(i32),
    sorted_instances: DeviceBuffer(i32),
    instance_to_sorted: DeviceBuffer(i32),

    // Prompt KV Cache buffers (Phase 7B)
    prompt_k_cache: DeviceBuffer(f32),
    prompt_v_cache: DeviceBuffer(f32),

    // Phase 9: Self-Conditioning device scratch buffers (Device 0)
    sc_soft_emb: DeviceBuffer(f32),
    sc_geglu_buf: DeviceBuffer(f32),
    sc_sig_buf: DeviceBuffer(f32),

    // Phase 19: Canvas initialization and self-conditioning backup buffers
    canvas_init_emb: DeviceBuffer(f32),
    canvas_raw_ans: DeviceBuffer(f32),

    // Final answer norm hidden output
    norm_hidden: DeviceBuffer(f32),
    allocator: std.mem.Allocator,

    pub const max_device_layers: usize = 30;
    pub const max_cached_prompt_tokens: usize = 512;
    pub const max_k_dim_cached: usize = 2048; // Max across SWA (8*256=2048) and Global (2*512=1024) layers (R26-01)
    pub const cache_floats_per_layer: usize = max_cached_prompt_tokens * max_k_dim_cached;
    pub const cache_total_floats: usize = max_device_layers * cache_floats_per_layer;

    pub fn init(allocator: std.mem.Allocator, device_id: i32, max_seq_len: usize) !CudaDeviceWorkspace {
        const stream = try CudaStream.init(device_id);
        errdefer {
            var s = stream;
            s.deinit();
        }

        const emb_size = max_seq_len * ModelConfig.embedding_dim;
        var emb_buf_0 = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer emb_buf_0.deinit();
        var emb_buf_1 = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer emb_buf_1.deinit();
        var norm_x = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer norm_x.deinit();

        var q_buf = try DeviceBuffer(f32).init(device_id, max_seq_len * infer.max_q_dim);
        errdefer q_buf.deinit();
        var k_buf = try DeviceBuffer(f32).init(device_id, max_seq_len * infer.max_k_dim);
        errdefer k_buf.deinit();
        var v_buf = try DeviceBuffer(f32).init(device_id, max_seq_len * infer.max_k_dim);
        errdefer v_buf.deinit();
        var head_out = try DeviceBuffer(f32).init(device_id, max_seq_len * infer.max_q_dim);
        errdefer head_out.deinit();
        var attn_out = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer attn_out.deinit();

        var dense_in = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer dense_in.deinit();
        var gate_buf = try DeviceBuffer(f32).init(device_id, max_seq_len * 2112);
        errdefer gate_buf.deinit();
        var up_buf = try DeviceBuffer(f32).init(device_id, max_seq_len * 2112);
        errdefer up_buf.deinit();
        var geglu_buf = try DeviceBuffer(f32).init(device_id, max_seq_len * 2112);
        errdefer geglu_buf.deinit();
        var dense_out = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer dense_out.deinit();
        var h1 = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer h1.deinit();

        var r_in = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer r_in.deinit();
        var router_logits = try DeviceBuffer(f32).init(device_id, max_seq_len * 128);
        errdefer router_logits.deinit();
        var top_indices = try DeviceBuffer(i32).init(device_id, max_seq_len * 8);
        errdefer top_indices.deinit();
        var top_weights = try DeviceBuffer(f32).init(device_id, max_seq_len * 8);
        errdefer top_weights.deinit();
        var moe_in = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer moe_in.deinit();
        var exp_act_buf = try DeviceBuffer(f32).init(device_id, max_seq_len * 8 * 704);
        errdefer exp_act_buf.deinit();
        var moe_accum = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer moe_accum.deinit();
        var expert_counts = try DeviceBuffer(i32).init(device_id, 128);
        errdefer expert_counts.deinit();
        var expert_offsets = try DeviceBuffer(i32).init(device_id, 128);
        errdefer expert_offsets.deinit();
        var sorted_instances = try DeviceBuffer(i32).init(device_id, max_seq_len * 8);
        errdefer sorted_instances.deinit();
        var instance_to_sorted = try DeviceBuffer(i32).init(device_id, max_seq_len * 8);
        errdefer instance_to_sorted.deinit();
        var prompt_k_cache = try DeviceBuffer(f32).init(device_id, cache_total_floats);
        errdefer prompt_k_cache.deinit();
        var prompt_v_cache = try DeviceBuffer(f32).init(device_id, cache_total_floats);
        errdefer prompt_v_cache.deinit();
        var norm_hidden = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer norm_hidden.deinit();

        var sc_soft_emb = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer sc_soft_emb.deinit();
        var sc_geglu_buf = try DeviceBuffer(f32).init(device_id, ModelConfig.self_cond_intermediate_dim);
        errdefer sc_geglu_buf.deinit();
        var sc_sig_buf = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer sc_sig_buf.deinit();

        var canvas_init_emb = try DeviceBuffer(f32).init(device_id, emb_size);
        errdefer canvas_init_emb.deinit();
        var canvas_raw_ans = try DeviceBuffer(f32).init(device_id, ModelConfig.embedding_dim);
        errdefer canvas_raw_ans.deinit();

        return CudaDeviceWorkspace{
            .device_id = device_id,
            .stream = stream,
            .max_seq_len = max_seq_len,
            .emb_buf_0 = emb_buf_0,
            .emb_buf_1 = emb_buf_1,
            .current_emb = emb_buf_0,
            .next_emb = emb_buf_1,
            .norm_x = norm_x,
            .q_buf = q_buf,
            .k_buf = k_buf,
            .v_buf = v_buf,
            .head_out = head_out,
            .attn_out = attn_out,
            .dense_in = dense_in,
            .gate_buf = gate_buf,
            .up_buf = up_buf,
            .geglu_buf = geglu_buf,
            .dense_out = dense_out,
            .h1 = h1,
            .r_in = r_in,
            .router_logits = router_logits,
            .top_indices = top_indices,
            .top_weights = top_weights,
            .moe_in = moe_in,
            .exp_act_buf = exp_act_buf,
            .moe_accum = moe_accum,
            .expert_counts = expert_counts,
            .expert_offsets = expert_offsets,
            .sorted_instances = sorted_instances,
            .instance_to_sorted = instance_to_sorted,
            .prompt_k_cache = prompt_k_cache,
            .prompt_v_cache = prompt_v_cache,
            .norm_hidden = norm_hidden,
            .sc_soft_emb = sc_soft_emb,
            .sc_geglu_buf = sc_geglu_buf,
            .sc_sig_buf = sc_sig_buf,
            .canvas_init_emb = canvas_init_emb,
            .canvas_raw_ans = canvas_raw_ans,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CudaDeviceWorkspace) void {
        self.canvas_raw_ans.deinit();
        self.canvas_init_emb.deinit();
        self.sc_sig_buf.deinit();
        self.sc_geglu_buf.deinit();
        self.sc_soft_emb.deinit();
        self.norm_hidden.deinit();
        self.prompt_v_cache.deinit();
        self.prompt_k_cache.deinit();
        self.instance_to_sorted.deinit();
        self.sorted_instances.deinit();
        self.expert_offsets.deinit();
        self.expert_counts.deinit();
        self.moe_accum.deinit();
        self.exp_act_buf.deinit();
        self.moe_in.deinit();
        self.top_weights.deinit();
        self.top_indices.deinit();
        self.router_logits.deinit();
        self.r_in.deinit();
        self.h1.deinit();
        self.dense_out.deinit();
        self.geglu_buf.deinit();
        self.up_buf.deinit();
        self.gate_buf.deinit();
        self.dense_in.deinit();
        self.attn_out.deinit();
        self.head_out.deinit();
        self.v_buf.deinit();
        self.k_buf.deinit();
        self.q_buf.deinit();
        self.norm_x.deinit();
        self.emb_buf_1.deinit();
        self.emb_buf_0.deinit();
        self.stream.deinit();
    }

    pub fn swapBuffers(self: *CudaDeviceWorkspace) void {
        const tmp = self.current_emb;
        self.current_emb = self.next_emb;
        self.next_emb = tmp;
    }

    pub fn resetBufferPointers(self: *CudaDeviceWorkspace) void {
        self.current_emb = self.emb_buf_0;
        self.next_emb = self.emb_buf_1;
    }

    /// Returns the exact total bytes of device activation memory allocated across all buffers.
    pub fn activationWorkspaceBytes(max_seq_len: usize) usize {
        const d_emb = ModelConfig.embedding_dim;
        const q_dim = infer.max_q_dim;
        const k_dim = infer.max_k_dim;
        return (max_seq_len * d_emb * @sizeOf(f32) * 11) + // emb_buf_0, emb_buf_1, norm_x, attn_out, dense_in, dense_out, h1, r_in, moe_in, moe_accum, canvas_init_emb (Phase 19)
            (max_seq_len * q_dim * @sizeOf(f32) * 2) + // q_buf, head_out
            (max_seq_len * k_dim * @sizeOf(f32) * 2) + // k_buf, v_buf
            (max_seq_len * 2112 * @sizeOf(f32) * 3) + // gate_buf, up_buf, geglu_buf
            (max_seq_len * 128 * @sizeOf(f32)) + // router_logits
            (max_seq_len * 8 * @sizeOf(i32)) + // top_indices
            (max_seq_len * 8 * @sizeOf(f32)) + // top_weights
            (max_seq_len * 8 * 704 * @sizeOf(f32)) + // exp_act_buf (exp_gate_up_buf eliminated in Phase 10 WP2)
            (128 * @sizeOf(i32) * 2) + // expert_counts, expert_offsets
            (max_seq_len * 8 * @sizeOf(i32) * 2) + // sorted_instances, instance_to_sorted
            (cache_total_floats * @sizeOf(f32) * 2) + // prompt_k_cache, prompt_v_cache (Phase 7B)
            (d_emb * @sizeOf(f32) * 2) + // norm_hidden, canvas_raw_ans (Phase 19)
            (d_emb * @sizeOf(f32) * 2) + // sc_soft_emb, sc_sig_buf (Phase 9)
            (ModelConfig.self_cond_intermediate_dim * @sizeOf(f32)); // sc_geglu_buf (Phase 9)
    }
};

/// Public injection seams for testing CUDA Graph fallback paths (R24-05, R40-01, R41-01, R42-01).
pub var inject_graph_capture_failure: bool = false;
pub var inject_graph_replay_failure: bool = false;
pub var inject_graph1_replay_failure: bool = false;
pub var inject_stream_sync_failure: bool = false;
pub var inject_dev0_sync_failure: bool = false;
pub var inject_dev1_sync_failure: bool = false;
pub var inject_device_sync_failure: bool = false;

/// Teardown safety policy: quarantined pinned staging allocations retained when stream
/// synchronization fails during workspace deinitialization.
pub const QuarantinedPinnedBuffer = struct {
    ptr: [*]f32,
    len: usize,
};
pub const MAX_QUARANTINED_BUFFERS: usize = 16;
pub var quarantined_buffers: [MAX_QUARANTINED_BUFFERS]?QuarantinedPinnedBuffer = @splat(null);
pub var quarantined_buffers_count: usize = 0;

pub const MAX_OVERFLOW_BUFFERS: usize = 64;
pub var overflow_quarantined_buffers: [MAX_OVERFLOW_BUFFERS]?QuarantinedPinnedBuffer = @splat(null);
pub var overflow_quarantined_buffers_count: usize = 0;

/// R45-03: Durable dynamic linked list node for pinned buffers quarantined when both fixed tables
/// (primary 16 + overflow 64) are exhausted and emergency device synchronization fails.
pub const DurableQuarantineNode = struct {
    ptr: [*]f32,
    len: usize,
    next: ?*DurableQuarantineNode = null,
};
pub var dynamic_quarantined_head: ?*DurableQuarantineNode = null;
pub var dynamic_quarantined_count: usize = 0;

pub fn hasQuarantinedBuffers() bool {
    return quarantined_buffers_count > 0 or overflow_quarantined_buffers_count > 0 or dynamic_quarantined_count > 0;
}

pub fn quarantinePinnedBuffer(buf: *cuda_mem.PinnedBuffer(f32)) void {
    if (comptime !cuda.is_enabled) return;
    if (buf.ptr) |p| {
        var placed = false;
        for (&quarantined_buffers) |*slot| {
            if (slot.* == null) {
                slot.* = .{
                    .ptr = p,
                    .len = buf.len,
                };
                quarantined_buffers_count += 1;
                placed = true;
                break;
            }
        }
        if (!placed) {
            // Quarantine capacity reached. Attempt an opportunistic drain of quiescent slots.
            drainQuarantinedPinnedBuffers() catch {};
            for (&quarantined_buffers) |*slot| {
                if (slot.* == null) {
                    slot.* = .{
                        .ptr = p,
                        .len = buf.len,
                    };
                    quarantined_buffers_count += 1;
                    placed = true;
                    break;
                }
            }
        }
        if (placed) {
            _ = buf.move();
            return;
        }

        // R43-01 & R44-04: Explicit teardown when primary quarantine capacity remains exhausted even after drain.
        // Synchronize all devices to establish whole-device quiescence and deinit directly
        // so we never leak or drop ownership of host memory.
        std.log.warn("quarantinePinnedBuffer: primary quarantine capacity ({d}) exhausted; synchronizing devices for immediate safe teardown of buffer {*} (len={d})", .{
            MAX_QUARANTINED_BUFFERS,
            p,
            buf.len,
        });
        var dev_count: c_int = 0;
        if (cuda.c.zeno_cuda_get_device_count(&dev_count) != cuda.c.ZENO_CUDA_SUCCESS or dev_count <= 0) {
            dev_count = 1;
        }
        var d: c_int = 0;
        var sync_ok = !inject_device_sync_failure;
        if (sync_ok) {
            while (d < dev_count) : (d += 1) {
                if (cuda.c.zeno_cuda_device_synchronize(d) != cuda.c.ZENO_CUDA_SUCCESS) {
                    sync_ok = false;
                    break;
                }
            }
        }

        if (sync_ok) {
            buf.deinit();
            return;
        }

        // Emergency device synchronization failed: in-flight DMA may still be active.
        // R44-04: Place into durable overflow quarantine to maintain clear, tracked ownership.
        var placed_overflow = false;
        for (&overflow_quarantined_buffers) |*slot| {
            if (slot.* == null) {
                slot.* = .{
                    .ptr = p,
                    .len = buf.len,
                };
                overflow_quarantined_buffers_count += 1;
                placed_overflow = true;
                break;
            }
        }

        if (placed_overflow) {
            std.log.warn("quarantinePinnedBuffer: emergency device synchronization failed; buffer {*} retained in durable overflow quarantine ({d}/{d})", .{
                p,
                overflow_quarantined_buffers_count,
                MAX_OVERFLOW_BUFFERS,
            });
            _ = buf.move();
        } else {
            // R45-03: Primary (16) and overflow (64) fixed tables are exhausted AND emergency device sync failed.
            // Allocate a dynamic quarantine node to guarantee durable ownership and tracking across workspace teardown.
            if (std.heap.c_allocator.create(DurableQuarantineNode)) |node| {
                node.* = .{
                    .ptr = p,
                    .len = buf.len,
                    .next = dynamic_quarantined_head,
                };
                dynamic_quarantined_head = node;
                dynamic_quarantined_count += 1;
                std.log.warn("quarantinePinnedBuffer: emergency sync failed and fixed tables ({d}+{d}) exhausted; buffer {*} tracked in dynamic overflow quarantine ({d})", .{
                    MAX_QUARANTINED_BUFFERS,
                    MAX_OVERFLOW_BUFFERS,
                    p,
                    dynamic_quarantined_count,
                });
                _ = buf.move();
            } else |_| {
                std.log.warn("quarantinePinnedBuffer: emergency sync failed, fixed tables exhausted, and dynamic node allocation failed; buffer {*} ownership retained by caller", .{p});
            }
        }
    }
}

pub fn drainQuarantinedPinnedBuffers() !void {
    if (comptime !cuda.is_enabled) return;

    if (inject_device_sync_failure) {
        std.log.warn("drainQuarantinedPinnedBuffers: injected device synchronization failure; retaining quarantine", .{});
        return error.DeviceNotQuiescent;
    }

    // R42-01: Confirm whole-device quiescence across all available CUDA devices before releasing any quarantined host memory.
    // If device synchronization fails on any device, in-flight DMA may still be referencing the pinned allocation.
    // In that event, abort drainage and retain the quarantined slots.
    var dev_count: c_int = 0;
    if (cuda.c.zeno_cuda_get_device_count(&dev_count) != cuda.c.ZENO_CUDA_SUCCESS or dev_count <= 0) {
        dev_count = 1;
    }
    var d: c_int = 0;
    while (d < dev_count) : (d += 1) {
        const rc = cuda.c.zeno_cuda_device_synchronize(d);
        if (rc != cuda.c.ZENO_CUDA_SUCCESS) {
            std.log.warn("drainQuarantinedPinnedBuffers: device {d} failed to synchronize (rc={d}); retaining quarantine", .{ d, rc });
            return error.DeviceNotQuiescent;
        }
    }

    // Devices are confirmed quiescent; release quarantined pinned buffers.
    // R42-01 & R42-02: Only clear the slot and decrement count if host free succeeds; retain slot if free fails so pointer is not lost.
    for (&quarantined_buffers) |*slot| {
        if (slot.*) |q| {
            const rc = cuda.c.zeno_cuda_host_free(q.ptr);
            if (rc == cuda.c.ZENO_CUDA_SUCCESS) {
                slot.* = null;
                if (quarantined_buffers_count > 0) quarantined_buffers_count -= 1;
            } else {
                std.log.warn("drainQuarantinedPinnedBuffers: failed to free quarantined host buffer {*} (rc={d}); retaining slot", .{ q.ptr, rc });
            }
        }
    }

    // R44-04: Also release any overflow quarantined pinned buffers once devices are quiescent.
    for (&overflow_quarantined_buffers) |*slot| {
        if (slot.*) |q| {
            const rc = cuda.c.zeno_cuda_host_free(q.ptr);
            if (rc == cuda.c.ZENO_CUDA_SUCCESS) {
                slot.* = null;
                if (overflow_quarantined_buffers_count > 0) overflow_quarantined_buffers_count -= 1;
            } else {
                std.log.warn("drainQuarantinedPinnedBuffers: failed to free overflow quarantined host buffer {*} (rc={d}); retaining slot", .{ q.ptr, rc });
            }
        }
    }

    // R45-03: Also release any dynamically quarantined pinned buffers once devices are quiescent.
    var curr_dynamic = dynamic_quarantined_head;
    var prev_dynamic: ?*DurableQuarantineNode = null;
    while (curr_dynamic) |node| {
        const next_node = node.next;
        const rc = cuda.c.zeno_cuda_host_free(node.ptr);
        if (rc == cuda.c.ZENO_CUDA_SUCCESS) {
            if (prev_dynamic) |p_node| {
                p_node.next = next_node;
            } else {
                dynamic_quarantined_head = next_node;
            }
            std.heap.c_allocator.destroy(node);
            if (dynamic_quarantined_count > 0) dynamic_quarantined_count -= 1;
        } else {
            std.log.warn("drainQuarantinedPinnedBuffers: failed to free dynamic quarantined host buffer {*} (rc={d}); retaining node", .{ node.ptr, rc });
            prev_dynamic = node;
        }
        curr_dynamic = next_node;
    }
}

/// High-level multi-device execution workspace coordinating device workspaces and inter-device transfers.
pub const CudaWorkspace = struct {
    allocator: std.mem.Allocator,
    topology: ExecutionTopology,
    dev0: CudaDeviceWorkspace,
    dev1: ?CudaDeviceWorkspace,
    boundary_event: ?cuda_mem.CudaEvent,
    host_staging_pinned: cuda_mem.PinnedBuffer(f32),
    host_staging_buf: []f32,
    host_norm_hidden: []f32,
    peer_access_enabled: bool,
    graph0: ?cuda_mem.CudaGraph,
    graph1: ?cuda_mem.CudaGraph,
    graph0_step2: ?cuda_mem.CudaGraph,
    cached_graph_prompt_len: usize,
    cached_graph_canvas_len: usize,
    cached_graph_answer_pos: usize,
    cached_graph_step2_prompt_len: usize,
    cached_graph_step2_canvas_len: usize,
    cached_graph_step2_answer_pos: usize,

    // Prompt KV Cache tracking (Phase 7B)
    cached_prompt_hash: u64,
    cached_prompt_len: usize,
    prompt_cache_valid: bool,

    // Stage-level profiling timings (Review Round 34 Option 1)
    last_stage_timings: decision.StageTimings,
    last_canvas_ms: f32,

    pub fn synchronize(self: *CudaWorkspace) !void {
        if (inject_stream_sync_failure) return error.StreamSyncFailed;
        var err0: ?anyerror = null;
        if (inject_dev0_sync_failure) {
            err0 = error.StreamSyncFailed;
        } else {
            self.dev0.stream.synchronize() catch |err| {
                err0 = err;
            };
        }
        var err1: ?anyerror = null;
        if (self.dev1) |*d1| {
            if (inject_dev1_sync_failure) {
                err1 = error.StreamSyncFailed;
            } else {
                d1.stream.synchronize() catch |err| {
                    err1 = err;
                };
            }
        }
        if (err0) |e| return e;
        if (err1) |e| return e;

        // R43-01: Streams successfully synchronized; opportunistically drain any quarantined pinned buffers
        if (hasQuarantinedBuffers()) {
            drainQuarantinedPinnedBuffers() catch {};
        }
    }

    pub fn init(allocator: std.mem.Allocator, topology: ExecutionTopology, devices: [2]i32, max_seq_len: usize) !*CudaWorkspace {
        // Clean slate: opportunistically drain any previously quarantined host pinned allocations.
        if (hasQuarantinedBuffers()) {
            drainQuarantinedPinnedBuffers() catch {};
        }

        const ws_ptr = try allocator.create(CudaWorkspace);
        errdefer allocator.destroy(ws_ptr);

        var dev0 = try CudaDeviceWorkspace.init(allocator, devices[0], max_seq_len);
        errdefer dev0.deinit();

        var dev1: ?CudaDeviceWorkspace = null;
        var peer_access_enabled = false;
        if (topology == .dual_gpu_split) {
            const rc0 = cuda.c.zeno_cuda_enable_peer_access(devices[0], devices[1]);
            const rc1 = cuda.c.zeno_cuda_enable_peer_access(devices[1], devices[0]);
            if (rc0 == 0 and rc1 == 0) {
                peer_access_enabled = true;
            }
            dev1 = try CudaDeviceWorkspace.init(allocator, devices[1], max_seq_len);
        }
        errdefer if (dev1) |*d1| d1.deinit();

        var boundary_event: ?cuda_mem.CudaEvent = null;
        if (topology == .dual_gpu_split) {
            boundary_event = try cuda_mem.CudaEvent.init(devices[0]);
        }
        errdefer if (boundary_event) |*ev| ev.deinit();

        var host_staging_pinned = try cuda_mem.PinnedBuffer(f32).init(max_seq_len * ModelConfig.embedding_dim);
        errdefer host_staging_pinned.deinit();

        const host_norm = try allocator.alloc(f32, ModelConfig.embedding_dim);
        errdefer allocator.free(host_norm);

        ws_ptr.allocator = allocator;
        ws_ptr.topology = topology;
        ws_ptr.dev0 = dev0;
        ws_ptr.dev1 = dev1;
        ws_ptr.boundary_event = boundary_event;
        ws_ptr.host_staging_pinned = host_staging_pinned;
        ws_ptr.host_staging_buf = host_staging_pinned.slice();
        ws_ptr.host_norm_hidden = host_norm;
        ws_ptr.peer_access_enabled = peer_access_enabled;
        ws_ptr.graph0 = null;
        ws_ptr.graph1 = null;
        ws_ptr.graph0_step2 = null;
        ws_ptr.cached_graph_prompt_len = 0;
        ws_ptr.cached_graph_canvas_len = 0;
        ws_ptr.cached_graph_answer_pos = 0;
        ws_ptr.cached_graph_step2_prompt_len = 0;
        ws_ptr.cached_graph_step2_canvas_len = 0;
        ws_ptr.cached_graph_step2_answer_pos = 0;
        ws_ptr.cached_prompt_hash = 0;
        ws_ptr.cached_prompt_len = 0;
        ws_ptr.prompt_cache_valid = false;
        ws_ptr.last_stage_timings = .{};
        ws_ptr.last_canvas_ms = 0.0;

        return ws_ptr;
    }

    pub fn invalidatePromptCache(self: *CudaWorkspace) void {
        self.cached_prompt_hash = 0;
        self.cached_prompt_len = 0;
        self.prompt_cache_valid = false;
        self.last_stage_timings = .{};
        self.last_canvas_ms = 0.0;
        if (self.graph0_step2) |*g| {
            g.deinit();
            self.graph0_step2 = null;
        }
        if (self.graph1) |*g1| {
            g1.deinit();
            self.graph1 = null;
        }
        if (self.graph0) |*g0| {
            g0.deinit();
            self.graph0 = null;
        }
        self.cached_graph_prompt_len = 0;
        self.cached_graph_canvas_len = 0;
        self.cached_graph_answer_pos = 0;
        self.cached_graph_step2_prompt_len = 0;
        self.cached_graph_step2_canvas_len = 0;
        self.cached_graph_step2_answer_pos = 0;
    }

    pub fn deinit(self: *CudaWorkspace) void {
        // R40-01 / R41-01: Independently drain each referencing device stream before teardown
        var sync_failed = false;
        if (self.synchronize()) |_| {} else |err| {
            sync_failed = true;
            std.log.warn("CudaWorkspace.deinit: stream synchronization failed with {s}", .{@errorName(err)});
        }
        if (self.graph0_step2) |*g| g.deinit();
        if (self.graph1) |*g1| g1.deinit();
        if (self.graph0) |*g0| g0.deinit();
        self.allocator.free(self.host_norm_hidden);

        // R41-01: Explicit teardown policy for failed stream synchronization:
        // Freeing pinned host memory while GPU DMA may still be in flight risks host memory corruption.
        // If stream synchronization could not be verified on all referencing streams, quarantine
        // the pinned allocation (retain the allocation without calling cudaFreeHost).
        if (!sync_failed) {
            self.host_staging_pinned.deinit();
            if (hasQuarantinedBuffers()) {
                drainQuarantinedPinnedBuffers() catch {};
            }
        } else {
            std.log.warn("CudaWorkspace.deinit: quarantining host_staging_pinned ({*} len={d}) due to failed stream synchronization", .{
                self.host_staging_pinned.ptr,
                self.host_staging_pinned.len,
            });
            quarantinePinnedBuffer(&self.host_staging_pinned);
        }

        if (self.boundary_event) |*ev| ev.deinit();
        if (self.dev1) |*d1| d1.deinit();
        self.dev0.deinit();
        self.allocator.destroy(self);
    }
};

/// Executes a single transformer layer forward pass on the GPU.
pub fn runLayerCuda(
    ws: *CudaDeviceWorkspace,
    layer: *const CudaLayerWeights,
    prompt_len: usize,
    num_tokens: usize,
    rope_freqs: ?DeviceSlice(f32),
    save_prompt_kv_layer_idx: ?usize,
) !void {
    const stream = &ws.stream;
    const is_global = layer.is_global;
    const q_dim = layer.q_dim;
    const k_dim = layer.k_dim;
    const head_dim = layer.head_dim;
    const num_q_heads = layer.num_q_heads;
    const num_kv_heads = layer.num_kv_heads;
    const theta: f32 = if (is_global) 1000000.0 else 10000.0;

    const emb_elements = num_tokens * ModelConfig.embedding_dim;
    const curr_slice = try ws.current_emb.slice().subslice(0, emb_elements);
    const nx_slice = try ws.norm_x.slice().subslice(0, emb_elements);

    // 1. Input normalization
    try cuda_mem.rmsNorm(curr_slice, layer.attn_norm.slice(), nx_slice, num_tokens, ModelConfig.embedding_dim, 1e-6, stream);

    // 2. Q, K, V projections
    const q_slice = try ws.q_buf.slice().subslice(0, num_tokens * q_dim);
    const k_slice = try ws.k_buf.slice().subslice(0, num_tokens * k_dim);
    const v_slice = try ws.v_buf.slice().subslice(0, num_tokens * k_dim);

    if (layer.q_weight_type == GGML_TYPE_Q4_K and layer.k_weight_type == GGML_TYPE_Q4_K and (is_global or (layer.v_weight != null and layer.v_weight_type == GGML_TYPE_Q4_K))) {
        const wv_slice = if (!is_global and layer.v_weight != null) layer.v_weight.?.slice() else null;
        try cuda_mem.gemvQ4_K_FusedQKV(
            layer.q_weight.slice(),
            layer.k_weight.slice(),
            wv_slice,
            nx_slice,
            q_slice,
            k_slice,
            v_slice,
            q_dim,
            k_dim,
            k_dim,
            ModelConfig.embedding_dim,
            num_tokens,
            stream,
        );
    } else if (layer.q_weight_type == GGML_TYPE_Q4_K and layer.k_weight_type == GGML_TYPE_Q4_K) {
        try cuda_mem.gemvQ4_K_FusedQK(
            layer.q_weight.slice(),
            layer.k_weight.slice(),
            nx_slice,
            q_slice,
            k_slice,
            null,
            q_dim,
            k_dim,
            ModelConfig.embedding_dim,
            num_tokens,
            stream,
        );
        if (layer.v_weight) |vw| {
            try gemvTensorCuda(layer.v_weight_type, vw.slice(), nx_slice, v_slice, k_dim, ModelConfig.embedding_dim, num_tokens, stream);
        }
    } else {
        try gemvTensorCuda(layer.q_weight_type, layer.q_weight.slice(), nx_slice, q_slice, q_dim, ModelConfig.embedding_dim, num_tokens, stream);
        try gemvTensorCuda(layer.k_weight_type, layer.k_weight.slice(), nx_slice, k_slice, k_dim, ModelConfig.embedding_dim, num_tokens, stream);
        if (is_global) {
            try v_slice.copyFromDevice(k_slice, stream);
        } else {
            try gemvTensorCuda(layer.v_weight_type, layer.v_weight.?.slice(), nx_slice, v_slice, k_dim, ModelConfig.embedding_dim, num_tokens, stream);
        }
    }

    // 3. Per-head RMSNorms & RoPE (Phase 15 WP3)
    const factors = if (is_global) rope_freqs else null;
    try cuda_mem.fusedHeadNormRope(q_slice, layer.q_norm.slice(), num_tokens, num_q_heads, head_dim, 0, theta, factors, 1e-6, stream);
    try cuda_mem.fusedHeadNormRope(k_slice, layer.k_norm.slice(), num_tokens, num_kv_heads, head_dim, 0, theta, factors, 1e-6, stream);
    try cuda_mem.rmsNorm(v_slice, null, v_slice, num_tokens * num_kv_heads, head_dim, 1e-6, stream);

    // Save prompt KV prefix cache if requested (Phase 7B)
    if (save_prompt_kv_layer_idx) |dev_layer_idx| {
        const offset = dev_layer_idx * CudaDeviceWorkspace.cache_floats_per_layer;
        const dst_k = try ws.prompt_k_cache.slice().subslice(offset, num_tokens * k_dim);
        const dst_v = try ws.prompt_v_cache.slice().subslice(offset, num_tokens * k_dim);
        try dst_k.copyFromDevice(k_slice, stream);
        try dst_v.copyFromDevice(v_slice, stream);
    }

    // 5. GQA Attention
    const head_out_slice = try ws.head_out.slice().subslice(0, num_tokens * q_dim);
    try cuda_mem.attention(
        q_slice,
        k_slice,
        v_slice,
        head_out_slice,
        num_tokens,
        prompt_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        q_dim,
        k_dim,
        k_dim,
        q_dim,
        stream,
    );

    // 6. Attention output projection & post norm & residual connection
    const a_out_slice = try ws.attn_out.slice().subslice(0, emb_elements);
    try gemvTensorCuda(layer.o_weight_type, layer.o_weight.slice(), head_out_slice, a_out_slice, ModelConfig.embedding_dim, q_dim, num_tokens, stream);
    try cuda_mem.fusedPostAttnNormResidual(a_out_slice, layer.post_attention_norm.slice(), curr_slice, a_out_slice, num_tokens, ModelConfig.embedding_dim, 1e-6, stream);

    // x_mid is a_out_slice
    const x_mid = a_out_slice;

    // 7. Dense FFN & MoE Pre-Norms (Phase 15 WP1)
    const dense_in_slice = try ws.dense_in.slice().subslice(0, emb_elements);
    const moe_in_slice = try ws.moe_in.slice().subslice(0, emb_elements);
    const r_in_slice = try ws.r_in.slice().subslice(0, emb_elements);
    const root_size: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ModelConfig.embedding_dim)));
    try cuda_mem.fusedPreFfnNorm(
        x_mid,
        layer.ffn_norm.slice(),
        layer.pre_ffw_norm_2.slice(),
        layer.router_scale.slice(),
        dense_in_slice,
        moe_in_slice,
        r_in_slice,
        root_size,
        num_tokens,
        ModelConfig.embedding_dim,
        1e-6,
        stream,
    );

    const geglu_slice = try ws.geglu_buf.slice().subslice(0, num_tokens * 2112);

    if (layer.ffn_gate_type == GGML_TYPE_Q4_K and layer.ffn_up_type == GGML_TYPE_Q4_K) {
        try cuda_mem.denseGateUpGegluQ4_K(
            layer.ffn_gate.slice(),
            layer.ffn_up.slice(),
            dense_in_slice,
            geglu_slice,
            2112,
            ModelConfig.embedding_dim,
            num_tokens,
            stream,
        );
    } else {
        const gate_slice = try ws.gate_buf.slice().subslice(0, num_tokens * 2112);
        const up_slice = try ws.up_buf.slice().subslice(0, num_tokens * 2112);
        try gemvTensorCuda(layer.ffn_gate_type, layer.ffn_gate.slice(), dense_in_slice, gate_slice, 2112, ModelConfig.embedding_dim, num_tokens, stream);
        try gemvTensorCuda(layer.ffn_up_type, layer.ffn_up.slice(), dense_in_slice, up_slice, 2112, ModelConfig.embedding_dim, num_tokens, stream);
        try cuda_mem.geglu(gate_slice, up_slice, geglu_slice, stream);
    }

    const dense_out_slice = try ws.dense_out.slice().subslice(0, emb_elements);
    const h1_slice = try ws.h1.slice().subslice(0, emb_elements);
    try gemvTensorCuda(layer.ffn_down_type, layer.ffn_down.slice(), geglu_slice, dense_out_slice, ModelConfig.embedding_dim, 2112, num_tokens, stream);
    try cuda_mem.rmsNorm(dense_out_slice, layer.post_ffw_norm_1.slice(), h1_slice, num_tokens, ModelConfig.embedding_dim, 1e-6, stream);

    // 8. MoE router
    const router_logits_slice = try ws.router_logits.slice().subslice(0, num_tokens * 128);
    try gemvTensorCuda(layer.router_gate_type, layer.router_gate.slice(), r_in_slice, router_logits_slice, 128, ModelConfig.embedding_dim, num_tokens, stream);

    const top_idx_slice = try ws.top_indices.slice().subslice(0, num_tokens * 8);
    const top_w_slice = try ws.top_weights.slice().subslice(0, num_tokens * 8);
    try cuda_mem.moeRouterTop8(router_logits_slice, layer.expert_down_scale.slice(), top_idx_slice, top_w_slice, num_tokens, stream);

    // 9. Evaluate active MoE experts on device using token-sorted projections (Phase 7A & Phase 13 WP1)
    const num_instances = num_tokens * 8;
    const exp_counts_slice = try ws.expert_counts.slice().subslice(0, 128);
    const exp_offsets_slice = try ws.expert_offsets.slice().subslice(0, 128);
    const sorted_inst_slice = try ws.sorted_instances.slice().subslice(0, num_instances);
    const inst_to_sorted_slice = try ws.instance_to_sorted.slice().subslice(0, num_instances);

    try cuda_mem.moeBuildExpertBins(
        top_idx_slice,
        exp_counts_slice,
        exp_offsets_slice,
        sorted_inst_slice,
        inst_to_sorted_slice,
        null,
        null,
        num_tokens,
        stream,
    );

    const exp_act_slice = try ws.exp_act_buf.slice().subslice(0, num_instances * 704);
    try cuda_mem.moeGemvGateUpGegluSorted(
        layer.expert_gate_up.slice(),
        moe_in_slice,
        exp_counts_slice,
        exp_offsets_slice,
        sorted_inst_slice,
        exp_act_slice,
        num_tokens,
        ModelConfig.embedding_dim,
        704,
        stream,
    );

    // 9. MoE Fused Down GEMV & Scaled Accumulation (Phase 13 WP1)
    const moe_accum_slice = try ws.moe_accum.slice().subslice(0, emb_elements);
    try cuda_mem.moeGemvDownAccumSorted(
        layer.expert_down.slice(),
        layer.expert_down_type,
        exp_act_slice,
        exp_counts_slice,
        exp_offsets_slice,
        sorted_inst_slice,
        top_w_slice,
        moe_accum_slice,
        num_tokens,
        704,
        ModelConfig.embedding_dim,
        stream,
    );

    // 10. Fused Post-FFN Combine (Phase 15 WP2)
    const next_slice = try ws.next_emb.slice().subslice(0, emb_elements);
    try cuda_mem.fusedPostFfnCombine(
        moe_accum_slice,
        layer.post_ffw_norm_2.slice(),
        h1_slice,
        layer.post_ffw_norm.slice(),
        x_mid,
        next_slice,
        prompt_len,
        num_tokens,
        ModelConfig.embedding_dim,
        layer.enc_layer_output_scale,
        layer.layer_output_scale,
        1e-6,
        stream,
    );

    // 13. Double-buffer swap
    ws.swapBuffers();
}

/// Executes a single transformer layer forward pass on the GPU for canvas tokens ONLY,
/// querying both the cached prompt KV and canvas KV (Phase 7B).
pub fn runLayerCanvasCuda(
    ws: *CudaDeviceWorkspace,
    layer: *const CudaLayerWeights,
    dev_layer_idx: usize,
    prompt_len: usize,
    canvas_len: usize,
    rope_freqs: ?DeviceSlice(f32),
) !void {
    const stream = &ws.stream;
    const is_global = layer.is_global;
    const q_dim = layer.q_dim;
    const k_dim = layer.k_dim;
    const head_dim = layer.head_dim;
    const num_q_heads = layer.num_q_heads;
    const num_kv_heads = layer.num_kv_heads;
    const theta: f32 = if (is_global) 1000000.0 else 10000.0;

    const emb_elements = canvas_len * ModelConfig.embedding_dim;
    const curr_slice = try ws.current_emb.slice().subslice(0, emb_elements);
    const nx_slice = try ws.norm_x.slice().subslice(0, emb_elements);

    // 1. Input normalization
    try cuda_mem.rmsNorm(curr_slice, layer.attn_norm.slice(), nx_slice, canvas_len, ModelConfig.embedding_dim, 1e-6, stream);

    // 2. Q, K, V projections for canvas tokens
    const q_slice = try ws.q_buf.slice().subslice(0, canvas_len * q_dim);
    const k_slice = try ws.k_buf.slice().subslice(0, canvas_len * k_dim);
    const v_slice = try ws.v_buf.slice().subslice(0, canvas_len * k_dim);

    if (layer.q_weight_type == GGML_TYPE_Q4_K and layer.k_weight_type == GGML_TYPE_Q4_K and (is_global or (layer.v_weight != null and layer.v_weight_type == GGML_TYPE_Q4_K))) {
        const wv_slice = if (!is_global and layer.v_weight != null) layer.v_weight.?.slice() else null;
        try cuda_mem.gemvQ4_K_FusedQKV(
            layer.q_weight.slice(),
            layer.k_weight.slice(),
            wv_slice,
            nx_slice,
            q_slice,
            k_slice,
            v_slice,
            q_dim,
            k_dim,
            k_dim,
            ModelConfig.embedding_dim,
            canvas_len,
            stream,
        );
    } else if (layer.q_weight_type == GGML_TYPE_Q4_K and layer.k_weight_type == GGML_TYPE_Q4_K) {
        try cuda_mem.gemvQ4_K_FusedQK(
            layer.q_weight.slice(),
            layer.k_weight.slice(),
            nx_slice,
            q_slice,
            k_slice,
            null,
            q_dim,
            k_dim,
            ModelConfig.embedding_dim,
            canvas_len,
            stream,
        );
        if (layer.v_weight) |vw| {
            try gemvTensorCuda(layer.v_weight_type, vw.slice(), nx_slice, v_slice, k_dim, ModelConfig.embedding_dim, canvas_len, stream);
        }
    } else {
        try gemvTensorCuda(layer.q_weight_type, layer.q_weight.slice(), nx_slice, q_slice, q_dim, ModelConfig.embedding_dim, canvas_len, stream);
        try gemvTensorCuda(layer.k_weight_type, layer.k_weight.slice(), nx_slice, k_slice, k_dim, ModelConfig.embedding_dim, canvas_len, stream);
        if (is_global) {
            try v_slice.copyFromDevice(k_slice, stream);
        } else {
            try gemvTensorCuda(layer.v_weight_type, layer.v_weight.?.slice(), nx_slice, v_slice, k_dim, ModelConfig.embedding_dim, canvas_len, stream);
        }
    }

    // 3. Per-head RMSNorms & RoPE (Phase 15 WP3)
    const factors = if (is_global) rope_freqs else null;
    try cuda_mem.fusedHeadNormRope(q_slice, layer.q_norm.slice(), canvas_len, num_q_heads, head_dim, prompt_len, theta, factors, 1e-6, stream);
    try cuda_mem.fusedHeadNormRope(k_slice, layer.k_norm.slice(), canvas_len, num_kv_heads, head_dim, prompt_len, theta, factors, 1e-6, stream);
    try cuda_mem.rmsNorm(v_slice, null, v_slice, canvas_len * num_kv_heads, head_dim, 1e-6, stream);

    // 5. GQA Attention with cached prompt KV
    const cache_offset = dev_layer_idx * CudaDeviceWorkspace.cache_floats_per_layer;
    const prompt_k_slice = try ws.prompt_k_cache.slice().subslice(cache_offset, prompt_len * k_dim);
    const prompt_v_slice = try ws.prompt_v_cache.slice().subslice(cache_offset, prompt_len * k_dim);
    const head_out_slice = try ws.head_out.slice().subslice(0, canvas_len * q_dim);

    try cuda_mem.attentionCached(
        q_slice,
        prompt_k_slice,
        prompt_v_slice,
        k_slice,
        v_slice,
        head_out_slice,
        prompt_len,
        canvas_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        q_dim,
        k_dim,
        k_dim,
        q_dim,
        stream,
    );

    // 6. Attention output projection & post norm & residual connection
    const a_out_slice = try ws.attn_out.slice().subslice(0, emb_elements);
    try gemvTensorCuda(layer.o_weight_type, layer.o_weight.slice(), head_out_slice, a_out_slice, ModelConfig.embedding_dim, q_dim, canvas_len, stream);
    try cuda_mem.fusedPostAttnNormResidual(a_out_slice, layer.post_attention_norm.slice(), curr_slice, a_out_slice, canvas_len, ModelConfig.embedding_dim, 1e-6, stream);

    const x_mid = a_out_slice;

    // 7. Dense FFN & MoE Pre-Norms (Phase 15 WP1)
    const dense_in_slice = try ws.dense_in.slice().subslice(0, emb_elements);
    const moe_in_slice = try ws.moe_in.slice().subslice(0, emb_elements);
    const r_in_slice = try ws.r_in.slice().subslice(0, emb_elements);
    const root_size: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ModelConfig.embedding_dim)));
    try cuda_mem.fusedPreFfnNorm(
        x_mid,
        layer.ffn_norm.slice(),
        layer.pre_ffw_norm_2.slice(),
        layer.router_scale.slice(),
        dense_in_slice,
        moe_in_slice,
        r_in_slice,
        root_size,
        canvas_len,
        ModelConfig.embedding_dim,
        1e-6,
        stream,
    );

    const geglu_slice = try ws.geglu_buf.slice().subslice(0, canvas_len * 2112);

    if (layer.ffn_gate_type == GGML_TYPE_Q4_K and layer.ffn_up_type == GGML_TYPE_Q4_K) {
        try cuda_mem.denseGateUpGegluQ4_K(
            layer.ffn_gate.slice(),
            layer.ffn_up.slice(),
            dense_in_slice,
            geglu_slice,
            2112,
            ModelConfig.embedding_dim,
            canvas_len,
            stream,
        );
    } else {
        const gate_slice = try ws.gate_buf.slice().subslice(0, canvas_len * 2112);
        const up_slice = try ws.up_buf.slice().subslice(0, canvas_len * 2112);
        try gemvTensorCuda(layer.ffn_gate_type, layer.ffn_gate.slice(), dense_in_slice, gate_slice, 2112, ModelConfig.embedding_dim, canvas_len, stream);
        try gemvTensorCuda(layer.ffn_up_type, layer.ffn_up.slice(), dense_in_slice, up_slice, 2112, ModelConfig.embedding_dim, canvas_len, stream);
        try cuda_mem.geglu(gate_slice, up_slice, geglu_slice, stream);
    }

    const dense_out_slice = try ws.dense_out.slice().subslice(0, emb_elements);
    const h1_slice = try ws.h1.slice().subslice(0, emb_elements);
    try gemvTensorCuda(layer.ffn_down_type, layer.ffn_down.slice(), geglu_slice, dense_out_slice, ModelConfig.embedding_dim, 2112, canvas_len, stream);
    try cuda_mem.rmsNorm(dense_out_slice, layer.post_ffw_norm_1.slice(), h1_slice, canvas_len, ModelConfig.embedding_dim, 1e-6, stream);

    // 8. MoE router
    const router_logits_slice = try ws.router_logits.slice().subslice(0, canvas_len * 128);
    try gemvTensorCuda(layer.router_gate_type, layer.router_gate.slice(), r_in_slice, router_logits_slice, 128, ModelConfig.embedding_dim, canvas_len, stream);

    const top_idx_slice = try ws.top_indices.slice().subslice(0, canvas_len * 8);
    const top_w_slice = try ws.top_weights.slice().subslice(0, canvas_len * 8);
    try cuda_mem.moeRouterTop8(router_logits_slice, layer.expert_down_scale.slice(), top_idx_slice, top_w_slice, canvas_len, stream);

    // 9. Evaluate active MoE experts on device using token-sorted projections (Phase 7A & Phase 13 WP1)
    const num_instances = canvas_len * 8;
    const exp_counts_slice = try ws.expert_counts.slice().subslice(0, 128);
    const exp_offsets_slice = try ws.expert_offsets.slice().subslice(0, 128);
    const sorted_inst_slice = try ws.sorted_instances.slice().subslice(0, num_instances);
    const inst_to_sorted_slice = try ws.instance_to_sorted.slice().subslice(0, num_instances);

    try cuda_mem.moeBuildExpertBins(
        top_idx_slice,
        exp_counts_slice,
        exp_offsets_slice,
        sorted_inst_slice,
        inst_to_sorted_slice,
        null,
        null,
        canvas_len,
        stream,
    );

    const exp_act_slice = try ws.exp_act_buf.slice().subslice(0, num_instances * 704);
    try cuda_mem.moeGemvGateUpGegluSorted(
        layer.expert_gate_up.slice(),
        moe_in_slice,
        exp_counts_slice,
        exp_offsets_slice,
        sorted_inst_slice,
        exp_act_slice,
        canvas_len,
        ModelConfig.embedding_dim,
        704,
        stream,
    );

    // 9. MoE Fused Down GEMV & Scaled Accumulation (Phase 13 WP1)
    const moe_accum_slice = try ws.moe_accum.slice().subslice(0, emb_elements);
    try cuda_mem.moeGemvDownAccumSorted(
        layer.expert_down.slice(),
        layer.expert_down_type,
        exp_act_slice,
        exp_counts_slice,
        exp_offsets_slice,
        sorted_inst_slice,
        top_w_slice,
        moe_accum_slice,
        canvas_len,
        704,
        ModelConfig.embedding_dim,
        stream,
    );

    // 10. Fused Post-FFN Combine (Phase 15 WP2)
    const next_slice = try ws.next_emb.slice().subslice(0, emb_elements);
    try cuda_mem.fusedPostFfnCombine(
        moe_accum_slice,
        layer.post_ffw_norm_2.slice(),
        h1_slice,
        layer.post_ffw_norm.slice(),
        x_mid,
        next_slice,
        0,
        canvas_len,
        ModelConfig.embedding_dim,
        layer.enc_layer_output_scale,
        layer.layer_output_scale,
        1e-6,
        stream,
    );

    // 13. Double-buffer swap
    ws.swapBuffers();
}

/// Transfers intermediate hidden state across GPU devices at partition boundary with async P2P & event coordination (R22-02, R39-01).
/// NOTE (R39-01 Lifetime Contract): When staging through pinned host memory, this function enqueues
/// asynchronous D2H on src_stream, records a hardware synchronization event, orders dst_stream to
/// wait on that event, and enqueues asynchronous H2D on dst_stream. The host buffer remains in-use
/// by dst_stream until dst_stream is synchronized. Callers must drain active streams (via ws.synchronize())
/// before modifying staging_buf or destroying the workspace.
pub fn transferHiddenState(
    src_slice: DeviceSlice(f32),
    dst_slice: DeviceSlice(f32),
    src_stream: *const CudaStream,
    dst_stream: *const CudaStream,
    boundary_event: ?*const cuda_mem.CudaEvent,
    staging_buf: []f32,
    peer_access_enabled: bool,
) !void {
    if (src_slice.len != dst_slice.len) return error.InvalidParam;
    if (src_slice.len == 0) return;

    if (src_slice.device_id == dst_slice.device_id) {
        try dst_slice.copyFromDevice(src_slice, src_stream);
        return;
    }

    // Direct P2P DMA copy when peer access is enabled on the devices
    if (peer_access_enabled) {
        const rc = cuda.c.zeno_cuda_memcpy_peer_async(
            dst_slice.ptr,
            dst_slice.device_id,
            src_slice.ptr,
            src_slice.device_id,
            src_slice.len * @sizeOf(f32),
            src_stream.handle,
        );
        if (rc == 0) {
            // Coordinate stream execution on GPU hardware via boundary event without blocking host CPU
            if (boundary_event) |ev| {
                try ev.record(src_stream);
                try dst_stream.waitEvent(ev);
            } else {
                try src_stream.synchronize();
            }
            return;
        }
    }

    // Fallback: Hardware topology does not support direct P2P DMA (e.g. GeForce RTX 3060 without NVLink).
    // Staged through pinned host memory with asynchronous hardware DMA transfer over PCIe Gen4 (Phase 14 WP2).
    if (staging_buf.len < src_slice.len) return error.InvalidParam;
    const stage = staging_buf[0..src_slice.len];

    if (boundary_event) |ev| {
        try src_slice.copyToHostAsync(stage, src_stream);
        try ev.record(src_stream);
        try dst_stream.waitEvent(ev);
        try dst_slice.copyFromHostAsync(stage, dst_stream);
    } else {
        try src_stream.synchronize();
        const rc_d2h = cuda.c.zeno_cuda_memcpy_d2h(stage.ptr, src_slice.ptr, stage.len * @sizeOf(f32), src_slice.device_id);
        if (rc_d2h != 0) return cuda.mapCudaErrorCode(rc_d2h);

        const rc_h2d = cuda.c.zeno_cuda_memcpy_h2d(dst_slice.ptr, stage.ptr, stage.len * @sizeOf(f32), dst_slice.device_id);
        if (rc_h2d != 0) return cuda.mapCudaErrorCode(rc_h2d);
        try dst_stream.synchronize();
    }
}

/// Evaluates prompt tokens through all layers on device and populates prompt KV cache (Phase 7B).
pub fn prefillPromptCuda(
    ws: *CudaWorkspace,
    model: *const CudaModel,
    cpu_model: DiffusionGemmaModel,
    prompt_tokens: []const u32,
) !void {
    const P = prompt_tokens.len;
    if (P == 0) return;
    if (P > ws.dev0.max_seq_len or P > CudaDeviceWorkspace.max_cached_prompt_tokens) return error.InvalidParam;

    // R39-01: Drain any pending stream operations before overwriting pinned host staging memory
    try ws.synchronize();

    const host_emb = ws.host_staging_buf[0 .. P * ModelConfig.embedding_dim];
    const emb_scale: f32 = @sqrt(@as(f32, @floatFromInt(ModelConfig.embedding_dim)));

    for (prompt_tokens, 0..) |tok, i| {
        const blocks = cpu_model.getEmbeddingBlocks(tok);
        const row = host_emb[i * ModelConfig.embedding_dim .. (i + 1) * ModelConfig.embedding_dim];
        quant.dequantizeRowQ6_K(blocks, row);
        for (row) |*val| val.* *= emb_scale;
    }

    ws.dev0.resetBufferPointers();
    if (ws.dev1) |*d1| d1.resetBufferPointers();

    try ws.synchronize();
    const t_prefill_start = decision.getMonotonicNanos();

    const d0_emb_slice = try ws.dev0.current_emb.slice().subslice(0, P * ModelConfig.embedding_dim);
    try d0_emb_slice.copyFromHost(host_emb);

    if (model.topology == .single_gpu) {
        for (0..ModelConfig.num_layers) |l| {
            const rf = if (model.layers[l].is_global) model.rope_freqs_dev0 else null;
            const rf_slice = if (rf) |r| r.slice() else null;
            try runLayerCuda(&ws.dev0, &model.layers[l], P, P, rf_slice, l);
        }
        try ws.dev0.stream.synchronize();
        const t_end = decision.getMonotonicNanos();
        const dur = @as(f32, @floatFromInt(t_end - t_prefill_start)) / 1_000_000.0;
        ws.last_stage_timings.prompt_p0_ms = dur;
        ws.last_stage_timings.prompt_transfer_ms = 0.0;
        ws.last_stage_timings.prompt_p1_ms = 0.0;
        ws.last_stage_timings.prompt_prefill_ms = dur;
        ws.last_stage_timings.prompt_cache_hit = false;
    } else {
        var dev1 = &ws.dev1.?;
        for (0..15) |l| {
            const rf = if (model.layers[l].is_global) model.rope_freqs_dev0 else null;
            const rf_slice = if (rf) |r| r.slice() else null;
            try runLayerCuda(&ws.dev0, &model.layers[l], P, P, rf_slice, l);
        }
        try ws.dev0.stream.synchronize();
        const t_p0 = decision.getMonotonicNanos();

        const d0_out_slice = try ws.dev0.emb_buf_1.slice().subslice(0, P * ModelConfig.embedding_dim);
        const d1_in_slice = try dev1.emb_buf_0.slice().subslice(0, P * ModelConfig.embedding_dim);
        const ev_opt = if (ws.boundary_event) |*ev| ev else null;
        try transferHiddenState(d0_out_slice, d1_in_slice, &ws.dev0.stream, &dev1.stream, ev_opt, ws.host_staging_buf, ws.peer_access_enabled);
        try dev1.stream.synchronize();
        try ws.dev0.stream.synchronize();
        const t_tr = decision.getMonotonicNanos();

        for (15..30) |l| {
            const dev_layer_idx = l - 15;
            const rf = if (model.layers[l].is_global) model.rope_freqs_dev1 else null;
            const rf_slice = if (rf) |r| r.slice() else null;
            try runLayerCuda(dev1, &model.layers[l], P, P, rf_slice, dev_layer_idx);
        }

        try dev1.stream.synchronize();
        try ws.dev0.stream.synchronize();
        const t_p1 = decision.getMonotonicNanos();

        ws.last_stage_timings.prompt_p0_ms = @as(f32, @floatFromInt(t_p0 - t_prefill_start)) / 1_000_000.0;
        ws.last_stage_timings.prompt_transfer_ms = @as(f32, @floatFromInt(t_tr - t_p0)) / 1_000_000.0;
        ws.last_stage_timings.prompt_p1_ms = @as(f32, @floatFromInt(t_p1 - t_tr)) / 1_000_000.0;
        ws.last_stage_timings.prompt_prefill_ms = @as(f32, @floatFromInt(t_p1 - t_prefill_start)) / 1_000_000.0;
        ws.last_stage_timings.prompt_cache_hit = false;
    }
}

fn runSingleGpuLayersEager(ws: *CudaWorkspace, model: *const CudaModel, P: usize, N: usize, answer_idx: usize) !void {
    ws.dev0.resetBufferPointers();
    for (0..ModelConfig.num_layers) |l| {
        const rf = if (model.layers[l].is_global) model.rope_freqs_dev0 else null;
        const rf_slice = if (rf) |r| r.slice() else null;
        try runLayerCuda(&ws.dev0, &model.layers[l], P, N, rf_slice, null);
    }
    const answer_slice = try ws.dev0.current_emb.slice().subslice(answer_idx * ModelConfig.embedding_dim, ModelConfig.embedding_dim);
    try cuda_mem.rmsNorm(answer_slice, model.output_norm.slice(), ws.dev0.norm_hidden.slice(), 1, ModelConfig.embedding_dim, 1e-6, &ws.dev0.stream);
}

fn runPartition0Eager(ws: *CudaWorkspace, model: *const CudaModel, P: usize, N: usize) !void {
    ws.dev0.resetBufferPointers();
    for (0..15) |l| {
        const rf = if (model.layers[l].is_global) model.rope_freqs_dev0 else null;
        const rf_slice = if (rf) |r| r.slice() else null;
        try runLayerCuda(&ws.dev0, &model.layers[l], P, N, rf_slice, null);
    }
}

fn runPartition1Eager(ws: *CudaWorkspace, model: *const CudaModel, P: usize, N: usize, answer_idx: usize) !void {
    var dev1 = &ws.dev1.?;
    dev1.resetBufferPointers();
    for (15..30) |l| {
        const rf = if (model.layers[l].is_global) model.rope_freqs_dev1 else null;
        const rf_slice = if (rf) |r| r.slice() else null;
        try runLayerCuda(dev1, &model.layers[l], P, N, rf_slice, null);
    }
    const answer_slice = try dev1.current_emb.slice().subslice(answer_idx * ModelConfig.embedding_dim, ModelConfig.embedding_dim);
    try cuda_mem.rmsNorm(answer_slice, model.output_norm.slice(), dev1.norm_hidden.slice(), 1, ModelConfig.embedding_dim, 1e-6, &dev1.stream);
}

fn runSingleGpuCanvasEager(ws: *CudaWorkspace, model: *const CudaModel, P: usize, C: usize, answer_pos: usize) !void {
    ws.dev0.resetBufferPointers();
    for (0..ModelConfig.num_layers) |l| {
        const rf = if (model.layers[l].is_global) model.rope_freqs_dev0 else null;
        const rf_slice = if (rf) |r| r.slice() else null;
        try runLayerCanvasCuda(&ws.dev0, &model.layers[l], l, P, C, rf_slice);
    }
    const answer_slice = try ws.dev0.current_emb.slice().subslice(answer_pos * ModelConfig.embedding_dim, ModelConfig.embedding_dim);
    try cuda_mem.rmsNorm(answer_slice, model.output_norm.slice(), ws.dev0.norm_hidden.slice(), 1, ModelConfig.embedding_dim, 1e-6, &ws.dev0.stream);
}

fn runPartition0CanvasEager(ws: *CudaWorkspace, model: *const CudaModel, P: usize, C: usize) !void {
    ws.dev0.resetBufferPointers();
    for (0..15) |l| {
        const rf = if (model.layers[l].is_global) model.rope_freqs_dev0 else null;
        const rf_slice = if (rf) |r| r.slice() else null;
        try runLayerCanvasCuda(&ws.dev0, &model.layers[l], l, P, C, rf_slice);
    }
}

fn runPartition1CanvasEager(ws: *CudaWorkspace, model: *const CudaModel, P: usize, C: usize, answer_pos: usize) !void {
    var dev1 = &ws.dev1.?;
    dev1.resetBufferPointers();
    for (15..30) |l| {
        const dev_layer_idx = l - 15;
        const rf = if (model.layers[l].is_global) model.rope_freqs_dev1 else null;
        const rf_slice = if (rf) |r| r.slice() else null;
        try runLayerCanvasCuda(dev1, &model.layers[l], dev_layer_idx, P, C, rf_slice);
    }
    const answer_slice = try dev1.current_emb.slice().subslice(answer_pos * ModelConfig.embedding_dim, ModelConfig.embedding_dim);
    try cuda_mem.rmsNorm(answer_slice, model.output_norm.slice(), dev1.norm_hidden.slice(), 1, ModelConfig.embedding_dim, 1e-6, &dev1.stream);
}

/// Phase 19: Executes device-resident self-conditioning MLP operations on stream without host synchronization.
pub fn runDeviceSelfConditioning(
    ws: *CudaDeviceWorkspace,
    model: *const CudaModel,
    target_pos: usize,
) !void {
    const sc_pre_norm = model.sc_pre_norm orelse return error.MissingSelfCondTensor;
    const sc_gate = model.sc_gate orelse return error.MissingSelfCondTensor;
    const sc_up = model.sc_up orelse return error.MissingSelfCondTensor;
    const sc_down = model.sc_down orelse return error.MissingSelfCondTensor;

    const emb_dim = ModelConfig.embedding_dim;
    const intermediate_dim = ModelConfig.self_cond_intermediate_dim;

    // 1. Pre-norm on Device: sc_soft_emb -> sc_sig_buf
    try cuda_mem.rmsNorm(
        ws.sc_soft_emb.slice(),
        sc_pre_norm.slice(),
        ws.sc_sig_buf.slice(),
        1,
        emb_dim,
        1e-6,
        &ws.stream,
    );

    // 2. Fused Gate + Up + GeGLU on Device: sc_sig_buf -> sc_geglu_buf
    try cuda_mem.denseGateUpGegluQ4_K(
        sc_gate.slice(),
        sc_up.slice(),
        ws.sc_sig_buf.slice(),
        ws.sc_geglu_buf.slice(),
        intermediate_dim,
        emb_dim,
        1,
        &ws.stream,
    );

    // 3. Down projection GEMV on Device (Q5_0): sc_geglu_buf -> sc_sig_buf
    try cuda_mem.gemvQ5_0(
        sc_down.slice(),
        ws.sc_geglu_buf.slice(),
        ws.sc_sig_buf.slice(),
        emb_dim,
        intermediate_dim,
        1,
        &ws.stream,
    );

    // 4. Add residual sc_sig_buf to current_emb at target_pos
    const ans_slice = try ws.current_emb.slice().subslice(target_pos * emb_dim, emb_dim);
    try cuda_mem.addResidual(
        ans_slice,
        ws.sc_sig_buf.slice(),
        &ws.stream,
    );

    // 5. Unweighted RMSNorm in-place on current_emb at target_pos
    try cuda_mem.rmsNorm(
        ans_slice,
        null,
        ans_slice,
        1,
        emb_dim,
        1e-6,
        &ws.stream,
    );
}

/// Applies device-resident self-conditioning on Device 0 at target_pos.
pub fn applyDeviceSelfConditioning(
    ws: *CudaWorkspace,
    model: *const CudaModel,
    cpu_model: DiffusionGemmaModel,
    candidate_token_ids: []const u32,
    candidate_probs: []const f32,
    target_pos: usize,
) !void {
    const emb_dim = ModelConfig.embedding_dim;
    const emb_scale: f32 = @sqrt(@as(f32, @floatFromInt(emb_dim)));

    // R39-01: Drain any pending stream operations before overwriting pinned host staging memory
    try ws.synchronize();

    // 1. Construct soft embedding on host staging buffer and copy to Device 0 sc_soft_emb
    const soft_emb = ws.host_staging_buf[0..emb_dim];
    @memset(soft_emb, 0.0);
    const row_temp = ws.host_staging_buf[emb_dim .. 2 * emb_dim];
    for (candidate_token_ids, candidate_probs) |tok, p| {
        if (p > 0.0) {
            const blocks = cpu_model.getEmbeddingBlocks(tok);
            quant.dequantizeRowQ6_K(blocks, row_temp);
            infer.accumulateScaledF32(soft_emb, row_temp, p * emb_scale);
        }
    }
    try ws.dev0.sc_soft_emb.copyFromHost(soft_emb);
    try runDeviceSelfConditioning(&ws.dev0, model, target_pos);
    try ws.dev0.stream.synchronize();
}

fn runSingleGpuStep2CanvasEager(ws: *CudaWorkspace, model: *const CudaModel, P: usize, C: usize, answer_pos: usize) !void {
    const emb_dim = ModelConfig.embedding_dim;
    ws.dev0.resetBufferPointers();
    const d0_cur = try ws.dev0.current_emb.slice().subslice(0, C * emb_dim);
    const d0_init = try ws.dev0.canvas_init_emb.slice().subslice(0, C * emb_dim);
    try d0_cur.copyFromDevice(d0_init, &ws.dev0.stream);
    const ans_slice = try ws.dev0.current_emb.slice().subslice(answer_pos * emb_dim, emb_dim);
    try ans_slice.copyFromDevice(ws.dev0.canvas_raw_ans.slice(), &ws.dev0.stream);
    try runDeviceSelfConditioning(&ws.dev0, model, answer_pos);
    for (0..ModelConfig.num_layers) |l| {
        const rf = if (model.layers[l].is_global) model.rope_freqs_dev0 else null;
        const rf_slice = if (rf) |r| r.slice() else null;
        try runLayerCanvasCuda(&ws.dev0, &model.layers[l], l, P, C, rf_slice);
    }
    try cuda_mem.rmsNorm(ans_slice, model.output_norm.slice(), ws.dev0.norm_hidden.slice(), 1, ModelConfig.embedding_dim, 1e-6, &ws.dev0.stream);
}

fn runPartition0Step2CanvasEager(ws: *CudaWorkspace, model: *const CudaModel, P: usize, C: usize, answer_pos: usize) !void {
    const emb_dim = ModelConfig.embedding_dim;
    ws.dev0.resetBufferPointers();
    const d0_cur = try ws.dev0.current_emb.slice().subslice(0, C * emb_dim);
    const d0_init = try ws.dev0.canvas_init_emb.slice().subslice(0, C * emb_dim);
    try d0_cur.copyFromDevice(d0_init, &ws.dev0.stream);
    const ans_slice = try ws.dev0.current_emb.slice().subslice(answer_pos * emb_dim, emb_dim);
    try ans_slice.copyFromDevice(ws.dev0.canvas_raw_ans.slice(), &ws.dev0.stream);
    try runDeviceSelfConditioning(&ws.dev0, model, answer_pos);
    for (0..15) |l| {
        const rf = if (model.layers[l].is_global) model.rope_freqs_dev0 else null;
        const rf_slice = if (rf) |r| r.slice() else null;
        try runLayerCanvasCuda(&ws.dev0, &model.layers[l], l, P, C, rf_slice);
    }
}

pub var enable_prompt_cache: bool = true;

/// Executes a structured read forward pass across all 30 layers using CUDA acceleration.
pub fn executeStructuredReadCuda(
    ws: *CudaWorkspace,
    model: *const CudaModel,
    cpu_model: DiffusionGemmaModel,
    prompt_tokens: []const u32,
    canvas_tokens: []const u32,
    answer_pos: usize,
    candidate_token_ids: []const u32,
    candidate_probs: ?[]const f32,
    out_raw_logits: []f32,
    out_softcapped_logits: []f32,
) !void {
    if (!is_enabled) return error.CudaDisabledAtBuild;
    if (answer_pos >= canvas_tokens.len) return error.InvalidParam;

    // R43-01: Drain any previously quarantined pinned buffers once devices are confirmed quiescent
    if (hasQuarantinedBuffers()) {
        drainQuarantinedPinnedBuffers() catch {};
    }

    const P = prompt_tokens.len;
    const C = canvas_tokens.len;
    const N = P + C;
    if (N > ws.dev0.max_seq_len or N > 512) return error.InvalidParam;

    // Phase 7B: Prompt KV prefix caching and canvas-only forward pass execution
    if (enable_prompt_cache and P > 0 and P <= CudaDeviceWorkspace.max_cached_prompt_tokens) {
        const prompt_hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(prompt_tokens));
        const prompt_hit = ws.prompt_cache_valid and ws.cached_prompt_hash == prompt_hash and ws.cached_prompt_len == P;

        if (!prompt_hit) {
            ws.invalidatePromptCache();
            try prefillPromptCuda(ws, model, cpu_model, prompt_tokens);
            ws.cached_prompt_hash = prompt_hash;
            ws.cached_prompt_len = P;
            ws.prompt_cache_valid = true;
        } else {
            ws.last_stage_timings.prompt_p0_ms = 0.0;
            ws.last_stage_timings.prompt_transfer_ms = 0.0;
            ws.last_stage_timings.prompt_p1_ms = 0.0;
            ws.last_stage_timings.prompt_prefill_ms = 0.0;
            ws.last_stage_timings.prompt_cache_hit = true;
        }

        if (candidate_probs == null) {
            // Step 1: Canvas tokens embedding on host & initial device backup (Phase 19)
            // R39-01: Drain any pending stream operations before embedding canvas tokens into pinned host staging memory
            try ws.synchronize();

            // Embed canvas tokens on host
            const host_emb = ws.host_staging_buf[0 .. C * ModelConfig.embedding_dim];
            const emb_scale: f32 = @sqrt(@as(f32, @floatFromInt(ModelConfig.embedding_dim)));
            for (canvas_tokens, 0..) |tok, i| {
                const blocks = cpu_model.getEmbeddingBlocks(tok);
                const row = host_emb[i * ModelConfig.embedding_dim .. (i + 1) * ModelConfig.embedding_dim];
                quant.dequantizeRowQ6_K(blocks, row);
                for (row) |*val| val.* *= emb_scale;
                if (i == answer_pos) {
                    const raw_row = ws.host_staging_buf[C * ModelConfig.embedding_dim .. (C + 1) * ModelConfig.embedding_dim];
                    @memcpy(raw_row, row);
                }
                ops.unweightedRmsNorm(row, 1e-6, row);
            }

            ws.dev0.resetBufferPointers();
            if (ws.dev1) |*d1| d1.resetBufferPointers();

            try ws.synchronize();
            const t_canvas_start = decision.getMonotonicNanos();

            const d0_cur_slice = try ws.dev0.current_emb.slice().subslice(0, C * ModelConfig.embedding_dim);
            const d0_init_slice = try ws.dev0.canvas_init_emb.slice().subslice(0, C * ModelConfig.embedding_dim);
            try d0_init_slice.copyFromHost(host_emb);
            try d0_cur_slice.copyFromDevice(d0_init_slice, &ws.dev0.stream);

            const raw_row = ws.host_staging_buf[C * ModelConfig.embedding_dim .. (C + 1) * ModelConfig.embedding_dim];
            try ws.dev0.canvas_raw_ans.slice().copyFromHost(raw_row);

            const graph_valid = (ws.graph0 != null and ws.cached_graph_prompt_len == P and ws.cached_graph_canvas_len == C and ws.cached_graph_answer_pos == answer_pos);

            if (model.topology == .single_gpu) {
                var replay_success = false;
                if (graph_valid and !inject_graph_replay_failure) {
                    if (ws.graph0.?.launch(&ws.dev0.stream)) |_| {
                        replay_success = true;
                    } else |_| {
                        if (ws.graph0) |*g0| {
                            g0.deinit();
                            ws.graph0 = null;
                        }
                        ws.cached_graph_prompt_len = 0;
                        ws.cached_graph_canvas_len = 0;
                        ws.cached_graph_answer_pos = 0;
                        try ws.dev0.stream.synchronize();
                    }
                }

                if (replay_success) {
                    try ws.dev0.stream.synchronize();
                    try ws.dev0.norm_hidden.copyToHost(ws.host_norm_hidden);
                } else {
                    if (ws.graph0) |*g0| {
                        g0.deinit();
                        ws.graph0 = null;
                    }

                    var capture0_ok = false;
                    if (cuda_mem.CudaGraph.beginCapture(&ws.dev0.stream)) |_| {
                        capture0_ok = true;
                    } else |_| {}

                    if (capture0_ok) {
                        errdefer cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                        try runSingleGpuCanvasEager(ws, model, P, C, answer_pos);

                        var g0_instantiated = false;
                        if (!inject_graph_capture_failure) {
                            if (cuda_mem.CudaGraph.endCapture(&ws.dev0.stream)) |g0| {
                                ws.graph0 = g0;
                                var launch_ok = false;
                                if (!inject_graph_replay_failure) {
                                    if (ws.graph0.?.launch(&ws.dev0.stream)) |_| {
                                        launch_ok = true;
                                    } else |_| {}
                                }
                                if (launch_ok) {
                                    g0_instantiated = true;
                                    ws.cached_graph_prompt_len = P;
                                    ws.cached_graph_canvas_len = C;
                                    ws.cached_graph_answer_pos = answer_pos;
                                } else {
                                    ws.graph0.?.deinit();
                                    ws.graph0 = null;
                                    ws.cached_graph_prompt_len = 0;
                                    ws.cached_graph_canvas_len = 0;
                                    ws.cached_graph_answer_pos = 0;
                                }
                            } else |_| {
                                ws.cached_graph_prompt_len = 0;
                                ws.cached_graph_canvas_len = 0;
                                ws.cached_graph_answer_pos = 0;
                            }
                        } else {
                            cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                            ws.cached_graph_prompt_len = 0;
                            ws.cached_graph_canvas_len = 0;
                            ws.cached_graph_answer_pos = 0;
                        }

                        if (!g0_instantiated) {
                            try runSingleGpuCanvasEager(ws, model, P, C, answer_pos);
                        }
                    } else {
                        try runSingleGpuCanvasEager(ws, model, P, C, answer_pos);
                    }

                    try ws.dev0.stream.synchronize();
                    try ws.dev0.norm_hidden.copyToHost(ws.host_norm_hidden);
                }
            } else {
                var dev1 = &ws.dev1.?;
                const d0_out_slice = try ws.dev0.emb_buf_1.slice().subslice(0, C * ModelConfig.embedding_dim);
                const d1_in_slice = try dev1.emb_buf_0.slice().subslice(0, C * ModelConfig.embedding_dim);
                const ev_opt = if (ws.boundary_event) |*ev| ev else null;

                var replay_success = false;
                if (graph_valid and ws.graph1 != null and !inject_graph_replay_failure) {
                    var l0_ok = false;
                    if (ws.graph0.?.launch(&ws.dev0.stream)) |_| {
                        l0_ok = true;
                    } else |_| {}

                    var tr_ok = false;
                    if (l0_ok) {
                        if (transferHiddenState(d0_out_slice, d1_in_slice, &ws.dev0.stream, &dev1.stream, ev_opt, ws.host_staging_buf, ws.peer_access_enabled)) |_| {
                            tr_ok = true;
                        } else |_| {}

                        if (tr_ok) {
                            if (!inject_graph1_replay_failure) {
                                if (ws.graph1.?.launch(&dev1.stream)) |_| {
                                    replay_success = true;
                                } else |_| {}
                            }
                        }
                    }
                    if (!replay_success) {
                        if (ws.graph0) |*g0| {
                            g0.deinit();
                            ws.graph0 = null;
                        }
                        if (ws.graph1) |*g1| {
                            g1.deinit();
                            ws.graph1 = null;
                        }
                        ws.cached_graph_prompt_len = 0;
                        ws.cached_graph_canvas_len = 0;
                        ws.cached_graph_answer_pos = 0;

                        // R40-01 / R41-01: Drain all referencing device streams before entering eager fallback.
                        try ws.synchronize();
                    }
                }

                if (!replay_success) {
                    // --- Partition 0 (Device 0, Layers 0..14) ---
                    var g0_instantiated = false;
                    var capture0_ok = false;
                    if (cuda_mem.CudaGraph.beginCapture(&ws.dev0.stream)) |_| {
                        capture0_ok = true;
                    } else |_| {}

                    if (capture0_ok) {
                        errdefer cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                        try runPartition0CanvasEager(ws, model, P, C);

                        if (!inject_graph_capture_failure) {
                            if (cuda_mem.CudaGraph.endCapture(&ws.dev0.stream)) |g0| {
                                ws.graph0 = g0;
                                var p0_launch_ok = false;
                                if (!inject_graph_replay_failure) {
                                    if (ws.graph0.?.launch(&ws.dev0.stream)) |_| {
                                        p0_launch_ok = true;
                                    } else |_| {}
                                }
                                if (p0_launch_ok) {
                                    g0_instantiated = true;
                                } else {
                                    ws.graph0.?.deinit();
                                    ws.graph0 = null;
                                }
                            } else |_| {}
                        } else {
                            cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                        }

                        if (!g0_instantiated) {
                            try runPartition0CanvasEager(ws, model, P, C);
                        }
                    } else {
                        try runPartition0CanvasEager(ws, model, P, C);
                    }

                    // Boundary transfer H_15 to Device 1 (canvas length C)
                    // R39-01: Ensure streams are drained if subsequent execution fails
                    errdefer ws.synchronize() catch {};
                    try transferHiddenState(d0_out_slice, d1_in_slice, &ws.dev0.stream, &dev1.stream, ev_opt, ws.host_staging_buf, ws.peer_access_enabled);

                    // --- Partition 1 (Device 1, Layers 15..29 + Output RMSNorm) ---
                    var capture1_ok = false;
                    if (cuda_mem.CudaGraph.beginCapture(&dev1.stream)) |_| {
                        capture1_ok = true;
                    } else |_| {}

                    if (capture1_ok) {
                        errdefer cuda_mem.CudaGraph.cancelCapture(&dev1.stream);
                        try runPartition1CanvasEager(ws, model, P, C, answer_pos);

                        var g1_instantiated = false;
                        if (!inject_graph_capture_failure) {
                            if (cuda_mem.CudaGraph.endCapture(&dev1.stream)) |g1| {
                                ws.graph1 = g1;
                                var p1_launch_ok = false;
                                if (!inject_graph_replay_failure) {
                                    if (ws.graph1.?.launch(&dev1.stream)) |_| {
                                        p1_launch_ok = true;
                                    } else |_| {}
                                }
                                if (p1_launch_ok and g0_instantiated) {
                                    g1_instantiated = true;
                                    ws.cached_graph_prompt_len = P;
                                    ws.cached_graph_canvas_len = C;
                                    ws.cached_graph_answer_pos = answer_pos;
                                } else {
                                    if (ws.graph1) |*g| {
                                        g.deinit();
                                        ws.graph1 = null;
                                    }
                                    if (ws.graph0) |*g| {
                                        g.deinit();
                                        ws.graph0 = null;
                                    }
                                    ws.cached_graph_prompt_len = 0;
                                    ws.cached_graph_canvas_len = 0;
                                    ws.cached_graph_answer_pos = 0;
                                }
                            } else |_| {
                                if (ws.graph0) |*g| {
                                    g.deinit();
                                    ws.graph0 = null;
                                }
                                ws.cached_graph_prompt_len = 0;
                                ws.cached_graph_canvas_len = 0;
                                ws.cached_graph_answer_pos = 0;
                            }
                        } else {
                            cuda_mem.CudaGraph.cancelCapture(&dev1.stream);
                            if (ws.graph0) |*g| {
                                g.deinit();
                                ws.graph0 = null;
                            }
                            ws.cached_graph_prompt_len = 0;
                            ws.cached_graph_canvas_len = 0;
                            ws.cached_graph_answer_pos = 0;
                        }

                        if (!g1_instantiated) {
                            try runPartition1CanvasEager(ws, model, P, C, answer_pos);
                        }
                    } else {
                        if (ws.graph0) |*g| {
                            g.deinit();
                            ws.graph0 = null;
                        }
                        ws.cached_graph_prompt_len = 0;
                        ws.cached_graph_canvas_len = 0;
                        ws.cached_graph_answer_pos = 0;
                        try runPartition1CanvasEager(ws, model, P, C, answer_pos);
                    }
                }

                // Copy normalized hidden state to host asynchronously on dev1.stream, then wait for completion
                try dev1.norm_hidden.copyToHost(ws.host_norm_hidden);
                try dev1.stream.synchronize();
                try ws.dev0.stream.synchronize();
            }

            // Project candidate logits on host using Q6_K dot products
            infer.readCandidateLogits(cpu_model, ws.host_norm_hidden, candidate_token_ids, out_raw_logits, out_softcapped_logits);

            const t_canvas_end = decision.getMonotonicNanos();
            ws.last_canvas_ms = @as(f32, @floatFromInt(t_canvas_end - t_canvas_start)) / 1_000_000.0;
            return;
        } else {
            // Step 2: Multi-Step Graph Chaining with Device-Resident Self-Conditioning (Phase 19)
            const sc_probs = candidate_probs.?;
            const emb_dim = ModelConfig.embedding_dim;
            const emb_scale: f32 = @sqrt(@as(f32, @floatFromInt(emb_dim)));

            // 1. Construct soft embedding on host staging buffer
            const soft_emb = ws.host_staging_buf[0..emb_dim];
            @memset(soft_emb, 0.0);
            const row_temp = ws.host_staging_buf[emb_dim .. 2 * emb_dim];
            for (candidate_token_ids, sc_probs) |tok, p| {
                if (p > 0.0) {
                    const blocks = cpu_model.getEmbeddingBlocks(tok);
                    quant.dequantizeRowQ6_K(blocks, row_temp);
                    infer.accumulateScaledF32(soft_emb, row_temp, p * emb_scale);
                }
            }

            ws.dev0.resetBufferPointers();
            if (ws.dev1) |*d1| d1.resetBufferPointers();

            const t_canvas_start = decision.getMonotonicNanos();

            // 2. Asynchronously copy soft embedding into device buffer on ws.dev0.stream
            try ws.dev0.sc_soft_emb.slice().copyFromHostAsync(soft_emb, &ws.dev0.stream);

            const step2_graph_valid = (ws.graph0_step2 != null and
                ws.cached_graph_step2_prompt_len == P and
                ws.cached_graph_step2_canvas_len == C and
                ws.cached_graph_step2_answer_pos == answer_pos);

            if (model.topology == .single_gpu) {
                var replay_success = false;
                if (step2_graph_valid and !inject_graph_replay_failure) {
                    if (ws.graph0_step2.?.launch(&ws.dev0.stream)) |_| {
                        replay_success = true;
                    } else |_| {
                        if (ws.graph0_step2) |*g| {
                            g.deinit();
                            ws.graph0_step2 = null;
                        }
                        ws.cached_graph_step2_prompt_len = 0;
                        ws.cached_graph_step2_canvas_len = 0;
                        ws.cached_graph_step2_answer_pos = 0;
                        try ws.dev0.stream.synchronize();
                    }
                }

                if (replay_success) {
                    try ws.dev0.stream.synchronize();
                    try ws.dev0.norm_hidden.copyToHost(ws.host_norm_hidden);
                } else {
                    if (ws.graph0_step2) |*g| {
                        g.deinit();
                        ws.graph0_step2 = null;
                    }

                    var capture0_ok = false;
                    if (cuda_mem.CudaGraph.beginCapture(&ws.dev0.stream)) |_| {
                        capture0_ok = true;
                    } else |_| {}

                    if (capture0_ok) {
                        errdefer cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                        try runSingleGpuStep2CanvasEager(ws, model, P, C, answer_pos);

                        var g0_instantiated = false;
                        if (!inject_graph_capture_failure) {
                            if (cuda_mem.CudaGraph.endCapture(&ws.dev0.stream)) |g0| {
                                ws.graph0_step2 = g0;
                                var launch_ok = false;
                                if (!inject_graph_replay_failure) {
                                    if (ws.graph0_step2.?.launch(&ws.dev0.stream)) |_| {
                                        launch_ok = true;
                                    } else |_| {}
                                }
                                if (launch_ok) {
                                    g0_instantiated = true;
                                    ws.cached_graph_step2_prompt_len = P;
                                    ws.cached_graph_step2_canvas_len = C;
                                    ws.cached_graph_step2_answer_pos = answer_pos;
                                } else {
                                    ws.graph0_step2.?.deinit();
                                    ws.graph0_step2 = null;
                                    ws.cached_graph_step2_prompt_len = 0;
                                    ws.cached_graph_step2_canvas_len = 0;
                                    ws.cached_graph_step2_answer_pos = 0;
                                }
                            } else |_| {
                                ws.cached_graph_step2_prompt_len = 0;
                                ws.cached_graph_step2_canvas_len = 0;
                                ws.cached_graph_step2_answer_pos = 0;
                            }
                        } else {
                            cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                            ws.cached_graph_step2_prompt_len = 0;
                            ws.cached_graph_step2_canvas_len = 0;
                            ws.cached_graph_step2_answer_pos = 0;
                        }

                        if (!g0_instantiated) {
                            try runSingleGpuStep2CanvasEager(ws, model, P, C, answer_pos);
                        }
                    } else {
                        try runSingleGpuStep2CanvasEager(ws, model, P, C, answer_pos);
                    }

                    try ws.dev0.stream.synchronize();
                    try ws.dev0.norm_hidden.copyToHost(ws.host_norm_hidden);
                }
            } else {
                var dev1 = &ws.dev1.?;
                const d0_out_slice = try ws.dev0.emb_buf_1.slice().subslice(0, C * ModelConfig.embedding_dim);
                const d1_in_slice = try dev1.emb_buf_0.slice().subslice(0, C * ModelConfig.embedding_dim);
                const ev_opt = if (ws.boundary_event) |*ev| ev else null;

                var replay_success = false;
                if (step2_graph_valid and ws.graph1 != null and !inject_graph_replay_failure) {
                    var l0_ok = false;
                    if (ws.graph0_step2.?.launch(&ws.dev0.stream)) |_| {
                        l0_ok = true;
                    } else |_| {}

                    var tr_ok = false;
                    if (l0_ok) {
                        if (transferHiddenState(d0_out_slice, d1_in_slice, &ws.dev0.stream, &dev1.stream, ev_opt, ws.host_staging_buf, ws.peer_access_enabled)) |_| {
                            tr_ok = true;
                        } else |_| {}

                        if (tr_ok) {
                            if (!inject_graph1_replay_failure) {
                                if (ws.graph1.?.launch(&dev1.stream)) |_| {
                                    replay_success = true;
                                } else |_| {}
                            }
                        }
                    }
                    if (!replay_success) {
                        if (ws.graph0_step2) |*g| {
                            g.deinit();
                            ws.graph0_step2 = null;
                        }
                        ws.cached_graph_step2_prompt_len = 0;
                        ws.cached_graph_step2_canvas_len = 0;
                        ws.cached_graph_step2_answer_pos = 0;

                        try ws.synchronize();
                    }
                }

                if (!replay_success) {
                    var g0_instantiated = false;
                    var capture0_ok = false;
                    if (cuda_mem.CudaGraph.beginCapture(&ws.dev0.stream)) |_| {
                        capture0_ok = true;
                    } else |_| {}

                    if (capture0_ok) {
                        errdefer cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                        try runPartition0Step2CanvasEager(ws, model, P, C, answer_pos);

                        if (!inject_graph_capture_failure) {
                            if (cuda_mem.CudaGraph.endCapture(&ws.dev0.stream)) |g0| {
                                ws.graph0_step2 = g0;
                                var p0_launch_ok = false;
                                if (!inject_graph_replay_failure) {
                                    if (ws.graph0_step2.?.launch(&ws.dev0.stream)) |_| {
                                        p0_launch_ok = true;
                                    } else |_| {}
                                }
                                if (p0_launch_ok) {
                                    g0_instantiated = true;
                                    ws.cached_graph_step2_prompt_len = P;
                                    ws.cached_graph_step2_canvas_len = C;
                                    ws.cached_graph_step2_answer_pos = answer_pos;
                                } else {
                                    ws.graph0_step2.?.deinit();
                                    ws.graph0_step2 = null;
                                }
                            } else |_| {}
                        } else {
                            cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                        }

                        if (!g0_instantiated) {
                            try runPartition0Step2CanvasEager(ws, model, P, C, answer_pos);
                        }
                    } else {
                        try runPartition0Step2CanvasEager(ws, model, P, C, answer_pos);
                    }

                    errdefer ws.synchronize() catch {};
                    try transferHiddenState(d0_out_slice, d1_in_slice, &ws.dev0.stream, &dev1.stream, ev_opt, ws.host_staging_buf, ws.peer_access_enabled);

                    var p1_done = false;
                    if (ws.graph1 != null and !inject_graph1_replay_failure) {
                        if (ws.graph1.?.launch(&dev1.stream)) |_| {
                            p1_done = true;
                        } else |_| {}
                    }
                    if (!p1_done) {
                        try runPartition1CanvasEager(ws, model, P, C, answer_pos);
                    }
                }

                try dev1.norm_hidden.copyToHost(ws.host_norm_hidden);
                try dev1.stream.synchronize();
                try ws.dev0.stream.synchronize();
            }

            infer.readCandidateLogits(cpu_model, ws.host_norm_hidden, candidate_token_ids, out_raw_logits, out_softcapped_logits);
            const t_canvas_end = decision.getMonotonicNanos();
            ws.last_canvas_ms = @as(f32, @floatFromInt(t_canvas_end - t_canvas_start)) / 1_000_000.0;
            return;
        }
    }

    // --- Uncached Fallback Execution Path (when enable_prompt_cache == false or P > max_prompt_tokens) ---
    ws.invalidatePromptCache();
    const answer_idx = P + answer_pos;

    // R39-01: Drain any pending stream operations before overwriting pinned host staging memory
    try ws.synchronize();

    // 1. Dual-region input embedding on host (takes ~0.05ms)
    const host_emb = ws.host_staging_buf[0 .. N * ModelConfig.embedding_dim];
    const emb_scale: f32 = @sqrt(@as(f32, @floatFromInt(ModelConfig.embedding_dim)));

    for (prompt_tokens, 0..) |tok, i| {
        const blocks = cpu_model.getEmbeddingBlocks(tok);
        const row = host_emb[i * ModelConfig.embedding_dim .. (i + 1) * ModelConfig.embedding_dim];
        quant.dequantizeRowQ6_K(blocks, row);
        for (row) |*val| val.* *= emb_scale;
    }
    for (canvas_tokens, 0..) |tok, i| {
        const blocks = cpu_model.getEmbeddingBlocks(tok);
        const row = host_emb[(P + i) * ModelConfig.embedding_dim .. (P + i + 1) * ModelConfig.embedding_dim];
        quant.dequantizeRowQ6_K(blocks, row);
        for (row) |*val| val.* *= emb_scale;
        if (candidate_probs == null or i != answer_pos) {
            ops.unweightedRmsNorm(row, 1e-6, row);
        }
    }

    // 2. Reset buffer pointers to initial configuration and synchronize streams
    ws.dev0.resetBufferPointers();
    if (ws.dev1) |*d1| d1.resetBufferPointers();

    try ws.synchronize();
    const t_fallback_start = decision.getMonotonicNanos();

    const d0_emb_slice = try ws.dev0.current_emb.slice().subslice(0, N * ModelConfig.embedding_dim);
    try d0_emb_slice.copyFromHost(host_emb);
    try ws.dev0.stream.synchronize();

    // Phase 9: If self-conditioning is enabled (steps t >= 2), apply device-resident SC MLP at answer position
    if (candidate_probs) |sc_probs| {
        try applyDeviceSelfConditioning(ws, model, cpu_model, candidate_token_ids, sc_probs, answer_idx);
    }

    // 3. Execute Transformer Blocks
    const graph_valid = (ws.graph0 != null and ws.cached_graph_prompt_len == P and ws.cached_graph_canvas_len == C and ws.cached_graph_answer_pos == answer_pos);

    if (model.topology == .single_gpu) {
        var replay_success = false;
        if (graph_valid and !inject_graph_replay_failure) {
            if (ws.graph0.?.launch(&ws.dev0.stream)) |_| {
                replay_success = true;
            } else |_| {
                if (ws.graph0) |*g0| {
                    g0.deinit();
                    ws.graph0 = null;
                }
                ws.cached_graph_prompt_len = 0;
                ws.cached_graph_canvas_len = 0;
                ws.cached_graph_answer_pos = 0;
                try ws.dev0.stream.synchronize();
            }
        }

        if (replay_success) {
            try ws.dev0.stream.synchronize();
            try ws.dev0.norm_hidden.copyToHost(ws.host_norm_hidden);
        } else {
            if (ws.graph0) |*g0| {
                g0.deinit();
                ws.graph0 = null;
            }

            const answer_idx_sg = P + answer_pos;
            var capture0_ok = false;
            if (cuda_mem.CudaGraph.beginCapture(&ws.dev0.stream)) |_| {
                capture0_ok = true;
            } else |_| {}

            if (capture0_ok) {
                errdefer cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                try runSingleGpuLayersEager(ws, model, P, N, answer_idx_sg);

                var g0_instantiated = false;
                if (!inject_graph_capture_failure) {
                    if (cuda_mem.CudaGraph.endCapture(&ws.dev0.stream)) |g0| {
                        ws.graph0 = g0;
                        ws.cached_graph_prompt_len = P;
                        ws.cached_graph_canvas_len = C;
                        ws.cached_graph_answer_pos = answer_pos;
                        var launch_ok = false;
                        if (!inject_graph_replay_failure) {
                            if (ws.graph0.?.launch(&ws.dev0.stream)) |_| {
                                launch_ok = true;
                            } else |_| {}
                        }
                        if (launch_ok) {
                            g0_instantiated = true;
                        } else {
                            ws.graph0.?.deinit();
                            ws.graph0 = null;
                        }
                    } else |_| {}
                } else {
                    cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                }

                // If graph finalization, initial launch, or injected capture failure occurred:
                if (!g0_instantiated) {
                    try runSingleGpuLayersEager(ws, model, P, N, answer_idx_sg);
                }
            } else {
                try runSingleGpuLayersEager(ws, model, P, N, answer_idx_sg);
            }

            try ws.dev0.stream.synchronize();
            try ws.dev0.norm_hidden.copyToHost(ws.host_norm_hidden);
        }
    } else {
        var dev1 = &ws.dev1.?;
        const d0_out_slice = try ws.dev0.emb_buf_1.slice().subslice(0, N * ModelConfig.embedding_dim);
        const d1_in_slice = try dev1.emb_buf_0.slice().subslice(0, N * ModelConfig.embedding_dim);
        const ev_opt = if (ws.boundary_event) |*ev| ev else null;

        var replay_success = false;
        if (graph_valid and ws.graph1 != null and !inject_graph_replay_failure) {
            var l0_ok = false;
            if (ws.graph0.?.launch(&ws.dev0.stream)) |_| {
                l0_ok = true;
            } else |_| {}

            var tr_ok = false;
            if (l0_ok) {
                if (transferHiddenState(d0_out_slice, d1_in_slice, &ws.dev0.stream, &dev1.stream, ev_opt, ws.host_staging_buf, ws.peer_access_enabled)) |_| {
                    tr_ok = true;
                } else |_| {}

                if (tr_ok) {
                    if (!inject_graph1_replay_failure) {
                        if (ws.graph1.?.launch(&dev1.stream)) |_| {
                            replay_success = true;
                        } else |_| {}
                    }
                }
            }
            if (!replay_success) {
                if (ws.graph0) |*g0| {
                    g0.deinit();
                    ws.graph0 = null;
                }
                if (ws.graph1) |*g1| {
                    g1.deinit();
                    ws.graph1 = null;
                }
                ws.cached_graph_prompt_len = 0;
                ws.cached_graph_canvas_len = 0;
                ws.cached_graph_answer_pos = 0;

                // R40-01 / R41-01: Drain all referencing device streams before entering eager fallback.
                // If stream synchronization fails, stop immediately to prevent unsafe reuse
                // of host_staging_buf while asynchronous transfers may still be pending.
                try ws.synchronize();
            }
        }

        if (!replay_success) {
            // --- Partition 0 (Device 0, Layers 0..14) ---
            var capture0_ok = false;
            if (cuda_mem.CudaGraph.beginCapture(&ws.dev0.stream)) |_| {
                capture0_ok = true;
            } else |_| {}

            if (capture0_ok) {
                errdefer cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                try runPartition0Eager(ws, model, P, N);

                var g0_instantiated = false;
                if (!inject_graph_capture_failure) {
                    if (cuda_mem.CudaGraph.endCapture(&ws.dev0.stream)) |g0| {
                        ws.graph0 = g0;
                        var p0_launch_ok = false;
                        if (!inject_graph_replay_failure) {
                            if (ws.graph0.?.launch(&ws.dev0.stream)) |_| {
                                p0_launch_ok = true;
                            } else |_| {}
                        }
                        if (p0_launch_ok) {
                            g0_instantiated = true;
                        } else {
                            ws.graph0.?.deinit();
                            ws.graph0 = null;
                        }
                    } else |_| {}
                } else {
                    cuda_mem.CudaGraph.cancelCapture(&ws.dev0.stream);
                }

                if (!g0_instantiated) {
                    try runPartition0Eager(ws, model, P, N);
                }
            } else {
                try runPartition0Eager(ws, model, P, N);
            }

            // Boundary transfer H_15 to Device 1
            // R39-01: Ensure streams are drained if subsequent execution fails
            errdefer ws.synchronize() catch {};
            try transferHiddenState(d0_out_slice, d1_in_slice, &ws.dev0.stream, &dev1.stream, ev_opt, ws.host_staging_buf, ws.peer_access_enabled);

            // --- Partition 1 (Device 1, Layers 15..29 + Output RMSNorm) ---
            var capture1_ok = false;
            if (cuda_mem.CudaGraph.beginCapture(&dev1.stream)) |_| {
                capture1_ok = true;
            } else |_| {}

            if (capture1_ok) {
                errdefer cuda_mem.CudaGraph.cancelCapture(&dev1.stream);
                try runPartition1Eager(ws, model, P, N, answer_idx);

                var g1_instantiated = false;
                if (!inject_graph_capture_failure) {
                    if (cuda_mem.CudaGraph.endCapture(&dev1.stream)) |g1| {
                        ws.graph1 = g1;
                        ws.cached_graph_prompt_len = P;
                        ws.cached_graph_canvas_len = C;
                        ws.cached_graph_answer_pos = answer_pos;
                        var p1_launch_ok = false;
                        if (!inject_graph_replay_failure) {
                            if (ws.graph1.?.launch(&dev1.stream)) |_| {
                                p1_launch_ok = true;
                            } else |_| {}
                        }
                        if (p1_launch_ok) {
                            g1_instantiated = true;
                        } else {
                            ws.graph1.?.deinit();
                            ws.graph1 = null;
                        }
                    } else |_| {}
                } else {
                    cuda_mem.CudaGraph.cancelCapture(&dev1.stream);
                }

                if (!g1_instantiated) {
                    try runPartition1Eager(ws, model, P, N, answer_idx);
                }
            } else {
                try runPartition1Eager(ws, model, P, N, answer_idx);
            }
        }

        try dev1.stream.synchronize();
        try ws.dev0.stream.synchronize();

        // Copy normalized hidden state to host
        try dev1.norm_hidden.copyToHost(ws.host_norm_hidden);
        try dev1.stream.synchronize();
    }

    // 4. Project candidate logits on host using Q6_K dot products
    infer.readCandidateLogits(cpu_model, ws.host_norm_hidden, candidate_token_ids, out_raw_logits, out_softcapped_logits);

    const t_fallback_end = decision.getMonotonicNanos();
    ws.last_canvas_ms = @as(f32, @floatFromInt(t_fallback_end - t_fallback_start)) / 1_000_000.0;
    ws.last_stage_timings.prompt_p0_ms = 0.0;
    ws.last_stage_timings.prompt_transfer_ms = 0.0;
    ws.last_stage_timings.prompt_p1_ms = 0.0;
    ws.last_stage_timings.prompt_prefill_ms = 0.0;
    ws.last_stage_timings.prompt_cache_hit = false;
}

// ---------------------------------------------------------------------------
// Unit Tests & Parity Verification
// ---------------------------------------------------------------------------

test "CudaDeviceWorkspace and CudaWorkspace allocation lifecycle" {
    if (!is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const topo = cuda.getTopology() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (topo.reported_devices == 0) return error.SkipZigTest;
    const dev0 = topo.devices[0].device_id;

    var dev_ws = try CudaDeviceWorkspace.init(allocator, dev0, 32);
    defer dev_ws.deinit();

    try std.testing.expectEqual(@as(usize, 32), dev_ws.max_seq_len);
    try std.testing.expectEqual(@as(usize, 32 * ModelConfig.embedding_dim), dev_ws.current_emb.len);

    var ws = try CudaWorkspace.init(allocator, .single_gpu, [_]i32{ dev0, dev0 }, 32);
    defer ws.deinit();
    try std.testing.expect(ws.dev1 == null);

    if (topo.reported_devices >= 2) {
        const dev1 = topo.devices[1].device_id;
        var ws_multi = try CudaWorkspace.init(allocator, .dual_gpu_split, [_]i32{ dev0, dev1 }, 32);
        defer ws_multi.deinit();
        try std.testing.expect(ws_multi.dev1 != null);
    }
}

test "transferHiddenState intra-device and cross-device parity" {
    if (!is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const topo = cuda.getTopology() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (topo.reported_devices == 0) return error.SkipZigTest;
    const dev0 = topo.devices[0].device_id;

    const count: usize = 2816 * 2;
    const host_src = try allocator.alloc(f32, count);
    defer allocator.free(host_src);
    const host_dst = try allocator.alloc(f32, count);
    defer allocator.free(host_dst);
    var staging_pinned = try cuda_mem.PinnedBuffer(f32).init(count);
    defer staging_pinned.deinit();
    const staging = staging_pinned.slice();

    for (host_src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 1000)) * 0.1;
    }

    // Intra-device transfer on Device 0
    var stream0 = try CudaStream.init(dev0);
    defer stream0.deinit();

    var buf_src0 = try DeviceBuffer(f32).init(dev0, count);
    defer buf_src0.deinit();
    var buf_dst0 = try DeviceBuffer(f32).init(dev0, count);
    defer buf_dst0.deinit();

    try buf_src0.copyFromHost(host_src);
    try transferHiddenState(buf_src0.slice(), buf_dst0.slice(), &stream0, &stream0, null, staging, false);
    try stream0.synchronize();
    try buf_dst0.copyToHost(host_dst);

    for (host_src, host_dst) |s, d| {
        try std.testing.expectEqual(s, d);
    }

    // Cross-device transfer if 2 devices present
    if (topo.reported_devices >= 2) {
        const dev1 = topo.devices[1].device_id;
        var stream1 = try CudaStream.init(dev1);
        defer stream1.deinit();
        var ev = try cuda_mem.CudaEvent.init(dev0);
        defer ev.deinit();

        var buf_dst1 = try DeviceBuffer(f32).init(dev1, count);
        defer buf_dst1.deinit();

        // Test host-staged transfer path
        try transferHiddenState(buf_src0.slice(), buf_dst1.slice(), &stream0, &stream1, &ev, staging, false);
        try stream1.synchronize();
        @memset(host_dst, 0.0);
        try buf_dst1.copyToHost(host_dst);

        for (host_src, host_dst) |s, d| {
            try std.testing.expectEqual(s, d);
        }

        // Test peer transfer path if enabled
        const p2p_rc = cuda.c.zeno_cuda_enable_peer_access(dev0, dev1);
        const p2p_ok = (p2p_rc == 0);
        try transferHiddenState(buf_src0.slice(), buf_dst1.slice(), &stream0, &stream1, &ev, staging, p2p_ok);
        try stream1.synchronize();
        @memset(host_dst, 0.0);
        try buf_dst1.copyToHost(host_dst);

        for (host_src, host_dst) |s, d| {
            try std.testing.expectEqual(s, d);
        }
    }
}

test "runLayerCuda single-layer forward pass on local model layer 0 if present" {
    if (!is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var cpu_model = DiffusionGemmaModel.loadFromFile(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return error.SkipZigTest;
        return err;
    };
    defer cpu_model.deinit();

    const topo = cuda.getTopology() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (topo.reported_devices == 0) return error.SkipZigTest;
    const dev0 = topo.devices[0].device_id;

    const num_tokens: usize = 4;
    const prompt_len: usize = 2;
    const total_elements = num_tokens * ModelConfig.embedding_dim;

    // Load Layer 0 onto Device 0
    var layer0 = try CudaLayerWeights.initFromModelLayer(cpu_model, 0, dev0);
    defer layer0.deinit();

    var dev_ws = try CudaDeviceWorkspace.init(allocator, dev0, num_tokens);
    defer dev_ws.deinit();

    // Prepare test input activations
    const host_in = try allocator.alloc(f32, total_elements);
    defer allocator.free(host_in);
    for (host_in, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i + 1)) * 0.01) * 0.1;
    }

    // 1. Run CPU layer forward pass
    const cpu_in = try allocator.alloc(f32, total_elements);
    defer allocator.free(cpu_in);
    @memcpy(cpu_in, host_in);

    const cpu_out = try allocator.alloc(f32, total_elements);
    defer allocator.free(cpu_out);
    @memset(cpu_out, 0.0);

    const cpu_norm_x = try allocator.alloc(f32, total_elements);
    defer allocator.free(cpu_norm_x);
    const cpu_q = try allocator.alloc(f32, num_tokens * infer.max_q_dim);
    defer allocator.free(cpu_q);
    const cpu_k = try allocator.alloc(f32, num_tokens * infer.max_k_dim);
    defer allocator.free(cpu_k);
    const cpu_v = try allocator.alloc(f32, num_tokens * infer.max_k_dim);
    defer allocator.free(cpu_v);
    const cpu_attn_out = try allocator.alloc(f32, total_elements);
    defer allocator.free(cpu_attn_out);

    const fpw = infer.WorkerScratch.floatsPerWorker(num_tokens);
    const worker_storage = try allocator.alloc(f32, fpw);
    defer allocator.free(worker_storage);
    var worker_scratch = infer.WorkerScratch.sliceFrom(worker_storage, num_tokens);

    var cpu_ctx = infer.LayerContext{
        .model = cpu_model,
        .layer_idx = 0,
        .prompt_len = prompt_len,
        .total_len = num_tokens,
        .max_q_dim = infer.max_q_dim,
        .max_k_dim = infer.max_k_dim,
        .rope_freqs = null,
        .current_emb = cpu_in,
        .next_emb = cpu_out,
        .norm_x_buf = cpu_norm_x,
        .q_buf = cpu_q,
        .k_buf = cpu_k,
        .v_buf = cpu_v,
        .attn_out_buf = cpu_attn_out,
    };

    infer.runQKVWorker(&cpu_ctx, 0, num_tokens);
    infer.runAttnWorker(&cpu_ctx, &worker_scratch, 0, num_tokens);
    infer.runFFNWorker(&cpu_ctx, &worker_scratch, 0, num_tokens);

    // 2. Run GPU layer forward pass
    try dev_ws.current_emb.copyFromHost(host_in);
    const chk = try allocator.alloc(f32, total_elements);
    defer allocator.free(chk);
    try dev_ws.current_emb.copyToHost(chk);
    for (0..num_tokens) |t| {
        std.debug.print("host_in on GPU Token {d}: {any}\n", .{ t, chk[t * ModelConfig.embedding_dim .. t * ModelConfig.embedding_dim + 5] });
    }
    try runLayerCuda(&dev_ws, &layer0, prompt_len, num_tokens, null, null);
    try dev_ws.stream.synchronize();

    const gpu_out = try allocator.alloc(f32, total_elements);
    defer allocator.free(gpu_out);
    try dev_ws.current_emb.copyToHost(gpu_out);

    // 3. Assert parity
    for (0..num_tokens) |t| {
        var t_max: f32 = 0.0;
        const c_t = cpu_out[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
        const g_t = gpu_out[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
        for (c_t, g_t) |c, g| {
            const diff = @abs(c - g);
            if (diff > t_max) t_max = diff;
        }
        std.debug.print("[parity] Token {d}: max_diff = {e:.4}\n", .{ t, t_max });
    }
    var max_diff: f32 = 0.0;
    var max_diff_idx: usize = 0;
    for (cpu_out, gpu_out, 0..) |c, g, i| {
        const diff = @abs(c - g);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    std.debug.print("[parity] Single-Layer GPU Forward Pass (Layer 0): max_diff = {e:.4} (cpu={d:.5}, gpu={d:.5} at idx {d})\n", .{
        max_diff, cpu_out[max_diff_idx], gpu_out[max_diff_idx], max_diff_idx,
    });
    try std.testing.expect(max_diff <= 1e-3);
}

test "runLayerCuda single-layer forward pass on local model layer 5 (global attention)" {
    if (!is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var cpu_model = DiffusionGemmaModel.loadFromFile(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return error.SkipZigTest;
        return err;
    };
    defer cpu_model.deinit();

    const topo = cuda.getTopology() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (topo.reported_devices == 0) return error.SkipZigTest;
    const dev0 = topo.devices[0].device_id;

    const num_tokens: usize = 4;
    const prompt_len: usize = 2;
    const total_elements = num_tokens * ModelConfig.embedding_dim;

    // Load Layer 5 onto Device 0
    var layer5 = try CudaLayerWeights.initFromModelLayer(cpu_model, 5, dev0);
    defer layer5.deinit();

    var dev_ws = try CudaDeviceWorkspace.init(allocator, dev0, num_tokens);
    defer dev_ws.deinit();

    var rope_freqs_dev: ?DeviceBuffer(f32) = null;
    var rf_floats: ?[*]const f32 = null;
    if (cpu_model.rope_freqs) |rf| {
        const rf_slice = getExactTensorSlice(cpu_model, rf);
        const num_floats = rf_slice.len / @sizeOf(f32);
        rf_floats = @ptrCast(@alignCast(rf_slice.ptr));
        var rf_buf = try DeviceBuffer(f32).init(dev0, num_floats);
        try rf_buf.copyFromHost(rf_floats.?[0..num_floats]);
        rope_freqs_dev = rf_buf;
    }
    defer if (rope_freqs_dev) |*rf| rf.deinit();

    // Prepare test input activations
    const host_in = try allocator.alloc(f32, total_elements);
    defer allocator.free(host_in);
    for (host_in, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i + 1)) * 0.01) * 0.1;
    }

    // 1. Run CPU layer forward pass
    const cpu_in = try allocator.alloc(f32, total_elements);
    defer allocator.free(cpu_in);
    @memcpy(cpu_in, host_in);

    const cpu_out = try allocator.alloc(f32, total_elements);
    defer allocator.free(cpu_out);
    @memset(cpu_out, 0.0);

    const cpu_norm_x = try allocator.alloc(f32, total_elements);
    defer allocator.free(cpu_norm_x);
    const cpu_q = try allocator.alloc(f32, num_tokens * infer.max_q_dim);
    defer allocator.free(cpu_q);
    const cpu_k = try allocator.alloc(f32, num_tokens * infer.max_k_dim);
    defer allocator.free(cpu_k);
    const cpu_v = try allocator.alloc(f32, num_tokens * infer.max_k_dim);
    defer allocator.free(cpu_v);
    const cpu_attn_out = try allocator.alloc(f32, total_elements);
    defer allocator.free(cpu_attn_out);

    const fpw = infer.WorkerScratch.floatsPerWorker(num_tokens);
    const worker_storage = try allocator.alloc(f32, fpw);
    defer allocator.free(worker_storage);
    var worker_scratch = infer.WorkerScratch.sliceFrom(worker_storage, num_tokens);

    var cpu_ctx = infer.LayerContext{
        .model = cpu_model,
        .layer_idx = 5,
        .prompt_len = prompt_len,
        .total_len = num_tokens,
        .max_q_dim = infer.max_q_dim,
        .max_k_dim = infer.max_k_dim,
        .rope_freqs = rf_floats,
        .current_emb = cpu_in,
        .next_emb = cpu_out,
        .norm_x_buf = cpu_norm_x,
        .q_buf = cpu_q,
        .k_buf = cpu_k,
        .v_buf = cpu_v,
        .attn_out_buf = cpu_attn_out,
    };

    infer.runQKVWorker(&cpu_ctx, 0, num_tokens);
    infer.runAttnWorker(&cpu_ctx, &worker_scratch, 0, num_tokens);
    infer.runFFNWorker(&cpu_ctx, &worker_scratch, 0, num_tokens);

    // 2. Run GPU layer forward pass
    try dev_ws.current_emb.copyFromHost(host_in);
    const chk5 = try allocator.alloc(f32, total_elements);
    defer allocator.free(chk5);
    try dev_ws.current_emb.copyToHost(chk5);
    for (0..num_tokens) |t| {
        std.debug.print("host_in on GPU L5 Token {d}: {any}\n", .{ t, chk5[t * ModelConfig.embedding_dim .. t * ModelConfig.embedding_dim + 5] });
    }
    const rf_slice = if (rope_freqs_dev) |*rf| rf.slice() else null;
    try runLayerCuda(&dev_ws, &layer5, prompt_len, num_tokens, rf_slice, null);
    try dev_ws.stream.synchronize();

    const gpu_out = try allocator.alloc(f32, total_elements);
    defer allocator.free(gpu_out);
    try dev_ws.current_emb.copyToHost(gpu_out);



    for (0..num_tokens) |t| {
        var t_max: f32 = 0.0;
        const c_t = cpu_out[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
        const g_t = gpu_out[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
        for (c_t, g_t) |c, g| {
            const diff = @abs(c - g);
            if (diff > t_max) t_max = diff;
        }
        std.debug.print("[parity] Layer 5 Token {d}: max_diff = {e:.4}\n", .{ t, t_max });
    }
    var max_diff: f32 = 0.0;
    var max_diff_idx: usize = 0;
    for (cpu_out, gpu_out, 0..) |c, g, i| {
        const diff = @abs(c - g);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    std.debug.print("[parity] Single-Layer GPU Forward Pass (Layer 5): max_diff = {e:.4} (cpu={d:.5}, gpu={d:.5} at idx {d})\n", .{
        max_diff, cpu_out[max_diff_idx], gpu_out[max_diff_idx], max_diff_idx,
    });
    try std.testing.expect(max_diff <= 1e-3);
}

test "executeStructuredReadCuda full 30-layer forward pass on local model if present" {
    if (!root.shouldRunModelTests()) return error.SkipZigTest;
    if (!is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var cpu_model = DiffusionGemmaModel.loadFromFile(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return error.SkipZigTest;
        return err;
    };
    defer cpu_model.deinit();

    std.debug.print("test: Initializing CudaModel...\n", .{});
    var cuda_model = try CudaModel.initFromCpuModel(allocator, cpu_model, false);
    defer cuda_model.deinit();
    std.debug.print("test: CudaModel initialized! Topology: {}\n", .{cuda_model.topology});

    const max_len: usize = 32;
    std.debug.print("test: Initializing CudaWorkspace...\n", .{});
    var ws = try CudaWorkspace.init(allocator, cuda_model.topology, cuda_model.devices, max_len);
    defer ws.deinit();
    std.debug.print("test: CudaWorkspace initialized!\n", .{});

    const prompt_tokens = [_]u32{ root.SpecialTokens.bos, 100, 200 };
    const canvas = try Canvas.seedSingleStep(&prompt_tokens, root.SpecialTokens.mask);
    const canvas_len: usize = 16;
    const candidate_tokens = [_]u32{ 10784, 3771 }; // Yes, No

    infer.dump_layers = true;
    defer infer.dump_layers = false;

    var cuda_raw: [2]f32 = undefined;
    var cuda_softcapped: [2]f32 = undefined;

    std.debug.print("test: Starting executeStructuredReadCuda...\n", .{});
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        null,
        &cuda_raw,
        &cuda_softcapped,
    );
    std.debug.print("test: executeStructuredReadCuda completed successfully!\n", .{});

    // Run CPU reference for comparison
    var cpu_ws = try infer.InferenceWorkspace.init(allocator, max_len, 4);
    defer cpu_ws.deinit();

    var cpu_raw: [2]f32 = undefined;
    var cpu_softcapped: [2]f32 = undefined;

    try infer.executeStructuredReadSlice(
        cpu_ws,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        null,
        &cpu_raw,
        &cpu_softcapped,
    );

    for (0..2) |i| {
        const raw_diff = @abs(cpu_raw[i] - cuda_raw[i]);
        const soft_diff = @abs(cpu_softcapped[i] - cuda_softcapped[i]);
        std.debug.print("[parity] Logit {d}: CPU={d:.4}, GPU={d:.4}, raw_diff={d:.4}, soft_diff={d:.4}\n", .{
            i, cpu_raw[i], cuda_raw[i], raw_diff, soft_diff,
        });
    }

    const cpu_pred = (cpu_raw[0] > cpu_raw[1]);
    const gpu_pred = (cuda_raw[0] > cuda_raw[1]);
    std.debug.print("[parity] Discrete Decision: CPU={s}, GPU={s}, Match={}\n", .{
        if (cpu_pred) "Yes" else "No",
        if (gpu_pred) "Yes" else "No",
        cpu_pred == gpu_pred,
    });
    try std.testing.expect(cpu_pred == gpu_pred);

    for (0..2) |i| {
        const raw_diff = @abs(cpu_raw[i] - cuda_raw[i]);
        const soft_diff = @abs(cpu_softcapped[i] - cuda_softcapped[i]);
        try std.testing.expect(raw_diff <= 0.05);
        try std.testing.expect(soft_diff <= 0.05);
    }

    // R24-05: Exercise injected graph capture failure fallback in executeStructuredReadCuda
    inject_graph_capture_failure = true;
    defer inject_graph_capture_failure = false;
    var cap_fail_raw: [2]f32 = undefined;
    var cap_fail_soft: [2]f32 = undefined;
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        null,
        &cap_fail_raw,
        &cap_fail_soft,
    );
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(cuda_raw[i], cap_fail_raw[i], 1e-4);
    }

    // R24-05: Exercise injected cached replay failure fallback in executeStructuredReadCuda
    inject_graph_capture_failure = false;
    // Establish fresh valid cached graph
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        null,
        &cap_fail_raw,
        &cap_fail_soft,
    );
    // Inject replay failure
    inject_graph_replay_failure = true;
    defer inject_graph_replay_failure = false;
    var rep_fail_raw: [2]f32 = undefined;
    var rep_fail_soft: [2]f32 = undefined;
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        null,
        &rep_fail_raw,
        &rep_fail_soft,
    );
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(cuda_raw[i], rep_fail_raw[i], 1e-4);
    }

    // R40-01: Exercise injected Device 1 graph replay failure after transferHiddenState is enqueued
    inject_graph_replay_failure = false;
    // Re-establish valid cached graphs on both devices
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        null,
        &cap_fail_raw,
        &cap_fail_soft,
    );
    try std.testing.expect(ws.graph0 != null);
    if (cuda_model.topology == .dual_gpu_split) {
        try std.testing.expect(ws.graph1 != null);
    }
    // Inject failure on Device 1 replay specifically
    inject_graph1_replay_failure = true;
    defer inject_graph1_replay_failure = false;
    var g1_rep_fail_raw: [2]f32 = undefined;
    var g1_rep_fail_soft: [2]f32 = undefined;
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        null,
        &g1_rep_fail_raw,
        &g1_rep_fail_soft,
    );
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(cuda_raw[i], g1_rep_fail_raw[i], 1e-4);
        try std.testing.expectApproxEqAbs(cuda_softcapped[i], g1_rep_fail_soft[i], 1e-4);
    }
    inject_graph1_replay_failure = false;

    // R41-01: Exercise injected Device 1 graph replay failure combined with stream synchronization failure
    // Re-establish valid cached graphs on both devices
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        null,
        &cap_fail_raw,
        &cap_fail_soft,
    );
    try std.testing.expect(ws.graph0 != null);
    if (cuda_model.topology == .dual_gpu_split) {
        try std.testing.expect(ws.graph1 != null);
        inject_graph1_replay_failure = true;
        inject_stream_sync_failure = true;
        defer {
            inject_graph1_replay_failure = false;
            inject_stream_sync_failure = false;
        }
        var sync_fail_raw: [2]f32 = undefined;
        var sync_fail_soft: [2]f32 = undefined;
        const res = executeStructuredReadCuda(
            ws,
            cuda_model,
            cpu_model,
            &prompt_tokens,
            canvas.tokens[0..canvas_len],
            canvas.answer_pos,
            &candidate_tokens,
            null,
            &sync_fail_raw,
            &sync_fail_soft,
        );
        try std.testing.expectError(error.StreamSyncFailed, res);
        inject_graph1_replay_failure = false;
        inject_stream_sync_failure = false;
    }

    // Phase 7B Parity Test: Compare cached structured read against uncached full forward pass
    enable_prompt_cache = false;
    defer enable_prompt_cache = true;
    var uncached_raw: [2]f32 = undefined;
    var uncached_soft: [2]f32 = undefined;
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        null,
        &uncached_raw,
        &uncached_soft,
    );
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(cuda_raw[i], uncached_raw[i], 1e-4);
        try std.testing.expectApproxEqAbs(cuda_softcapped[i], uncached_soft[i], 1e-4);
    }

    // Phase 19: Exercise Step 2 CUDA Graph Capture and Replay with Self-Conditioning
    enable_prompt_cache = true;
    const candidate_probs = [_]f32{ 0.75, 0.25 };
    var step2_cuda_raw: [2]f32 = undefined;
    var step2_cuda_soft: [2]f32 = undefined;

    // Step 2 Run 1: Graph Capture
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        &candidate_probs,
        &step2_cuda_raw,
        &step2_cuda_soft,
    );
    try std.testing.expect(ws.graph0_step2 != null);

    // Step 2 Run 2: Graph Replay
    var step2_replay_raw: [2]f32 = undefined;
    var step2_replay_soft: [2]f32 = undefined;
    try executeStructuredReadCuda(
        ws,
        cuda_model,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        &candidate_probs,
        &step2_replay_raw,
        &step2_replay_soft,
    );
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(step2_cuda_raw[i], step2_replay_raw[i], 1e-4);
        try std.testing.expectApproxEqAbs(step2_cuda_soft[i], step2_replay_soft[i], 1e-4);
    }

    // Compare with CPU reference for Step 2
    var step2_cpu_raw: [2]f32 = undefined;
    var step2_cpu_soft: [2]f32 = undefined;
    try infer.executeStructuredReadSlice(
        cpu_ws,
        cpu_model,
        &prompt_tokens,
        canvas.tokens[0..canvas_len],
        canvas.answer_pos,
        &candidate_tokens,
        &candidate_probs,
        &step2_cpu_raw,
        &step2_cpu_soft,
    );
    for (0..2) |i| {
        const raw_diff = @abs(step2_cpu_raw[i] - step2_cuda_raw[i]);
        const soft_diff = @abs(step2_cpu_soft[i] - step2_cuda_soft[i]);
        try std.testing.expect(raw_diff <= 0.05);
        try std.testing.expect(soft_diff <= 0.05);
    }
}

test "CudaDeviceWorkspace: activationWorkspaceBytes matches audited footprint (R24-01, R26-01, R26-04)" {
    const bytes_512 = CudaDeviceWorkspace.activationWorkspaceBytes(512);
    // Audited exact byte sum for N=512: 381,932,800 bytes (~381.93 MB, 364.24 MiB)
    // Includes SWA-sized prompt KV prefix cache (k_dim=2048), Phase 9 self-conditioning scratch buffers,
    // and Phase 19 device-resident canvas initialization backup buffers (canvas_init_emb: 5.77 MB, canvas_raw_ans: 11.26 KB).
    // In Phase 10 WP2 & Phase 15 WP2, dead scratch buffers exp_gate_up_buf (23.07 MB), h2 (5.77 MB),
    // ffn_comb (5.77 MB), and post_ffn (5.77 MB) were eliminated via kernel fusions (-40.37 MB total).
    try std.testing.expectEqual(@as(usize, 381932800), bytes_512);
}

test "CudaDeviceWorkspace: prompt KV cache stride isolates SWA layers with long prompt (R26-01)" {
    if (!is_enabled) return error.SkipZigTest;
    const count = cuda.getDeviceCount() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (count == 0) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var ws = try CudaDeviceWorkspace.init(allocator, 0, 512);
    defer ws.deinit();

    // Exercise prompt length > 256 tokens with SWA k_dim = 2048 (e.g. 270 tokens)
    const prompt_len: usize = 270;
    const swa_k_dim: usize = 2048;
    const floats_per_layer = prompt_len * swa_k_dim; // 552,960 floats

    const host_pattern = try allocator.alloc(f32, floats_per_layer);
    defer allocator.free(host_pattern);

    // Fill layer 0 with pattern 1.0f
    @memset(host_pattern, 1.0);
    const l0_offset = 0 * CudaDeviceWorkspace.cache_floats_per_layer;
    const l0_k = try ws.prompt_k_cache.slice().subslice(l0_offset, floats_per_layer);
    try l0_k.copyFromHost(host_pattern);

    // Fill layer 1 with pattern 2.0f
    @memset(host_pattern, 2.0);
    const l1_offset = 1 * CudaDeviceWorkspace.cache_floats_per_layer;
    const l1_k = try ws.prompt_k_cache.slice().subslice(l1_offset, floats_per_layer);
    try l1_k.copyFromHost(host_pattern);

    // Verify layer 0 remained completely untouched (no overlap or corruption from layer 1 write)
    const host_check_l0 = try allocator.alloc(f32, floats_per_layer);
    defer allocator.free(host_check_l0);
    try l0_k.copyToHost(host_check_l0);

    for (host_check_l0) |val| {
        try std.testing.expectEqual(@as(f32, 1.0), val);
    }

    // Verify layer 1 is completely intact
    const host_check_l1 = try allocator.alloc(f32, floats_per_layer);
    defer allocator.free(host_check_l1);
    try l1_k.copyToHost(host_check_l1);

    for (host_check_l1) |val| {
        try std.testing.expectEqual(@as(f32, 2.0), val);
    }
}

test "R40-01: CudaWorkspace independent stream synchronization and safe pinned memory deinit" {
    if (!is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const topo = cuda.getTopology() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (topo.reported_devices < 2) return error.SkipZigTest;

    const devices = [2]i32{ topo.devices[0].device_id, topo.devices[1].device_id };
    const ws = try CudaWorkspace.init(allocator, .dual_gpu_split, devices, 256);

    // Verify independent synchronization on both streams succeeds
    try ws.synchronize();

    // Enqueue an async boundary transfer between Dev 0 and Dev 1 into pinned memory
    const count: usize = 2816;
    var buf_src0 = try DeviceBuffer(f32).init(devices[0], count);
    defer buf_src0.deinit();
    var buf_dst1 = try DeviceBuffer(f32).init(devices[1], count);
    defer buf_dst1.deinit();

    const ev = ws.boundary_event orelse return error.InvalidParam;
    try transferHiddenState(buf_src0.slice(), buf_dst1.slice(), &ws.dev0.stream, &ws.dev1.?.stream, &ev, ws.host_staging_buf, ws.peer_access_enabled);

    // Verify that workspace deinit successfully drains active transfers on both streams without memory corruption
    ws.deinit();
}

test "R41-01: CudaWorkspace quarantine policy and failed stream synchronization" {
    if (!is_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const topo = cuda.getTopology() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (topo.reported_devices < 2) return error.SkipZigTest;

    const devices = [2]i32{ topo.devices[0].device_id, topo.devices[1].device_id };

    // Test 1: Injected Device 0 sync failure causes synchronize() to fail
    {
        const ws = try CudaWorkspace.init(allocator, .dual_gpu_split, devices, 256);
        inject_dev0_sync_failure = true;
        defer inject_dev0_sync_failure = false;
        try std.testing.expectError(error.StreamSyncFailed, ws.synchronize());
        inject_dev0_sync_failure = false;
        ws.deinit();
    }

    // Test 2: R41-01 / R42-01 / R42-02: Active transfer before sync failure, quarantine retention on quiescence failure, and slot reuse
    {
        const ws = try CudaWorkspace.init(allocator, .dual_gpu_split, devices, 256);
        const prev_quarantine_count = quarantined_buffers_count;

        // Enqueue an active async transfer using ws.host_staging_pinned so it is actively referenced by DMA
        const count: usize = 2816;
        var buf_src0 = try DeviceBuffer(f32).init(devices[0], count);
        defer buf_src0.deinit();
        var buf_dst1 = try DeviceBuffer(f32).init(devices[1], count);
        defer buf_dst1.deinit();
        const ev = ws.boundary_event orelse return error.InvalidParam;
        try transferHiddenState(buf_src0.slice(), buf_dst1.slice(), &ws.dev0.stream, &ws.dev1.?.stream, &ev, ws.host_staging_buf, ws.peer_access_enabled);

        // Inject Device 1 synchronization failure during deinit: triggers quarantine policy
        inject_dev1_sync_failure = true;
        defer inject_dev1_sync_failure = false;
        ws.deinit();
        inject_dev1_sync_failure = false;

        try std.testing.expectEqual(prev_quarantine_count + 1, quarantined_buffers_count);
        try std.testing.expect(quarantined_buffers[0] != null);

        // R42-01: When device quiescence cannot be established, drainQuarantinedPinnedBuffers must abort and retain slots
        inject_device_sync_failure = true;
        try std.testing.expectError(error.DeviceNotQuiescent, drainQuarantinedPinnedBuffers());
        try std.testing.expectEqual(prev_quarantine_count + 1, quarantined_buffers_count);
        try std.testing.expect(quarantined_buffers[0] != null);
        inject_device_sync_failure = false;

        // When device quiescence is confirmed, drainQuarantinedPinnedBuffers releases memory and clears slot
        try drainQuarantinedPinnedBuffers();
        try std.testing.expectEqual(prev_quarantine_count, quarantined_buffers_count);
        try std.testing.expect(quarantined_buffers[0] == null);

        // R42-02: Verify that slot 0 is reused on subsequent quarantine
        const ws2 = try CudaWorkspace.init(allocator, .dual_gpu_split, devices, 256);
        inject_dev1_sync_failure = true;
        ws2.deinit();
        inject_dev1_sync_failure = false;

        try std.testing.expectEqual(prev_quarantine_count + 1, quarantined_buffers_count);
        try std.testing.expect(quarantined_buffers[0] != null); // Reused slot 0!
        try drainQuarantinedPinnedBuffers();
        try std.testing.expectEqual(prev_quarantine_count, quarantined_buffers_count);
    }

    // Test 3: R43-01: Quarantine capacity overflow handling and automatic reclamation
    {
        try drainQuarantinedPinnedBuffers();
        try std.testing.expectEqual(@as(usize, 0), quarantined_buffers_count);

        // Fill all 16 slots
        var mock_buffers: [MAX_QUARANTINED_BUFFERS]cuda_mem.PinnedBuffer(f32) = undefined;
        for (0..MAX_QUARANTINED_BUFFERS) |i| {
            mock_buffers[i] = try cuda_mem.PinnedBuffer(f32).init(64);
            quarantinePinnedBuffer(&mock_buffers[i]);
            try std.testing.expect(mock_buffers[i].ptr == null); // Pointer was moved into quarantine
        }
        try std.testing.expectEqual(@as(usize, MAX_QUARANTINED_BUFFERS), quarantined_buffers_count);

        // Allocate a 17th pinned buffer: quarantine table is full
        var buf17 = try cuda_mem.PinnedBuffer(f32).init(64);
        // With devices quiescent, quarantinePinnedBuffer attempts opportunistic drain first or forces safe teardown
        quarantinePinnedBuffer(&buf17);
        try std.testing.expect(buf17.ptr == null); // buf17 safely handled without memory leak

        // Final cleanup
        try drainQuarantinedPinnedBuffers();
        try std.testing.expectEqual(@as(usize, 0), quarantined_buffers_count);
    }

    // Test 4: R44-04: Quarantine overflow with failed emergency sync moves to durable overflow quarantine
    {
        try drainQuarantinedPinnedBuffers();
        try std.testing.expectEqual(@as(usize, 0), quarantined_buffers_count);
        try std.testing.expectEqual(@as(usize, 0), overflow_quarantined_buffers_count);

        // Fill all 16 slots with sync failure so opportunistic drain is skipped
        inject_device_sync_failure = true;

        var mock_buffers: [MAX_QUARANTINED_BUFFERS]cuda_mem.PinnedBuffer(f32) = undefined;
        for (0..MAX_QUARANTINED_BUFFERS) |i| {
            mock_buffers[i] = try cuda_mem.PinnedBuffer(f32).init(64);
            quarantinePinnedBuffer(&mock_buffers[i]);
            try std.testing.expect(mock_buffers[i].ptr == null);
        }
        try std.testing.expectEqual(@as(usize, MAX_QUARANTINED_BUFFERS), quarantined_buffers_count);

        // Allocate a 17th pinned buffer: primary quarantine full, device sync fails
        var buf17 = try cuda_mem.PinnedBuffer(f32).init(64);
        quarantinePinnedBuffer(&buf17);
        // buf17 should be moved into overflow_quarantined_buffers
        try std.testing.expect(buf17.ptr == null);
        try std.testing.expectEqual(@as(usize, 1), overflow_quarantined_buffers_count);

        // Now restore device sync and drain all
        inject_device_sync_failure = false;
        try drainQuarantinedPinnedBuffers();
        try std.testing.expectEqual(@as(usize, 0), quarantined_buffers_count);
        try std.testing.expectEqual(@as(usize, 0), overflow_quarantined_buffers_count);
    }

    // Test 5: R45-03: Primary (16) and overflow (64) quarantine saturation with emergency sync failure
    // transfers to dynamic overflow quarantine, surviving workspace teardown
    {
        try drainQuarantinedPinnedBuffers();
        try std.testing.expectEqual(@as(usize, 0), quarantined_buffers_count);
        try std.testing.expectEqual(@as(usize, 0), overflow_quarantined_buffers_count);
        try std.testing.expectEqual(@as(usize, 0), dynamic_quarantined_count);

        inject_device_sync_failure = true;

        // 1. Fill all 16 primary slots
        var primary_mock_buffers: [MAX_QUARANTINED_BUFFERS]cuda_mem.PinnedBuffer(f32) = undefined;
        for (0..MAX_QUARANTINED_BUFFERS) |i| {
            primary_mock_buffers[i] = try cuda_mem.PinnedBuffer(f32).init(64);
            quarantinePinnedBuffer(&primary_mock_buffers[i]);
            try std.testing.expect(primary_mock_buffers[i].ptr == null);
        }
        try std.testing.expectEqual(@as(usize, MAX_QUARANTINED_BUFFERS), quarantined_buffers_count);

        // 2. Fill all 64 overflow slots
        var overflow_mock_buffers: [MAX_OVERFLOW_BUFFERS]cuda_mem.PinnedBuffer(f32) = undefined;
        for (0..MAX_OVERFLOW_BUFFERS) |i| {
            overflow_mock_buffers[i] = try cuda_mem.PinnedBuffer(f32).init(64);
            quarantinePinnedBuffer(&overflow_mock_buffers[i]);
            try std.testing.expect(overflow_mock_buffers[i].ptr == null);
        }
        try std.testing.expectEqual(@as(usize, MAX_OVERFLOW_BUFFERS), overflow_quarantined_buffers_count);

        // 3. Both fixed tables (16 + 64 = 80 buffers) are now completely saturated.
        // Allocate an 81st pinned buffer: must be accepted by dynamic overflow quarantine
        var buf81 = try cuda_mem.PinnedBuffer(f32).init(64);
        const original_ptr = buf81.ptr.?;
        quarantinePinnedBuffer(&buf81);

        // Assert buf81 ownership was transferred (ptr set to null) into dynamic quarantine
        try std.testing.expect(buf81.ptr == null);
        try std.testing.expectEqual(@as(usize, 1), dynamic_quarantined_count);
        try std.testing.expect(dynamic_quarantined_head != null);
        try std.testing.expectEqual(original_ptr, dynamic_quarantined_head.?.ptr);

        // 4. Verify that when CudaWorkspace deinitialization encounters failed stream sync,
        // its host_staging_pinned buffer is also accepted into dynamic quarantine and survives
        // workspace destruction without leaking or losing ownership tracking.
        {
            var ws = try CudaWorkspace.init(std.testing.allocator, .single_gpu, [_]i32{ 0, 0 }, 32);
            const ws_pinned_ptr = ws.host_staging_pinned.ptr.?;

            // Inject stream synchronization failure during deinit
            inject_stream_sync_failure = true;
            ws.deinit();
            inject_stream_sync_failure = false;

            // Workspace is destroyed; verify its pinned allocation was retained in dynamic quarantine
            try std.testing.expectEqual(@as(usize, 2), dynamic_quarantined_count);
            try std.testing.expect(dynamic_quarantined_head != null);
            try std.testing.expectEqual(ws_pinned_ptr, dynamic_quarantined_head.?.ptr);
        }

        // 5. Restore device synchronization and drain all 82 quarantined buffers cleanly
        inject_device_sync_failure = false;
        try drainQuarantinedPinnedBuffers();
        try std.testing.expectEqual(@as(usize, 0), quarantined_buffers_count);
        try std.testing.expectEqual(@as(usize, 0), overflow_quarantined_buffers_count);
        try std.testing.expectEqual(@as(usize, 0), dynamic_quarantined_count);
        try std.testing.expect(dynamic_quarantined_head == null);
    }
}


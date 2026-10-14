//! Zeno — DiffusionGemma Model representation and tensor mapping.
//!
//! Maps GGUF tensors to transformer layers (SWA vs Global), embeddings, and norms
//! using zero-copy memory mapping.

const std = @import("std");
const posix = std.posix;
const linux = std.os.linux;
const gguf = @import("gguf.zig");
const quant = @import("quant.zig");
const root = @import("root.zig");

pub const ModelConfig = root.ModelConfig;
pub const TensorInfo = gguf.TensorInfo;

/// Represents a single transformer block in DiffusionGemma.
pub const TransformerLayer = struct {
    layer_index: usize,
    is_global: bool,

    // Attention sub-layer
    attn_norm: TensorInfo,
    q_norm: TensorInfo,
    k_norm: TensorInfo,
    q_weight: TensorInfo,
    k_weight: TensorInfo,
    v_weight: ?TensorInfo, // Null for global layers (layers 5, 11, 17, 23, 29) where V = K weights
    o_weight: TensorInfo,
    post_attention_norm: TensorInfo,

    // Shared dense FFN sub-layer
    ffn_norm: TensorInfo,
    ffn_gate: TensorInfo,
    ffn_up: TensorInfo,
    ffn_down: TensorInfo,
    post_ffw_norm_1: TensorInfo,

    // MoE routed sub-layer
    pre_ffw_norm_2: TensorInfo,
    router_gate: TensorInfo,
    router_scale: TensorInfo,
    expert_gate_up: TensorInfo,
    expert_down: TensorInfo,
    expert_down_scale: TensorInfo,
    post_ffw_norm_2: TensorInfo,

    // Block-level output norm and scale
    post_ffw_norm: TensorInfo,
    layer_output_scale: TensorInfo,
    enc_layer_output_scale: TensorInfo,
};

pub const ModelLoadError = error{
    InvalidArchitecture,
    IncompatibleDimensions,
    MissingTensor,
    FileNotFound,
    MmapFailed,
};

pub const DiffusionGemmaModel = struct {
    allocator: std.mem.Allocator,
    gguf_file: gguf.GgufFile,
    mmap_data: []align(std.heap.page_size_min) const u8,
    file_fd: ?posix.fd_t,

    // Global model tensors
    token_embd: TensorInfo,
    output_norm: TensorInfo,
    rope_freqs: ?TensorInfo,

    // Self-conditioning tensors (Phase 9)
    self_cond_pre_norm: ?TensorInfo,
    self_cond_down: ?TensorInfo,
    self_cond_gate: ?TensorInfo,
    self_cond_up: ?TensorInfo,

    // 30 transformer layers
    layers: [ModelConfig.num_layers]TransformerLayer,

    /// Loads and memory-maps a DiffusionGemma model from a GGUF file path.
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !DiffusionGemmaModel {
        const fd = posix.openat(posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0) catch {
            return error.FileNotFound;
        };
        errdefer _ = linux.close(fd);

        const file_size: usize = @intCast(linux.lseek(fd, 0, linux.SEEK.END));
        _ = linux.lseek(fd, 0, linux.SEEK.SET);

        if (file_size < 1024) {
            return error.MmapFailed;
        }

        const mapped = posix.mmap(null, file_size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0) catch {
            return error.MmapFailed;
        };

        var model = try loadFromMemory(allocator, mapped);
        model.file_fd = fd;
        return model;
    }

    /// Loads model structure from an in-memory mapped GGUF byte buffer.
    pub fn loadFromMemory(allocator: std.mem.Allocator, mapped_data: []align(std.heap.page_size_min) const u8) !DiffusionGemmaModel {
        var gfile = try gguf.parseGguf(allocator, mapped_data);
        errdefer gfile.deinit();

        // Validate architecture
        const arch = gfile.getMetadataString("general.architecture") orelse return error.InvalidArchitecture;
        if (!std.mem.eql(u8, arch, "diffusion-gemma")) {
            return error.InvalidArchitecture;
        }

        // Validate block count
        const block_count = gfile.getMetadataU32("diffusion-gemma.block_count") orelse return error.IncompatibleDimensions;
        if (block_count != ModelConfig.num_layers) {
            return error.IncompatibleDimensions;
        }

        // Validate embedding length
        const emb_len = gfile.getMetadataU32("diffusion-gemma.embedding_length") orelse return error.IncompatibleDimensions;
        if (emb_len != ModelConfig.embedding_dim) {
            return error.IncompatibleDimensions;
        }

        // Validate non-causal bidirectional attention invariant
        const is_causal = gfile.getMetadataBool("diffusion-gemma.attention.causal") orelse return error.IncompatibleDimensions;
        if (is_causal) {
            return error.IncompatibleDimensions;
        }

        // Locate global tensors
        const token_embd = gfile.findTensor("token_embd.weight") orelse return error.MissingTensor;
        const output_norm = gfile.findTensor("output_norm.weight") orelse return error.MissingTensor;
        const rope_freqs = gfile.findTensor("rope_freqs.weight");

        // Locate self-conditioning tensors (Phase 9)
        const sc_pre_norm = gfile.findTensor("self_cond_pre_norm.weight");
        const sc_down = gfile.findTensor("self_cond_down.weight");
        const sc_gate = gfile.findTensor("self_cond_gate.weight");
        const sc_up = gfile.findTensor("self_cond_up.weight");

        // Map all 30 layers
        var layers: [ModelConfig.num_layers]TransformerLayer = undefined;
        var name_buf: [64]u8 = undefined;

        for (0..ModelConfig.num_layers) |l| {
            const is_global = (l == 5 or l == 11 or l == 17 or l == 23 or l == 29);

            const attn_norm_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm.weight", .{l});
            const attn_norm = gfile.findTensor(attn_norm_name) orelse return error.MissingTensor;

            const q_norm_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q_norm.weight", .{l});
            const q_norm = gfile.findTensor(q_norm_name) orelse return error.MissingTensor;

            const k_norm_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k_norm.weight", .{l});
            const k_norm = gfile.findTensor(k_norm_name) orelse return error.MissingTensor;

            const q_weight_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q.weight", .{l});
            const q_weight = gfile.findTensor(q_weight_name) orelse return error.MissingTensor;

            const k_weight_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k.weight", .{l});
            const k_weight = gfile.findTensor(k_weight_name) orelse return error.MissingTensor;

            var v_weight: ?TensorInfo = null;
            if (!is_global) {
                const v_weight_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{l});
                v_weight = gfile.findTensor(v_weight_name) orelse return error.MissingTensor;
            }

            const o_weight_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_output.weight", .{l});
            const o_weight = gfile.findTensor(o_weight_name) orelse return error.MissingTensor;

            const post_attn_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.post_attention_norm.weight", .{l});
            const post_attn = gfile.findTensor(post_attn_name) orelse return error.MissingTensor;

            const ffn_norm_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm.weight", .{l});
            const ffn_norm = gfile.findTensor(ffn_norm_name) orelse return error.MissingTensor;

            const ffn_gate_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate.weight", .{l});
            const ffn_gate = gfile.findTensor(ffn_gate_name) orelse return error.MissingTensor;

            const ffn_up_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up.weight", .{l});
            const ffn_up = gfile.findTensor(ffn_up_name) orelse return error.MissingTensor;

            const ffn_down_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down.weight", .{l});
            const ffn_down = gfile.findTensor(ffn_down_name) orelse return error.MissingTensor;

            const post_ffw_1_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.post_ffw_norm_1.weight", .{l});
            const post_ffw_1 = gfile.findTensor(post_ffw_1_name) orelse return error.MissingTensor;

            const pre_ffw_2_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.pre_ffw_norm_2.weight", .{l});
            const pre_ffw_2 = gfile.findTensor(pre_ffw_2_name) orelse return error.MissingTensor;

            const router_gate_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate_inp.weight", .{l});
            const router_gate = gfile.findTensor(router_gate_name) orelse return error.MissingTensor;

            const router_scale_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate_inp.scale", .{l});
            const router_scale = gfile.findTensor(router_scale_name) orelse return error.MissingTensor;

            const exp_gate_up_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate_up_exps.weight", .{l});
            const exp_gate_up = gfile.findTensor(exp_gate_up_name) orelse return error.MissingTensor;

            const exp_down_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down_exps.weight", .{l});
            const exp_down = gfile.findTensor(exp_down_name) orelse return error.MissingTensor;

            const exp_down_scale_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down_exps.scale", .{l});
            const exp_down_scale = gfile.findTensor(exp_down_scale_name) orelse return error.MissingTensor;

            const post_ffw_2_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.post_ffw_norm_2.weight", .{l});
            const post_ffw_2 = gfile.findTensor(post_ffw_2_name) orelse return error.MissingTensor;

            const post_ffw_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.post_ffw_norm.weight", .{l});
            const post_ffw = gfile.findTensor(post_ffw_name) orelse return error.MissingTensor;

            const layer_out_scale_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.layer_output_scale.weight", .{l});
            const layer_out_scale = gfile.findTensor(layer_out_scale_name) orelse return error.MissingTensor;

            const enc_layer_out_scale_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.enc_layer_output_scale.weight", .{l});
            const enc_layer_out_scale = gfile.findTensor(enc_layer_out_scale_name) orelse return error.MissingTensor;

            layers[l] = .{
                .layer_index = l,
                .is_global = is_global,
                .attn_norm = attn_norm,
                .q_norm = q_norm,
                .k_norm = k_norm,
                .q_weight = q_weight,
                .k_weight = k_weight,
                .v_weight = v_weight,
                .o_weight = o_weight,
                .post_attention_norm = post_attn,
                .ffn_norm = ffn_norm,
                .ffn_gate = ffn_gate,
                .ffn_up = ffn_up,
                .ffn_down = ffn_down,
                .post_ffw_norm_1 = post_ffw_1,
                .pre_ffw_norm_2 = pre_ffw_2,
                .router_gate = router_gate,
                .router_scale = router_scale,
                .expert_gate_up = exp_gate_up,
                .expert_down = exp_down,
                .expert_down_scale = exp_down_scale,
                .post_ffw_norm_2 = post_ffw_2,
                .post_ffw_norm = post_ffw,
                .layer_output_scale = layer_out_scale,
                .enc_layer_output_scale = enc_layer_out_scale,
            };
        }

        return DiffusionGemmaModel{
            .allocator = allocator,
            .gguf_file = gfile,
            .mmap_data = mapped_data,
            .file_fd = null,
            .token_embd = token_embd,
            .output_norm = output_norm,
            .rope_freqs = rope_freqs,
            .self_cond_pre_norm = sc_pre_norm,
            .self_cond_down = sc_down,
            .self_cond_gate = sc_gate,
            .self_cond_up = sc_up,
            .layers = layers,
        };
    }

    /// Releases resources, unmapping file memory and closing file descriptors.
    pub fn deinit(self: *DiffusionGemmaModel) void {
        self.gguf_file.deinit();
        posix.munmap(self.mmap_data);
        if (self.file_fd) |fd| {
            _ = linux.close(fd);
            self.file_fd = null;
        }
    }

    /// Returns the 11 Q6_K quantization blocks for a given token ID.
    /// Each row in token_embd.weight is 2816 elements = 11 blocks of 256 values (2310 bytes).
    pub fn getEmbeddingBlocks(self: DiffusionGemmaModel, token_id: u32) []const quant.BlockQ6_K {
        std.debug.assert(token_id < ModelConfig.vocab_size);
        const blocks_per_row = ModelConfig.embedding_dim / quant.QK_K; // 2816 / 256 = 11
        const bytes_per_row = blocks_per_row * @sizeOf(quant.BlockQ6_K); // 11 * 210 = 2310 bytes

        const tensor_slice = self.gguf_file.getTensorSlice(self.token_embd);
        const row_offset = @as(usize, token_id) * bytes_per_row;
        const row_bytes = tensor_slice[row_offset .. row_offset + bytes_per_row];

        return @as([*]const quant.BlockQ6_K, @ptrCast(@alignCast(row_bytes.ptr)))[0..blocks_per_row];
    }
};

test "DiffusionGemmaModel: verify layer specialization on local GGUF if present" {
    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    // Attempt to open; skip test if local model file is not present
    var model = DiffusionGemmaModel.loadFromFile(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer model.deinit();

    // Verify global layer V=K sharing across all 30 layers
    for (model.layers) |layer| {
        if (layer.is_global) {
            try std.testing.expect(layer.v_weight == null);
        } else {
            try std.testing.expect(layer.v_weight != null);
        }
    }

    // Verify token embedding lookup for BOS (2) and MASK (4)
    const bos_blocks = model.getEmbeddingBlocks(root.SpecialTokens.bos);
    try std.testing.expectEqual(@as(usize, 11), bos_blocks.len);

    const mask_blocks = model.getEmbeddingBlocks(root.SpecialTokens.mask);
    try std.testing.expectEqual(@as(usize, 11), mask_blocks.len);
}

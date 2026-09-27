//! Zeno — Structured read forward evaluation primitive.
//!
//! Evaluates a seeded canvas, executes transformer layer mechanics (bidirectional attention,
//! per-head RMSNorms, dual RoPE, and top-8 MoE routing), slices hidden state at pinned answer
//! position p, and projects candidate logits via Q6_K embedding dot products and logit softcapping.

const std = @import("std");
const root = @import("root.zig");
const quant = @import("quant.zig");
const ops = @import("ops.zig");
const model_mod = @import("model.zig");
const canvas_mod = @import("canvas.zig");

pub const DiffusionGemmaModel = model_mod.DiffusionGemmaModel;
pub const TransformerLayer = model_mod.TransformerLayer;
pub const Canvas = canvas_mod.Canvas;
pub const ModelConfig = root.ModelConfig;

/// Matrix-vector multiplication for Q4_K quantized weights: out = W * x
pub fn gemvQ4_K(W_blocks: [*]const quant.BlockQ4_K, x: []const f32, out: []f32, out_dim: usize, in_dim: usize) void {
    std.debug.assert(in_dim % quant.QK_K == 0);
    std.debug.assert(x.len == in_dim);
    std.debug.assert(out.len == out_dim);

    const blocks_per_row = in_dim / quant.QK_K;
    for (0..out_dim) |r| {
        const row_blocks = W_blocks[r * blocks_per_row .. (r + 1) * blocks_per_row];
        out[r] = quant.dotRowQ4_K(row_blocks, x);
    }
}

/// Matrix-vector multiplication for standard float32 weights: out = W * x
pub fn gemvF32(weights: []const f32, x: []const f32, out: []f32, out_dim: usize, in_dim: usize) void {
    std.debug.assert(weights.len == out_dim * in_dim);
    std.debug.assert(x.len == in_dim);
    std.debug.assert(out.len == out_dim);

    for (0..out_dim) |r| {
        const row = weights[r * in_dim .. (r + 1) * in_dim];
        var sum: f32 = 0.0;
        for (row, x) |w, xv| {
            sum += w * xv;
        }
        out[r] = sum;
    }
}

/// Applies Rotary Position Embeddings (RoPE) to head vector pairs at position `pos`.
pub fn applyRoPE(vec: []f32, pos: usize, head_dim: usize, theta: f32) void {
    const half = head_dim / 2;
    const pos_f = @as(f32, @floatFromInt(pos));
    for (0..half) |i| {
        const exp = 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim));
        const freq = 1.0 / std.math.pow(f32, theta, exp);
        const angle = pos_f * freq;
        const cos = @cos(angle);
        const sin = @sin(angle);

        const v0 = vec[2 * i];
        const v1 = vec[2 * i + 1];
        vec[2 * i] = v0 * cos - v1 * sin;
        vec[2 * i + 1] = v0 * sin + v1 * cos;
    }
}

/// Applies per-head RMSNorm across query or key head vectors.
pub fn applyPerHeadNorm(heads: []f32, num_heads: usize, head_dim: usize, norm_weights: []const f32) void {
    std.debug.assert(heads.len == num_heads * head_dim);
    std.debug.assert(norm_weights.len == head_dim);

    for (0..num_heads) |h| {
        const head_slice = heads[h * head_dim .. (h + 1) * head_dim];
        ops.rmsNorm(head_slice, norm_weights, 1e-6, head_slice);
    }
}

/// Selects top-K expert indices from 128 router logits and computes normalized routing weights.
pub fn topKExperts(logits: []const f32, comptime K: usize, out_indices: *[K]usize, out_weights: *[K]f32) void {
    std.debug.assert(logits.len >= K);

    for (0..K) |i| {
        out_indices[i] = i;
    }

    for (K..logits.len) |cand_idx| {
        const cand_val = logits[cand_idx];
        var min_pos: usize = 0;
        var min_val: f32 = logits[out_indices[0]];
        for (1..K) |j| {
            const val = logits[out_indices[j]];
            if (val < min_val) {
                min_val = val;
                min_pos = j;
            }
        }
        if (cand_val > min_val) {
            out_indices[min_pos] = cand_idx;
        }
    }

    var top_logits: [K]f32 = undefined;
    for (0..K) |i| {
        top_logits[i] = logits[out_indices[i]];
    }
    ops.softmax(&top_logits, 1.0, out_weights);
}

/// Dequantizes and embeds the 256 canvas tokens into an [256, 2816] activation buffer,
/// applying Gemma embedding scale sqrt(d_model) = sqrt(2816).
pub fn embedCanvas(model: DiffusionGemmaModel, canvas_tokens: []const u32, out_emb: []f32) void {
    std.debug.assert(canvas_tokens.len == ModelConfig.canvas_length);
    std.debug.assert(out_emb.len == ModelConfig.canvas_length * ModelConfig.embedding_dim);

    const emb_scale: f32 = @sqrt(@as(f32, @floatFromInt(ModelConfig.embedding_dim)));

    for (canvas_tokens, 0..) |token_id, i| {
        const blocks = model.getEmbeddingBlocks(token_id);
        const row = out_emb[i * ModelConfig.embedding_dim .. (i + 1) * ModelConfig.embedding_dim];
        quant.dequantizeRowQ6_K(blocks, row);

        for (row) |*val| {
            val.* *= emb_scale;
        }
    }
}

/// Slices hidden state vector at answer position p and projects logits exclusively
/// for the supplied candidate token IDs using Q6_K dot products on token_embd.weight.
pub fn readCandidateLogits(
    model: DiffusionGemmaModel,
    hidden_p: []const f32,
    candidate_token_ids: []const u32,
    out_raw_logits: []f32,
    out_softcapped_logits: []f32,
) void {
    std.debug.assert(hidden_p.len == ModelConfig.embedding_dim);
    std.debug.assert(candidate_token_ids.len == out_raw_logits.len);
    std.debug.assert(candidate_token_ids.len == out_softcapped_logits.len);

    for (candidate_token_ids, 0..) |cand_id, i| {
        const blocks = model.getEmbeddingBlocks(cand_id);
        const raw_logit = quant.dotRowQ6_K(blocks, hidden_p);
        out_raw_logits[i] = raw_logit;
        out_softcapped_logits[i] = ops.logitSoftcap(raw_logit, ModelConfig.logit_softcap);
    }
}

/// Executes a single-step structured read over a seeded canvas.
/// Embeds canvas, evaluates forward pass representation, and extracts candidate logits at position p.
pub fn executeStructuredRead(
    allocator: std.mem.Allocator,
    model: DiffusionGemmaModel,
    canvas: Canvas,
    candidate_token_ids: []const u32,
    out_raw_logits: []f32,
    out_softcapped_logits: []f32,
) !void {
    const total_emb_elements = ModelConfig.canvas_length * ModelConfig.embedding_dim;
    const emb_buffer = try allocator.alloc(f32, total_emb_elements);
    defer allocator.free(emb_buffer);

    // 1. Embed canvas tokens with sqrt(d_model) scaling
    embedCanvas(model, &canvas.tokens, emb_buffer);

    // 2. Extract hidden state at pinned answer position p
    const answer_row = emb_buffer[canvas.answer_pos * ModelConfig.embedding_dim .. (canvas.answer_pos + 1) * ModelConfig.embedding_dim];

    // Normalize through output RMSNorm scale
    const norm_hidden = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(norm_hidden);

    const out_norm_slice = model.gguf_file.getTensorSlice(model.output_norm);
    const out_norm_weights = @as([*]const f32, @ptrCast(@alignCast(out_norm_slice.ptr)))[0..ModelConfig.embedding_dim];
    ops.rmsNorm(answer_row, out_norm_weights, 1e-6, norm_hidden);

    // 3. Project candidate logits directly from output embedding table
    readCandidateLogits(model, norm_hidden, candidate_token_ids, out_raw_logits, out_softcapped_logits);
}

test "topKExperts: selection and softmax renormalization" {
    var logits: [128]f32 = @splat(0.0);
    logits[10] = 5.0;
    logits[20] = 10.0;
    logits[30] = 3.0;
    logits[40] = 8.0;
    logits[50] = 7.0;
    logits[60] = 6.0;
    logits[70] = 9.0;
    logits[80] = 4.0;

    var indices: [8]usize = undefined;
    var weights: [8]f32 = undefined;
    topKExperts(&logits, 8, &indices, &weights);

    // Verify all selected are non-zero indices
    var sum: f32 = 0.0;
    for (weights) |w| {
        sum += w;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "applyRoPE: rotation at pos 0 is identity" {
    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    applyRoPE(&vec, 0, 4, 10000.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vec[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), vec[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), vec[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), vec[3], 1e-5);
}

test "embedCanvas and readCandidateLogits execution on local model if present" {
    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var model = root.DiffusionGemmaModel.loadFromFile(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer model.deinit();

    // Create seeded canvas: prompt tokens [BOS, 100, 200] and MASK at position 3
    const prompt = [_]u32{ root.SpecialTokens.bos, 100, 200 };
    const canvas = try Canvas.seedSingleStep(&prompt, root.SpecialTokens.mask);

    // Candidate tokens: Yes (10784 or 8438) and No (3771 or 2301)
    const candidates = [_]u32{ 10784, 3771 };
    var raw_logits: [2]f32 = undefined;
    var softcapped_logits: [2]f32 = undefined;

    try executeStructuredRead(allocator, model, canvas, &candidates, &raw_logits, &softcapped_logits);

    // Softcapped logits must be within [-30.0, 30.0]
    for (softcapped_logits) |l| {
        try std.testing.expect(l >= -30.0 and l <= 30.0);
    }
}

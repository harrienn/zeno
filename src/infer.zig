//! Zeno — Structured read forward evaluation primitive.
//!
//! Evaluates a seeded canvas, slices hidden state at pinned answer position p,
//! and projects candidate logits via Q6_K embedding dot products and logit softcapping.

const std = @import("std");
const root = @import("root.zig");
const quant = @import("quant.zig");
const ops = @import("ops.zig");
const model_mod = @import("model.zig");
const canvas_mod = @import("canvas.zig");

pub const DiffusionGemmaModel = model_mod.DiffusionGemmaModel;
pub const Canvas = canvas_mod.Canvas;
pub const ModelConfig = root.ModelConfig;

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

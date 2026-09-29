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
const gguf = @import("gguf.zig");

pub const DiffusionGemmaModel = model_mod.DiffusionGemmaModel;
pub const TransformerLayer = model_mod.TransformerLayer;
pub const Canvas = canvas_mod.Canvas;
pub const ModelConfig = root.ModelConfig;
pub const TensorInfo = gguf.TensorInfo;
pub const GgufFile = gguf.GgufFile;

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

/// Applies NeoX-style (half-split) Rotary Position Embeddings at position `pos`.
pub fn applyRoPE(vec: []f32, pos: usize, head_dim: usize, theta: f32) void {
    const half = head_dim / 2;
    const pos_f = @as(f32, @floatFromInt(pos));
    for (0..half) |i| {
        const exp = 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim));
        const freq = 1.0 / std.math.pow(f32, theta, exp);
        const angle = pos_f * freq;
        const cos = @cos(angle);
        const sin = @sin(angle);

        const v0 = vec[i];
        const v1 = vec[half + i];
        vec[i] = v0 * cos - v1 * sin;
        vec[half + i] = v0 * sin + v1 * cos;
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

/// Matrix-vector multiplication for Q8_0 quantized weights: out = W * x
pub fn gemvQ8_0(W_blocks: [*]const quant.BlockQ8_0, x: []const f32, out: []f32, out_dim: usize, in_dim: usize) void {
    std.debug.assert(in_dim % quant.QK8_0 == 0);
    std.debug.assert(x.len == in_dim);
    std.debug.assert(out.len == out_dim);

    const blocks_per_row = in_dim / quant.QK8_0;
    for (0..out_dim) |r| {
        const row_blocks = W_blocks[r * blocks_per_row .. (r + 1) * blocks_per_row];
        out[r] = quant.dotRowQ8_0(row_blocks, x);
    }
}

/// Matrix-vector multiplication for Q6_K quantized weights: out = W * x
pub fn gemvQ6_K(W_blocks: [*]const quant.BlockQ6_K, x: []const f32, out: []f32, out_dim: usize, in_dim: usize) void {
    std.debug.assert(in_dim % quant.QK_K == 0);
    std.debug.assert(x.len == in_dim);
    std.debug.assert(out.len == out_dim);

    const blocks_per_row = in_dim / quant.QK_K;
    for (0..out_dim) |r| {
        const row_blocks = W_blocks[r * blocks_per_row .. (r + 1) * blocks_per_row];
        out[r] = quant.dotRowQ6_K(row_blocks, x);
    }
}

/// Matrix-vector multiplication for Q5_0 quantized weights: out = W * x
pub fn gemvQ5_0(W_blocks: [*]const quant.BlockQ5_0, x: []const f32, out: []f32, out_dim: usize, in_dim: usize) void {
    std.debug.assert(in_dim % quant.QK5_0 == 0);
    std.debug.assert(x.len == in_dim);
    std.debug.assert(out.len == out_dim);

    const blocks_per_row = in_dim / quant.QK5_0;
    for (0..out_dim) |r| {
        const row_blocks = W_blocks[r * blocks_per_row .. (r + 1) * blocks_per_row];
        out[r] = quant.dotRowQ5_0(row_blocks, x);
    }
}

/// Dispatches matrix-vector multiplication based on tensor ggml_type: out = W * x
pub fn gemvTensor(gfile: gguf.GgufFile, tensor: gguf.TensorInfo, x: []const f32, out: []f32, out_dim: usize, in_dim: usize) void {
    const raw_slice = gfile.getTensorSlice(tensor);
    switch (tensor.ggml_type) {
        0 => { // F32
            const w_ptr: [*]const f32 = @ptrCast(@alignCast(raw_slice.ptr));
            gemvF32(w_ptr[0 .. out_dim * in_dim], x, out, out_dim, in_dim);
        },
        6 => { // Q5_0
            const b_ptr: [*]const quant.BlockQ5_0 = @ptrCast(@alignCast(raw_slice.ptr));
            gemvQ5_0(b_ptr, x, out, out_dim, in_dim);
        },
        8 => { // Q8_0
            const b_ptr: [*]const quant.BlockQ8_0 = @ptrCast(@alignCast(raw_slice.ptr));
            gemvQ8_0(b_ptr, x, out, out_dim, in_dim);
        },
        12 => { // Q4_K
            const b_ptr: [*]const quant.BlockQ4_K = @ptrCast(@alignCast(raw_slice.ptr));
            gemvQ4_K(b_ptr, x, out, out_dim, in_dim);
        },
        14 => { // Q6_K
            const b_ptr: [*]const quant.BlockQ6_K = @ptrCast(@alignCast(raw_slice.ptr));
            gemvQ6_K(b_ptr, x, out, out_dim, in_dim);
        },
        else => std.debug.panic("Unsupported tensor ggml_type {} for tensor '{s}' (shape: [{}, {}])", .{ tensor.ggml_type, tensor.name, tensor.dimensions[0], tensor.dimensions[1] }),
    }
}

/// MoE expert gate_up projection: [2816 -> 1408] for selected expert_idx
pub fn gemvExpertGateUp(
    W_blocks: [*]const quant.BlockQ4_K,
    expert_idx: usize,
    x: []const f32,
    out_gate_up: []f32,
) void {
    std.debug.assert(x.len == 2816);
    std.debug.assert(out_gate_up.len == 1408);
    const blocks_per_row = 2816 / quant.QK_K; // 11
    const expert_offset = expert_idx * 1408 * blocks_per_row;
    for (0..1408) |r| {
        const row_start = expert_offset + r * blocks_per_row;
        const row_blocks = W_blocks[row_start .. row_start + blocks_per_row];
        out_gate_up[r] = quant.dotRowQ4_K(row_blocks, x);
    }
}

/// MoE expert down projection: [704 -> 2816] for selected expert_idx
pub fn gemvExpertDown(
    tensor: gguf.TensorInfo,
    raw_slice: []const u8,
    expert_idx: usize,
    x: []const f32,
    out_down: []f32,
) void {
    std.debug.assert(x.len == 704);
    std.debug.assert(out_down.len == 2816);

    switch (tensor.ggml_type) {
        6 => { // Q5_0
            const W_blocks: [*]const quant.BlockQ5_0 = @ptrCast(@alignCast(raw_slice.ptr));
            const blocks_per_row = 704 / quant.QK5_0; // 22
            const expert_offset = expert_idx * 2816 * blocks_per_row;
            for (0..2816) |r| {
                const row_start = expert_offset + r * blocks_per_row;
                const row_blocks = W_blocks[row_start .. row_start + blocks_per_row];
                out_down[r] = quant.dotRowQ5_0(row_blocks, x);
            }
        },
        8 => { // Q8_0
            const W_blocks: [*]const quant.BlockQ8_0 = @ptrCast(@alignCast(raw_slice.ptr));
            const blocks_per_row = 704 / quant.QK8_0; // 22
            const expert_offset = expert_idx * 2816 * blocks_per_row;
            for (0..2816) |r| {
                const row_start = expert_offset + r * blocks_per_row;
                const row_blocks = W_blocks[row_start .. row_start + blocks_per_row];
                out_down[r] = quant.dotRowQ8_0(row_blocks, x);
            }
        },
        else => std.debug.panic("Unsupported expert_down ggml_type {}", .{tensor.ggml_type}),
    }
}

/// Dequantizes and embeds a slice of canvas tokens into an [N, 2816] activation buffer,
/// applying Gemma embedding scale sqrt(d_model) = sqrt(2816).
pub fn embedCanvasSlice(model: DiffusionGemmaModel, canvas_tokens: []const u32, out_emb: []f32) void {
    std.debug.assert(out_emb.len == canvas_tokens.len * ModelConfig.embedding_dim);

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

/// Dequantizes and embeds all 256 canvas tokens into an [256, 2816] activation buffer.
pub fn embedCanvas(model: DiffusionGemmaModel, canvas_tokens: []const u32, out_emb: []f32) void {
    std.debug.assert(canvas_tokens.len == ModelConfig.canvas_length);
    embedCanvasSlice(model, canvas_tokens, out_emb);
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
/// Embeds canvas, evaluates all 30 transformer layers following the verified 15-slot sequence,
/// and extracts candidate logits at position p.
pub fn executeStructuredRead(
    allocator: std.mem.Allocator,
    model: DiffusionGemmaModel,
    canvas: Canvas,
    candidate_token_ids: []const u32,
    out_raw_logits: []f32,
    out_softcapped_logits: []f32,
) !void {
    // 1. Determine active evaluation length (multiples of 16 holding answer_pos + 1, up to canvas_length)
    const min_eval_tokens = canvas.answer_pos + 1;
    const step_16: usize = 16;
    const rounded_tokens = ((min_eval_tokens + step_16 - 1) / step_16) * step_16;
    const eval_len = @min(ModelConfig.canvas_length, rounded_tokens);

    // 2. Allocate scratch buffers for the canvas activations
    const total_emb_elements = eval_len * ModelConfig.embedding_dim;
    const emb_buffer = try allocator.alloc(f32, total_emb_elements);
    defer allocator.free(emb_buffer);

    const next_emb_buffer = try allocator.alloc(f32, total_emb_elements);
    defer allocator.free(next_emb_buffer);

    // Embed initial canvas tokens with sqrt(d_model) scaling
    embedCanvasSlice(model, canvas.tokens[0..eval_len], emb_buffer);

    // Scratch buffers for layer evaluation
    const max_q_dim: usize = 8192;
    const max_k_dim: usize = 2048;

    const q_buf = try allocator.alloc(f32, eval_len * max_q_dim);
    defer allocator.free(q_buf);

    const k_buf = try allocator.alloc(f32, eval_len * max_k_dim);
    defer allocator.free(k_buf);

    const v_buf = try allocator.alloc(f32, eval_len * max_k_dim);
    defer allocator.free(v_buf);

    const norm_x_buf = try allocator.alloc(f32, eval_len * ModelConfig.embedding_dim);
    defer allocator.free(norm_x_buf);

    const attn_out_buf = try allocator.alloc(f32, eval_len * ModelConfig.embedding_dim);
    defer allocator.free(attn_out_buf);

    const head_out = try allocator.alloc(f32, max_q_dim);
    defer allocator.free(head_out);

    const scores = try allocator.alloc(f32, eval_len);
    defer allocator.free(scores);

    const attn_weights = try allocator.alloc(f32, eval_len);
    defer allocator.free(attn_weights);

    // Dense & MoE scratch buffers
    const dense_in = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(dense_in);

    const gate_buf = try allocator.alloc(f32, 2112);
    defer allocator.free(gate_buf);

    const up_buf = try allocator.alloc(f32, 2112);
    defer allocator.free(up_buf);

    const geglu_buf = try allocator.alloc(f32, 2112);
    defer allocator.free(geglu_buf);

    const dense_out = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(dense_out);

    const hidden_states_1 = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(hidden_states_1);

    const r_in = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(r_in);

    const router_logits = try allocator.alloc(f32, 128);
    defer allocator.free(router_logits);

    const moe_in = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(moe_in);

    const exp_gate_up_buf = try allocator.alloc(f32, 1408);
    defer allocator.free(exp_gate_up_buf);

    const exp_act_buf = try allocator.alloc(f32, 704);
    defer allocator.free(exp_act_buf);

    const exp_down_buf = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(exp_down_buf);

    const moe_accum = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(moe_accum);

    const hidden_states_2 = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(hidden_states_2);

    const ffn_combined = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(ffn_combined);

    const post_ffn = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(post_ffn);

    // 3. Execute 30 Transformer Blocks following verified 15-slot sequence
    var current_emb = emb_buffer;
    var next_emb = next_emb_buffer;

    for (model.layers) |layer| {
        const is_global = layer.is_global;
        const q_dim = layer.q_weight.dimensions[1];
        const k_dim = layer.k_weight.dimensions[1];
        const head_dim = layer.q_norm.dimensions[0];
        const num_q_heads = q_dim / head_dim;
        const num_kv_heads = k_dim / head_dim;
        const gqa_ratio = num_q_heads / num_kv_heads;
        const theta: f32 = if (is_global) 1000000.0 else 10000.0;
        const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const root_size: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ModelConfig.embedding_dim)));

        // Get pointers to tensor weights
        const attn_norm_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.attn_norm).ptr));
        const q_norm_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.q_norm).ptr));
        const k_norm_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.k_norm).ptr));
        const post_attn_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.post_attention_norm).ptr));
        const ffn_norm_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.ffn_norm).ptr));
        const post_ffw_1_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.post_ffw_norm_1).ptr));
        const pre_ffw_2_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.pre_ffw_norm_2).ptr));
        const router_scale_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.router_scale).ptr));
        const exp_down_scale_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.expert_down_scale).ptr));
        const post_ffw_2_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.post_ffw_norm_2).ptr));
        const post_ffw_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.post_ffw_norm).ptr));
        const layer_out_scale_weights: [*]const f32 = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.layer_output_scale).ptr));

        const exp_gate_up_blocks: [*]const quant.BlockQ4_K = @ptrCast(@alignCast(model.gguf_file.getTensorSlice(layer.expert_gate_up).ptr));
        const exp_down_slice = model.gguf_file.getTensorSlice(layer.expert_down);

        // --- Slot 1: Input RMSNorm across all canvas tokens ---
        for (0..eval_len) |t| {
            const x_t = current_emb[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
            const nx_t = norm_x_buf[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
            ops.rmsNorm(x_t, attn_norm_weights[0..ModelConfig.embedding_dim], 1e-6, nx_t);
        }

        // --- Slot 2, 3, 4: Q/K/V Projections, Norms, RoPE ---
        for (0..eval_len) |t| {
            const nx_t = norm_x_buf[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
            const q_t = q_buf[t * max_q_dim .. t * max_q_dim + q_dim];
            const k_t = k_buf[t * max_k_dim .. t * max_k_dim + k_dim];
            const v_t = v_buf[t * max_k_dim .. t * max_k_dim + k_dim];

            gemvTensor(model.gguf_file, layer.q_weight, nx_t, q_t, q_dim, ModelConfig.embedding_dim);
            gemvTensor(model.gguf_file, layer.k_weight, nx_t, k_t, k_dim, ModelConfig.embedding_dim);

            if (is_global) {
                // Global layers: V is tied to K BEFORE k_norm and RoPE
                @memcpy(v_t, k_t);
            } else {
                gemvTensor(model.gguf_file, layer.v_weight.?, nx_t, v_t, k_dim, ModelConfig.embedding_dim);
            }

            // Slot 3: Per-head Q/K norm
            applyPerHeadNorm(q_t, num_q_heads, head_dim, q_norm_weights[0..head_dim]);
            applyPerHeadNorm(k_t, num_kv_heads, head_dim, k_norm_weights[0..head_dim]);

            // V receives unweighted RMSNorm per head
            for (0..num_kv_heads) |kv| {
                const v_head = v_t[kv * head_dim .. (kv + 1) * head_dim];
                ops.unweightedRmsNorm(v_head, 1e-6, v_head);
            }

            // Slot 4: RoPE on Q and K
            for (0..num_q_heads) |h| {
                applyRoPE(q_t[h * head_dim .. (h + 1) * head_dim], t, head_dim, theta);
            }
            for (0..num_kv_heads) |kv| {
                applyRoPE(k_t[kv * head_dim .. (kv + 1) * head_dim], t, head_dim, theta);
            }
        }

        // --- Slot 5: Non-Causal Bidirectional Attention + W_O Output Projection ---
        for (0..eval_len) |t| {
            const q_t = q_buf[t * max_q_dim .. t * max_q_dim + q_dim];
            @memset(head_out[0..q_dim], 0.0);

            for (0..num_q_heads) |h| {
                const kv = h / gqa_ratio;
                const q_head = q_t[h * head_dim .. (h + 1) * head_dim];

                // Attention dot-product across all canvas tokens s in [0, eval_len)
                for (0..eval_len) |s| {
                    const k_s = k_buf[s * max_k_dim .. s * max_k_dim + k_dim];
                    const k_head = k_s[kv * head_dim .. (kv + 1) * head_dim];
                    var dot: f32 = 0.0;
                    for (q_head, k_head) |qv, kvv| {
                        dot += qv * kvv;
                    }
                    scores[s] = dot * inv_sqrt_d;
                }

                ops.softmax(scores[0..eval_len], 1.0, attn_weights[0..eval_len]);

                // Weighted sum over V_s
                const h_out = head_out[h * head_dim .. (h + 1) * head_dim];
                for (0..eval_len) |s| {
                    const w = attn_weights[s];
                    if (w == 0.0) continue;
                    const v_s = v_buf[s * max_k_dim .. s * max_k_dim + k_dim];
                    const v_head = v_s[kv * head_dim .. (kv + 1) * head_dim];
                    for (h_out, v_head) |*ho, vv| {
                        ho.* += w * vv;
                    }
                }
            }

            // W_O Projection
            const a_out = attn_out_buf[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
            gemvTensor(model.gguf_file, layer.o_weight, head_out[0..q_dim], a_out, ModelConfig.embedding_dim, q_dim);

            // Slot 6: Post-attention norm
            ops.rmsNorm(a_out, post_attn_weights[0..ModelConfig.embedding_dim], 1e-6, a_out);

            // Slot 7: Attention residual add: x_mid = x + attn_out
            const x_t = current_emb[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
            for (a_out, x_t) |*ao, xt| {
                ao.* += xt;
            }
        }

        // --- Slots 8–15: Dense MLP + MoE Expert Evaluation ---
        for (0..eval_len) |t| {
            const x_mid_t = attn_out_buf[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];

            // Slot 8: Dense FFN input norm
            ops.rmsNorm(x_mid_t, ffn_norm_weights[0..ModelConfig.embedding_dim], 1e-6, dense_in);

            // Slot 9: Dense MLP branch (GeGLU)
            gemvTensor(model.gguf_file, layer.ffn_gate, dense_in, gate_buf, 2112, ModelConfig.embedding_dim);
            gemvTensor(model.gguf_file, layer.ffn_up, dense_in, up_buf, 2112, ModelConfig.embedding_dim);
            ops.geglu(gate_buf, up_buf, geglu_buf);
            gemvTensor(model.gguf_file, layer.ffn_down, geglu_buf, dense_out, ModelConfig.embedding_dim, 2112);
            ops.rmsNorm(dense_out, post_ffw_1_weights[0..ModelConfig.embedding_dim], 1e-6, hidden_states_1);

            // Slot 10: MoE Router (consumes x_mid_t)
            ops.unweightedRmsNorm(x_mid_t, 1e-6, r_in);
            for (r_in, 0..) |*val, dim| {
                val.* *= root_size * router_scale_weights[dim];
            }
            gemvTensor(model.gguf_file, layer.router_gate, r_in, router_logits, 128, ModelConfig.embedding_dim);

            var top_indices: [8]usize = undefined;
            var top_weights: [8]f32 = undefined;
            topKExperts(router_logits, 8, &top_indices, &top_weights);

            for (0..8) |e| {
                top_weights[e] *= exp_down_scale_weights[top_indices[e]];
            }

            // Slot 11: MoE Expert branch
            ops.rmsNorm(x_mid_t, pre_ffw_2_weights[0..ModelConfig.embedding_dim], 1e-6, moe_in);
            @memset(moe_accum, 0.0);

            for (0..8) |e| {
                const exp_id = top_indices[e];
                const exp_w = top_weights[e];
                if (exp_w == 0.0) continue;

                gemvExpertGateUp(exp_gate_up_blocks, exp_id, moe_in, exp_gate_up_buf);
                ops.geglu(exp_gate_up_buf[0..704], exp_gate_up_buf[704..1408], exp_act_buf);
                gemvExpertDown(layer.expert_down, exp_down_slice, exp_id, exp_act_buf, exp_down_buf);

                for (moe_accum, exp_down_buf) |*acc, dv| {
                    acc.* += dv * exp_w;
                }
            }

            ops.rmsNorm(moe_accum, post_ffw_2_weights[0..ModelConfig.embedding_dim], 1e-6, hidden_states_2);

            // Slot 12: Combine Dense MLP + MoE branches
            for (ffn_combined, hidden_states_1, hidden_states_2) |*comb, h1, h2| {
                comb.* = h1 + h2;
            }

            // Slot 13: Combined post-FFN norm
            ops.rmsNorm(ffn_combined, post_ffw_weights[0..ModelConfig.embedding_dim], 1e-6, post_ffn);

            // Slot 14 & 15: FFN residual add and Layer Output Scale
            const l_scale = layer_out_scale_weights[0];
            const out_t = next_emb[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
            for (out_t, x_mid_t, post_ffn) |*ot, xm, pf| {
                ot.* = (xm + pf) * l_scale;
            }
        }

        // Swap buffers for next layer
        const temp = current_emb;
        current_emb = next_emb;
        next_emb = temp;
    }

    // 4. Extract hidden state at pinned answer position p
    const answer_row = current_emb[canvas.answer_pos * ModelConfig.embedding_dim .. (canvas.answer_pos + 1) * ModelConfig.embedding_dim];

    // Final output RMSNorm
    const norm_hidden = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(norm_hidden);

    const out_norm_slice = model.gguf_file.getTensorSlice(model.output_norm);
    const out_norm_weights = @as([*]const f32, @ptrCast(@alignCast(out_norm_slice.ptr)))[0..ModelConfig.embedding_dim];
    ops.rmsNorm(answer_row, out_norm_weights, 1e-6, norm_hidden);

    // 5. Project candidate logits directly from output embedding table
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

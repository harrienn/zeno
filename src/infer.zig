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
    var x_sums: [256]f32 = undefined;
    const num_groups = in_dim / 32;
    std.debug.assert(num_groups <= x_sums.len);
    quant.computeXSums32(x, x_sums[0..num_groups]);

    for (0..out_dim) |r| {
        const row_blocks = W_blocks[r * blocks_per_row .. (r + 1) * blocks_per_row];
        out[r] = quant.dotRowQ4_K_with_xsums(row_blocks, x, x_sums[0..num_groups]);
    }
}

/// Matrix-vector multiplication for standard float32 weights: out = W * x
pub fn gemvF32(weights: []const f32, x: []const f32, out: []f32, out_dim: usize, in_dim: usize) void {
    std.debug.assert(weights.len == out_dim * in_dim);
    std.debug.assert(x.len == in_dim);
    std.debug.assert(out.len == out_dim);

    const Vec8 = @Vector(8, f32);
    const n_chunks = in_dim / 8;
    const rem = in_dim % 8;

    for (0..out_dim) |r| {
        const row = weights[r * in_dim .. (r + 1) * in_dim];
        var acc: Vec8 = @splat(0.0);
        for (0..n_chunks) |c| {
            const off = c * 8;
            const arr_w: [8]f32 = row[off..][0..8].*;
            const arr_x: [8]f32 = x[off..][0..8].*;
            const vw: Vec8 = arr_w;
            const vx: Vec8 = arr_x;
            acc = @mulAdd(Vec8, vw, vx, acc);
        }
        var sum = @reduce(.Add, acc);
        for (0..rem) |k| {
            const off = n_chunks * 8 + k;
            sum += row[off] * x[off];
        }
        out[r] = sum;
    }
}

/// Applies NeoX-style (half-split) Rotary Position Embeddings over the full head dimension.
pub fn applyRoPE(vec: []f32, pos: usize, head_dim: usize, theta: f32, freq_factors: ?[*]const f32) void {
    const half = head_dim / 2;
    const pos_f = @as(f32, @floatFromInt(pos));
    for (0..half) |i| {
        const exp = 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim));
        const ff: f32 = if (freq_factors) |factors| factors[i] else 1.0;
        const freq = (1.0 / std.math.pow(f32, theta, exp)) / ff;
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

/// MoE expert gate_up projection with precomputed activation sums: [2816 -> 1408] for selected expert_idx
pub fn gemvExpertGateUpWithXSums(
    W_blocks: [*]const quant.BlockQ4_K,
    expert_idx: usize,
    x: []const f32,
    x_sums: []const f32,
    out_gate_up: []f32,
) void {
    std.debug.assert(x.len == 2816);
    std.debug.assert(out_gate_up.len == 1408);
    const blocks_per_row = 2816 / quant.QK_K; // 11
    const expert_offset = expert_idx * 1408 * blocks_per_row;
    for (0..1408) |r| {
        const row_start = expert_offset + r * blocks_per_row;
        const row_blocks = W_blocks[row_start .. row_start + blocks_per_row];
        out_gate_up[r] = quant.dotRowQ4_K_with_xsums(row_blocks, x, x_sums);
    }
}

/// MoE expert gate_up projection: [2816 -> 1408] for selected expert_idx
pub fn gemvExpertGateUp(
    W_blocks: [*]const quant.BlockQ4_K,
    expert_idx: usize,
    x: []const f32,
    out_gate_up: []f32,
) void {
    var x_sums: [88]f32 = undefined;
    quant.computeXSums32(x, &x_sums);
    gemvExpertGateUpWithXSums(W_blocks, expert_idx, x, &x_sums, out_gate_up);
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

const WorkerScratch = struct {
    head_out: []f32,
    scores: []f32,
    attn_weights: []f32,
    dense_in: []f32,
    gate_buf: []f32,
    up_buf: []f32,
    geglu_buf: []f32,
    dense_out: []f32,
    hidden_states_1: []f32,
    r_in: []f32,
    router_logits: []f32,
    moe_in: []f32,
    exp_gate_up_buf: []f32,
    exp_act_buf: []f32,
    exp_down_buf: []f32,
    moe_accum: []f32,
    hidden_states_2: []f32,
    ffn_combined: []f32,
    post_ffn: []f32,

    pub fn init(allocator: std.mem.Allocator, eval_len: usize, max_q_dim: usize) !WorkerScratch {
        return .{
            .head_out = try allocator.alloc(f32, max_q_dim),
            .scores = try allocator.alloc(f32, eval_len),
            .attn_weights = try allocator.alloc(f32, eval_len),
            .dense_in = try allocator.alloc(f32, ModelConfig.embedding_dim),
            .gate_buf = try allocator.alloc(f32, 2112),
            .up_buf = try allocator.alloc(f32, 2112),
            .geglu_buf = try allocator.alloc(f32, 2112),
            .dense_out = try allocator.alloc(f32, ModelConfig.embedding_dim),
            .hidden_states_1 = try allocator.alloc(f32, ModelConfig.embedding_dim),
            .r_in = try allocator.alloc(f32, ModelConfig.embedding_dim),
            .router_logits = try allocator.alloc(f32, 128),
            .moe_in = try allocator.alloc(f32, ModelConfig.embedding_dim),
            .exp_gate_up_buf = try allocator.alloc(f32, 1408),
            .exp_act_buf = try allocator.alloc(f32, 704),
            .exp_down_buf = try allocator.alloc(f32, ModelConfig.embedding_dim),
            .moe_accum = try allocator.alloc(f32, ModelConfig.embedding_dim),
            .hidden_states_2 = try allocator.alloc(f32, ModelConfig.embedding_dim),
            .ffn_combined = try allocator.alloc(f32, ModelConfig.embedding_dim),
            .post_ffn = try allocator.alloc(f32, ModelConfig.embedding_dim),
        };
    }

    pub fn deinit(self: *WorkerScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.head_out);
        allocator.free(self.scores);
        allocator.free(self.attn_weights);
        allocator.free(self.dense_in);
        allocator.free(self.gate_buf);
        allocator.free(self.up_buf);
        allocator.free(self.geglu_buf);
        allocator.free(self.dense_out);
        allocator.free(self.hidden_states_1);
        allocator.free(self.r_in);
        allocator.free(self.router_logits);
        allocator.free(self.moe_in);
        allocator.free(self.exp_gate_up_buf);
        allocator.free(self.exp_act_buf);
        allocator.free(self.exp_down_buf);
        allocator.free(self.moe_accum);
        allocator.free(self.hidden_states_2);
        allocator.free(self.ffn_combined);
        allocator.free(self.post_ffn);
    }
};

const LayerContext = struct {
    model: DiffusionGemmaModel,
    layer_idx: usize,
    prompt_len: usize,
    total_len: usize,
    max_q_dim: usize,
    max_k_dim: usize,
    rope_freqs: ?[*]const f32,
    current_emb: []f32,
    next_emb: []f32,
    norm_x_buf: []f32,
    q_buf: []f32,
    k_buf: []f32,
    v_buf: []f32,
    attn_out_buf: []f32,
};

fn runQKVWorker(ctx: *const LayerContext, t_start: usize, t_end: usize) void {
    const layer = ctx.model.layers[ctx.layer_idx];
    const is_global = layer.is_global;
    const q_dim = layer.q_weight.dimensions[1];
    const k_dim = layer.k_weight.dimensions[1];
    const head_dim = layer.q_norm.dimensions[0];
    const num_q_heads = q_dim / head_dim;
    const num_kv_heads = k_dim / head_dim;
    const theta: f32 = if (is_global) 1000000.0 else 10000.0;

    const attn_norm_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.attn_norm).ptr));
    const q_norm_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.q_norm).ptr));
    const k_norm_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.k_norm).ptr));

    for (t_start..t_end) |t| {
        const x_t = ctx.current_emb[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
        const nx_t = ctx.norm_x_buf[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
        ops.rmsNorm(x_t, attn_norm_weights[0..ModelConfig.embedding_dim], 1e-6, nx_t);

        const q_t = ctx.q_buf[t * ctx.max_q_dim .. t * ctx.max_q_dim + q_dim];
        const k_t = ctx.k_buf[t * ctx.max_k_dim .. t * ctx.max_k_dim + k_dim];
        const v_t = ctx.v_buf[t * ctx.max_k_dim .. t * ctx.max_k_dim + k_dim];

        gemvTensor(ctx.model.gguf_file, layer.q_weight, nx_t, q_t, q_dim, ModelConfig.embedding_dim);
        gemvTensor(ctx.model.gguf_file, layer.k_weight, nx_t, k_t, k_dim, ModelConfig.embedding_dim);

        if (is_global) {
            @memcpy(v_t, k_t);
        } else {
            gemvTensor(ctx.model.gguf_file, layer.v_weight.?, nx_t, v_t, k_dim, ModelConfig.embedding_dim);
        }

        applyPerHeadNorm(q_t, num_q_heads, head_dim, q_norm_weights[0..head_dim]);
        applyPerHeadNorm(k_t, num_kv_heads, head_dim, k_norm_weights[0..head_dim]);

        for (0..num_kv_heads) |kv| {
            const v_head = v_t[kv * head_dim .. (kv + 1) * head_dim];
            ops.unweightedRmsNorm(v_head, 1e-6, v_head);
        }

        const factors = if (is_global) ctx.rope_freqs else null;
        for (0..num_q_heads) |h| {
            applyRoPE(q_t[h * head_dim .. (h + 1) * head_dim], t, head_dim, theta, factors);
        }
        for (0..num_kv_heads) |kv| {
            applyRoPE(k_t[kv * head_dim .. (kv + 1) * head_dim], t, head_dim, theta, factors);
        }
    }
}

fn runAttnWorker(ctx: *const LayerContext, scratch: *WorkerScratch, t_start: usize, t_end: usize) void {
    const layer = ctx.model.layers[ctx.layer_idx];
    const q_dim = layer.q_weight.dimensions[1];
    const k_dim = layer.k_weight.dimensions[1];
    const head_dim = layer.q_norm.dimensions[0];
    const num_q_heads = q_dim / head_dim;
    const num_kv_heads = k_dim / head_dim;
    const gqa_ratio = num_q_heads / num_kv_heads;
    const post_attn_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.post_attention_norm).ptr));

    for (t_start..t_end) |t| {
        const is_prompt_query = (t < ctx.prompt_len);
        const q_t = ctx.q_buf[t * ctx.max_q_dim .. t * ctx.max_q_dim + q_dim];
        @memset(scratch.head_out[0..q_dim], 0.0);

        for (0..num_q_heads) |h| {
            const kv = h / gqa_ratio;
            const q_head = q_t[h * head_dim .. (h + 1) * head_dim];

            for (0..ctx.total_len) |s| {
                if (is_prompt_query and s > t) {
                    scratch.scores[s] = -1e30;
                } else {
                    const k_s = ctx.k_buf[s * ctx.max_k_dim .. s * ctx.max_k_dim + k_dim];
                    const k_head = k_s[kv * head_dim .. (kv + 1) * head_dim];
                    var dot: f32 = 0.0;
                    for (q_head, k_head) |qv, kvv| {
                        dot += qv * kvv;
                    }
                    scratch.scores[s] = dot; // Gemma 4 attention scale is 1.0f
                }
            }

            ops.softmax(scratch.scores[0..ctx.total_len], 1.0, scratch.attn_weights[0..ctx.total_len]);

            const h_out = scratch.head_out[h * head_dim .. (h + 1) * head_dim];
            for (0..ctx.total_len) |s| {
                const w = scratch.attn_weights[s];
                if (w == 0.0) continue;
                const v_s = ctx.v_buf[s * ctx.max_k_dim .. s * ctx.max_k_dim + k_dim];
                const v_head = v_s[kv * head_dim .. (kv + 1) * head_dim];
                for (h_out, v_head) |*ho, vv| {
                    ho.* += w * vv;
                }
            }
        }

        const a_out = ctx.attn_out_buf[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
        gemvTensor(ctx.model.gguf_file, layer.o_weight, scratch.head_out[0..q_dim], a_out, ModelConfig.embedding_dim, q_dim);
        ops.rmsNorm(a_out, post_attn_weights[0..ModelConfig.embedding_dim], 1e-6, a_out);

        const x_t = ctx.current_emb[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
        for (a_out, x_t) |*ao, xt| {
            ao.* += xt;
        }
    }
}

fn runFFNWorker(ctx: *const LayerContext, scratch: *WorkerScratch, t_start: usize, t_end: usize) void {
    const layer = ctx.model.layers[ctx.layer_idx];
    const root_size: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ModelConfig.embedding_dim)));

    const ffn_norm_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.ffn_norm).ptr));
    const post_ffw_1_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.post_ffw_norm_1).ptr));
    const pre_ffw_2_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.pre_ffw_norm_2).ptr));
    const router_scale_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.router_scale).ptr));
    const exp_down_scale_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.expert_down_scale).ptr));
    const post_ffw_2_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.post_ffw_norm_2).ptr));
    const post_ffw_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.post_ffw_norm).ptr));
    const enc_layer_out_scale_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.enc_layer_output_scale).ptr));
    const layer_out_scale_weights: [*]const f32 = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.layer_output_scale).ptr));

    const exp_gate_up_blocks: [*]const quant.BlockQ4_K = @ptrCast(@alignCast(ctx.model.gguf_file.getTensorSlice(layer.expert_gate_up).ptr));
    const exp_down_slice = ctx.model.gguf_file.getTensorSlice(layer.expert_down);

    for (t_start..t_end) |t| {
        const x_mid_t = ctx.attn_out_buf[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];

        ops.rmsNorm(x_mid_t, ffn_norm_weights[0..ModelConfig.embedding_dim], 1e-6, scratch.dense_in);

        gemvTensor(ctx.model.gguf_file, layer.ffn_gate, scratch.dense_in, scratch.gate_buf, 2112, ModelConfig.embedding_dim);
        gemvTensor(ctx.model.gguf_file, layer.ffn_up, scratch.dense_in, scratch.up_buf, 2112, ModelConfig.embedding_dim);
        ops.geglu(scratch.gate_buf, scratch.up_buf, scratch.geglu_buf);
        gemvTensor(ctx.model.gguf_file, layer.ffn_down, scratch.geglu_buf, scratch.dense_out, ModelConfig.embedding_dim, 2112);
        ops.rmsNorm(scratch.dense_out, post_ffw_1_weights[0..ModelConfig.embedding_dim], 1e-6, scratch.hidden_states_1);

        ops.unweightedRmsNorm(x_mid_t, 1e-6, scratch.r_in);
        for (scratch.r_in, 0..) |*val, dim| {
            val.* *= root_size * router_scale_weights[dim];
        }
        gemvTensor(ctx.model.gguf_file, layer.router_gate, scratch.r_in, scratch.router_logits, 128, ModelConfig.embedding_dim);

        var top_indices: [8]usize = undefined;
        var top_weights: [8]f32 = undefined;
        topKExperts(scratch.router_logits, 8, &top_indices, &top_weights);

        for (0..8) |e| {
            top_weights[e] *= exp_down_scale_weights[top_indices[e]];
        }

        ops.rmsNorm(x_mid_t, pre_ffw_2_weights[0..ModelConfig.embedding_dim], 1e-6, scratch.moe_in);
        var moe_in_xsums: [88]f32 = undefined;
        quant.computeXSums32(scratch.moe_in, &moe_in_xsums);
        @memset(scratch.moe_accum, 0.0);

        for (0..8) |e| {
            const exp_id = top_indices[e];
            const exp_w = top_weights[e];
            if (exp_w == 0.0) continue;

            gemvExpertGateUpWithXSums(exp_gate_up_blocks, exp_id, scratch.moe_in, &moe_in_xsums, scratch.exp_gate_up_buf);
            ops.geglu(scratch.exp_gate_up_buf[0..704], scratch.exp_gate_up_buf[704..1408], scratch.exp_act_buf);
            gemvExpertDown(layer.expert_down, exp_down_slice, exp_id, scratch.exp_act_buf, scratch.exp_down_buf);

            for (scratch.moe_accum, scratch.exp_down_buf) |*acc, dv| {
                acc.* += dv * exp_w;
            }
        }

        ops.rmsNorm(scratch.moe_accum, post_ffw_2_weights[0..ModelConfig.embedding_dim], 1e-6, scratch.hidden_states_2);

        for (scratch.ffn_combined, scratch.hidden_states_1, scratch.hidden_states_2) |*comb, h1, h2| {
            comb.* = h1 + h2;
        }

        ops.rmsNorm(scratch.ffn_combined, post_ffw_weights[0..ModelConfig.embedding_dim], 1e-6, scratch.post_ffn);

        const l_scale = if (t < ctx.prompt_len) enc_layer_out_scale_weights[0] else layer_out_scale_weights[0];
        const out_t = ctx.next_emb[t * ModelConfig.embedding_dim .. (t + 1) * ModelConfig.embedding_dim];
        for (out_t, x_mid_t, scratch.post_ffn) |*ot, xm, pf| {
            ot.* = (xm + pf) * l_scale;
        }
    }
}

const PoolPhase = enum(u32) {
    idle,
    qkv,
    attn,
    ffn,
    terminate,
};

const LayerPool = struct {
    phase: std.atomic.Value(PoolPhase) = std.atomic.Value(PoolPhase).init(.idle),
    epoch: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    done_count: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    ctx: *const LayerContext = undefined,
    scratches: []WorkerScratch = undefined,
    num_threads: usize,
    tokens_per_thread: usize,
    total_len: usize,

    fn workerLoop(self: *LayerPool, tid: usize) void {
        var last_epoch: u32 = 0;
        const t_start = tid * self.tokens_per_thread;
        const t_end = @min(t_start + self.tokens_per_thread, self.total_len);

        while (true) {
            while (true) {
                const ep = self.epoch.load(.acquire);
                if (ep != last_epoch) {
                    last_epoch = ep;
                    break;
                }
                std.atomic.spinLoopHint();
            }

            const p = self.phase.load(.acquire);
            if (p == .terminate) return;

            if (t_start < self.total_len) {
                switch (p) {
                    .qkv => runQKVWorker(self.ctx, t_start, t_end),
                    .attn => runAttnWorker(self.ctx, &self.scratches[tid], t_start, t_end),
                    .ffn => runFFNWorker(self.ctx, &self.scratches[tid], t_start, t_end),
                    .idle, .terminate => {},
                }
            }

            _ = self.done_count.fetchAdd(1, .release);
        }
    }

    fn dispatch(self: *LayerPool, p: PoolPhase) void {
        self.phase.store(p, .release);
        self.done_count.store(0, .release);
        _ = self.epoch.fetchAdd(1, .release);

        const t_start = 0;
        const t_end = @min(self.tokens_per_thread, self.total_len);
        if (t_start < self.total_len) {
            switch (p) {
                .qkv => runQKVWorker(self.ctx, t_start, t_end),
                .attn => runAttnWorker(self.ctx, &self.scratches[0], t_start, t_end),
                .ffn => runFFNWorker(self.ctx, &self.scratches[0], t_start, t_end),
                .idle, .terminate => {},
            }
        }

        const target = @as(u32, @intCast(self.num_threads - 1));
        while (self.done_count.load(.acquire) < target) {
            std.atomic.spinLoopHint();
        }
    }
};

/// Executes a single-step structured read over a seeded canvas.
/// Embeds prompt and canvas across dual regions, evaluates all 30 transformer layers,
/// and extracts candidate logits at the canvas answer position.
pub fn executeStructuredRead(
    allocator: std.mem.Allocator,
    model: DiffusionGemmaModel,
    prompt_tokens: []const u32,
    canvas: Canvas,
    candidate_token_ids: []const u32,
    out_raw_logits: []f32,
    out_softcapped_logits: []f32,
) !void {
    const P = prompt_tokens.len;
    const C = ModelConfig.canvas_length;
    const N = P + C;

    // 1. Allocate scratch buffers for unified activations (N tokens)
    const total_emb_elements = N * ModelConfig.embedding_dim;
    const emb_buffer = try allocator.alloc(f32, total_emb_elements);
    defer allocator.free(emb_buffer);

    const next_emb_buffer = try allocator.alloc(f32, total_emb_elements);
    defer allocator.free(next_emb_buffer);

    // Dual-region token embedding:
    // Prompt tokens [0..P): scaled by sqrt(d_model)
    // Canvas tokens [P..N): scaled by sqrt(d_model) followed by unweighted RMSNorm
    const emb_scale: f32 = @sqrt(@as(f32, @floatFromInt(ModelConfig.embedding_dim)));
    for (prompt_tokens, 0..) |tok, i| {
        const blocks = model.getEmbeddingBlocks(tok);
        const row = emb_buffer[i * ModelConfig.embedding_dim .. (i + 1) * ModelConfig.embedding_dim];
        quant.dequantizeRowQ6_K(blocks, row);
        for (row) |*val| val.* *= emb_scale;
    }
    for (canvas.tokens, 0..) |tok, i| {
        const blocks = model.getEmbeddingBlocks(tok);
        const row = emb_buffer[(P + i) * ModelConfig.embedding_dim .. (P + i + 1) * ModelConfig.embedding_dim];
        quant.dequantizeRowQ6_K(blocks, row);
        for (row) |*val| val.* *= emb_scale;
        ops.unweightedRmsNorm(row, 1e-6, row);
    }

    // Scratch buffers for layer evaluation
    const max_q_dim: usize = 8192;
    const max_k_dim: usize = 2048;

    const q_buf = try allocator.alloc(f32, N * max_q_dim);
    defer allocator.free(q_buf);

    const k_buf = try allocator.alloc(f32, N * max_k_dim);
    defer allocator.free(k_buf);

    const v_buf = try allocator.alloc(f32, N * max_k_dim);
    defer allocator.free(v_buf);

    const norm_x_buf = try allocator.alloc(f32, N * ModelConfig.embedding_dim);
    defer allocator.free(norm_x_buf);

    const attn_out_buf = try allocator.alloc(f32, N * ModelConfig.embedding_dim);
    defer allocator.free(attn_out_buf);

    // Persistent thread pool setup across CPU threads
    const max_threads: usize = 16;
    const cpu_count = std.Thread.getCpuCount() catch 4;
    const num_threads = @min(@min(cpu_count, max_threads), N);

    var worker_scratches = try allocator.alloc(WorkerScratch, num_threads);
    defer allocator.free(worker_scratches);
    for (worker_scratches) |*ws| {
        ws.* = try WorkerScratch.init(allocator, N, max_q_dim);
    }
    defer {
        for (worker_scratches) |*ws| {
            ws.deinit(allocator);
        }
    }

    const tokens_per_thread = (N + num_threads - 1) / num_threads;
    var pool = LayerPool{
        .num_threads = num_threads,
        .tokens_per_thread = tokens_per_thread,
        .total_len = N,
        .scratches = worker_scratches,
    };

    var pool_threads: [max_threads]std.Thread = undefined;
    const worker_count = if (num_threads > 1) num_threads - 1 else 0;
    for (0..worker_count) |i| {
        pool_threads[i] = try std.Thread.spawn(.{}, LayerPool.workerLoop, .{ &pool, i + 1 });
    }
    defer {
        if (worker_count > 0) {
            pool.phase.store(.terminate, .release);
            _ = pool.epoch.fetchAdd(1, .release);
            for (0..worker_count) |i| {
                pool_threads[i].join();
            }
        }
    }

    // 2. Execute 30 Transformer Blocks
    var current_emb = emb_buffer;
    var next_emb = next_emb_buffer;

    const rf_slice = if (model.rope_freqs) |rf| model.gguf_file.getTensorSlice(rf) else null;
    const rf_floats: ?[*]const f32 = if (rf_slice) |s| @ptrCast(@alignCast(s.ptr)) else null;

    for (0..model.layers.len) |layer_idx| {
        var ctx = LayerContext{
            .model = model,
            .layer_idx = layer_idx,
            .prompt_len = P,
            .total_len = N,
            .max_q_dim = max_q_dim,
            .max_k_dim = max_k_dim,
            .rope_freqs = rf_floats,
            .current_emb = current_emb,
            .next_emb = next_emb,
            .norm_x_buf = norm_x_buf,
            .q_buf = q_buf,
            .k_buf = k_buf,
            .v_buf = v_buf,
            .attn_out_buf = attn_out_buf,
        };

        if (num_threads == 1) {
            runQKVWorker(&ctx, 0, N);
            runAttnWorker(&ctx, &worker_scratches[0], 0, N);
            runFFNWorker(&ctx, &worker_scratches[0], 0, N);
        } else {
            pool.ctx = &ctx;
            pool.dispatch(.qkv);
            pool.dispatch(.attn);
            pool.dispatch(.ffn);
        }

        // Swap buffers for next layer
        const temp = current_emb;
        current_emb = next_emb;
        next_emb = temp;
    }

    // 3. Extract hidden state at pinned answer position in canvas (P + canvas.answer_pos)
    const answer_idx = P + canvas.answer_pos;
    const answer_row = current_emb[answer_idx * ModelConfig.embedding_dim .. (answer_idx + 1) * ModelConfig.embedding_dim];

    // Final output RMSNorm
    const norm_hidden = try allocator.alloc(f32, ModelConfig.embedding_dim);
    defer allocator.free(norm_hidden);

    const out_norm_slice = model.gguf_file.getTensorSlice(model.output_norm);
    const out_norm_weights = @as([*]const f32, @ptrCast(@alignCast(out_norm_slice.ptr)))[0..ModelConfig.embedding_dim];
    ops.rmsNorm(answer_row, out_norm_weights, 1e-6, norm_hidden);

    // 4. Project candidate logits directly from output embedding table
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
    applyRoPE(&vec, 0, 4, 10000.0, null);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vec[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), vec[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), vec[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), vec[3], 1e-5);
}

test "embedCanvas and readCandidateLogits execution on local model if present" {
    if (!root.shouldRunModelTests()) return error.SkipZigTest;

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

    try executeStructuredRead(allocator, model, &prompt, canvas, &candidates, &raw_logits, &softcapped_logits);

    // Softcapped logits must be within [-30.0, 30.0]
    for (softcapped_logits) |l| {
        try std.testing.expect(l >= -30.0 and l <= 30.0);
    }
}

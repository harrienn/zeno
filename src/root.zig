//! Zeno — Zig-based, Jev-style structured decision engine on DiffusionGemma.
//!
//! Root module exposing library types, configurations, constants, and execution primitives.

const std = @import("std");

pub const decision = @import("decision.zig");
pub const quant = @import("quant.zig");
pub const ops = @import("ops.zig");
pub const gguf = @import("gguf.zig");
pub const model = @import("model.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const canvas = @import("canvas.zig");
pub const infer = @import("infer.zig");
pub const engine = @import("engine.zig");
pub const fixture = @import("fixture.zig");
pub const server = @import("server.zig");
pub const build_options = @import("build_options");

/// Returns true if heavy 26B full-model forward pass tests are enabled.
/// Controlled via `zig build test -Dtest-model=true` or `zig build test-model`.
pub fn shouldRunModelTests() bool {
    return build_options.test_model;
}

// Re-export common types
pub const DecisionRequest = decision.DecisionRequest;
pub const DecisionResponse = decision.DecisionResponse;
pub const CandidateSpec = decision.CandidateSpec;
pub const CandidateResult = decision.CandidateResult;
pub const SchemaType = decision.SchemaType;
pub const GgufFile = gguf.GgufFile;
pub const DiffusionGemmaModel = model.DiffusionGemmaModel;
pub const Tokenizer = tokenizer.Tokenizer;
pub const TokenId = tokenizer.TokenId;
pub const Canvas = canvas.Canvas;
pub const DecisionEngine = engine.DecisionEngine;
pub const InferenceWorkspace = infer.InferenceWorkspace;
pub const LayerPool = infer.LayerPool;
pub const GoldenFixture = fixture.GoldenFixture;
pub const Server = server.Server;

/// Semantic version of the Zeno runtime.
pub const version = std.SemanticVersion{
    .major = 0,
    .minor = 1,
    .patch = 0,
};

/// Model architecture constants for DiffusionGemma 26B-A4B.
pub const ModelConfig = struct {
    /// Pinned model canvas length (in tokens). Default is 256.
    pub const canvas_length: usize = 256;
    /// Transformer embedding dimension d_model.
    pub const embedding_dim: usize = 2816;
    /// Total number of transformer layers (30 blocks).
    pub const num_layers: usize = 30;
    /// Total MoE experts per layer.
    pub const num_experts: usize = 128;
    /// Active MoE experts per token routing.
    pub const num_active_experts: usize = 8;
    /// Total vocabulary size.
    pub const vocab_size: usize = 262_144;
    /// Logit softcapping value.
    pub const logit_softcap: f32 = 30.0;
    /// Attention sliding window size (1024 tokens).
    pub const sliding_window: usize = 1024;
    /// Maximum evaluated sequence length (sliding window capacity).
    pub const max_seq_len: usize = sliding_window;
    /// Maximum prompt tokens permitted before canvas overflow.
    pub const max_prompt_tokens: usize = max_seq_len - canvas_length;
};

/// Special token IDs defined by the Gemma 4 / DiffusionGemma tokenizer.
pub const SpecialTokens = struct {
    pub const pad: u32 = 0;
    pub const eos: u32 = 1;
    pub const bos: u32 = 2;
    pub const unk: u32 = 3;
    pub const mask: u32 = 4;
    pub const turn_close: u32 = 106;
};

test {
    _ = decision;
    _ = quant;
    _ = ops;
    _ = gguf;
    _ = model;
    _ = tokenizer;
    _ = canvas;
    _ = infer;
    _ = engine;
    _ = fixture;
}

test "model configuration invariants" {
    try std.testing.expectEqual(@as(usize, 256), ModelConfig.canvas_length);
    try std.testing.expectEqual(@as(usize, 2816), ModelConfig.embedding_dim);
    try std.testing.expectEqual(@as(usize, 30), ModelConfig.num_layers);
    try std.testing.expectEqual(@as(usize, 128), ModelConfig.num_experts);
    try std.testing.expectEqual(@as(usize, 8), ModelConfig.num_active_experts);
    try std.testing.expectEqual(@as(usize, 262_144), ModelConfig.vocab_size);
    try std.testing.expectEqual(@as(f32, 30.0), ModelConfig.logit_softcap);
}

test "special token invariants" {
    try std.testing.expectEqual(@as(u32, 0), SpecialTokens.pad);
    try std.testing.expectEqual(@as(u32, 1), SpecialTokens.eos);
    try std.testing.expectEqual(@as(u32, 2), SpecialTokens.bos);
    try std.testing.expectEqual(@as(u32, 3), SpecialTokens.unk);
    try std.testing.expectEqual(@as(u32, 4), SpecialTokens.mask);
    try std.testing.expectEqual(@as(u32, 106), SpecialTokens.turn_close);
}

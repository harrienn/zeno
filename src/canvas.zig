//! Zeno — Canvas templates, prompt formatting, and seeding with answer-position pinning.
//!
//! Implements the 256-token canvas geometry specified in specs/canvas-templates.md.

const std = @import("std");
const root = @import("root.zig");
const decision = @import("decision.zig");
const tokenizer = @import("tokenizer.zig");

pub const CanvasError = error{
    CanvasOverflow,
    MultiTokenCandidateNotAllowed,
    DuplicateCandidateToken,
    TokenNotFound,
};

/// 256-token canvas buffer holding seeded tokens, answer position, and pinning status.
pub const Canvas = struct {
    tokens: [root.ModelConfig.canvas_length]u32,
    answer_pos: usize,
    is_pinned: bool,
    prompt_len: usize,

    /// Initializes a blank canvas filled with PAD (0) tokens.
    pub fn initEmpty() Canvas {
        return Canvas{
            .tokens = @splat(root.SpecialTokens.pad),
            .answer_pos = 0,
            .is_pinned = false,
            .prompt_len = 0,
        };
    }

    /// Seeds the canvas with answer token at position 0, following TURN_CLOSE at position 1, and PAD tail.
    /// In DiffusionGemma's dual-region architecture, prompt tokens (length P) precede the 256-token canvas.
    pub fn seedSingleStep(prompt_tokens: []const u32, seed_token: u32) CanvasError!Canvas {
        var canvas = Canvas{
            .tokens = @splat(root.SpecialTokens.pad),
            .answer_pos = 0,
            .is_pinned = true,
            .prompt_len = prompt_tokens.len,
        };

        canvas.tokens[0] = seed_token;
        canvas.tokens[1] = root.SpecialTokens.turn_close;

        return canvas;
    }

    /// Formats a complete structured decision prompt conforming to Gemma 4 turn delimiters:
    /// <|turn>user\n{context}\n\nQuestion: {query}\nChoices:\n- {cand}\n<turn|>\n<|turn>model\nAnswer:
    pub fn formatDecisionPrompt(
        allocator: std.mem.Allocator,
        query: []const u8,
        context: ?[]const u8,
        candidates: []const decision.CandidateSpec,
    ) ![]const u8 {
        var list: std.ArrayList(u8) = .empty;
        errdefer list.deinit(allocator);

        // User turn
        try list.appendSlice(allocator, "<|turn>user\n");
        if (context) |ctx| {
            if (ctx.len > 0) {
                try list.appendSlice(allocator, ctx);
                try list.appendSlice(allocator, "\n\n");
            }
        }

        try list.appendSlice(allocator, "Question: ");
        try list.appendSlice(allocator, query);
        try list.appendSlice(allocator, "\nChoices:\n");

        for (candidates) |cand| {
            try list.appendSlice(allocator, "- ");
            try list.appendSlice(allocator, cand.label);
            try list.appendSlice(allocator, "\n");
        }

        try list.appendSlice(allocator, "<turn|>\n<|turn>model\nAnswer:");
        return list.toOwnedSlice(allocator);
    }

    /// Validates all candidate specifications and resolves each candidate to a unique single token ID.
    pub fn validateCandidates(
        tok: tokenizer.Tokenizer,
        candidates: []const decision.CandidateSpec,
        out_token_ids: []u32,
    ) CanvasError!void {
        std.debug.assert(out_token_ids.len >= candidates.len);

        for (candidates, 0..) |cand, i| {
            const token_id = try tok.resolveCandidateToken(cand.token);

            // Check for duplicate token ID among earlier resolved candidates
            for (out_token_ids[0..i]) |prev_id| {
                if (prev_id == token_id) {
                    return error.DuplicateCandidateToken;
                }
            }

            out_token_ids[i] = token_id;
        }
    }
};

test "Canvas: seedSingleStep geometry and padding" {
    const prompt = [_]u32{ 2, 100, 200, 300 }; // 4 tokens: BOS + 3 words
    const seed = root.SpecialTokens.mask; // MASK (4)

    const canvas = try Canvas.seedSingleStep(&prompt, seed);

    try std.testing.expectEqual(@as(usize, 0), canvas.answer_pos);
    try std.testing.expectEqual(@as(usize, 4), canvas.prompt_len);
    try std.testing.expect(canvas.is_pinned);

    try std.testing.expectEqual(seed, canvas.tokens[0]); // Answer slot at canvas position 0
    try std.testing.expectEqual(root.SpecialTokens.turn_close, canvas.tokens[1]);

    // Slots after TURN_CLOSE must be PAD (0)
    for (canvas.tokens[2..]) |tok| {
        try std.testing.expectEqual(@as(u32, root.SpecialTokens.pad), tok);
    }
}

test "Canvas: formatDecisionPrompt includes turn tags and candidates" {
    const allocator = std.testing.allocator;
    const candidates = [_]decision.CandidateSpec{
        .{ .token = "Yes", .label = "Yes" },
        .{ .token = "No", .label = "No" },
    };

    const prompt_text = try Canvas.formatDecisionPrompt(allocator, "Is service active?", "Context: OK", &candidates);
    defer allocator.free(prompt_text);

    try std.testing.expect(std.mem.indexOf(u8, prompt_text, "<|turn>user\nContext: OK\n\nQuestion: Is service active?\nChoices:\n- Yes\n- No\n<turn|>\n<|turn>model\nAnswer:") != null);
}

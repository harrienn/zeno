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
        return seedSingleStepAt(prompt_tokens, seed_token, 0);
    }

    /// Seeds the canvas with answer token at a designated position in [0, canvas_length),
    /// adjacent TURN_CLOSE, and PAD tail.
    pub fn seedSingleStepAt(prompt_tokens: []const u32, seed_token: u32, answer_pos: usize) CanvasError!Canvas {
        if (prompt_tokens.len > root.ModelConfig.max_prompt_tokens) {
            return error.CanvasOverflow;
        }
        if (answer_pos >= root.ModelConfig.canvas_length) {
            return error.CanvasOverflow;
        }

        var canvas = Canvas{
            .tokens = @splat(root.SpecialTokens.pad),
            .answer_pos = answer_pos,
            .is_pinned = true,
            .prompt_len = prompt_tokens.len,
        };

        canvas.tokens[answer_pos] = seed_token;
        if (answer_pos + 1 < root.ModelConfig.canvas_length) {
            canvas.tokens[answer_pos + 1] = root.SpecialTokens.turn_close;
        } else if (answer_pos > 0) {
            canvas.tokens[answer_pos - 1] = root.SpecialTokens.turn_close;
        }

        return canvas;
    }

    /// Preflight calculation of the rendered decision prompt byte length.
    /// Uses overflow-safe arithmetic. Returns null if arithmetic overflows usize.
    pub fn calculatePromptByteLength(
        query: []const u8,
        context: ?[]const u8,
        candidates: []const decision.CandidateSpec,
    ) ?usize {
        var total: usize = 0;
        const boilerplate_len = comptime (
            "<|turn>user\n".len +
            "Question: ".len +
            "\nChoices:\n".len +
            "<turn|>\n<|turn>model\nAnswer:".len
        );
        total = std.math.add(usize, total, boilerplate_len) catch return null;
        total = std.math.add(usize, total, query.len) catch return null;

        if (context) |ctx| {
            if (ctx.len > 0) {
                total = std.math.add(usize, total, ctx.len) catch return null;
                total = std.math.add(usize, total, comptime "\n\n".len) catch return null;
            }
        }

        for (candidates) |cand| {
            total = std.math.add(usize, total, comptime ("- ".len + "\n".len)) catch return null;
            if (!std.mem.eql(u8, cand.token, cand.label)) {
                total = std.math.add(usize, total, cand.token.len) catch return null;
                total = std.math.add(usize, total, comptime ": ".len) catch return null;
            }
            total = std.math.add(usize, total, cand.label.len) catch return null;
        }

        return total;
    }

    /// Formats a complete structured decision prompt conforming to Gemma 4 turn delimiters:
    /// <|turn>user\n{context}\n\nQuestion: {query}\nChoices:\n- {cand}\n<turn|>\n<|turn>model\nAnswer:
    pub fn formatDecisionPrompt(
        allocator: std.mem.Allocator,
        query: []const u8,
        context: ?[]const u8,
        candidates: []const decision.CandidateSpec,
    ) ![]const u8 {
        const preflight_len = calculatePromptByteLength(query, context, candidates) orelse return error.CanvasOverflow;
        if (preflight_len > root.ModelConfig.max_prompt_bytes) {
            return error.CanvasOverflow;
        }

        var list: std.ArrayList(u8) = .empty;
        errdefer list.deinit(allocator);
        try list.ensureTotalCapacity(allocator, preflight_len);

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
            if (!std.mem.eql(u8, cand.token, cand.label)) {
                try list.appendSlice(allocator, cand.token);
                try list.appendSlice(allocator, ": ");
            }
            try list.appendSlice(allocator, cand.label);
            try list.appendSlice(allocator, "\n");
        }

        try list.appendSlice(allocator, "<turn|>\n<|turn>model\nAnswer:");
        std.debug.assert(list.items.len == preflight_len);
        return list.toOwnedSlice(allocator);
    }

    /// Validates all candidate specifications and resolves each candidate to a unique single token ID
    /// by calculating the candidate continuation at the answer boundary of the formatted prompt.
    pub fn validateCandidateContinuations(
        tok: tokenizer.Tokenizer,
        allocator: std.mem.Allocator,
        prompt_text: []const u8,
        prompt_tokens: []const tokenizer.TokenId,
        candidates: []const decision.CandidateSpec,
        out_token_ids: []u32,
    ) CanvasError!void {
        std.debug.assert(out_token_ids.len >= candidates.len);

        for (candidates, 0..) |cand, i| {
            const token_id = tok.resolveCandidateContinuation(
                allocator,
                prompt_text,
                prompt_tokens,
                cand.token,
            ) catch |err| switch (err) {
                error.MultiTokenCandidateNotAllowed => return error.MultiTokenCandidateNotAllowed,
                error.TokenNotFound => return error.TokenNotFound,
            };

            // Check for duplicate token ID among earlier resolved candidates
            for (out_token_ids[0..i]) |prev_id| {
                if (prev_id == token_id) {
                    return error.DuplicateCandidateToken;
                }
            }

            out_token_ids[i] = token_id;
        }
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

test "Canvas: formatDecisionPrompt renders coded choices with token prefix (R56-01)" {
    const allocator = std.testing.allocator;
    const candidates = [_]decision.CandidateSpec{
        .{ .token = "A", .label = "Credit Card Fraud" },
        .{ .token = "B", .label = "Identity Theft" },
        .{ .token = "C", .label = "Account Takeover" },
    };

    const prompt_text = try Canvas.formatDecisionPrompt(allocator, "Classify this fraud report.", null, &candidates);
    defer allocator.free(prompt_text);

    const expected_snippet =
        "Choices:\n" ++
        "- A: Credit Card Fraud\n" ++
        "- B: Identity Theft\n" ++
        "- C: Account Takeover\n" ++
        "<turn|>\n<|turn>model\nAnswer:";

    try std.testing.expect(std.mem.indexOf(u8, prompt_text, expected_snippet) != null);
}

test "Canvas: seedSingleStep rejects prompt exceeding max_prompt_tokens with CanvasOverflow" {
    var long_prompt: [root.ModelConfig.max_prompt_tokens + 1]u32 = @splat(100);
    try std.testing.expectError(error.CanvasOverflow, Canvas.seedSingleStep(&long_prompt, root.SpecialTokens.mask));
}

test "Canvas: calculatePromptByteLength matches formatDecisionPrompt exactly" {
    const allocator = std.testing.allocator;
    const candidates = [_]decision.CandidateSpec{
        .{ .token = "Yes", .label = "Yes Option A" },
        .{ .token = "No", .label = "No Option B" },
    };

    // With context
    const preflight_with_ctx = Canvas.calculatePromptByteLength("Is system ready?", "All systems nominal", &candidates).?;
    const formatted_with_ctx = try Canvas.formatDecisionPrompt(allocator, "Is system ready?", "All systems nominal", &candidates);
    defer allocator.free(formatted_with_ctx);
    try std.testing.expectEqual(formatted_with_ctx.len, preflight_with_ctx);

    // Without context
    const preflight_no_ctx = Canvas.calculatePromptByteLength("Is system ready?", null, &candidates).?;
    const formatted_no_ctx = try Canvas.formatDecisionPrompt(allocator, "Is system ready?", null, &candidates);
    defer allocator.free(formatted_no_ctx);
    try std.testing.expectEqual(formatted_no_ctx.len, preflight_no_ctx);
}

test "Canvas: formatDecisionPrompt rejects oversized candidate labels before allocation (R8-02)" {
    const allocator = std.testing.allocator;
    var huge_label: [root.ModelConfig.max_prompt_bytes + 10]u8 = @splat('A');
    const candidates = [_]decision.CandidateSpec{
        .{ .token = "Yes", .label = &huge_label },
        .{ .token = "No", .label = "No" },
    };

    try std.testing.expectError(error.CanvasOverflow, Canvas.formatDecisionPrompt(allocator, "Short query", null, &candidates));
}

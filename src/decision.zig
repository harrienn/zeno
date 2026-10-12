//! Zeno — Decision API types and validation.
//!
//! Implements the Jev-compatible /v1/systemone contracts defined in specs/decision-api.md.

const std = @import("std");

/// High-level decision schema type.
pub const SchemaType = enum {
    boolean,
    choice,
    score,
};

/// Candidate option specification from caller.
pub const CandidateSpec = struct {
    /// Rendered token text (must resolve to exactly 1 token in the tokenizer).
    token: []const u8,
    /// User-facing display label.
    label: []const u8,
};

/// Evaluated candidate result with token ID, probability, and raw logit.
pub const CandidateResult = struct {
    label: []const u8,
    token_id: u32,
    probability: f32,
    raw_logit: f32,
};

/// Decision request payload.
pub const DecisionRequest = struct {
    /// Query or question to resolve.
    query: []const u8,
    /// Optional background context or application state.
    context: ?[]const u8 = null,
    /// Decision schema type.
    schema_type: SchemaType,
    /// Candidate options (must contain between 2 and 32 items).
    candidates: []const CandidateSpec,
    /// Max diffusion steps. Version 0 only supports 1.
    step_cap: u32 = 1,
    /// Positive, finite logit scaling factor before candidate normalization.
    temperature: f32 = 1.0,
    /// Optional threshold on normalized entropy in [0, 1] for flagging uncertainty.
    entropy_threshold: ?f32 = null,
};

/// Decision response payload.
pub const DecisionResponse = struct {
    /// Display label of the highest scoring candidate.
    decision: []const u8,
    /// Candidate-normalized score for the selected candidate (not an empirical calibration guarantee).
    decision_probability: f32,
    /// Shannon entropy across the candidate set, in nats.
    entropy_nats: f32,
    /// Normalized entropy H(P) / ln(N) in [0, 1].
    normalized_entropy: f32,
    /// True if normalized_entropy >= entropy_threshold.
    uncertain: bool,
    /// Detailed results per candidate.
    candidates: []const CandidateResult,
    /// Number of diffusion/forward steps executed (always 1 in v0).
    steps_executed: u32,
    /// Number of tokens evaluated up to and including the pinned answer token (prompt_len + 1).
    canvas_tokens_used: u32,
};

/// Validation errors for DecisionRequest.
pub const ValidationError = error{
    TooFewCandidates,
    TooManyCandidates,
    DuplicateCandidateToken,
    DuplicateCandidateLabel,
    UnsupportedStepCap,
    InvalidTemperature,
    InvalidEntropyThreshold,
    EmptyQuery,
};

/// Validates a DecisionRequest against the specification constraints.
pub fn validateRequest(req: DecisionRequest) ValidationError!void {
    if (req.query.len == 0) {
        return error.EmptyQuery;
    }
    if (req.candidates.len < 2) {
        return error.TooFewCandidates;
    }
    if (req.candidates.len > 32) {
        return error.TooManyCandidates;
    }
    if (req.step_cap != 1) {
        return error.UnsupportedStepCap;
    }
    if (!std.math.isFinite(req.temperature) or req.temperature <= 0.0) {
        return error.InvalidTemperature;
    }
    if (req.entropy_threshold) |th| {
        if (!std.math.isFinite(th) or th < 0.0 or th > 1.0) {
            return error.InvalidEntropyThreshold;
        }
    }

    // Check for duplicate candidate tokens or labels
    for (req.candidates, 0..) |c1, i| {
        for (req.candidates[i + 1 ..]) |c2| {
            if (std.mem.eql(u8, c1.token, c2.token)) {
                return error.DuplicateCandidateToken;
            }
            if (std.mem.eql(u8, c1.label, c2.label)) {
                return error.DuplicateCandidateLabel;
            }
        }
    }
}

test "validateRequest: valid boolean request" {
    const candidates = [_]CandidateSpec{
        .{ .token = "Yes", .label = "Yes" },
        .{ .token = "No", .label = "No" },
    };
    const req = DecisionRequest{
        .query = "Is this connection alive?",
        .schema_type = .boolean,
        .candidates = &candidates,
        .step_cap = 1,
        .temperature = 1.0,
        .entropy_threshold = 0.5,
    };
    try validateRequest(req);
}

test "validateRequest: reject too few candidates" {
    const candidates = [_]CandidateSpec{
        .{ .token = "Yes", .label = "Yes" },
    };
    const req = DecisionRequest{
        .query = "Is this connection alive?",
        .schema_type = .boolean,
        .candidates = &candidates,
    };
    try std.testing.expectError(error.TooFewCandidates, validateRequest(req));
}

test "validateRequest: reject duplicate tokens" {
    const candidates = [_]CandidateSpec{
        .{ .token = "Yes", .label = "Label1" },
        .{ .token = "Yes", .label = "Label2" },
    };
    const req = DecisionRequest{
        .query = "Test query",
        .schema_type = .choice,
        .candidates = &candidates,
    };
    try std.testing.expectError(error.DuplicateCandidateToken, validateRequest(req));
}

test "validateRequest: reject unsupported step cap" {
    const candidates = [_]CandidateSpec{
        .{ .token = "A", .label = "Option A" },
        .{ .token = "B", .label = "Option B" },
    };
    const req = DecisionRequest{
        .query = "Test query",
        .schema_type = .choice,
        .candidates = &candidates,
        .step_cap = 2,
    };
    try std.testing.expectError(error.UnsupportedStepCap, validateRequest(req));
}

test "validateRequest: reject invalid temperature" {
    const candidates = [_]CandidateSpec{
        .{ .token = "A", .label = "Option A" },
        .{ .token = "B", .label = "Option B" },
    };
    const req = DecisionRequest{
        .query = "Test query",
        .schema_type = .choice,
        .candidates = &candidates,
        .temperature = -0.5,
    };
    try std.testing.expectError(error.InvalidTemperature, validateRequest(req));
}

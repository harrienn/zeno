//! Zeno — Golden fixture parser and local-engine comparison against stored values.
//!
//! Conforms to zeno.fixture/v1, zeno.tolerance/v1, and zeno.tolerance/v2 defined in specs/UNRESOLVED.md.

const std = @import("std");
const root = @import("root.zig");
const decision = @import("decision.zig");
const engine_mod = @import("engine.zig");

pub const FixtureModel = struct {
    path: []const u8,
    sha256: []const u8,
};

pub const FixtureReference = struct {
    repo: []const u8,
    commit: []const u8,
};

pub const FixtureCandidate = struct {
    label: []const u8,
    token: []const u8,
    token_id: u32,
};

pub const FixtureExpected = struct {
    decision: []const u8,
    decision_probability: f32,
    candidate_logits_raw: []const f32,
    candidate_logits_softcapped: []const f32,
    probabilities: []const f32,
    entropy_nats: f32,
    normalized_entropy: f32,
    uncertain: bool,
};

pub const GoldenFixture = struct {
    schema: []const u8,
    policy: []const u8,
    description: []const u8,
    model: FixtureModel,
    reference: FixtureReference,
    query: []const u8,
    context: ?[]const u8 = null,
    schema_type: []const u8,
    canvas_length: usize,
    answer_pos: usize,
    answer_seed_token: u32,
    entropy_threshold: ?f32 = null,
    prompt_tokens: ?[]const i32 = null,
    candidates: []const FixtureCandidate,
    expected: FixtureExpected,
    local_engine_baseline: ?FixtureExpected = null,
};

pub const ParityReport = struct {
    decision_matches: bool,
    max_raw_logit_delta: f32,
    max_softcapped_logit_delta: f32,
    max_prob_delta: f32,
    entropy_delta: f32,
    norm_entropy_delta: f32,
    uncertain_matches: bool,
    meets_tolerances: bool,
};

pub fn evaluateReferenceParity(
    fix: GoldenFixture,
    resp: decision.DecisionResponse,
) ParityReport {
    const dec_match = std.mem.eql(u8, fix.expected.decision, resp.decision);
    const len_match = (fix.expected.candidate_logits_raw.len == resp.candidates.len) and
        (fix.expected.candidate_logits_softcapped.len == resp.candidates.len) and
        (fix.expected.probabilities.len == resp.candidates.len);

    var max_raw: f32 = if (len_match) 0.0 else std.math.inf(f32);
    var max_sc: f32 = if (len_match) 0.0 else std.math.inf(f32);
    var max_prob: f32 = if (len_match) 0.0 else std.math.inf(f32);
    if (len_match) {
        for (fix.expected.candidate_logits_raw, resp.candidates, 0..) |exp_raw, cand, i| {
            const raw_diff = @abs(exp_raw - cand.raw_logit);
            if (raw_diff > max_raw) max_raw = raw_diff;

            const cand_sc = root.ops.logitSoftcap(cand.raw_logit, root.ModelConfig.logit_softcap);
            const sc_diff = @abs(fix.expected.candidate_logits_softcapped[i] - cand_sc);
            if (sc_diff > max_sc) max_sc = sc_diff;

            const prob_diff = @abs(fix.expected.probabilities[i] - cand.probability);
            if (prob_diff > max_prob) max_prob = prob_diff;
        }
    }

    const h_diff = @abs(fix.expected.entropy_nats - resp.entropy_nats);
    const h_norm_diff = @abs(fix.expected.normalized_entropy - resp.normalized_entropy);
    const unc_match = fix.expected.uncertain == resp.uncertain;

    const is_v2 = std.mem.eql(u8, fix.policy, "zeno.tolerance/v2");
    const raw_limit: f32 = if (is_v2) 10.0 else 0.5;
    const sc_limit: f32 = if (is_v2) 7.0 else 0.5;
    const prob_limit: f32 = if (is_v2) 0.15 else 0.05;
    const h_limit: f32 = if (is_v2) 0.35 else 0.05;
    const h_norm_limit: f32 = if (is_v2) 0.25 else 0.05;

    const meets_tol = len_match and dec_match and unc_match and
        (max_raw <= raw_limit) and (max_sc <= sc_limit) and
        (max_prob <= prob_limit) and (h_diff <= h_limit) and (h_norm_diff <= h_norm_limit);

    return .{
        .decision_matches = dec_match,
        .max_raw_logit_delta = max_raw,
        .max_softcapped_logit_delta = max_sc,
        .max_prob_delta = max_prob,
        .entropy_delta = h_diff,
        .norm_entropy_delta = h_norm_diff,
        .uncertain_matches = unc_match,
        .meets_tolerances = meets_tol,
    };
}

pub const FixtureError = error{
    InvalidSchema,
    UnsupportedPolicy,
    CandidateMismatch,
    ParityMismatch,
    PromptTokenMismatch,
};

const posix = std.posix;
const linux = std.os.linux;

/// Loads and parses a golden fixture JSON file from disk.
pub fn loadFixture(allocator: std.mem.Allocator, path: []const u8) !std.json.Parsed(GoldenFixture) {
    const fd = posix.openat(posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0) catch {
        return error.FileNotFound;
    };
    defer _ = linux.close(fd);

    const file_size: usize = @intCast(linux.lseek(fd, 0, linux.SEEK.END));
    _ = linux.lseek(fd, 0, linux.SEEK.SET);

    const content = try allocator.alloc(u8, file_size);
    defer allocator.free(content);

    const bytes_read = try posix.read(fd, content);
    if (bytes_read != file_size) return error.UnexpectedEof;

    const parsed = try std.json.parseFromSlice(GoldenFixture, allocator, content, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    });

    if (!std.mem.eql(u8, parsed.value.schema, "zeno.fixture/v1")) {
        parsed.deinit();
        return error.InvalidSchema;
    }

    if (!std.mem.eql(u8, parsed.value.policy, "zeno.tolerance/v1") and
        !std.mem.eql(u8, parsed.value.policy, "zeno.tolerance/v2"))
    {
        parsed.deinit();
        return error.UnsupportedPolicy;
    }

    return parsed;
}

/// Executes a decision query defined in a fixture and compares the local engine
/// with its stored values. Verifies prompt token sequence integrity, asserts discrete
/// decision agreement with reference expected, and validates regression against local_engine_baseline.
pub fn validateFixture(
    allocator: std.mem.Allocator,
    fixture_path: []const u8,
    engine: *engine_mod.DecisionEngine,
) !void {
    const parsed = try loadFixture(allocator, fixture_path);
    defer parsed.deinit();
    const fix = parsed.value;

    // Build CandidateSpec list
    const candidate_specs = try allocator.alloc(decision.CandidateSpec, fix.candidates.len);
    defer allocator.free(candidate_specs);

    for (fix.candidates, 0..) |cand, i| {
        candidate_specs[i] = .{
            .label = cand.label,
            .token = cand.token,
        };
    }

    const schema_t: decision.SchemaType = if (std.mem.eql(u8, fix.schema_type, "boolean"))
        .boolean
    else if (std.mem.eql(u8, fix.schema_type, "choice"))
        .choice
    else if (std.mem.eql(u8, fix.schema_type, "score"))
        .score
    else
        return error.UnsupportedPolicy;

    const req = decision.DecisionRequest{
        .query = fix.query,
        .context = fix.context,
        .schema_type = schema_t,
        .candidates = candidate_specs,
        .step_cap = 1,
        .temperature = 1.0,
        .entropy_threshold = fix.entropy_threshold,
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    // 1. Verify prompt token integrity if prompt_tokens is recorded
    if (fix.prompt_tokens) |recorded_prompt| {
        const formatted_prompt = try root.Canvas.formatDecisionPrompt(arena.allocator(), fix.query, fix.context, candidate_specs);
        const encoded_tokens = try engine.tokenizer.encode(arena.allocator(), formatted_prompt);
        try std.testing.expectEqual(recorded_prompt.len, encoded_tokens.len + 1);
        try std.testing.expectEqual(recorded_prompt[0], @as(i32, @intCast(root.SpecialTokens.bos)));
        for (encoded_tokens, 0..) |tok, i| {
            try std.testing.expectEqual(recorded_prompt[i + 1], @as(i32, @intCast(tok)));
        }
    }

    const resp = try engine.decide(arena.allocator(), req);

    // 2. Assert exact discrete decision string match against reference expected
    try std.testing.expectEqualStrings(fix.expected.decision, resp.decision);

    // 3. Assert regression against local_engine_baseline
    const baseline = fix.local_engine_baseline orelse fix.expected;
    try std.testing.expectEqualStrings(baseline.decision, resp.decision);
    try std.testing.expectApproxEqAbs(baseline.decision_probability, resp.decision_probability, 0.05);
    try std.testing.expectEqual(baseline.uncertain, resp.uncertain);
    try std.testing.expectEqual(baseline.candidate_logits_raw.len, resp.candidates.len);
    try std.testing.expectEqual(baseline.candidate_logits_softcapped.len, resp.candidates.len);
    try std.testing.expectEqual(baseline.probabilities.len, resp.candidates.len);

    for (baseline.candidate_logits_raw, resp.candidates, 0..) |expected_raw, actual_cand, i| {
        try std.testing.expectApproxEqAbs(expected_raw, actual_cand.raw_logit, 0.5);
        const actual_softcapped = root.ops.logitSoftcap(actual_cand.raw_logit, root.ModelConfig.logit_softcap);
        try std.testing.expectApproxEqAbs(baseline.candidate_logits_softcapped[i], actual_softcapped, 0.5);
        try std.testing.expectApproxEqAbs(baseline.probabilities[i], actual_cand.probability, 0.05);
    }

    try std.testing.expectApproxEqAbs(baseline.entropy_nats, resp.entropy_nats, 0.05);
    try std.testing.expectApproxEqAbs(baseline.normalized_entropy, resp.normalized_entropy, 0.05);

    try std.testing.expectEqual(fix.canvas_length, root.ModelConfig.canvas_length);
    try std.testing.expectEqual(@as(usize, 0), fix.answer_pos);
    try std.testing.expectEqual(root.SpecialTokens.mask, fix.answer_seed_token);
    if (fix.prompt_tokens) |recorded_prompt| {
        try std.testing.expectEqual(@as(u32, @intCast(recorded_prompt.len + fix.answer_pos + 1)), resp.canvas_tokens_used);
    } else {
        try std.testing.expect(fix.answer_pos + 1 <= resp.canvas_tokens_used);
    }
}

test "loadFixture: parses zeno.fixture/v1 correctly" {
    const allocator = std.testing.allocator;
    const fixture_path = "specs/fixtures/system_health_boolean.json";

    const parsed = loadFixture(allocator, fixture_path) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer parsed.deinit();

    try std.testing.expectEqualStrings("zeno.fixture/v1", parsed.value.schema);
    try std.testing.expectEqualStrings("zeno.tolerance/v2", parsed.value.policy);
    try std.testing.expectEqualStrings("Yes", parsed.value.expected.decision);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.candidates.len);

    const mc_parsed = loadFixture(allocator, "specs/fixtures/incident_severity_multiclass.json") catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer mc_parsed.deinit();

    try std.testing.expectEqualStrings("zeno.fixture/v1", mc_parsed.value.schema);
    try std.testing.expectEqualStrings("zeno.tolerance/v2", mc_parsed.value.policy);
    try std.testing.expectEqualStrings("Medium", mc_parsed.value.expected.decision);
    try std.testing.expectEqual(@as(usize, 3), mc_parsed.value.candidates.len);

    const boundary_parsed = loadFixture(allocator, "specs/fixtures/boundary_256_incident_triage.json") catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer boundary_parsed.deinit();

    try std.testing.expectEqualStrings("zeno.fixture/v1", boundary_parsed.value.schema);
    try std.testing.expectEqualStrings("zeno.tolerance/v2", boundary_parsed.value.policy);
    try std.testing.expectEqualStrings("No", boundary_parsed.value.expected.decision);
    try std.testing.expectEqual(@as(usize, 2), boundary_parsed.value.candidates.len);

    const uncertain_parsed = loadFixture(allocator, "specs/fixtures/maintenance_priority_uncertain.json") catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer uncertain_parsed.deinit();

    try std.testing.expectEqualStrings("zeno.fixture/v1", uncertain_parsed.value.schema);
    try std.testing.expectEqualStrings("zeno.tolerance/v2", uncertain_parsed.value.policy);
    try std.testing.expectEqualStrings("Low", uncertain_parsed.value.expected.decision);
    try std.testing.expectEqual(true, uncertain_parsed.value.expected.uncertain);
    try std.testing.expectEqual(@as(usize, 3), uncertain_parsed.value.candidates.len);

    const score_parsed = loadFixture(allocator, "specs/fixtures/satisfaction_score_5class.json") catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer score_parsed.deinit();

    try std.testing.expectEqualStrings("zeno.fixture/v1", score_parsed.value.schema);
    try std.testing.expectEqualStrings("zeno.tolerance/v2", score_parsed.value.policy);
    try std.testing.expectEqualStrings("5", score_parsed.value.expected.decision);
    try std.testing.expectEqual(false, score_parsed.value.expected.uncertain);
    try std.testing.expectEqual(@as(usize, 5), score_parsed.value.candidates.len);
}

test "validateFixture: compares local model with stored fixture values if present" {
    if (!root.shouldRunModelTests()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var engine = engine_mod.DecisionEngine.init(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer engine.deinit() catch {};

    try validateFixture(allocator, "specs/fixtures/system_health_boolean.json", &engine);
    try validateFixture(allocator, "specs/fixtures/incident_severity_multiclass.json", &engine);
    try validateFixture(allocator, "specs/fixtures/boundary_256_incident_triage.json", &engine);
    try validateFixture(allocator, "specs/fixtures/maintenance_priority_uncertain.json", &engine);
    try validateFixture(allocator, "specs/fixtures/satisfaction_score_5class.json", &engine);
}

test "validateFixture: compares CUDA model with stored fixture values if present" {
    if (!root.shouldRunModelTests()) return error.SkipZigTest;
    if (!root.cuda.is_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var engine = engine_mod.DecisionEngine.initWithOptions(allocator, model_path, .{ .backend = .cuda }) catch |err| {
        if (err == error.FileNotFound or err == error.NoDevice) return;
        return err;
    };
    defer engine.deinit() catch {};

    try validateFixture(allocator, "specs/fixtures/system_health_boolean.json", &engine);
    try validateFixture(allocator, "specs/fixtures/incident_severity_multiclass.json", &engine);
    try validateFixture(allocator, "specs/fixtures/boundary_256_incident_triage.json", &engine);
    try validateFixture(allocator, "specs/fixtures/maintenance_priority_uncertain.json", &engine);
    try validateFixture(allocator, "specs/fixtures/satisfaction_score_5class.json", &engine);
}

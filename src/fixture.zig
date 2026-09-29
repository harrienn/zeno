//! Zeno — Golden fixture parser and parity validation against pinned references.
//!
//! Conforms to zeno.fixture/v1 and zeno.tolerance/v1 defined in specs/UNRESOLVED.md.

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
    candidates: []const FixtureCandidate,
    expected: FixtureExpected,
};

pub const FixtureError = error{
    InvalidSchema,
    UnsupportedPolicy,
    CandidateMismatch,
    ParityMismatch,
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

    if (!std.mem.eql(u8, parsed.value.policy, "zeno.tolerance/v1")) {
        parsed.deinit();
        return error.UnsupportedPolicy;
    }

    return parsed;
}

/// Executes a decision query defined in a golden fixture and asserts numerical
/// parity according to the zeno.tolerance/v1 specification in specs/UNRESOLVED.md §6.
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

    const req = decision.DecisionRequest{
        .query = fix.query,
        .context = fix.context,
        .schema_type = .boolean,
        .candidates = candidate_specs,
        .step_cap = 1,
        .temperature = 1.0,
        .entropy_threshold = 0.5,
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const resp = try engine.decide(arena.allocator(), req);

    // Assert exact decision string match
    try std.testing.expectEqualStrings(fix.expected.decision, resp.decision);

    // Assert decision probability within tolerance 0.05
    try std.testing.expectApproxEqAbs(fix.expected.decision_probability, resp.decision_probability, 0.05);

    // Assert exact uncertainty boolean match
    try std.testing.expectEqual(fix.expected.uncertain, resp.uncertain);

    // Assert candidates length match
    try std.testing.expectEqual(fix.expected.candidate_logits_raw.len, resp.candidates.len);

    // Assert per-candidate raw logits and probabilities within tolerances
    for (fix.expected.candidate_logits_raw, resp.candidates, 0..) |expected_raw, actual_cand, i| {
        try std.testing.expectApproxEqAbs(expected_raw, actual_cand.raw_logit, 0.5);
        try std.testing.expectApproxEqAbs(fix.expected.probabilities[i], actual_cand.probability, 0.05);
    }

    // Assert entropy metrics within tolerance 0.05
    try std.testing.expectApproxEqAbs(fix.expected.entropy_nats, resp.entropy_nats, 0.05);
    try std.testing.expectApproxEqAbs(fix.expected.normalized_entropy, resp.normalized_entropy, 0.05);
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
    try std.testing.expectEqualStrings("zeno.tolerance/v1", parsed.value.policy);
    try std.testing.expectEqualStrings("No", parsed.value.expected.decision);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.candidates.len);
}

test "validateFixture: verifies parity on local model if present" {
    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";
    const fixture_path = "specs/fixtures/system_health_boolean.json";

    var engine = engine_mod.DecisionEngine.init(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer engine.deinit();

    try validateFixture(allocator, fixture_path, &engine);
}

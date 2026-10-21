//! Zeno — Labeled dataset loader and result serializer for empirical calibration.
//!
//! Conforms to zeno.dataset/v1 schema.

const std = @import("std");
const decision = @import("decision.zig");
const root = @import("root.zig");

pub const DatasetCandidate = struct {
    token: []const u8,
    label: []const u8,
};

pub const DatasetExample = struct {
    id: []const u8,
    category: []const u8,
    query: []const u8,
    context: ?[]const u8 = null,
    schema_type: []const u8,
    candidates: []const DatasetCandidate,
    ground_truth: []const u8,
};

pub const BenchmarkDataset = struct {
    schema: []const u8,
    name: []const u8,
    description: []const u8,
    examples: []const DatasetExample,
};

pub const EvaluationResult = struct {
    id: []const u8,
    category: []const u8,
    ground_truth: []const u8,
    decision: []const u8,
    is_correct: bool,
    decision_probability: f32,
    entropy_nats: f32,
    normalized_entropy: f32,
    uncertain: bool,
    steps_executed: u32,
    latency_ms: f64,
    candidates: []const decision.CandidateResult,
    stage_timings: ?decision.StageTimings = null,
};

pub const EvaluationSummary = struct {
    total_examples: usize,
    correct_count: usize,
    accuracy: f64,
    mean_latency_ms: f64,
    median_latency_ms: f64,
    mean_normalized_entropy: f64,
    step_counts: [5]usize, // indices 1..4 for steps_executed
    uncertain_count: usize,
};

/// Loads and parses a benchmark dataset JSON file from disk.
pub fn loadDataset(allocator: std.mem.Allocator, path: []const u8) !std.json.Parsed(BenchmarkDataset) {
    const fd = std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0) catch {
        return error.FileNotFound;
    };
    defer _ = std.os.linux.close(fd);

    const file_size: usize = @intCast(std.os.linux.lseek(fd, 0, std.os.linux.SEEK.END));
    _ = std.os.linux.lseek(fd, 0, std.os.linux.SEEK.SET);

    const content = try allocator.alloc(u8, file_size);
    defer allocator.free(content);

    const bytes_read = try std.posix.read(fd, content);
    if (bytes_read != file_size) return error.UnexpectedEof;

    return try std.json.parseFromSlice(BenchmarkDataset, allocator, content, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    });
}

pub fn escapeJsonString(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var list: std.ArrayList(u8) = .empty;
    for (input) |byte| {
        switch (byte) {
            '"' => try list.appendSlice(allocator, "\\\""),
            '\\' => try list.appendSlice(allocator, "\\\\"),
            '\n' => try list.appendSlice(allocator, "\\n"),
            '\r' => try list.appendSlice(allocator, "\\r"),
            '\t' => try list.appendSlice(allocator, "\\t"),
            0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => {
                const hex = try std.fmt.allocPrint(allocator, "\\u00{x:0>2}", .{byte});
                defer allocator.free(hex);
                try list.appendSlice(allocator, hex);
            },
            else => try list.append(allocator, byte),
        }
    }
    return list.toOwnedSlice(allocator);
}

/// Serializes evaluation results and summary statistics to a structured JSON file.
pub fn writeEvaluationResultsJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    backend: []const u8,
    step_cap: u32,
    threshold: ?f32,
    temperature: f32,
    canvas_len: usize,
    results: []const EvaluationResult,
    summary: EvaluationSummary,
) !void {
    var json_list: std.ArrayList(u8) = .empty;

    const th_str = if (threshold) |t| try std.fmt.allocPrint(allocator, "{d:.4}", .{t}) else "null";

    const header = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "schema": "zeno.evaluation/v1",
        \\  "config": {{
        \\    "backend": "{s}",
        \\    "step_cap": {d},
        \\    "entropy_threshold": {s},
        \\    "temperature": {d:.4},
        \\    "canvas_length": {d}
        \\  }},
        \\  "summary": {{
        \\    "total_examples": {d},
        \\    "correct_count": {d},
        \\    "accuracy": {d:.4},
        \\    "mean_latency_ms": {d:.2},
        \\    "median_latency_ms": {d:.2},
        \\    "mean_normalized_entropy": {d:.4},
        \\    "uncertain_count": {d},
        \\    "step_distribution": {{
        \\      "step_1": {d},
        \\      "step_2": {d},
        \\      "step_3": {d},
        \\      "step_4": {d}
        \\    }}
        \\  }},
        \\  "results": [
        \\
    , .{
        backend,
        step_cap,
        th_str,
        temperature,
        canvas_len,
        summary.total_examples,
        summary.correct_count,
        summary.accuracy,
        summary.mean_latency_ms,
        summary.median_latency_ms,
        summary.mean_normalized_entropy,
        summary.uncertain_count,
        summary.step_counts[1],
        summary.step_counts[2],
        summary.step_counts[3],
        summary.step_counts[4],
    });
    try json_list.appendSlice(allocator, header);

    for (results, 0..) |res, r_i| {
        const comma_res = if (r_i + 1 < results.len) "," else "";
        const esc_id = try escapeJsonString(allocator, res.id);
        const esc_cat = try escapeJsonString(allocator, res.category);
        const esc_gt = try escapeJsonString(allocator, res.ground_truth);
        const esc_dec = try escapeJsonString(allocator, res.decision);

        const item_hdr = try std.fmt.allocPrint(allocator,
            \\    {{
            \\      "id": "{s}",
            \\      "category": "{s}",
            \\      "ground_truth": "{s}",
            \\      "decision": "{s}",
            \\      "is_correct": {},
            \\      "decision_probability": {d:.6},
            \\      "entropy_nats": {d:.6},
            \\      "normalized_entropy": {d:.6},
            \\      "uncertain": {},
            \\      "steps_executed": {d},
            \\      "latency_ms": {d:.2},
            \\      "candidates": [
            \\
        , .{
            esc_id,
            esc_cat,
            esc_gt,
            esc_dec,
            res.is_correct,
            res.decision_probability,
            res.entropy_nats,
            res.normalized_entropy,
            res.uncertain,
            res.steps_executed,
            res.latency_ms,
        });
        try json_list.appendSlice(allocator, item_hdr);

        for (res.candidates, 0..) |c, c_i| {
            const sc = root.ops.logitSoftcap(c.raw_logit, root.ModelConfig.logit_softcap);
            const comma_c = if (c_i + 1 < res.candidates.len) "," else "";
            const esc_lbl = try escapeJsonString(allocator, c.label);
            const cand_str = try std.fmt.allocPrint(allocator,
                \\        {{
                \\          "label": "{s}",
                \\          "token_id": {d},
                \\          "raw_logit": {d:.6},
                \\          "softcapped_logit": {d:.6},
                \\          "probability": {d:.6}
                \\        }}{s}
                \\
            , .{
                esc_lbl,
                c.token_id,
                c.raw_logit,
                sc,
                c.probability,
                comma_c,
            });
            try json_list.appendSlice(allocator, cand_str);
        }

        if (res.stage_timings) |st| {
            const tim_str = try std.fmt.allocPrint(allocator,
                \\      ],
                \\      "stage_timings": {{
                \\        "prompt_prefill_ms": {d:.3},
                \\        "prompt_p0_ms": {d:.3},
                \\        "prompt_transfer_ms": {d:.3},
                \\        "prompt_p1_ms": {d:.3},
                \\        "canvas_step1_ms": {d:.3},
                \\        "canvas_step2_ms": {d:.3},
                \\        "total_engine_ms": {d:.3},
                \\        "prompt_cache_hit": {}
                \\      }}
                \\    }}{s}
                \\
            , .{
                st.prompt_prefill_ms,
                st.prompt_p0_ms,
                st.prompt_transfer_ms,
                st.prompt_p1_ms,
                st.canvas_step1_ms,
                st.canvas_step2_ms,
                st.total_engine_ms,
                st.prompt_cache_hit,
                comma_res,
            });
            try json_list.appendSlice(allocator, tim_str);
        } else {
            const item_ftr = try std.fmt.allocPrint(allocator,
                \\      ]
                \\    }}{s}
                \\
            , .{comma_res});
            try json_list.appendSlice(allocator, item_ftr);
        }
    }

    try json_list.appendSlice(allocator, "  ]\n}\n");

    const out_fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    defer _ = std.os.linux.close(out_fd);
    var written: usize = 0;
    while (written < json_list.items.len) {
        const rc = std.os.linux.write(out_fd, json_list.items.ptr + written, json_list.items.len - written);
        const signed_rc: isize = @bitCast(rc);
        if (signed_rc < 0) return error.WriteFailed;
        if (rc == 0) return error.WriteFailed;
        written += rc;
    }
}

test "dataset loader validates 50-example benchmark" {
    const allocator = std.testing.allocator;
    const parsed = try loadDataset(allocator, "specs/fixtures/calibration_benchmark.json");
    defer parsed.deinit();

    try std.testing.expectEqualStrings("zeno.dataset/v1", parsed.value.schema);
    try std.testing.expectEqual(@as(usize, 50), parsed.value.examples.len);

    var bool_count: usize = 0;
    var choice_count: usize = 0;
    var score_count: usize = 0;

    for (parsed.value.examples) |ex| {
        try std.testing.expect(ex.id.len > 0);
        try std.testing.expect(ex.query.len > 0);
        try std.testing.expect(ex.candidates.len >= 2);
        try std.testing.expect(ex.ground_truth.len > 0);

        // Verify ground_truth exists in candidates
        var found = false;
        for (ex.candidates) |c| {
            if (std.mem.eql(u8, c.label, ex.ground_truth)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);

        if (std.mem.eql(u8, ex.schema_type, "boolean")) {
            bool_count += 1;
        } else if (std.mem.eql(u8, ex.schema_type, "choice")) {
            choice_count += 1;
        } else if (std.mem.eql(u8, ex.schema_type, "score")) {
            score_count += 1;
        }
    }

    try std.testing.expectEqual(@as(usize, 20), bool_count);
    try std.testing.expectEqual(@as(usize, 15), choice_count);
    try std.testing.expectEqual(@as(usize, 15), score_count);
}

const std = @import("std");
const Io = std.Io;
const zeno = @import("zeno");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena: std.mem.Allocator = init.arena.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    try stdout_writer.print("Zeno v{d}.{d}.{d} — Jev-style structured decision engine on DiffusionGemma\n", .{
        zeno.version.major,
        zeno.version.minor,
        zeno.version.patch,
    });

    const args = try init.minimal.args.toSlice(arena);
    if (args.len > 1 and std.mem.eql(u8, args[1], "serve")) {
        if (args.len < 3) {
            try stdout_writer.print("Usage: zeno serve <path/to/model.gguf> [--host <ip>] [--port <port>]\n", .{});
            try stdout_writer.flush();
            return;
        }

        const model_path = args[2];
        var host: []const u8 = "127.0.0.1";
        var port: u16 = 8000;

        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--host") and i + 1 < args.len) {
                i += 1;
                host = args[i];
            } else if (std.mem.eql(u8, args[i], "--port") and i + 1 < args.len) {
                i += 1;
                port = std.fmt.parseInt(u16, args[i], 10) catch 8000;
            }
        }

        try stdout_writer.print("Initializing decision engine with model: {s}...\n", .{model_path});
        try stdout_writer.flush();

        var engine = zeno.DecisionEngine.init(arena, model_path) catch |err| {
            try stdout_writer.print("Error initializing engine: {}\n", .{err});
            try stdout_writer.flush();
            return;
        };
        defer engine.deinit();

        var server_inst = zeno.Server.init(arena, &engine, host, port);
        try server_inst.listen();
        defer server_inst.deinit();

        try stdout_writer.print("Zeno System One Decision API server listening on http://{s}:{d}\n", .{ host, port });
        try stdout_writer.print("Endpoints:\n", .{});
        try stdout_writer.print("  GET  /health           - Liveness probe\n", .{});
        try stdout_writer.print("  POST /v1/systemone     - Jev-compatible decision API\n\n", .{});
        try stdout_writer.flush();

        try server_inst.serve();
        return;
    } else if (args.len > 1 and std.mem.eql(u8, args[1], "decide")) {
        if (args.len < 4) {
            try stdout_writer.print("Usage: zeno decide <path/to/model.gguf> <query> [cand1] [cand2] ...\n", .{});
            try stdout_writer.flush();
            return;
        }

        const model_path = args[2];
        const query = args[3];

        const candidate_specs = try parseCandidateSpecs(arena, if (args.len > 4) args[4..] else &.{});

        try stdout_writer.print("Initializing decision engine with model: {s}...\n", .{model_path});
        try stdout_writer.flush();

        var engine = zeno.DecisionEngine.init(arena, model_path) catch |err| {
            try stdout_writer.print("Error initializing engine: {}\n", .{err});
            try stdout_writer.flush();
            return;
        };
        defer engine.deinit();

        const req = zeno.DecisionRequest{
            .query = query,
            .schema_type = .boolean,
            .candidates = candidate_specs,
            .step_cap = 1,
            .temperature = 1.0,
            .entropy_threshold = 0.5,
        };

        const resp = engine.decide(arena, req) catch |err| {
            try stdout_writer.print("Decision evaluation error: {}\n", .{err});
            try stdout_writer.flush();
            return;
        };

        try stdout_writer.print("\n--- Decision Result ---\n", .{});
        try stdout_writer.print("Query:               \"{s}\"\n", .{query});
        try stdout_writer.print("Decision:            {s}\n", .{resp.decision});
        try stdout_writer.print("Decision Prob:       {d:.4}\n", .{resp.decision_probability});
        try stdout_writer.print("Entropy (nats):      {d:.4}\n", .{resp.entropy_nats});
        try stdout_writer.print("Normalized Entropy:  {d:.4}\n", .{resp.normalized_entropy});
        try stdout_writer.print("Uncertain:           {}\n", .{resp.uncertain});
        try stdout_writer.print("Steps:               {d}\n", .{resp.steps_executed});
        try stdout_writer.print("Canvas Tokens:       {d}\n", .{resp.canvas_tokens_used});
        try stdout_writer.print("Candidates:\n", .{});
        for (resp.candidates) |c| {
            try stdout_writer.print("  - {s} (ID {d}): prob={d:.4}, raw_logit={d:.4}\n", .{
                c.label,
                c.token_id,
                c.probability,
                c.raw_logit,
            });
        }
    } else if (args.len > 1 and (std.mem.eql(u8, args[1], "metadata") or std.mem.eql(u8, args[1], "meta"))) {
        if (args.len < 3) {
            try stdout_writer.print("Usage: zeno metadata <path/to/model.gguf>\n", .{});
            try stdout_writer.flush();
            return;
        }
        const model_path = args[2];
        var model = zeno.DiffusionGemmaModel.loadFromFile(arena, model_path) catch |err| {
            try stdout_writer.print("Error loading model: {}\n", .{err});
            try stdout_writer.flush();
            return;
        };
        defer model.deinit();

        try stdout_writer.print("GGUF Metadata Entries ({d} keys):\n", .{model.gguf_file.metadata.len});
        for (model.gguf_file.metadata) |entry| {
            try stdout_writer.print("  - {s} ({s}): ", .{ entry.key, @tagName(entry.value_type) });
            switch (entry.value_type) {
                .uint32 => try stdout_writer.print("{d}\n", .{model.gguf_file.getMetadataU32(entry.key) orelse 0}),
                .float32 => try stdout_writer.print("{d}\n", .{model.gguf_file.getMetadataF32(entry.key) orelse 0.0}),
                .bool_val => try stdout_writer.print("{}\n", .{model.gguf_file.getMetadataBool(entry.key) orelse false}),
                .string => {
                    const s = model.gguf_file.getMetadataString(entry.key) orelse "";
                    if (s.len > 120) {
                        try stdout_writer.print("\"{s}...\" (len={d})\n", .{ s[0..120], s.len });
                    } else {
                        try stdout_writer.print("\"{s}\"\n", .{s});
                    }
                },
                .array => try stdout_writer.print("<array>\n", .{}),
                else => try stdout_writer.print("<other>\n", .{}),
            }
        }
        try stdout_writer.flush();
        return;
    } else if (args.len > 1 and std.mem.eql(u8, args[1], "dump-eval")) {
        if (args.len < 6) {
            try stdout_writer.print("Usage: zeno dump-eval <model.gguf> <query> <prompt_out.i32> <canvas_out.i32>\n", .{});
            try stdout_writer.flush();
            return;
        }
        const model_path = args[2];
        const query = args[3];
        const prompt_path = args[4];
        const canvas_path = args[5];

        var engine = try zeno.DecisionEngine.init(arena, model_path);
        defer engine.deinit();

        const candidate_specs = try parseCandidateSpecs(arena, if (args.len > 6) args[6..] else &.{});
        const formatted = try zeno.Canvas.formatDecisionPrompt(arena, query, null, candidate_specs);
        const prompt_tokens = try encodePromptWithBOS(arena, engine.tokenizer, formatted);

        try writeTokensToFile(prompt_path, prompt_tokens);

        const canvas_tokens = makeCanvasTokens(zeno.SpecialTokens.mask);
        try writeTokensToFile(canvas_path, &canvas_tokens);

        try stdout_writer.print("Exported {d} prompt tokens to {s} and 256 canvas tokens to {s}\n", .{
            prompt_tokens.len,
            prompt_path,
            canvas_path,
        });
        try stdout_writer.flush();
        return;
    } else if (args.len > 1 and std.mem.eql(u8, args[1], "dump-fixture")) {
        if (args.len < 6) {
            try stdout_writer.print("Usage: zeno dump-fixture <model.gguf> <fixture.json> <prompt_out.i32> <canvas_out.i32>\n", .{});
            try stdout_writer.flush();
            return;
        }
        const model_path = args[2];
        const fixture_path = args[3];
        const prompt_path = args[4];
        const canvas_path = args[5];

        const parsed = try zeno.fixture.loadFixture(arena, fixture_path);
        defer parsed.deinit();
        const fix = parsed.value;

        var engine = try zeno.DecisionEngine.init(arena, model_path);
        defer engine.deinit();

        const candidate_specs = try candidateSpecsFromFixture(arena, fix);
        const formatted = try zeno.Canvas.formatDecisionPrompt(arena, fix.query, fix.context, candidate_specs);
        const prompt_tokens = try encodePromptWithBOS(arena, engine.tokenizer, formatted);

        // Verify freshly formatted tokens against fixture stored tokens
        if (fix.prompt_tokens) |expected_prompt| {
            if (expected_prompt.len != prompt_tokens.len) {
                try stdout_writer.print("Error: Prompt token length mismatch for '{s}': expected {d}, got {d}\n", .{
                    fixture_path,
                    expected_prompt.len,
                    prompt_tokens.len,
                });
                try stdout_writer.flush();
                return error.PromptTokenMismatch;
            }
            for (expected_prompt, prompt_tokens, 0..) |exp_tok, act_tok, idx| {
                if (exp_tok != act_tok) {
                    try stdout_writer.print("Error: Prompt token mismatch at index {d} for '{s}': expected {d}, got {d}\n", .{
                        idx,
                        fixture_path,
                        exp_tok,
                        act_tok,
                    });
                    try stdout_writer.flush();
                    return error.PromptTokenMismatch;
                }
            }
        }

        try writeTokensToFile(prompt_path, prompt_tokens);

        const canvas_tokens = makeCanvasTokens(fix.answer_seed_token);
        try writeTokensToFile(canvas_path, &canvas_tokens);

        try stdout_writer.print("Exported fixture '{s}': {d} prompt tokens (verified matching) to {s} and 256 canvas tokens to {s}\n", .{
            fixture_path,
            prompt_tokens.len,
            prompt_path,
            canvas_path,
        });
        try stdout_writer.flush();
        return;
    } else if (args.len > 1 and std.mem.eql(u8, args[1], "tokenize")) {
        if (args.len < 4) {
            try stdout_writer.print("Usage: zeno tokenize <model.gguf> <text>\n", .{});
            try stdout_writer.flush();
            return;
        }
        const model_path = args[2];
        const text = args[3];

        var model = try zeno.DiffusionGemmaModel.loadFromFile(arena, model_path);
        defer model.deinit();
        var tokenizer = try zeno.Tokenizer.initFromGguf(arena, model.gguf_file);
        defer tokenizer.deinit(arena);

        const tokens = try tokenizer.encode(arena, text);
        try stdout_writer.print("Encoded {d} tokens for \"{s}\":\n", .{ tokens.len, text });
        for (tokens, 0..) |tok, i| {
            const piece = tokenizer.decode(tok) orelse "<unk>";
            try stdout_writer.print("  [{d}] {d} (\"{s}\")\n", .{ i, tok, piece });
        }

        if (tokenizer.resolveCandidateToken(text)) |cand_id| {
            const piece = tokenizer.decode(cand_id) orelse "<unk>";
            try stdout_writer.print("Candidate resolution: ID {d} (\"{s}\")\n", .{ cand_id, piece });
        } else |err| {
            try stdout_writer.print("Candidate resolution failed: {}\n", .{err});
        }
        try stdout_writer.flush();
        return;
    } else if (args.len > 1 and std.mem.eql(u8, args[1], "decide-fixture")) {
        if (args.len < 4) {
            try stdout_writer.print("Usage: zeno decide-fixture <model.gguf> <fixture.json> [out.json]\n", .{});
            try stdout_writer.flush();
            return;
        }
        const model_path = args[2];
        const fixture_path = args[3];

        const parsed = try zeno.fixture.loadFixture(arena, fixture_path);
        defer parsed.deinit();
        const fix = parsed.value;

        var engine = try zeno.DecisionEngine.init(arena, model_path);
        defer engine.deinit();

        const candidate_specs = try candidateSpecsFromFixture(arena, fix);
        const schema_t = try parseSchemaType(fix.schema_type);

        const req = zeno.DecisionRequest{
            .query = fix.query,
            .context = fix.context,
            .schema_type = schema_t,
            .candidates = candidate_specs,
            .step_cap = 1,
            .temperature = 1.0,
            .entropy_threshold = fix.entropy_threshold,
        };

        const resp = try engine.decide(arena, req);

        try stdout_writer.print("Decision:           {s}\n", .{resp.decision});
        try stdout_writer.print("Decision Prob:      {d:.4}\n", .{resp.decision_probability});
        try stdout_writer.print("Entropy (nats):     {d:.4}\n", .{resp.entropy_nats});
        try stdout_writer.print("Normalized Entropy: {d:.4}\n", .{resp.normalized_entropy});
        try stdout_writer.print("Uncertain:          {}\n", .{resp.uncertain});
        try stdout_writer.print("Candidates:\n", .{});
        for (resp.candidates) |c| {
            const sc = zeno.ops.logitSoftcap(c.raw_logit, zeno.ModelConfig.logit_softcap);
            try stdout_writer.print("  - {s} (ID {d}): raw={d:.4}, sc={d:.4}, prob={d:.4}\n", .{
                c.label,
                c.token_id,
                c.raw_logit,
                sc,
                c.probability,
            });
        }
        try stdout_writer.flush();

        if (args.len > 4) {
            try writeDecisionResponseJson(arena, args[4], resp);
        }
        return;
    } else if (args.len > 1) {
        var model_path: []const u8 = args[1];
        if (std.mem.eql(u8, args[1], "info") and args.len > 2) {
            model_path = args[2];
        }

        try stdout_writer.print("\nLoading and inspecting GGUF model: {s}...\n", .{model_path});
        try stdout_writer.flush();

        var model = zeno.DiffusionGemmaModel.loadFromFile(arena, model_path) catch |err| {
            try stdout_writer.print("Error loading model: {}\n", .{err});
            try stdout_writer.flush();
            return;
        };
        defer model.deinit();

        const arch = model.gguf_file.getMetadataString("general.architecture") orelse "unknown";
        const name = model.gguf_file.getMetadataString("general.name") orelse "unknown";
        const alignment = model.gguf_file.alignment;
        const total_tensors = model.gguf_file.tensor_count;

        try stdout_writer.print("  Architecture: {s}\n", .{arch});
        try stdout_writer.print("  Model name:   {s}\n", .{name});
        try stdout_writer.print("  Alignment:    {d} bytes\n", .{alignment});
        try stdout_writer.print("  Tensors:      {d} descriptors parsed\n", .{total_tensors});

        var global_count: usize = 0;
        var swa_count: usize = 0;
        for (model.layers) |l| {
            if (l.is_global) {
                global_count += 1;
            } else {
                swa_count += 1;
            }
        }
        try stdout_writer.print("  Layer verification: {d} total ({d} SWA, {d} Global with tied V=K)\n", .{
            model.layers.len,
            swa_count,
            global_count,
        });

        const emb_blocks = model.getEmbeddingBlocks(zeno.SpecialTokens.bos);
        try stdout_writer.print("  Token embedding: {d} Q6_K blocks per token ({d} values)\n", .{
            emb_blocks.len,
            emb_blocks.len * zeno.quant.QK_K,
        });
        try stdout_writer.print("Model inspection successful.\n", .{});
    } else {
        try stdout_writer.print("Config: canvas={d} tokens | vocab={d} | d_model={d} | layers={d} | experts={d} (top-{d})\n", .{
            zeno.ModelConfig.canvas_length,
            zeno.ModelConfig.vocab_size,
            zeno.ModelConfig.embedding_dim,
            zeno.ModelConfig.num_layers,
            zeno.ModelConfig.num_experts,
            zeno.ModelConfig.num_active_experts,
        });
        try stdout_writer.print("\nUsage:\n", .{});
        try stdout_writer.print("  zeno [info] <path/to/model.gguf>\n", .{});
        try stdout_writer.print("  zeno decide <path/to/model.gguf> \"<query>\" [cand1] [cand2] ...\n", .{});
        try stdout_writer.print("  zeno serve <path/to/model.gguf> [--host <ip>] [--port <port>]\n", .{});
    }

    try stdout_writer.flush();
}

fn parseCandidateSpecs(allocator: std.mem.Allocator, cand_args: []const []const u8) ![]zeno.CandidateSpec {
    if (cand_args.len == 0) {
        const defaults = try allocator.alloc(zeno.CandidateSpec, 2);
        defaults[0] = .{ .token = "Yes", .label = "Yes" };
        defaults[1] = .{ .token = "No", .label = "No" };
        return defaults;
    }
    const specs = try allocator.alloc(zeno.CandidateSpec, cand_args.len);
    for (cand_args, 0..) |cand_str, i| {
        specs[i] = .{ .token = cand_str, .label = cand_str };
    }
    return specs;
}

fn candidateSpecsFromFixture(allocator: std.mem.Allocator, fix: zeno.GoldenFixture) ![]zeno.CandidateSpec {
    const specs = try allocator.alloc(zeno.CandidateSpec, fix.candidates.len);
    for (fix.candidates, 0..) |cand, i| {
        specs[i] = .{ .label = cand.label, .token = cand.token };
    }
    return specs;
}

fn parseSchemaType(schema_str: []const u8) !zeno.SchemaType {
    if (std.mem.eql(u8, schema_str, "boolean")) return .boolean;
    if (std.mem.eql(u8, schema_str, "choice")) return .choice;
    if (std.mem.eql(u8, schema_str, "score")) return .score;
    return error.InvalidCharacter;
}

fn encodePromptWithBOS(allocator: std.mem.Allocator, tokenizer: zeno.Tokenizer, formatted_prompt: []const u8) ![]i32 {
    const raw_tokens = try tokenizer.encode(allocator, formatted_prompt);
    var prompt_tokens = try allocator.alloc(i32, raw_tokens.len + 1);
    prompt_tokens[0] = @intCast(zeno.SpecialTokens.bos);
    for (raw_tokens, 0..) |tok, i| {
        prompt_tokens[i + 1] = @intCast(tok);
    }
    return prompt_tokens;
}

fn makeCanvasTokens(seed_token: u32) [256]i32 {
    var canvas: [256]i32 = undefined;
    canvas[0] = @intCast(seed_token);
    canvas[1] = @intCast(zeno.SpecialTokens.turn_close);
    @memset(canvas[2..], @intCast(zeno.SpecialTokens.pad));
    return canvas;
}

fn writeTokensToFile(path: []const u8, tokens: []const i32) !void {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    defer _ = std.os.linux.close(fd);
    const bytes = std.mem.sliceAsBytes(tokens);
    _ = std.os.linux.write(fd, bytes.ptr, bytes.len);
}

fn writeDecisionResponseJson(allocator: std.mem.Allocator, out_json_path: []const u8, resp: zeno.DecisionResponse) !void {
    var json_list: std.ArrayList(u8) = .empty;

    const header = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "decision": "{s}",
        \\  "decision_probability": {d:.6},
        \\  "entropy_nats": {d:.6},
        \\  "normalized_entropy": {d:.6},
        \\  "uncertain": {},
        \\  "steps_executed": {d},
        \\  "canvas_tokens_used": {d},
        \\  "candidates": [
        \\
    , .{
        resp.decision,
        resp.decision_probability,
        resp.entropy_nats,
        resp.normalized_entropy,
        resp.uncertain,
        resp.steps_executed,
        resp.canvas_tokens_used,
    });
    try json_list.appendSlice(allocator, header);

    for (resp.candidates, 0..) |c, i| {
        const sc = zeno.ops.logitSoftcap(c.raw_logit, zeno.ModelConfig.logit_softcap);
        const comma = if (i + 1 < resp.candidates.len) "," else "";
        const cand_str = try std.fmt.allocPrint(allocator,
            \\    {{
            \\      "label": "{s}",
            \\      "token_id": {d},
            \\      "raw_logit": {d:.6},
            \\      "softcapped_logit": {d:.6},
            \\      "probability": {d:.6}
            \\    }}{s}
            \\
        , .{
            c.label,
            c.token_id,
            c.raw_logit,
            sc,
            c.probability,
            comma,
        });
        try json_list.appendSlice(allocator, cand_str);
    }

    try json_list.appendSlice(allocator, "  ]\n}\n");

    const out_fd = try std.posix.openat(std.posix.AT.FDCWD, out_json_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    defer _ = std.os.linux.close(out_fd);
    _ = std.os.linux.write(out_fd, json_list.items.ptr, json_list.items.len);
}

test "main module compilation test" {
    try std.testing.expect(zeno.version.major == 0);
}

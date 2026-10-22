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

    if (init.environ_map.get("ZENO_DUMP_LAYERS") != null) {
        zeno.infer.dump_layers = true;
    }
    if (init.environ_map.get("ZENO_DUMP_LAYERS_BIN")) |path| {
        zeno.infer.dump_layers_path = path;
    }

    const args = try init.minimal.args.toSlice(arena);
    if (args.len > 1 and std.mem.eql(u8, args[1], "cuda-info")) {
        const res = printCudaInfo(stdout_writer);
        try stdout_writer.flush();
        try res;
        return;
    } else if (args.len > 1 and std.mem.eql(u8, args[1], "serve")) {
        if (args.len < 3) {
            try stdout_writer.print("Usage: zeno serve <path/to/model.gguf> [--cuda] [--host <ip>] [--port <port>]\n", .{});
            try stdout_writer.flush();
            return;
        }

        const model_path = args[2];
        var host: []const u8 = "127.0.0.1";
        var port: u16 = 8000;
        var use_cuda = false;

        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--cuda")) {
                use_cuda = true;
            } else if (std.mem.eql(u8, args[i], "--host")) {
                if (i + 1 >= args.len) {
                    try stdout_writer.print("Error: --host requires an IP address argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                i += 1;
                host = args[i];
            } else if (std.mem.eql(u8, args[i], "--port")) {
                if (i + 1 >= args.len) {
                    try stdout_writer.print("Error: --port requires a port number argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                i += 1;
                port = std.fmt.parseInt(u16, args[i], 10) catch {
                    try stdout_writer.print("Error: Invalid port number '{s}' (must be 0-65535)\n", .{args[i]});
                    try stdout_writer.flush();
                    return;
                };
            } else {
                try stdout_writer.print("Error: Unknown option '{s}'\n", .{args[i]});
                try stdout_writer.flush();
                return;
            }
        }

        try stdout_writer.print("Initializing decision engine with model: {s}...\n", .{model_path});
        if (use_cuda) {
            if (!zeno.cuda.is_enabled) {
                try stdout_writer.print("Error: CUDA support was disabled at build time (-Dcuda=false)\n", .{});
                try stdout_writer.flush();
                return;
            }
            const topo = zeno.cuda.getTopology() catch |err| {
                try stdout_writer.print("Error discovering CUDA devices: {}\n", .{err});
                try stdout_writer.flush();
                return;
            };
            try stdout_writer.print("Backend: CUDA ({d} device(s) detected, topology: {s})\n", .{
                topo.reported_devices,
                if (topo.reported_devices >= 2) "dual_gpu_split" else "single_gpu",
            });
        } else {
            try stdout_writer.print("Backend: CPU\n", .{});
        }
        try stdout_writer.flush();

        var engine = zeno.DecisionEngine.initWithOptions(arena, model_path, .{
            .backend = if (use_cuda) .cuda else .cpu,
        }) catch |err| {
            try stdout_writer.print("Error initializing engine: {}\n", .{err});
            try stdout_writer.flush();
            return;
        };
        defer engine.deinit() catch |err| {
            std.log.err("engine deinitialization failed: {s}", .{@errorName(err)});
        };

        var server_inst = zeno.Server.init(std.heap.page_allocator, &engine, host, port);
        try server_inst.listen();
        defer server_inst.deinit();

        if (!std.mem.eql(u8, host, "127.0.0.1") and !std.mem.eql(u8, host, "localhost")) {
            try stdout_writer.print("WARNING: Binding to non-loopback address '{s}' without authentication. Local/trusted network use only.\n", .{host});
        }

        try stdout_writer.print("Zeno System One Decision API server listening on http://{s}:{d}\n", .{ host, server_inst.port });
        try stdout_writer.print("Endpoints:\n", .{});
        try stdout_writer.print("  GET  /health           - Liveness probe\n", .{});
        try stdout_writer.print("  POST /v1/systemone     - Jev-compatible decision API\n\n", .{});
        try stdout_writer.flush();

        try server_inst.serve();
        return;
    } else if (args.len > 1 and std.mem.eql(u8, args[1], "decide")) {
        if (args.len < 4) {
            try stdout_writer.print("Usage: zeno decide <path/to/model.gguf> <query> [--cuda] [--canvas <N>] [cand1] [cand2] ...\n", .{});
            try stdout_writer.flush();
            return;
        }

        const model_path = args[2];
        const query = args[3];

        var canvas_override: ?usize = null;
        var answer_pos: usize = 0;
        var use_cuda = false;
        var step_cap: u32 = 1;
        var raw_candidates: std.ArrayList([]const u8) = .empty;
        var arg_i: usize = 4;
        while (arg_i < args.len) : (arg_i += 1) {
            if (std.mem.eql(u8, args[arg_i], "--cuda")) {
                use_cuda = true;
            } else if (std.mem.eql(u8, args[arg_i], "--steps")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --steps requires an integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                step_cap = std.fmt.parseInt(u32, args[arg_i], 10) catch {
                    try stdout_writer.print("Error: Invalid step count '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else if (std.mem.eql(u8, args[arg_i], "--canvas")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --canvas requires an integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                canvas_override = std.fmt.parseInt(usize, args[arg_i], 10) catch {
                    try stdout_writer.print("Error: Invalid canvas length '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else if (std.mem.eql(u8, args[arg_i], "--answer-pos")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --answer-pos requires an integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                answer_pos = std.fmt.parseInt(usize, args[arg_i], 10) catch {
                    try stdout_writer.print("Error: Invalid answer position '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else {
                try raw_candidates.append(arena, args[arg_i]);
            }
        }

        const candidate_specs = try parseCandidateSpecs(arena, raw_candidates.items);

        try stdout_writer.print("Initializing decision engine with model: {s}...\n", .{model_path});
        if (use_cuda) {
            if (!zeno.cuda.is_enabled) {
                try stdout_writer.print("Error: CUDA support was disabled at build time (-Dcuda=false)\n", .{});
                try stdout_writer.flush();
                return;
            }
            const topo = zeno.cuda.getTopology() catch |err| {
                try stdout_writer.print("Error discovering CUDA devices: {}\n", .{err});
                try stdout_writer.flush();
                return;
            };
            try stdout_writer.print("Backend: CUDA ({d} device(s) detected, topology: {s})\n", .{
                topo.reported_devices,
                if (topo.reported_devices >= 2) "dual_gpu_split" else "single_gpu",
            });
        } else {
            try stdout_writer.print("Backend: CPU\n", .{});
        }
        try stdout_writer.flush();

        var engine = zeno.DecisionEngine.initWithOptions(arena, model_path, .{
            .backend = if (use_cuda) .cuda else .cpu,
        }) catch |err| {
            try stdout_writer.print("Error initializing engine: {}\n", .{err});
            try stdout_writer.flush();
            return;
        };
        defer engine.deinit() catch |err| {
            std.log.err("engine deinitialization failed: {s}", .{@errorName(err)});
        };

        const req = zeno.DecisionRequest{
            .query = query,
            .schema_type = .boolean,
            .candidates = candidate_specs,
            .step_cap = step_cap,
            .temperature = 1.0,
            .entropy_threshold = 0.5,
            .canvas_length = canvas_override,
            .answer_position = answer_pos,
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
        if (resp.timings) |t| {
            try stdout_writer.print("Timings:\n", .{});
            try stdout_writer.print("  - Prompt prefill:    {d:.2} ms (cache hit: {})\n", .{ t.prompt_prefill_ms, t.prompt_cache_hit });
            try stdout_writer.print("  - Canvas Step 1:     {d:.2} ms\n", .{ t.canvas_step1_ms });
            if (t.canvas_step2_ms > 0) {
                try stdout_writer.print("  - Canvas Step 2:     {d:.2} ms\n", .{ t.canvas_step2_ms });
            }
            try stdout_writer.print("  - Total Engine:      {d:.2} ms\n", .{ t.total_engine_ms });
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
        defer engine.deinit() catch |err| {
            std.log.err("engine deinitialization failed: {s}", .{@errorName(err)});
        };

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
        defer engine.deinit() catch |err| {
            std.log.err("engine deinitialization failed: {s}", .{@errorName(err)});
        };

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
            try stdout_writer.print("Usage: zeno decide-fixture <model.gguf> <fixture.json> [--cuda] [--canvas <N>] [out.json]\n", .{});
            try stdout_writer.flush();
            return;
        }
        const model_path = args[2];
        const fixture_path = args[3];

        var canvas_override: ?usize = null;
        var answer_pos: usize = 0;
        var step_cap: u32 = 1;
        var out_json_path: ?[]const u8 = null;
        var use_cuda = false;
        var arg_i: usize = 4;
        while (arg_i < args.len) : (arg_i += 1) {
            if (std.mem.eql(u8, args[arg_i], "--cuda")) {
                use_cuda = true;
            } else if (std.mem.eql(u8, args[arg_i], "--steps")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --steps requires an integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                step_cap = std.fmt.parseInt(u32, args[arg_i], 10) catch {
                    try stdout_writer.print("Error: Invalid step count '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else if (std.mem.eql(u8, args[arg_i], "--canvas")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --canvas requires an integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                canvas_override = std.fmt.parseInt(usize, args[arg_i], 10) catch {
                    try stdout_writer.print("Error: Invalid canvas length '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else if (std.mem.eql(u8, args[arg_i], "--answer-pos")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --answer-pos requires an integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                answer_pos = std.fmt.parseInt(usize, args[arg_i], 10) catch {
                    try stdout_writer.print("Error: Invalid answer position '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else {
                out_json_path = args[arg_i];
            }
        }

        const parsed = try zeno.fixture.loadFixture(arena, fixture_path);
        defer parsed.deinit();
        const fix = parsed.value;

        if (use_cuda and !zeno.cuda.is_enabled) {
            try stdout_writer.print("Error: CUDA support was disabled at build time (-Dcuda=false)\n", .{});
            try stdout_writer.flush();
            return;
        }

        var engine = try zeno.DecisionEngine.initWithOptions(arena, model_path, .{
            .backend = if (use_cuda) .cuda else .cpu,
        });
        defer engine.deinit() catch |err| {
            std.log.err("engine deinitialization failed: {s}", .{@errorName(err)});
        };

        const candidate_specs = try candidateSpecsFromFixture(arena, fix);
        const schema_t = try parseSchemaType(fix.schema_type);

        const req = zeno.DecisionRequest{
            .query = fix.query,
            .context = fix.context,
            .schema_type = schema_t,
            .candidates = candidate_specs,
            .step_cap = step_cap,
            .temperature = 1.0,
            .entropy_threshold = fix.entropy_threshold,
            .canvas_length = canvas_override,
            .answer_position = answer_pos,
        };

        const resp = try engine.decide(arena, req);

        try stdout_writer.print("Decision:           {s}\n", .{resp.decision});
        try stdout_writer.print("Decision Prob:      {d:.4}\n", .{resp.decision_probability});
        try stdout_writer.print("Entropy (nats):     {d:.4}\n", .{resp.entropy_nats});
        try stdout_writer.print("Normalized Entropy: {d:.4}\n", .{resp.normalized_entropy});
        try stdout_writer.print("Uncertain:          {}\n", .{resp.uncertain});
        try stdout_writer.print("Steps Executed:     {d}\n", .{resp.steps_executed});
        if (resp.timings) |t| {
            try stdout_writer.print("Timings (ms):       Prefill: {d:.2} (P0: {d:.2}, Transfer: {d:.2}, P1: {d:.2}, Hit: {}) | Canvas Step1: {d:.2} | Step2: {d:.2} | Total Engine: {d:.2}\n", .{
                t.prompt_prefill_ms,
                t.prompt_p0_ms,
                t.prompt_transfer_ms,
                t.prompt_p1_ms,
                t.prompt_cache_hit,
                t.canvas_step1_ms,
                t.canvas_step2_ms,
                t.total_engine_ms,
            });
        }
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

        if (out_json_path) |p| {
            try writeDecisionResponseJson(arena, p, resp);
        }
        return;
    } else if (args.len > 1 and std.mem.eql(u8, args[1], "evaluate-dataset")) {
        if (args.len < 4) {
            try stdout_writer.print("Usage: zeno evaluate-dataset <model.gguf> <dataset.json> [--cuda] [--steps <N>] [--threshold <tau>] [--canvas <N>] [--out <results.json>]\n", .{});
            try stdout_writer.flush();
            return;
        }
        const model_path = args[2];
        const dataset_path = args[3];

        var canvas_override: ?usize = null;
        var answer_pos: usize = 0;
        var step_cap: u32 = 1;
        var entropy_threshold: ?f32 = null;
        var eval_temperature: f32 = 1.0;
        var out_json_path: ?[]const u8 = null;
        var use_cuda = false;
        var arg_i: usize = 4;
        while (arg_i < args.len) : (arg_i += 1) {
            if (std.mem.eql(u8, args[arg_i], "--cuda")) {
                use_cuda = true;
            } else if (std.mem.eql(u8, args[arg_i], "--steps")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --steps requires an integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                step_cap = std.fmt.parseInt(u32, args[arg_i], 10) catch {
                    try stdout_writer.print("Error: Invalid step count '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else if (std.mem.eql(u8, args[arg_i], "--temperature")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --temperature requires a float argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                eval_temperature = std.fmt.parseFloat(f32, args[arg_i]) catch {
                    try stdout_writer.print("Error: Invalid temperature '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
                if (!std.math.isFinite(eval_temperature) or eval_temperature <= 0.0) {
                    try stdout_writer.print("Error: Temperature must be a positive finite float\n", .{});
                    try stdout_writer.flush();
                    return;
                }
            } else if (std.mem.eql(u8, args[arg_i], "--threshold")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --threshold requires a float argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                entropy_threshold = std.fmt.parseFloat(f32, args[arg_i]) catch {
                    try stdout_writer.print("Error: Invalid entropy threshold '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else if (std.mem.eql(u8, args[arg_i], "--canvas")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --canvas requires an integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                canvas_override = std.fmt.parseInt(usize, args[arg_i], 10) catch {
                    try stdout_writer.print("Error: Invalid canvas length '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else if (std.mem.eql(u8, args[arg_i], "--answer-pos")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --answer-pos requires an integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                answer_pos = std.fmt.parseInt(usize, args[arg_i], 10) catch {
                    try stdout_writer.print("Error: Invalid answer position '{s}'\n", .{args[arg_i]});
                    try stdout_writer.flush();
                    return;
                };
            } else if (std.mem.eql(u8, args[arg_i], "--out")) {
                if (arg_i + 1 >= args.len) {
                    try stdout_writer.print("Error: --out requires an output path argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                arg_i += 1;
                out_json_path = args[arg_i];
            } else {
                out_json_path = args[arg_i];
            }
        }

        const parsed = try zeno.dataset.loadDataset(arena, dataset_path);
        defer parsed.deinit();
        const ds = parsed.value;

        if (use_cuda and !zeno.cuda.is_enabled) {
            try stdout_writer.print("Error: CUDA support was disabled at build time (-Dcuda=false)\n", .{});
            try stdout_writer.flush();
            return;
        }

        try stdout_writer.print("Initializing decision engine with model: {s} (backend: {s})...\n", .{
            model_path,
            if (use_cuda) "CUDA" else "CPU",
        });
        try stdout_writer.flush();

        var engine = try zeno.DecisionEngine.initWithOptions(arena, model_path, .{
            .backend = if (use_cuda) .cuda else .cpu,
        });
        defer engine.deinit() catch |err| {
            std.log.err("engine deinitialization failed: {s}", .{@errorName(err)});
        };

        const eff_canvas = canvas_override orelse 256;
        try stdout_writer.print("Evaluating dataset '{s}' ({d} examples, step_cap={d}, canvas={d}, answer_pos={d})\n\n", .{
            ds.name,
            ds.examples.len,
            step_cap,
            eff_canvas,
            answer_pos,
        });
        try stdout_writer.flush();

        // Warmup query using first example
        if (ds.examples.len > 0) {
            const first = ds.examples[0];
            var warmup_cands = try arena.alloc(zeno.CandidateSpec, first.candidates.len);
            for (first.candidates, 0..) |c, i| {
                warmup_cands[i] = .{ .token = c.token, .label = c.label };
            }
            const warmup_req = zeno.DecisionRequest{
                .query = first.query,
                .context = first.context,
                .schema_type = parseSchemaType(first.schema_type) catch .boolean,
                .candidates = warmup_cands,
                .step_cap = step_cap,
                .temperature = eval_temperature,
                .entropy_threshold = entropy_threshold,
                .canvas_length = canvas_override,
                .answer_position = answer_pos,
            };
            _ = try engine.decide(arena, warmup_req);
        }

        var results = try arena.alloc(zeno.dataset.EvaluationResult, ds.examples.len);
        var correct_count: usize = 0;
        var step_counts = [_]usize{ 0, 0, 0, 0, 0 };
        var total_latency_ms: f64 = 0.0;
        var latencies = try arena.alloc(f64, ds.examples.len);
        var sum_norm_entropy: f64 = 0.0;
        var uncertain_count: usize = 0;
        var total_prefill_ms: f64 = 0.0;
        var cold_prefill_count: usize = 0;
        var total_canvas_step1_ms: f64 = 0.0;
        var total_canvas_step2_ms: f64 = 0.0;
        var step2_count: usize = 0;

        for (ds.examples, 0..) |ex, idx| {
            var cands = try arena.alloc(zeno.CandidateSpec, ex.candidates.len);
            for (ex.candidates, 0..) |c, i| {
                cands[i] = .{ .token = c.token, .label = c.label };
            }
            const schema_t = try parseSchemaType(ex.schema_type);

            const req = zeno.DecisionRequest{
                .query = ex.query,
                .context = ex.context,
                .schema_type = schema_t,
                .candidates = cands,
                .step_cap = step_cap,
                .temperature = eval_temperature,
                .entropy_threshold = entropy_threshold,
                .canvas_length = canvas_override,
                .answer_position = answer_pos,
            };

            const t_start = getMonotonicMs();
            const resp = try engine.decide(arena, req);
            const latency = getMonotonicMs() - t_start;

            latencies[idx] = latency;
            total_latency_ms += latency;
            sum_norm_entropy += resp.normalized_entropy;

            if (resp.timings) |t| {
                if (!t.prompt_cache_hit) {
                    total_prefill_ms += t.prompt_prefill_ms;
                    cold_prefill_count += 1;
                }
                total_canvas_step1_ms += t.canvas_step1_ms;
                if (t.canvas_step2_ms > 0.0) {
                    total_canvas_step2_ms += t.canvas_step2_ms;
                    step2_count += 1;
                }
            }

            const is_correct = std.mem.eql(u8, resp.decision, ex.ground_truth);
            if (is_correct) correct_count += 1;
            if (resp.uncertain) uncertain_count += 1;

            if (resp.steps_executed <= 4) {
                step_counts[resp.steps_executed] += 1;
            }

            results[idx] = .{
                .id = ex.id,
                .category = ex.category,
                .ground_truth = ex.ground_truth,
                .decision = resp.decision,
                .is_correct = is_correct,
                .decision_probability = resp.decision_probability,
                .entropy_nats = resp.entropy_nats,
                .normalized_entropy = resp.normalized_entropy,
                .uncertain = resp.uncertain,
                .steps_executed = resp.steps_executed,
                .latency_ms = latency,
                .candidates = resp.candidates,
                .stage_timings = resp.timings,
            };

            if (resp.timings) |t| {
                const hit_str = if (t.prompt_cache_hit) " [cache hit]" else "";
                try stdout_writer.print("[{d:>2}/{d}] ({s:<10}) Pred: '{s:<8}' | Truth: '{s:<8}' | Match: {s} | Prob: {d:.4} | H_norm: {d:.4} | Steps: {d} | {d:6.1}ms (prefill={d:.1}ms{s}, c1={d:.1}ms, c2={d:.1}ms)\n", .{
                    idx + 1,
                    ds.examples.len,
                    ex.id,
                    resp.decision,
                    ex.ground_truth,
                    if (is_correct) "YES" else "NO ",
                    resp.decision_probability,
                    resp.normalized_entropy,
                    resp.steps_executed,
                    latency,
                    t.prompt_prefill_ms,
                    hit_str,
                    t.canvas_step1_ms,
                    t.canvas_step2_ms,
                });
            } else {
                try stdout_writer.print("[{d:>2}/{d}] ({s:<10}) Pred: '{s:<8}' | Truth: '{s:<8}' | Match: {s} | Prob: {d:.4} | H_norm: {d:.4} | Steps: {d} | {d:6.1}ms\n", .{
                    idx + 1,
                    ds.examples.len,
                    ex.id,
                    resp.decision,
                    ex.ground_truth,
                    if (is_correct) "YES" else "NO ",
                    resp.decision_probability,
                    resp.normalized_entropy,
                    resp.steps_executed,
                    latency,
                });
            }
            try stdout_writer.flush();
        }

        const stats = calculateStats(latencies);
        const accuracy = if (ds.examples.len > 0) @as(f64, @floatFromInt(correct_count)) / @as(f64, @floatFromInt(ds.examples.len)) else 0.0;
        const mean_latency = if (ds.examples.len > 0) total_latency_ms / @as(f64, @floatFromInt(ds.examples.len)) else 0.0;
        const mean_norm_ent = if (ds.examples.len > 0) sum_norm_entropy / @as(f64, @floatFromInt(ds.examples.len)) else 0.0;

        const summary = zeno.dataset.EvaluationSummary{
            .total_examples = ds.examples.len,
            .correct_count = correct_count,
            .accuracy = accuracy,
            .mean_latency_ms = mean_latency,
            .median_latency_ms = stats.median,
            .mean_normalized_entropy = mean_norm_ent,
            .step_counts = step_counts,
            .uncertain_count = uncertain_count,
        };

        try stdout_writer.print("\n=== Dataset Evaluation Summary ===\n", .{});
        try stdout_writer.print("Total Examples:        {d}\n", .{summary.total_examples});
        try stdout_writer.print("Correct Decisions:     {d} ({d:.2}% accuracy)\n", .{ summary.correct_count, summary.accuracy * 100.0 });
        try stdout_writer.print("Latency:               Mean: {d:.2}ms | Median: {d:.2}ms | Spread: {d:.2}ms\n", .{ summary.mean_latency_ms, summary.median_latency_ms, stats.spread });
        if (ds.examples.len > 0 and results[0].stage_timings != null) {
            const mean_prefill = if (cold_prefill_count > 0) total_prefill_ms / @as(f64, @floatFromInt(cold_prefill_count)) else 0.0;
            const mean_c1 = total_canvas_step1_ms / @as(f64, @floatFromInt(ds.examples.len));
            const mean_c2 = if (step2_count > 0) total_canvas_step2_ms / @as(f64, @floatFromInt(step2_count)) else 0.0;
            try stdout_writer.print("Stage Timings:         Prefill Mean: {d:.2}ms (N={d}) | Canvas Step1 Mean: {d:.2}ms | Canvas Step2 Mean: {d:.2}ms (N={d})\n", .{
                mean_prefill,
                cold_prefill_count,
                mean_c1,
                mean_c2,
                step2_count,
            });
        }
        try stdout_writer.print("Mean Norm Entropy:     {d:.4}\n", .{summary.mean_normalized_entropy});
        try stdout_writer.print("Uncertain Queries:     {d} / {d}\n", .{ summary.uncertain_count, summary.total_examples });
        try stdout_writer.print("Step Distribution:     T=1: {d} | T=2: {d} | T=3: {d} | T=4: {d}\n", .{
            summary.step_counts[1],
            summary.step_counts[2],
            summary.step_counts[3],
            summary.step_counts[4],
        });
        try stdout_writer.print("==================================\n", .{});
        try stdout_writer.flush();

        if (out_json_path) |p| {
            try zeno.dataset.writeEvaluationResultsJson(
                arena,
                p,
                if (use_cuda) "cuda" else "cpu",
                step_cap,
                entropy_threshold,
                eval_temperature,
                eff_canvas,
                answer_pos,
                results,
                summary,
            );
            try stdout_writer.print("Saved detailed results JSON to: {s}\n", .{p});
            try stdout_writer.flush();
        }
        return;
    } else if (args.len > 1 and (std.mem.eql(u8, args[1], "benchmark-canvas") or std.mem.eql(u8, args[1], "benchmark"))) {
        if (args.len < 3) {
            try stdout_writer.print("Usage: zeno benchmark <model.gguf> [--cuda] [--samples N] [fixture.json]\n", .{});
            try stdout_writer.flush();
            return;
        }
        const model_path = args[2];
        var specific_fixture: ?[]const u8 = null;
        var use_cuda = false;
        var num_samples: usize = 3;

        var b_idx: usize = 3;
        while (b_idx < args.len) : (b_idx += 1) {
            if (std.mem.eql(u8, args[b_idx], "--cuda")) {
                use_cuda = true;
            } else if (std.mem.eql(u8, args[b_idx], "--samples")) {
                if (b_idx + 1 >= args.len) {
                    try stdout_writer.print("Error: --samples requires a positive integer argument\n", .{});
                    try stdout_writer.flush();
                    return;
                }
                b_idx += 1;
                num_samples = std.fmt.parseInt(usize, args[b_idx], 10) catch {
                    try stdout_writer.print("Error: Invalid sample count '{s}'\n", .{args[b_idx]});
                    try stdout_writer.flush();
                    return;
                };
                if (num_samples == 0) {
                    try stdout_writer.print("Error: --samples must be at least 1\n", .{});
                    try stdout_writer.flush();
                    return;
                }
            } else {
                specific_fixture = args[b_idx];
            }
        }

        const all_fixtures = [_][]const u8{
            "specs/fixtures/system_health_boolean.json",
            "specs/fixtures/incident_severity_multiclass.json",
            "specs/fixtures/satisfaction_score_5class.json",
            "specs/fixtures/maintenance_priority_uncertain.json",
            "specs/fixtures/boundary_256_incident_triage.json",
        };

        const target_fixtures: []const []const u8 = if (specific_fixture) |f|
            try arena.dupe([]const u8, &.{f})
        else
            &all_fixtures;

        try stdout_writer.print("========================================================================================\n", .{});
        try stdout_writer.print("Zeno Shortened Canvas Prefix Benchmark ({s})\n", .{if (use_cuda) "CUDA" else "CPU"});
        try stdout_writer.print("Model: {s}\n", .{model_path});
        if (use_cuda) {
            if (!zeno.cuda.is_enabled) {
                try stdout_writer.print("Error: CUDA support was disabled at build time (-Dcuda=false)\n", .{});
                try stdout_writer.flush();
                return;
            }
            const topo = zeno.cuda.getTopology() catch |err| {
                try stdout_writer.print("Error discovering CUDA devices: {}\n", .{err});
                try stdout_writer.flush();
                return;
            };
            try stdout_writer.print("Topology: {s} ({d} device(s) detected)\n", .{
                if (topo.reported_devices >= 2) "dual_gpu_split" else "single_gpu",
                topo.reported_devices,
            });
            for (0..topo.reported_devices) |dev_idx| {
                const dev = &topo.devices[dev_idx];
                try stdout_writer.print("  Device {d}: {s} (Compute {d}.{d}, {d} MB VRAM)\n", .{
                    dev.device_id,
                    dev.name(),
                    dev.compute_major,
                    dev.compute_minor,
                    dev.total_memory_bytes / (1024 * 1024),
                });
            }
            if (topo.reported_devices >= 2) {
                const p2p_rc = zeno.cuda.c.zeno_cuda_enable_peer_access(0, 1);
                try stdout_writer.print("  Inter-GPU Transfer: {s}\n", .{
                    if (p2p_rc == 0) "Direct P2P DMA (enabled)" else "Host-staged fallback (P2P unsupported by hardware topology)",
                });
            }
        }
        try stdout_writer.print("Benchmark Configuration: {d} sample(s) per width post-warmup (reporting median & spread)\n", .{num_samples});
        try stdout_writer.print("Evaluating canvas widths C in {{16, 32, 64, 128, 256}}\n", .{});
        try stdout_writer.print("========================================================================================\n\n", .{});
        try stdout_writer.flush();

        var engine = try zeno.DecisionEngine.initWithOptions(arena, model_path, .{
            .backend = if (use_cuda) .cuda else .cpu,
        });
        defer engine.deinit() catch |err| {
            std.log.err("engine deinitialization failed: {s}", .{@errorName(err)});
        };

        const canvas_widths = [_]usize{ 16, 32, 64, 128, 256 };

        for (target_fixtures) |fixture_path| {
            try stdout_writer.print("--- Fixture: {s} ---\n", .{fixture_path});
            const parsed = try zeno.fixture.loadFixture(arena, fixture_path);
            defer parsed.deinit();
            const fix = parsed.value;

            const candidate_specs = try candidateSpecsFromFixture(arena, fix);
            const schema_t = try parseSchemaType(fix.schema_type);

            // 1. Evaluate baseline C=256 first
            const baseline_req = zeno.DecisionRequest{
                .query = fix.query,
                .context = fix.context,
                .schema_type = schema_t,
                .candidates = candidate_specs,
                .step_cap = 1,
                .temperature = 1.0,
                .entropy_threshold = fix.entropy_threshold,
                .canvas_length = 256,
            };

            // Warmup baseline (for both CPU and CUDA backends, R24-03)
            _ = try engine.decide(arena, baseline_req);

            var baseline_samples = try arena.alloc(f64, num_samples);
            var baseline_resp: zeno.DecisionResponse = undefined;
            for (0..num_samples) |s_i| {
                const t_base_start = getMonotonicMs();
                baseline_resp = try engine.decide(arena, baseline_req);
                const t_base_end = getMonotonicMs();
                baseline_samples[s_i] = t_base_end - t_base_start;
            }
            const baseline_stats = calculateStats(baseline_samples);

            try stdout_writer.print("Baseline (C=256): Decision='{s}' Prob={d:.4} Median={d:.2}ms Spread={d:.2}ms (N={d})\n\n", .{
                baseline_resp.decision,
                baseline_resp.decision_probability,
                baseline_stats.median,
                baseline_stats.spread,
                num_samples,
            });

            try stdout_writer.print("| Canvas C | Decision | Agreement | Prob    | dProb  | dLogit(raw) | Median (ms) | Spread (ms) | Speedup |\n", .{});
            try stdout_writer.print("|:---------|:---------|:----------|:--------|:-------|:------------|:------------|:------------|:--------|\n", .{});

            for (canvas_widths) |c_len| {
                const req = zeno.DecisionRequest{
                    .query = fix.query,
                    .context = fix.context,
                    .schema_type = schema_t,
                    .candidates = candidate_specs,
                    .step_cap = 1,
                    .temperature = 1.0,
                    .entropy_threshold = fix.entropy_threshold,
                    .canvas_length = c_len,
                };

                // Warmup per canvas width (for both CPU and CUDA backends, R24-03)
                _ = try engine.decide(arena, req);
                var samples = try arena.alloc(f64, num_samples);
                var resp: zeno.DecisionResponse = undefined;
                for (0..num_samples) |s_i| {
                    const t_start = getMonotonicMs();
                    resp = try engine.decide(arena, req);
                    const t_end = getMonotonicMs();
                    samples[s_i] = t_end - t_start;
                }
                const stats = calculateStats(samples);
                const speedup = if (stats.median > 0.0) baseline_stats.median / stats.median else 1.0;

                const agreement = std.mem.eql(u8, resp.decision, baseline_resp.decision);
                var max_prob_diff: f32 = 0.0;
                var max_logit_diff: f32 = 0.0;
                for (resp.candidates, baseline_resp.candidates) |cand, bcand| {
                    const pdiff = @abs(cand.probability - bcand.probability);
                    if (pdiff > max_prob_diff) max_prob_diff = pdiff;
                    const ldiff = @abs(cand.raw_logit - bcand.raw_logit);
                    if (ldiff > max_logit_diff) max_logit_diff = ldiff;
                }

                try stdout_writer.print("| {d:>8} | {s:<8} | {s:<9} | {d:.4} | {d:.4} | {d:11.4} | {d:11.2} | {d:11.2} | {d:6.2}x |\n", .{
                    c_len,
                    resp.decision,
                    if (agreement) "YES" else "NO",
                    resp.decision_probability,
                    max_prob_diff,
                    max_logit_diff,
                    stats.median,
                    stats.spread,
                    speedup,
                });
                try stdout_writer.flush();
            }
            try stdout_writer.print("\n", .{});
            try stdout_writer.flush();
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
        try stdout_writer.print("  zeno decide <path/to/model.gguf> \"<query>\" [--cuda] [--canvas <N>] [cand1] [cand2] ...\n", .{});
        try stdout_writer.print("  zeno decide-fixture <path/to/model.gguf> <fixture.json> [--cuda] [--canvas <N>] [out.json]\n", .{});
        try stdout_writer.print("  zeno evaluate-dataset <path/to/model.gguf> <dataset.json> [--cuda] [--steps <N>] [--threshold <tau>] [--temperature <T>] [--canvas <N>] [--out <results.json>]\n", .{});
        try stdout_writer.print("  zeno benchmark <path/to/model.gguf> [--cuda] [fixture.json]\n", .{});
        try stdout_writer.print("  zeno serve <path/to/model.gguf> [--cuda] [--host <ip>] [--port <port>]\n", .{});
        try stdout_writer.print("  zeno cuda-info\n", .{});
    }

    try stdout_writer.flush();
}

fn printCudaInfo(writer: anytype) !void {
    try writer.print("\n======================================================================\n", .{});
    try writer.print(" Zeno CUDA Hardware Topology & Capabilities\n", .{});
    try writer.print("======================================================================\n", .{});

    if (!zeno.cuda.is_enabled) {
        try writer.print("CUDA Backend:        DISABLED in this binary build.\n", .{});
        try writer.print("Target Architecture: none\n", .{});
        try writer.print("\nTo enable CUDA support, rebuild with: zig build -Dcuda=true\n", .{});
        try writer.print("======================================================================\n", .{});
        return;
    }

    try writer.print("CUDA Backend:        ENABLED (Compiled with -Dcuda=true, {s})\n", .{zeno.cuda.target_arch});

    const topo = zeno.cuda.getTopology() catch |err| {
        try writer.print("Error querying CUDA device topology: {}\n", .{err});
        try writer.print("======================================================================\n", .{});
        return err;
    };

    try writer.print("Visible Devices:     {d} (Reported: {d})\n", .{ topo.total_visible_devices, topo.reported_devices });

    for (0..topo.reported_devices) |i| {
        const dev = &topo.devices[i];
        const total_mib = dev.total_memory_bytes / (1024 * 1024);
        const free_mib = dev.free_memory_bytes / (1024 * 1024);
        const total_gb = @as(f64, @floatFromInt(dev.total_memory_bytes)) / (1000.0 * 1000.0 * 1000.0);

        try writer.print("\n[Device {d}]\n", .{dev.device_id});
        try writer.print("  Model:             {s}\n", .{dev.name()});
        try writer.print("  Compute Cap:       {d}.{d}\n", .{ dev.compute_major, dev.compute_minor });
        try writer.print("  SM Count:          {d}\n", .{dev.multi_processor_count});
        try writer.print("  Warp Size:         {d}\n", .{dev.warp_size});
        try writer.print("  Total VRAM:        {d} MiB ({d:.2} GB)\n", .{ total_mib, total_gb });
        try writer.print("  Free VRAM:         {d} MiB\n", .{free_mib});
    }

    if (topo.reported_devices > 0) {
        try writer.print("\n----------------------------------------------------------------------\n", .{});
        try writer.print(" Inter-GPU Peer-to-Peer (P2P) Access Capability Matrix:\n", .{});
        try writer.print(" (Capability query only; does not imply P2P is currently enabled)\n", .{});
        try writer.print("----------------------------------------------------------------------\n", .{});

        // Header row
        try writer.print("             ", .{});
        for (0..topo.reported_devices) |j| {
            try writer.print("    GPU {d}   ", .{j});
        }
        try writer.print("\n", .{});

        // Matrix rows
        for (0..topo.reported_devices) |i| {
            try writer.print("  GPU {d}      ", .{i});
            for (0..topo.reported_devices) |j| {
                if (i == j) {
                    try writer.print("      —       ", .{});
                } else if (topo.devices[i].can_access_peer[j]) {
                    try writer.print("   Capable    ", .{});
                } else {
                    try writer.print(" Not Capable  ", .{});
                }
            }
            try writer.print("\n", .{});
        }
    }

    try writer.print("======================================================================\n", .{});
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

const BenchmarkStats = struct {
    median: f64,
    spread: f64,
};

fn calculateStats(samples: []f64) BenchmarkStats {
    if (samples.len == 0) return .{ .median = 0.0, .spread = 0.0 };
    std.mem.sort(f64, samples, {}, struct {
        fn lessThan(_: void, a: f64, b: f64) bool {
            return a < b;
        }
    }.lessThan);
    const mid = samples.len / 2;
    const med = if (samples.len % 2 == 1)
        samples[mid]
    else
        (samples[mid - 1] + samples[mid]) / 2.0;
    const spread = samples[samples.len - 1] - samples[0];
    return .{ .median = med, .spread = spread };
}

fn getMonotonicMs() f64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1000.0 + @as(f64, @floatFromInt(ts.nsec)) / 1_000_000.0;
}

fn writeAllToFd(fd: i32, bytes: []const u8) !void {
    var written: usize = 0;
    while (written < bytes.len) {
        const rc = std.os.linux.write(fd, bytes.ptr + written, bytes.len - written);
        const signed_rc: isize = @bitCast(rc);
        if (signed_rc < 0) return error.WriteFailed;
        if (rc == 0) return error.WriteFailed;
        written += rc;
    }
}

fn writeTokensToFile(path: []const u8, tokens: []const i32) !void {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    defer _ = std.os.linux.close(fd);
    const bytes = std.mem.sliceAsBytes(tokens);
    try writeAllToFd(fd, bytes);
}

fn writeDecisionResponseJson(allocator: std.mem.Allocator, out_json_path: []const u8, resp: zeno.DecisionResponse) !void {
    var json_list: std.ArrayList(u8) = .empty;
    const esc_decision = try zeno.dataset.escapeJsonString(allocator, resp.decision);

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
        esc_decision,
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
        const esc_label = try zeno.dataset.escapeJsonString(allocator, c.label);
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
            esc_label,
            c.token_id,
            c.raw_logit,
            sc,
            c.probability,
            comma,
        });
        try json_list.appendSlice(allocator, cand_str);
    }

    try json_list.appendSlice(allocator, "  ]");
    if (resp.timings) |t| {
        const tim_str = try std.fmt.allocPrint(allocator,
            \\,
            \\  "timings": {{
            \\    "prompt_prefill_ms": {d:.3},
            \\    "prompt_p0_ms": {d:.3},
            \\    "prompt_transfer_ms": {d:.3},
            \\    "prompt_p1_ms": {d:.3},
            \\    "canvas_step1_ms": {d:.3},
            \\    "canvas_step2_ms": {d:.3},
            \\    "total_engine_ms": {d:.3},
            \\    "prompt_cache_hit": {}
            \\  }}
        , .{
            t.prompt_prefill_ms,
            t.prompt_p0_ms,
            t.prompt_transfer_ms,
            t.prompt_p1_ms,
            t.canvas_step1_ms,
            t.canvas_step2_ms,
            t.total_engine_ms,
            t.prompt_cache_hit,
        });
        try json_list.appendSlice(allocator, tim_str);
    }
    try json_list.appendSlice(allocator, "\n}\n");

    const out_fd = try std.posix.openat(std.posix.AT.FDCWD, out_json_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    defer _ = std.os.linux.close(out_fd);
    try writeAllToFd(out_fd, json_list.items);
}

test "main module compilation test" {
    try std.testing.expect(zeno.version.major == 0);
}

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
    if (args.len > 1 and std.mem.eql(u8, args[1], "decide")) {
        if (args.len < 4) {
            try stdout_writer.print("Usage: zeno decide <path/to/model.gguf> <query> [cand1] [cand2] ...\n", .{});
            try stdout_writer.flush();
            return;
        }

        const model_path = args[2];
        const query = args[3];

        var candidates_list: std.ArrayList(zeno.CandidateSpec) = .empty;
        if (args.len > 4) {
            for (args[4..]) |cand_str| {
                try candidates_list.append(arena, .{ .token = cand_str, .label = cand_str });
            }
        } else {
            // Default boolean candidates
            try candidates_list.append(arena, .{ .token = "Yes", .label = "Yes" });
            try candidates_list.append(arena, .{ .token = "No", .label = "No" });
        }

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
            .candidates = candidates_list.items,
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

        var cands_list: std.ArrayList(zeno.CandidateSpec) = .empty;
        if (args.len > 6) {
            for (args[6..]) |cand_str| {
                try cands_list.append(arena, .{ .token = cand_str, .label = cand_str });
            }
        } else {
            try cands_list.append(arena, .{ .token = "Yes", .label = "Yes" });
            try cands_list.append(arena, .{ .token = "No", .label = "No" });
        }

        const formatted = try zeno.Canvas.formatDecisionPrompt(arena, query, null, cands_list.items);
        const raw_tokens = try engine.tokenizer.encode(arena, formatted);

        var prompt_tokens = try arena.alloc(i32, raw_tokens.len + 1);
        prompt_tokens[0] = @intCast(zeno.SpecialTokens.bos);
        for (raw_tokens, 0..) |tok, i| {
            prompt_tokens[i + 1] = @intCast(tok);
        }

        const posix = std.posix;
        const linux = std.os.linux;

        const p_fd = try posix.openat(posix.AT.FDCWD, prompt_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        defer _ = linux.close(p_fd);
        const p_bytes = std.mem.sliceAsBytes(prompt_tokens);
        _ = linux.write(p_fd, p_bytes.ptr, p_bytes.len);

        var canvas_tokens: [256]i32 = undefined;
        canvas_tokens[0] = @intCast(zeno.SpecialTokens.mask);
        canvas_tokens[1] = @intCast(zeno.SpecialTokens.turn_close);
        for (2..256) |i| {
            canvas_tokens[i] = @intCast(zeno.SpecialTokens.pad);
        }

        const c_fd = try posix.openat(posix.AT.FDCWD, canvas_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        defer _ = linux.close(c_fd);
        const c_bytes = std.mem.sliceAsBytes(&canvas_tokens);
        _ = linux.write(c_fd, c_bytes.ptr, c_bytes.len);

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

        const candidate_specs = try arena.alloc(zeno.CandidateSpec, fix.candidates.len);
        for (fix.candidates, 0..) |cand, i| {
            candidate_specs[i] = .{
                .label = cand.label,
                .token = cand.token,
            };
        }

        const formatted = try zeno.Canvas.formatDecisionPrompt(arena, fix.query, fix.context, candidate_specs);
        const raw_tokens = try engine.tokenizer.encode(arena, formatted);

        var prompt_tokens = try arena.alloc(i32, raw_tokens.len + 1);
        prompt_tokens[0] = @intCast(zeno.SpecialTokens.bos);
        for (raw_tokens, 0..) |tok, i| {
            prompt_tokens[i + 1] = @intCast(tok);
        }

        const posix = std.posix;
        const linux = std.os.linux;

        const p_fd = try posix.openat(posix.AT.FDCWD, prompt_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        defer _ = linux.close(p_fd);
        const p_bytes = std.mem.sliceAsBytes(prompt_tokens);
        _ = linux.write(p_fd, p_bytes.ptr, p_bytes.len);

        var canvas_tokens: [256]i32 = undefined;
        canvas_tokens[0] = @intCast(fix.answer_seed_token);
        canvas_tokens[1] = @intCast(zeno.SpecialTokens.turn_close);
        for (2..256) |i| {
            canvas_tokens[i] = @intCast(zeno.SpecialTokens.pad);
        }

        const c_fd = try posix.openat(posix.AT.FDCWD, canvas_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        defer _ = linux.close(c_fd);
        const c_bytes = std.mem.sliceAsBytes(&canvas_tokens);
        _ = linux.write(c_fd, c_bytes.ptr, c_bytes.len);

        try stdout_writer.print("Exported fixture '{s}': {d} prompt tokens to {s} and 256 canvas tokens to {s}\n", .{
            fixture_path,
            prompt_tokens.len,
            prompt_path,
            canvas_path,
        });
        try stdout_writer.flush();
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
    }

    try stdout_writer.flush();
}

test "main module compilation test" {
    try std.testing.expect(zeno.version.major == 0);
}

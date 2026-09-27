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

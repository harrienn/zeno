//! Zeno — Jev-style System One Decision Engine.
//!
//! Orchestrates request validation, candidate resolution, seeded canvas construction,
//! structured read forward evaluation, candidate softmax scoring, and entropy evaluation.

const std = @import("std");
const root = @import("root.zig");
const decision = @import("decision.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const canvas_mod = @import("canvas.zig");
const infer = @import("infer.zig");
const ops = @import("ops.zig");

pub const DecisionRequest = decision.DecisionRequest;
pub const DecisionResponse = decision.DecisionResponse;
pub const CandidateResult = decision.CandidateResult;
pub const DiffusionGemmaModel = model_mod.DiffusionGemmaModel;
pub const Tokenizer = tokenizer_mod.Tokenizer;
pub const Canvas = canvas_mod.Canvas;
pub const Mutex = struct {
    inner: std.atomic.Mutex = .unlocked,

    pub fn lock(self: *Mutex) void {
        while (!self.inner.tryLock()) {
            std.Thread.yield() catch {};
        }
    }

    pub fn unlock(self: *Mutex) void {
        self.inner.unlock();
    }
};

pub const DecisionEngine = struct {
    allocator: std.mem.Allocator,
    model: DiffusionGemmaModel,
    tokenizer: Tokenizer,
    workspace: *infer.InferenceWorkspace,
    mutex: Mutex = .{},

    /// Initializes decision engine by memory-mapping the GGUF model, loading tokenizer vocabulary,
    /// and pre-allocating an inference workspace with persistent thread pool.
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !DecisionEngine {
        var model = try DiffusionGemmaModel.loadFromFile(allocator, model_path);
        errdefer model.deinit();

        var tok = try Tokenizer.initFromGguf(allocator, model.gguf_file);
        errdefer tok.deinit(allocator);

        var ws = try infer.InferenceWorkspace.init(allocator, 512, null);
        errdefer ws.deinit();

        return DecisionEngine{
            .allocator = allocator,
            .model = model,
            .tokenizer = tok,
            .workspace = ws,
            .mutex = .{},
        };
    }

    /// Releases workspace, tokenizer vocabulary, model mapping, and memory.
    pub fn deinit(self: *DecisionEngine) void {
        self.workspace.deinit();
        self.tokenizer.deinit(self.allocator);
        self.model.deinit();
    }

    /// Executes a deterministic single-step software decision query conforming to the Jev /v1/systemone contract.
    /// Thread-safe: serializes concurrent decide calls on the same engine instance to prevent workspace tensor races (R5-01).
    pub fn decide(self: *DecisionEngine, request_allocator: std.mem.Allocator, req: DecisionRequest) !DecisionResponse {
        self.mutex.lock();
        defer self.mutex.unlock();

        // 1. Validate request bounds and constraints
        try decision.validateRequest(req);

        // 2. Validate and resolve candidate tokens
        const cand_count = req.candidates.len;
        const candidate_tokens = try request_allocator.alloc(u32, cand_count);
        defer request_allocator.free(candidate_tokens);

        try Canvas.validateCandidates(self.tokenizer, req.candidates, candidate_tokens);

        // 3. Format decision prompt and tokenize into seeded canvas
        const formatted_prompt = try Canvas.formatDecisionPrompt(
            request_allocator,
            req.query,
            req.context,
            req.candidates,
        );
        defer request_allocator.free(formatted_prompt);

        const raw_prompt_tokens = try self.tokenizer.encode(request_allocator, formatted_prompt);
        defer request_allocator.free(raw_prompt_tokens);

        const prompt_tokens = try request_allocator.alloc(u32, raw_prompt_tokens.len + 1);
        defer request_allocator.free(prompt_tokens);
        prompt_tokens[0] = root.SpecialTokens.bos;
        @memcpy(prompt_tokens[1..], raw_prompt_tokens);

        const canvas = try Canvas.seedSingleStep(prompt_tokens, root.SpecialTokens.mask);

        // 4. Execute structured read over canvas for candidate tokens
        const raw_logits = try request_allocator.alloc(f32, cand_count);
        defer request_allocator.free(raw_logits);

        const softcapped_logits = try request_allocator.alloc(f32, cand_count);
        defer request_allocator.free(softcapped_logits);

        try infer.executeStructuredRead(
            self.workspace,
            self.model,
            prompt_tokens,
            canvas,
            candidate_tokens,
            raw_logits,
            softcapped_logits,
        );

        // 5. Candidate-restricted softmax normalization
        const probs = try request_allocator.alloc(f32, cand_count);
        defer request_allocator.free(probs);

        ops.softmax(softcapped_logits, req.temperature, probs);

        // 6. Candidate Shannon entropy in nats and normalized entropy in [0, 1]
        const h_nats = ops.shannonEntropyNats(probs);
        const h_norm = ops.normalizedEntropy(h_nats, cand_count);

        const uncertain = if (req.entropy_threshold) |th| (h_norm >= th) else false;

        // 7. Determine highest-scoring decision candidate
        var best_idx: usize = 0;
        var max_prob: f32 = probs[0];
        for (probs[1..], 1..) |p, i| {
            if (p > max_prob) {
                max_prob = p;
                best_idx = i;
            }
        }

        // 8. Build detailed candidate result slice allocated with request_allocator
        const candidate_results = try request_allocator.alloc(CandidateResult, cand_count);
        for (0..cand_count) |i| {
            candidate_results[i] = CandidateResult{
                .label = req.candidates[i].label,
                .token_id = candidate_tokens[i],
                .probability = probs[i],
                .raw_logit = raw_logits[i],
            };
        }

        return DecisionResponse{
            .decision = req.candidates[best_idx].label,
            .decision_probability = max_prob,
            .entropy_nats = h_nats,
            .normalized_entropy = h_norm,
            .uncertain = uncertain,
            .candidates = candidate_results,
            .steps_executed = 1,
            .canvas_tokens_used = @intCast(canvas.prompt_len + 1),
        };
    }
};

test "DecisionEngine: end-to-end boolean decision execution on local model if present" {
    if (!root.shouldRunModelTests()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var engine = DecisionEngine.init(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer engine.deinit();

    const candidates = [_]decision.CandidateSpec{
        .{ .token = "Yes", .label = "Yes" },
        .{ .token = "No", .label = "No" },
    };

    const req = DecisionRequest{
        .query = "Is this service operating normally?",
        .context = "Heartbeat timestamp: 2026-09-27T20:10:00Z. Errors: 0.",
        .schema_type = .boolean,
        .candidates = &candidates,
        .step_cap = 1,
        .temperature = 1.0,
        .entropy_threshold = 0.8,
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const resp = try engine.decide(arena.allocator(), req);

    try std.testing.expect(std.mem.eql(u8, resp.decision, "Yes") or std.mem.eql(u8, resp.decision, "No"));
    try std.testing.expect(resp.decision_probability > 0.0 and resp.decision_probability <= 1.0);
    try std.testing.expect(resp.entropy_nats >= 0.0);
    try std.testing.expect(resp.normalized_entropy >= 0.0 and resp.normalized_entropy <= 1.0);
    try std.testing.expect(resp.canvas_tokens_used > 0);
    try std.testing.expectEqual(@as(usize, 2), resp.candidates.len);
}

test "DecisionEngine: normative tie-breaking selects lowest candidate index (U-19)" {
    // When candidates have identical probabilities, lowest index must win
    const probs = [_]f32{ 0.5, 0.5 };
    var best_idx: usize = 0;
    var max_prob: f32 = probs[0];
    for (probs[1..], 1..) |p, i| {
        if (p > max_prob) {
            max_prob = p;
            best_idx = i;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), best_idx);
}

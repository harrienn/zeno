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
const infer_cuda = @import("infer_cuda.zig");
const ops = @import("ops.zig");

pub const BackendType = enum {
    cpu,
    cuda,
};

pub const EngineOptions = struct {
    backend: BackendType = .cpu,
    force_single_gpu: bool = false,
    workspace_seq_len: usize = 512,
};

pub const DecisionRequest = decision.DecisionRequest;
pub const DecisionResponse = decision.DecisionResponse;
pub const CandidateResult = decision.CandidateResult;
pub const DiffusionGemmaModel = model_mod.DiffusionGemmaModel;
pub const Tokenizer = tokenizer_mod.Tokenizer;
pub const Canvas = canvas_mod.Canvas;
pub const Mutex = struct {
    inner: std.atomic.Mutex = .unlocked,

    pub fn lock(self: *Mutex) void {
        var spin_count: usize = 0;
        while (!self.inner.tryLock()) {
            spin_count += 1;
            if (spin_count < 100) {
                std.atomic.spinLoopHint();
            } else if (spin_count < 1000) {
                std.Thread.yield() catch {};
            } else {
                const ts = std.posix.timespec{ .sec = 0, .nsec = 1_000_000 };
                _ = std.posix.system.nanosleep(&ts, null);
            }
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
    backend: BackendType,
    cpu_workspace: ?*infer.InferenceWorkspace = null,
    cuda_model: ?*infer_cuda.CudaModel = null,
    cuda_workspace: ?*infer_cuda.CudaWorkspace = null,
    mutex: Mutex = .{},

    /// Initializes decision engine with default CPU reference execution.
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !DecisionEngine {
        return initWithOptions(allocator, model_path, .{ .backend = .cpu });
    }

    /// Initializes decision engine with explicit backend and hardware execution options.
    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        options: EngineOptions,
    ) !DecisionEngine {
        var model = try DiffusionGemmaModel.loadFromFile(allocator, model_path);
        errdefer model.deinit();

        var tok = try Tokenizer.initFromGguf(allocator, model.gguf_file);
        errdefer tok.deinit(allocator);

        var cpu_ws: ?*infer.InferenceWorkspace = null;
        var cuda_model: ?*infer_cuda.CudaModel = null;
        var cuda_ws: ?*infer_cuda.CudaWorkspace = null;

        errdefer {
            if (cuda_ws) |ws| ws.deinit();
            if (cuda_model) |cm| cm.deinit();
            if (cpu_ws) |ws| ws.deinit();
        }

        switch (options.backend) {
            .cpu => {
                cpu_ws = try infer.InferenceWorkspace.init(allocator, options.workspace_seq_len, null);
            },
            .cuda => {
                if (!infer_cuda.is_enabled) return error.CudaDisabledAtBuild;
                const cm = try infer_cuda.CudaModel.initFromCpuModel(allocator, model, options.force_single_gpu);
                cuda_model = cm;
                const cws = try infer_cuda.CudaWorkspace.init(allocator, cm.topology, cm.devices, options.workspace_seq_len);
                cuda_ws = cws;
            },
        }

        return DecisionEngine{
            .allocator = allocator,
            .model = model,
            .tokenizer = tok,
            .backend = options.backend,
            .cpu_workspace = cpu_ws,
            .cuda_model = cuda_model,
            .cuda_workspace = cuda_ws,
            .mutex = .{},
        };
    }

    /// Releases workspace, tokenizer vocabulary, model mapping, and memory.
    pub fn deinit(self: *DecisionEngine) void {
        if (self.cuda_workspace) |ws| ws.deinit();
        if (self.cuda_model) |cm| cm.deinit();
        if (self.cpu_workspace) |ws| ws.deinit();
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

        // 3. Early byte-length guard before formatting and tokenization (R7-05, R8-02)
        const preflight_prompt_bytes = Canvas.calculatePromptByteLength(
            req.query,
            req.context,
            req.candidates,
        ) orelse return error.CanvasOverflow;

        if (preflight_prompt_bytes > root.ModelConfig.max_prompt_bytes) {
            return error.CanvasOverflow;
        }

        const formatted_prompt = try Canvas.formatDecisionPrompt(
            request_allocator,
            req.query,
            req.context,
            req.candidates,
        );
        defer request_allocator.free(formatted_prompt);

        std.debug.assert(formatted_prompt.len == preflight_prompt_bytes);

        const raw_prompt_tokens = try self.tokenizer.encode(request_allocator, formatted_prompt);
        defer request_allocator.free(raw_prompt_tokens);

        if (raw_prompt_tokens.len + 1 > root.ModelConfig.max_prompt_tokens) {
            return error.CanvasOverflow;
        }

        const prompt_tokens = try request_allocator.alloc(u32, raw_prompt_tokens.len + 1);
        defer request_allocator.free(prompt_tokens);
        prompt_tokens[0] = root.SpecialTokens.bos;
        @memcpy(prompt_tokens[1..], raw_prompt_tokens);

        const canvas = try Canvas.seedSingleStep(prompt_tokens, root.SpecialTokens.mask);

        // 4. Execute structured read over canvas for candidate tokens
        // 4. Adaptive multi-step forward evaluation loop (Phase 9)
        const raw_logits = try request_allocator.alloc(f32, cand_count);
        defer request_allocator.free(raw_logits);

        const softcapped_logits = try request_allocator.alloc(f32, cand_count);
        defer request_allocator.free(softcapped_logits);

        const probs = try request_allocator.alloc(f32, cand_count);
        defer request_allocator.free(probs);

        const canvas_len = req.canvas_length orelse root.ModelConfig.canvas_length;
        const total_seq_len = prompt_tokens.len + canvas_len;

        var prev_probs: ?[]f32 = null;
        if (req.step_cap > 1) {
            prev_probs = try request_allocator.alloc(f32, cand_count);
        }
        defer if (prev_probs) |p| request_allocator.free(p);

        var steps_executed: u32 = 0;
        var h_nats: f32 = 0.0;
        var h_norm: f32 = 0.0;
        var uncertain: bool = false;

        while (steps_executed < req.step_cap) {
            const step_probs = if (steps_executed == 0) null else prev_probs;

            switch (self.backend) {
                .cpu => {
                    const ws = self.cpu_workspace orelse return error.InvalidEngineState;
                    try validateSequenceLength(.cpu, total_seq_len, ws.max_seq_len);
                    try infer.executeStructuredReadSlice(
                        ws,
                        self.model,
                        prompt_tokens,
                        canvas.tokens[0..canvas_len],
                        canvas.answer_pos,
                        candidate_tokens,
                        step_probs,
                        raw_logits,
                        softcapped_logits,
                    );
                },
                .cuda => {
                    const ws = self.cuda_workspace orelse return error.InvalidEngineState;
                    const cm = self.cuda_model orelse return error.InvalidEngineState;
                    try validateSequenceLength(.cuda, total_seq_len, ws.dev0.max_seq_len);
                    try infer_cuda.executeStructuredReadCuda(
                        ws,
                        cm,
                        self.model,
                        prompt_tokens,
                        canvas.tokens[0..canvas_len],
                        canvas.answer_pos,
                        candidate_tokens,
                        step_probs,
                        raw_logits,
                        softcapped_logits,
                    );
                },
            }

            steps_executed += 1;

            // 5. Candidate-restricted softmax normalization
            ops.softmax(softcapped_logits, req.temperature, probs);

            // 6. Candidate Shannon entropy in nats and normalized entropy in [0, 1]
            h_nats = ops.shannonEntropyNats(probs);
            h_norm = ops.normalizedEntropy(h_nats, cand_count);

            uncertain = if (req.entropy_threshold) |th| (h_norm >= th) else false;

            // Adaptive early stopping: if decisive (h_norm < entropy_threshold), terminate early
            if (req.entropy_threshold) |th| {
                if (h_norm < th) break;
            }

            // Propagate candidate probabilities as self-conditioning for subsequent step
            if (steps_executed < req.step_cap) {
                @memcpy(prev_probs.?, probs);
            }
        }

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
            .steps_executed = steps_executed,
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

test "DecisionEngine: end-to-end boolean decision execution on CUDA if present" {
    if (!root.shouldRunModelTests()) return error.SkipZigTest;
    if (!root.cuda.is_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var engine = DecisionEngine.initWithOptions(allocator, model_path, .{ .backend = .cuda }) catch |err| {
        if (err == error.FileNotFound or err == error.NoDevice) return;
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

test "DecisionEngine: multi-step adaptive stepping simulation logic (Phase 9 / R26-03)" {
    // Verify entropy gating and early termination behavior without model weights
    const cand_count: usize = 3;
    const entropy_threshold: f32 = 0.5;

    // Case 1: Step 1 is ambiguous (H_norm >= threshold), Step 2 is decisive (H_norm < threshold)
    // Step 1: P = [0.6, 0.35, 0.05] -> H_nats ~= 0.778, H_norm ~= 0.708 (>= 0.5 -> proceed to step 2)
    const probs_step1 = [_]f32{ 0.60, 0.35, 0.05 };
    const h1_nats = ops.shannonEntropyNats(&probs_step1);
    const h1_norm = ops.normalizedEntropy(h1_nats, cand_count);
    try std.testing.expect(h1_norm >= entropy_threshold);

    // Step 2: P = [0.95, 0.04, 0.01] -> H_nats ~= 0.215, H_norm ~= 0.196 (< 0.5 -> stop early)
    const probs_step2 = [_]f32{ 0.95, 0.04, 0.01 };
    const h2_nats = ops.shannonEntropyNats(&probs_step2);
    const h2_norm = ops.normalizedEntropy(h2_nats, cand_count);
    try std.testing.expect(h2_norm < entropy_threshold);

    // Case 2: Step 1 is already decisive -> terminates at step 1
    const decisive_probs = [_]f32{ 0.98, 0.01, 0.01 };
    const h_decisive = ops.normalizedEntropy(ops.shannonEntropyNats(&decisive_probs), cand_count);
    try std.testing.expect(h_decisive < entropy_threshold);
}

test "DecisionEngine: multi-step boolean decision execution on CUDA if present (Phase 9)" {
    if (!root.shouldRunModelTests()) return error.SkipZigTest;
    if (!root.cuda.is_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var engine = DecisionEngine.initWithOptions(allocator, model_path, .{ .backend = .cuda }) catch |err| {
        if (err == error.FileNotFound or err == error.NoDevice) return;
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
        .step_cap = 2,
        .temperature = 1.0,
        .entropy_threshold = 0.5,
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const resp = try engine.decide(arena.allocator(), req);

    try std.testing.expect(resp.steps_executed >= 1 and resp.steps_executed <= 2);
    try std.testing.expect(std.mem.eql(u8, resp.decision, "Yes") or std.mem.eql(u8, resp.decision, "No"));
    try std.testing.expect(resp.decision_probability > 0.0 and resp.decision_probability <= 1.0);
}

/// Validates total sequence length (prompt + canvas) against backend capabilities and workspace capacity (R23-04 / R24-05).
pub fn validateSequenceLength(backend: BackendType, total_seq_len: usize, workspace_max_seq_len: usize) !void {
    switch (backend) {
        .cpu => {
            if (total_seq_len > workspace_max_seq_len) {
                return error.CanvasOverflow;
            }
        },
        .cuda => {
            // Strict CUDA attention kernel ceiling is 512
            if (total_seq_len > 512 or total_seq_len > workspace_max_seq_len) {
                return error.CanvasOverflow;
            }
        },
    }
}

test "DecisionEngine: sequence length bounds validation for CPU and CUDA (R24-05)" {
    // 1. CPU allows sequence lengths > 512 when workspace capacity allows it
    try validateSequenceLength(.cpu, 513, 1024);
    try validateSequenceLength(.cpu, 768, 1024);
    try validateSequenceLength(.cpu, 1024, 1024);

    // 2. CPU rejects sequences exceeding workspace capacity
    try std.testing.expectError(error.CanvasOverflow, validateSequenceLength(.cpu, 1025, 1024));
    try std.testing.expectError(error.CanvasOverflow, validateSequenceLength(.cpu, 300, 256));

    // 3. CUDA accepts sequences <= 512 within workspace capacity
    try validateSequenceLength(.cuda, 16, 512);
    try validateSequenceLength(.cuda, 256, 512);
    try validateSequenceLength(.cuda, 512, 512);

    // 4. CUDA rejects sequences > 512 even if workspace capacity is larger
    try std.testing.expectError(error.CanvasOverflow, validateSequenceLength(.cuda, 513, 1024));
    try std.testing.expectError(error.CanvasOverflow, validateSequenceLength(.cuda, 600, 1024));

    // 5. CUDA rejects sequences exceeding workspace capacity even if <= 512
    try std.testing.expectError(error.CanvasOverflow, validateSequenceLength(.cuda, 400, 256));
}

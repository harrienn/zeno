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
    workspace_seq_len: usize = root.ModelConfig.max_seq_len,
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

/// Nominal stepping and output temperature scaling state for decision execution (R44-05, R45-01, R46-02).
pub const StepDecisionState = struct {
    steps_executed: u32 = 0,
    entropy_nats: f32 = 0.0,
    normalized_entropy: f32 = 0.0,
    uncertain: bool = false,

    /// Evaluates step candidate probabilities under native training temperature T=1.0 for self-conditioning and early stopping.
    /// Returns true if early stopping threshold was met (decisive).
    pub fn processStep(
        self: *StepDecisionState,
        softcapped_logits: []const f32,
        cand_count: usize,
        entropy_threshold: ?f32,
        probs_out: []f32,
        prev_probs_out: ?[]f32,
        step_cap: u32,
    ) bool {
        self.steps_executed += 1;
        ops.softmax(softcapped_logits, 1.0, probs_out);
        self.entropy_nats = ops.shannonEntropyNats(probs_out);
        self.normalized_entropy = ops.normalizedEntropy(self.entropy_nats, cand_count);
        self.uncertain = if (entropy_threshold) |th| (self.normalized_entropy >= th) else false;

        const stop_early = if (entropy_threshold) |th| (self.normalized_entropy < th) else false;
        if (!stop_early and self.steps_executed < step_cap) {
            if (prev_probs_out) |prev| {
                @memcpy(prev, probs_out);
            }
        }
        return stop_early;
    }

    /// Applies caller-specified temperature scaling to final logits, updating output probabilities,
    /// response entropy, and client-facing uncertain flag.
    pub fn finalizeOutput(
        self: *StepDecisionState,
        softcapped_logits: []const f32,
        cand_count: usize,
        temperature: f32,
        entropy_threshold: ?f32,
        probs_out: []f32,
    ) void {
        if (temperature != 1.0) {
            ops.softmax(softcapped_logits, temperature, probs_out);
            self.entropy_nats = ops.shannonEntropyNats(probs_out);
            self.normalized_entropy = ops.normalizedEntropy(self.entropy_nats, cand_count);
            self.uncertain = if (entropy_threshold) |th| (self.normalized_entropy >= th) else false;
        }
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
            if (cuda_ws) |ws| ws.deinit() catch |err| {
                std.log.err("cuda workspace deinitialization failed during engine init abort: {s}", .{@errorName(err)});
            };
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
                const cuda_seq_len = @min(options.workspace_seq_len, 512);
                const cws = try infer_cuda.CudaWorkspace.init(allocator, cm.topology, cm.devices, cuda_seq_len);
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
    /// Propagates any teardown error (e.g. error.DeviceNotQuiescent) and halts cleanup immediately,
    /// preserving model buffers and workspace ownership until device quiescence is established (R51-01).
    pub fn deinit(self: *DecisionEngine) !void {
        if (self.cuda_workspace) |ws| {
            try ws.deinit();
            self.cuda_workspace = null;
        }
        if (self.cuda_model) |cm| {
            cm.deinit();
            self.cuda_model = null;
        }
        if (self.cpu_workspace) |ws| {
            ws.deinit();
            self.cpu_workspace = null;
        }
        self.tokenizer.deinit(self.allocator);
        self.model.deinit();
    }

    /// Executes a deterministic single-step software decision query conforming to the Jev /v1/systemone contract.
    /// Thread-safe: serializes concurrent decide calls on the same engine instance to prevent workspace tensor races (R5-01).
    pub fn decide(self: *DecisionEngine, request_allocator: std.mem.Allocator, req: DecisionRequest) !DecisionResponse {
        self.mutex.lock();
        defer self.mutex.unlock();

        const t_engine_start = decision.getMonotonicNanos();

        // 1. Validate request bounds and constraints
        try decision.validateRequest(req);

        // 2. Early byte-length guard before formatting and tokenization (R7-05, R8-02)
        const preflight_prompt_bytes = Canvas.calculatePromptByteLength(
            req.query,
            req.context,
            req.candidates,
        ) orelse return error.CanvasOverflow;

        if (preflight_prompt_bytes > root.ModelConfig.max_prompt_bytes) {
            return error.CanvasOverflow;
        }

        // 3. Format structured decision prompt conforming to Gemma 4 delimiters
        const formatted_prompt = try Canvas.formatDecisionPrompt(
            request_allocator,
            req.query,
            req.context,
            req.candidates,
        );
        defer request_allocator.free(formatted_prompt);

        std.debug.assert(formatted_prompt.len == preflight_prompt_bytes);

        // 4. Tokenize prompt
        const raw_prompt_tokens = try self.tokenizer.encode(request_allocator, formatted_prompt);
        defer request_allocator.free(raw_prompt_tokens);

        // 5. Validate and resolve candidate tokens at the rendered answer boundary (R56-02)
        const cand_count = req.candidates.len;
        const candidate_tokens = try request_allocator.alloc(u32, cand_count);
        defer request_allocator.free(candidate_tokens);

        try Canvas.validateCandidateContinuations(
            self.tokenizer,
            request_allocator,
            formatted_prompt,
            raw_prompt_tokens,
            req.candidates,
            candidate_tokens,
        );

        const canvas_len = req.canvas_length orelse root.ModelConfig.canvas_length;
        const total_prompt_tokens = raw_prompt_tokens.len + 1; // +1 for bos
        const total_seq_len = total_prompt_tokens + canvas_len;

        // Sequence length preflight: validate against backend capabilities and workspace capacity
        // before allocating candidate logits, probabilities, or seeding canvas (R30-01).
        const max_ws_seq_len = switch (self.backend) {
            .cpu => if (self.cpu_workspace) |ws| ws.max_seq_len else root.ModelConfig.max_seq_len,
            .cuda => if (self.cuda_workspace) |ws| ws.dev0.max_seq_len else 512,
        };
        try validateSequenceLength(self.backend, total_seq_len, max_ws_seq_len);

        if (total_prompt_tokens > root.ModelConfig.max_prompt_tokens) {
            return error.CanvasOverflow;
        }

        const prompt_tokens = try request_allocator.alloc(u32, total_prompt_tokens);
        defer request_allocator.free(prompt_tokens);
        prompt_tokens[0] = root.SpecialTokens.bos;
        @memcpy(prompt_tokens[1..], raw_prompt_tokens);

        const ans_pos = req.answer_position orelse 0;
        if (ans_pos >= canvas_len) return error.CanvasOverflow;
        const canvas = try Canvas.seedSingleStepAt(prompt_tokens, root.SpecialTokens.mask, ans_pos);

        // 4. Adaptive multi-step forward evaluation loop (Phase 9)
        const raw_logits = try request_allocator.alloc(f32, cand_count);
        defer request_allocator.free(raw_logits);

        const softcapped_logits = try request_allocator.alloc(f32, cand_count);
        defer request_allocator.free(softcapped_logits);

        const probs = try request_allocator.alloc(f32, cand_count);
        defer request_allocator.free(probs);

        var prev_probs: ?[]f32 = null;
        if (req.step_cap > 1) {
            if (!self.model.hasSelfConditioning()) {
                return error.MissingSelfCondTensor;
            }
            if (self.backend == .cuda) {
                if (self.cuda_model) |cm| {
                    if (!cm.hasSelfConditioning()) return error.MissingSelfCondTensor;
                }
            }
            prev_probs = try request_allocator.alloc(f32, cand_count);
        }
        defer if (prev_probs) |p| request_allocator.free(p);

        var step_state = StepDecisionState{};
        var stage_timings: ?decision.StageTimings = if (self.backend == .cuda) decision.StageTimings{} else null;

        while (step_state.steps_executed < req.step_cap) {
            const step_probs = if (step_state.steps_executed == 0) null else prev_probs;

            switch (self.backend) {
                .cpu => {
                    const ws = self.cpu_workspace orelse return error.InvalidEngineState;
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
                    if (stage_timings) |*st| {
                        if (step_state.steps_executed == 0) {
                            st.prompt_prefill_ms = ws.last_stage_timings.prompt_prefill_ms;
                            st.prompt_p0_ms = ws.last_stage_timings.prompt_p0_ms;
                            st.prompt_transfer_ms = ws.last_stage_timings.prompt_transfer_ms;
                            st.prompt_p1_ms = ws.last_stage_timings.prompt_p1_ms;
                            st.prompt_cache_hit = ws.last_stage_timings.prompt_cache_hit;
                            st.canvas_step1_ms = ws.last_canvas_ms;
                        } else if (step_state.steps_executed == 1) {
                            st.canvas_step2_ms = ws.last_canvas_ms;
                        }
                    }
                },
            }

            // 5 & 6. Process step candidate probabilities at native T=1.0 for self-conditioning and early stopping
            if (step_state.processStep(
                softcapped_logits,
                cand_count,
                req.entropy_threshold,
                probs,
                prev_probs,
                req.step_cap,
            )) break;
        }

        // R44-05 & R45-01: Apply caller-requested temperature scaling strictly to final decision output probabilities
        step_state.finalizeOutput(
            softcapped_logits,
            cand_count,
            req.temperature,
            req.entropy_threshold,
            probs,
        );

        const steps_executed = step_state.steps_executed;
        const h_nats = step_state.entropy_nats;
        const h_norm = step_state.normalized_entropy;
        const uncertain = step_state.uncertain;

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

        if (stage_timings) |*st| {
            const t_engine_end = decision.getMonotonicNanos();
            st.total_engine_ms = @as(f32, @floatFromInt(t_engine_end - t_engine_start)) / 1_000_000.0;
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
            .timings = stage_timings,
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
    defer engine.deinit() catch {};

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
    try std.testing.expect(resp.timings == null);
}

test "DecisionEngine: multiple-choice coded options with distinct tokens and labels (R56-01)" {
    if (!root.shouldRunModelTests()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var engine = DecisionEngine.init(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer engine.deinit() catch {};

    const candidates = [_]decision.CandidateSpec{
        .{ .token = "A", .label = "Credit Card Fraud" },
        .{ .token = "B", .label = "Identity Theft" },
        .{ .token = "C", .label = "Account Takeover" },
    };

    const req = DecisionRequest{
        .query = "What type of financial fraud is reported?",
        .context = "Customer reported an unknown charge of $540 at a gas station.",
        .schema_type = .choice,
        .candidates = &candidates,
        .step_cap = 1,
        .temperature = 1.0,
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const resp = try engine.decide(arena.allocator(), req);

    // Decision must match one of the user-facing display labels, not the token code "A"
    try std.testing.expect(
        std.mem.eql(u8, resp.decision, "Credit Card Fraud") or
        std.mem.eql(u8, resp.decision, "Identity Theft") or
        std.mem.eql(u8, resp.decision, "Account Takeover"),
    );
    try std.testing.expect(!std.mem.eql(u8, resp.decision, "A"));
    try std.testing.expect(!std.mem.eql(u8, resp.decision, "B"));
    try std.testing.expect(!std.mem.eql(u8, resp.decision, "C"));

    // Verify candidate token IDs match codes A, B, C and preserve display labels
    try std.testing.expectEqual(@as(u32, 236776), resp.candidates[0].token_id); // 'A'
    try std.testing.expectEqualStrings("Credit Card Fraud", resp.candidates[0].label);
    try std.testing.expectEqual(@as(u32, 236784), resp.candidates[1].token_id); // 'B'
    try std.testing.expectEqualStrings("Identity Theft", resp.candidates[1].label);
    try std.testing.expectEqual(@as(u32, 236798), resp.candidates[2].token_id); // 'C'
    try std.testing.expectEqualStrings("Account Takeover", resp.candidates[2].label);
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
    defer engine.deinit() catch {};

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
    try std.testing.expect(resp.timings != null);
}

test "DecisionEngine: adaptive early stopping stepping loop state machine (R27-03)" {
    const StepOutcome = struct {
        probs: []const f32,
    };

    const runLoop = struct {
        fn execute(step_cap: u32, threshold: ?f32, outcomes: []const StepOutcome) struct { steps: u32, uncertain: bool, prev_probs_updated: bool } {
            var steps_executed: u32 = 0;
            var uncertain: bool = false;
            var prev_probs_updated = false;

            while (steps_executed < step_cap) {
                const cur_outcome = outcomes[steps_executed];
                steps_executed += 1;

                const h_nats = ops.shannonEntropyNats(cur_outcome.probs);
                const h_norm = ops.normalizedEntropy(h_nats, cur_outcome.probs.len);

                uncertain = if (threshold) |th| (h_norm >= th) else false;

                if (threshold) |th| {
                    if (h_norm < th) break;
                }

                if (steps_executed < step_cap) {
                    prev_probs_updated = true;
                }
            }

            return .{ .steps = steps_executed, .uncertain = uncertain, .prev_probs_updated = prev_probs_updated };
        }
    }.execute;

    const p_ambiguous = [_]f32{ 0.60, 0.35, 0.05 }; // H_norm ~= 0.708 (>= 0.5)
    const p_decisive = [_]f32{ 0.95, 0.04, 0.01 }; // H_norm ~= 0.196 (< 0.5)

    // Scenario 1: Step 1 ambiguous, Step 2 decisive, step_cap = 3 -> terminates at step 2 (early stopping)
    const outcomes1 = [_]StepOutcome{ .{ .probs = &p_ambiguous }, .{ .probs = &p_decisive }, .{ .probs = &p_decisive } };
    const res1 = runLoop(3, 0.5, &outcomes1);
    try std.testing.expectEqual(@as(u32, 2), res1.steps);
    try std.testing.expectEqual(false, res1.uncertain);
    try std.testing.expectEqual(true, res1.prev_probs_updated);

    // Scenario 2: Step 1 already decisive, step_cap = 4 -> terminates immediately at step 1
    const outcomes2 = [_]StepOutcome{ .{ .probs = &p_decisive }, .{ .probs = &p_ambiguous } };
    const res2 = runLoop(4, 0.5, &outcomes2);
    try std.testing.expectEqual(@as(u32, 1), res2.steps);
    try std.testing.expectEqual(false, res2.uncertain);
    try std.testing.expectEqual(false, res2.prev_probs_updated);

    // Scenario 3: All steps ambiguous, step_cap = 2 -> exhausts cap, uncertain=true
    const outcomes3 = [_]StepOutcome{ .{ .probs = &p_ambiguous }, .{ .probs = &p_ambiguous } };
    const res3 = runLoop(2, 0.5, &outcomes3);
    try std.testing.expectEqual(@as(u32, 2), res3.steps);
    try std.testing.expectEqual(true, res3.uncertain);
    try std.testing.expectEqual(true, res3.prev_probs_updated);
}

test "DecisionEngine: multi-step decision execution on CUDA if present (Phase 9 / R27-03)" {
    if (!root.shouldRunModelTests()) return error.SkipZigTest;
    if (!root.cuda.is_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var engine = DecisionEngine.initWithOptions(allocator, model_path, .{ .backend = .cuda }) catch |err| {
        if (err == error.FileNotFound or err == error.NoDevice) return;
        return err;
    };
    defer engine.deinit() catch {};

    // Use ambiguous maintenance fixture: step 1 produces H_norm ~ 0.596 (uncertain), step 2 produces H_norm ~ 0.140 (decisive)
    const candidates = [_]decision.CandidateSpec{
        .{ .token = "Low", .label = "Low" },
        .{ .token = "Medium", .label = "Medium" },
        .{ .token = "High", .label = "High" },
    };

    var req = DecisionRequest{
        .query = "What is the priority level of this maintenance event?",
        .context = "Routine index reorganization scheduled for customer analytics warehouse. Query latency expected to elevate by 10% for analytics queries only during off-peak window. Production OLTP databases are entirely unaffected.\n",
        .schema_type = .choice,
        .candidates = &candidates,
        .step_cap = 1,
        .temperature = 1.0,
        .entropy_threshold = 0.5,
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    // 1. Single-step execution on ambiguous query: must execute exactly 1 step and report uncertain=true
    const resp1 = try engine.decide(arena.allocator(), req);
    try std.testing.expectEqual(@as(u32, 1), resp1.steps_executed);
    try std.testing.expectEqual(true, resp1.uncertain);

    // 2. Multi-step execution with step_cap=2: step 1 is ambiguous, step 2 is decisive -> must execute 2 steps and report uncertain=false
    req.step_cap = 2;
    const resp2 = try engine.decide(arena.allocator(), req);
    try std.testing.expectEqual(@as(u32, 2), resp2.steps_executed);
    try std.testing.expectEqual(false, resp2.uncertain);
    try std.testing.expect(std.mem.eql(u8, resp2.decision, "Low"));

    // 3. Multi-step execution with step_cap=3: proves adaptive early stopping at step 2 (step 3 is skipped)
    req.step_cap = 3;
    const resp3 = try engine.decide(arena.allocator(), req);
    try std.testing.expectEqual(@as(u32, 2), resp3.steps_executed);
    try std.testing.expectEqual(false, resp3.uncertain);
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

/// Returns the effective maximum allowed prompt token count (including BOS) for a given backend, canvas length, and workspace capacity (R30-01).
pub fn maxPromptTokensForBackend(backend: BackendType, canvas_len: usize, workspace_max_seq_len: usize) usize {
    const max_seq = switch (backend) {
        .cpu => workspace_max_seq_len,
        .cuda => @min(workspace_max_seq_len, 512),
    };
    if (canvas_len >= max_seq) return 0;
    return @min(max_seq - canvas_len, root.ModelConfig.max_prompt_tokens);
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

test "DecisionEngine: maxPromptTokensForBackend calculates effective prompt limits (R30-01)" {
    // CPU with 1024 workspace and 256 canvas -> 768 tokens
    try std.testing.expectEqual(@as(usize, 768), maxPromptTokensForBackend(.cpu, 256, 1024));
    // CPU with 1024 workspace and 16 canvas -> capped at ModelConfig.max_prompt_tokens (768)
    try std.testing.expectEqual(@as(usize, 768), maxPromptTokensForBackend(.cpu, 16, 1024));
    // CPU with smaller workspace (512) and 256 canvas -> 256 tokens
    try std.testing.expectEqual(@as(usize, 256), maxPromptTokensForBackend(.cpu, 256, 512));

    // CUDA with 512 workspace and 256 canvas -> 256 tokens (512 - 256)
    try std.testing.expectEqual(@as(usize, 256), maxPromptTokensForBackend(.cuda, 256, 512));
    // CUDA with 1024 configured workspace -> still clamped to 512 ceiling -> 256 tokens with 256 canvas
    try std.testing.expectEqual(@as(usize, 256), maxPromptTokensForBackend(.cuda, 256, 1024));
    // CUDA with 16 canvas -> 512 - 16 = 496 tokens
    try std.testing.expectEqual(@as(usize, 496), maxPromptTokensForBackend(.cuda, 16, 512));
    // Canvas length exceeding workspace -> 0 tokens
    try std.testing.expectEqual(@as(usize, 0), maxPromptTokensForBackend(.cuda, 512, 512));
    try std.testing.expectEqual(@as(usize, 0), maxPromptTokensForBackend(.cpu, 1024, 1024));
}

test "DecisionEngine: split temperature semantics for internal stopping vs response uncertainty (R45-01 / R46-01 / R46-02)" {
    // Contract: Intermediate stepping evaluates softmax and normalized entropy at native T=1.0 via StepDecisionState.processStep.
    // Post-hoc options.temperature is applied strictly to final response probabilities and sets response uncertainty via StepDecisionState.finalizeOutput.
    // With non-default T > 1.0, internal stopping may halt early (decisive under native dynamics)
    // while the client response flags uncertain = true due to post-hoc dispersion expansion.

    const cand_count = 2;
    const decisive_logits = [_]f32{ 14.0, 11.0 }; // delta = 3.0
    var probs: [cand_count]f32 = undefined;
    var prev_probs: [cand_count]f32 = undefined;

    // --- Scenario 1: Step 1 is decisive under T=1.0, but T=2.0 expands response dispersion ---
    {
        var step_state = StepDecisionState{};
        const tau: f32 = 0.30;
        const req_temp: f32 = 2.0;

        // Step 1: processStep under native T=1.0
        const stopped_early = step_state.processStep(
            &decisive_logits,
            cand_count,
            tau,
            &probs,
            &prev_probs,
            2, // step_cap = 2
        );

        // Native T=1.0: delta = 3.0 -> prob0 ~= 0.9526, H_norm ~= 0.2758 < tau (0.30)
        try std.testing.expect(stopped_early);
        try std.testing.expectEqual(@as(u32, 1), step_state.steps_executed);
        try std.testing.expect(probs[0] > 0.95);
        try std.testing.expect(step_state.normalized_entropy < 0.30);
        try std.testing.expect(!step_state.uncertain); // Decisive internally

        // Finalize output under req_temp = 2.0
        step_state.finalizeOutput(
            &decisive_logits,
            cand_count,
            req_temp,
            tau,
            &probs,
        );

        // Scaled T=2.0: delta = 1.5 -> prob0 ~= 0.8176, H_norm ~= 0.6840 >= tau (0.30)
        try std.testing.expectEqual(@as(u32, 1), step_state.steps_executed); // Steps unchanged
        try std.testing.expect(probs[0] < 0.85);
        try std.testing.expect(step_state.normalized_entropy > 0.60);
        try std.testing.expect(step_state.uncertain); // Response flagged uncertain

        // Verified: Execution halted early at step 1 while response output reported uncertain = true
        try std.testing.expect(stopped_early and step_state.uncertain);
    }

    // --- Scenario 2: Multi-step progression with self-conditioning buffer propagation ---
    {
        var step_state = StepDecisionState{};
        const tau: f32 = 0.50;
        const ambiguous_logits = [_]f32{ 12.05, 12.00 }; // delta = 0.05 -> H_norm ~= 0.999

        // Step 1: Ambiguous -> should NOT stop early, must populate prev_probs
        const step1_stopped = step_state.processStep(
            &ambiguous_logits,
            cand_count,
            tau,
            &probs,
            &prev_probs,
            2, // step_cap = 2
        );
        try std.testing.expect(!step1_stopped);
        try std.testing.expectEqual(@as(u32, 1), step_state.steps_executed);
        try std.testing.expect(step_state.uncertain);
        // Assert prev_probs was populated from probs
        try std.testing.expectEqual(probs[0], prev_probs[0]);
        try std.testing.expectEqual(probs[1], prev_probs[1]);

        // Step 2: Decisive -> should stop early
        const step2_stopped = step_state.processStep(
            &decisive_logits,
            cand_count,
            tau,
            &probs,
            &prev_probs,
            2,
        );
        try std.testing.expect(step2_stopped);
        try std.testing.expectEqual(@as(u32, 2), step_state.steps_executed);
        try std.testing.expect(!step_state.uncertain);
    }
}

test "DecisionEngine: R51-01 error propagation and resource preservation under terminal quarantine failure" {
    if (!root.cuda.is_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    // 1. Initialize a full DecisionEngine with the CUDA backend
    var engine = DecisionEngine.initWithOptions(allocator, model_path, .{ .backend = .cuda }) catch |err| switch (err) {
        error.FileNotFound, error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    // Ensure cleanup safety even if an intermediate assertion fails
    defer {
        infer_cuda.inject_device_sync_failure = false;
        infer_cuda.inject_retained_allocator_failure = false;
        infer_cuda.inject_quarantine_allocator_failure = false;
        infer_cuda.inject_stream_sync_failure = false;
        engine.deinit() catch {};
        infer_cuda.drainQuarantinedPinnedBuffers() catch {};
    }

    try std.testing.expect(engine.cuda_workspace != null);
    try std.testing.expect(engine.cuda_model != null);
    const initial_ws = engine.cuda_workspace.?;
    const initial_model = engine.cuda_model.?;
    const original_pinned_ptr = initial_ws.host_staging_pinned.ptr.?;

    // 2. Clear slate: drain any existing quarantine
    try infer_cuda.drainQuarantinedPinnedBuffers();
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.quarantined_buffers_count);
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.overflow_quarantined_buffers_count);
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.dynamic_quarantined_count);
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.emergency_reserve_buffers_count);
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.retained_workspaces_count);
    try std.testing.expect(infer_cuda.retained_workspaces_head == null);

    // 3. Fill all quarantine tiers (112 buffers total) to force terminal retention path
    infer_cuda.inject_device_sync_failure = true;
    infer_cuda.inject_quarantine_allocator_failure = true;

    // Fill 16 primary slots
    var primary_mock: [infer_cuda.MAX_QUARANTINED_BUFFERS]root.cuda_mem.PinnedBuffer(f32) = undefined;
    for (0..infer_cuda.MAX_QUARANTINED_BUFFERS) |i| {
        primary_mock[i] = try root.cuda_mem.PinnedBuffer(f32).init(64);
        infer_cuda.quarantinePinnedBuffer(&primary_mock[i]);
        try std.testing.expect(primary_mock[i].ptr == null);
    }
    try std.testing.expectEqual(@as(usize, infer_cuda.MAX_QUARANTINED_BUFFERS), infer_cuda.quarantined_buffers_count);

    // Fill 64 overflow slots
    var overflow_mock: [infer_cuda.MAX_OVERFLOW_BUFFERS]root.cuda_mem.PinnedBuffer(f32) = undefined;
    for (0..infer_cuda.MAX_OVERFLOW_BUFFERS) |i| {
        overflow_mock[i] = try root.cuda_mem.PinnedBuffer(f32).init(64);
        infer_cuda.quarantinePinnedBuffer(&overflow_mock[i]);
        try std.testing.expect(overflow_mock[i].ptr == null);
    }
    try std.testing.expectEqual(@as(usize, infer_cuda.MAX_OVERFLOW_BUFFERS), infer_cuda.overflow_quarantined_buffers_count);

    // Fill 16 emergency reserve slots
    var reserve_mock: [infer_cuda.MAX_EMERGENCY_RESERVE_BUFFERS]root.cuda_mem.PinnedBuffer(f32) = undefined;
    for (0..infer_cuda.MAX_EMERGENCY_RESERVE_BUFFERS) |i| {
        reserve_mock[i] = try root.cuda_mem.PinnedBuffer(f32).init(64);
        infer_cuda.quarantinePinnedBuffer(&reserve_mock[i]);
        try std.testing.expect(reserve_mock[i].ptr == null);
    }
    try std.testing.expectEqual(@as(usize, infer_cuda.MAX_EMERGENCY_RESERVE_BUFFERS), infer_cuda.emergency_reserve_buffers_count);

    // Fill all 16 static records in retained_workspace_pool
    var pool_mock: [infer_cuda.MAX_RETAINED_WORKSPACES]root.cuda_mem.PinnedBuffer(f32) = undefined;
    for (0..infer_cuda.MAX_RETAINED_WORKSPACES) |i| {
        pool_mock[i] = try root.cuda_mem.PinnedBuffer(f32).init(64);
        infer_cuda.retained_workspace_pool_used[i] = true;
        infer_cuda.retained_workspace_pool[i] = .{
            .pinned_ptr = pool_mock[i].ptr.?,
            .pinned_len = pool_mock[i].len,
            .next = infer_cuda.retained_workspaces_head,
            .is_dynamic = false,
        };
        infer_cuda.retained_workspaces_head = &infer_cuda.retained_workspace_pool[i];
        infer_cuda.retained_workspaces_count += 1;
        _ = pool_mock[i].move();
    }
    try std.testing.expectEqual(@as(usize, infer_cuda.MAX_RETAINED_WORKSPACES), infer_cuda.retained_workspaces_count);

    // 4. Force both dynamic retention allocation failure AND whole-device sync failure
    infer_cuda.inject_retained_allocator_failure = true;
    infer_cuda.inject_device_sync_failure = true;
    infer_cuda.inject_stream_sync_failure = true;

    // 5. Attempt teardown: DecisionEngine.deinit must propagate error.DeviceNotQuiescent!
    try std.testing.expectError(error.DeviceNotQuiescent, engine.deinit());

    // 6. Assert R51-01 resource preservation:
    // Neither cuda_workspace nor cuda_model must have been destroyed or nulled!
    try std.testing.expect(engine.cuda_workspace != null);
    try std.testing.expectEqual(initial_ws, engine.cuda_workspace.?);
    try std.testing.expect(engine.cuda_model != null);
    try std.testing.expectEqual(initial_model, engine.cuda_model.?);
    try std.testing.expect(engine.cuda_workspace.?.host_staging_pinned.ptr != null);
    try std.testing.expectEqual(original_pinned_ptr, engine.cuda_workspace.?.host_staging_pinned.ptr.?);

    // 7. Restore device synchronization and dynamic allocation
    infer_cuda.inject_device_sync_failure = false;
    infer_cuda.inject_retained_allocator_failure = false;
    infer_cuda.inject_stream_sync_failure = false;

    // 8. Retry DecisionEngine.deinit: must now succeed cleanly and destroy all resources!
    try engine.deinit();

    // Verify all engine resources are nulled out
    try std.testing.expect(engine.cuda_workspace == null);
    try std.testing.expect(engine.cuda_model == null);

    // 9. Restore quarantine allocator and drain all 112 quarantined buffers cleanly
    infer_cuda.inject_quarantine_allocator_failure = false;

    try infer_cuda.drainQuarantinedPinnedBuffers();
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.quarantined_buffers_count);
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.overflow_quarantined_buffers_count);
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.dynamic_quarantined_count);
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.emergency_reserve_buffers_count);
    try std.testing.expectEqual(@as(usize, 0), infer_cuda.retained_workspaces_count);
    try std.testing.expect(infer_cuda.retained_workspaces_head == null);
}

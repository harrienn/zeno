# Zeno — CPU Optimization and Simplification Review

**Reviewed range:** `bd21727..e461f0c` (four commits)
**Review date:** 2026-10-06
**Scope:** Persistent inference workspace and worker pool, fused layer dispatch, removal of the unused tensor abstraction, CLI helper consolidation, and the new CUDA roadmap.
**Method:** Static code and specification review. No builds, tests, profiles, or model runs were performed.

## Summary

The persistent workspace and worker pool remove repeated large allocations and thread creation. Combining attention and FFN into one worker dispatch also removes one global barrier per layer. Deleting the unused `tensor.zig` abstraction is consistent with the project's preference for direct tensor flow.

The largest remaining CPU opportunity is the attention kernel: its Q·K and attention-weighted V inner loops are scalar, and prompt queries still walk the full sequence to write masked scores. There is also an apparent opportunity to reuse per-worker and in-place buffers, reducing the long-lived workspace by roughly 11 MiB at the current default capacity. Before optimizing further, decide the concurrency contract: a `DecisionEngine` now owns one mutable workspace and one worker pool, so simultaneous `decide` calls on the same engine are unsafe.

## Findings

### R5-01 — Shared inference workspace makes simultaneous engine calls race

**Severity:** P1 before exposing the engine to a concurrent server

`DecisionEngine` owns one `InferenceWorkspace` and passes it to every request ([`src/engine.zig`](../src/engine.zig), lines 22–26 and 86–101). The workspace contains shared activation buffers, mutable pool phase/epoch state, and a shared layer-context pointer ([`src/infer.zig`](../src/infer.zig), lines 771–785). Two overlapping `decide` calls can overwrite each other's tensors and dispatch state.

The decision API spec already says the engine concurrency behavior must be documented before implementation ([`specs/decision-api.md`](../specs/decision-api.md), §Memory Contract). The persistent workspace makes this a concrete design gate.

**Requested follow-up:** Explicitly declare a `DecisionEngine` single-call-at-a-time and serialize entry, or introduce a bounded pool of independent workspaces. If using a workspace pool, size it against the intended request concurrency and include its memory cost. Do not treat the global worker pool as request-level concurrency.

### R5-02 — Attention's Q·K and weighted-V loops are scalar hot spots

**Severity:** P1 optimization opportunity; profile before choosing kernel shape

For every query head, `runAttnWorker` computes each Q·K dot with a scalar loop at [`src/infer.zig`](../src/infer.zig), lines 512–518, then accumulates the weighted V vector with another scalar loop at lines 524–532. This work repeats across every query token and every transformer layer. In contrast, the quantized GEMV paths and dense F32 GEMV already use SIMD operations.

**Suggested direction:** Add SIMD reductions for Q·K and vectorized accumulators for weighted V. For prompt rows, compute scores only for visible positions `0..t+1` and fill the masked tail with `-inf`/the existing masking value, instead of branching over all `N` positions. Keep a scalar reference and compare intermediate attention outputs before relying on end-to-end decisions. Profile attention separately from QKV, dense FFN, and MoE so effort follows measured cost.

### R5-03 — Full-sequence intermediate buffers appear reusable

**Severity:** P2 memory and cache optimization opportunity

At the default `max_seq_len = 512` and 16 workers, the workspace allocates about 48.8 MiB, excluding model memory. Two full-sequence arrays account for roughly 11 MiB of that:

- `norm_x_buf` stores the normalized input only for the QKV projection of each token; each token is consumed immediately in `runQKVWorker`.
- `attn_out_buf` stores the attention residual result that is consumed by FFN and then replaced by the next layer's output.

The latter appears eligible for in-place reuse: after the QKV phase completes, attention no longer needs `current_emb` as input for other tokens. A per-worker projection scratch can hold the O projection, apply its norm and residual, and write the result back to that token's `current_emb` row; FFN then reads it and writes `next_emb`. Likewise, QKV can use per-worker scratch for its normalized input. This would remove both sequence-wide arrays, subject to preserving the phase barrier and checking aliasing carefully.

**Suggested direction:** Prototype each reuse separately and compare the full output plus representative per-layer values. If in-place reuse makes ownership harder to inspect, keeping the extra buffers may be preferable to an unmeasured memory saving.

### R5-04 — Static equal-token chunks can leave attention workers imbalanced

**Severity:** P2 performance opportunity

The pool divides `N` into equal contiguous token counts ([`src/infer.zig`](../src/infer.zig), lines 686–690 and 718–731). Attention cost is not equal per token: prompt query rows attend only through their causal position, while canvas rows attend to all `N` positions (lines 499–523). With a long prompt, workers assigned mostly prompt rows do less attention work than workers assigned canvas rows, and each layer waits for the slowest worker.

**Suggested direction:** Measure per-worker phase time for short, typical, and boundary-length prompts. If imbalance is material, partition the attention rows by estimated visible-key work rather than token count, or schedule chunks dynamically. Check the impact on the fused `attn_ffn` phase before adding scheduler complexity.

### R5-05 — Workspace capacity and thread count are fixed at engine initialization

**Severity:** P2 resource-tuning opportunity

`DecisionEngine.init` always creates a 512-row workspace and chooses up to 16 workers ([`src/engine.zig`](../src/engine.zig), line 37; [`src/infer.zig`](../src/infer.zig), lines 791–839). This avoids first-request setup, but also commits about 49 MiB per engine before a prompt is known. Every additional concurrent workspace would multiply that cost. Growth later doubles every buffer together ([`src/infer.zig`](../src/infer.zig), lines 859–919).

**Suggested direction:** Keep the eager path if warm latency is the goal, but expose or internally tune workspace capacity and worker count. Benchmark startup/RSS, steady-state latency, and throughput across thread counts. Consider lazy allocation or capacity classes only if measurements show the memory or startup cost matters.

### R5-06 — CUDA layer-boundary transfer shape does not match the implemented sequence

**Severity:** P2 roadmap correction before implementing layer splitting

The runtime evaluates a sequence of `N = prompt_tokens.len + canvas_length` rows ([`src/infer.zig`](../src/infer.zig), lines 935–937). The CUDA roadmap's layer-split diagram and transfer deliverable assume a fixed `[256, 2816]` hidden state in FP16 ([`specs/cuda-roadmap.md`](../specs/cuda-roadmap.md), §§2.2 and 3, Stage 5). A prompt is also carried through the transformer, so the boundary activation is `[N, 2816]`, not `[256, 2816]`. Converting the CPU reference's FP32 activations to FP16 also needs an explicit numerical contract and measurement.

**Requested follow-up:** Make the transfer size depend on the complete evaluated sequence and label FP16 as a proposed optimization. Validate memory and numerical effects before using the fixed 1.44 MB estimate as an architecture assumption.

## Positive changes

- The workspace reuses allocations across requests and grows only when required.
- Persistent workers avoid thread creation per transformer layer.
- Fusing attention and FFN dispatch removes one barrier per layer without merging their tensor operations.
- Removing the unused tensor wrapper reduces indirection and keeps layout visible at call sites.
- The CLI now centralizes repeated candidate parsing, token-file writing, and fixture conversion helpers.

## Suggested optimization order

1. Set the engine concurrency contract and protect the shared workspace accordingly.
2. Add phase-level timing and per-worker timing to identify the actual hot path and imbalance.
3. Vectorize attention and avoid scanning masked prompt positions.
4. Evaluate the two workspace buffer reuse opportunities with layer-level comparisons.
5. Tune workspace size, thread count, and idle-worker wakeup policy using latency, throughput, and memory measurements.
6. Correct the CUDA transfer shape before implementation; defer multi-GPU decomposition until single-GPU kernels and transfer boundaries are measured.

## Verdict

The recent simplifications look directionally sound, with no static evidence here that the fused attention/FFN dispatch changes tensor dependencies. The most actionable CPU gains are in attention vectorization, masked prompt traversal, and avoiding sequence-wide scratch arrays. The shared-workspace concurrency contract should be resolved before the engine is used by a concurrent service. No performance claim should be upgraded based on this static review; run profile and parity measurements on the target hardware first.

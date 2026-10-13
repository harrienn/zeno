# Zeno — Development Plan

## 1. North Star

Build a lightweight Zig runtime implementing a **Jev-style structured decision engine** on DiffusionGemma. Reproducible decisions are a goal; probability calibration must be measured rather than assumed.

The first successful milestone is not maximum conversational output.

The first successful milestone is:

```text
application state + query
          ↓
seeded canvas using the model's configured width
          ↓
single-step forward read at a selected answer position
          ↓
candidate logit extraction and entropy (calibration to be evaluated)
          ↓
typed decision + candidate scores and entropy
```

running through Zeno with behavior compared against [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) and [llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423). These references do not by themselves validate Zeno's implementation.

---

## 2. Development Strategy

Zeno development follows this order:

```text
semantic understanding
        ↓
minimal implementation
        ↓
reference validation
        ↓
end-to-end execution
        ↓
CUDA execution
        ↓
profiling
        ↓
optimization
```

Do not reverse this order without an explicit reason.

---

## 3. Phase 0 — Repository Foundation

Goal:

Create the smallest useful Zig project structure.

Expected outcomes:

* build system works (`build.zig`);
* tests work;
* basic project conventions exist;
* `README.md`, `AGENTS.md`, and `ARCHITECTURE.md` are present and aligned;
* reference material is documented;
* no speculative framework code is introduced.

Exit condition:

```text
zig build
zig build test
```

work reliably in the development environment.

The repository has progressed beyond the initial scaffold: source modules now cover GGUF parsing, tokenizer data, tensors, quantization, operations, model representation, canvases, and early decision flow. These components are not yet evidence of a complete or reference-validated inference path; Phase 3 remains the end-to-end structured-read milestone.

---

## 4. Phase 1 — Runtime Primitives

Goal:

Establish the minimum runtime primitives required by the model.

Likely areas:

```text
types
memory (ownership and arena allocation)
tensor representation (shape, stride, layout)
dtype handling (f32, quantized block formats like Q4_K, dequantization)
basic operations
error handling
```

This phase should establish semantic contracts, not optimize kernels.

Exit condition:

Core operations can be tested independently of the full model.

---

## 5. Phase 2 — Model Representation

Goal:

Represent the required DiffusionGemma model structure in Zig.

Focus on:

* GGUF container parsing (header, metadata, tensor descriptors);
* model configuration decoding (`diffusion.canvas_length = 256`, 30 blocks, 128 MoE experts with 8 active);
* tokenizer vocabulary and special tokens (`mask_token_id = 4`, `bos_token_id = 2`, `eos_token_id = 1`);
* **tokenizer `encode()`** — the BPE encoder itself: pre-tokenization, merge rank semantics, and byte fallback. *Added after review: the vocabulary alone cannot place query text on the canvas, and this deliverable was previously owned by no phase.* See [`specs/UNRESOLVED.md`](specs/UNRESOLVED.md) U-04;
* model weights and tensor mapping;
* model components (token embeddings, bidirectional non-causal attention, MoE feed-forward blocks, RMSNorm);
* data flow between components.

Do not yet introduce broad multi-model abstractions.

Exit condition:

The model structure and weights can be loaded, represented, and inspected without executing the complete generation loop.

---

## 6. Phase 3 — Single-Step Structured Read (System One Primitive)

Goal:

Implement one semantically correct structured read over a seeded canvas.

The core execution questions are:

```text
1. Given an application state and typed query schema,
   how is the configured canvas seeded, and how are answer positions represented and pinned?

2. How does a single forward evaluation over the canvas
   produce logits across candidate decision tokens?

3. How are candidate probabilities and Shannon entropy
   derived at answer positions, and how well do those probabilities calibrate?
```

Validation strategy:

Compare intermediate activations, output logits, and entropy scores against [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) and [llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423), while documenting differences between their execution paths. The vLLM reference requires each candidate answer to be a single token (or represented by a one-token code).

Exit condition:

A seeded canvas query produces CPU results that match the chosen reference behavior within documented tolerances; calibration is evaluated on a labeled dataset and reported separately from normalization and entropy.

**Parity gate.** A parity claim requires all Phase 3 implementation questions and parity questions U-09 through U-18 and U-22/U-23 to be `resolved` with evidence, plus independently captured reference fixtures and validated per-field comparisons. U-11 pins the vLLM oracle; U-12 pins llama.cpp. Implementation gating questions U-01 through U-12, U-14 through U-20, and U-23 are resolved with evidence; all 5 golden fixtures reproduce exact discrete decisions and uncertainty flags. However, numerical reference raw-logit parity gates U-13 and U-22 remain open in [`specs/UNRESOLVED.md`](specs/UNRESOLVED.md) because measured raw-logit differences against GGML SIMD dequantization exceed the recorded reference tolerance. Per [`AGENTS.md`](AGENTS.md) §5, the internal CPU implementation serves as the authoritative semantic baseline for subsequent backend work while external raw-logit reconciliation proceeds.

Parity is asserted exclusively through the golden-fixture format in [`specs/UNRESOLVED.md`](specs/UNRESOLVED.md) §5.

---

## 7. Phase 4 — Adaptive Decisions & Iterative Refinement

Goal:

Investigate whether follow-up reads or iterative refinement improve structured decisions, then implement only the empirically supported step-control behavior.

Conceptually:

```text
receive query schema (boolean, classification, score)
                ↓
seed configured canvas and select answer positions
                ↓
step 1: parallel forward read
                ↓
evaluate candidate entropy at answer positions
                ↓
[entropy <= threshold?] ──► return decision with measured candidate scores
                │
         (high entropy / multi-slot)
                ↓
follow-up read, new noise draw, or specified refinement (select by experiment)
                ↓
return resolved typed decision
```

Focus on:

* schema-to-canvas template mapping;
* compare one read, repeated reads with new noise, and same-canvas refinement before selecting a step policy;
* state lifetime and per-step arena allocators;
* Jev-compatible `/v1/systemone` request/response interface;
* typed schema validation.

Exit condition:

Zeno can execute end-to-end structured decision queries on the CPU reference path, with candidate probabilities and entropy reported separately from any calibration metrics.

---

## 8. Phase 5 — CUDA & Multi-GPU Execution

Goal:

Accelerate structured reads using CUDA while strictly preserving the established semantic behavior of the CPU reference implementation.

Research topics informed by the [separate experimental community patch for llama.cpp PR #24423](https://github.com/readwrightexecute/llamacpp-diffusiongemma-multigpu):

* **Single-device baseline**: establish end-to-end correctness, operator coverage, and precise memory footprint on a single GPU first;
* **Prompt cache lifetime**: establish which context can be reused and whether reuse applies within a query, across steps, or across queries;
* **Device-resident self-conditioning**: measure whether retaining state on GPU devices across steps helps the chosen execution path (Phase 4+);
* **Multi-GPU decomposition**: evaluate pipeline layer splitting across available devices if memory fit requires sharding; the community patch tests a specific two-GPU layer split;
* **Communication cost**: measure P2P transfer latency over PCIe before relying on multi-GPU pipelining.

Approach:

```text
CPU reference implementation (semantic ground truth)
      ↓
CUDA build integration & device discovery spike (-Dcuda, nvcc, C ABI)
      ↓
Single-device elementwise and quantized GEMV kernels (Q4_K, Q6_K, Q8_0)
      ↓
Single-device transformer forward pass
      ↓
Multi-GPU pipeline layer splitting (if sharding required by memory)
      ↓
latency profiling & reproducible benchmarking
```

Exit condition:

CUDA execution produces decisions and candidate metrics matching the CPU reference implementation within documented numerical tolerances; any throughput and latency gains are demonstrated by reproducible benchmarks.

*Status: COMPLETED (Stages 1–6)*. Full 30-layer forward pass implemented with dual-GPU layer partitioning (Strategy A), device-side batched MoE, 5/5 golden decision fixtures passing, and benchmark harness established.

---

## 9. Phase 6 — Profiling and Optimization

Optimization begins only after:

* end-to-end execution is correct;
* CPU/reference behavior is stable;
* CUDA behavior is validated.

Measure before changing architecture.

*Status: COMPLETED (Phase 6 Optimizations & Graph Capture)*.
* **WP1 (Scale Precomputation & 4x Batched GEMV Token Tiling)**: Precomputed Q4_K scales with 1-cycle `__shfl_sync` warp broadcasts across `gemv.cu` and `moe.cu`. Implemented 4x token batching for `gemv_q4k`, `gemv_q8_0`, `gemv_q6k`, and `gemv_q5_0`, slashing DRAM weight traffic by up to 4x.
* **WP2 (1x4 Warp Coarsening for MoE Projections)**: Coarsened `moe_gemv_down_batched_kernel` and `moe_gemv_gate_up_batched_kernel` to compute 4 output rows per warp, cutting threadblock scheduler launches by 4x and maximizing L1/register activation reuse.
* **WP3 (CUDA Graph Capture & Zero-Overhead Launch)**: Implemented stream graph capture and replay for partition streams on Device 0 and Device 1, eliminating hundreds of individual driver launches per step.
* **Redundant Host Stalls Eliminated**: Removed 52 synchronous blocking `cudaMemset` calls on intermediate buffers that are unconditionally written by their producing kernels.
* **Benchmark Impact (Exploratory Measurements)**:
  * *(Environment: Dual NVIDIA GeForce RTX 3060 12GB GPUs, driver 550.120, CUDA 12.4, ReleaseFast build. Harness updated in Round 23 to execute repeated post-warmup samples ($N=3$) and report median and spread to establish multi-sample statistical rigor.)*
  * $C=16$ single-step latency reduced from **738.2 ms** to **600.93 ms** (**18.6% speedup**, shaving ~137 ms).
  * $C=256$ baseline latency reduced from **4668 ms** to **3801 ms** (shaving **867 ms**).
  * 100% discrete decision match and probability $\Delta \le 0.05$ preserved across all 5 golden fixtures.

Approach:

```text
hypothesis
    ↓
implementation
    ↓
benchmark
    ↓
comparison
```

---

## 10. Research Track

Zeno is also a research project into discrete diffusion as a decision paradigm.

### Reported Model Metadata (Artifact Provenance Needed)

The following values are reported from model/GGUF inspection, but the inspected artifact and revision must be recorded before treating them as project-wide invariants:

* **Canvas length**: 256 tokens (`diffusion.canvas_length = 256`); serving implementations may support a shorter request canvas.
* **Non-causal attention**: attention is bidirectional (`diffusion-gemma.attention.causal = false`).
* **Special tokens**: BOS is 2, EOS is 1, mask token is 4 (`tokenizer.ggml.mask_token_id = 4`).
* **Backbone**: Gemma 4 MoE (30 transformer blocks, 128 experts with 8 active, embedding dimension 2816).

### Open Research Questions

The following questions should remain visible during implementation:

```text
Calibration & Entropy: How well-calibrated are pre-softmax / softmax logits for binary and multi-class decisions in a single pass?

Step Capping: What is the empirical trade-off between single-step read latency and multi-step denoising accuracy?

Canvas Geometry: How does answer-position placement within the configured canvas affect attention and decision quality?

Multi-GPU Communication: What is the optimal tensor-parallel decomposition for 128 MoE experts during 1-step reads?

Comparison with Autoregressive Models: How does structured diffusion compare against grammar-constrained autoregressive decoding in latency, throughput, and error rates?
```

Research findings should be written down in `specs/` when they become sufficiently supported.

Hypotheses must be clearly distinguished from verified behavior.

---

## 11. Decision Rules

When multiple implementations are possible, prefer the one that:

1. makes the semantic model easier to understand;
2. introduces fewer abstractions;
3. makes ownership easier to reason about;
4. keeps the reference path straightforward;
5. can later be optimized without changing semantics.

Do not optimize for hypothetical future requirements.

---

## 12. Current Work Rule

At any point, keep one clearly defined immediate objective.

The active objective should fit in one sentence.

Example:

```text
Implement and validate the CPU representation of one denoising step.
```

**Current objective:**

```text
Implement token-sorted expert batched GEMV/GEMM and prompt KV reuse to compress MoE weight streaming toward the sub-100ms decision latency target.
```

This active objective builds upon the completed Phase 6 CUDA Graph capture, 4x GEMV batching, and 1x4 MoE warp coarsening (which brought C=16 latency to 600ms), focusing on token sorting by expert to load expert weights exactly once per layer.

---

## 13. Definition of Done

A task is not complete merely because code exists.

A task should be considered complete when:

```text
implementation
    +
compiler validation
    +
relevant tests
    +
reference comparison where applicable
    +
documented assumptions
```

are all satisfied.

For experimental work, record what was learned even when the implementation is discarded.

---

## 14. Plan Maintenance

This document describes project direction, not detailed implementation.

Update `PLAN.md` when:

* a phase is completed;
* the project direction changes;
* a major research question is answered;
* an architectural decision changes sequencing;
* a previously assumed requirement is disproven.

Do not put low-level API details in this file.

Detailed behavior belongs in `specs/`.

In particular: when a specification defers a question, the deferral must be registered in [`specs/UNRESOLVED.md`](specs/UNRESOLVED.md) rather than left as inline prose. Unregistered deferrals are not visible from the plan, do not gate any phase, and are the mechanism by which a faithful implementation ends up incorrect.

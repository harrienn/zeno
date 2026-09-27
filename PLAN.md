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

The initial build scaffold and configuration constants are present. GGUF loading, tokenizer support, and model execution remain future work.

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

Accelerate structured reads using CUDA and multi-GPU tensor parallelism while strictly preserving the established semantic behavior.

Research topics informed by the [separate experimental community patch for llama.cpp PR #24423](https://github.com/readwrightexecute/llamacpp-diffusiongemma-multigpu):

* **Prompt cache lifetime**: establish which context can be reused and whether reuse applies within a query, across steps, or across queries;
* **Device-resident self-conditioning**: measure whether retaining state on GPU devices across steps helps the chosen execution path;
* **Sharding strategy**: compare layer and tensor splits for the target model and supported hardware; the community patch tests a specific two-GPU layer split;
* **Communication cost**: measure transfer and collective costs before choosing an optimization.

Approach:

```text
CPU reference
      ↓
CUDA single-device forward pass
      ↓
Multi-GPU tensor parallel sharding
      ↓
verify numerical equivalence with reference
      ↓
latency profiling & benchmarking
```

Exit condition:

CUDA and multi-GPU execution produces decisions and candidate metrics within documented tolerances of the CPU reference implementation; any throughput and latency gains are demonstrated by reproducible benchmarks.

---

## 9. Phase 6 — Profiling and Optimization

Optimization begins only after:

* end-to-end execution is correct;
* CPU/reference behavior is stable;
* CUDA behavior is validated.

Measure before changing architecture.

Potential targets include:

```text
memory movement
kernel launch overhead
matrix operations
attention
normalization
embedding operations
denoising loop overhead
allocation
```

Each optimization should have:

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

Avoid parallel architectural changes while the active objective is unresolved.

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

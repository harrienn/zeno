# Zeno — Architecture

## 1. Purpose

Zeno is a research project for a lightweight, Jev-style structured decision runtime built in Zig and based on DiffusionGemma.

Rather than serving as a conversational text generator, the architecture is organized around **structured decision execution** (structured reads) over a seeded token canvas:

```text
Application State + Query Schema
               ↓
Seeded Token Canvas (pinned noisy decision slots)
               ↓
Parallel Forward Read (single-step / capped diffusion)
               ↓
Candidate Logit Extraction & Entropy (calibration to be evaluated)
               ↓
Structured Decision (typed label, boolean, or score)
```

The runtime makes this execution process explicit, reproducible, inspectable, and efficient. The repository provides a complete CPU reference implementation and an accelerated sub-100ms dual-GPU CUDA runtime.

Zeno remains smaller in scope than a general-purpose conversational framework, focusing exclusively on deterministic System One decision execution.

---

## 2. Architectural Goals

Zeno has five primary goals:

1. reproduce the structured-reads and discrete diffusion behavior of DiffusionGemma (reference: vLLM PR #57250);
2. execute reproducible, schema-constrained "System One" decisions and report candidate scores and entropy, with calibration to be evaluated;
3. provide a clean CPU/reference execution path for canvas seeding, forward evaluation, and logit extraction;
4. investigate CUDA and multi-GPU execution, using llama.cpp PR #24423 and a separate experimental community patch as references; cache lifetime, self-conditioning placement, and sharding strategy remain design questions;
5. maintain a small, understandable foundation without unnecessary abstraction layers.

Performance matters for System One decisions, but semantic correctness and model understanding come first. Calibration and performance must be measured.

---

## 3. High-Level Structure

The system is conceptually divided into the following layers:

```text
┌─────────────────────────────────────────┐
│                 Frontend                │
│    CLI / Jev System One Decision API    │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│             Decision Engine             │
│  canvas seeding / pinned slots / read   │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│              Model Runtime              │
│       canvas state / execution step     │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│               Model Layer               │
│   embeddings / MoE blocks / attention   │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│                Tensor                   │
│     shape / dtype / layout / storage    │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│                Backend                  │
│       CPU reference / CUDA / Multi-GPU  │
└─────────────────────────────────────────┘
```

These are conceptual boundaries, not necessarily separate packages or modules.

The implementation should remain free to collapse layers when doing so improves simplicity.

---

## 4. Frontend

The frontend provides the interface for structured decision queries and runtime diagnostics.

Responsibilities include:

* receiving structured query requests (e.g. Jev-compatible `/v1/systemone` format or native Zig API calls);
* parsing query schemas (boolean judgments, multiple-choice classification, ordinal scoring);
* configuring step limits (e.g. single-step read vs capped iterative refinement);
* formatting responses with typed labels, candidate scores, normalized probabilities where appropriate, and Shannon entropy; calibration must be evaluated separately;
* exposing basic runtime diagnostics and performance metrics.

The frontend does not contain model execution semantics.

It should be possible to change the CLI without changing the denoising engine.

---

## 5. Model Runtime

The planned model runtime will own the execution process.

Its primary responsibility is to coordinate:

```text
input
  ↓
initial canvas/state
  ↓
denoising step
  ↓
updated state
  ↓
repeat
  ↓
final decoded representation
```

The runtime controls temporal state.

It should not own low-level kernel implementation details.

---

## 6. Model Layer

The planned model layer represents the computation performed by DiffusionGemma. The reference execution path has distinct causal encoder/prefill and bidirectional decoder/denoising modes, with KV state and self-conditioning crossing those phases. A Zeno forward-pass contract must preserve this distinction rather than treating inference as one undifferentiated bidirectional pass.

Conceptually:

```text
model configuration (GGUF metadata)
        +
weights (quantized tensor storage)
        +
runtime tensors
        ↓
model computation (embeddings, bidirectional attention, MoE blocks, RMSNorm)
        ↓
denoising prediction / model output
```

Model code should express model semantics rather than backend-specific optimization.

The target DiffusionGemma weights and configuration are expected in GGUF containers (for example, Q4_K_M quantized tensors). Model facts and tensor mappings should be tied to a named model artifact and revision before implementation. The target architecture uses a Gemma-based Mixture-of-Experts (MoE) network with bidirectional (non-causal) attention.

The same model operation should have the same meaning regardless of whether it executes on CPU or CUDA.

---

## 7. Tensor Layer

Tensor is a fundamental semantic abstraction.

A tensor is not merely a pointer to memory.

The runtime must be able to reason about:

```text
shape
dtype
layout
storage
device
```

These properties must not be hidden when they affect correctness.

Tensor abstractions should remain lightweight.

Avoid creating a large tensor framework unless Zeno actually requires one.

---

## 8. Backend Layer & Multi-GPU Execution

The backend provides execution primitives.

The initial conceptual backends are:

```text
CPU
CUDA (Single-GPU)
CUDA (Multi-GPU Tensor Parallel)
```

The CPU path is intended to serve as the semantic reference implementation.

CUDA implementations must preserve the exact semantic contract of the reference path.

For multi-GPU execution, Zeno may investigate lessons from the [separate community patch for llama.cpp PR #24423](https://github.com/readwrightexecute/llamacpp-diffusiongemma-multigpu). That patch describes device-resident self-conditioning and prompt-KV caching for a specific two-GPU layer-split setup. It does not establish a general tensor-splitting strategy or behavior across other hardware layouts. Zeno's cache lifetime, sharding, and communication strategy remain open design questions.

---

## 9. Decision Engine & Structured Reads

The decision engine is the conceptual center of Zeno.

Rather than generating sequential conversational text token-by-token:

```text
token 1
   ↓
token 2
   ↓
token 3
   ↓
...
```

Zeno is intended to operate over a seeded token canvas (the inspected model metadata reports 256 tokens) and execute **structured reads**, using [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) as a behavioral reference. Treat the canvas length as model metadata, not a universal fixed width.

### 9.1 Structured Reads Mode (System One / Jev-Style Decisions)

In structured-reads mode, the engine evaluates software decisions in a single or step-capped pass:

```text
1. Seed Canvas: Place input context and an answer template onto the configured canvas.
2. Select Answer Positions: Mark the target positions and establish their exact seed/pinning behavior from the reference implementation.
3. Forward Read: Execute a single parallel forward pass with bidirectional attention.
4. Logit Readout: Read logits exclusively at the pinned slot index for allowed candidate tokens.
5. Score: Normalize candidate logits if appropriate and compute Shannon entropy. Calibration requires separate empirical validation.
```

If the implementation restricts readout to the schema's candidate token IDs and validates the mapping, the selected label can be constrained to those choices. The vLLM reference requires each candidate answer to be a single token; clients can map longer labels to one-token codes. This constraint alone does not establish correctness or probability calibration.

### 9.2 Follow-up Reads and Iterative Refinement (Research Question)

The project has not established whether a high-entropy result should trigger another read of the same canvas, a new noise draw, or a denoising step. These are distinct strategies and should be compared experimentally before one becomes runtime behavior. Multi-token structured infilling is also outside the currently established read semantics.

```text
initial seeded canvas
        ↓
read step 1 (evaluate entropy at pinned slots)
        ↓
[entropy > threshold?]
        ↓
perform a separately specified follow-up read or refinement
        ↓
converged decision state
```

Questions to resolve:
* **Bidirectional decoder attention**: the decoder evaluates the canvas bidirectionally; causal prompt prefill and its KV state remain a distinct phase.
* **Cache behavior**: determine which prompt/model state can be reused and for what lifetime; do not assume caching across queries.
* **Direct logit readout**: classification judgments are extracted directly from logits at pinned positions without free-form token emission.

The exact mathematical semantics of structured reads and any denoising behavior belong in dedicated specifications in `specs/` before those behaviors are implemented.

---

## 10. Compile-Time vs Runtime Boundary

Zeno intentionally separates static structure from runtime state.

### Compile-time information

Potential examples include:

```text
model configuration
static type information
specialized dimensions
dispatch structure
kernel configuration
compile-time validation
```

### Runtime information

Examples include:

```text
weights
input
token canvas
denoising state
activations
sampling state
GPU memory
allocator state
```

The architecture should avoid turning runtime model state into compile-time structure merely for abstraction convenience.

---

## 11. Reference vs Optimized Paths

The project should maintain a distinction between:

```text
semantic reference
        ↓
optimized implementation
```

For example:

```text
CPU reference attention
        ↓
CUDA attention kernel
```

The optimized implementation should be judged against the reference behavior.

This distinction is especially important while the model semantics are still being investigated.

---

## 12. External References

Zeno relies on two key external implementations as behavioral references:

1. **[vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250)** (`examples/features/structured_diffusion/structured_server.py`):
   - Reference for a prototype **structured-reads mode**, seeded canvas construction, selected answer positions, and a sample interposer providing the Jev-compatible `/v1/systemone` API. The PR notes that this is not a standard vLLM endpoint and that candidate answers must be single tokens.

2. **[llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423)** and the [separate experimental Multi-GPU patch](https://github.com/readwrightexecute/llamacpp-diffusiongemma-multigpu):
   - PR reference for DiffusionGemma GGUF execution (`llama-diffusion-cli`) and non-causal attention. The community patch explores a specific two-GPU setup; its behavior should not be generalized without independent validation.

However:

```text
External reference architecture
              ≠
       Zeno architecture
```

Zeno borrows verified semantics while intentionally avoiding unnecessary historical or framework complexity.

The reference runtimes are oracles for behavior, not templates for internal design.

---

## 13. Data Flow

The structured decision data flow is:

```text
Application State & Typed Query
              ↓
Schema Mapping & Template Construction
              ↓
Seeded Token Canvas (pinned noisy slots)
              ↓
Parallel Model Evaluation (bidirectional attention)
              ↓
Pinned Slot Logit Readout (restricted to candidate tokens)
              ↓
Candidate Normalization & Shannon Entropy (calibration evaluated separately)
              ↓
Decision Output with Candidate Scores and Entropy (calibration to be evaluated)
```

Each transition corresponds to a mechanically verifiable runtime step.

---

## 14. Ownership of Responsibilities

A useful responsibility split is:

```text
Frontend
    owns user interaction, HTTP endpoints, and CLI queries

Decision Engine / Schema Mapper
    owns schema-to-canvas mapping, answer-position selection, logit extraction, and any empirically validated calibration

Runtime
    owns execution state, canvas lifetime, and step control flow

Model
    owns model semantics, MoE routing, and bidirectional attention

Tokenizer
    owns encoding input text to tokens and decoding candidate labels

Tensor
    owns tensor representation and quantized layout

Backend
    owns execution mechanics (CPU, CUDA, and Multi-GPU tensor splitting)

Allocator / Memory
    owns lifetime mechanics and step arena allocations
```

These responsibilities may be implemented with fewer concrete modules.

Conceptual separation is more important than file-count separation.

---

## 15. Architectural Constraints

The following constraints are intentional:

* Do not build a general inference framework.
* Do not optimize before semantic correctness is established.
* Do not hide memory ownership.
* Do not hide tensor layout when it matters.
* Do not duplicate model semantics across backends.
* Do not introduce abstractions solely to support hypothetical future models.
* Do not allow GPU kernels to silently redefine the model.
* Keep reference implementations available while semantics are under investigation.

---

## 16. Evolution Strategy

Zeno should evolve approximately in this direction:

```text
Understand
    ↓
Reference implementation
    ↓
Minimal CPU runtime
    ↓
End-to-end denoising
    ↓
CUDA execution
    ↓
Profiling
    ↓
Selective optimization
```

Optimization may later change internal structure.

Model semantics should remain the stable center.

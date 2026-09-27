# Zeno — Architecture

## 1. Purpose

Zeno is a lightweight, Jev-style structured decision engine built on DiffusionGemma implemented in Zig.

Rather than serving as a conversational text generator, the architecture is organized around **structured decision execution** (structured reads) over a seeded token canvas:

```text
Application State + Query Schema
               ↓
Seeded Token Canvas (pinned noisy decision slots)
               ↓
Parallel Forward Read (single-step / capped diffusion)
               ↓
Logit Extraction & Calibrated Confidence
               ↓
Structured Decision (typed label, boolean, or score)
```

The runtime exists to make this execution process explicit, deterministic, inspectable, and high-performance.

Zeno remains substantially simpler and faster than a general-purpose conversational framework.

---

## 2. Architectural Goals

Zeno has five primary goals:

1. reproduce the structured-reads and discrete diffusion behavior of DiffusionGemma (reference: vLLM PR #57250);
2. execute deterministic, schema-constrained "System One" decisions with calibrated probabilities and entropy metrics;
3. provide a clean CPU/reference execution path for canvas seeding, forward evaluation, and logit extraction;
4. provide an efficient CUDA and multi-GPU execution path (reference: Multi-GPU patch for llama.cpp PR #24423) that preserves prompt caching, device-resident self-conditioning, and tensor parallelism;
5. maintain a small, understandable foundation without unnecessary abstraction layers.

Performance is critical for System One decisions, but correctness, calibrated confidence, and model understanding come first.

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
* formatting responses with typed labels, calibrated probabilities, and Shannon entropy;
* exposing basic runtime diagnostics and performance metrics.

The frontend does not contain model execution semantics.

It should be possible to change the CLI without changing the denoising engine.

---

## 5. Model Runtime

The model runtime owns the execution process.

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

The model layer represents the computation performed by DiffusionGemma.

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

DiffusionGemma weights and configuration are loaded from GGUF containers (e.g. Q4_K_M quantized tensors). The underlying architecture is a Gemma-based Mixture-of-Experts (MoE) network with bidirectional (non-causal) attention.

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

The CPU path serves as the reference implementation.

CUDA implementations must preserve the exact semantic contract of the reference path.

For multi-GPU execution, Zeno incorporates lessons from the **Multi-GPU patch for llama.cpp PR #24423**:
* **Device-resident state**: self-conditioning tensors and layer activations remain resident on GPU devices to avoid host synchronization.
* **Prompt-cache awareness**: static prompt contexts are cached across decision steps without being invalidated by multi-GPU layer splitting.
* **Balanced tensor splitting**: transformer blocks and large MoE feed-forward weight tensors (128 experts) are split across available devices while minimizing cross-device collective communication overhead during single-step forward reads.

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

Zeno operates over a seeded token canvas (256 tokens in DiffusionGemma) and executes **structured reads** (as pioneered in **vLLM PR #57250**):

### 9.1 Structured Reads Mode (System One / Jev-Style Decisions)

In structured-reads mode, the engine evaluates software decisions in a single or step-capped pass:

```text
1. Seed Canvas: Place input context + schema template onto the 256 canvas.
2. Pin Slots: Leave target decision positions as noise / mask tokens (mask_token_id = 4).
3. Forward Read: Execute a single parallel forward pass with bidirectional attention.
4. Logit Readout: Read logits exclusively at the pinned slot index for allowed candidate tokens.
5. Calibrate: Apply softmax over candidates and compute Shannon entropy.
```

Because output is strictly constrained to the candidate tokens defined by the query schema, outputs are guaranteed to be type-safe and never go "off-schema."

### 9.2 Confidence-Guided Iterative Denoising

When a decision slot exhibits high entropy (uncertainty) or when a query requests multi-token structured infilling, the engine can adaptively execute capped iterative steps:

```text
initial seeded canvas
        ↓
read step 1 (evaluate entropy at pinned slots)
        ↓
[entropy > threshold?]
        ↓
refine step 2 (denoise ambiguous positions)
        ↓
converged decision state
```

Key execution distinctions:
* **Bidirectional attention**: the network attends across the entire token canvas without causal masking.
* **No accumulating KV cache**: generation does not append new keys and values step-by-step; each step refines the canvas representation in parallel.
* **Direct logit readout**: classification judgments are extracted directly from logits at pinned positions without free-form token emission.

The exact mathematical semantics of structured reads and denoising belong in dedicated specifications in `specs/`.

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

1. **vLLM PR #57250** (`examples/features/structured_diffusion/structured_server.py`):
   - Reference for **structured-reads mode**, seeded canvas construction, pinned mask positions, and the Jev-compatible `/v1/systemone` decision API.

2. **llama.cpp PR #24423 & Multi-GPU patch**:
   - Reference for DiffusionGemma GGUF execution (`llama-diffusion-cli`), non-causal attention implementation, and multi-GPU tensor splitting with preserved self-conditioning.

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
Softmax Normalization & Shannon Entropy
              ↓
Calibrated Decision & Confidence Output
```

Each transition corresponds to a mechanically verifiable runtime step.

---

## 14. Ownership of Responsibilities

A useful responsibility split is:

```text
Frontend
    owns user interaction, HTTP endpoints, and CLI queries

Decision Engine / Schema Mapper
    owns schema-to-canvas mapping, pinned slot placement, logit extraction, and calibration

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

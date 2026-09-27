# Zeno — Architecture

## 1. Purpose

Zeno is a small inference runtime for DiffusionGemma implemented in Zig.

The architecture is organized around one central execution model:

```text
DiffusionGemma
      ↓
token canvas
      ↓
iterative denoising
      ↓
text
```

The runtime exists to make this execution process explicit, inspectable, and eventually efficient.

Zeno should remain substantially simpler than a general-purpose inference framework.

---

## 2. Architectural Goals

Zeno has five primary goals:

1. reproduce the semantic behavior of DiffusionGemma;
2. make the denoising execution model understandable;
3. provide a clean CPU/reference execution path;
4. provide a CUDA execution path without changing model semantics;
5. create a small foundation for future performance work.

Performance is important, but correctness and model understanding come first.

---

## 3. High-Level Structure

The system is conceptually divided into the following layers:

```text
┌─────────────────────────────────────────┐
│                 Frontend                │
│          CLI / inference request        │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│              Model Runtime              │
│      model state / denoising loop       │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│               Model Layer               │
│   embeddings / transformer / outputs    │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│                Tensor                   │
│     shape / dtype / layout / storage    │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│                Backend                  │
│         CPU reference / CUDA            │
└─────────────────────────────────────────┘
```

These are conceptual boundaries, not necessarily separate packages or modules.

The implementation should remain free to collapse layers when doing so improves simplicity.

---

## 4. Frontend

The frontend provides the user-facing inference interface.

Responsibilities include:

* receiving input;
* configuring generation;
* selecting a model;
* reporting output;
* exposing basic runtime diagnostics.

The frontend should not contain model semantics.

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

## 8. Backend Layer

The backend provides execution primitives.

The initial conceptual backends are:

```text
CPU
CUDA
```

The CPU path serves as the reference implementation.

CUDA is an execution backend, not a separate model implementation.

For example:

```text
Model operation
      │
      ├── CPU implementation
      │
      └── CUDA implementation
```

The operation's semantic contract remains the same.

---

## 9. Denoising Engine

The denoising engine is the conceptual center of Zeno.

Unlike autoregressive generation, generation is not modeled as:

```text
token 1
   ↓
token 2
   ↓
token 3
   ↓
...
```

Instead, the runtime operates over a fixed token canvas (256 tokens in DiffusionGemma) and repeatedly updates its state:

```text
initial canvas (prompt + mask tokens)
      ↓
denoise step 1
      ↓
denoise step 2
      ↓
denoise step 3
      ↓
...
      ↓
converged canvas
```

Key execution distinctions:
* **Bidirectional attention**: the network attends across the entire token canvas without causal masking.
* **No accumulating KV cache**: generation does not append new keys and values step-by-step; each step refines the canvas representation in parallel.
* **Discrete token refinement**: positions transition from noise/mask placeholders into high-confidence token identities.

The exact mathematical semantics of a denoising step belong in the dedicated denoising specification, not in this document.

This document only establishes that denoising is an iterative state transformation.

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

llama.cpp is a valuable reference for understanding and validating runtime behavior.

However:

```text
llama.cpp architecture
        ≠
Zeno architecture
```

Zeno should borrow verified semantics while intentionally avoiding unnecessary historical or framework complexity.

The reference runtime is an oracle for behavior where appropriate, not a mandate for internal structure.

---

## 13. Data Flow

The intended conceptual data flow is:

```text
User Input
    ↓
Input Representation
    ↓
Initial Token Canvas
    ↓
Model Evaluation
    ↓
Denoising Update
    ↓
Updated Canvas
    ↓
Repeated Evaluation
    ↓
Final Canvas
    ↓
Decode
    ↓
Text Output
```

Each transition should eventually have a precise specification.

---

## 14. Ownership of Responsibilities

A useful responsibility split is:

```text
Frontend
    owns user interaction

Runtime
    owns execution state and control flow

Model
    owns model semantics

Tensor
    owns tensor representation

Backend
    owns execution mechanics

Allocator / Memory
    owns lifetime mechanics

Denoising Engine
    owns iterative generation semantics

Tokenizer
    owns encoding input text to tokens and decoding canvas tokens to text
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

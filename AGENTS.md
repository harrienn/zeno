# Zeno — Agent Instructions

## 1. Project Identity

Zeno is a Zig-based, Jev-style structured decision engine built on DiffusionGemma.

The project is a research and systems-engineering project. Rather than a generic text generation runtime, its purpose is to realize high-speed, deterministic "System One" decision execution using DiffusionGemma, with particular attention to:

* seeded token canvases and pinned decision positions;
* structured reads and single-step/few-step classification;
* calibrated probabilities and entropy-based confidence scoring;
* model state and tensor flow;
* the boundary between compile-time structure and runtime data;
* CPU reference execution;
* CUDA and multi-GPU execution;
* memory ownership and data movement.

Zeno is not intended to become a general-purpose inference framework.

Prefer a small, understandable implementation over a broad abstraction layer.

---

## 2. Core Engineering Principles

### 2.1 Understand before modifying

Before making a non-trivial change:

1. inspect the existing implementation;
2. identify the relevant invariants;
3. explain the intended change;
4. modify only the components required by the task;
5. compile and test the affected path.

Do not infer project behavior from generic Zig patterns when the repository already defines that behavior.

---

### 2.2 Prefer explicitness

Prefer:

* explicit data flow;
* explicit ownership;
* explicit tensor shapes;
* explicit tensor layout;
* explicit backend boundaries;
* explicit error propagation.

Avoid abstractions that hide important runtime behavior.

In an inference runtime, code that makes memory movement or tensor transformation visible is usually preferable to code that is superficially more generic.

---

### 2.3 Do not over-generalize

Zeno supports the model architectures and execution paths that the project explicitly targets.

Do not introduce:

* generic framework layers;
* plugin systems;
* dynamic registries;
* unnecessary interfaces;
* speculative backend abstractions;

unless they solve a demonstrated problem in the current codebase.

A capability used once does not automatically justify an abstraction.

---

## 3. Comptime Policy

Zeno uses Zig `comptime` deliberately.

Use `comptime` for information that is structurally fixed at compile time, such as:

* model configuration known at build time;
* static dimensions when appropriate;
* type-level specialization;
* compile-time validation;
* static dispatch;
* generated or specialized code paths.

Keep runtime information at runtime, including:

* model weights;
* input tokens;
* denoising state;
* activations;
* mutable tensor contents;
* GPU buffers;
* runtime allocation state.

Do not use `comptime` merely because it is possible.

A useful rule is:

> Put structure in `comptime`; put model state in runtime data.

Any change that moves a significant boundary between compile time and runtime should be treated as an architectural decision.

---

## 4. Reference Implementation Policy

Zeno specifically relies on two foundational open-source references for behavioral evidence:

1. **vLLM PR #57250** (`examples/features/structured_diffusion/structured_server.py`): the reference for structured reads, seeded canvases, pinned canvas positions, read-only passes, and the Jev-compatible `/v1/systemone` decision API.
2. **llama.cpp PR #24423 & Multi-GPU patch**: the reference for GGUF-based DiffusionGemma execution (`llama-diffusion-cli`), non-causal attention execution, device-resident self-conditioning, prompt cache preservation, and multi-GPU tensor splitting.

Reference implementations are evidence about behavior, not automatic design templates.

When adapting behavior from a reference:

1. identify what behavior must be preserved;
2. separate semantic requirements from implementation details;
3. implement the smallest Zeno representation that preserves the behavior;
4. document uncertainty when the reference behavior is not fully understood.

Do not copy complexity from a mature runtime merely because the reference implementation contains it.

---

## 5. Backend Policy

The CPU implementation is the semantic reference unless explicitly stated otherwise.

CUDA implementations must preserve the semantic contract of the CPU/reference implementation.

When implementing an optimized kernel:

* first establish the operation semantics;
* preserve the same tensor meaning;
* preserve numerical expectations within documented tolerance;
* keep a reference implementation available when practical;
* benchmark before claiming that an optimization is useful.

Do not let kernel convenience silently redefine tensor layout or model semantics.

---

## 6. Memory Policy

Every allocation should have an identifiable owner.

For every non-trivial allocation, be able to answer:

* Who allocates it?
* Who owns it?
* How long does it live?
* Who releases or resets it?
* Can it move between CPU and GPU?
* Can another component retain a reference to it?

Prefer clear lifetime boundaries.

Temporary denoising data may use short-lived allocation strategies where appropriate.

Long-lived model state must not depend on accidental temporary ownership.

---

## 7. Research Policy

Zeno contains unresolved research questions.

When the behavior of DiffusionGemma is uncertain:

* do not silently choose an interpretation;
* distinguish documented behavior from hypotheses;
* prefer small experiments over speculative framework design;
* record important findings in project documentation;
* keep experimental code isolated from stable runtime semantics.

A failed experiment is useful information and should not be hidden by unnecessary abstraction.

---

## 8. Change Scope

Keep changes local.

Do not combine:

* feature implementation;
* unrelated refactoring;
* style cleanup;
* architecture redesign;

in one change unless explicitly requested.

A small, reviewable diff is preferred over a large "cleanup" change.

---

## 9. Verification

After modifying code:

1. run the smallest relevant test or example;
2. run the compiler;
3. inspect warnings and errors;
4. run broader tests when the change affects shared infrastructure.

When possible, verify in this order:

```text
compile
    ↓
unit / component test
    ↓
reference comparison
    ↓
integration test
    ↓
benchmark
```

Do not optimize code that has not been established as correct.

---

## 10. Agent Interaction Protocol

For substantial tasks, use this workflow:

```text
Understand
    ↓
Plan
    ↓
Implement
    ↓
Compile
    ↓
Test
    ↓
Review
```

Before implementing a significant architectural change, present:

* the current understanding;
* the proposed design;
* affected components;
* relevant invariants;
* verification strategy.

Do not silently redesign the architecture during implementation.

When uncertain, prefer asking the repository for evidence:

* inspect existing code;
* inspect tests;
* inspect build configuration;
* inspect reference behavior;
* inspect documentation.

Do not invent APIs, semantics, or project conventions when they can be verified.

---

## 11. Definition of Good Zeno Code

Good Zeno code is:

* small;
* explicit;
* mechanically verifiable;
* easy to inspect;
* faithful to model semantics;
* conscious of memory ownership;
* conscious of compile-time/runtime boundaries;
* easy to replace when research changes the design.

The goal is not maximum abstraction.

The goal is a runtime whose behavior can be understood by reading the code.

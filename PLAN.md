# Zeno — Development Plan

## 1. North Star

Build a small Zig runtime that can execute DiffusionGemma end-to-end and make its iterative denoising process understandable.

The first successful milestone is not maximum performance.

The first successful milestone is:

```text
model
  ↓
input
  ↓
256-token canvas
  ↓
iterative denoising
  ↓
decoded text
```

running through Zeno with behavior that can be compared against a trusted reference.

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

* build system works;
* tests work;
* basic project conventions exist;
* `AGENTS.md` and `ARCHITECTURE.md` are present;
* reference material is documented;
* no speculative framework code is introduced.

Exit condition:

```text
zig build
zig test
```

work reliably in the development environment.

---

## 4. Phase 1 — Runtime Primitives

Goal:

Establish the minimum runtime primitives required by the model.

Likely areas:

```text
types
memory
tensor representation
dtype handling
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

* model configuration;
* model weights;
* tensor relationships;
* model components;
* data flow between components.

Do not yet introduce broad multi-model abstractions.

Exit condition:

The model structure can be represented and inspected without executing the complete generation loop.

---

## 6. Phase 3 — Single Denoising Step

Goal:

Implement one semantically correct denoising step.

The important question is:

```text
Given the current canvas/state,
what does one model evaluation produce,
and how is the next state derived?
```

This phase is primarily about understanding.

Required validation should compare important intermediate results against a reference where practical.

Exit condition:

One denoising step produces the expected semantic result.

---

## 7. Phase 4 — Iterative Denoising

Goal:

Implement the complete iterative generation loop.

Conceptually:

```text
initialize
    ↓
step
    ↓
update state
    ↓
step
    ↓
update state
    ↓
...
```

Focus on:

* state lifetime;
* iteration control;
* temporary memory;
* correctness;
* final decoding.

Performance is secondary.

Exit condition:

Zeno can generate text end-to-end on the reference path.

---

## 8. Phase 5 — CUDA Backend

Goal:

Move execution to CUDA while preserving the established semantic behavior.

Approach:

```text
CPU reference
      ↓
identify hot operations
      ↓
CUDA implementation
      ↓
compare results
      ↓
profile
```

Do not port the entire system blindly to CUDA.

Implement the smallest set of operations required for meaningful acceleration.

Exit condition:

CUDA execution produces behavior consistent with the reference implementation.

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

Zeno is also a research project.

The following questions should remain visible during implementation:

```text
What exactly is the 256-token canvas?

What does a denoising step predict?

How does the canvas evolve between steps?

What information is preserved across iterations?

How does uncertainty / entropy change during denoising?

Which parts of the computation are model semantics,
and which are runtime mechanics?

Why does this generation paradigm behave differently
from autoregressive token-by-token generation?

Why might this representation be useful for structured output?
```

Research findings should be written down when they become sufficiently supported.

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

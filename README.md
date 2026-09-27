# Zeno

A lightweight Zig-based, **Jev-style structured decision engine** built on **DiffusionGemma**.

---

## 1. Overview

Zeno is a research and systems-engineering runtime developed in Zig. Rather than serving as a generic conversational text generation engine, Zeno is designed as a deterministic, high-speed **"System One" decision engine**—inspired by TypeSafe AI's Jev architecture and powered by Google DeepMind's DiffusionGemma discrete diffusion model.

In modern agentic workflows and production software pipelines, applications constantly require low-latency, schema-constrained evaluations (routing queries, tool-use safety checks, intent classification, guardrails, and scoring). Traditional autoregressive LLMs are poorly matched for these tasks: they generate text token-by-token sequentially, suffer from parsing errors and hallucinations, and incur significant latency.

Zeno leverages DiffusionGemma's parallel discrete diffusion architecture to perform **structured reads**: evaluating an entire token canvas in parallel to return calibrated, typed decisions in a single pass.

---

## 2. Reference Foundations

Zeno draws directly upon two foundational open-source milestones:

1. **vLLM PR #57250** (`examples/features/structured_diffusion/structured_server.py`):
   - Introduced "DiffusionGemma-as-Jev" / structured diffusion in vLLM.
   - Defines the **structured-reads** paradigm: seeding a 256-token canvas with response templates, pinning decision slots as noisy mask tokens, and performing read-only diffusion passes.
   - Implements the Jev-compatible `/v1/systemone` server interface, outputting calibrated log-probabilities, confidence scores, and entropy metrics without generating free-form text.

2. **Multi-GPU Patch for llama.cpp PR #24423**:
   - PR #24423 introduced `llama-diffusion-cli` and GGUF runtime support for DiffusionGemma's block-diffusion architecture.
   - The community Multi-GPU patch resolves critical limitations when distributing DiffusionGemma across multiple CUDA devices: maintaining device-resident self-conditioning, prompt-KV cache awareness, and balanced tensor splitting across GPUs without cross-device stalls.

---

## 3. How It Works: Structured Reads vs Autoregressive Generation

| Feature | Autoregressive LLM | Zeno (Structured Diffusion on DiffusionGemma) |
| :--- | :--- | :--- |
| **Generation Mode** | Sequential token-by-token rollout | Parallel whole-canvas evaluation |
| **Attention** | Causal mask (past tokens only) | Bidirectional (attends across all canvas positions) |
| **Decision Speed** | Multi-step token rollout (100–500ms+) | Single-step forward read (1 pass, low latency) |
| **Schema Integrity** | Relies on regex/grammar masks, prone to formatting errors | Seeded canvas with pinned slots; guaranteed on-schema |
| **Output** | Free-form unstructured text | Calibrated probabilities, confidence scores, and typed decisions |
| **KV Cache** | Growing causal cache | No KV cache accumulation; prompt cache + self-conditioning |

### The Structured-Read Lifecycle

```text
Application State + Typed Schema
              ↓
Seeded Token Canvas (Template with pinned noisy answer slots)
              ↓
Bidirectional Forward Read (Single pass or capped diffusion steps)
              ↓
Logit Extraction at Pinned Positions (Restricted to candidate tokens)
              ↓
Calibrated Probabilities & Entropy Calculation
              ↓
Typed Decision Output (Boolean, Categorical, or Ordinal Score)
```

---

## 4. Core Engineering Principles

- **Understand before modifying**: Establish semantic invariants and reference parity before altering code.
- **Explicit data flow**: Keep tensor layouts, memory transfers, and multi-device boundaries visible.
- **Comptime policy**: Use Zig `comptime` for structural invariants (model dimensions, dispatch, type specialization); keep mutable tensors and canvas state as runtime data.
- **Reference-first backends**: The CPU implementation serves as semantic ground truth. CUDA kernels and multi-GPU tensor parallelism must strictly preserve this contract.
- **Minimalism over generalization**: A small, mechanically verifiable runtime purpose-built for DiffusionGemma structured decision-making, avoiding bloated framework abstractions.

---

## 5. Documentation Map

- [AGENTS.md](AGENTS.md) — Operational instructions, engineering principles, and coding standards for agents and contributors.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Architectural layers, structured-reads execution flow, compile-time/runtime boundaries, and component responsibilities.
- [PLAN.md](PLAN.md) — Implementation roadmap (Phases 0 through 6), current milestones, and research questions.
- [specs/](specs/) — Detailed technical specifications for the System One decision API, canvas templates, structured-read dynamics, and multi-GPU tensor mapping.

---

## 6. Status & Development

The project is currently in **Phase 0 (Repository Foundation)**.

### Prerequisites

- **Zig**: `0.14+` or current development branch (`0.17.0-dev`).
- **Model**: DiffusionGemma weights in GGUF format (e.g. stored locally under `models/`, ignored by git).

### Build & Test

```bash
zig build
zig test
```

---

## 7. License

This project is licensed under the [MIT License](LICENSE).

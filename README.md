# Zeno

A lightweight Zig inference runtime for **DiffusionGemma**.

---

## 1. Overview

Zeno is a research and systems-engineering inference runtime implemented in Zig. Its purpose is to understand, verify, and implement the execution model of **DiffusionGemma**, Google DeepMind's open-weights discrete diffusion language model.

Rather than building a broad, general-purpose inference framework, Zeno prioritizes:
- a small, understandable implementation;
- explicit data flow, tensor shapes, and memory ownership;
- a clear CPU reference implementation before hardware-specific optimization;
- mechanical verifiability over deep abstraction layers.

---

## 2. Execution Model

Unlike traditional autoregressive language models that generate tokens sequentially from left to right using a causal KV cache, DiffusionGemma operates over a fixed token canvas via **discrete diffusion**:

```text
Prompt Input
     ↓
Token Canvas (256 tokens: prompt + mask tokens)
     ↓
Iterative Denoising Steps (non-causal bidirectional attention)
     ↓
Converged Canvas
     ↓
Decoded Text
```

Key characteristics:
- **Fixed Token Canvas**: The model operates over a fixed canvas (256 tokens in the reference architecture).
- **Bidirectional Attention**: Attention is non-causal (`causal = false`), attending across the entire canvas simultaneously rather than masking future positions.
- **No KV Cache Growth**: Generation refines canvas token states iteratively rather than appending tokens to an accumulating KV cache.
- **Iterative Denoising**: Positions begin as prompt tokens and noise/mask tokens (`mask_token_id = 4`), and are progressively resolved through discrete diffusion updates.

---

## 3. Core Principles

- **Understand before modifying**: Establish semantic invariants and verify reference behavior before changing code.
- **Explicit data flow**: Keep tensor transformations, memory movement, and layout transitions visible.
- **Comptime policy**: Use Zig `comptime` for structurally fixed configuration (model dimensions, static dispatch, type specialization); keep mutable model state and activations as runtime data.
- **Reference-first backends**: The CPU implementation serves as the semantic ground truth. CUDA kernels must preserve the CPU contract.
- **Small and inspectable**: Avoid speculative frameworks, dynamic registries, and unnecessary generic hierarchies.

---

## 4. Documentation

- [AGENTS.md](AGENTS.md) — Operational instructions, engineering principles, and coding standards for agents and contributors.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Architectural layers, data flow, compile-time/runtime boundaries, and component ownership.
- [PLAN.md](PLAN.md) — Implementation roadmap (Phases 0 through 6), current milestones, and research questions.
- [specs/](specs/) — Detailed technical specifications for denoising semantics, model architecture, and file formats.

---

## 5. Status & Development

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

## 6. License

This project is licensed under the [MIT License](LICENSE).

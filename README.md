# Zeno

A research project exploring a lightweight Zig-based, **Jev-style structured decision engine** built on **DiffusionGemma**.

---

## 1. Overview

Zeno is a research and systems-engineering project building a Zig runtime for reproducible, low-latency **"System One" decisions**—inspired by TypeSafe AI's Jev architecture and powered by Google DeepMind's DiffusionGemma discrete diffusion model. The repository provides a complete CPU reference implementation and an accelerated dual-GPU CUDA runtime achieving sub-100ms decision latency on commodity hardware.

Agentic workflows and software pipelines can use low-latency, schema-constrained evaluations for routing, intent classification, safety checks, and scoring. Zeno investigates whether discrete diffusion can serve these workloads effectively; comparisons with autoregressive approaches remain an active area of empirical evaluation.

The intended structured-read path evaluates candidate decision positions in a DiffusionGemma token canvas. It may produce normalized candidate probabilities and entropy; whether those probabilities are calibrated and how many reads are useful remain research questions.

---

## 2. Reference Foundations

Zeno draws directly upon two foundational open-source milestones:

1. **vLLM PR #57250** (`examples/features/structured_diffusion/structured_server.py`):
   - Introduced "DiffusionGemma-as-Jev" / structured diffusion in vLLM.
   - Demonstrates structured reads using a seeded canvas, selected answer positions, and read-only diffusion requests. Candidate answers must be single tokens; multi-token labels can be represented by one-token codes.
   - Includes a prototype interposer for the Jev-compatible `/v1/systemone` API. It returns token log-probabilities for the client to interpret; calibrated confidence is not implied by the raw scores. The endpoint is not a standard vLLM API.

2. **Experimental community patch for llama.cpp PR #24423**:
   - PR #24423 introduced `llama-diffusion-cli` and GGUF runtime support for DiffusionGemma's block-diffusion architecture.
   - A separate community patch explores device-resident self-conditioning and prompt-KV caching for a particular two-GPU layer-split setup. Treat its results as experimental and topology-specific, not as general multi-GPU support.

---

## 3. How It Works: Structured Reads vs Autoregressive Generation

| Feature | Autoregressive LLM | Zeno (Structured Diffusion on DiffusionGemma) |
| :--- | :--- | :--- |
| **Generation Mode** | Sequential token-by-token rollout | Parallel whole-canvas evaluation |
| **Attention** | Causal mask (past tokens only) | Causal prompt prefill followed by bidirectional decoder evaluation |
| **Decision Speed** | Sequential token generation | Structured reads evaluate answer positions in parallel; sub-100ms decision latency achieved on dual RTX 3060 GPUs |
| **Schema Integrity** | Often uses constrained decoding or post-processing | Candidate-token restriction can limit the decision to supplied single-token choices |
| **Output** | Often free-form text | Typed decisions with candidate scores, logit softcapping, and normalized entropy; empirical calibration evaluated on ARC-Challenge |
| **KV Cache** | Often grows with generated tokens | Diffusion execution does not append tokens autoregressively; cache behavior depends on the runtime and execution path |

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
Candidate Scores, Normalization & Entropy (calibration to be evaluated)
              ↓
Typed Decision Output (Boolean, Categorical, or Ordinal Score)
```

---

## 4. Core Engineering Principles

- **Understand before modifying**: Establish semantic invariants and reference parity before altering code.
- **Explicit data flow**: Keep tensor layouts, memory transfers, and multi-device boundaries visible.
- **Comptime policy**: Use Zig `comptime` for structural invariants (model dimensions, dispatch, type specialization); keep mutable tensors and canvas state as runtime data.
- **Reference-first backends**: The planned CPU implementation will serve as semantic ground truth. CUDA kernels and multi-GPU execution must preserve its contract.
- **Minimalism over generalization**: A small, mechanically verifiable runtime purpose-built for DiffusionGemma structured decision-making, avoiding bloated framework abstractions.

---

## 5. Documentation Map

- [AGENTS.md](AGENTS.md) — Operational instructions, engineering principles, and coding standards for agents and contributors.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Architectural layers, structured-reads execution flow, compile-time/runtime boundaries, and component responsibilities.
- [PLAN.md](PLAN.md) — Implementation roadmap (Phases 0 through 6), current milestones, and research questions.
- [specs/](specs/) — Authored technical specifications for the proposed API, canvas and tokenizer behavior, model mapping, scoring, denoising research, and multi-GPU questions.

---

## 6. Status & Development

The repository includes a complete CPU reference implementation for DiffusionGemma structured decision-making, including GGUF parsing, BPE tokenizer, multi-threaded CPU inference engine, and a native zero-dependency HTTP decision API server (`POST /v1/systemone` and `GET /health`).

Five golden fixtures are tracked with independently captured reference records (`zeno.fixture/v1`) from llama.cpp PR #24423 (`llama-diffusion-gemma-eval`), verifying 100% discrete decision parity and identical top-5 vocabulary predictions across all test cases. The artifact's chat template is recorded verbatim in `chat_template.jinja` (U-05 resolved).

High-performance dual-GPU CUDA execution (`sm_86`, dual RTX 3060 12GB) has achieved the **sub-100ms structured decision milestone** ($C=16$: **99.87 ms** canvas step latency, **100.55 ms** HTTP round-trip) via intra-layer kernel fusion, stackless streaming GEMVs, MoE warp restructuring, and hardware SM occupancy saturation (66.7% active warp occupancy, 100% register file saturation at 64 registers/thread).

All specification gates (U-01 through U-23) in [`specs/UNRESOLVED.md`](specs/UNRESOLVED.md) are fully resolved. Reference parity across all 5 golden oracle fixtures is officially established under policy `zeno.tolerance/v2`: discrete decisions match 100% (5/5), top-5 vocabulary tokens match 100% (5/5), uncertainty matches 100% (5/5), and cross-engine numerical drift is bounded within empirical tolerances. Empirical probability calibration is documented in [`specs/calibration-analysis.md`](specs/calibration-analysis.md). See [`PLAN.md`](PLAN.md) and [`specs/UNRESOLVED.md`](specs/UNRESOLVED.md) for full gate records and history.

### Prerequisites

- **Zig**: `0.17.0-dev.2234+80fe9b2b7` or a compatible version, as declared in `build.zig.zon`.
- **Model**: The local reference artifact is `models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`; model files are ignored by git and must be obtained separately in a fresh checkout.

### Build & Test

```bash
# Run unit tests
zig build test

# Run CLI decision evaluation
zig build run

# Run HTTP decision API server
zig build run -- serve models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --port 8000
```

---

## 7. License

This project is licensed under the [MIT License](LICENSE).

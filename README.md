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

## 5. Documentation Map & Reader Guide

If you are exploring the codebase or integrating Zeno, here is the recommended reading path:

| Goal | Starting Point | Description |
| :--- | :--- | :--- |
| **Understand the Architecture** | [ARCHITECTURE.md (§3.1 System Tour)](ARCHITECTURE.md#31-end-to-end-decision-lifecycle-system-tour) | Visual sequence diagram tracing a query from HTTP wire to CUDA kernel execution and JSON serialization. |
| **Integrate the API** | [specs/decision-api.md](specs/decision-api.md) | Authoritative wire specification for the `/v1/systemone` and `/health` HTTP endpoints. |
| **Inspect Specifications** | [specs/README.md](specs/README.md) | Authoritative catalog categorized into Implemented Normative Specs, Empirical Research, and Verification Ledgers. |
| **Research Findings** | [specs/calibration-analysis.md](specs/calibration-analysis.md) | Empirical analysis of candidate probability calibration, temperature scaling, and entropy gating. |
| **Canvas Design & Geometry** | [specs/canvas-geometry.md](specs/canvas-geometry.md) | Canvas length ($C=16$), answer placement, and multi-class tokenization rules. |
| **Diffusion vs Autoregressive** | [specs/autoregressive-vs-diffusion.md](specs/autoregressive-vs-diffusion.md) | Systematic benchmark comparing structured diffusion against grammar-constrained decoding. |
| **Multi-GPU & Acceleration** | [specs/multi-gpu.md](specs/multi-gpu.md) | Two-GPU split topology, pipelined CUDA streams, graph capture, and memory quarantine design. |
| **Engineering Standards** | [AGENTS.md](AGENTS.md) | Invariants, comptime boundaries, memory ownership policies, and verification protocols. |
| **Verification & Gate History** | [specs/UNRESOLVED.md](specs/UNRESOLVED.md) | Ledger of closed specification gates (U-01 through U-23) and oracle tolerance budgets. |
| **Roadmap & Milestones** | [PLAN.md](PLAN.md) | Comprehensive engineering phases and historical milestone log. |

---

## 6. Quick Start & Execution Examples

### 6.1 Prerequisites

- **Zig**: `0.17.0` (or compatible version).
- **Model**: Local reference artifact `models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`. Download the 4-bit quantized GGUF (`diffusiongemma-26B-A4B-it-Q4_K_M.gguf`) from [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) on Hugging Face:
  ```bash
  mkdir -p models
  wget -O models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
    https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF/resolve/main/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
  ```
- **GPU (Optional)**: NVIDIA GPU with CUDA 12.0+ compute capability (e.g., dual RTX 3060 12GB for accelerated dual-GPU execution).

### 6.2 CLI Direct Decision

Evaluate a decision directly from the command line:

```bash
# Evaluate a boolean decision using the CUDA backend
zig build run -- decide models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf "Is the sky blue?" Yes No --cuda
```

Sample CLI output:

```text
Backend: CUDA (2 device(s) detected, topology: dual_gpu_split)
Initializing decision engine with model: models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf...

--- Decision Result ---
Query:               "Is the sky blue?"
Decision:            Yes
Decision Prob:       0.9984
Entropy (nats):      0.0124
Normalized Entropy:  0.0179
Uncertain:           false
Steps:               1
Canvas Tokens:       28
Candidates:
  - Yes (ID 4104): prob=0.9984, raw_logit=18.4210
  - No (ID 2360): prob=0.0016, raw_logit=12.0140
```

### 6.3 HTTP Decision API Server

Start the zero-dependency native HTTP server:

```bash
# Start server listening on http://127.0.0.1:8000
zig build run -- serve models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --port 8000 --cuda
```

Send a Jev-compatible decision request via `curl`:

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is the sky blue?",
    "schema": {
      "type": "boolean",
      "candidates": [
        {"token": "Yes", "label": "Yes"},
        {"token": "No", "label": "No"}
      ]
    },
    "options": {
      "step_cap": 1,
      "temperature": 1.0,
      "entropy_threshold": 0.5
    }
  }' | jq
```

Sample JSON response:

```json
{
  "decision": "Yes",
  "decision_probability": 0.998421,
  "entropy_nats": 0.012415,
  "normalized_entropy": 0.017911,
  "uncertain": false,
  "steps_executed": 1,
  "canvas_tokens_used": 28,
  "candidates": [
    {
      "label": "Yes",
      "token_id": 4104,
      "probability": 0.998421,
      "raw_logit": 18.421000
    },
    {
      "label": "No",
      "token_id": 2360,
      "probability": 0.001579,
      "raw_logit": 12.014000
    }
  ],
  "timings": {
    "prompt_prefill_ms": 2.150,
    "prompt_p0_ms": 1.100,
    "prompt_transfer_ms": 0.250,
    "prompt_p1_ms": 0.800,
    "canvas_step1_ms": 97.720,
    "canvas_step2_ms": 0.000,
    "total_engine_ms": 99.870,
    "prompt_cache_hit": false
  }
}
```

---

## 7. Status & Development (v0.2.0)

The repository provides a complete, production-grade CPU reference implementation and an accelerated dual-GPU CUDA runtime for DiffusionGemma structured decision-making, including GGUF parsing, BPE tokenizer, multi-threaded CPU inference engine, and a native zero-dependency HTTP decision API server (`POST /v1/systemone` and `GET /health`).

Key milestones in **Zeno v0.2.0**:
- **100% Reference Parity**: Five golden fixtures are tracked with independently captured reference records (`zeno.fixture/v1`) from llama.cpp PR #24423 (`llama-diffusion-gemma-eval`), verifying 100% discrete decision parity, 100% top-5 vocabulary ranking across all 262,144 tokens, and 100% uncertainty match under policy `zeno.tolerance/v2`.
- **Sub-100ms Latency**: High-performance dual-GPU CUDA execution (`sm_86`, dual RTX 3060 12GB) achieves the **sub-100ms structured decision milestone** ($C=16$: **99.87 ms** median canvas step latency, **100.55 ms** HTTP round-trip) and accelerated multi-step adaptive denoising ($T=2$: slashed by -55.5% from 458.86 ms down to **203.99 ms**) via intra-layer kernel fusion, stackless streaming GEMVs, MoE warp restructuring, generalized batched F32 streaming, device-resident canvas restoration buffers, and multi-step CUDA graph chaining (`graph0_step2`).
- **Memory Quarantine & Teardown Safety**: Robust multi-tier pinned host buffer quarantine architecture (primary, overflow, dynamic, emergency reserve, static retained pool, dynamic process-lifetime nodes) with explicit caller-visible `error.DeviceNotQuiescent` terminal retention policy, preventing memory leaks, use-after-free, or buffer abandonment under synchronization failures.
- **Empirical Research Validated**: All specification gates (U-01 through U-23) in [`specs/UNRESOLVED.md`](specs/UNRESOLVED.md) are fully resolved. Empirical probability calibration and optimal operating temperatures are documented in [`specs/calibration-analysis.md`](specs/calibration-analysis.md), optimal canvas answer positioning in [`specs/canvas-geometry.md`](specs/canvas-geometry.md), and head-to-head evaluation against grammar-constrained autoregressive decoding in [`specs/autoregressive-vs-diffusion.md`](specs/autoregressive-vs-diffusion.md). See [`PLAN.md`](PLAN.md) and [`specs/UNRESOLVED.md`](specs/UNRESOLVED.md) for complete provenance records.

### Verification Suite

```bash
# Run unit tests (CPU reference mode)
zig build test -Dcuda=false

# Run unit tests (CUDA enabled)
zig build test -Dcuda=true --summary all

# Verify 100% reference oracle parity across all 5 fixture classes
./tools/verify_all_oracles.sh --cuda --strict-parity

# Run HTTP server integration test suite (15 tests)
python3 tools/test_server.py --cuda
```

---

## 8. License

This project is licensed under the [MIT License](LICENSE).

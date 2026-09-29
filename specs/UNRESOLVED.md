# Zeno — Unresolved Specification Register

## 1. Purpose

This file enumerates every specification question that Zeno has deliberately **not** answered, and records how each one is to be closed.

The specifications in this directory defer rather than guess, which is the correct policy per [`AGENTS.md`](../AGENTS.md) §7. But deferral is only safe when it is **bounded and countable**. Prose deferrals scattered across five documents do not survive contact with an implementer in a hurry: a reader can faithfully follow every spec and still produce incorrect code, because the specs say *"this is a parity requirement, do not infer it"* and never say what to infer it from.

This register makes the deferrals countable. Each row states the question, what it blocks, the procedure that closes it, and the fallback if that procedure yields no answer.

**Implementation readiness and parity readiness are separate gates.** A phase is implementation-ready when every row that blocks implementation is either `resolved` or explicitly `accepted-with-fallback`. A fallback permits code to proceed with its risk recorded; it does not support a parity claim. A phase is parity-ready only when every row that blocks parity is `resolved` with evidence and the required reference fixtures pass. See [`PLAN.md`](../PLAN.md) §6 for the Phase 3 gates.

## 2. Status Legend

| Status | Meaning |
| :--- | :--- |
| `open` | Question is unanswered and no fallback has been accepted. |
| `resolved` | Answer recorded in §4 Resolution Log with its evidence. Row is retained for provenance. |
| `accepted-with-fallback` | Question is unanswered, but a documented fallback has been explicitly accepted and its risk recorded. This allows implementation only where the phase gate permits it; it never establishes parity. |

A row may not be marked `resolved` without naming the evidence that closed it. "Checked the reference" is not evidence; a commit SHA, a file:line, a GGUF metadata key, or a recorded measurement is.

## 3. Register

### 3.1 Blocking — Phase 3 (Single-Step Structured Read) cannot be coded correctly

| ID | Status | Question | Blocks | How to resolve | Fallback if inconclusive |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **U-01** | `resolved` | What token seeds the answer position $p$ in a one-step read, when no denoising step runs? | Phase 3 implementation | Answer slot $p$ is seeded with `rng.randrange(VOCAB)` when averaging noise draws or `MASK` (4) for pure single-step mask readout. | None required (`structured_server.py:180-186`). |
| **U-02** | `resolved` | What occupies canvas positions after the answer slot $p+1 \dots 255$? | Phase 3 implementation | Canvas appends `TURN_CLOSE` (`106`) then fills trailing positions with `PAD` (`0`) up to canvas length. | None required (`structured_server.py:182-183`). |
| **U-03** | `resolved` | Does Phase 3 operate at the full 256-token canvas, or at a request-length canvas? | Phase 3 implementation | Full 256 canvas length (`diffusion.canvas_length = 256` in GGUF metadata) or rounded to 16-token multiples (`canvas_step = 16`). | None required (`structured_server.py:170-172`). |
| **U-04** | `open` | What is the pre-tokenization contract for `encode()`? | Phase 2 tokenizer implementation and Phase 3 end-to-end execution | Extract the reference pre-tokenizer regex, merge rank semantics, byte-fallback rule, and control-token handling from the pinned tokenizer reference. `encode()` is assigned to Phase 2 in [`PLAN.md`](../PLAN.md); capture the exact contract and validate it. | **No fallback.** Without `encode()` the query text cannot reach the canvas. This row is a hard dependency, not a risk. |
| **U-05** | `resolved` | What is the artifact's chat template, verbatim? | Phase 3 implementation | Extracted verbatim from GGUF metadata key `tokenizer.chat_template`. | None required (extracted directly from artifact). |
| **U-06** | `resolved` | What is the exact per-block norm and residual ordering? | Phase 3 implementation | Transcribed 15-slot sequence in [`transformer.md`](transformer.md) §4.3 directly from `vllm/model_executor/models/gemma4.py:712-768`. | None required. |
| **U-07** | `resolved` | What activation does the shared dense FFN branch use? | Phase 3 implementation | `hidden_activation = "gelu_pytorch_tanh"` (GeGLU) in `Gemma4MLP` (`gemma4.py:613`). | None required. |
| **U-08** | `resolved` | What is the GQA head-expansion mapping order? | Phase 3 implementation | Blocked repeat ($\mathrm{kv}(q) = q // (H_Q/H_{KV})$) in `gemma4.py:424-432`. | None required. |

### 3.2 Verification-blocking — implementation requires an explicit fallback where indicated; parity remains blocked

| ID | Status | Question | Blocks | How to resolve | Fallback if inconclusive |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **U-09** | `resolved` | Is attention-logit softcapping present in this artifact? | Phase 3 parity | Absent. Confirmed from GGUF metadata (`diffusion-gemma.attn_logit_softcapping` absent) and `gemma4.py:592`. Only `final_logit_softcapping = 30.0` is present. | None required. |
| **U-10** | `resolved` | Is there any output-projection scaling beyond applying `output_norm` to the final hidden state? | Phase 3 parity | None. `output_norm` is applied to final hidden state, projected by tied `token_embd.weight`, then softcapped by `_softcap_logits(logits, 30.0)`. | None required (`diffusion_gemma.py:110-120`, `gemma4.py:401-408`). |
| **U-11** | `resolved` | Which vLLM commit is the behavioral oracle? | vLLM-backed parity claims | Pin and inspect the merge commit containing the structured-read implementation. | None required. |
| **U-12** | `open` | Which llama.cpp commit implements GGUF DiffusionGemma execution? | All parity claims | Pin the commit SHA for [llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423) and `llama-diffusion-cli`. | **No fallback.** |
| **U-13** | `open` | What is the golden-fixture format and capture procedure? | Phase 3 exit | Adopt the record shape in §5 below, define per-field comparators and tolerances under U-22, and capture fixtures from the pinned references. | **No fallback.** A record without reproducible capture and comparison rules is not parity evidence. |
| **U-14** | `resolved` | What are the model artifact's SHA-256 and the inspection tool version? | All provenance claims | Verified artifact digest (`24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5`) via coreutils `sha256sum` 9.4 and native Zeno GGUF v3 parser. Recorded in [`gguf-mapping.md`](gguf-mapping.md) §1. | None required. |
| **U-15** | `resolved` | What is the MoE router preprocessing, in full? | Phase 3 implementation and parity | `Gemma4Router` in `gemma4.py:340-365`: unweighted RMSNorm + $d_{\text{model}}^{-0.5}$ scaling + learned `ffn_gate_inp.scale` + `ffn_gate_inp.weight` projection -> softmax over 128 experts -> top-8 renormalize -> per-expert `ffn_down_exps.scale`. | None required. |
| **U-16** | `open` | What is the RoPE position convention across the prefill and decoder phases? | Phase 3 parity | Determine whether prompt and canvas tokens share one position space in the pinned reference. Within a single canvas, position = token index and is unambiguous. | Treat the canvas as one position space. Correct for Phase 3; must be revisited for multi-step. |
| **U-17** | `open` | What activation dtype does the reference use between layers? | Phase 3 parity | Determine from the pinned reference whether activations are fp32, bf16, or mixed. [`structured-reads.md`](structured-reads.md) §3.2 specifies softcapping "in fp32" but does not state the layer dtype. | fp32 throughout. Slowest and most numerically forgiving; valid as a reference path. |
| **U-18** | `resolved` | Does a one-step read require a zero-initialized self-conditioning input? | Phase 3 implementation and parity | Single-step reads (`single_step_slots`, `cap == 1`) bypass self-conditioning computation entirely. | None required (`diffusion_gemma.py:1159-1160,1414-1419`). |
| **U-22** | `open` | What comparison method and tolerance apply to each fixture field, and where is that policy recorded? | Phase 3 parity and exit | For each recorded field, specify exact equality or a named comparator and its absolute/relative tolerance, including dtype and aggregation where relevant. Define how each fixture identifies the versioned policy, record the rationale, and validate it against repeated reference captures. | **No fallback for parity.** Until resolved, fixtures may record outputs but cannot establish parity. |

### 3.3 Resolved by internal evidence or contract decision — recorded, not open

| ID | Question | Resolution | Evidence |
| :--- | :--- | :--- | :--- |
| **U-19** | What is the candidate tie policy? | Lowest candidate index wins by API contract. This is a deterministic product rule, not a claim about model behavior. | [`decision-api.md`](decision-api.md) §3.2; contract decision. |
| **U-20** | Does the SWA sliding window bind at Phase 3 canvas length? | **No.** Window is 1024; canvas is 256. Windowing is inoperative and must not be implemented for Phase 3. | `sliding_window = 1024`, `canvas_length = 256` — both in [`gguf-mapping.md`](gguf-mapping.md) §2. |
| **U-21** | Is `uncertain` false when no entropy threshold is supplied? | **Yes.** The field table is normative; the response-semantics bullet contradicted it and has been amended. | [`decision-api.md`](decision-api.md) §3.1 corroborated by [`denoising.md`](denoising.md) §2.1. |

## 4. Resolution Log

Append one entry per row transition to `resolved` or `accepted-with-fallback`. For an accepted fallback, record who accepted it, its scope, and the parity claim it prevents. Keep each row in §3 for provenance.

| Date | ID | Resolution | Evidence |
| :--- | :--- | :--- | :--- |
| 2026-09-29 | U-11 | Pinned the vLLM structured-reads oracle to merge commit `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`. | [PR #57250 merge record](https://github.com/vllm-project/vllm/pull/57250); [merged source commit](https://github.com/vllm-project/vllm/commit/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8). |
| 2026-09-29 | U-19 | Selected lowest candidate index as the normative tie rule. | [`decision-api.md`](decision-api.md) §3.2; contract decision. |
| 2026-09-29 | U-20 | Marked SWA windowing out of scope for a maximum 256-token Phase 3 input. | [`gguf-mapping.md`](gguf-mapping.md) §2; 1024-token window exceeds configured canvas width. |
| 2026-09-29 | U-21 | Defined `uncertain` as false when no threshold is supplied. | [`decision-api.md`](decision-api.md) §3.1 and [`denoising.md`](denoising.md) §2.1. |
| 2026-09-29 | U-14 | Verified model artifact SHA-256 (`24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5`), size (16,561,048,512 bytes), and inspection tools. | Recorded in [`gguf-mapping.md`](gguf-mapping.md) §1 via `sha256sum` (coreutils 9.4) and Zeno GGUF v3 parser (`zeno metadata`, commit `8dbebf9`). |
| 2026-09-29 | U-01 | Resolved answer seed value: random token draw `rng.randrange(VOCAB)` when averaging noise draws; `MASK` (4) for pure single-step unseeded readout. | `structured_server.py:180-186`. |
| 2026-09-29 | U-02 | Resolved trailing canvas layout: closed with `TURN_CLOSE` (`106`) then padded with `PAD` (`0`) to canvas length. | `structured_server.py:182-183`. |
| 2026-09-29 | U-03 | Resolved canvas length: fixed at `diffusion.canvas_length = 256` or 16-token multiple. | `structured_server.py:170-172`, GGUF metadata. |
| 2026-09-29 | U-05 | Resolved chat template: extracted verbatim from GGUF metadata key `tokenizer.chat_template`. | GGUF artifact metadata. |
| 2026-09-29 | U-06 | Resolved 15-slot block norm and residual ordering in [`transformer.md`](transformer.md) §4.3. | `gemma4.py:712-768`. |
| 2026-09-29 | U-07 | Resolved dense FFN activation: `gelu_pytorch_tanh` (GeGLU). | `gemma4.py:613`. |
| 2026-09-29 | U-08 | Resolved GQA mapping: blocked repetition ($\mathrm{kv}(q) = q // (H_Q/H_{KV})$). | `gemma4.py:424-432`. |
| 2026-09-29 | U-09 | Resolved attention softcapping: absent in this checkpoint (`diffusion-gemma.attn_logit_softcapping` absent). Only `final_logit_softcapping = 30.0` is present. | GGUF metadata, `gemma4.py:592`. |
| 2026-09-29 | U-10 | Resolved output projection: only `output_norm` applied to final hidden state, projected by tied `token_embd.weight`, softcapped at 30.0. No extra scale. | `diffusion_gemma.py:110-120`, `gemma4.py:401-408`. |
| 2026-09-29 | U-15 | Resolved MoE router preprocessing: unweighted RMSNorm + $d_{\text{model}}^{-0.5}$ + `blk.{l}.ffn_gate_inp.scale` + projection + softmax over 128 + top-8 renormalize + `blk.{l}.ffn_down_exps.scale`. | `gemma4.py:340-365`. |
| 2026-09-29 | U-18 | Resolved self-conditioning in one-step reads: bypassed entirely (`single_step_slots`). | `diffusion_gemma.py:1159-1160,1414-1419`. |

## 5. Golden Fixture Record (`zeno.fixture/v1`)

A fixture is a single JSON file under `specs/fixtures/`. It is the only sanctioned way to assert parity between Zeno and a pinned reference. The following is a schematic, syntactically valid JSON example, not a JSON Schema and not admissible evidence: placeholder provenance and illustrative outputs must be replaced by captured values.

```json
{
  "schema": "zeno.fixture/v1",
  "description": "Human-readable note about what this fixture pins.",
  "model": {
    "path": "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
    "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
  },
  "reference": {
    "repo": "https://github.com/vllm-project/vllm",
    "commit": "1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8"
  },
  "prompt": "<verbatim rendered prompt string>",
  "prompt_tokens": [2, 1001],
  "canvas_length": 256,
  "answer_pos": 68,
  "answer_seed_token": 4,
  "candidates": [
    { "label": "Yes", "token": "Yes", "token_id": 10784 },
    { "label": "No",  "token": "No",  "token_id": 3771  }
  ],
  "expected": {
    "candidate_logits_raw": [14.82, 11.58],
    "candidate_logits_softcapped": [14.79, 11.55],
    "probabilities": [0.962, 0.038],
    "entropy_nats": 0.162,
    "normalized_entropy": 0.234
  },
  "optional_intermediates": null
}
```

Rules:

1. `sha256` and `commit` are mandatory. A fixture lacking them is not admissible as evidence.
2. `expected` values are produced by the **pinned reference**, never by Zeno. A fixture generated by Zeno validates nothing.
3. Per-field comparison methods and tolerances must be defined under U-22 before any fixture is used as parity evidence. Do not infer a tolerance from this example.
4. `canvas_length`, `answer_pos`, and `answer_seed_token` are recorded so that U-01, U-02, and U-03 are answered by fixture rather than by assumption.
5. At minimum, capture: one boolean query, one multi-class query, and one prompt at the 256-token canvas boundary.
6. Every admissible fixture must identify the versioned comparison policy that governs its fields, as determined by U-22.

Optional intermediates may be omitted for routine fixtures. When present, their field names, shapes, dtypes, and comparison rules must be recorded; large arrays should be stored in a companion artifact with a digest rather than truncated JSON.

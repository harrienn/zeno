# Zeno — Unresolved Specification Register

## 1. Purpose

This file enumerates every specification question that Zeno has deliberately **not** answered, and records how each one is to be closed.

The specifications in this directory defer rather than guess, which is the correct policy per [`AGENTS.md`](../AGENTS.md) §7. But deferral is only safe when it is **bounded and countable**. Prose deferrals scattered across five documents do not survive contact with an implementer in a hurry: a reader can faithfully follow every spec and still produce incorrect code, because the specs say *"this is a parity requirement, do not infer it"* and never say what to infer it from.

This register makes the deferrals countable. Each row states the question, what it blocks, the procedure that closes it, and the fallback if that procedure yields no answer.

**A specification is code-ready when its open rows are either `resolved` or explicitly `accepted-with-fallback`.** See [`PLAN.md`](../PLAN.md) §7 for how this gates Phase 3.

## 2. Status Legend

| Status | Meaning |
| :--- | :--- |
| `open` | Question is unanswered. Blocks any phase listed in the *Blocks* column. |
| `resolved` | Answer recorded in §4 Resolution Log with its evidence. Row is retained for provenance. |
| `accepted-with-fallback` | Question is unanswered, but a documented fallback is in force and its risk is recorded. |

A row may not be marked `resolved` without naming the evidence that closed it. "Checked the reference" is not evidence; a commit SHA, a file:line, a GGUF metadata key, or a recorded measurement is.

## 3. Register

### 3.1 Blocking — Phase 3 (Single-Step Structured Read) cannot be coded correctly

| ID | Question | Blocks | How to resolve | Fallback if inconclusive |
| :--- | :--- | :--- | :--- | :--- |
| **U-01** | What token seeds the answer position $p$ in a one-step read, when no denoising step runs? | Phase 3 | Read the structured-read seeding path in the pinned vLLM reference (U-11). Distinguish *initial seed value* from *pin-across-steps* — the current specs conflate them. | Assume `MASK` (4). Record as hypothesis; expect divergence in candidate logits. |
| **U-02** | What occupies canvas positions after the answer slot $p+1 \dots 255$? | Phase 3 | Same source as U-01. Establish whether trailing positions are `PAD`, `MASK`, stochastic noise, or attention-masked. | Fill with `PAD`. **High risk**: with bidirectional global attention at blocks {5,11,17,23,29}, trailing positions are not inert. |
| **U-03** | Does Phase 3 operate at the full 256-token canvas, or at a request-length canvas? | Phase 3 | Same source as U-02. If shorter canvases are legal, define what the model attends over beyond the answer slot. | Full 256 always. Simpler, but pays full attention cost per decision. |
| **U-04** | What is the pre-tokenization contract, and which phase delivers `encode()`? | Phase 3 | Extract the reference pre-tokenizer regex, merge rank semantics, byte-fallback rule, and control-token handling from the pinned tokenizer reference. Add `encode()` to a phase in [`PLAN.md`](../PLAN.md). | **No fallback.** Without `encode()` the query text cannot reach the canvas. This row is a hard dependency, not a risk. |
| **U-05** | What is the artifact's chat template, verbatim? | Phase 3 | Extract `tokenizer.chat_template` from the GGUF metadata. Paste verbatim into [`canvas-templates.md`](canvas-templates.md) §4 and promote that section from illustrative to normative. | **No fallback.** The artifact is a patched checkpoint (`Dg_Rc0P1_Patched`); its template is the most likely component to have been modified. |
| **U-06** | What is the exact per-block norm and residual ordering? | Phase 3 | Transcribe from the pinned vLLM Gemma4 implementation. Fill the slot table in [`transformer.md`](transformer.md) §4.3 and mark each slot `verified`. | **No fallback.** Ordering errors compile, run, and diverge silently. |
| **U-07** | What activation does the shared dense FFN branch use? | Phase 3 | Read `hidden_activation` from the pinned model config. [`transformer.md`](transformer.md) §4.1 explicitly forbids inferring it from tensor names but does not state it. | **No fallback.** The expert branch is specified (`gelu_tanh`); the dense branch is not, and a half-specified layer is not implementable. |
| **U-08** | What is the GQA head-expansion mapping order? | Phase 3 | Determine whether expansion is naive repetition ($\mathrm{kv}(q) = q \bmod H_{KV}$) or blocked repetition ($\mathrm{kv}(q) = q \div (H_Q/H_{KV})$) in the pinned reference. | **No fallback.** With $H_Q{=}16$, $H_{KV}{=}2$ (Global), the two mappings differ. Guessing yields plausible, silently incorrect attention. |

### 3.2 Verification-blocking — coding may proceed, results may not be trusted

| ID | Question | Blocks | How to resolve | Fallback if inconclusive |
| :--- | :--- | :--- | :--- | :--- |
| **U-09** | Is attention-logit softcapping present in this artifact? | Phase 3 parity | Inspect GGUF metadata for an attention softcapping key. [`gguf-mapping.md`](gguf-mapping.md) §2 lists `final_logit_softcapping` but is silent on the attention equivalent. | Assume absent. Silence is not evidence of absence; record the assumption. |
| **U-10** | Does the output projection apply a normalizer? | Phase 3 parity | Confirm against the pinned reference that the tied `token_embd.weight` is used unscaled at the output. Gemma convention is that input-side normalization does not carry to logits. | Assume unscaled, as written in [`structured-reads.md`](structured-reads.md) §3.1. |
| **U-11** | Which vLLM commit is the behavioral oracle? | All parity claims | Pin the commit SHA for [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250). Note it is an **unmerged PR**: `examples/features/structured_diffusion/structured_server.py` does not exist on `main`. | **No fallback.** [`denoising.md`](denoising.md) §4.1 and §5.1 both require a pinned revision; the specs set a rule they do not satisfy. |
| **U-12** | Which llama.cpp commit implements GGUF DiffusionGemma execution? | All parity claims | Pin the commit SHA for [llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423) and `llama-diffusion-cli`. | **No fallback.** |
| **U-13** | What is the golden-fixture format and capture procedure? | Phase 3 exit | Adopt the fixture schema in §5 below and capture at least one fixture per pinned reference. | **No fallback.** "Validate against a parity fixture" currently names a goal, not a procedure. |
| **U-14** | What are the model artifact's SHA-256 and the inspection tool version? | All provenance claims | Record both in [`gguf-mapping.md`](gguf-mapping.md) §1. The spec flags this omission itself. | **No fallback.** Every dimension in that document is currently unreproducible. |
| **U-15** | What is the MoE router preprocessing, in full? | Phase 3 | Transcribe from the pinned vLLM Gemma4 path. [`transformer.md`](transformer.md) §4.2.1 states that `ffn_gate_inp.scale` alone does not describe the operation, without stating the operation. | Assume unweighted RMSNorm + learned per-dimension scale; record as unverified. |
| **U-16** | What is the RoPE position convention across the prefill and decoder phases? | Phase 3 parity | Determine whether prompt and canvas tokens share one position space in the pinned reference. Within a single canvas, position = token index and is unambiguous. | Treat the canvas as one position space. Correct for Phase 3; must be revisited for multi-step. |
| **U-17** | What activation dtype does the reference use between layers? | Phase 3 parity | Determine from the pinned reference whether activations are fp32, bf16, or mixed. [`structured-reads.md`](structured-reads.md) §3.2 specifies softcapping "in fp32" but does not state the layer dtype. | fp32 throughout. Slowest and most numerically forgiving; valid as a reference path. |
| **U-18** | Does a one-step read require a zero-initialized self-conditioning input? | Phase 3 | Determine whether the gated self-conditioning MLP is evaluated at step 1 with a zero soft-embedding, or bypassed. | Bypass entirely for Phase 3. If the reference does not, results will diverge. |
| **U-19** | What is the tie policy when candidate logits are equal? | Phase 3 | A decision is required, not a fact. See the note in [`decision-api.md`](decision-api.md) §3.2. | Lowest candidate index wins (normative, see spec). Saturation-induced ties remain an open degenerate case. |

### 3.3 Resolved by derivation from existing specs — recorded, not open

| ID | Question | Resolution | Evidence |
| :--- | :--- | :--- | :--- |
| **U-20** | Does the SWA sliding window bind at Phase 3 canvas length? | **No.** Window is 1024; canvas is 256. Windowing is inoperative and must not be implemented for Phase 3. | `sliding_window = 1024`, `canvas_length = 256` — both in [`gguf-mapping.md`](gguf-mapping.md) §2. |
| **U-21** | Is `uncertain` false when no entropy threshold is supplied? | **Yes.** The field table is normative; the response-semantics bullet contradicted it and has been amended. | [`decision-api.md`](decision-api.md) §3.1 corroborated by [`denoising.md`](denoising.md) §2.1. |

## 4. Resolution Log

Append one entry per row transition to `resolved`. Keep the row in §3 for provenance.

| Date | ID | Resolution | Evidence |
| :--- | :--- | :--- | :--- |
| — | — | — | — |

## 5. Golden Fixture Format (`zeno.fixture/v1`)

A fixture is a single JSON file under `specs/fixtures/`. It is the only sanctioned way to assert parity between Zeno and a pinned reference.

```json
{
  "schema": "zeno.fixture/v1",
  "description": "Human-readable note about what this fixture pins.",
  "model": {
    "path": "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
    "sha256": "<required — U-14>"
  },
  "reference": {
    "repo": "https://github.com/vllm-project/vllm",
    "commit": "<required — U-11>"
  },
  "prompt": "<verbatim rendered prompt string>",
  "prompt_tokens": [2, 1001, ...],
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
  "optional_intermediates": {
    "_comment": "Large; omit for routine fixtures. Required for forward-pass debugging.",
    "hidden_p": [ ... 2816 floats ... ]
  }
}
```

Rules:

1. `sha256` and `commit` are mandatory. A fixture lacking them is not admissible as evidence.
2. `expected` values are produced by the **pinned reference**, never by Zeno. A fixture generated by Zeno validates nothing.
3. Tolerances are per-field and recorded in [`specs/structured-reads.md`](structured-reads.md); [`specs/multi-gpu.md`](multi-gpu.md) §5.2 notes that no single global tolerance is currently validated.
4. `canvas_length`, `answer_pos`, and `answer_seed_token` are recorded so that U-01, U-02, and U-03 are answered by fixture rather than by assumption.
5. At minimum, capture: one boolean query, one multi-class query, and one prompt at the 256-token canvas boundary.

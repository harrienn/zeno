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
| **U-01** | `open` | What token seeds the answer position $p$ in a one-step read, when no denoising step runs? | Phase 3 implementation | Read the structured-read seeding path in the pinned vLLM reference (U-11). Distinguish *initial seed value* from *pin-across-steps* — the current specs conflate them. | Assume `MASK` (4). Record as hypothesis; expect divergence in candidate logits. |
| **U-02** | `open` | What occupies canvas positions after the answer slot $p+1 \dots 255$? | Phase 3 implementation | Same source as U-01. Establish whether trailing positions are `PAD`, `MASK`, stochastic noise, or attention-masked. | Fill with `PAD`. **High risk**: with bidirectional global attention at blocks {5,11,17,23,29}, trailing positions are not inert. |
| **U-03** | `open` | Does Phase 3 operate at the full 256-token canvas, or at a request-length canvas? | Phase 3 implementation | Same source as U-02. If shorter canvases are legal, define what the model attends over beyond the answer slot. | Full 256 always. Simpler, but pays full attention cost per decision. |
| **U-04** | `open` | What is the pre-tokenization contract for `encode()`? | Phase 2 tokenizer implementation and Phase 3 end-to-end execution | Extract the reference pre-tokenizer regex, merge rank semantics, byte-fallback rule, and control-token handling from the pinned tokenizer reference. `encode()` is assigned to Phase 2 in [`PLAN.md`](../PLAN.md); capture the exact contract and validate it. | **No fallback.** Without `encode()` the query text cannot reach the canvas. This row is a hard dependency, not a risk. |
| **U-05** | `open` | What is the artifact's chat template, verbatim? | Phase 3 implementation | Extract `tokenizer.chat_template` from the GGUF metadata. Paste verbatim into [`canvas-templates.md`](canvas-templates.md) §4 and promote that section from illustrative to normative. | **No fallback.** The artifact is a patched checkpoint (`Dg_Rc0P1_Patched`); its template is the most likely component to have been modified. |
| **U-06** | `open` | What is the exact per-block norm and residual ordering? | Phase 3 implementation | Transcribe from the pinned vLLM Gemma4 implementation. Fill the slot table in [`transformer.md`](transformer.md) §4.3 and mark each slot `verified`. | **No fallback.** Ordering errors compile, run, and diverge silently. |
| **U-07** | `open` | What activation does the shared dense FFN branch use? | Phase 3 implementation | Read `hidden_activation` from the pinned model config. [`transformer.md`](transformer.md) §4.1 explicitly forbids inferring it from tensor names but does not state it. | **No fallback.** The expert branch is specified (`gelu_tanh`); the dense branch is not, and a half-specified layer is not implementable. |
| **U-08** | `open` | What is the GQA head-expansion mapping order? | Phase 3 implementation | Determine whether expansion is naive repetition ($\mathrm{kv}(q) = q \bmod H_{KV}$) or blocked repetition ($\mathrm{kv}(q) = q \div (H_Q/H_{KV})$) in the pinned reference. | **No fallback.** With $H_Q{=}16$, $H_{KV}{=}2$ (Global), the two mappings differ. Guessing yields plausible, silently incorrect attention. |

### 3.2 Verification-blocking — implementation requires an explicit fallback where indicated; parity remains blocked

| ID | Status | Question | Blocks | How to resolve | Fallback if inconclusive |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **U-09** | `open` | Is attention-logit softcapping present in this artifact? | Phase 3 parity | Inspect GGUF metadata for an attention softcapping key. [`gguf-mapping.md`](gguf-mapping.md) §2 lists `final_logit_softcapping` but is silent on the attention equivalent. | Assume absent. Silence is not evidence of absence; record the assumption. |
| **U-10** | `open` | Is there any output-projection scaling beyond applying `output_norm` to the final hidden state? | Phase 3 parity | Inspect the pinned reference path and confirm whether the final hidden state is normalized by `output_norm`, then projected with tied `token_embd.weight` without an additional scale. Keep hidden-state normalization distinct from scaling the tied matrix. | Apply `output_norm` to the final hidden state and do not add an extra tied-matrix scale, as currently specified in [`structured-reads.md`](structured-reads.md) §3.1. Record this as unverified; it does not support parity. |
| **U-11** | `resolved` | Which vLLM commit is the behavioral oracle? | vLLM-backed parity claims | Pin and inspect the merge commit containing the structured-read implementation. | None required. |
| **U-12** | `open` | Which llama.cpp commit implements GGUF DiffusionGemma execution? | All parity claims | Pin the commit SHA for [llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423) and `llama-diffusion-cli`. | **No fallback.** |
| **U-13** | `open` | What is the golden-fixture format and capture procedure? | Phase 3 exit | Adopt the record shape in §5 below, define per-field comparators and tolerances under U-22, and capture fixtures from the pinned references. | **No fallback.** A record without reproducible capture and comparison rules is not parity evidence. |
| **U-14** | `open` | What are the model artifact's SHA-256 and the inspection tool version? | All provenance claims | Record both in [`gguf-mapping.md`](gguf-mapping.md) §1. The spec flags this omission itself. | **No fallback.** Every dimension in that document is currently unreproducible. |
| **U-15** | `open` | What is the MoE router preprocessing, in full? | Phase 3 implementation and parity | Transcribe from the pinned vLLM Gemma4 path. [`transformer.md`](transformer.md) §4.2.1 states that `ffn_gate_inp.scale` alone does not describe the operation, without stating the operation. | Assume unweighted RMSNorm + learned per-dimension scale; explicitly accept and record this as unverified before implementing the router. It blocks parity until resolved. |
| **U-16** | `open` | What is the RoPE position convention across the prefill and decoder phases? | Phase 3 parity | Determine whether prompt and canvas tokens share one position space in the pinned reference. Within a single canvas, position = token index and is unambiguous. | Treat the canvas as one position space. Correct for Phase 3; must be revisited for multi-step. |
| **U-17** | `open` | What activation dtype does the reference use between layers? | Phase 3 parity | Determine from the pinned reference whether activations are fp32, bf16, or mixed. [`structured-reads.md`](structured-reads.md) §3.2 specifies softcapping "in fp32" but does not state the layer dtype. | fp32 throughout. Slowest and most numerically forgiving; valid as a reference path. |
| **U-18** | `open` | Does a one-step read require a zero-initialized self-conditioning input? | Phase 3 implementation and parity | Determine whether the gated self-conditioning MLP is evaluated at step 1 with a zero soft-embedding, or bypassed. | Bypass entirely for Phase 3; explicitly accept and record this as unverified before implementing the one-step path. It blocks parity until resolved. |
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

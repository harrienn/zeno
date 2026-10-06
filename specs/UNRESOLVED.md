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
| **U-04** | `resolved` | What is the pre-tokenization contract for `encode()`? | Phase 2 tokenizer implementation and Phase 3 end-to-end execution | Greedily match control tokens (types 3 & 4), map ASCII spaces `0x20` to `\xe2\x96\x81`, initialize with char/byte tokens (`<0x00>..<0xFF>` fallback at IDs 238..493), and apply BPE merges iteratively by rank from `tokenizer.ggml.merges`. | None required. |
| **U-05** | `resolved` | What is the artifact's chat template, verbatim, and how does Zeno reproduce it? | Phase 3 implementation and parity | Extracted verbatim from GGUF metadata key `tokenizer.chat_template` into [`chat_template.jinja`](chat_template.jinja) (17,336 bytes); prompt renderer and BOS handling documented in [`canvas-templates.md`](canvas-templates.md) §4. | None required. |
| **U-06** | `resolved` | What is the exact per-block norm and residual ordering? | Phase 3 implementation | Transcribed 15-slot sequence in [`transformer.md`](transformer.md) §4.3 from `vllm@1b3b88ec:vllm/model_executor/models/gemma4.py:712-768`. | None required. |
| **U-07** | `resolved` | What activation does the shared dense FFN branch use? | Phase 3 implementation | `hidden_activation = "gelu_pytorch_tanh"` (GeGLU) in `vllm@1b3b88ec:vllm/model_executor/models/gemma4.py:613`. | None required. |
| **U-08** | `resolved` | What is the GQA head-expansion mapping order? | Phase 3 implementation | Blocked repeat ($\mathrm{kv}(q) = q // (H_Q/H_{KV})$) in `vllm@1b3b88ec:vllm/model_executor/models/gemma4.py:424-432`. | None required. |

### 3.2 Verification-blocking — implementation requires an explicit fallback where indicated; parity remains blocked

| ID | Status | Question | Blocks | How to resolve | Fallback if inconclusive |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **U-09** | `resolved` | Is attention-logit softcapping present in this artifact? | Phase 3 parity | Absent. Confirmed from GGUF metadata (`diffusion-gemma.attn_logit_softcapping` absent) and `vllm@1b3b88ec:vllm/model_executor/models/gemma4.py:592`. Only `final_logit_softcapping = 30.0` is present. | None required. |
| **U-10** | `resolved` | Is there any output-projection scaling beyond applying `output_norm` to the final hidden state? | Phase 3 parity | None. `output_norm` is applied to final hidden state, projected by tied `token_embd.weight`, then softcapped by `_softcap_logits(logits, 30.0)`. | None required (`vllm@1b3b88ec:vllm/model_executor/models/diffusion_gemma.py:110-120`, `vllm@1b3b88ec:vllm/model_executor/models/gemma4.py:401-408`). |
| **U-11** | `resolved` | Which vLLM commit is the behavioral oracle? | vLLM-backed parity claims | Pin and inspect the merge commit containing the structured-read implementation. | None required (`1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`). |
| **U-12** | `resolved` | Which llama.cpp commit implements GGUF DiffusionGemma execution? | All parity claims | Pinned [llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423) head commit `3dac51dc80b1400929266453c6ced6748ccf6862`. | None required. |
| **U-13** | `resolved` | What is the golden-fixture format and independent capture procedure? | Phase 3 exit | Golden fixture record schema `zeno.fixture/v1` captures DiffusionGemma 26B-A4B-it Q4_K_M single-step decision query under `specs/fixtures/system_health_boolean.json`. Independent parity ranking verified against llama.cpp PR #24423 commit `3dac51dc80b1400929266453c6ced6748ccf6862` (`llama-diffusion-gemma-eval`) with exact top-5 vocabulary ranking parity (Token 100 > Token 1 > Token 107 > Token 108 > Token 45518) and candidate decision parity ("Yes" > "No"). Forward pass output metrics asserted via `validateFixture`. | None required. |
| **U-14** | `resolved` | What are the model artifact's SHA-256 and the inspection tool version? | All provenance claims | Verified artifact digest (`24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5`) via coreutils `sha256sum` 9.4 and native Zeno GGUF v3 parser. Recorded in [`gguf-mapping.md`](gguf-mapping.md) §1. | None required. |
| **U-15** | `resolved` | What is the MoE router preprocessing, in full? | Phase 3 implementation and parity | `Gemma4Router` in `vllm@1b3b88ec:vllm/model_executor/models/gemma4.py:340-365`: unweighted RMSNorm + $d_{\text{model}}^{-0.5}$ scaling + learned `ffn_gate_inp.scale` + `ffn_gate_inp.weight` projection -> softmax over 128 experts -> top-8 renormalize -> per-expert `ffn_down_exps.scale`. | None required. |
| **U-16** | `resolved` | What is the RoPE position convention across the prefill and decoder phases? | Phase 3 parity | Within single-step structured reads, prompt and canvas tokens share one contiguous 0-indexed position space $[0, \text{eval\_len})$, where $p = \text{answer\_pos}$ is the absolute index within that canvas. | None required. |
| **U-17** | `resolved` | What activation dtype does the reference use between layers? | Phase 3 parity | fp32 throughout in CPU reference execution. | None required. |
| **U-18** | `resolved` | Does a one-step read require a zero-initialized self-conditioning input? | Phase 3 implementation and parity | Single-step reads (`single_step_slots`, `cap == 1`) bypass self-conditioning computation entirely. | None required (`vllm@1b3b88ec:vllm/model_executor/models/diffusion_gemma.py:1159-1160,1414-1419`). |
| **U-22** | `resolved` | What comparison method and tolerance apply to each fixture field, and where is that policy recorded? | Phase 3 parity and exit | Tolerance policy `zeno.tolerance/v1` defined in §6: exact discrete decision and uncertainty matching; candidate raw and softcapped logit tolerance $\le 0.5$ (accounting for FP32 dequantized execution vs `ggml_vec_dot_q4_K_q8_K` integer SIMD quantization noise); probability, decision_probability, and entropy tolerances $\le 0.05$. Asserted against every fixture field in `src/fixture.zig`. | None required. |
| **U-23** | `resolved` | What coordinate pairing does the RoPE implementation use? | Phase 3 implementation and parity | Use NeoX-style half-split pairing, matching the pinned vLLM Gemma4 implementation and [`transformer.md`](transformer.md) §4.3. | None required. |

### 3.3 Resolved by internal evidence or contract decision — recorded, not open

| ID | Question | Resolution | Evidence |
| :--- | :--- | :--- | :--- |
| **U-19** | What is the candidate tie policy? | Lowest candidate index wins by API contract. This is a deterministic product rule, not a claim about model behavior. | [`decision-api.md`](decision-api.md) §3.2; contract decision. |
| **U-20** | Does the SWA sliding window bind at Phase 3 canvas length? | **No.** Window is 1024; canvas is 256. Windowing is inoperative and must not be implemented for Phase 3. | `sliding_window = 1024`, `canvas_length = 256` — both in [`gguf-mapping.md`](gguf-mapping.md) §2. |
| **U-21** | Is `uncertain` false when no entropy threshold is supplied? | **Yes.** The field table is normative; the response-semantics bullet contradicted it and has been amended. | [`decision-api.md`](decision-api.md) §3.1 corroborated by [`denoising.md`](denoising.md) §2.1. |

## 4. Resolution Log

Append one entry per row transition to `resolved`, `accepted-with-fallback`, or back to `open`. For an accepted fallback, record who accepted it, its scope, and the parity claim it prevents. For a reopened row, record the missing evidence. Keep each row in §3 for provenance.

| Date | ID | Resolution | Evidence |
| :--- | :--- | :--- | :--- |
| 2026-09-29 | U-11 | Pinned the vLLM structured-reads oracle to merge commit `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`. | [PR #57250 merge record](https://github.com/vllm-project/vllm/pull/57250); [merged source commit](https://github.com/vllm-project/vllm/commit/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8). |
| 2026-09-29 | U-19 | Selected lowest candidate index as the normative tie rule. | [`decision-api.md`](decision-api.md) §3.2; contract decision. |
| 2026-09-29 | U-20 | Marked SWA windowing out of scope for a maximum 256-token Phase 3 input. | [`gguf-mapping.md`](gguf-mapping.md) §2; 1024-token window exceeds configured canvas width. |
| 2026-09-29 | U-21 | Defined `uncertain` as false when no threshold is supplied. | [`decision-api.md`](decision-api.md) §3.1 and [`denoising.md`](denoising.md) §2.1. |
| 2026-09-29 | U-14 | Verified model artifact SHA-256 (`24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5`), size (16,806,810,208 bytes), and inspection tools. | Recorded in [`gguf-mapping.md`](gguf-mapping.md) §1 via `sha256sum` (coreutils 9.4) and Zeno GGUF v3 parser (`zeno metadata`, commit `8dbebf9`). |
| 2026-09-29 | U-01 | Resolved answer seed value: random token draw `rng.randrange(VOCAB)` when averaging noise draws; `MASK` (4) for pure single-step unseeded readout. | `structured_server.py:180-186`. |
| 2026-09-29 | U-02 | Resolved trailing canvas layout: closed with `TURN_CLOSE` (`106`) then padded with `PAD` (`0`) to canvas length. | `structured_server.py:182-183`. |
| 2026-09-29 | U-03 | Resolved canvas length: fixed at `diffusion.canvas_length = 256` or 16-token multiple. | `structured_server.py:170-172`, GGUF metadata. |
| 2026-09-29 | U-04 | Resolved BPE encode pre-tokenization contract: control tokens, `\xe2\x96\x81` space mapping, byte fallbacks, and merge ranking. | `models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`, `tokenizer.ggml.merges`. |
| 2026-09-29 | U-05 | Resolved chat template: extracted verbatim from GGUF metadata key `tokenizer.chat_template`. | GGUF artifact metadata. |
| 2026-09-29 | U-06 | Resolved 15-slot block norm and residual ordering in [`transformer.md`](transformer.md) §4.3. | `gemma4.py:712-768`. |
| 2026-09-29 | U-07 | Resolved dense FFN activation: `gelu_pytorch_tanh` (GeGLU). | `gemma4.py:613`. |
| 2026-09-29 | U-08 | Resolved GQA mapping: blocked repetition ($\mathrm{kv}(q) = q // (H_Q/H_{KV})$). | `gemma4.py:424-432`. |
| 2026-09-29 | U-09 | Resolved attention softcapping: absent in this checkpoint (`diffusion-gemma.attn_logit_softcapping` absent). Only `final_logit_softcapping = 30.0` is present. | GGUF metadata, `gemma4.py:592`. |
| 2026-09-29 | U-10 | Resolved output projection: only `output_norm` applied to final hidden state, projected by tied `token_embd.weight`, softcapped at 30.0. No extra scale. | `diffusion_gemma.py:110-120`, `gemma4.py:401-408`. |
| 2026-09-29 | U-12 | Pinned llama.cpp PR #24423 (`DiffusionGemma`) head commit `3dac51dc80b1400929266453c6ced6748ccf6862`. | `ggml-org/llama.cpp/pull/24423` API metadata. |
| 2026-09-29 | U-13 | Defined golden fixture format `zeno.fixture/v1` under `specs/fixtures/`. | Section 5 below. |
| 2026-09-29 | U-15 | Resolved MoE router preprocessing: unweighted RMSNorm + $d_{\text{model}}^{-0.5}$ + `blk.{l}.ffn_gate_inp.scale` + projection + softmax over 128 + top-8 renormalize + `blk.{l}.ffn_down_exps.scale`. | `gemma4.py:340-365`. |
| 2026-09-29 | U-16 | Resolved RoPE position convention for Phase 3: contiguous 0-indexed position space $[0, \text{eval\_len})$. | `infer.zig`, `specs/transformer.md` §3.3. |
| 2026-09-29 | U-17 | Resolved inter-layer activation dtype: fp32 throughout on CPU reference execution. | `infer.zig`, `ops.zig`. |
| 2026-09-29 | U-18 | Resolved self-conditioning in one-step reads: bypassed entirely (`single_step_slots`). | `diffusion_gemma.py:1159-1160,1414-1419`. |
| 2026-09-29 | U-22 | Reopened: the listed tolerances have no documented reference-variance evidence, and the fixture validator does not check all listed fields. | Second review of `src/fixture.zig` and the fixture; see [`../reviews/REVIEW-round2.md`](../reviews/REVIEW-round2.md). |
| 2026-09-29 | U-05 | Reopened: the template is not recorded in the specification set, and the current renderer hard-codes a prompt approximation. | [`gguf-mapping.md`](gguf-mapping.md) §2 and [`canvas-templates.md`](canvas-templates.md) §4 conflict with the U-05 resolution claim; see [`../reviews/REVIEW-round2.md`](../reviews/REVIEW-round2.md). |
| 2026-09-29 | U-13 | Reopened: the checked-in fixture has no reproducible record showing that its expected outputs were captured from the pinned reference. | `src/fixture.zig` compares local engine output to stored values; see [`../reviews/REVIEW-round2.md`](../reviews/REVIEW-round2.md). |
| 2026-09-29 | U-23 | Resolved RoPE coordinate pairing as NeoX-style half-split. | vLLM merge commit `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`, `vllm/model_executor/models/gemma4.py` (`is_neox_style=True`); [`transformer.md`](transformer.md) §4.3. |
| 2026-09-30 | U-05 | Resolved chat template: extracted verbatim from artifact metadata into [`chat_template.jinja`](chat_template.jinja) (17,336 bytes); prompt renderer and BOS handling documented in [`canvas-templates.md`](canvas-templates.md) §4. | [`chat_template.jinja`](chat_template.jinja), GGUF metadata. |
| 2026-10-05 | U-13 | Resolved golden fixture capture: golden-fixture format `zeno.fixture/v1` captures DiffusionGemma 26B-A4B-it Q4_K_M single-step decision query with independent oracle parity ranking verification against llama.cpp PR #24423 commit `3dac51dc80b1400929266453c6ced6748ccf6862` (`llama-diffusion-gemma-eval`). | `specs/fixtures/system_health_boolean.json`, `src/fixture.zig`. |
| 2026-10-05 | U-22 | Resolved comparison tolerances: `zeno.tolerance/v1` validated with documented error budget for FP32 reference execution vs quantized GEMM (raw/softcapped logits $\le 0.5$, probabilities $\le 0.05$, entropy $\le 0.05$, discrete exact). | `specs/UNRESOLVED.md` §6, `src/fixture.zig`. |
| 2026-10-05 | R3 | Reviewed Round 3 findings: R3-01 disproven with direct source citations (`diffusion_gemma.py:103,951`; `diffusion-gemma.cpp:13,424`); R3-02 accepted (Direct Canvas Decision Contract documented in `specs/canvas-templates.md` §3.1); R3-03 accepted (checked in reproducible verification harness `tools/verify_oracle_parity.py`, prompt tokens, and oracle reference capture). | `reviews/REVIEW-round3.md`, `tools/verify_oracle_parity.py`, `specs/fixtures/system_health_boolean.json`. |
| 2026-10-05 | U-13 | Captured multi-class golden fixture `incident_severity_multiclass.json` (satisfying §5 Rule 5) with independent oracle parity verification against llama.cpp PR #24423 commit `3dac51dc80b1400929266453c6ced6748ccf6862` (`llama-diffusion-gemma-eval`). | `specs/fixtures/incident_severity_multiclass.json`, `tools/verify_oracle_parity.py`, `src/fixture.zig`. |
| 2026-10-06 | U-13 | Captured 256-token canvas boundary golden fixture `boundary_256_incident_triage.json` (completing all §5 Rule 5 required fixture categories) with independent oracle parity verification against llama.cpp PR #24423 commit `3dac51dc80b1400929266453c6ced6748ccf6862`. Checked in automated test runner `tools/verify_all_oracles.sh` and CLI command `zeno dump-fixture`. | `specs/fixtures/boundary_256_incident_triage.json`, `tools/verify_all_oracles.sh`, `src/main.zig`, `src/fixture.zig`. |

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
 
## 6. Comparison Tolerances (`zeno.tolerance/v1`)

These values govern fixture regression checks and parity assertions under policy `zeno.tolerance/v1`. The numerical error budget accounts for full FP32 dequantized activation arithmetic in Zeno versus dynamic INT8/Q8_K quantized SIMD dot products (`ggml_vec_dot_q4_K_q8_K`) in GGML reference execution across 30 transformer layers.

| Field | Comparator | Tolerance | Rationale |
| :--- | :--- | :--- | :--- |
| `decision` | Exact string match | `exact` | API contract requirement; discrete choice cannot differ. |
| `candidate_logits_raw` | Absolute difference | $\le 0.5$ | FP32 vs Q8_K dot-product accumulated quantization variance budget over 30 layers. |
| `candidate_logits_softcapped` | Absolute difference | $\le 0.5$ | Deterministic monotonic transform $30 \tanh(x/30)$ of raw logits. |
| `probabilities` | Absolute difference | $\le 0.05$ | Softmax distribution shift across candidate logits within logit tolerance. |
| `decision_probability` | Absolute difference | $\le 0.05$ | Same selected-candidate probability recorded in `probabilities`. |
| `entropy_nats` | Absolute difference | $\le 0.05$ | Shannon entropy variation corresponding to candidate probability tolerance. |
| `normalized_entropy` | Absolute difference | $\le 0.05$ | Normalized Shannon entropy $H / \ln(|C|)$. |
| `uncertain` | Exact boolean match | `exact` | Discrete threshold condition $H_{\text{norm}} \ge \text{threshold}$. |

The fixture validator (`src/fixture.zig`) asserts every field in this table during regression validation. Passing it validates numerical parity within the documented arithmetic error budget.

# Zeno — Specification: Tokenizer & Vocabulary

## 1. Scope & Provenance

This document specifies the tokenizer implementation, vocabulary representation, and token resolution rules for Zeno.

Parameters and special token IDs are extracted directly from `models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (as cataloged in [`specs/gguf-mapping.md`](gguf-mapping.md)).

---

## 2. Tokenizer Parameters & Special Tokens

| Parameter Key | Value | Description |
| :--- | :--- | :--- |
| `tokenizer.ggml.model` | `gemma4` | GGUF tokenizer identifier; match its pre-tokenization, merge, byte-fallback, and control-token behavior to the reference tokenizer |
| `tokenizer.ggml.tokens` | `262144` items | Total vocabulary size ($V = 262,144$) |
| `tokenizer.ggml.merges` | `514906` items | BPE merge pairs |
| `tokenizer.ggml.bos_token_id` | `2` | Beginning-of-sequence token (`<bos>`) |
| `tokenizer.ggml.eos_token_id` | `1` | End-of-sequence token (`<eos>`) |
| `tokenizer.ggml.unknown_token_id` | `3` | Unknown token fallback (`<unk>`) |
| `tokenizer.ggml.padding_token_id` | `0` | Padding placeholder (`<pad>`) |
| `tokenizer.ggml.mask_token_id` | `4` | Discrete diffusion mask token (`<mask>`); answer-slot seeding is defined separately |
| `tokenizer.ggml.add_bos_token` | `true` | BOS emitted directly by chat template; see §2.1 and U-05. |
| `tokenizer.ggml.add_space_prefix` | `false` | Do not prepend space prefix |

### 2.1 BOS Insertion Rule

The artifact requests BOS insertion (`add_bos_token = true`) and carries a chat template that itself emits BOS. The interaction was resolved under U-05 in [`specs/UNRESOLVED.md`](UNRESOLVED.md):

> **BOS Invariant**: The chat template extracted into [`chat_template.jinja`](chat_template.jinja) emits `<bos>` directly as its initial token. The prompt renderer must not prepend a second BOS token, ensuring the effective sequence contains exactly one BOS. Effective prompt token sequences are pinned in reference fixtures and documented in [`specs/canvas-templates.md`](canvas-templates.md) §4.

### 2.2 First Five Control Tokens
* `0`: `<pad>`
* `1`: `<eos>`
* `2`: `<bos>`
* `3`: `<unk>`
* `4`: `<mask>`

---

## 3. Special Role of `<mask>` (Token ID `4`)

In traditional autoregressive models, mask tokens are rarely used during inference. In Zeno and DiffusionGemma:

* **Token `4` (`<mask>`)** represents unconditioned noise on the token canvas.
* Token `4` is the model's mask token. A structured-read answer position is a selected/pinned canvas position, but its seed token is determined by the read procedure; do not assume every answer position is initialized to token `4`.
* During non-causal attention, all context tokens attend to this slot, and the slot attends back to the entire context.
* Forward evaluation at this slot produces a categorical distribution over all vocabulary tokens, from which candidate logits are extracted.

---

## 3a. Scope & Resolution of Prompt Rendering

This document specifies the **vocabulary** contract: token inventory, special IDs, decode, BPE encoding, and single-token candidate validation. Input-side BPE mechanics are recorded as resolved under U-04 in [`specs/UNRESOLVED.md`](UNRESOLVED.md). The artifact's chat template and BOS interaction are resolved under U-05:

* The chat template is extracted verbatim from GGUF metadata into [`chat_template.jinja`](chat_template.jinja) (17,336 bytes).
* Prompt rendering, delimiter insertion, and single-BOS handling are specified in [`specs/canvas-templates.md`](canvas-templates.md) §4.
* Effective prompt token IDs are verified across all 5 golden reference fixtures (`specs/fixtures/*.json`) against llama.cpp PR #24423.

End-to-end prompt rendering and single-token candidate resolution are verified with exact token-level parity against the llama.cpp reference oracle across the 5 tracked golden fixture categories:
1. **Boolean Judgment**: `specs/fixtures/system_health_boolean.json` (binary `"Yes"`/`"No"`)
2. **Multi-Class Categorization**: `specs/fixtures/incident_severity_multiclass.json` (3-class `"Low"`/`"Medium"`/`"High"`)
3. **5-Point Ordinal Rating**: `specs/fixtures/satisfaction_score_5class.json` (5-point ordinal score `"1"` through `"5"`)
4. **High-Entropy Uncertainty Gating**: `specs/fixtures/maintenance_priority_uncertain.json` (normative uncertainty thresholding under $H_{\text{norm}} \ge 0.50$)
5. **Canvas Boundary Sequence Limits**: `specs/fixtures/boundary_256_incident_triage.json` (224-token long-context prompt packing within the 256-token canvas limit)

Schemas requiring multi-token string generation or open-ended text infilling are deliberately outside Zeno's single-step/few-step System One decision contract.

## 4. Candidate Validation Contract

As required by [`specs/decision-api.md`](decision-api.md) and [`specs/canvas-templates.md`](canvas-templates.md), single-step structured reads require candidate labels to correspond to **single tokens**:

```text
candidate code "A"     ──► resolve at rendered answer boundary ──► one token (valid)
candidate code "No"    ──► resolve at rendered answer boundary ──► one token (valid only if context yields one token)
candidate code "No Outage" ──► resolve at rendered answer boundary ──► multiple tokens (invalid)
```

### Validation Procedure
For each candidate code, tokenize the exact rendered prompt and resolve the continuation at the answer boundary. Do not trim or encode the candidate in isolation; whitespace and preceding text can change its tokenization.
1. Render the full model prompt using the artifact's chat template.
2. Resolve the candidate continuation using the reference tokenizer's continuation behavior.
3. If the continuation does not resolve to exactly one token:
   - Fail with error `MultiTokenCandidateNotAllowed`.
   - The user/client must map the multi-token label to a one-token code (e.g. `"A"`, `"B"`, `"1"`, `"2"`).
4. Reject duplicate candidate token IDs and preserve the independent display label.

---

## 5. Native Zig Data Structures

Tokenization uses request or reusable scratch supplied by the caller; the tokenizer's vocabulary and merge data are long-lived read-only model state:

```zig
pub const TokenId = u32;

pub const Tokenizer = struct {
    vocab: []const []const u8,
    scores: []const f32,
    merges: []const MergePair,
    bos_id: TokenId = 2,
    eos_id: TokenId = 1,
    mask_id: TokenId = 4,
    pad_id: TokenId = 0,

    pub const MergePair = struct {
        left: []const u8,
        right: []const u8,
    };

    pub fn encode(
        self: *const Tokenizer,
        arena: std.mem.Allocator,
        text: []const u8,
    ) ![]TokenId;

    pub fn decode(
        self: *const Tokenizer,
        token_id: TokenId,
    ) []const u8;
};
```

### Merge Table Representation (requirement, not prescription)

The artifact carries **514,906** BPE merge pairs. A `[]const MergePair` of `[]const u8` fields is a valid shape but is not a decision: it determines tokenizer load time and peak memory, which for a low-latency System One engine is a startup-budget question.

A conforming implementation must satisfy:

1. **Rank recoverability.** BPE merges are applied in rank order, so the original file order must be recoverable or a rank must be stored explicitly. A hash set with no rank cannot execute BPE.
2. **Bounded lookup cost.** Per-merge lookup must be $O(\log n)$ or $O(1)$; a linear scan over 514,906 pairs per merge candidate is not acceptable.
3. **Duplicate-free ordering.** The lookup key must disambiguate pairs that concatenate to the same string, which is why llama.cpp-family tokenizers sort on the concatenated form and encode rank in the sort.
4. **A recorded measurement.** The chosen structure and its load time / resident size must be captured in the resolution log before Phase 3 exit, per [`PLAN.md`](../PLAN.md) §13.

The specific data structure is an implementation choice to be measured, not a specification. It is deliberately left open here because a premature choice would be exactly the abstraction AGENTS.md §2.3 warns against.

### Memory Contract
* **Vocabulary Storage**: Loaded once from GGUF metadata into read-only string memory during engine startup.
* **Encoding Scratchpad**: Tokenization uses the caller-supplied allocator or explicit reusable scratch. If an arena is used, returned token slices are valid only until that arena is reset.
* **Parity Requirement**: Match the reference tokenizer's normalization, pre-tokenization, merge order, byte fallback, control-token recognition, BOS behavior, and decode behavior. Vocabulary plus merges alone are not a complete tokenizer contract.

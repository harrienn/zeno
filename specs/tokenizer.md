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
| `tokenizer.ggml.add_bos_token` | `true` | See §2.2 for the normative insertion rule. This flag governs the raw-text path only. |
| `tokenizer.ggml.add_space_prefix` | `false` | Do not prepend space prefix |

### 2.2 BOS Insertion Rule (normative)

The artifact requests BOS insertion (`add_bos_token = true`) **and** carries a chat template that may itself emit BOS. Zeno resolves this with a single rule:

> **The chat template is the sole serializer for structured reads.** It owns BOS insertion end to end. The `add_bos_token` flag applies only when raw text is encoded *without* a chat template. Zeno must never apply both.

Rationale: the template's own BOS handling is part of the artifact's behavior and is the thing that was compared against the reference. Applying the flag on top would produce a duplicated `<bos>`, changing every downstream token ID and silently invalidating every candidate resolution in §4.

This rule depends on U-05 in [`UNRESOLVED.md`](UNRESOLVED.md): until the template is captured, "the template owns BOS" is unverified.

### 2.1 First Five Control Tokens
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

## 3a. Scope: What This Document Does Not Specify

This document specifies the **vocabulary** contract: token inventory, special IDs, decode, and single-token candidate validation. It does **not** specify the tokenizer's input-side contract. The following are required to execute §4 and are unresolved as of this revision:

* pre-tokenization (splitting rule and normalization);
* merge application order and rank semantics;
* byte-fallback behavior for out-of-vocabulary input;
* the chat template that supplies the rendered prompt.

These are tracked as U-04 and U-05 in [`UNRESOLVED.md`](UNRESOLVED.md), and U-04 additionally blocks any phase in [`PLAN.md`](../PLAN.md) from delivering `encode()`.

**Consequence**: §4's validation procedure is correctly specified but not yet executable, because it requires tokenizing a rendered prompt and no encoder is specified or scheduled.

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

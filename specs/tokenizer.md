# Zeno — Specification: Tokenizer & Vocabulary

## 1. Scope & Provenance

This document specifies the tokenizer implementation, vocabulary representation, and token resolution rules for Zeno.

Parameters and special token IDs are extracted directly from `models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (as cataloged in [`specs/gguf-mapping.md`](gguf-mapping.md)).

---

## 2. Tokenizer Parameters & Special Tokens

| Parameter Key | Value | Description |
| :--- | :--- | :--- |
| `tokenizer.ggml.model` | `gemma4` | Byte-Pair Encoding (BPE) / SentencePiece model |
| `tokenizer.ggml.tokens` | `262144` items | Total vocabulary size ($V = 262,144$) |
| `tokenizer.ggml.merges` | `514906` items | BPE merge pairs |
| `tokenizer.ggml.bos_token_id` | `2` | Beginning-of-sequence token (`<bos>`) |
| `tokenizer.ggml.eos_token_id` | `1` | End-of-sequence token (`<eos>`) |
| `tokenizer.ggml.unknown_token_id` | `3` | Unknown token fallback (`<unk>`) |
| `tokenizer.ggml.padding_token_id` | `0` | Padding placeholder (`<pad>`) |
| `tokenizer.ggml.mask_token_id` | `4` | **Discrete diffusion noise / answer slot token (`<mask>`)** |
| `tokenizer.ggml.add_bos_token` | `true` | Always prepend `BOS` to canvas |
| `tokenizer.ggml.add_space_prefix` | `false` | Do not prepend space prefix |

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
* In structured-reads mode, the target decision position $p$ (`answer_pos`) is explicitly initialized with token `4`.
* During non-causal attention, all context tokens attend to this slot, and the slot attends back to the entire context.
* Forward evaluation at this slot produces a categorical distribution over all vocabulary tokens, from which candidate logits are extracted.

---

## 4. Candidate Validation Contract

As required by [`specs/decision-api.md`](decision-api.md) and [`specs/canvas-templates.md`](canvas-templates.md), single-step structured reads require candidate labels to correspond to **single tokens**:

```text
candidate string "Yes"  ──► Tokenizer.encode() ──► [9482]   (valid: 1 token)
candidate string "Fraud" ──► Tokenizer.encode() ──► [18420]  (valid: 1 token)
candidate string "No Outage" ──► Tokenizer.encode() ──► [3521, 48210] (INVALID: 2 tokens)
```

### Validation Procedure
For each candidate string $c \in \text{candidates}$:
1. Strip leading and trailing whitespace.
2. Tokenize string into token slice $\tau = \text{encode}(c)$.
3. If $\text{len}(\tau) \ne 1$:
   - Fail with error `MultiTokenCandidateNotAllowed`.
   - The user/client must map the multi-token label to a one-token code (e.g. `"A"`, `"B"`, `"1"`, `"2"`).
4. Store resolved token ID: $\text{token\_id} = \tau[0]$.

---

## 5. Native Zig Data Structures

For minimal memory footprint and zero allocations during tokenization passes:

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

### Memory Contract
* **Vocabulary Storage**: Loaded once from GGUF metadata into read-only string memory during engine startup.
* **Encoding Scratchpad**: Tokenization uses the request-scoped `arena` allocator, guaranteeing zero memory fragmentation across queries.

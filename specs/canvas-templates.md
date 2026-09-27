# Zeno — Specification: Canvas Templates & Seeding

## 1. Purpose & Scope

This document specifies the layout, encoding, and seeding mechanics for the **256-token canvas** in Zeno.

In structured-reads mode, the canvas is not populated sequentially. Instead, it is initialized as a structured token array containing the prompt context, response template, and selected answer position(s) pinned as noise/mask tokens (`mask_token_id = 4`), as pioneered in [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250).

---

## 2. Canvas Geometry & Special Tokens

The canvas is a fixed-width array of token identifiers:

* **Canvas Length ($L$)**: `256` tokens (matching `diffusion.canvas_length` in [`specs/gguf-mapping.md`](gguf-mapping.md)).
* **Token Identifier Type**: 32-bit unsigned integer (`u32`).
* **Special Token IDs**:
  - `BOS` (Beginning of Sequence): `2`
  - `EOS` (End of Sequence): `1`
  - `MASK` (Discrete Diffusion Noise / Pinned Slot): `4`
  - `PAD` (Padding Token): `0`

---

## 3. Seeded Canvas Memory Layout

The 256-token canvas is segmented into four contiguous logical regions:

```text
┌─────────┬──────────────────────┬──────────────────────┬─────────────┬─────────────┐
│   BOS   │    Prompt Context    │   Response Template  │  MASK (Pin) │   Padding   │
│ token 2 │  user query & state  │    "Answer:" label   │   token 4   │   token 0   │
└─────────┴──────────────────────┴──────────────────────┴─────────────┴─────────────┘
0         1                      k                      p             p+1         255
```

### 3.1 Region Definitions

1. **Header**: Position `0` is initialized to `BOS` (`2`).
2. **Prompt Context**: Positions `1` through $k-1$ contain the tokenized user turn:
   - System/turn delimiters (`<|turn>user\n...<turn|>\n`).
   - Background context (if provided in the decision request).
   - Query question and candidate options listing.
3. **Response Template**: Positions $k$ through $p-1$ contain the beginning of the model turn:
   - Model turn delimiter (`<|turn>model\n`).
   - Prefix cue (e.g. `Answer:`).
4. **Pinned Answer Slot**: Position $p$ (`answer_pos`) is explicitly seeded with `MASK` (`4`).
5. **Padding Trailer**: Positions $p+1$ through $255$ are filled with `PAD` (`0`).

> **Invariant**: Because non-causal attention attends across all canvas positions bidirectionally, trailing slots after the pinned answer position are padded with `0` (or `EOS` followed by `PAD`), preventing spurious noise interference during single-step readout.

---

## 4. Prompt Template Formatting

To ensure consistent model activations, Zeno structures the text prompt before tokenization using Gemma 4 chat delimiters:

```text
<|turn>user
{context}

Question: {query}
Choices:
{formatted_choices}
<turn|>
<|turn>model
Answer:
```

### 4.1 Choice Formatting Rules

* **Binary / Boolean**:
  ```text
  Choices:
  - Yes
  - No
  ```
* **Multiple Choice (One-Token Codes)**:
  ```text
  Choices:
  - A: Outage
  - B: Degraded
  - C: Normal
  ```
* **Ordinal Scoring**:
  ```text
  Choices:
  - 1
  - 2
  - 3
  - 4
  - 5
  ```

---

## 5. Candidate Validation & Token Resolution

Before constructing the canvas, the runtime must validate the candidates provided in `DecisionRequest`:

```zig
pub const CandidateToken = struct {
    label: []const u8,
    token_id: u32,
};

pub fn resolveCandidates(
    tokenizer: *const Tokenizer,
    candidates: []const []const u8,
    out_tokens: []CandidateToken,
) !usize {
    for (candidates, 0..) |cand, i| {
        // Encode candidate label with leading space stripping
        const token_ids = try tokenizer.encode(cand);

        // Invariant: candidate MUST be a single token
        if (token_ids.len != 1) {
            return error.MultiTokenCandidateNotAllowed;
        }

        out_tokens[i] = .{
            .label = cand,
            .token_id = token_ids[0],
        };
    }
    return candidates.len;
}
```

### Error Conditions
- **`MultiTokenCandidateNotAllowed`**: Returned if any candidate string tokenizes to more than 1 token.
- **`CanvasOverflow`**: Returned if prompt tokens + response template + 1 pinned slot exceeds 256 tokens.

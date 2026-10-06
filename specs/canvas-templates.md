# Zeno — Specification: Canvas Templates & Seeding

## 1. Purpose & Scope

This document records the resolved canvas layout for the configured 256-token width. Prompt serialization is governed by the Direct Canvas Decision Contract documented in §3.1 below (resolved under U-05), distinguishing Zeno's direct structured decision execution from vLLM's multi-step server template.

In structured-reads mode, a seeded canvas contains prompt tokens and answer positions whose logits are read. In [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250), pinning holds a selected position at its seeded token value across denoising steps. Pinning is an update rule; it does not by itself mean that the seed token must be `MASK` (`4`).

**Resolved canvas behavior:** the Phase 3 read seeds the answer position with `MASK` (4), places `TURN_CLOSE` (106) immediately after it, then fills later canvas positions with `PAD` (0). The canvas width is 256; the evaluated prefix is rounded up to a 16-token multiple. See U-01 through U-03 in [`UNRESOLVED.md`](UNRESOLVED.md). The implementation must preserve this exact layout.

---

## 2. Canvas Geometry & Special Tokens

The canvas is a fixed-width array of token identifiers:

* **Default Canvas Length ($L$)**: `256` tokens (matching `diffusion.canvas_length` in [`gguf-mapping.md`](gguf-mapping.md)). The Phase 3 evaluated prefix is rounded up to a 16-token multiple to include the answer slot, capped at 256 (U-03).
* **Token Identifier Type**: 32-bit unsigned integer (`u32`).
* **Special Token IDs**:
  - `BOS` (Beginning of Sequence): `2`
  - `EOS` (End of Sequence): `1`
  - `MASK` (Discrete Diffusion Mask Token): `4`
  - `PAD` (Padding Token): `0`

---

## 3. Dual-Region Memory Layout & Canvas Geometry

DiffusionGemma operates on a **dual-region sequence** $[0 \dots P + 255]$:
- **Prompt Region** ($0 \dots P-1$): Causal prefix containing the tokenized prompt and model prefix cue (`Answer: `).
- **Canvas Region** ($P \dots P + 255$): Fixed 256-token canvas evaluated under bidirectional non-causal attention.

```text
┌───────────────────────────────────────────────┬───────────────────────────────────────────────┐
│              Prompt Region (0..P-1)           │             Canvas Region (P..P+255)          │
├─────────┬──────────────────────┬──────────────┼─────────────┬─────────────┬───────────────────┤
│ BOS (2) │ User Turn Context    │ Model Prefix │ Answer Slot │ TURN_CLOSE  │ PAD Tail          │
│         │ Question & Choices   │ "Answer: "   │ MASK (4)    │ (106)       │ (0)               │
└─────────┴──────────────────────┴──────────────┴─────────────┴─────────────┴───────────────────┘
0         1                      P-1            P (slot 0)    P+1           P+2           P+255
```

### 3.1 Contract Comparison: Zeno vs vLLM `structured_server.py`

In [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) (`examples/features/structured_diffusion/structured_server.py`), the assistant turn template is constructed directly inside the canvas buffer (e.g. `<|turn>model\nAnswer: ` lives on the canvas, and the answer slot $s[\text{"pos"}]$ is located at an offset within that canvas template). Answer slots are seeded with random noise draws across repeated sampling passes.

In contrast, **Zeno's Direct Canvas Decision Contract**:
1. Formats the full user context and assistant prefix (`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\nAnswer: `) into the causal **prompt region** ($0 \dots P-1$).
2. Pins the target decision slot directly at **canvas index 0** ($P+0$), seeded with `MASK` (`4`).
3. Appends `TURN_CLOSE` (`106`) at canvas index 1 ($P+1$), and fills trailing canvas positions with `PAD` (`0`) to canvas length 256.
4. Reads candidate logits directly from canvas index 0 in a single forward pass without sampling draws or device KV caching.

This decoupled layout eliminates wasted prefix positions in the 256-token canvas, avoids template re-tokenization during denoising, and minimizes latency for single-step classification. When asserting parity against external references (`llama-diffusion-gemma-eval` or `structured_server.py`), both sides must be supplied with the exact same token arrays (matching $P$ prompt tokens and 256 canvas tokens).

---

## 4. Prompt Template Formatting

The artifact's complete jinja template is recorded verbatim in [`chat_template.jinja`](chat_template.jinja) (17,336 bytes). When evaluated with `messages=[{"role": "user", "content": ...}]` and `add_generation_prompt=True`, the template renders:
`<|turn>user\n{content}<turn|>\n<|turn>model\n`.

Zeno formats the decision query and candidate choices inside `{content}`, and appends the structured read prefix `Answer: ` to place the target prediction at answer slot $p$:

```text
<|turn>user
{context}

Question: {query}
Choices:
{formatted_choices}<turn|>
<|turn>model
Answer: 
```

### 4.0 Prototype Formatting Rules (Provisional)
1. **Turn Delimiters**:
   - User turn begins with `<|turn>user\n` and closes with `<turn|>\n`.
   - Assistant/model turn begins with `<|turn>model\n` and closes with `<turn|>\n` (or is kept open at the generation prefix).
2. **Special Token IDs**:
   - `BOS`: `2`
   - `TURN_CLOSE` (`<turn|>`): `106`
   - `PAD`: `0`
   - `MASK`: `4`
3. **Answer Prefix**:
   - The model turn begins with `Answer: ` where the token immediately following this prefix corresponds to answer slot $p$.

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

The request supplies a one-token output code and a separate display label for each candidate. Resolve the output code at the actual answer boundary of the fully rendered prompt. Encoding a trimmed candidate string in isolation is insufficient because whitespace and preceding text can change tokenization.

For each candidate:

1. Render the model's exact prompt and answer prefix.
2. Resolve the candidate continuation using the selected tokenizer's continuation semantics.
3. Reject it with `MultiTokenCandidateNotAllowed` unless that continuation is exactly one token.
4. Reject duplicate token IDs with `DuplicateCandidateToken`, even when display labels differ.
5. Keep the resolved token ID associated with its display label.

The exact continuation procedure must be tested against the model tokenizer and a reference fixture before implementation.

### Error Conditions
- **`MultiTokenCandidateNotAllowed`**: Returned if a rendered candidate continuation is not exactly one token.
- **`DuplicateCandidateToken`**: Returned when two candidates resolve to the same token ID.
- **`CanvasOverflow`**: Returned if serialized prompt tokens and required answer positions exceed the configured canvas width.

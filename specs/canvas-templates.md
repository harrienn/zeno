# Zeno — Specification: Canvas Templates & Seeding

## 1. Purpose & Scope

This document records the resolved canvas layout for the configured 256-token width. Exact prompt serialization remains open under U-05; the current text format is provisional.

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

## 3. Seeded Canvas Memory Layout

The following diagram shows the resolved canvas geometry. The exact user/model prompt text remains provisional under U-05:

```text
┌─────────┬──────────────────────┬──────────────────────┬─────────────┬─────────────┐
│ BOS?    │    Prompt Context    │   Response Template  │ Answer seed │ TURN_CLOSE │ PAD tail │
│template │  user query & state  │    "Answer:" label   │ token + pin │    token   │          │
└─────────┴──────────────────────┴──────────────────────┴─────────────┴────────────┴──────────┘
0         1                      k                      p             p+1          p+2        255
```

### 3.1 Region Definitions

1. **Header**: The reference tokenizer/template determines BOS insertion. Do not insert BOS separately if the tokenizer already adds it.
2. **Prompt Context**: Positions `1` through $k-1$ contain the tokenized user turn:
   - System/turn delimiters (`<|turn>user\n...<turn|>\n`).
   - Background context (if provided in the decision request).
   - Query question and candidate options listing.
3. **Response Template**: Positions $k$ through $p-1$ contain the beginning of the model turn:
   - Model turn delimiter (`<|turn>model\n`).
   - Prefix cue (e.g. `Answer:`).
4. **Answer Position**: Position $p$ is seeded with the token specified by the chosen structured-read procedure and selected for logit readout. Pin it if the selected step policy requires preserving its seed value.
5. **Remaining Canvas**: Position $p+1$ contains `TURN_CLOSE` (106); later positions contain `PAD` (0). The evaluated prefix is rounded to a 16-token multiple. This is the reference-derived Phase 3 layout (U-02/U-03).

PAD positions included in the evaluated prefix remain token embeddings and can affect the answer through bidirectional attention. Do not truncate or mask them unless a separate reference contract establishes that behavior.

---

## 4. Prompt Template Formatting

The exact artifact template has not been captured in a reviewable specification (U-05). The following prompt is the current Zeno prototype format, not a normative reproduction of the artifact template.

For standard decision reads:

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

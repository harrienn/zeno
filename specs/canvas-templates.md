# Zeno — Specification: Canvas Templates & Seeding

## 1. Purpose & Scope

This document records a proposed layout for the model's configured canvas width (256 tokens in the inspected GGUF artifact). The exact prompt serialization and padding behavior remain subject to comparison with the selected reference implementation.

In structured-reads mode, a seeded canvas contains prompt tokens and answer positions whose logits are read. In [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250), pinning holds a selected position at its seeded token value across denoising steps. Pinning is an update rule; it does not by itself mean that the seed token must be `MASK` (`4`).

> **Known circularity.** This document and [`structured-reads.md`](structured-reads.md) §2 both defer the answer-position seed value to *"the structured-read procedure"*, but neither document *is* that procedure. The value of $x_p$ — the single most consequential byte in the system — is therefore undefined. This is tracked as **U-01** in [`UNRESOLVED.md`](UNRESOLVED.md), and it blocks Phase 3. Note that the correct answer may well be `MASK`; what is missing is the evidence that distinguishes *"seeded with MASK"* from *"pinned to its seed across steps"*, which are different properties.

---

## 2. Canvas Geometry & Special Tokens

The canvas is a fixed-width array of token identifiers:

* **Default Canvas Length ($L$)**: `256` tokens (matching `diffusion.canvas_length` in [`specs/gguf-mapping.md`](gguf-mapping.md)). Whether a shorter request canvas is legal — and if so, what the model attends over beyond the answer slot — is unresolved (**U-03** in [`UNRESOLVED.md`](UNRESOLVED.md)) and is coupled to U-02 above. Do not implement a truncated canvas until both are resolved.
* **Token Identifier Type**: 32-bit unsigned integer (`u32`).
* **Special Token IDs**:
  - `BOS` (Beginning of Sequence): `2`
  - `EOS` (End of Sequence): `1`
  - `MASK` (Discrete Diffusion Mask Token): `4`
  - `PAD` (Padding Token): `0`

---

## 3. Seeded Canvas Memory Layout

The following diagram is a template proposal, not a verified requirement for model inputs:

```text
┌─────────┬──────────────────────┬──────────────────────┬─────────────┬─────────────┐
│ BOS?    │    Prompt Context    │   Response Template  │ Answer seed │ Remaining   │
│template │  user query & state  │    "Answer:" label   │ token + pin │   canvas    │
└─────────┴──────────────────────┴──────────────────────┴─────────────┴─────────────┘
0         1                      k                      p             p+1         255
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
5. **Remaining Canvas**: Seed contents and attention treatment are not specified here (**U-02**). Establish them from the reference path and a golden fixture ([`UNRESOLVED.md`](UNRESOLVED.md) §5).

> **Open requirement (U-02)**: A `PAD` token is still a token embedding. Define whether unused positions are masked from attention, contain padding IDs, or contain diffusion noise. Do not assume that filling them with `0` prevents them from influencing bidirectional attention.
>
> This is **not** a low-risk default. Global full attention occurs at blocks {5, 11, 17, 23, 29} ([`transformer.md`](transformer.md) §3.1), so a canvas whose trailing ~250 positions are filled with `PAD` is *not* equivalent to a shorter canvas. The two produce different hidden states at position $p$. Which one the reference performs determines the entire numeric result of a structured read, so this must be resolved by evidence rather than preference.

---

## 4. Prompt Template Formatting

The normative prompt structure is governed by `tokenizer.chat_template` extracted from the model artifact metadata (resolving **U-05**).

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

### 4.0 Chat Template Rules (Normative)
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

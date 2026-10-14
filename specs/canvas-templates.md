# Zeno — Specification: Canvas Templates & Seeding

## 1. Purpose & Scope

This document records Zeno's direct structured-decision canvas contract and distinguishes it from vLLM's structured-read sampler. The model artifact's default canvas width is 256 tokens; Zeno evaluates a configurable prefix of that canvas, with `canvas_length` in `[16, 256]` and divisible by 16. Prompt serialization is governed by §3.1 below.

In structured-reads mode, a seeded canvas contains prompt tokens and answer positions whose logits are read. In [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250), pinning holds a selected position at its seeded token value across denoising steps. Pinning is an update rule; it does not by itself mean that the seed token must be `MASK` (`4`).

**Resolved Zeno behavior:** a read seeds canvas position 0 with `MASK` (4), places `TURN_CLOSE` (106) at position 1, and fills the remaining backing canvas with `PAD` (0). The default evaluated width is 256; callers may select a shorter evaluated prefix in 16-token increments. This width override is a Zeno execution option. For a reference comparison, record and use the same canvas width and token arrays on both sides. See U-01 through U-03 in [`UNRESOLVED.md`](UNRESOLVED.md).

---

## 2. Canvas Geometry & Special Tokens

The canvas is a fixed-width array of token identifiers:

* **Maximum / Default Canvas Length ($C_{\max}$)**: `256` tokens (matching `diffusion.canvas_length` in [`gguf-mapping.md`](gguf-mapping.md)). The evaluated width $C$ defaults to 256 and may be set from 16 through 256 in multiples of 16. The total evaluated sequence length is $N=P+C$.
* **Token Identifier Type**: 32-bit unsigned integer (`u32`).
* **Special Token IDs**:
  - `BOS` (Beginning of Sequence): `2`
  - `EOS` (End of Sequence): `1`
  - `MASK` (Discrete Diffusion Mask Token): `4`
  - `PAD` (Padding Token): `0`

---

## 3. Dual-Region Memory Layout & Canvas Geometry

DiffusionGemma operates on a **dual-region sequence** $[0 \dots P + C - 1]$:
- **Prompt Region** ($0 \dots P-1$): Causal prefix containing the tokenized prompt and model prefix cue (`Answer: `).
- **Canvas Region** ($P \dots P + C - 1$): Evaluated canvas prefix of width $C$ under the decoder's bidirectional attention path. The GGUF default/maximum is 256; Zeno's API can request a shorter width.

```text
┌───────────────────────────────────────────────┬───────────────────────────────────────────────┐
│              Prompt Region (0..P-1)           │             Canvas Region (P..P+C-1)          │
├─────────┬──────────────────────┬──────────────┼─────────────┬─────────────┬───────────────────┤
│ BOS (2) │ User Turn Context    │ Model Prefix │ Answer Slot │ TURN_CLOSE  │ PAD Tail          │
│         │ Question & Choices   │ "Answer: "   │ MASK (4)    │ (106)       │ (0)               │
└─────────┴──────────────────────┴──────────────┴─────────────┴─────────────┴───────────────────┘
0         1                      P-1            P (slot 0)    P+1           P+2           P+C-1
```

### 3.1 Contract Comparison: Zeno vs vLLM `structured_server.py`

In [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) (`examples/features/structured_diffusion/structured_server.py`), the assistant turn template is constructed directly inside the canvas buffer (e.g. `<|turn>model\nAnswer: ` lives on the canvas, and the answer slot $s[\text{"pos"}]$ is located at an offset within that canvas template). Answer slots are seeded with random noise draws across repeated sampling passes.

In contrast, **Zeno's Direct Canvas Decision Contract**:
1. Formats the full user context and assistant prefix (`<|turn>user\n...<turn|>\n<|turn>model\nAnswer: `) into the causal **prompt region** ($0 \dots P-1$).
2. Designates **canvas index 0** ($P+0$) as the answer/read position and seeds it with `MASK` (`4`). Zeno reads its logits without applying the reference sampler's iterative pin/accept/re-noise updates.
3. Appends `TURN_CLOSE` (`106`) at canvas index 1 ($P+1$), and fills the remaining backing canvas with `PAD` (`0`). The evaluated prefix has width $C$; only positions $0 \dots C-1$ participate in that request.
4. Reads candidate logits directly from canvas index 0 in one logical read, without iterative sampling draws. CUDA may use prompt KV prefix caching internally; that cache is an execution optimization and does not change the input-token contract.

This decoupled layout places the decision answer at canvas index 0 instead of embedding the assistant template inside the canvas as the vLLM example does. When asserting parity against an external reference (`llama-diffusion-gemma-eval` or `structured_server.py`), supply the exact same prompt tokens, canvas tokens, answer position, and evaluated canvas width $C$ to both sides. A matching candidate decision with a different layout is not a token-level parity comparison.

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

---

## 6. Empirical Evaluation of Shortened Canvas Prefixes

### 6.1 Motivation & Mechanics

In DiffusionGemma's dual-region architecture, prompt tokens ($0 \dots P-1$) attend causally while canvas tokens ($P \dots P+C-1$) use the decoder's bidirectional attention path. At the default 256-token width, positions after answer and `TURN_CLOSE` are filled with `PAD` (0). These positions are still evaluated through the transformer and MoE; shorter widths reduce that work but also change the evaluated sequence and can change logits.

In [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) (`examples/features/structured_diffusion/structured_server.py:399`), single-step requests are dynamically rounded to the smallest multiple of `CANVAS_STEP = 16`:
$$\text{canvas\_width} = \min(C_{\text{len}}, \lceil (|\text{template}| + 1) / 16 \rceil \times 16)$$
For a single answer slot and turn close ($|\text{template}| + 1 = 2$), vLLM executes with $C = 16$.

### 6.2 Empirical Benchmark Results (ReleaseFast CPU)

> [!NOTE]
> **Preliminary Exploratory Scope (R21-04)**: The latency numbers and discrete decision agreement recorded below are preliminary, exploratory single-sample observations on CPU (ReleaseFast) evaluated against the local CPU $C=256$ baseline run. They are not warm-up averaged distributions or comparisons against an independent oracle runtime. The default remains $C=256$.

The table below records empirical latency (ms) and discrete decision agreement across all 5 golden fixtures for $C \in \{16, 32, 64, 128, 256\}$ on CPU:

| Fixture | Prompt $P$ | Baseline $C=256$ Latency | $C=16$ Latency | $C=16$ Speedup | $C=16$ Agreement | Intermediate $C \in \{32..128\}$ Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `system_health_boolean` | 31 | 17,137 ms | 2,675 ms | **6.41x** | **YES** ("Yes", $P=0.85$) | Flips to "No" at $C=32$, drops to 52% at $C=64$ |
| `incident_severity_multiclass` | 49 | 22,752 ms | 4,532 ms | **5.02x** | **YES** ("Medium", $P=0.44$) | Stable "Medium" across all $C \in [16, 256]$ |
| `satisfaction_score_5class` | 86 | 24,151 ms | 7,580 ms | **3.19x** | **YES** ("5", $P=0.95$) | Stable "5" across all $C \in [16, 256]$ |
| `maintenance_priority_uncertain`| 72 | 21,197 ms | 6,575 ms | **3.22x** | **YES** ("Low", $P=0.70$) | Flips to "Medium" at $C \in \{32, 64, 128\}$ |
| `boundary_256_incident_triage` | 224 | 29,603 ms | 15,252 ms | **1.94x** | **YES** ("No", $P=1.00$) | Stable "No" across all $C \in [16, 256]$ |

### 6.3 Invariants & Operational Guidance

1. **Decision Parity at $C=16$ on Fixture Suite**: Across the 5 golden fixtures evaluated, $C=16$ produces **100% discrete decision agreement** with the local $C=256$ CPU run while achieving a **$3.2\times$ to $6.4\times$ latency reduction** on CPU. Broader decision and numerical stability across larger evaluation datasets remain unestablished.
2. **Intermediate Instability**: Canvas widths $C \in \{32, 64, 128\}$ exhibit representation drift due to intermediate ratios of unmasked `PAD` tokens in bidirectional attention. Intermediate canvas widths should not be used arbitrarily.
3. **API Contract**: The default remains $C = 256$. Applications may request `canvas_length` from 16 through 256 in multiples of 16 via `DecisionRequest`, `options.canvas_length` in POST `/v1/systemone`, or `--canvas` via CLI. Fixtures must record their width; the existing $C=256$ fixtures do not establish parity at other widths.

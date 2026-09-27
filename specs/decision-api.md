# Zeno — Specification: System One Decision API

## 1. Purpose & Scope

This document specifies the interface for Zeno's **System One** decision engine.

Inspired by TypeSafe AI's Jev and aligned with the prototype `/v1/systemone` interposer in [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250), Zeno executes fast, deterministic, schema-constrained software decisions without free-form conversational text generation.

The specification defines:
1. The HTTP REST API contract (`POST /v1/systemone`).
2. The in-process native Zig struct interface.
3. Schema types and candidate token constraints.
4. Response payloads, candidate score normalization, and entropy reporting.

---

## 2. Invariant: Single-Token Choice Constraint

As established in [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) and [`README.md`](../README.md), a single-step structured read evaluates logits at a single pinned answer position on the canvas.

- **Constraint**: Every candidate option provided in a query must resolve to **exactly one token** in the model vocabulary (e.g. `"Yes"`, `"No"`, `"True"`, `"False"`, `"1"`, `"A"`, `"B"`).
- **Multi-token mapping**: Queries with multi-word labels (e.g. `"Credit Card Fraud"`) must map choices to one-token codes (e.g. `"A": "Credit Card Fraud"`, `"B": "Identity Theft"`) within the template prompt.

The decision engine strictly validates that all candidate choices resolve to single token IDs before executing a forward pass. If a choice resolves to multiple tokens, the request fails with a schema validation error.

---

## 3. HTTP Protocol: `POST /v1/systemone`

### 3.1 Request Payload

```json
{
  "query": "Does this support ticket indicate an outage?",
  "context": "Customer report: All database connections timed out starting at 14:02 UTC.",
  "schema": {
    "type": "boolean",
    "candidates": ["Yes", "No"]
  },
  "options": {
    "step_cap": 1,
    "temperature": 1.0,
    "entropy_threshold": 0.5
  }
}
```

#### Fields

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `query` | `string` | **Yes** | The decision prompt or question to resolve. |
| `context` | `string` | No | Unstructured application state or background text. |
| `schema.type` | `string` | **Yes** | Decision schema: `"boolean"`, `"choice"`, or `"score"`. |
| `schema.candidates` | `string[]` | **Yes** | List of valid choices. Each must resolve to 1 token ($2 \le N \le 32$). |
| `options.step_cap` | `integer` | No | Max diffusion steps (default: `1`). |
| `options.temperature` | `float` | No | Logit scaling factor before softmax (default: `1.0`). |
| `options.entropy_threshold` | `float` | No | Shannon entropy threshold for flagging uncertainty. |

### 3.2 Response Payload

```json
{
  "decision": "Yes",
  "score": 0.962,
  "entropy": 0.162,
  "uncertain": false,
  "candidates": [
    {
      "label": "Yes",
      "token_id": 9482,
      "score": 0.962,
      "raw_logit": 14.82
    },
    {
      "label": "No",
      "token_id": 3521,
      "score": 0.038,
      "raw_logit": 11.58
    }
  ],
  "steps_executed": 1,
  "canvas_tokens_used": 68
}
```

#### Response Semantics

* **`decision`**: The candidate label with the highest score ($p_{\text{max}}$).
* **`score`**: Normalized softmax score across the allowed candidate set:
  $$p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^N \exp(z_j / T)}$$
  *(Note: This is a normalized candidate score; empirical calibration requires separate labeled evaluation).*
* **`entropy`**: Normalized Shannon entropy over the candidate set:
  $$H(P) = -\sum_{i=1}^N p_i \ln(p_i)$$
* **`uncertain`**: Boolean flag set to `true` if $H(P) \ge \text{entropy\_threshold}$.

---

## 4. Supported Schema Types

### 4.1 Boolean Judgment (`"boolean"`)
Used for guards, gates, assertions, and smart conditionals.
* Default candidates: `["Yes", "No"]` or `["True", "False"]`.

### 4.2 Single-Token Multiple Choice (`"choice"`)
Used for intent routing, ticket classification, and moderation.
* Candidates: Single-word labels (e.g. `["Spam", "Ham"]`, `["Bug", "Feature", "Question"]`) or single-letter codes (e.g. `["A", "B", "C", "D"]`).

### 4.3 Ordinal Scoring (`"score"`)
Used for LLM-as-a-judge, response grading, and severity ranking.
* Candidates: Numeric single-digit tokens (e.g. `["1", "2", "3", "4", "5"]`).

---

## 5. Native Zig Library Interface

For embedded, in-process execution without HTTP overhead, Zeno exposes a typed Zig interface:

```zig
pub const SchemaType = enum {
    boolean,
    choice,
    score,
};

pub const DecisionRequest = struct {
    query: []const u8,
    context: ?[]const u8 = null,
    schema_type: SchemaType,
    candidates: []const []const u8,
    step_cap: u32 = 1,
    temperature: f32 = 1.0,
    entropy_threshold: ?f32 = null,
};

pub const CandidateResult = struct {
    label: []const u8,
    token_id: u32,
    score: f32,
    raw_logit: f32,
};

pub const DecisionResponse = struct {
    decision: []const u8,
    score: f32,
    entropy: f32,
    uncertain: bool,
    candidates: []const CandidateResult,
    steps_executed: u32,
    canvas_tokens_used: u32,
};

pub const DecisionEngine = struct {
    pub fn decide(
        self: *DecisionEngine,
        arena: std.mem.Allocator,
        request: DecisionRequest,
    ) !DecisionResponse;
};
```

### Memory Contract

- **Zero Global State**: All scratch tensors, canvas buffers, and response objects are allocated from the supplied `arena` allocator.
- **Immediate Reclaim**: At the completion of `decide()`, resetting or freeing `arena` reclaims all temporary activation memory in $O(1)$.

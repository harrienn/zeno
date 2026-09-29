# Zeno — Specification: System One Decision API

## 1. Purpose & Scope

This document proposes a Zeno decision API inspired by TypeSafe AI's Jev and the prototype `/v1/systemone` interposer in [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250). The JSON shape below is a Zeno proposal; it is not wire-compatible with the vLLM interposer. Runtime speed and determinism remain to be measured.

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
    "candidates": [
      {"token": "Yes", "label": "Yes"},
      {"token": "No", "label": "No"}
    ]
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
| `schema.candidates` | object[] | **Yes** | Each item has a `token` string that resolves to one token and a user-facing `label`. Require 2–32 candidates with distinct token IDs. |
| `options.step_cap` | `integer` | No | Version 0 supports only `1`; reject other values until multi-step semantics are specified and validated. |
| `options.temperature` | `float` | No | Positive, finite logit scaling factor before candidate normalization (default: `1.0`). It does not control diffusion sampling in a one-step read. |
| `options.entropy_threshold` | `float` | No | Optional threshold on normalized entropy, in `[0, 1]`. If omitted, `uncertain` is `false`. |

### 3.2 Response Payload

```json
{
  "decision": "Yes",
  "decision_probability": 0.962,
  "entropy_nats": 0.162,
  "normalized_entropy": 0.234,
  "uncertain": false,
  "candidates": [
    {
      "label": "Yes",
      "token_id": 9482,
      "probability": 0.962,
      "raw_logit": 14.82
    },
    {
      "label": "No",
      "token_id": 3521,
      "probability": 0.038,
      "raw_logit": 11.58
    }
  ],
  "steps_executed": 1,
  "canvas_tokens_used": 68
}
```

#### Response Semantics

* **`decision`**: The candidate label with the highest normalized candidate score ($p_{\text{max}}$).
  **Tie rule (normative)**: when two or more candidates share the maximum score exactly, the **lowest candidate index** in request order wins. This guarantees determinism for a System One engine and must not be left to floating-point comparison order. Note that ties are *reachable* in practice — logit softcapping (§3.2 of [`structured-reads.md`](structured-reads.md)) saturates distinct raw logits to an identical value, which is a numerical degenerate case rather than a genuine model judgement. See [`UNRESOLVED.md`](UNRESOLVED.md) U-19.
* **`decision_probability`**: Candidate-normalized softmax score for the selected label. It is not an empirically calibrated probability:
  $$p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^N \exp(z_j / T)}$$
  *(Empirical calibration requires separate labeled evaluation.)*
* **`entropy_nats`**: Shannon entropy over the candidate distribution, in nats:
  $$H(P) = -\sum_{i=1}^N p_i \ln(p_i)$$
* **`normalized_entropy`**: $H(P) / \ln(N)$, in $[0,1]$ for $N \ge 2$.
* **`uncertain`**: Set to `true` **only when `options.entropy_threshold` was supplied** and `normalized_entropy >= entropy_threshold`. If the threshold is omitted, `uncertain` is always `false`. Entropy is a dispersion measure, not a correctness estimate. *An earlier revision of this document stated "supplied or defaulted", contradicting the field table in §3.1. The field table is normative, and this reading is corroborated by [`denoising.md`](denoising.md) §2.1. See [`UNRESOLVED.md`](UNRESOLVED.md) U-21.*
* **`canvas_tokens_used`**: The number of canvas positions carrying prompt or answer content, i.e. `answer_pos + 1`. Trailing canvas fill is excluded, because whether trailing positions hold `PAD`, `MASK`, or noise is an open question ([`UNRESOLVED.md`](UNRESOLVED.md) U-02). This field therefore measures *seeded content length*, not canvas occupancy.

---

## 4. Supported Schema Types

### 4.1 Boolean Judgment (`"boolean"`)
Used for guards, gates, assertions, and smart conditionals.
* Default candidates: `["Yes", "No"]` or `["True", "False"]`.

### 4.2 Single-Token Multiple Choice (`"choice"`)
Used for intent routing, ticket classification, and moderation.
* Candidates: Each candidate contains a one-token `token` code and a display `label`. For example, `{"token":"A", "label":"Credit Card Fraud"}`.

### 4.3 Ordinal Scoring (`"score"`)
Used for LLM-as-a-judge, response grading, and severity ranking.
* Candidates: Ordered candidate objects, usually with numeric one-token codes. The `decision_probability` remains the normalized probability of the selected code; it is not the ordinal score itself.

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
    candidates: []const CandidateSpec,
    step_cap: u32 = 1,
    temperature: f32 = 1.0,
    entropy_threshold: ?f32 = null,
};

pub const CandidateSpec = struct {
    token: []const u8,
    label: []const u8,
};

pub const CandidateResult = struct {
    label: []const u8,
    token_id: u32,
    probability: f32,
    raw_logit: f32,
};

pub const DecisionResponse = struct {
    decision: []const u8,
    decision_probability: f32,
    entropy_nats: f32,
    normalized_entropy: f32,
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

- The engine owns long-lived model weights, tokenizer data, and backend state; these do not use the request allocator.
- Per-request scratch and response slices use the caller-supplied allocator. The response remains valid until the caller resets or frees that allocator.
- The API must document whether an engine may be called concurrently and how callers initialize/load and release it before those operations are implemented.

### Initial Error Contract

Reject malformed schemas, fewer than two or more than 32 candidates, duplicate resolved token IDs, candidates that do not resolve to one token in their rendered answer context, non-finite or non-positive temperatures, canvas overflow, and `step_cap` values other than `1`. HTTP status codes and a stable JSON error envelope must be specified before exposing the endpoint.

A candidate string that is absent from the vocabulary must **not** be reported as `MultiTokenCandidateNotAllowed`; the two conditions have different remediations (choose a different label versus map the label to a one-token code). See [`UNRESOLVED.md`](UNRESOLVED.md).

## 7. Known Gaps in This Specification

This contract is not yet implementable end-to-end. The following dependencies are tracked in [`UNRESOLVED.md`](UNRESOLVED.md):

* **U-04** — no `encode()` exists in any phase of [`PLAN.md`](../PLAN.md); the query text cannot yet be tokenized.
* **U-05** — the artifact's chat template is not captured, so §3 rendering has no normative source.
* **U-01 / U-02 / U-03** — canvas seeding and tail contents are undefined, so `canvas_tokens_used` and `steps_executed` semantics rest on an unresolved basis.

The request/response shape, the single-token constraint (§2), and the scoring fields are specified and may be implemented. The **engine that produces the `expected` values is not**.

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
* **`canvas_tokens_used`**: The number of prompt and answer positions, i.e. `answer_pos + 1`. It excludes the `TURN_CLOSE` and padded tail positions specified by U-02 in [`UNRESOLVED.md`](UNRESOLVED.md).

### 3.3 HTTP Error Envelope and Status Codes

All non-200 responses return a standard RFC 8259-compliant JSON error object:

```json
{
  "error": {
    "code": "canvas_overflow",
    "message": "Prompt exceeds configured canvas length"
  }
}
```

#### Implemented Status Codes

| HTTP Status | Error Code | Trigger Condition |
| :--- | :--- | :--- |
| `200 OK` | *(none)* | Decision executed successfully, or liveness probe passed (`GET`/`HEAD /health`). |
| `400 Bad Request` | `bad_request` | Malformed HTTP request line, invalid path, unparseable Content-Length, or duplicate Content-Length header. |
| `400 Bad Request` | `invalid_json` | Unparseable JSON body, or missing required schema fields (`query`, `schema`, `schema.type`, `schema.candidates`, `token`, `label`). |
| `400 Bad Request` | `invalid_schema_type` | Schema type is not `"boolean"`, `"choice"`, or `"score"`. |
| `400 Bad Request` | `empty_query` | Query string is empty (`""`). |
| `400 Bad Request` | `too_few_candidates` | Fewer than 2 candidates provided. |
| `400 Bad Request` | `too_many_candidates` | More than 32 candidates provided. |
| `400 Bad Request` | `duplicate_candidate_token` | Multiple candidates resolve to identical token IDs. |
| `400 Bad Request` | `duplicate_candidate_label` | Multiple candidates share identical display labels. |
| `400 Bad Request` | `token_not_found` | Candidate token text is not found in tokenizer vocabulary. |
| `400 Bad Request` | `multi_token_candidate` | Candidate resolves to more than 1 vocabulary token. |
| `400 Bad Request` | `unsupported_step_cap` | `options.step_cap` is set to any value other than 1. |
| `400 Bad Request` | `invalid_temperature` | Temperature is non-finite, $\le 0.0$, or NaN. |
| `400 Bad Request` | `invalid_entropy_threshold` | Threshold is non-finite, $< 0.0$, or $> 1.0$. |
| `400 Bad Request` | `canvas_overflow` | Rendered prompt exceeds maximum allowed prompt bytes (16 KiB) or prompt tokens (768 tokens, capping total sequence to 1024). |
| `400 Bad Request` | `read_error` / `incomplete_body` | Socket read failure or client disconnected prematurely during body transmission. |
| `404 Not Found` | `not_found` | Request path is not `/health` or `/v1/systemone`. |
| `405 Method Not Allowed` | `method_not_allowed` | Method other than GET or HEAD for `/health`, or other than POST for `/v1/systemone`. |
| `408 Request Timeout` | `request_timeout` | Total request read time exceeded the 30-second deadline. |
| `411 Length Required` | `length_required` | POST request omitted `Content-Length` header. |
| `413 Payload Too Large` | `payload_too_large` | Request body exceeds maximum allowed buffer size (64 KiB). |
| `500 Internal Server Error` | `internal_error` | Internal engine or inference failure. |
| `501 Not Implemented` | `not_implemented` | Request specifies a `Transfer-Encoding` header (all transfer codings, including `chunked`, are unsupported). |
| `505 Version Not Supported` | `http_version_not_supported` | Request line specifies an HTTP version other than `HTTP/1.0` or `HTTP/1.1`. |

### 3.4 Wire Format, Escaping & Numerical Precision

- **RFC 8259 String Escaping**: All JSON strings (including user-supplied `decision` labels, candidate `label` strings, and error messages) are strictly escaped (`\"`, `\\`, control codes `\n`, `\r`, `\t`, `\u00XX`), preventing JSON structure injection.
- **Float Rounding in JSON**: API response floating-point fields (`decision_probability`, `probability`, `raw_logit`, `entropy_nats`, `normalized_entropy`) are serialized using 6 decimal places of precision (`{d:.6}`). Downstream automated tools comparing golden fixtures against IEEE 754 float baselines must use the native Zig interface or account for decimal formatting tolerance.

### 3.5 Framing, Limits & Security Boundaries

- **Framing Invariants**: HTTP request lines must contain exactly three tokens (`METHOD PATH HTTP/VERSION`). Paths must start with `/`. Repeated or conflicting `Content-Length` headers are rejected with HTTP 400 Bad Request to prevent HTTP request smuggling. Both `GET` and `HEAD` methods are accepted on `/health` (with `HEAD` omitting the response body per RFC 7231).
- **Socket Idle Timeouts & Request Deadlines**: Accepted connections enforce a 15-second socket idle timeout and a strict, un-overrunnable 30-second request read deadline implemented via `poll` capped to the remaining deadline. Read deadline expiration returns HTTP 408 Request Timeout (`request_timeout`), while idle client connections during header reading close silently. Rejected requests (such as 400, 404, 405, 408, 411, 413, 501, 505) immediately close the connection without draining unread client bodies.
- **Sequence & Workspace Bounding**: Request payloads are capped at 64 KiB (`payload_too_large`). Query, context, and candidate labels are preflight-bounded at 16 KiB total prompt bytes (`max_prompt_bytes`) before template formatting, and tokenized prompts are bounded at 768 tokens (`max_prompt_tokens`), preventing both tokenization/allocation CPU DOS and activation tensor reallocation beyond 1024 tokens (`max_seq_len`).
- **Serial Execution & Trust Boundary**: The server is a single-threaded synchronous local decision engine without authentication or transport encryption (TLS). In-flight model evaluations on `/v1/systemone` execute synchronously on CPU; concurrent incoming requests (including `/health`) are queued in the OS TCP socket listen backlog until the current forward pass completes. It is intended for loopback operation (`127.0.0.1`). If exposed to external network interfaces (`--host 0.0.0.0`), it must be placed behind an authenticated reverse proxy (e.g. Envoy, Nginx, Caddy).

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

### Memory and Concurrency Contract

- The engine owns long-lived model weights, tokenizer data, and a pre-allocated `InferenceWorkspace` containing activation tensors and the persistent worker thread pool; these do not use the request allocator.
- **Concurrency Contract (R5-01)**: A single `DecisionEngine` instance owns one mutable workspace and executes requests sequentially. Access to `engine.decide(...)` is serialized via an internal mutex to guarantee race-free execution across concurrent caller threads. Applications requiring parallel evaluation across multiple queries simultaneously must allocate multiple `DecisionEngine` instances (accounting for ~49 MiB workspace overhead per instance) or route queries through an external queue.
- Per-request scratch and response slices use the caller-supplied allocator. The response remains valid until the caller resets or frees that allocator.

### Error Contract

The native interface returns typed errors:
- `error.EmptyQuery`: query string is empty (`""`).
- `error.TooFewCandidates`: fewer than 2 candidates provided.
- `error.TooManyCandidates`: more than 32 candidates provided.
- `error.DuplicateCandidateToken`: candidate options resolve to duplicate vocabulary token IDs.
- `error.DuplicateCandidateLabel`: candidate options share identical display labels.
- `error.UnsupportedStepCap`: `step_cap` is not 1.
- `error.InvalidTemperature`: temperature $\le 0.0$ or non-finite.
- `error.InvalidEntropyThreshold`: threshold $< 0.0$, $> 1.0$, or non-finite.
- `error.TokenNotFound`: candidate string absent from model vocabulary.
- `error.MultiTokenCandidateNotAllowed`: candidate resolves to multiple tokens.
- `error.CanvasOverflow`: prompt byte length exceeds `max_prompt_bytes` (16 KiB) or tokenized prompt exceeds `max_prompt_tokens` (768).

A candidate string that is absent from the vocabulary is reported as `TokenNotFound`, **not** `MultiTokenCandidateNotAllowed`, preserving distinct failure semantics (choose a different label versus map the label to a one-token code).

## 6. Known Gaps in This Specification

The request path is implemented as a CPU prototype with verified discrete decision matching against reference oracles. Remaining dependencies are tracked in [`UNRESOLVED.md`](UNRESOLVED.md):

* **U-05** — *Resolved*: The artifact's exact chat template is recorded verbatim in [`chat_template.jinja`](chat_template.jinja) and documented in [`canvas-templates.md`](canvas-templates.md) §4.
* **U-13 / U-22** — *Open*: Golden fixture captures and schema `zeno.fixture/v1` are established against llama.cpp PR #24423 (`llama-diffusion-gemma-eval`) with exact candidate decision matching and identical top-5 vocabulary predictions. However, candidate raw logit divergence between FP32 dequantized execution and GGML SIMD execution (1.19 to 11.07 across fixtures) exceeds the $\le 0.5$ tolerance limit in `zeno.tolerance/v1`. Full Phase 3 parity exit remains open until numerical logit divergence is reconciled.

The request/response shape, single-token constraint (§2), scoring fields, HTTP error envelope, and framing limits are specified. RoPE pairing is resolved to NeoX-style half-split under U-23 and implemented accordingly.

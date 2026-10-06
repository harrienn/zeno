# Review Round 6 — API Server and Whole-Project Readiness

**Reviewed revision:** `f0d1b47` (`feat(server): implement native zero-dependency HTTP decision API server`), including changes since `4cbca28`.

**Scope:** Current project state, with focus on the new HTTP server, its integration with the decision engine, API/spec consistency, and outstanding inference readiness gates. This was a static review; no build or tests were run.

## Verdict

The server adds a useful local API prototype and exposes the existing CPU decision path through a small, dependency-free interface. The API shape and internal decision call are understandable. The endpoint should not yet be treated as safe for untrusted or network-facing use: successful responses can be malformed by caller-controlled labels, a single client can monopolize the server indefinitely, and request size is not bounded at the token/workspace level.

The wider inference project remains a CPU prototype rather than a reference-validated runtime. The unresolved register still has U-13 and U-22 open because raw-logit differences exceed the documented tolerance, even though the checked fixtures report matching discrete decisions. The API can expose results, but should not imply reference parity or calibrated confidence.

## Findings

### R6-01 — Caller-controlled labels are interpolated into JSON without escaping

**Severity: High**
**Location:** `src/server.zig:368-416`

`decision` and candidate `label` are written into quoted JSON strings using `{s}`. Labels are supplied in the request, then copied into the response by the engine. A label containing a quote, backslash, or control character can make the response invalid JSON; crafted labels can also inject additional JSON fields into the response body.

**Recommendation:** Serialize the response with Zig's JSON encoder (including strings) instead of hand-building quoted values. Add a regression case with quotes, backslashes, and control characters.

### R6-02 — A single slow connection or inference blocks the entire server

**Severity: High**
**Location:** `src/server.zig:141-156`, `166-183`, `245-280`, `477-488`

The accept loop handles one client synchronously. Header and body reads have no deadlines, so a client can connect and send a partial header/body, pinning the only server thread indefinitely. A normal long-running decision also prevents `/health` and all other requests from being handled. Unknown paths, wrong methods, and oversized bodies call `drainRemaining` before responding, which can block on data the server does not need to consume; this is especially problematic for the 413 path.

**Recommendation:** Set explicit read deadlines and return errors without draining rejected request bodies when the connection will be closed. If concurrent request handling is added, bound the worker/queue count and keep engine workspace serialization intentional. Until then, document that the endpoint is single-request-at-a-time and local/trusted-use only.

### R6-03 — HTTP body cap does not bound inference workspace growth

**Severity: High**
**Location:** `src/server.zig:239-280`, `src/engine.zig:86-103`, `src/infer.zig:897-929`

The server accepts bodies up to 10 MiB, but there is no query/context/token-count limit before tokenization and inference. `Canvas.seedSingleStep` does not reject long prompts, and the engine passes prompt plus canvas length to `InferenceWorkspace.ensureCapacity`, which doubles and reallocates all sequence-sized buffers. A modest JSON body can therefore request a very large workspace and exhaust process memory; the declared `CanvasOverflow` error is not raised by the current canvas path.

**Recommendation:** Define and enforce a maximum rendered prompt/token count before workspace growth. Make the limit consistent with model semantics and the API contract, and either return the documented overflow error or remove that promise if prompts are intentionally unbounded.

### R6-04 — The API can be bound to a non-loopback address without an access boundary

**Severity: High when exposed beyond loopback**
**Location:** `src/main.zig:28-53`; `src/server.zig:95-120`

The default host is `127.0.0.1`, which is appropriate for a local prototype, but `--host` permits binding to any IPv4 address. There is no authentication or transport protection in this server. Exposing it to an untrusted network would allow arbitrary callers to consume inference and memory resources.

**Recommendation:** Clearly state the trust boundary in CLI help and API documentation. If remote use is intended, require an authenticated front end or implement an explicit access-control mechanism before recommending direct exposure.

### R6-05 — CLI port parsing silently hides invalid configuration and logs the wrong port for `--port 0`

**Severity: Medium**
**Location:** `src/main.zig:31-39`, `52-57`; `src/server.zig:123-129`

An invalid `--port` value silently resets the port to 8000; unknown options and options missing values are also silently ignored. The server supports port 0 and updates `server_inst.port` to the assigned port, but the startup message prints the original local `port` variable, so it reports port 0 rather than the actual listener.

**Recommendation:** Reject malformed or incomplete CLI options with a clear error. Print `server_inst.port` after `listen()`.

### R6-06 — HTTP framing validation is too permissive for a network-facing parser

**Severity: Medium**
**Location:** `src/server.zig:194-213`

The request line is split into method/path without validating the version or number of fields. Invalid `Content-Length` is treated as if it were absent, and repeated `Content-Length` headers overwrite one another. This produces ambiguous handling for malformed requests and is unsafe if the endpoint is ever placed behind a proxy with different framing rules.

**Recommendation:** Reject malformed request lines, invalid or duplicate `Content-Length`, and unsupported transfer encodings explicitly. Keep the parser contract narrow and document that chunked requests are unsupported if that is intentional.

### R6-07 — Current tests do not cover the server's highest-risk paths

**Severity: Medium**
**Location:** `tools/test_server.py`; `src/server.zig:541-619`

The integration script covers health, routing errors, two validation cases, and one happy-path decision. It does not exercise escaped labels, malformed/ambiguous HTTP framing, incomplete/oversized bodies, prompt limits, or whether health remains responsive during a stalled or long-running decision. The unit serializer fixture uses only plain ASCII labels, so it cannot expose R6-01.

**Recommendation:** Add focused server-level regression coverage for the findings above and make the live-decision assertion depend on an explicit fixture/reference contract rather than an unexplained expected label.

### R6-08 — API specification and README status statements are stale

**Severity: Medium**
**Location:** `specs/decision-api.md:179-192`; `README.md:80-87`

`decision-api.md` says HTTP statuses and a stable error envelope must be specified before exposing the endpoint, although the implementation now returns both. Its “Known Gaps” says U-05 is unrecorded and fixtures lack independent captures, while `specs/UNRESOLVED.md:33,46,52` records U-05 as resolved and describes oracle-captured fixtures; U-13/U-22 remain open for numerical tolerance/parity. `README.md` repeats the obsolete U-05 and fixture-provenance claims.

**Recommendation:** Update both documents to describe the implemented status/error envelope and reflect the current gate precisely: U-05 is resolved; U-13/U-22 are still open because measured numerical divergence exceeds the agreed tolerance.

## Non-blocking implementation notes

- The internal engine mutex correctly protects the shared mutable workspace, but it is a yield-loop spin lock (`src/engine.zig:21-33`). If the library is called concurrently, waiters consume CPU while the full inference runs under the lock. Prefer a blocking mutex if the Zig version/API permits.
- API response serialization rounds numeric fields to six decimal places. This may be adequate for display, but the wire contract should say whether values are rounded and fixture comparisons should use the engine values rather than expecting lossless float output.
- The server tests and CLI demonstrate basic wiring, but this review did not execute a build or test suite. No claim about compilation or runtime behavior is made here.

## Recommended order

1. Fix JSON string encoding and enforce a rendered-prompt/token limit.
2. Add read deadlines and ensure rejected requests do not block while draining bodies.
3. Make CLI configuration errors explicit and correct the port-0 startup message.
4. Tighten request framing and add regression tests for malformed input and concurrency/responsiveness.
5. Bring `README.md` and `specs/decision-api.md` into sync with the implementation and open parity gates.

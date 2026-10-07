# Review Round 7 — Follow-up to API Server Fixes

**Reviewed revision:** `440f93a` (`feat(server): address Codex review round 6 findings`), compared with `ac6710f`.

**Scope:** Follow-up review of Round 6 fixes and related API documentation, CLI usage, and server tests. This was a static review; no build or tests were run.

## Verdict

The patch makes meaningful progress: success and error strings are escaped, prompt token count is capped, rejected requests no longer drain bodies, CLI option errors are explicit, and request-line/transfer-encoding checks are tighter. The non-loopback warning and API trust-boundary text are useful.

I found one likely test/runtime allocator leak, a still-incomplete server starvation mitigation, and contract/documentation mismatches introduced while documenting the API. The CPU/reference parity status is stated more accurately than before: U-13/U-22 remain open.

## Findings

### R7-01 — Response serialization leaks its `allocPrint` temporaries

**Severity: High**
**Location:** `src/server.zig:437-455`, `461-475`, tests at `592-648`

`serializeDecisionResponse` allocates `header` and one `cand_str` per candidate using `std.fmt.allocPrint`, appends their contents to `list`, and never frees those temporary slices. The serializer tests pass `std.testing.allocator`, which checks for outstanding allocations; these calls are therefore expected to report leaks, and ordinary non-arena callers of this public function also leak memory on each request. The production HTTP path happens to pass an arena, which masks the ownership problem until arena teardown.

**Recommendation:** Avoid the temporary allocations by formatting directly into the output list, or free each temporary after appending it. Keep the returned JSON allocation owned by the caller and document that ownership.

### R7-02 — Documented error codes and native errors do not match implementation

**Severity: Medium**
**Location:** `specs/decision-api.md:115-139`, `230-244`; implementation at `src/server.zig:329-390`, `src/decision.zig:68-78`

The new error table does not describe the actual wire contract. Examples:

- An unsupported schema type returns `invalid_schema_type`, not `unsupported_schema_type`.
- Missing `schema` or candidate fields fail JSON struct parsing and return `invalid_json`, not `missing_schema` or `missing_candidate_token`.
- An empty query returns `empty_query`; it is not reported as `bad_request`.
- Duplicate labels return `duplicate_candidate_label`, which is absent from the table.
- HTTP 505 returns `http_version_not_supported`, not `unsupported_version`.
- `/health` accepts `HEAD`, while the table says only `GET` is accepted.
- The native error list names `InvalidSchema` and `DuplicateCandidate`, which are not the error names in `decision.zig`.

**Recommendation:** Derive the table and native error list from the actual branches/error sets, or change the implementation to match a deliberately chosen contract. Add a table-driven API test for each listed error so future edits cannot silently drift.

### R7-03 — README server command does not match the CLI

**Severity: Medium**
**Location:** `README.md:93-103`; CLI parser at `src/main.zig:20-29`

The README instructs users to run `zig build run -- --serve --port 8000`. The executable expects the subcommand `serve` followed by a model path, and it does not accept `--serve`. As written, this command cannot start the documented server.

**Recommendation:** Show the actual form, including a model path, for example `zig build run -- serve models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --port 8000`.

### R7-04 — Socket idle timeouts do not prevent the synchronous server from being monopolized

**Severity: Medium**
**Location:** `src/server.zig:140-160`, `173-187`, `285-320`; API spec at `specs/decision-api.md:146-151`

`SO_RCVTIMEO` limits how long an individual blocking `read` waits without progress; it is not a total request deadline. A client can continue sending small pieces before each 15-second idle timeout and keep the sole accept/dispatch thread occupied. Separately, one normal inference blocks `/health` and every other request for the full inference duration. The API spec discloses single-threaded operation, but its timeout wording can be read as a request deadline and the change does not provide health responsiveness during inference.

**Recommendation:** Describe these as idle socket timeouts, not general request deadlines. If health responsiveness or resistance to slow trickle clients is required, use an absolute request deadline and isolate request processing from the accept/health path with a bounded execution policy. Otherwise keep the explicit local-only, serial-use limitation prominent.

### R7-05 — Prompt limit is checked only after formatting and tokenization

**Severity: Medium**
**Location:** `src/engine.zig:86-100`; HTTP body cap at `src/server.zig:280-320`

The new 768-token bound prevents oversized prompts from growing the inference workspace, which closes the unbounded-workspace issue. However, the engine first formats and tokenizes the entire request, then checks the token count. The HTTP layer permits a 10 MiB body, so a request that is guaranteed to be rejected can still cause substantial temporary allocation and tokenizer work before `canvas_overflow` is returned.

**Recommendation:** Add an earlier byte-size guard for query/context/rendered prompt, or use a bounded tokenizer path that stops once the token limit is exceeded. Keep the token-count check as the semantic final guard.

### R7-06 — `HEAD /health` sends a response body

**Severity: Low**
**Location:** `src/server.zig:253-260`; documented methods at `specs/decision-api.md:133-135`

The handler accepts `HEAD` but routes it through `sendJson`, which sends the JSON body. HTTP `HEAD` responses should carry the headers a corresponding `GET` would produce without a response body. The documentation also omits this supported method.

**Recommendation:** Either remove `HEAD` support and document GET only, or return headers with the correct content length and no body for `HEAD`.

## Verification notes

- The added integration cases cover malformed request lines, duplicate content length, chunked transfer rejection, prompt overflow, and escaped labels. They do not test the 15-second timeout behavior or health responsiveness during a decision.
- The live API case still asserts `uncertain == false` for a model-generated distribution with a supplied entropy threshold. That assertion depends on a particular model output and may be brittle across numerical/backend changes; either pin it to a fixture or assert the threshold semantics independently.
- No build or tests were run for this review, so R7-01 is a static allocator-ownership finding rather than a reported test failure.

## Recommended order

1. Fix serializer temporary ownership and run the focused server tests.
2. Make the API error table match actual wire/native errors.
3. Correct the README command.
4. Clarify idle timeout guarantees and decide whether request/health isolation is in scope.
5. Bound input work before full formatting/tokenization and settle the `HEAD` behavior.

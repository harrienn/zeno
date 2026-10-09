# Review Round 9 — Follow-up to API Server Deadline and Prompt Bounds

**Reviewed revision:** `b2edab2` (`feat(server): address Codex review round 8 findings`), compared with `c2115f2`.

**Scope:** Follow-up review of the request deadline helper, prompt-byte preflight, related API documentation, and new tests. This was a static review; no build or tests were run.

## Verdict

The latest patch addresses both Round 8 findings in the normal path. Poll waits are capped to the remaining request deadline, the prompt-byte calculation now includes candidate labels before formatting, and the added tests cover these cases. The README/spec and API contract improvements from prior rounds remain in place.

I found one edge case in the strict deadline guarantee and one small discrepancy in how unsupported transfer encodings are documented.

## Findings

### R9-01 — Interrupted waits retry with the original timeout instead of the remaining deadline

**Severity: Medium**
**Location:** `src/server.zig:199-203`, `214-223`

`readChunkWithDeadline` calculates `wait_ms` once, then retries `poll` with that same timeout after `EINTR`. The inner `read` loop also retries after `EINTR` without rechecking the absolute deadline. Repeated signals can therefore extend the nominal 30-second deadline, contrary to the spec's “strict, un-overrunnable” wording. This is most relevant when the server is embedded in a host process that handles signals while requests are arriving.

**Recommendation:** Recompute the monotonic time and remaining deadline after every `EINTR`, before retrying either `poll` or `read`. Consider using one helper that returns deadline expiry whenever the absolute time has passed.

### R9-02 — Transfer-Encoding documentation is narrower than parser behavior

**Severity: Low**
**Location:** `src/server.zig:301-323`; `specs/decision-api.md:141`

The parser rejects any header whose name begins with `Transfer-Encoding:` and returns 501. The error message and API table describe only `Transfer-Encoding: chunked`. Other transfer codings are also rejected, so the wire contract is narrower than actual behavior.

**Recommendation:** State that all `Transfer-Encoding` request headers are unsupported, or explicitly parse and reject only the transfer codings the implementation cannot handle. Add a non-chunked transfer-encoding case if the broader rejection is the intended contract.

## Confirmed improvements

- `calculatePromptByteLength` includes template bytes, context, query, and candidate labels using checked arithmetic; the engine checks it before formatting.
- The serializer writes formatted fields directly into the output list, avoiding the temporary allocations identified in R7-01.
- The latest tests add prompt-label overflow, incomplete-body, HEAD, and wire-error cases. This review did not run those tests.
- The API specification explicitly documents the synchronous server and the remaining U-13/U-22 parity gates.

## Verification notes

`git diff --check` passed for the review report. No build or test suite was run; findings are based on source and documentation inspection.

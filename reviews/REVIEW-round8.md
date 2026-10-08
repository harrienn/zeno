# Review Round 8 — Follow-up to API Server Review

**Reviewed revision:** `4169e5a` (`feat(server): address Codex review round 7 findings`), compared with `f1e3e06`.

**Scope:** Follow-up review of the latest API server, engine, test, README, and API specification changes. This was a static review; no build or tests were run.

## Verdict

The changes address the concrete Round 7 findings well: response serialization now writes directly into its output list, the error-code/native-error documentation is much closer to the implementation, the README command matches the CLI, and `HEAD /health` omits its body. The test no longer assumes a specific model entropy result.

I found two remaining boundary issues: the 30-second deadline is not a strict deadline when a blocking read is already in progress, and the byte guard can still allocate a large formatted prompt before rejecting oversized candidate labels through the native API. The synchronous server limitation is now stated plainly in the API specification; I treat that as a documented prototype constraint rather than a new defect.

## Findings

### R8-01 — The absolute request-read deadline can overrun and does not always return 408

**Severity: Medium**
**Location:** `src/server.zig:170-190`, `314-330`; `specs/decision-api.md:152`

The code checks the monotonic clock before each blocking `read`, but the socket read timeout remains fixed at 15 seconds. If a read starts just before the 30-second deadline and receives no data, it can block for another 15 seconds. If that read times out, the header path closes the connection without a response, while the body path returns `400 read_error`; neither returns the documented `408 request_timeout`. The time comparison also uses whole seconds and `>` rather than an absolute timespec deadline.

**Recommendation:** Compute the remaining deadline before each read and cap the socket wait to that duration, or use `poll`/`ppoll` with an absolute deadline. Map deadline expiry to the documented 408 response; keep idle-timeout behavior distinct if it intentionally closes silently.

### R8-02 — Candidate labels are not included in the early prompt-byte guard

**Severity: Medium for native API callers**
**Location:** `src/engine.zig:86-105`

The pre-format guard checks only `query.len + context.len`. Candidate labels are inserted into the rendered prompt, but their lengths are considered only by the later `formatted_prompt.len` check, after `Canvas.formatDecisionPrompt` has already allocated and copied them. The HTTP body cap bounds this route, but an in-process caller can pass arbitrarily large labels and trigger a large allocation before the `max_prompt_bytes` error is returned.

**Recommendation:** Include every candidate label and the fixed template overhead in an overflow-safe preflight byte calculation before formatting. Keep the final rendered-length check as a defensive assertion/guard.

## Verification notes

- The serializer no longer creates the leaked `allocPrint` temporaries identified in R7-01.
- The API error table now covers the actual error names noted in R7-02, and the README invocation supplies the required `serve` subcommand and model path.
- The added `HEAD` case checks that the response has a content length and no body. The new test suite still does not exercise slow-trickle clients, idle timeout behavior, or the 30-second deadline path.
- No build or tests were run for this review; the conclusions above are from source and documentation inspection.

## Recommended order

1. Make the absolute deadline effective during blocking reads and return the documented timeout response.
2. Include candidate-label bytes in pre-format prompt-size validation.
3. Add focused tests for deadline behavior and oversized native candidate labels.

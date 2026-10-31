# Review Round 57 — CUDA Teardown Concurrency and Contract Documentation

**Reviewed revision:** `6c55275` (`fix(decision): remediate Review Round 56 decision contract, continuation resolution, and CUDA target admission`)

**Scope:** Read-only review of the current project, with emphasis on the Round 56 remediation, CUDA resource teardown, HTTP request concurrency, and consistency between the runtime and its documentation.

## Verdict

The Round 56 changes address the reviewed decision-contract issues in the implementation: distinct candidate codes are rendered with their labels, candidate IDs are resolved against the formatted prompt, fixture checks compare those IDs, and CUDA target parsing now distinguishes PTX from SASS-bearing targets.

One resource-safety issue remains in the CUDA failure teardown path: process-wide quarantine bookkeeping is mutated without synchronization. I also found two documentation statements that no longer describe the current code. These do not invalidate ordinary single-engine inference, but the teardown race weakens the safety guarantee precisely when CUDA synchronization has already failed.

## Findings

### R57-01 — Process-wide CUDA quarantine state is unsynchronized

**Severity: Medium — concurrent CUDA workspace teardown can race while recording or draining pinned buffers.**

The quarantine tables, retained-workspace pool, linked-list heads, and counts are mutable module-level globals (`src/infer_cuda.zig:1013-1016`). `quarantinePinnedBuffer` writes those structures (`:1026-1058`), `drainQuarantinedPinnedBuffers` traverses and edits the shared retained list (`:1251-1278`), and `CudaWorkspace.deinit` adds retained records (`:1471-1508`). These operations have no common lock.

One `DecisionEngine` serializes its own `decide` calls, but that mutex does not protect teardown or multiple engine/workspace instances. If two CUDA workspaces are deinitialized concurrently—especially after a synchronization failure—the shared list or pool can be overwritten or traversed inconsistently, potentially losing ownership tracking for a pinned allocation or attempting duplicate cleanup.

**Recommendation:** Protect all accesses to the process-wide quarantine state with one clearly owned mutex, or make the quarantine registry an explicitly shared owner passed to each CUDA workspace. Add a concurrent teardown test that exercises both insertion and draining; preserve the existing single-threaded error-path tests as well.

### R57-02 — API documentation describes the old single-threaded server

**Severity: Low — the documented behavior of `/health` during inference is incorrect.**

`specs/decision-api.md:182` says the server is single-threaded and that concurrent requests, including `/health`, wait in the listen backlog during an inference. The current server reserves a connection slot and spawns a worker thread for each accepted connection (`src/server.zig:175-205`). Decision execution is serialized by the engine mutex (`src/engine.zig:193-197`), while `/health` is handled by its connection worker without calling the engine. Thus decision calls queue at the engine lock, but health requests can be served during a long inference.

**Recommendation:** Update the spec to describe concurrent connection handling, serialized decision execution, and responsive health probes. Keep the documented 64-connection limit and overload response aligned with the server implementation.

### R57-03 — Architecture overview retains the old answer-prefix spacing

**Severity: Low — one architecture example contradicts the current prompt contract.**

`ARCHITECTURE.md:225` still shows `Answer: ` with a trailing space. The renderer and current canvas/tokenizer specification terminate the prompt at `Answer:` without that space (`src/canvas.zig:143`, `specs/canvas-templates.md:84-85`).

**Recommendation:** Remove the trailing space in the architecture example and, while editing that paragraph, clarify that the assistant prefix is in the causal prompt and the `MASK` answer position is in the canvas.

## Round 56 Remediation Status

- **R56-01, coded choices:** Addressed in `Canvas.formatDecisionPrompt`; code and label are both shown when they differ.
- **R56-02, continuation resolution and candidate-ID checks:** Addressed in the engine path and fixture/oracle validation code.
- **R56-03, CUDA code-image admission:** The parser and compatibility rules now distinguish SASS-bearing, PTX-only, and architecture-specific targets. Runtime execution on additional GPU generations still requires hardware validation.

## Verification Limits

Inspected revision `6c55275`, the related engine, canvas, tokenizer, fixture, CUDA, server, and specification code, and the latest review report. No builds, tests, oracle comparisons, or GPU runs were performed for this review. The worktree was clean before this report was added.

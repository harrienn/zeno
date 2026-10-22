# Review Round 52 — Documentation Consistency and Reader Onboarding

**Reviewed revision:** `616e9ca` (`fix: remediate Review Round 51 engine teardown and tokenizer spec findings`), compared with Round 51 review commit `6ad7e91`.

**Scope:** Follow-up review of the engine teardown propagation and U-05 documentation update, with a focus on whether the repository lets a new contributor understand the current system and follow its execution. No implementation or documentation changes were made.

## Findings

### R52-01 — CLI call sites silently discard teardown failures

**Severity: Low — the library now propagates teardown errors, but the executable hides them.**

`DecisionEngine.deinit` now propagates `DeviceNotQuiescent`, and its implementation stops before freeing dependent engine resources. The CLI and server paths in `src/main.zig`, however, all use `defer engine.deinit() catch {}`. A terminal teardown failure is therefore invisible to CLI and server users, despite the release documentation describing the error as caller-visible.

**Recommendation:** Log cleanup failures in these top-level paths, or use explicit shutdown handling where the command can report an unsuccessful teardown. Keep best-effort cleanup in tests separate from user-facing shutdown behavior.

### R52-02 — Current behavior and implementation status remain inconsistent across the docs

**Severity: Medium — new readers cannot tell which prompt and decision contracts are final.**

The updated `specs/tokenizer.md` says end-to-end prompt rendering is fully reference-faithful across all supported schemas. Yet `specs/canvas-templates.md` still labels formatting rules “Prototype” and “Provisional” and says candidate continuation must be tested before implementation. `specs/README.md` describes the canvas layout as proposed and lists open parity requirements. `ARCHITECTURE.md` still describes the model layer as planned and follow-up reads as unestablished. These statements conflict with the resolved U-05 status and the implemented single- and multi-step decision paths.

The new tokenizer statement also generalizes from five golden fixtures to “all supported schemas.” State the tested scope explicitly: which schema kinds and fixture cases are covered, and what remains outside the evidence.

**Recommendation:** Make the architecture and specification index describe the current implementation. For each behavior, keep one normative source and label it `implemented`, `experimental`, or `open`; link to code/tests and record exactly what fixture coverage supports parity claims. Move obsolete proposals and roadmap material into clearly marked history rather than leaving it next to active contracts.

## Documentation Direction

Organize documentation around a reader's path:

1. **Start here (`README.md`):** purpose, prerequisites, build, one CLI decision, one API request, expected output, and links to the next document.
2. **System tour:** trace one request from HTTP/native API through prompt formatting and tokenization, canvas seeding, CPU/CUDA execution, candidate scoring, and response. Include a sequence diagram and define terms such as structured read, canvas, pinned slot, denoising step, and entropy.
3. **As-built architecture (`ARCHITECTURE.md`):** current modules, backend boundaries, data flow, ownership/lifetimes, and teardown behavior. Separate current behavior from experiments and planned work.
4. **Normative specs:** exact model, tokenizer, canvas, API, and kernel contracts, with status and evidence links at the top of each page.
5. **Verification and performance:** commands, model digest, hardware, fixture count and schema coverage, tolerance policy, sample counts, and limitations for each reported result.
6. **Contributor guide:** how to build CPU and CUDA paths, run the narrow and full checks, add an oracle fixture, and update the relevant contract.

The unresolved register should remain a gate and evidence ledger, not the default first document for someone trying to understand what Zeno currently does.

## Review Verdict

The R51 engine-level error propagation is a substantial improvement, and the U-05 correction records the captured template and single-BOS rule. The command-line call sites still hide teardown errors. More broadly, the docs still mix current contracts with old proposals and plans, so a newcomer has no dependable end-to-end path from purpose to a running decision and then to implementation detail. Reconcile the status language and add that reader path before presenting the documentation as a coherent description of v0.2.0.

## Verification Limits

The review inspected `DecisionEngine.deinit`, its command-line call sites, the U-05 tokenizer update, the canvas-template spec, the specs index, and `ARCHITECTURE.md`. The commit reports passing builds, oracle checks, and server tests; none were rerun for this review.

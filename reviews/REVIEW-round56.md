# Review Round 56 — Decision Contract and CUDA Portability

**Reviewed revision:** `0d2577a` (`fix(cuda): remediate Review Round 55 architecture compatibility gate R55-01`)

**Scope:** Read-only project review of the public decision contract, prompt and candidate-token path, oracle fixture validation, and CUDA architecture admission. Inspected active API/tokenizer/canvas specs and the related engine, tokenizer, CUDA selection, fixture, and verification code.

## Verdict

The latest CUDA change improves compatibility handling for ordinary `sm_XX` cubin targets and rejects malformed target strings. The broader project still has a mismatch between the documented choice API and the prompt actually sent to the model, and the candidate-token path does not implement the specified continuation-at-answer-boundary rule. The new CUDA admission logic also treats virtual PTX targets as if they were SASS-only targets.

These affect core decision semantics and GPU portability claims. I would not yet describe the end-to-end decision contract or support beyond the recorded runtime host as fully verified.

## Findings

### R56-01 — Choice codes are not shown in the rendered prompt

**Severity: High — multiple-choice requests can score a code that the model was never told corresponds to the displayed choice.**

The API contract separates a one-token output `token` from its user-facing `label`; its example maps `token: "A"` to `label: "Credit Card Fraud"`. The canvas specification says the choices should be rendered as coded options such as `A: Outage`. `Canvas.formatDecisionPrompt`, however, writes only `cand.label` for each option. It later scores the model's logit for `cand.token`.

For a request with token `A` and label `Credit Card Fraud`, the model therefore sees `Credit Card Fraud` with no `A` association, while Zeno chooses based on the logit for `A`. Boolean examples where token and label are identical do not expose this issue.

**Recommendation:** Render the output code together with its label using the specified choice format, while keeping the response decision equal to the intended display label. Add a multiple-choice end-to-end fixture where token codes and labels differ, and verify the prompt includes each mapping.

### R56-02 — Candidate resolution is not continuation-aware, and fixture checks omit token-ID equality

**Severity: High — candidate logits may refer to a different token from the one implied by the prompt contract, while existing verification does not directly assert the candidate ID.**

The tokenizer spec requires resolving a candidate continuation at the answer boundary of the fully rendered prompt. `DecisionEngine.decide` resolves candidates before it formats or tokenizes that prompt. `Tokenizer.resolveCandidateToken` searches the vocabulary for the exact string first, then tries a leading SentencePiece space prefix; it does not tokenize the rendered prompt followed by the candidate and calculate the continuation. There is also a literal template mismatch: the spec's answer prefix is `Answer: `, but the renderer appends `Answer:` with no trailing space.

The fixture path checks that prompt tokens match the recorded prompt and compares candidate scores, but it does not assert that `resp.candidates[i].token_id` equals the fixture candidate's recorded `token_id`. The Python parity verifier uses fixture token IDs to read oracle logits and compares Zeno's scores by candidate order, but likewise does not compare Zeno's returned candidate IDs. Score tolerances can therefore obscure a candidate-token mapping error.

**Recommendation:** Make the rendered answer prefix match the normative spec. Resolve candidate codes by tokenizing the continuation in that exact context and require precisely one continuation token. Assert candidate token IDs against fixture/reference IDs in both relevant verification paths.

### R56-03 — CUDA admission does not model the code images produced by NVCC targets

**Severity: Medium — a valid PTX-only or PTX-containing build can be rejected on a GPU that could execute it.**

`build.zig` passes the configured value as `nvcc -arch=<target>`. The current parser accepts both `sm_XX` and `compute_XX`, but `TargetArch` records only major/minor and an `exact_only` flag. `isDeviceArchCompatible` then requires the device to have the same major compute-capability version. As a result, `compute_80` is treated as SASS targeting CC 8.x, even though NVCC documents `-arch=compute_80` as PTX-only and forward-compatible with later GPUs. NVCC also documents `-arch=sm_XX` as producing both a cubin and PTX; the gate can reject a later-major device even where the embedded PTX is usable. See NVIDIA's [NVCC compilation documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html).

**Recommendation:** Represent whether a target emits SASS, PTX, or both, then admit a device when at least one embedded code image is compatible under CUDA's rules. Keep architecture-specific targets such as `sm_90a` constrained according to their narrower compatibility. Add tests for `compute_80` on a later major, `sm_90` with PTX fallback on a later major, and `sm_90a` exact-only behavior.

## Review Conclusion

The Round 55 change addresses the plain-cubin cases it targeted, but the project-wide review finds two unresolved decision-contract issues and one remaining CUDA target-model issue. Prioritize the choice prompt and candidate continuation fixes before relying on the API for arbitrary labels or treating oracle fixture success as proof of candidate-token parity. Treat PTX-compatible GPU execution as build-configurable until selection models the actual embedded code forms and hardware runtime checks confirm it.

## Verification Limits

Inspected commit `0d2577a`, the request/response and canvas/tokenizer specs, `src/engine.zig`, `src/canvas.zig`, `src/tokenizer.zig`, `src/infer_cuda.zig`, `src/fixture.zig`, `tools/verify_oracle_parity.py`, and the CUDA build configuration. No builds, tests, or GPU runs were performed for this review.

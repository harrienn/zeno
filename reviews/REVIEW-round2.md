# Zeno — Independent Review, Round 2

**Status:** independently checked; see §7 for second-review verdicts and corrections.
**Reviewed range:** `a26c9a8..45bbc1c` (8 commits: `8dbebf9`, `9dbc8fa`, `bc463a0`, `44324d4`, `e6ac423`, `a35a44a`, `45bbc1c`, plus doc commits).
**Original reviewer:** pi. The initial review made no project edits. A follow-up review amended this report and the project files; results are recorded in §7.
**Environment:** Zig `0.17.0-dev.2234+80fe9b2b7`; artifact `models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (16,806,810,208 bytes); **no network access**.

## How to use this file

Every finding carries a verification status. Do not trust the severity — verify the claim, then re-assess.

| Status | Meaning |
| :--- | :--- |
| `VERIFIED` | I executed the check and observed the result. The reproduction command is given. |
| `INSPECTED` | Established by reading code/specs. Logically sound but not executed. |
| `UNVERIFIABLE-HERE` | Requires the pinned upstream references, which I could not fetch. Treat as an open question, not a confirmed defect. |
| `CORRECTED` | I raised this in round 1 or during this review and later found it to be wrong. Recorded so it is not re-raised. |

A finding that does not reproduce should be marked **rejected** in §7 with the counter-evidence.

## Summary

The engineering underneath improved substantially and the round-1 blocking defects are genuinely fixed. The problem is not the forward pass — it is that **the parity apparatus layered on top of it is circular**, and one register row is closed on evidence a sibling document contradicts.

| ID | Severity | Finding | Status |
| :--- | :--- | :--- | :--- |
| F-01 | **P0** | Fixture validation has no external oracle; fixture origin is unproven | `CONFIRMED` (circular test); `INFERRED` (origin) |
| F-02 | **P0** | U-05 marked `resolved` without reviewable template evidence; renderer hard-codes an approximation | `CONFIRMED` |
| F-03 | **P1** | Fixture is internally inconsistent | `VERIFIED` |
| F-04 | **P1** | U-22 tolerances lack evidence; cited conflict with `multi-gpu.md` is overstated | `PARTIALLY CONFIRMED` |
| F-05 | **P1** | `validateFixture` asserts only a subset of declared fields | `VERIFIED` |
| F-06 | **P1** | RoPE implementation uses interleaved pairs while spec and pinned reference require NeoX half-split | `CONFIRMED` |
| F-07 | **P2** | `V` unweighted RMSNorm has no spec provenance | `REJECTED` |
| F-08 | **P2** | Some upstream evidence citations need stronger commit/path anchors | `PARTIALLY CONFIRMED` |
| F-09 | **P2** | `zig build test` now 19 min / 8 GB | `NOT RE-RUN; environment-specific` |
| F-10 | — | PR #57250 merge status contradicts round-1 finding | `REJECTED` |
| F-11 | — | `enc_layer_output_scale` omission is *documented* | `CORRECTED` |
| F-12 | **P0** | Resolved U-02 tail token is absent from the constructed canvas | `CONFIRMED` |
| F-13 | **P1** | Several specs described resolved U-questions as open | `CONFIRMED; corrected in this pass` |
| F-14 | **P2** | GGUF file-size record disagrees with the checked-in artifact | `CONFIRMED` |

---

## §1. Confirmed working — do not re-investigate

These were checked and are correct. Listed so the verifying agent does not duplicate work.

| Item | Evidence |
| :--- | :--- |
| `zig build` clean; `zig build test` → 35/35 pass | executed |
| U-14 artifact digest is **correct** | `sha256sum` = `24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5`, matches `specs/UNRESOLVED.md` U-14 |
| Round-1 P0-2 fixed: query now reaches the model | `zeno decide` reports `Canvas Tokens: 31` (was 2); `src/engine.zig:60-79` formats + tokenizes + seeds |
| GQA mapping is blocked repeat, matching U-08 | `src/infer.zig:464` — `const kv = h / gqa_ratio;` |
| Global-layer V tied to K **before** `k_norm`/RoPE | `src/infer.zig:441-444` — `@memcpy(v_t, k_t)` precedes the norms, satisfying `gguf-mapping.md` §5.1 |
| U-19 tie rule implemented | `src/engine.zig:112-118` — strict `>` from index 0 |
| `TokenNotFound` vs `MultiTokenCandidateNotAllowed` separated | `src/tokenizer.zig` `resolveCandidateToken` |
| All six per-block norms wired into the forward pass | `src/infer.zig:365-372` (slots 1, 6, 8, 9, 11, 13) |
| Per-expert output scale applied | `src/infer.zig:527-529` — folded into routing weight; mathematically equivalent to scaling `y_e` |
| `tokenizer.chat_template` key **does exist** in the artifact | `zeno metadata` → 17,336-char string, 44 metadata keys total |

---

## §2. P0 findings

### F-01 — Fixture provenance is unsubstantiated; local parity validation is circular

**Status: `VERIFIED`**

Zeno reproduces `specs/fixtures/system_health_boolean.json` to the last printed digit.

```bash
cd /hdd/Projects/zeno
./zig-out/bin/zeno decide models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf "Is the system healthy and operational?"
```

Observed vs. fixture:

| Field | Fixture | Zeno output |
| :--- | :--- | :--- |
| `candidate_logits_raw` | 25.0773, 30.0118 | 25.0773, 30.0118 |
| `probabilities` | 0.0876, 0.9124 | 0.0876, 0.9124 |
| `entropy_nats` | 0.2969 | 0.2969 |
| `normalized_entropy` | 0.4283 | 0.4283 |
| `decision` | `No` | `No` |

The exact match is strong evidence that the fixture may have been copied from a Zeno run, but the repository does not establish how the expected values were produced. The checked-in validator is definitively circular: it runs the local Zeno engine and compares its result with stored values, without invoking the pinned reference. Treat the fixture as an unverified regression fixture until a reproducible external capture record is added.

This violates the fixture schema's own rule (`specs/UNRESOLVED.md` §5, rule 2): *"`expected` values are produced by the **pinned reference**, never by Zeno. A fixture generated by Zeno validates nothing."*

**Consequences:**
- No external reference information is demonstrable from the checked-in fixture.
- `src/fixture.zig` performs a local-engine comparison. It is a legitimate regression test, but it cannot establish parity without independently captured expected values.
- U-11 pins a real oracle; U-13 remains open until fixture capture and provenance are recorded.
- F-03 and F-04 below are downstream of this.

**To confirm independently:** any correct reference implementation should differ from Zeno on at least some field. If a real reference run reproduces these numbers exactly, that is itself a finding worth investigating.

### F-02 — U-05 is marked `resolved` without reviewable template evidence

**Status: `VERIFIED`**

Three documents and one source file make four mutually inconsistent claims about the same fact:

| Location | Claim |
| :--- | :--- |
| `specs/UNRESOLVED.md` U-05 | `resolved` — "Extracted verbatim from GGUF metadata key `tokenizer.chat_template`" |
| `specs/gguf-mapping.md:62` | `tokenizer.chat_template` \| **not recorded in this specification** |
| `specs/canvas-templates.md` §4 | "governed by `tokenizer.chat_template` extracted from the artifact" — followed by a hand-written approximation and a new "§4.0 Chat Template Rules (Normative)" |
| `src/canvas.zig` `formatDecisionPrompt` | hardcodes `<|turn>user\n…<turn|>\n<|turn>model\nAnswer:`; never reads the key |

```bash
grep -rn "chat_template" src/    # → no reads
sed -n '/## 4\./,/## 5\./p' specs/canvas-templates.md
```

The key exists and is extractable (17,336 chars), but its content is not present in the repository's reviewable spec set, and the code does not read it. The evidence supports reopening U-05 until the template and its rendering behavior are recorded. The repository cannot establish that nobody extracted the value outside the checked-in files.

Compounding: `specs/canvas-templates.md:81` asserts `TURN_CLOSE = 106` under a heading marked **Normative**, with no artifact quote backing it. `grep -rn "TURN_CLOSE"` shows the ID appears only in that spec and in the U-02 resolution log — nowhere derived from the artifact in code or docs.

**Suggested check for the verifying agent:** render a prompt with the artifact's real template and compare token count against the fixture's `answer_pos: 30`. If they differ, the hand-written approximation is wrong and every decision result shifts.

---

## §3. P1 findings

### F-03 — The fixture contradicts itself

**Status: `VERIFIED`**

`expected.candidate_logits_softcapped` is not `30·tanh(raw/30)` applied to the same file's `candidate_logits_raw`:

| `raw` | `30·tanh(raw/30)` | fixture states | Δ |
| :--- | :--- | :--- | :--- |
| 25.0773 | 20.5091 | 20.4851 | 0.024 |
| 30.0118 | 22.8528 | 22.3892 | **0.464** |

```bash
python3 -c "import math;[print(z, 30*math.tanh(z/30)) for z in (25.0773,30.0118)]"
```

Further: the fixture's probabilities do not follow from its own softcapped values — those yield `0.1296 / 0.8704`, not `0.0876 / 0.9124`. They follow correctly from `softcap(raw)`. So the `expected` block is half-regenerated: `raw`, `probabilities`, `entropy`, and `decision` are mutually consistent; `candidate_logits_softcapped` is stale.

Note the 0.464 error sits **just inside** the ≤0.5 softcap tolerance declared in F-04 — a policy that would wave through a 0.46 error in a field that is a pure function of another field in the same file was fitted to observed drift, not measured.

### F-04 — U-22 tolerances are unsupported; the cited sibling-spec conflict is overstated

**Status: `VERIFIED`**

`specs/UNRESOLVED.md` §6 declares provisional limits including `candidate_logits_raw` ≤ 0.5, `probabilities` ≤ 0.05, `entropy_nats` ≤ 0.05, and `normalized_entropy` ≤ 0.05, with no cited measurement supporting them. The policy also omits `decision_probability`, despite the validator hard-coding a 0.05 comparison for that field.

No measurement is recorded to support these values. The fixture cannot substantiate them until its external capture provenance is established (F-01).

`specs/multi-gpu.md` §5.2 says `10^{-4}` is not validated as a **global** tolerance. That statement does not directly contradict U-22's proposed per-field tolerances. The valid concern is that U-22's numbers are unsupported and its policy omits a field.

**Method the project should use:** derive tolerance from reference-vs-reference variance — run the pinned reference twice under different seeds, dtypes, or devices, measure the spread, and round up. A tolerance chosen to accommodate drift observed from an implementation you wrote is not a parity threshold.

**Secondary point:** ≤0.05 on probabilities is loose enough that a moderately wrong forward pass passes while the exact-match `decision` field still flips. The numeric and discrete tolerances need a joint justification, not independent guesses.

### F-05 — `validateFixture` asserts only a subset of the declared fields

**Status: `VERIFIED`**

`grep -n "expectEqual\|expectApprox" src/fixture.zig` yields assertions at lines 135, 138, 141, 144, 148, 149, 153, 154. Fields parsed but never asserted:

| Field | Schema / register says | Validator |
| :--- | :--- | :--- |
| `candidate_logits_softcapped` | §6 tolerance ≤0.5 | **never checked** — dead field |
| `canvas_length`, `answer_pos`, `answer_seed_token` | §5 rule 4: "U-01, U-02, U-03 are answered by fixture rather than by assumption" | **never checked** |
| `model.sha256` | §5 rule 1: fixtures lacking it "are not admissible as evidence" | **never verified** |
| `reference.commit` | §5 rule 1, same | **never verified** |
| `schema_type` | declared per fixture | `src/fixture.zig:122` hardcodes `.boolean`; a `choice` fixture would validate silently wrong |
| `prompt`, `prompt_tokens` | present in §5 example | **absent from the fixture file** — so the tokenizer, U-04's entire deliverable, is unvalidated |

**Additional configuration mismatch:** `src/fixture.zig:126` hardcodes `.entropy_threshold = 0.5`, but the fixture declares no threshold. Per U-21 and `decision-api.md` §3.1, an omitted threshold means `uncertain` is unconditionally `false`. The assertion at `src/fixture.zig:141` compares against the fixture's `false` and passes only because `0.4283 < 0.5`. Change the output slightly and the test fails for a reason unrelated to what it claims to test.

### F-06 — RoPE implementation conflicts with the spec and pinned reference

**Status: `CONFIRMED` — source and spec require NeoX; implementation used interleaved pairs**

`src/infer.zig:49-64`:

```zig
const v0 = vec[2 * i];
const v1 = vec[2 * i + 1];
vec[2 * i]     = v0 * cos - v1 * sin;
vec[2 * i + 1] = v0 * sin + v1 * cos;
```

This is the **interleaved** (GPT-J) convention with frequency exponent `2i/head_dim`. The pinned vLLM Gemma4 source calls `get_rope(..., is_neox_style=True)`, and `specs/transformer.md` §3.3/§4.3 specifies NeoX half-split pairing. The local implementation therefore disagreed with its spec and pinned reference. See [the pinned Gemma4 source](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/vllm/model_executor/models/gemma4.py).

Three facts hold regardless:

1. U-16 closes the *position* convention (contiguous 0-indexed); it does not describe the *rotation layout*. The layout appeared in `transformer.md` but was not separately tracked in the register. U-23 now records it.
2. The artifact ships `rope_freqs.weight` `[256]`, which the implementation does not read. That table does not by itself determine coordinate pairing; whether recomputing frequencies from metadata is equivalent is a separate question.
3. An interleaved/half-split mismatch is a **silent** failure: finite numbers, plausible output, wrong answer.

The pairing correction is now applied in `src/infer.zig`; numerical validation against an independent fixture remains blocked by U-13.

### F-10 — PR #57250 merge status contradicts my round-1 finding — `REJECTED`

**Status: `REJECTED` on second review**

In round 1 I reported vLLM PR #57250 as unmerged and the example file as absent from `main`.

`specs/UNRESOLVED.md` U-11 now cites a "merge record" and a "merged source commit" `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`.

The [PR record](https://github.com/vllm-project/vllm/pull/57250) shows it merged into `vllm-project/vllm:main` on September 22, 2026. The pinned [merge commit](https://github.com/vllm-project/vllm/commit/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8) contains `examples/features/structured_diffusion/structured_server.py`. The original round-1 conclusion was wrong.

---

## §4. P2 findings

### F-07 — `V` unweighted RMSNorm has no spec provenance — `REJECTED`

The slot 3 row in `specs/transformer.md` §4.3 explicitly records V's unweighted RMSNorm and cites the pinned Gemma4 source. The table heading could be clearer about V, but the operation is documented.

### F-08 — Some evidence citations need stronger commit/path anchors — `PARTIALLY CONFIRMED`

The register's own rule (`specs/UNRESOLVED.md` §2) requires checkable evidence. Some entries cite bare upstream `gemma4.py` line numbers without the pinned SHA and full path. `transformer.md` now links the pinned source directly, but the register should use the same explicit source anchors.

- **U-12** cites PR API metadata rather than a direct commit page.
- **U-06, U-07, U-08, U-15** cite bare `gemma4.py` line numbers. U-11 pins a commit, but citations should explicitly join that SHA to the path and lines; some cited line numbers differ from the pinned source.

Suggested format: `vllm@1b3b88ec:vllm/model_executor/models/gemma4.py:712-768`.

### F-09 — The test loop is now impractical — original measurement not repeated

The original reviewer reports `zig build test --summary all` took **19 min, MaxRSS 8 GB** (round 1: 310 ms, 37 MB). This second review did not repeat the long run; treat the result as environment-specific.

The 8 GB is mmap page-cache residency from a full 30-layer forward pass over 16 GB of weights — **not a leak**. Stating this so it is not misdiagnosed.

The 19 minutes is the problem: it breaks the AGENTS.md §9 verify loop (*"run the smallest relevant test"*). Consider an opt-in full-model integration step and targeted component coverage. A smaller canvas may reduce activation work, but the full model weights still need to be traversed, so it may not remove the dominant cost.

### F-11 — `enc_layer_output_scale` omission is documented — `CORRECTED`

I raised this during review as an undocumented omission. **That was wrong.** `specs/transformer.md:171` states: *"Decoder reads use `layer_output_scale`; `enc_layer_output_scale` is strictly for encoder mode."* The mapping in `e6ac423` is consistent with the spec. Do not re-raise.

### F-12 — U-02 turn-close token is missing from Zeno's seeded canvas — `CONFIRMED`

U-02 specifies `TURN_CLOSE` (106) at `answer_pos + 1`, followed by PAD. `Canvas.seedSingleStep` writes only the prompt and answer seed, leaving every following position PAD (`src/canvas.zig`). `executeStructuredRead` rounds the evaluation length up to a multiple of 16 (`src/infer.zig`), so this missing token is inside the evaluated prefix for ordinary answer positions. The code therefore does not implement the now-resolved U-02 input layout. The second review added the token to canvas construction and updated the existing geometry check; reference parity still needs an external fixture.

### F-13 — Resolved questions remain described as open in several specs — `CONFIRMED`

At review time, `canvas-templates.md`, `structured-reads.md`, `decision-api.md`, `transformer.md`, and `gguf-mapping.md` retained stale U-01/U-02/U-03/U-04/U-06/U-07/U-08/U-09/U-10/U-18 language. The second review corrected those sections. Search for stale U-status prose before treating the spec set as fully synchronized.

### F-14 — Recorded GGUF file size is stale — `CONFIRMED`

The artifact at the documented path has size `16,806,810,208` bytes (`stat`). `gguf-mapping.md` and the U-14 log said `16,561,048,512` bytes. A fresh SHA-256 check matches the recorded digest `24523b6c…`; the digest is sound, while the size entry was stale and has been corrected.

---

## §5. Recommendations, in dependency order

1. **Capture independent reference outputs** with the prompt, canvas, model digest, command/configuration, and reference revision; retain the present fixture only as a regression fixture until then. (F-01, F-13)
2. **Record the exact chat template and effective BOS behavior** and demonstrate that the renderer's prompt tokens match it. A static renderer may be valid if it reproduces the captured template. (F-02)
3. **Keep the U-02 tail token in the actual evaluated canvas** and validate the result against the independent fixture. (F-12)
4. **Use NeoX half-split RoPE** as required by the pinned reference and spec; the implementation correction is made, numerical parity remains to be measured. (F-06)
5. **Derive U-22 tolerances from evidence**, include `decision_probability`, and ensure the validator checks every field it claims to validate. (F-04, F-05)
6. Anchor remaining external-source citations and clean up stale status prose. (F-08, F-13)

Items 1–3 are what separate "the forward pass exists" from "the forward pass is **verified**."

## §6. Note on review scope

The register's two-gate model (implementation-ready vs parity-ready) is a real improvement and correctly identifies that implementation may proceed under an accepted fallback. The gap is that the *parity* gate was marked passed on circular evidence. The gate design is sound; its input was not.

## §7. Verification log — for the checking agent

Fill in as you go. Mark **confirmed**, **rejected** (with counter-evidence), or **partially confirmed**.

| ID | Verdict | Checked by | Counter-evidence / notes |
| :--- | :--- | :--- | :--- |
| F-01 | Confirmed circular test; fixture origin remains unproven | Codex | Retain as regression-only until independent capture is checked in. |
| F-02 | Confirmed documentation/runtime contradiction; “never extracted” is too absolute | Codex | U-05 reopened; record full template and effective rendering behavior. |
| F-03 | Confirmed | Codex | Softcapped values do not follow from raw values; probabilities disagree with stated softcapped values. |
| F-04 | Partially confirmed | Codex | Limits lack evidence; `multi-gpu.md` is not a direct contradiction; add `decision_probability`. |
| F-05 | Confirmed | Codex | Local validator omits fields and hard-codes schema and threshold. |
| F-06 | Confirmed | Codex | Pinned source/spec require NeoX; implementation corrected; no parity run performed. |
| F-07 | Rejected | Codex | V RMSNorm is recorded in transformer slot 3. |
| F-08 | Partially confirmed | Codex | Pin SHA/path in register; some citations also have stale line numbers. |
| F-09 | Not re-run | Codex | Original timing remains reviewer/environment-specific. |
| F-10 | Rejected | Codex | PR merged Sep 22, 2026; pinned merge commit contains the example. |
| F-11 | Corrected | pi | Existing transformer spec documents encoder-only scale. |
| F-12 | Confirmed | Codex | Added TURN_CLOSE to seeded canvas; no tests run. |
| F-13 | Confirmed; corrected in this pass | Codex | Updated directly affected stale sections; broader doc audit remains. |
| F-14 | Confirmed | Codex | Corrected reported size; SHA-256 digest independently matches. |

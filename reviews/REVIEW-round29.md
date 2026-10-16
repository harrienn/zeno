# Review Round 29 — Integrity and Robustness of the Decision, CLI, and Evaluation Paths

**Reviewed revision:** `7e3ba5b` (`fix(calibration): resolve Review Round 28 findings (R28-01 through R28-06)`).

**Scope:** Static review of the core decision path (`src/engine.zig`, `src/decision.zig`, `src/canvas.zig`, `src/ops.zig`, `src/infer.zig`), the frontend and tooling (`src/server.zig`, `src/main.zig`, `src/fixture.zig`, `src/dataset.zig`), the parsers and numerics (`src/gguf.zig`, `src/quant.zig`, `src/tokenizer.zig`), the CUDA memory and orchestration layers (`src/cuda_mem.zig`, `src/infer_cuda.zig`, `src/cuda.zig`), and the Python evaluation tools. `zig build test --summary all` was run and passes (cached). No fresh `-Dcuda` build, 26B model run, CUDA benchmark, or parity comparison was performed.

## Verdict

This is an unusually disciplined research codebase. The architecture documents, the `specs/UNRESOLVED.md` register, and the 28 prior review rounds show a project that separates *implementation-ready* from *parity-ready*, records provenance (artifact SHA-256, pinned reference commits), and refuses to overclaim calibration or CUDA parity. The core decision path (`engine.zig` → `infer.zig` → `ops.zig`) is clean, explicit, and faithful to the documented semantics; error handling and memory ownership are mostly careful.

The findings below are **not** on the happy-path correctness of a decision. They are integrity and robustness defects: one class silently changes a run's configuration, two writers can emit corrupt artifacts, the calibration analysis tool can fabricate a "100% / zero-risk" line, and several CUDA and CLI paths have latent leaks or unvalidated inputs. None are High severity, but the calibration and artifact writers matter because the project's credibility rests on the numbers being what the tool reports.

## Findings

### R29-01 — Malformed CLI option values are silently replaced by defaults

**Severity: High (for run reproducibility)**

**Location:** `src/main.zig:142`, `src/main.zig:147`, `src/main.zig:394`, `src/main.zig:399`, `src/main.zig:481`, `src/main.zig:486`, `src/main.zig:491`

```zig
step_cap = std.fmt.parseInt(u32, args[arg_i], 10) catch 1;
canvas_override = std.fmt.parseInt(usize, args[arg_i], 10) catch null;
entropy_threshold = std.fmt.parseFloat(f32, args[arg_i]) catch null;
```

A typo such as `--threshold 0,5` silently disables uncertainty gating (`null`), `--canvas abc` silently runs 256, and `--steps two` silently runs one step. The user receives a result for a configuration they did not request. `serve` already handles this correctly for `--port` (prints an error and returns), so the correct pattern exists in the same file. The missing-value cases (`--steps` at end of argv) are also silently skipped.

**Recommendation:** Fail loudly for `--steps`, `--canvas`, and `--threshold`, matching the `serve --port` pattern, and emit an error when the value argument is absent.

### R29-02 — `main.zig` writers ignore short writes and skip JSON escaping

**Severity: Medium**

**Location:** `src/main.zig:1067-1072` and `src/main.zig:1074-1127`

```zig
_ = std.os.linux.write(fd, bytes.ptr, bytes.len);         // return discarded
...
_ = std.os.linux.write(out_fd, json_list.items.ptr, json_list.items.len);
```

`write(2)` may return a short count, producing truncated `.i32` prompt/canvas files (which `tools/generate_golden_fixtures.py` feeds to the oracle) or truncated JSON, with no error reported. `dataset.zig:232-239` already implements the correct retry loop.

Separately, `writeDecisionResponseJson` interpolates `resp.decision` and `c.label` into JSON string literals without escaping, while `dataset.zig` (`escapeJsonString`) and `server.zig` (`writeEscapedJsonString`) both escape. Labels come from CLI arguments and fixture files, so a quote, backslash, or newline yields invalid JSON.

**Recommendation:** Loop until all bytes are written and propagate errors, reusing the pattern in `writeEvaluationResultsJson`. Route `resp.decision` and `c.label` through an escaping helper before interpolation.

### R29-03 — Calibration tool can report 100% selective accuracy at 0% coverage and hides missing ground-truth labels

**Severity: Medium**

**Location:** `tools/evaluate_calibration.py:108` and `tools/evaluate_calibration.py:89-94`

```python
acc_cov = (correct_cov / cov_count) * 100.0 if cov_count > 0 else 100.0
```

When no example passes an entropy cutoff, the table prints `Selective Acc = 100.00% / Selective Risk = 0.00%`. A reader can mistake "we declined to answer anything" for "we are perfect", which is the opposite of what a selective-risk curve conveys.

In the NLL path, if a result's `ground_truth` matches no candidate label, `prob_gt` stays at the `1e-12` default and the run silently contributes roughly `+27.6` nats instead of surfacing the data error while still appearing computed.

**Recommendation:** Emit `N/A` and skip the risk computation when `cov_count == 0`; distinguish "found" from "not found" for the ground-truth probability and report or raise on the missing-label case. The ECE/MCE/Brier math itself was recomputed against the checked-in result files and is correct.

### R29-04 — Documented maximum prompt length is unreachable with the default workspace

**Severity: Medium**

**Location:** `src/root.zig:80` versus `src/engine.zig:24`, `src/engine.zig:217`

`ModelConfig.max_prompt_tokens = max_seq_len - canvas_length = 1024 - 256 = 768`, and both `decide` and `Canvas.seedSingleStep` enforce 768. However `DecisionEngine`'s default `workspace_seq_len` is `512` (`EngineOptions.workspace_seq_len`), and `validateSequenceLength(.cpu, total_seq_len, ws.max_seq_len)` runs *before* `executeStructuredReadSlice` can grow the workspace via `ensureCapacity`. A 300-token prompt therefore passes the 768 check and is then rejected with `CanvasOverflow`; the effective CPU limit is 256 prompt tokens with a 256 canvas. CUDA is capped at 512 total regardless (`src/infer_cuda.zig:532` even declares a different `max_prompt_tokens = 512`).

**Recommendation:** Either raise the default `workspace_seq_len` to `max_seq_len` (1024) for CPU, or reduce `max_prompt_tokens` and the U-20 register claim of "maximum prompt 768" to match the effective limit, and make the two `max_prompt_tokens` constants agree.

### R29-05 — CUDA: `rope_freqs` device buffers leak if model load fails before the cleanup `errdefer` is registered

**Severity: Medium**

**Location:** `src/infer_cuda.zig:380-388` versus `src/infer_cuda.zig:428-438`

`rf0`/`rf1` are allocated and copied with no per-buffer `errdefer`. The cleanup that frees `rope_freqs_dev0/dev1` is a function-scope `errdefer` registered at line 428, so it does not cover an error from the device-1 allocation or copy at lines 385-386 (a Zig `errdefer` only guards code that runs after it is registered). The self-conditioning buffers above use immediate `errdefer sc_*.deinit()` correctly.

**Recommendation:** Add `errdefer rf0.deinit();` and `errdefer rf1.deinit();` immediately after each successful `DeviceBuffer(f32).init`.

### R29-06 — CUDA: `DeviceBuffer.deinit` retains the handle on a failed free in release builds

**Severity: Medium**

**Location:** `src/cuda_mem.zig:310-322` (and the same shape in `CudaStream.deinit`, `src/cuda_mem.zig:137-148`)

```zig
if (rc != 0) { std.log.err(...); std.debug.assert(rc == 0); return; }  // ptr stays set
```

In Debug/ReleaseSafe this aborts on any transient failure; in ReleaseFast the assert is stripped and the function returns with `self.ptr` still set, so the allocation is leaked with no remaining live reference and no retry path. The intent to "retain" defeats itself.

**Recommendation:** Null the handle regardless of `rc` (logging the failure), or change `deinit` to return an error and propagate it.

### R29-07 — CUDA: unvalidated `answer_pos` and missing cached-attention stride checks

**Severity: Low/Medium**

**Location:** `src/infer_cuda.zig:1574`; `src/cuda_mem.zig:993-1024`

`infer_cuda.zig:1574` guards `answer_pos` only with `std.debug.assert`, which is compiled out in ReleaseFast; an out-of-range position then reads or writes an uninitialized workspace region and yields wrong logits silently. Use a runtime `if (answer_pos >= canvas_tokens.len) return error.InvalidParam;`.

`attentionCached` validates lengths but omits the stride lower-bound checks that `attention` performs (`src/cuda_mem.zig:923-925`), so a too-small stride would pass the length check and index out of bounds. Latent today (callers pass full strides), but an inconsistent validation contract.

**Recommendation:** Replace the assert with a runtime check active in all build modes; add the same stride lower-bound validation to `attentionCached` as in `attention`.

### R29-08 — Fixture validator under-checks expected arrays and mis-validates answer position

**Severity: Low**

**Location:** `src/fixture.zig:218-232`; `src/fixture.zig:75-85`

`validateFixture` length-checks `candidate_logits_raw` and `candidate_logits_softcapped` but then indexes `baseline.probabilities[i]` with no length check, so a truncated fixture panics inside a function whose purpose is to validate fixtures gracefully. `evaluateReferenceParity` has the same unchecked indexing.

The final assertion `fix.answer_pos + 1 <= resp.canvas_tokens_used` is vacuous: the engine sets `canvas_tokens_used = canvas.prompt_len + 1`, so with `answer_pos == 0` it reduces to `0 <= prompt_len`. The fixture's `answer_pos` and `answer_seed_token` are effectively never validated.

**Recommendation:** Assert each expected array's length equals `resp.candidates.len` before indexing; validate the pinned position against the engine's actual canvas rather than `prompt_len + 1`.

### R29-09 — `--strict-parity` is a no-op without a Zeno JSON argument

**Severity: Low**

**Location:** `tools/verify_oracle_parity.py:99-169`

The only `sys.exit(1)` lives inside `if zeno_json_path:`. Running `verify_oracle_parity.py oracle.bin fixture.json --strict-parity` never performs a parity gate and exits 0, so a CI job that relies on the flag silently passes.

**Recommendation:** Error out (non-zero) when `--strict-parity` is set but no Zeno JSON was provided.

### R29-10 — `--host localhost` is recognized as loopback but fails to bind

**Severity: Low**

**Location:** `src/main.zig:109` versus `src/server.zig:121`, `src/server.zig:48-62`

`main.zig:109` special-cases `"localhost"` as trusted loopback (suppressing the non-loopback warning), but `Server.listen` calls `parseIp4`, which returns `error.InvalidIp` for `"localhost"`. The command then fails at `try server_inst.listen()`, which is uncaught, so it surfaces as a raw returned error rather than a message. Either resolve hostnames or drop the `"localhost"` branch.

**Recommendation:** Remove the `"localhost"` special case, or add hostname resolution so the documented host form binds.

## Process Observations

- **No CI, and the default test command does not exercise the model path.** There is no `.github/` directory and no workflow file. `zig build test` runs only lightweight unit tests; the golden-fixture comparisons (`validateFixture`), the end-to-end engine tests, and all CUDA tests are gated behind `shouldRunModelTests()` / `cuda.is_enabled` and require `-Dtest-model=true`. The green `zig build test` reported in the README therefore validates the plumbing, not the decisions. The project's most important verification is exactly what continuous integration would miss.
- **The review loop is working, but pattern-level fixes lag.** R28-03 fixed JSON escaping and short writes in `dataset.zig` but not in the sibling `main.zig` writers (R29-02); R28-02 disciplined a prose overclaim but not the tool that generated it (R29-03). A "when fixing X here, grep for X elsewhere" checklist would pay off.
- **The register and documentation consistency is a strength**, but R29-04 is a case where `root.zig`, `infer_cuda.zig`, and U-20 disagree on the supported sequence range.

## What Is Solid

- `engine.zig`, `decision.zig`, and `ops.zig` validation, tie-breaking (U-19), and entropy/softmax math are clear and correct; the R27-03 adaptive-term state machine has a good standalone test.
- The hand-rolled HTTP server (`server.zig`) is genuinely careful: duplicate `Content-Length` and `Transfer-Encoding` rejection, bounded body size, absolute plus idle deadlines, atomic connection reservation, and a drain-on-stop loop, all with tests.
- `quant.zig` block layouts carry `comptime` size assertions, and the dequantization math matches the llama.cpp formulas inspected.
- Honest scoping: U-13 and U-22 remain `open` with recorded evidence rather than inflated tolerances, and the calibration writeup is labeled exploratory.

## Empirical Interpretation

No new measurements were produced by this review. The checked-in calibration result files were only used to cross-check that the ECE/MCE/Brier/NLL arithmetic in `tools/evaluate_calibration.py` reproduces; it does. No inference, fixture, or benchmark run was performed, so this review provides no new evidence of model parity.

## Verification Notes

- `zig build test --summary all` passes (cached); the heavy `-Dtest-model=true` and `-Dcuda` paths were not built or run.
- Findings are based on static reading of the exact files and line ranges cited. Each was traced through the surrounding code before being recorded; nothing here is a style preference.
- Suggested first change: R29-01 through R29-03, which are small and protect run reproducibility and reported-number integrity, followed by the CUDA `errdefer`/`deinit`/validation cluster (R29-05 through R29-07), and a `zig build test -Dtest-model=true` CI job so the parity tests actually run.

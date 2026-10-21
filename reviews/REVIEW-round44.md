# Review Round 44 — Parity Gate, Canvas Geometry, Calibration, and CUDA Phase 19

**Reviewed revisions:** `6ae7610` (calibration analysis and Round 43 fixes), `7888bf9` (tolerance v2 and U-22), `a54ff32` (canvas geometry study), `65442bb` (CUDA Phase 19), and `2c1e84c` (structured diffusion versus autoregressive study).

**Scope:** Reviewed the changes since Round 43, focusing on new empirical claims, parity enforcement, GGUF validation, calibration behavior, and CUDA resource ownership. This review inspected source and documentation. No source changes were made, and no builds or tests were run.

## Findings

### R44-01 — The v2 parity gate does not enforce exact top-five vocabulary parity

**Severity: High — the reported parity pass omits a normative invariant.**

`specs/UNRESOLVED.md` makes `top_5_vocab_tokens` an exact cross-engine invariant under `zeno.tolerance/v2` (section 6.2). `tools/verify_oracle_parity.py` checks the pinned reference's top five against the recorded oracle reference, but it never compares Zeno's top five with the oracle. The `passed_parity` expression also omits this field (`tools/verify_oracle_parity.py`, around lines 209–219). Consequently, the printed “Full reference parity PASS” can succeed without establishing the stated top-five invariant.

**Recommendation:** Include the Zeno-versus-oracle top-five comparison in the parity gate and ensure the Zeno output contains the required unconstrained ranking. If that ranking is not available for these runs, revise the normative claim and the “U-22 resolved”/full-parity language until it is measured.

### R44-02 — Canvas geometry conclusions do not describe the tested token layouts

**Severity: High — the position sweep does not isolate answer position as claimed.**

`tools/evaluate_geometry.py` sweeps `answer_pos` by invoking `decide-fixture`. `Canvas.seedSingleStepAt` puts `TURN_CLOSE` at `answer_pos + 1`, except at the final canvas slot, where it puts the token at `answer_pos - 1` (`src/canvas.zig`, lines 57–62). Thus at position 1, the actual layout is `PAD, MASK, TURN_CLOSE`, not `TURN_CLOSE, MASK`; at position 15, `TURN_CLOSE` is at 14, not a canvas made entirely of preceding PADs. The study's explanation and production recommendation in `specs/canvas-geometry.md` (sections 3.2 and 6) therefore do not match the evaluated inputs. For most positions the sweep also moves the closing token together with the answer, so it does not measure answer-position effects independently.

**Recommendation:** Record the exact canvas token layout for every sweep point. If the intended experiment is to compare answer position while holding turn boundary placement fixed, add an explicit layout control and rerun it. Otherwise, revise the claims to describe the joint answer/closing-token shift that was actually measured.

### R44-03 — GGUF parsing validates tensor starts but not full payload extents

**Severity: High — a truncated tensor can reach unchecked slicing during model loading.**

`parseGguf` checks that `data_offset + tensor.offset` is not beyond the file, but does not verify that the tensor's complete byte range fits (`src/gguf.zig`, lines 279–283). `getTensorSlice` returns the remaining file tail, and `getExactTensorSlice` slices that tail to the computed tensor size without first checking its length (`src/gguf.zig`, lines 123–129; `src/infer_cuda.zig`, lines 44–48). A file with a valid tensor start and truncated payload can therefore pass parsing and fail at a later unchecked slice.

**Recommendation:** Compute each tensor's byte length from its dimensions and GGML type using overflow-safe arithmetic, validate the full range during parsing (or before any exact slice), and add a truncated-payload case whose tensor start is inside the file.

### R44-04 — Quarantine overflow still loses pinned-buffer ownership if emergency synchronization fails

**Severity: Medium — the failure path leaks an allocation while claiming it is retained.**

When all quarantine slots remain occupied after a drain attempt, `quarantinePinnedBuffer` tries to synchronize every device. If synchronization fails, it logs that the buffer is retained, then falls through to `buf.move()` without putting the pointer into any owner or registry (`src/infer_cuda.zig`, lines 794–822). This loses the only tracked owner of the pinned allocation. The earlier Round 43 lifecycle improvements make this an exceptional path, but do not make this full-table failure safe.

**Recommendation:** Keep ownership in a durable overflow owner or propagate a teardown failure without moving the buffer. Ensure the log describes the actual ownership outcome.

### R44-05 — Post-hoc T=2 temperature metrics do not represent T=2 serving at that temperature

**Severity: Medium — the calibration report's T=2 estimates do not match the runtime control.**

`tools/evaluate_calibration.py` optimizes temperature by rescaling saved logits. In serving, `engine.zig` applies the requested temperature to each pass and copies the resulting probabilities into `prev_probs` for self-conditioning (`src/engine.zig`, lines 275–292). For adaptive two-step inference, changing temperature can therefore change the second pass's input and logits, not just its final displayed probabilities. The reported post-hoc T=2 metrics do not establish the behavior of the `--temperature` setting at runtime.

**Recommendation:** Separate the final-output calibration temperature from the self-conditioning temperature, or rerun the full adaptive inference for each candidate serving temperature and report those end-to-end results.

### R44-06 — Temperature fitting and reported calibration metrics use the same labeled examples

**Severity: Medium — reported calibration improvements are in-sample.**

`optimize_temperature` searches for the temperature that minimizes NLL over all supplied labeled results and then reports calibrated metrics on that same collection (`tools/evaluate_calibration.py`, lines 125–205). The values in `specs/calibration-analysis.md` therefore show fit-set performance, not evidence that the selected temperatures generalize to unseen queries. Cross-evaluating the listed runs may be informative, but does not replace a held-out evaluation if each temperature was selected using the target set's labels.

**Recommendation:** Fit temperature on a calibration split and report metrics on a separate held-out split. Until then, label the reported gains as exploratory in-sample results and avoid presenting them as serving recommendations.

## Review Verdict

The latest work adds valuable parity policy, calibration analysis, canvas experiments, and CUDA graph paths, but several claims currently exceed what the checks establish. Fix the top-five parity gate, align the canvas study with its actual input layouts, and reject truncated tensor extents before model loading. Treat temperature recommendations as exploratory until calibration uses held-out data and the T=2 serving semantics are evaluated end to end. The pinned-buffer overflow path also needs an ownership-preserving failure policy.

## Verification Limits

The review inspected the five commits listed above, the relevant canvas, engine, GGUF, CUDA, and parity-verifier code, and the calibration and geometry documentation. No build, test, CUDA runtime check, reference comparison, or benchmark was performed.

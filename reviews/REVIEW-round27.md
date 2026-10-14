# Review Round 27 — Round 26 Fix Follow-up

**Reviewed revision:** `3f8cb17` (`fix(cuda,spec): resolve review round 26 findings R26-01 to R26-04`), compared with `b569064`.

**Scope:** Static review of the Round 26 corrective changes, including CUDA prompt-cache sizing, multi-step API behavior, self-conditioning documentation, and the new stepping/cache tests. No builds, tests, CUDA execution, or benchmarks were run for this review.

## Verdict

The CUDA prompt-cache stride and workspace accounting fixes address the corresponding Round 26 findings. The self-conditioning injection description now reflects the code's add-then-unweighted-RMSNorm order, and the API documentation was updated for `step_cap`.

Two contract details still need correction: the API promises a missing-tensor fallback that the backends do not provide, and the documented GeGLU equation reverses the gate/up operands relative to both the implementation and pinned vLLM source. The added stepping tests also do not establish that the engine actually performs or stops after a multi-step run.

## Findings

### R27-01 — Missing self-conditioning tensors do not trigger the documented single-step fallback

**Severity: Medium**

**Location:** `specs/decision-api.md:57`, `src/engine.zig:203-237`, `src/infer.zig:1102-1108`, `src/infer_cuda.zig:1472-1475`, `src/infer_cuda.zig:1609-1612`

The API now states that `step_cap > 1` falls back to one step if the model artifact lacks self-conditioning tensors. There is no such fallback in the engine. On CUDA, the second-step call reaches `applyDeviceSelfConditioning`, which returns `error.MissingSelfCondTensor` when any tensor is absent. On CPU, if `self_cond_pre_norm` is absent, the injection is skipped but the engine continues its loop, potentially repeating the same unconditioned forward pass and reporting multiple steps; if only some tensors are present, signal computation returns `MissingSelfCondTensor`.

**Recommendation:** Decide the intended behavior and implement it consistently. Either detect incomplete self-conditioning tensors before entering the loop and execute exactly one step, or return a documented error for `step_cap > 1`. Do not report multiple executed steps for repeated unconditioned passes as though self-conditioning occurred.

### R27-02 — GeGLU equation still swaps the gate and up branches

**Severity: Medium**

**Location:** `specs/denoising.md:89`, `src/ops.zig:120-127`

The specification defines $M = G \odot \text{GELU}_{\text{tanh}}(U)$. The implementation computes $M = \text{GELU}_{\text{tanh}}(G) \odot U$. The latter also matches the pinned vLLM implementation at [`diffusion_gemma.py` commit `1b3b88ec`](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/vllm/model_executor/models/diffusion_gemma.py), which applies GELU to `gate_proj(x)` and multiplies by `up_proj(x)`.

**Recommendation:** Correct the equation and make the `G`/`U` naming explicit in the surrounding text so future kernel changes preserve the branch order.

### R27-03 — New tests do not assert that the engine executes multi-step or stops early

**Severity: Low**

**Location:** `src/engine.zig:392-456`

The “multi-step adaptive stepping simulation logic” test computes entropy for manually supplied probability arrays; it does not invoke the `DecisionEngine` loop or the logic that propagates probabilities. The CUDA end-to-end test allows `steps_executed` to be either 1 or 2, so it can pass without entering step 2. It also does not assert the expected early-stop result.

**Recommendation:** Test the engine's step transition and stopping logic directly, or add a deterministic model-backed case that requires two steps and a second case that proves early termination. Keep backend numerical parity as a separate assertion.

## Previously Reported Findings

- **R26-01, prompt KV cache stride:** Addressed by increasing the per-layer cache stride to cover the maximum SWA K dimension and adding a long-prompt slot-isolation test.
- **R26-02, self-conditioning injection equation:** The documented add-then-unweighted-post-norm behavior now agrees with the implementation. The GeGLU operand-order mismatch in R27-02 remains.
- **R26-03, API step-cap documentation:** The supported range and response field are now documented. The missing-tensor behavior in R27-01 is still inaccurate.
- **R26-04, workspace accounting:** The table and asserted workspace total now include the larger cache and Phase 9 scratch buffers.

## Verification Notes

This is a static review. The commit message reports CPU/CUDA builds and oracle checks as passing, but those commands were not independently rerun here. The oracle checks establish the existing one-step decision fixtures; they do not by themselves establish multi-step sampler parity.

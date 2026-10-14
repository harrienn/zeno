# Review Round 26 — Phase 9 Multi-Step Denoising Follow-up

**Reviewed revisions:** `8d2c6b4..c702102`, including CUDA Phases 7–8 follow-up fixes and Phase 9 multi-step adaptive denoising/self-conditioning.

**Scope:** Static review of the changes in `src/engine.zig`, `src/decision.zig`, `src/infer.zig`, `src/infer_cuda.zig`, model tensor loading, the CUDA prompt KV cache, API/spec consistency, and the updated workspace budget. This review did not execute builds, tests, CUDA kernels, or benchmarks.

## Verdict

The implementation adds a useful multi-step experiment and appears to keep the CPU and CUDA self-conditioning paths structurally similar. The current changes should not yet be treated as a validated reference-compatible denoising implementation. One valid CUDA input can corrupt prompt KV cache data, and the implemented self-conditioning equations do not match the equations written in the Phase 9 specification. The API specification and workspace budget also need synchronization with the code.

## Findings

### R26-01 — Prompt KV cache stride is undersized for SWA layers

**Severity: High**

**Location:** `src/infer_cuda.zig:523-527`, `src/infer_cuda.zig:889-895`, `src/infer_cuda.zig:1113-1115`

The cache allocates a fixed per-layer stride of `512 * 1024` floats. However, SWA layers have `k_dim = 2048` (`8 KV heads * 256`), while global-attention layers have `k_dim = 1024` (`2 KV heads * 512`); see `specs/transformer.md:68-71` and `specs/gguf-mapping.md:107`.

For a valid CUDA request with a prompt longer than 256 tokens and a short canvas (for example, prompt length 257 and canvas length 16, total sequence length 273), an SWA cache record is larger than its slot. Its write overlaps the next layer's record. Later layer writes can overwrite the excess keys/values, so the cached attention read can combine data from different layers. The current short-prompt parity test does not exercise this boundary.

**Recommendation:** Allocate using the maximum K dimension (2048), or give each layer a correctly sized offset/record based on its actual K dimension. Add a cache parity case with a prompt over 256 tokens and a short canvas.

### R26-02 — Self-conditioning implementation differs from its documented equations

**Severity: High**

**Location:** `specs/denoising.md:88-94`, `src/infer.zig:1055-1058`, `src/infer.zig:1102-1108`, `src/infer_cuda.zig:1501-1536`

The Phase 9 specification defines `R = E_cand + D`, then `H_sc = RMSNorm(R)`, and injects `H_sc` into the answer embedding. Both implementations instead return only the down-projection `D`; they add it to the existing answer embedding and RMS-normalize that sum. Neither constructs the specified `E_cand + D` residual followed by its own RMSNorm.

Consequently, CPU/CUDA probability agreement only demonstrates agreement between these two implementations; it does not show that either implements the written contract or the pinned reference behavior.

**Recommendation:** Confirm the intended operation against the pinned reference and GGUF tensors, then align both backends and the equations. Until that is done, describe Phase 9 as an experimental candidate-restricted adaptation rather than reference parity.

### R26-03 — Public API contract still rejects multi-step requests

**Severity: Medium**

**Location:** `specs/decision-api.md:57`, `specs/decision-api.md:130`, `specs/decision-api.md:241`, `src/decision.zig:64`

The implementation accepts `step_cap` from 1 through 4, but the API field and error contract still say that only 1 is supported. The `DecisionResponse.steps_executed` comment also says it is always 1. `specs/denoising.md` §2.1 says multi-step policy remains disabled, while §6 documents the new implementation.

**Recommendation:** Update the normative API and response descriptions, including behavior when self-conditioning tensors are absent, and reconcile the initial policy text with the Phase 9 section. If multi-step remains experimental, identify that status explicitly in the API contract.

### R26-04 — Multi-GPU workspace budget omits Phase 9 scratch buffers

**Severity: Low**

**Location:** `specs/multi-gpu.md:27-41`, `src/infer_cuda.zig:728-731`

The spec reports 336,801,792 bytes, but the current `activationWorkspaceBytes(512)` assertion is 336,832,768 bytes. The 30,976-byte difference is exactly the two `embedding_dim` self-conditioning buffers and one `self_cond_intermediate_dim` buffer now included in the implementation. The table does not list those buffers either.

**Recommendation:** Refresh the total, table rows, and buffer count from the current workspace calculation.

## Readiness Notes

- The existing `step_cap` test checks request validation; no checked-in test exercises a multi-step CPU or CUDA request or self-conditioning numerics. The reported empirical run is useful evidence, but it is not an automated regression test.
- CPU/CUDA parity for Phase 9 is not the same as parity with vLLM's full sampler. The current path propagates candidate-restricted probabilities at one answer position, while the documented reference sampler operates over canvas positions and has re-noising/acceptance behavior.
- The review is based on source and specification inspection only. No tests, builds, or CUDA runtime checks were run.

## Overall Assessment

Phase 9 can continue as a clearly labeled experiment after correcting the CUDA cache layout and resolving the self-conditioning contract. Update the API/specification and workspace accounting before relying on the new public step-cap behavior or documenting the memory footprint as exact. Reference-sampler parity remains a separate validation gate.

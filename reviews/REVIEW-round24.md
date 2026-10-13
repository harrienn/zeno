# Review Round 24 — CUDA Round 23 Follow-up

**Reviewed revision:** `ae88da1` (`fix(cuda): resolve review round 23 findings R23-01 to R23-05`), including its changes to `src/infer_cuda.zig`, `src/cuda/ops.cu`, `src/engine.zig`, `src/main.zig`, `src/cuda_mem.zig`, `specs/multi-gpu.md`, and `PLAN.md`.

**Scope:** Static review of the full latest change set and the directly related CUDA workspace, decision-engine, benchmark, and specification contracts. No build, tests, CUDA execution, sanitizer run, or benchmark was performed.

## Verdict

This revision addresses the main issues from Round 23: capture/first-launch failures now trigger eager execution, direct peer access is attempted, the multi-GPU spec describes the CPU/GPU boundary, the CPU and CUDA sequence limits are checked separately, and the benchmark accepts repeated samples. The implementation is clearer as a result.

I would keep the Phase 6 measurements marked exploratory. The new benchmark still skips CPU warm-up, the CUDA graph fallback does not cover failures during cached graph replay, and the documented per-device workspace budget is below the amount allocated by the current workspace code. Peer-access setup errors and the new fallback tests also need clearer reporting/evidence before those paths can be considered verified.

## Findings

### R24-01 — Documented CUDA activation workspace is materially undersized

**Severity: Medium**

**Location:** `specs/multi-gpu.md:25-27`; `src/infer_cuda.zig:462-518`

The spec says each device's workspace is under 100 MB at `N = 512`, but `CudaDeviceWorkspace.init` allocates many full-sequence FP32 buffers. Summing the declared capacities at `max_seq_len = 512` gives approximately **211 MB (201 MiB) per device**, before CUDA runtime/graph allocations. This includes the two embedding buffers, the sequence-sized intermediate buffers, Q/K/V and attention outputs, and the batched expert buffers. The `exp_down_buf` alone accounts for about 46 MB, so the listed 46 MB formula is only one component of the workspace.

The current RTX 3060 headroom may still be adequate, but the budget is inaccurate and could mislead future device selection or capacity planning.

**Recommendation:** Recalculate the budget from all `DeviceBuffer` allocations at the configured maximum sequence length, state bytes versus MiB consistently, and keep the estimate tied to a small auditable table or helper that sums the same capacities.

### R24-02 — CUDA Graph recovery covers only finalization and first launch

**Severity: Medium**

**Location:** `src/infer_cuda.zig:973-982`, `src/infer_cuda.zig:955-960`, and `src/infer_cuda.zig:1023-1027`

The new eager rerun handles graph finalization and the first graph launch. It does not handle an operation failing while a graph is being built: each `runLayerCuda` call is `try`-propagated, so the function exits without ending or cancelling the capture. Once a graph is cached, the `graph_valid` branches also use `try graph.launch(...)`; a later replay failure returns an error without trying eager execution. The replay case fails visibly rather than returning stale output, but capture-body failures can also leave stream capture state uncleared for later requests.

**Recommendation:** Centralize capture cleanup and recovery so errors from operations inside capture, graph finalization, initial launch, and cached replay all leave the stream usable. For recoverable cases, invalidate/destroy the graph, reset buffer pointers, and execute the affected partition eagerly; otherwise return an error only after cleaning up capture state.

### R24-03 — CPU benchmark samples are reported as post-warmup without a CPU warm-up

**Severity: Low**

**Location:** `src/main.zig:548-551` and `src/main.zig:585-588`

The benchmark prints that samples are post-warmup, but the warm-up call is conditional on `use_cuda`. CPU samples therefore include the first cold request. This makes the report misleading and can skew CPU latency and speedup comparisons.

**Recommendation:** Warm up both backends before collecting samples, or report backend-specific warm-up behavior.

### R24-04 — Peer-access setup failures are silently converted to success

**Severity: Low**

**Location:** `src/cuda/ops.cu:641-655`; `src/infer_cuda.zig:623-628`

`zeno_cuda_enable_peer_access` returns success when peer capability detection fails, when the devices cannot access one another, and when `cudaDeviceEnablePeerAccess` fails for an unexpected reason. The Zig call site discards the return value as well. The copy path may still fall back to host staging if the later peer copy fails, but the runtime cannot distinguish a deliberately unsupported topology from a failed attempt to enable peer access. This also makes it difficult to verify claims that a transfer used direct P2P.

**Recommendation:** Return a distinct unsupported status for the normal no-P2P case and propagate genuine CUDA errors. Record which transfer path was selected in diagnostics or benchmark output.

### R24-05 — The new tests do not exercise the production fallback or sequence guard

**Severity: Low**

**Location:** `src/cuda_mem.zig:2831-2900`; `src/engine.zig:367-373`

The graph test captures a successful graph, discards it, then manually reruns RMSNorm and residual addition. It checks eager kernel parity, but does not induce a failure in `executeStructuredReadCuda` or prove its fallback branch executes. The CPU sequence test only checks that `EngineOptions.workspace_seq_len` stores 1024; it does not submit a request whose total sequence length exceeds 512.

**Recommendation:** Add a narrow injectable graph-launch/finalization failure seam and test the actual inference fallback. For the engine bound, test the validation logic directly or run a request with a controlled tokenizer/workspace so the over-512 path is exercised.

### R24-06 — New embedding and readout timing estimates lack a measurement source

**Severity: Low**

**Location:** `specs/multi-gpu.md:17`; `src/infer_cuda.zig:923-939` and `src/infer_cuda.zig:1137-1138`

The spec adds estimates of approximately 0.05 ms for host embedding work and 0.01 ms for candidate readout. The benchmark times the entire `DecisionEngine.decide` request and does not separately measure these operations, so the repo does not provide evidence for these component timings.

**Recommendation:** Cite a recorded profiler measurement with hardware, build, input size, and method, or remove the numbers and retain the qualitative description of the hybrid CPU/GPU boundary.

## Improvements confirmed

- Capture finalization and first-launch failures now attempt eager execution rather than proceeding with unexecuted captured work.
- The boundary transfer retains event-based cross-stream ordering for a successful peer copy and uses a synchronous host-staged path when the copy API reports failure.
- The multi-GPU diagram now matches the current CPU embedding and candidate-readout implementation.
- The global 512-token engine check is now backend-specific, and the CPU path uses its configured workspace limit.
- Benchmark output now includes repeated-sample median and range, and `PLAN.md` labels the recorded optimization measurements exploratory.

## Readiness assessment

| Area | Assessment |
|---|---|
| CUDA graph initial-capture failure handling | Improved in source; fallback path is not directly tested. |
| CUDA graph failure handling | Finalization/first-launch failures retry eagerly; capture-body and cached-replay errors lack equivalent cleanup/recovery. |
| Multi-GPU peer transfer | Event ordered for the direct-copy path; enablement errors are not observable; host staging remains synchronous. |
| Workspace VRAM estimate | Spec says under 100 MB; source allocations total about 211 MB (201 MiB) per device at sequence length 512. |
| CPU/CUDA sequence limits | Dispatch-specific checks are now in place; the new CPU test does not exercise request behavior. |
| Benchmark repeatability | Better harness; CPU lacks warm-up, and the recorded numbers remain exploratory rather than independently reproducible evidence. |
| External oracle parity | Not assessed or changed by this revision. |

## Verification notes

Static review only. No build, test, CUDA execution, compute-sanitizer run, or benchmark was performed. The worktree was clean before this review document was added.

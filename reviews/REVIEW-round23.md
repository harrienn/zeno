# Review Round 23 — CUDA Stage 6 Integration and Optimizations

**Reviewed revision:** `145a2aa` (`perf(cuda): implement Phase 6 kernel optimizations and CUDA Graph capture`), including `68ea120` (Stage 6 `DecisionEngine` integration and batched MoE).

**Scope:** Static review of the CUDA graph capture/replay path, peer transfer coordination, new device-side MoE kernels, GEMV batching, `DecisionEngine`/CLI integration, benchmark harness, and updated roadmap/spec claims. No build, tests, CUDA execution, sanitizer run, or benchmark was performed for this review.

## Verdict

This revision materially advances the backend: CUDA can now be selected through the engine and CLI, MoE routes and expert projections stay on the device, and the work adds 4-token GEMV batching and graph replay. The full CUDA path is now connected to the decision API.

I would not yet accept the CUDA Graph and multi-GPU claims as production-ready. The graph path silently continues after graph finalization fails, even though capture suppresses execution of the captured work. The nominally asynchronous boundary copy synchronizes both device streams before returning, and the runtime never enables peer access. The updated multi-GPU spec also describes token embeddings and candidate readout as GPU-resident, but the implementation still performs both on the CPU. Finally, the benchmark harness times one post-warm-up sample per case, which is not enough evidence for the repeatable latency claims recorded in `PLAN.md`.

## Findings

### R23-01 — Graph finalization failure is swallowed after captured work was not executed

**Severity: High**
**Location:** `src/infer_cuda.zig`, graph capture paths at lines 979–986, 1027–1031, and 1057–1064

When `beginCapture` succeeds, layer kernels are captured rather than executed. If `endCapture` fails while ending or instantiating the graph, the code ignores the error and continues to synchronize/copy output buffers. Since it never replays the captured work or reruns the layers normally, the request can use stale workspace data as if inference succeeded. The same pattern exists for both single-device and dual-device partitions.

**Recommendation:** Treat graph finalization failure as an execution failure, or rerun that partition outside capture before returning a result. Add an injected/forced capture-failure test proving that the fallback produces the same output as ordinary execution.

### R23-02 — Boundary transfer still blocks the host and does not enable direct peer access

**Severity: Medium**
**Location:** `src/infer_cuda.zig`, `transferHiddenState`, lines 866–894; `src/cuda/ops.cu`, `zeno_cuda_memcpy_peer_async`

After enqueueing `cudaMemcpyPeerAsync`, the helper immediately synchronizes both the source and destination streams. The host therefore waits for the transfer and cannot overlap it with subsequent work. The host-staged fallback also uses blocking `cudaMemcpy` calls. In addition, the CUDA runtime only queries `cudaDeviceCanAccessPeer`; it never calls `cudaDeviceEnablePeerAccess`. NVIDIA documents that enabling peer access is what allows peer copies to avoid host staging, so a capable topology alone does not establish the direct-transfer path ([CUDA Programming Guide: multi-GPU peer transfers](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)).

**Recommendation:** Enable peer access for the selected device pair when supported, preserve stream/event ordering without immediately synchronizing both streams, and label/measure the host-staged fallback separately. If synchronization is intentionally retained for correctness, do not claim host-nonblocking transfer or overlap.

### R23-03 — The multi-GPU spec overstates GPU residency and host-free execution

**Severity: Medium**
**Location:** `specs/multi-gpu.md`, lines 9, 18–19, and 35–59; `src/infer_cuda.zig`, lines 918–945 and 1076

The spec says token embeddings and candidate readout are placed on the GPUs and describes the pipeline as zero-host-intervention. The implementation dequantizes token embeddings on the CPU before copying activations to Device 0, then computes candidate logits on the CPU after copying the normalized answer vector back. Device-side routing removes the earlier MoE round trips, but the complete inference request is still a hybrid CPU/GPU path. This also means the listed per-device memory budget does not describe the actual placement.

**Recommendation:** Document the current hybrid boundaries and recalculate the memory budget from actual device allocations. Reserve “host-free” for the layer execution interval, or move embedding and candidate readout to CUDA before making an end-to-end claim.

### R23-04 — The engine applies the CUDA 512-token limit to every backend

**Severity: Medium**
**Location:** `src/engine.zig`, lines 185–188; `EngineOptions.workspace_seq_len`, lines 21–25; `specs/decision-api.md`, lines 153 and 246

`decide` rejects requests above 512 tokens before backend dispatch, including CPU requests. This overrides the configurable workspace length and conflicts with the API spec, which documents support up to 1024 total sequence tokens and 768 prompt tokens. The strict 512 ceiling is a CUDA attention-kernel restriction; it should not silently become a shared CPU/API contract.

**Recommendation:** Apply the 512 check only to CUDA (or derive the bound from the selected backend and allocated workspace), and reconcile the public request limits with the actual backend capacities.

### R23-05 — Recorded benchmark claims exceed the harness's sampling method

**Severity: Low**
**Location:** `src/main.zig`, benchmark path, lines 530–565; `PLAN.md`, lines 249–257

The CUDA benchmark warms each case once and records one timed request for the baseline and each canvas width. It reports no repeated samples, distribution, device/build metadata, or paired-run procedure, while `PLAN.md` calls the results reproducible benchmark evidence and records precise before/after latency changes.

**Recommendation:** Run repeated timed samples after warm-up, report a median and spread, and record GPU model, clocks/power mode where available, build revision/options, and exact invocation. Until then, label the numbers exploratory single-run measurements.

## Confirmed improvements

- `DecisionEngine` now has explicit CPU/CUDA backend selection, with CLI options for `decide`, `decide-fixture`, `benchmark`, and `serve`.
- The earlier per-layer host copies of MoE routing indices and weights have been removed; the new projection, activation, down-projection, and weighted accumulation kernels operate on device buffers.
- The 4-token GEMV kernels retain a separate path for batches smaller than four and guard tail tokens in larger batches.
- The engine checks total CUDA sequence length before entering the attention path.
- The five CUDA fixture checks reuse the existing versioned comparison policy; oracle numerical divergence remains separately open in `specs/UNRESOLVED.md` U-13/U-22.

## Readiness assessment

| Scope | Status |
|---|---|
| CUDA backend API/CLI integration | Present in source. |
| Batched device-side MoE | Implemented in source; no tests or benchmarks were run in this review. |
| CUDA Graph replay | Implemented, but failure handling under R23-01 can return invalid results. |
| Multi-GPU boundary transfer | Has async API and event ordering, but immediate stream synchronization prevents host overlap; direct peer enablement is absent. |
| End-to-end host-free execution | Not implemented; embeddings and candidate logits remain CPU operations. |
| CPU/CUDA sequence contracts | Inconsistent with the documented 1024-token API bound under R23-04. |
| CUDA-to-CPU fixture agreement | Project reports 5/5 fixture decisions and policy thresholds; not rerun in this review. |
| External llama.cpp raw-logit parity | Still open under U-13/U-22. |
| Performance claims | Not independently verified; benchmark sampling needs stronger repeatability evidence. |

## Verification notes

Static review only. No build, test, CUDA execution, compute-sanitizer run, or benchmark was performed. The source tree was clean before this review file was added.

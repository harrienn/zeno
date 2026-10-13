# Review Round 22 — CUDA Stage 4 and Stage 5

**Reviewed revision:** `dc499ce` (`feat(cuda): implement Stage 5 execution pipeline and multi-gpu layer partitioning`), including `e257bcb` (Stage 4 attention and MoE) and `da82163` (Round 21 fixes).

**Scope:** Static review of the new CUDA attention and MoE kernels, `src/infer_cuda.zig`, multi-GPU transfer path, related tests, and stage documentation. No build, tests, CUDA execution, sanitizer run, or benchmark was performed for this review.

## Verdict

This is a substantial implementation step. The CUDA layer path follows the CPU reference's central operations and order: Q/K/V projection and normalization, prompt-causal/canvas-bidirectional attention, dense and MoE branches, and dual-region output scaling. The source includes one-layer checks for an SWA layer and a global-attention layer, plus a full-model comparison test.

The implementation is not yet ready to accept as the Stage 5 exit gate or as an application CUDA backend. The MoE path synchronizes and transfers routing decisions to the host at every layer, then launches expert work serially per token. The boundary transfer is also synchronous. These choices conflict with the roadmap's batched expert and no-host-intervention goals. The execution helper is not connected to `DecisionEngine` or CLI/backend selection, which the roadmap leaves to Stage 6. The checked-in full-model test is a useful smoke/parity check, but its single two-candidate case and 0.5 logit tolerance do not establish fixture-suite parity.

## Findings

### R22-01 — Per-layer host routing and serial expert launches violate the Stage 4/5 execution contract

**Severity: High**
**Location:** `src/infer_cuda.zig`, `runLayerCuda`, lines 761–798

After computing top-8 routes on the GPU, each layer synchronizes its stream, copies indices and weights to the CPU, loops over every token and selected expert on the CPU, and launches separate expert gate/up, GeGLU, down, and accumulation operations. This creates a host round trip for every transformer layer and up to eight serial expert evaluations per token. The Stage 4 roadmap calls for batched active-expert execution; Stage 5's exit gate says inference should proceed without host intervention.

**Recommendation:** Keep route selection and expert dispatch device-side, and batch work by active expert or another measured GPU-friendly grouping. If a host-driven prototype is intentional, revise the stage gate and make its latency/throughput limitations explicit before claiming Stage 5 complete.

### R22-02 — Multi-GPU boundary transfer is synchronous, not the specified asynchronous stream transfer

**Severity: Medium**
**Location:** `src/infer_cuda.zig`, `transferHiddenState`, lines 831–862; `src/cuda/ops.cu`, `zeno_cuda_memcpy_peer`

The peer path calls synchronous `cudaMemcpyPeer`; the fallback uses synchronous device-to-host and host-to-device `cudaMemcpy` calls. The transfer helper takes no stream and has no cross-device event coordination. This can preserve data correctness, but it does not implement the roadmap's asynchronous P2P and stream-coordination deliverable, and it blocks the host between the two layer partitions.

**Recommendation:** Use a peer-async transfer tied to explicit source/destination stream ordering, with a similarly explicit asynchronous host-staging fallback. Measure each path on the supported topology and update the gate to match the tested capability.

### R22-03 — The advertised sequence limit exceeds the attention kernel's accepted range

**Severity: Medium**
**Location:** `src/cuda/attention.cu`, `zeno_cuda_attention`, line 237; `src/infer_cuda.zig`, `executeStructuredReadCuda`, lines 880–883; `specs/cuda-roadmap.md`, line 116

The roadmap describes the Stage 5 transfer path for `N <= 1024`, and the orchestration checks only that `N` fits the allocated workspace. The attention ABI rejects `num_tokens > 512`. A request with a larger workspace and `512 < N <= 1024` therefore passes the orchestration check but fails when the first layer invokes attention.

**Recommendation:** Either raise the kernel's supported limit safely (including its fixed shared-memory arrays and launch resource use), or make the 512-token limit explicit and enforce it at the top-level CUDA entry point. Align the roadmap's supported range with the implementation.

### R22-04 — The full-model test is too permissive to enforce the reported numerical result

**Severity: Medium**
**Location:** `src/infer_cuda.zig`, full-model test, lines 1320–1408

The test covers one locally constructed prompt/canvas and two candidates. It asserts that the winning candidate matches and that each raw and softcapped logit differs by at most `0.5`; the commit message reports a maximum difference of `0.0003`. The test does not assert that reported tighter bound, evaluate the golden fixture suite, or compare against an independently captured CUDA/reference fixture. A large numerical regression could still pass while preserving the same winner.

**Recommendation:** Assert the documented measured tolerance in the test, add the intended canvas/prompt boundary cases, and run the CUDA path through the golden fixture suite once Stage 6 backend selection is available. Keep CPU parity, discrete-decision agreement, and external-oracle parity as separate claims.

### R22-05 — Multi-GPU specification still says no GPU path exists

**Severity: Low**
**Location:** `specs/multi-gpu.md`, line 5; `specs/cuda-roadmap.md`, Stages 4–6

The multi-GPU spec still describes the GPU path as nonexistent and the layer split as only a candidate, while the new code implements a two-device split and the roadmap describes Stage 5 as complete. At the same time, the code does not meet the roadmap's asynchronous/no-host-intervention transfer gate. The documents currently cannot tell a reader which parts are implemented, which are experimentally verified, and which acceptance criteria remain open.

**Recommendation:** Update the multi-GPU spec with the implemented topology and actual transfer/host-routing behavior, preserving the distinction between source implementation and validated support. Record the remaining Stage 5 gate items and Stage 6 integration status in the roadmap.

## Confirmed improvements

- CUDA attention implements the same prompt-causal and canvas-bidirectional visibility rule as the CPU path.
- The CUDA layer path carries the global-layer V-from-K behavior and per-head V normalization used by the CPU implementation.
- Layer 0 and layer 5 GPU tests compare against the CPU implementation, covering both attention variants.
- The workspace sizes activation buffers by the configured maximum sequence length, including MoE intermediates.
- The earlier Round 21 connection reservation, shutdown-drain, and GEMV grid-Y fixes are present in `da82163`; they were not runtime-tested here.

## Readiness assessment

| Scope | Status |
|---|---|
| Stage 4 attention and layer-level CUDA source | Implemented; CPU comparison tests are present but were not rerun. |
| Stage 4 batched MoE execution | Not met; current dispatch copies routes to the host and launches experts serially. |
| Stage 5 layer partitioning | Implemented in source for the first two reported devices. |
| Stage 5 asynchronous/no-host-intervention gate | Not met by the synchronous boundary copy and host-driven MoE path. |
| Full-model CPU comparison | A gated test exists for one local model/input; the recorded project result was not independently verified in this review. |
| CUDA backend in `DecisionEngine` / CLI | Not integrated; Stage 6 remains. |
| Golden fixture and external oracle parity | Not established by this implementation or review. |
| Performance and multi-GPU benefit | Not established; no benchmark was run. |

## Verification notes

Static review only. No build, test, CUDA execution, compute-sanitizer run, or benchmark was performed. The working tree was clean before this review file was added.

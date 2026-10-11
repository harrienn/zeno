# Review Round 10 — CUDA Backend Readiness

**Reviewed revision:** `7ac16f9` (`feat(server): address Codex review round 9 findings`), compared with `a83ec3c`.

**Scope:** Readiness of the current project and CUDA roadmap for beginning CUDA backend work. Static review only; no build or tests were run.

## Verdict

The project is **not ready to begin the CUDA backend under its current implementation plan**. `PLAN.md` says to avoid parallel architectural changes while the current parity objective remains unresolved, and the CUDA roadmap still has material contract errors. A device-discovery/toolchain spike could be run as an explicitly scoped research exception, but that is different from the project being ready to start backend implementation.

The repository already has a CPU execution path and model tensor inventory, and the CUDA roadmap proposes useful early steps. However, it is still a draft, contains concrete tensor and validation-script errors, and conflicts with the current parity gate and execution plan. The latest Antigravity changes address API server behavior only; they do not provide new CUDA implementation or readiness evidence. There is no CUDA source or CUDA build integration in `src/` or `build.zig` yet.

## Findings

### R10-01 — GEMV roadmap assigns Q5_0 to the wrong projection and omits Q8_0

**Severity: High**  
**Location:** `specs/cuda-roadmap.md:90-95`; `specs/gguf-mapping.md:75-80, 115-120`

Stage 3 calls for `gemv_q5_0.cu` for the expert down projection. The model inventory identifies `blk.{i}.ffn_down_exps.weight` as **Q8_0**, and the shared `ffn_down.weight` is Q8_0 as well. The listed Q5_0 tensor is `self_cond_down.weight`, which is Phase 4+ and bypassed by the current one-step read under U-18. The current GEMV list therefore misses a quantized matrix operation needed by the active path and assigns a kernel to the wrong tensor.

**Recommendation:** Derive the Stage 3 kernel list from the active-path tensor inventory, including Q8_0 expert/shared down projections. Keep Q5_0 explicitly scoped to the later self-conditioning path unless another active Q5_0 matrix is identified.

### R10-02 — The maximum layer-boundary transfer estimate is understated

**Severity: Medium**  
**Location:** `specs/cuda-roadmap.md:52, 115`; `specs/decision-api.md:153`

The roadmap uses `N = prompt_len + 256` and claims approximately 2.88–5.77 MB for FP32 activations. The API contract permits up to 768 prompt tokens, so the maximum sequence extent is 1,024. At `[N, 2816]` FP32, that is 11,534,336 bytes (about 11.0 MiB), twice the stated upper estimate. The smaller estimates correspond to shorter sequence lengths and should not be used as the maximum memory or transfer budget.

**Recommendation:** State the supported range and compute transfer/workspace budgets at `N = 1024`; distinguish decimal MB from MiB. Treat actual latency as unmeasured until benchmarked.

### R10-03 — The documented CUDA parity command is not supported by the script

**Severity: High**  
**Location:** `specs/cuda-roadmap.md:119-123`; `tools/verify_all_oracles.sh:7, 69-80`

The Stage 6 command `./tools/verify_all_oracles.sh --backend=cuda` is not implemented. The script treats its first argument as the model path, runs `decide-fixture` through the CPU engine, and has no backend selector. As written, the documented command interprets `--backend=cuda` as a GGUF path and fails before selecting a backend.

**Recommendation:** Define and implement an explicit backend selection interface before making this an exit-gate command. Ensure the verifier reports which engine actually ran, so a CPU pass cannot be mistaken for CUDA validation.

### R10-04 — The CUDA execution plan conflicts on parity status and multi-GPU decomposition

**Severity: High**  
**Location:** `PLAN.md:157-159, 203-232, 334-342`; `specs/UNRESOLVED.md:11, 32-45`; `specs/cuda-roadmap.md:35-60, 107-117`; `specs/multi-gpu.md:1-18`

`PLAN.md` says every U-01–U-23 gate is resolved, while the unresolved register still marks U-13 and U-22 open because measured raw-logit differences exceed the current tolerance. The Plan's current objective also still says to close U-05, U-13, and U-22 even though U-05 is resolved. Separately, Phase 5 describes multi-GPU tensor parallelism, while the CUDA roadmap and multi-GPU spec describe a candidate pipeline layer split and explicitly say no Zeno GPU path or validated transfer plan exists.

The open CPU/reference numerical parity does not by itself prevent a CUDA prototype from matching the CPU implementation. It does prevent claiming external-reference parity, and the contradictory Plan makes the intended gate for beginning integrated backend work unclear.

**Recommendation:** Reconcile the phase gates and current objective against `UNRESOLVED.md`. Choose one multi-GPU decomposition as an experiment only after single-GPU execution and measured memory use; do not present tensor parallelism and layer splitting as the same plan. Preserve the distinction between CPU-equivalent kernel validation and llama.cpp/vLLM parity.

### R10-05 — The roadmap assumes the target hardware and deployment shape before validating them

**Severity: Medium**  
**Location:** `specs/cuda-roadmap.md:4-5, 17, 69-76`; `specs/multi-gpu.md:1-18`

The roadmap's first exit gate requires two RTX 3060 devices with P2P confirmed, but the multi-GPU spec says runtime memory fit and decomposition are unvalidated. The roadmap also combines a fixed local two-device gate with a broad goal of configurable single- and multi-GPU support. Those are separate scopes and should have separate evidence and acceptance criteria.

**Recommendation:** Make Stage 1 report the actual visible device count, capabilities, memory, and peer-access matrix without making dual-GPU availability a prerequisite for developing single-GPU kernels. Keep single-device bring-up as the first backend gate; gate multi-GPU work on measured fit and transfer behavior.

## Implementation language recommendation

For the first production CUDA path, use **Zig for the host engine and CUDA C++ for GPU kernels**, connected through a small C ABI. Keep request handling, inference orchestration, ownership, device selection, and the CPU reference implementation in Zig. Put CUDA-specific kernels behind explicit C-compatible entry points so the engine does not depend on CUDA C++ types or templates.

This is a project-specific risk choice: Zeno has an established Zig CPU path, but no CUDA build integration yet, while the planned kernels include quantized GEMV and attention. NVIDIA's `nvcc` is the standard compiler driver for CUDA C/C++ and PTX; the LLVM NVPTX backend supports a defined subset of LLVM IR and needs the host-side CUDA Driver API or another runtime path to load and launch kernels ([NVIDIA `nvcc` guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html), [LLVM NVPTX guide](https://llvm.org/docs/NVPTXUsage.html)). Using CUDA C++ for initial kernels reduces toolchain risk while leaving the engine and public project structure Zig-first.

Do not commit the whole backend to Zig + NVPTX64 before a representative prototype demonstrates reliable compilation, inspectable generated code, correct CPU comparisons, and a concrete maintenance or performance benefit. Zig-generated PTX remains a reasonable later option for selected kernels, especially simple operations where direct control is useful. Compare implementations with the same inputs, tolerances, and hardware before choosing per-kernel language boundaries; language choice alone does not establish performance.

This recommendation does not require converting the entire CUDA roadmap to C++ or introducing a broad abstraction layer. Keep the C ABI narrow, retain a CPU-only build when CUDA is unavailable, and make CUDA compilation an explicit build option.

## Readiness split

| Work | Readiness | Reason |
|---|---|---|
| CUDA compiler/device discovery and build-flag prototype | Possible as a separate, bounded research spike | It would be an exception to the current Plan rule against parallel architectural changes; preserve a CPU-only build path. |
| Small elementwise/GEMV kernel experiments against CPU operations | Defer until scope and operation contracts are corrected | First correct the quantized type list and document tensor layouts/dtypes; keep each experiment independently reviewable. |
| Integrated single-GPU model execution | Not yet specified well enough for a dependable implementation | Need active-path operator/tensor coverage, ownership and residency boundaries, and consistent phase gates. |
| Multi-GPU execution and latency claims | Not ready | No validated Zeno GPU path, memory measurements, or chosen decomposition; current targets are roadmap hypotheses. |
| CUDA reference parity claim | Not ready | U-13/U-22 remain open, and the documented CUDA verification command does not select CUDA. |

## Recommended next steps

1. Correct the GEMV type inventory and maximum activation-transfer calculation.
2. Reconcile `PLAN.md` with `UNRESOLVED.md`; explicitly state whether CPU-equivalent CUDA bring-up may proceed while external numerical parity remains open.
3. Close or deliberately reprioritize the current Plan objective before treating CUDA as active implementation work. If an early spike is chosen, scope it to device/toolchain discovery, with the CPU-only build unaffected when CUDA is unavailable.
4. Defer multi-GPU architecture and performance claims until single-GPU correctness and actual memory use are measured.
5. Add a backend-aware oracle/fixture path before using it as a CUDA exit gate.

## Verification notes

This review was based on repository inspection. No compiler, build, or test commands were run. The reported sequence-size estimate is calculated from the documented maximum of 768 prompt tokens plus a 256-token canvas at 2,816 FP32 values per token.

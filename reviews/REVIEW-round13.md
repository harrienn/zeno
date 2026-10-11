# Review Round 13 — Stage 1 CUDA Implementation Readiness

**Reviewed revision:** `dd2b425` (`docs(cuda): address Codex review round 12 findings`), compared with `b07ed51`.

**Scope:** Static follow-up review of the latest CUDA roadmap and Stage 1 RFC changes, with a readiness assessment for beginning implementation. No builds or tests were run.

## Verdict

**Stage 1 (CUDA build integration and device discovery) is ready to implement.** The latest changes align the roadmap and RFC on direct object-file linkage, make initial current-device query failure explicit, report a dedicated error when successful-path restoration fails, and correct the earlier acceptance and topology issues. The one remaining device-restoration caveat below should be stated in the API contract, but it does not block beginning this bounded spike.

This does **not** mean the full CUDA inference backend is ready or validated. Stages 2–6 still require implementation and CPU comparisons; the multi-GPU design remains gated on actual device availability, memory measurements, and transfer validation. External llama.cpp numerical parity also remains open under U-13/U-22, separately from CUDA-to-CPU semantic validation.

## Finding

### R13-01 — Restoration on query-error paths is best-effort and its failure is discarded

**Severity: Low**
**Location:** `specs/cuda-stage1-rfc.md:169-205`

The successful path now returns `ZENO_CUDA_ERR_RESTORE_DEVICE_FAILED` if restoring the caller's device fails. On property, set-device, memory, or peer-query errors, however, the code casts the restoration result to `void` and returns only the primary error. For failures after switching devices, callers may therefore receive an error while the current device remains changed, with no status indicating that restoration also failed. The code comments call this best-effort, but the Zig API/error contract does not describe the resulting state guarantee.

**Recommendation:** Document that a primary query failure may leave the current device changed if best-effort restoration also fails, or add a combined/secondary restoration status. A simple precedence rule is sufficient for this Stage 1 utility.

## Confirmed Round 12 resolutions

- The roadmap and RFC now consistently compile `zeno_cuda.o` and attach it directly to the executable.
- Failure to query the current device now returns an explicit error rather than substituting device 0.
- Successful-path restore failure now has its own error code and error string.
- The roadmap and RFC agree on visible versus reported device counts, P2P capability terminology, the single-device discovery gate, and provisional transfer latency.
- `PLAN.md` now distinguishes exact discrete decisions/uncertainty flags from open numerical reference-logit parity.

## Readiness split

| Scope | Readiness |
|---|---|
| Stage 1 build flag, nvcc compilation, C ABI, and device discovery | Ready to implement; preserve the default CPU-only build and validate each stated gate. |
| CUDA elementwise and quantized kernels | Not yet validated; begin only in their later stages with per-operation contracts and CPU comparisons. |
| End-to-end single-GPU inference | Not yet implemented or verified. |
| Multi-GPU inference and performance claims | Still gated on measured memory fit, topology, P2P behavior, and benchmarks. |
| External reference numerical parity | Still open under U-13/U-22; CUDA-to-CPU agreement does not close it. |

## Verification notes

This was a documentation and contract review only. No compiler, build, or tests were run.

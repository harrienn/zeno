# Review Round 39 — Phase 14 CUDA Optimizations

**Reviewed revisions:** `ce45444` (`docs(review): remediate Review Round 38 findings R38-01 through R38-04`) and `276bf2f` (`Implement CUDA Phase 14: MoE shared-memory activation staging & pinned boundary DMA`).

**Scope:** Review of Phase 14's shared-memory activation staging, pinned-host cross-GPU transfers, benchmark interpretation, and repeatability claims. No implementation changes were made during this review.

## Findings

### R39-01 — Pinned staging memory can be reused or released before queued transfers are known to be complete

**Severity: High — make the pinned-buffer lifetime explicit across success, error, reuse, and teardown paths.**

`transferHiddenState` enqueues an asynchronous device-to-host copy into the shared pinned staging buffer, records an event on the source stream, makes the destination stream wait for that event, and then enqueues the host-to-device copy. The function can return while the destination stream is still consuming the host buffer. The normal inference paths synchronize both device streams later, but that synchronization is not guaranteed on every error return.

There are two concrete lifetime boundaries to address:

- `CudaWorkspace.deinit()` calls `host_staging_pinned.deinit()` before destroying the device streams and without first draining the work that may still reference the buffer (`src/infer_cuda.zig`, `CudaWorkspace.deinit`).
- `prefillPromptCuda` writes new embedding data into `host_staging_buf` before calling `ws.synchronize()` (`src/infer_cuda.zig`, `prefillPromptCuda`). If a prior request returned through an error path after queuing the asynchronous transfer, the next request can overwrite the staging bytes while the destination H2D copy still reads them.

**Recommendation:** Track which streams may reference the pinned buffer and drain them before host-buffer reuse and before freeing it. Ensure cleanup also drains queued transfers on error paths. Keep the ownership and synchronization requirement visible in the workspace API so later callers cannot accidentally reuse the buffer early.

### R39-02 — The activation-cache and synchronization analysis is arithmetically and structurally inaccurate

**Severity: Medium — correct the footprint and barrier accounting, then profile before explaining the result.**

`specs/denoising.md` §14.3 says the C=16 activation footprint is 176 KB and fits in the RTX 3060's 128 KB per-SM L1. At the stated dimensions, `16 × 2816 × 4 = 180,224` bytes, or 176 KiB, which is larger than 128 KiB. The whole activation tensor therefore cannot fit in that L1 capacity as claimed. Cache behavior may still help individual accesses, but this calculation does not establish that explanation.

The same section attributes the kernel to 11 `__syncthreads()` calls. Static inspection of `src/cuda/moe.cu` shows 11 barriers after staging a K-superblock's activation tiles, plus 10 barriers between consuming a tile and overwriting it for the next row group, and an additional barrier at the batch boundary. The barriers protect different producer/consumer phases, so the count and their cost should be described from the actual loop structure.

**Recommendation:** Correct the cache-capacity comparison and count barriers from the implementation. Treat the cache and synchronization explanation as a hypothesis until profiler evidence supports it; report measured latency separately.

### R39-03 — The benchmark does not isolate the two Phase 14 changes, and the stated improvements are not consistent across canvas widths

**Severity: Medium — qualify the conclusion and measure each optimization independently.**

Phase 14 changes both MoE activation staging and cross-GPU pinned asynchronous DMA. The table in `specs/denoising.md` §14.2 compares combined Phase 14 timings with Phase 13, so it cannot attribute an observed difference to either change.

The data also does not support the §14.3 statement that the two changes yield consistent improvements at intermediate canvas widths. C=32 improves by 1.01 ms against a 0.55 ms reported spread, C=64 improves by 8.44 ms but has a 39.53 ms spread, and C=16 is effectively unchanged. C=128 and C=256 are slightly slower. These results are useful preliminary observations, but the C=64 difference is especially noisy and the paired changes are confounded.

**Recommendation:** Present the current results as preliminary end-to-end measurements. Use controlled comparisons that change one optimization at a time, retain individual samples and run metadata, and repeat enough runs to assess the observed spread before claiming a stable gain.

### R39-04 — Five runs on one fixture do not establish the broadly worded determinism guarantees

**Severity: Medium — narrow the contract to measured evidence or broaden the evaluation.**

`specs/denoising.md` §13.3.1 measures five process executions of one incident-severity fixture, then states guaranteed bit-exact discrete decisions and global tolerance bounds for probabilities, entropy, and logits. That experiment supports repeatability for the measured fixture and configuration; it does not establish the same guarantee for all inputs, especially near decision ties where schedule-dependent floating-point atomic accumulation can affect the selected class.

**Recommendation:** Either scope the wording to the tested fixture, model, hardware, and execution configuration, or evaluate a broader corpus that includes near-tie cases and define the guarantee from those results. Keep observed measurements distinct from the intended contract.

## Positive Changes

- The shared-memory path stages activation data cooperatively and uses vectorized loads. The inspected tail and partial-batch handling appear coherent from static review.
- The cross-device transfer now uses pinned host memory with a source-stream event and destination-stream wait, which expresses the intended D2H-to-H2D dependency without a host-side wait in the transfer function.
- Round 38's corrected logical intermediate-traffic arithmetic and qualified L2 scheduling explanation are reflected in the updated documentation.

## Review Verdict

Phase 14 is a useful implementation step, but the pinned-buffer lifetime needs an explicit correctness fix before relying on asynchronous transfers across failed requests or workspace teardown. The performance and determinism conclusions should also be narrowed to match the evidence. The static review did not identify an obvious indexing error in the shared-memory staging path.

## Verification Limits

The review inspected the Phase 14 implementation, transfer and workspace lifecycle paths, the associated CUDA kernel, benchmark summaries, and relevant determinism documentation. The C=16 footprint was recomputed from the documented dimensions. No CUDA compilation, build, tests, profiler run, or benchmark was performed during this review.

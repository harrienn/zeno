# Review Round 21 — CUDA GEMV, Canvas Prefix, and Server Concurrency

**Reviewed revision:** `a6266d2` (`feat(cuda): implement native quantized GEMV kernels (Stage 3)`), including `253f2a9` (server concurrency) and `f8d3a29` (shortened canvas prefix).

**Scope:** Static review of the recent CUDA GEMV kernels and wrappers, canvas-prefix API and benchmark, and detached HTTP worker lifecycle. No builds, tests, CUDA runs, sanitizer checks, or benchmarks were run during this review.

## Verdict

Stage 3 adds a useful low-level CUDA GEMV foundation for Q8_0, Q4_K, Q6_K, and Q5_0, with matching CPU reference functions and synthetic parity tests. The backend is not yet integrated into `DecisionEngine`, so these kernels do not establish end-to-end CUDA inference or performance gains.

The shortened canvas prefix is opt-in and leaves the default at 256, which limits the behavior change. The reported C=16 agreement is measured against the local C=256 CPU baseline over five fixtures; it does not establish numerical parity or broad decision stability.

The server concurrency change has two lifecycle issues that should be fixed before relying on its 64-connection bound or graceful shutdown behavior.

## Findings

### R21-01 — Active-connection limit is racy during connection bursts

**Severity: High**
**Location:** `src/server.zig`, `serve` and `workerConnectionHandler`

The accept loop checks `active_connections` before spawning a worker, but the counter is incremented only after the detached worker starts. Several accepted connections can therefore all observe a count below 64 and spawn workers before any increment occurs. Under a burst, the server can exceed its advertised limit and create an unbounded number of threads relative to that limit.

**Recommendation:** Reserve a connection slot atomically in the accept thread before spawning. If thread creation fails, release the reservation before continuing. The worker should release the already-reserved slot on exit rather than acquiring it after it starts.

### R21-02 — Timed shutdown can deinitialize state still used by detached workers

**Severity: High**
**Location:** `src/server.zig`, `stop`, `deinit`, and detached connection workers

Workers retain pointers to `Server` and its `DecisionEngine`, but `stop()` waits at most 500 ms and then returns even if requests remain active. Inference requests can take longer than that. A caller that deinitializes the server and engine after this timeout can leave a worker accessing freed state. The counter race in R21-01 also means shutdown can observe zero before a newly spawned worker increments it.

**Recommendation:** Track accepted workers before they can run and join/drain them before destroying `Server` or `DecisionEngine`. If shutdown must have a deadline, define a safe cancellation or ownership-transfer path for workers that outlive it; do not return from deinitialization while they still reference these objects.

### R21-03 — GEMV batch dimension can exceed CUDA grid-Y limits

**Severity: Medium**
**Location:** `src/cuda/gemv.cu`, each `zeno_cuda_gemv_*` launch; `src/cuda_mem.zig`, `gemvQ8_0`, `gemvQ4_K`, `gemvQ6_K`, and `gemvQ5_0`

The Zig wrappers accept any positive `batch_size` up to `c_int`, then the C ABI uses `batch_size` directly as `grid.y`. CUDA devices have a much smaller grid-Y limit (65,535 on the current target). A representable and allocatable batch above that limit passes wrapper validation but fails at kernel launch.

**Recommendation:** Validate the launch limit and return `error.InvalidParam`, or flatten batch and output-row indexing into a grid dimension with a supported range. Add a boundary case to the wrapper tests.

### R21-04 — Canvas-prefix timing results are preliminary single-run measurements

**Severity: Low**
**Location:** `src/main.zig`, `benchmark-canvas`; `specs/canvas-templates.md` §6

The benchmark measures each canvas width once, in a fixed order, with the C=256 baseline first. The five-fixture results are useful exploratory evidence, but a single sample per case is sensitive to warm caches, CPU scheduling, and thermal state. Also, C=16's recorded agreement is against a local C=256 run, not an independent oracle execution.

**Recommendation:** Label the current numbers preliminary. For a performance claim, warm up, repeat runs, vary or alternate measurement order, and report a distribution. Keep the comparison explicitly scoped to local C=256 decisions unless external-reference results are captured separately.

## Confirmed improvements

- The CUDA code represents the four quantized block layouts with compile-time size assertions.
- Zig GEMV wrappers validate dimensions, quantization-block divisibility, buffer lengths, and device/stream affinity before calling the C ABI.
- Tests cover each format against its CPU reference and include a batched Q8_0 case, although the shapes are small and synthetic.
- `DecisionRequest.canvas_length` is opt-in and defaults to the existing 256-token canvas.
- The `/health` handler can now be served independently from a connection that is executing inference.

## Readiness assessment

| Scope | Status |
|---|---|
| Stage 3 GEMV source implementation | Present for Q8_0, Q4_K, Q6_K, and Q5_0. |
| GEMV correctness | Synthetic CPU comparison tests are present; not rerun in this review. |
| CUDA performance | Not established; no Stage 3 benchmark was run or recorded here. |
| End-to-end CUDA inference | Not established; GEMV is not yet wired into `DecisionEngine`. |
| Canvas C=16 behavior | Matches local C=256 decisions on the five recorded fixtures; broader and numerical stability remain unestablished. |
| Server admission and shutdown | Needs R21-01 and R21-02 before relying on the configured connection bound and drain behavior. |

## Verification notes

Static review only. No build, unit test, HTTP integration test, CUDA execution, compute-sanitizer run, or benchmark was run by this reviewer.

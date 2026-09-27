# Zeno — Specification: Multi-GPU Architecture & Execution Dynamics

## 1. Purpose & Scope

This document lists multi-GPU research questions and candidate decompositions. It does not prescribe a validated Zeno architecture; no Zeno GPU execution path exists yet.

Multi-GPU research is informed by the [separate community patch for llama.cpp PR #24423](https://github.com/readwrightexecute/llamacpp-diffusiongemma-multigpu). That patch explored a specific two-GPU layer-split setup. Its results do not establish a general performance or correctness guarantee.

---

## 2. Hardware Constraints & VRAM Budget

The local GGUF file (`diffusiongemma-26B-A4B-it-Q4_K_M.gguf`, cataloged in [`gguf-mapping.md`](gguf-mapping.md)) is approximately **16 GB** on disk. Runtime memory requirements are not established by file size alone; they depend on weight representation, workspaces, logits, cache state, and allocator overhead.

* **Weight memory**: measure for the actual runtime representation and device placement.
* **Canvas activation memory**: measure for the implemented dtype, attention path, batch size, and scratch strategy.

Whether the model fits on a given GPU and whether it needs sharding must be determined from measured runtime memory on that hardware.

---

## 3. Sharding Strategies: Layer Splitting vs. Tensor Parallelism

For low-latency "System One" decision execution, the choice of multi-GPU decomposition determines whether communication overhead negates compute gains.

### 3.1 Strategy A: Pipeline Layer Splitting (Candidate)

Illustrative placement only: this omits phase-specific KV/self-conditioning state and is not a validated transfer plan. In layer splitting, transformer blocks could be partitioned contiguously across devices:

```text
[GPU 0: Device 0]
  - Token Embeddings (token_embd.weight)
  - Blocks 0 through 14 (15 transformer layers)
  - Inter-device transfer: Hidden state H_15 (shape [256, 2816], ~1.44 MB in FP16)
         │
         ▼  (Single PCIe / NVLink P2P transfer)
[GPU 1: Device 1]
  - Blocks 15 through 29 (15 transformer layers)
  - Final Output RMSNorm (output_norm.weight)
  - Output projection & candidate logit readout
```

#### Questions to Measure:
* Count boundary transfers for the actual model phases, KV state, and self-conditioning path.
* Measure P2P availability and host synchronization on each supported topology.
* Include embeddings, output projection, cache state, and workspaces in placement and memory estimates.

---

### 3.2 Strategy B: Tensor Parallelism (Intra-Layer Sharding)

In tensor parallelism, weight matrices within each block are sharded across devices:
* **Attention Heads**: Query/Key/Value heads are split ($H_Q / N$, $H_{KV} / N$).
* **MoE Experts**: The 128 experts are distributed ($64$ experts per GPU on a 2-GPU topology).

#### Questions to Measure:
* Identify collectives from the chosen operator decomposition rather than assuming a fixed count.
* Benchmark communication and compute on supported interconnects and batch sizes.

---

## 4. Prompt Cache Lifetime & State Preservation

In structured decision workloads, the input prompt frequently contains invariant background context (e.g. system instructions, policy documents, few-shot schemas) that repeats across sequential queries.

### Cache Lifetime Taxonomy

1. **Intra-Query Lifetime (if multi-step execution is implemented)**:
   - Intermediate activations and self-conditioning vectors ($H_{\text{sc}}$) remain resident in device memory across steps $1 \dots T$.
   - **Goal to evaluate**: Keep reusable state on device where this preserves semantics and improves measured latency; transfers remain an implementation choice.

2. **Inter-Query Lifetime (Prompt Prefix Caching)**:
   - Reuse across requests is a hypothesis for this model's causal-prefill/bidirectional-decoder path; do not assume ordinary causal KV-prefix caching applies unchanged.
   - Before implementation, specify cache keys, ownership, eviction, invalidation, concurrency, and a parity test against uncached execution.

---

## 5. Verification & Parity Constraints

In accordance with [AGENTS.md](../AGENTS.md):

1. **Semantic Equivalence**:
   Multi-GPU execution must preserve the exact mathematical semantics of the CPU reference implementation.
2. **Numerical Tolerance**:
   Output logits and candidate probabilities from multi-GPU execution must match single-GPU and CPU outputs within documented numerical floating-point tolerances:
   Choose numerical tolerances per operation and dtype from reference comparisons; `10^{-4}` is not a validated global tolerance for quantized model outputs.
3. **Benchmarking Mandate**:
   Any performance claims regarding multi-GPU acceleration must be substantiated with reproducible latency (milliseconds per decision) and throughput (queries per second) benchmarks against the single-GPU baseline.

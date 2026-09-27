# Zeno — Specification: Multi-GPU Architecture & Execution Dynamics

## 1. Purpose & Scope

This document specifies the research parameters, sharding strategies, cache lifetimes, and execution mechanics for accelerating Zeno across multiple GPUs.

As established in [`AGENTS.md`](../AGENTS.md#L107-L128) and [`ARCHITECTURE.md`](../ARCHITECTURE.md#L177-L204), multi-GPU execution is informed by experimental evidence from the [separate community patch for llama.cpp PR #24423](https://github.com/readwrightexecute/llamacpp-diffusiongemma-multigpu). That patch explored a specific two-GPU layer-split setup; this specification defines Zeno's sharding architecture while treating general multi-device scaling as an active research area.

---

## 2. Hardware Constraints & VRAM Budget

The primary model artifact (`diffusiongemma-26B-A4B-it-Q4_K_M.gguf`, cataloged in [`specs/gguf-mapping.md`](gguf-mapping.md)) occupies approximately **16 GB** of memory:

* **Weight Memory**: $\approx 15.8\text{ GB}$ (quantized across Q4_K, Q6_K, Q8_0, and F32).
* **Canvas Activation Memory ($L = 256$)**: $< 100\text{ MB}$ per forward pass.

On high-memory single GPUs (e.g. 24 GB RTX 3090/4090, 48 GB A40/A6000), the model executes in a single device context without inter-device communication. On dual 16 GB hardware (e.g. 2× RTX 4080, 2× V100 16GB), distributing the model across devices is mandatory.

---

## 3. Sharding Strategies: Layer Splitting vs. Tensor Parallelism

For low-latency "System One" decision execution, the choice of multi-GPU decomposition determines whether communication overhead negates compute gains.

### 3.1 Strategy A: Pipeline Layer Splitting (Recommended for 1-Step Reads)

In layer splitting, transformer blocks are partitioned contiguously across devices:

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

#### Communication Analysis:
* **Total Transfers per Decision**: Exactly **1 transfer** of $\approx 1.44\text{ MB}$ (hidden state at layer boundary).
* **Host Synchronization**: Zero host round-trips; transfer uses CUDA Peer-to-Peer (P2P) memory copies.
* **Suitability**: Highly optimal for single-step structured reads because it incurs zero intra-layer collective communication overhead.

---

### 3.2 Strategy B: Tensor Parallelism (Intra-Layer Sharding)

In tensor parallelism, weight matrices within each block are sharded across devices:
* **Attention Heads**: Query/Key/Value heads are split ($H_Q / N$, $H_{KV} / N$).
* **MoE Experts**: The 128 experts are distributed ($64$ experts per GPU on a 2-GPU topology).

#### Communication Analysis:
* **Collectives**: Requires an `All-Reduce` operation after every attention output projection and every MoE block reduction.
* **Overhead**: Across 30 blocks, a single forward pass requires **60 All-Reduce operations**.
* **Limitation**: On standard PCIe interconnects without high-bandwidth NVLink, 60 synchronization barriers introduce significant latency jitter, making Strategy B less attractive for single-step 100ms decision budgets.

---

## 4. Prompt Cache Lifetime & State Preservation

In structured decision workloads, the input prompt frequently contains invariant background context (e.g. system instructions, policy documents, few-shot schemas) that repeats across sequential queries.

### Cache Lifetime Taxonomy

1. **Intra-Query Lifetime (Multi-step Denoising)**:
   - Intermediate activations and self-conditioning vectors ($H_{\text{sc}}$) remain resident in device memory across steps $1 \dots T$.
   - **Invariant**: No tensor is copied back to CPU host memory during intermediate steps.

2. **Inter-Query Lifetime (Prompt Prefix Caching)**:
   - If sequential queries share an identical prompt prefix ($0 \dots k$), key/value representations or intermediate activations can be cached across requests.
   - **Research Requirement**: Caching across queries must establish explicit cache eviction, memory ownership, and cache invalidation policies to avoid stale state pollution.

---

## 5. Verification & Parity Constraints

In accordance with [`AGENTS.md` §5](file:///hdd/Projects/zeno/AGENTS.md#L124-L140):

1. **Semantic Equivalence**:
   Multi-GPU execution must preserve the exact mathematical semantics of the CPU reference implementation.
2. **Numerical Tolerance**:
   Output logits and candidate probabilities from multi-GPU execution must match single-GPU and CPU outputs within documented numerical floating-point tolerances:
   $$\max_{k} |z_k^{(\text{multi-gpu})} - z_k^{(\text{cpu})}| < 10^{-4}$$
3. **Benchmarking Mandate**:
   Any performance claims regarding multi-GPU acceleration must be substantiated with reproducible latency (milliseconds per decision) and throughput (queries per second) benchmarks against the single-GPU baseline.

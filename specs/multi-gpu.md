# Zeno — Specification: Multi-GPU Architecture & Execution Dynamics

## 1. Purpose & Scope

This document specifies the multi-GPU architecture, topology partitioning, and execution dynamics for Zeno's CUDA inference backend.

As of CUDA Stage 5/6, Zeno implements a dual-GPU pipeline layer-splitting architecture (Strategy A) targeting dual-GPU workstations (e.g. dual NVIDIA RTX 3060 12GB GPUs, total 24GB VRAM) for low-latency System One inference.

Multi-GPU research is informed by the [separate community patch for llama.cpp PR #24423](https://github.com/readwrightexecute/llamacpp-diffusiongemma-multigpu). That patch explored a specific two-GPU layer-split setup; Zeno implements a native, zero-host-intervention execution pipeline with asynchronous peer stream/event coordination.

---

## 2. Hardware Constraints & VRAM Budget

The GGUF model file (`diffusiongemma-26B-A4B-it-Q4_K_M.gguf`, cataloged in [`gguf-mapping.md`](gguf-mapping.md)) is approximately **16 GB** on disk and ~16.8 GB uncompressed in memory. It exceeds the 12GB VRAM limit of a single RTX 3060, necessitating sharding across two devices.

* **Dual-GPU Partitioning (Strategy A)**:
  - **Device 0**: Token embeddings (~1.5 GB) + Layers 0..14 (~7.3 GB) + Workspace (~100 MB) ≈ **8.9 GB VRAM**.
  - **Device 1**: Layers 15..29 (~7.3 GB) + Final Output RMSNorm & Output projection (~1.5 GB) + Workspace (~100 MB) ≈ **8.9 GB VRAM**.
  - Both devices comfortably fit within 12 GB VRAM with >3 GB headroom for OS desktop and CUDA driver allocations.
* **Single-GPU Topology**:
  - When running on a device with $\ge 24$ GB VRAM (e.g., RTX 3090, RTX 4090, A100), all 30 layers reside on Device 0.
* **Canvas Activation Workspace**:
  - Sequence limit $N \le 512$ is strictly enforced at the CUDA entry point.
  - Per-device workspace requires $< 100$ MB, including attention scratch and batched MoE intermediate activations ($512 \times 8 \times 2816 \times 4\text{ B} \approx 46\text{ MB}$).

---

## 3. Sharding Strategies: Layer Splitting vs. Tensor Parallelism

### 3.1 Strategy A: Pipeline Layer Splitting (Implemented)

For single-step and few-step System One reads ($T \in [1, 4]$), layer splitting provides minimal inter-device communication overhead compared to tensor parallelism: only **one boundary transfer per forward pass step**.

```text
[GPU 0: Device 0]
  - Token Embeddings (token_embd.weight)
  - Blocks 0 through 14 (15 transformer layers)
  - Device-side top-8 MoE batched projections
  - Inter-device transfer: Hidden state H_15 (shape [N, 2816], FP32)
         │
         ▼  (cudaMemcpyPeerAsync + cudaStreamWaitEvent)
[GPU 1: Device 1]
  - Blocks 15 through 29 (15 transformer layers)
  - Device-side top-8 MoE batched projections
  - Final Output RMSNorm (output_norm.weight)
  - Output projection & candidate logit readout (model.output)
```

#### Boundary Transfer Mechanics:
1. When peer-to-peer (P2P) access is enabled (`cudaDeviceCanAccessPeer` returns 1), transfer uses `cudaMemcpyPeerAsync` on Device 0's stream.
2. A CUDA event `boundary_event` is recorded on Device 0's stream.
3. Device 1's stream calls `cudaStreamWaitEvent(ws1.stream, boundary_event)`, ensuring GPU hardware scheduling without stalling the host CPU thread.
4. Fallback: If P2P is not supported by the hardware/PCIe topology, transfer is staged through host memory using asynchronous stream synchronization.

#### MoE Execution Mechanics:
- MoE routing computes `top_k=8` expert indices and softmax weights directly in device memory.
- Active experts are evaluated using batched GPU kernels (`zeno_cuda_moe_gemv_gate_up_batched`, `zeno_cuda_moe_geglu_batched`, `zeno_cuda_moe_gemv_down_batched`, `zeno_cuda_moe_accumulate_scaled_batched`).
- Zero host intervention: no routing decisions or intermediate activations are transferred to the host during layer execution.

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

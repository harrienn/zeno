# Zeno — Specification: Multi-GPU Architecture & Execution Dynamics

## 1. Purpose & Scope

This document specifies the multi-GPU architecture, topology partitioning, and execution dynamics for Zeno's CUDA inference backend.

As of CUDA Stage 5/6, Zeno implements a dual-GPU pipeline layer-splitting architecture (Strategy A) targeting dual-GPU workstations (e.g. dual NVIDIA RTX 3060 12GB GPUs, total 24GB VRAM) for low-latency System One inference.

Multi-GPU research is informed by the [separate community patch for llama.cpp PR #24423](https://github.com/readwrightexecute/llamacpp-diffusiongemma-multigpu). That patch explored a specific two-GPU layer-split setup; Zeno implements a native execution pipeline with device-side batched MoE, zero host round-trips during layer execution, and asynchronous peer stream/event coordination.

---

## 2. Hardware Constraints & VRAM Budget

The GGUF model file (`diffusiongemma-26B-A4B-it-Q4_K_M.gguf`, cataloged in [`gguf-mapping.md`](gguf-mapping.md)) is approximately **16 GB** on disk and ~16.8 GB uncompressed in memory. It exceeds the 12GB VRAM limit of a single RTX 3060, necessitating sharding across two devices.

Zeno implements a hybrid CPU/GPU architecture: input token embedding lookups and candidate logit dot-product readout are executed on the host CPU against mapped GGUF tables, while the compute-intensive 30-layer transformer forward pass runs entirely on the GPUs.

* **Dual-GPU Partitioning (Strategy A)**:
  - **Device 0**: Layers 0..14 (~7.3 GB) + Activation Workspace (~211 MB / 201 MiB) ≈ **7.5 GB VRAM**.
  - **Device 1**: Layers 15..29 (~7.3 GB) + Final Output RMSNorm (<1 MB) + Activation Workspace (~211 MB / 201 MiB) ≈ **7.5 GB VRAM**.
  - Both devices comfortably fit within 12 GB VRAM with >4.4 GB headroom for OS desktop and CUDA driver allocations.
* **Single-GPU Topology**:
  - When running on a device with $\ge 24$ GB VRAM (e.g., RTX 3090, RTX 4090, A100), all 30 layers reside on Device 0 (~14.8 GB VRAM).
* **Canvas Activation Workspace**:
  - Sequence limit $N \le 512$ is strictly enforced at the CUDA entry point.
  - Per-device workspace requires **210,938,880 bytes (~210.94 MB / 201.17 MiB)** at $N=512$, calculated explicitly via `CudaDeviceWorkspace.activationWorkspaceBytes(512)`:

| Buffer Group | Buffers | Dimensions ($N=512$) | Memory (MiB / MB) |
|---|---|---|---|
| Embedding & Residuals | `emb_buf_0`, `emb_buf_1`, `norm_x`, `attn_out`, `dense_in`, `dense_out`, `h1`, `r_in`, `moe_in`, `moe_accum`, `h2`, `ffn_comb`, `post_ffn` | $13 \times [N \times 2816]$, FP32 | 71.50 MiB (74.97 MB) |
| Attention Head & Projections | `q_buf`, `head_out` | $2 \times [N \times 8192]$, FP32 | 32.00 MiB (33.55 MB) |
| Attention KV Buffers | `k_buf`, `v_buf` | $2 \times [N \times 2048]$, FP32 | 8.00 MiB (8.39 MB) |
| Dense FFN Scratch | `gate_buf`, `up_buf`, `geglu_buf` | $3 \times [N \times 2112]$, FP32 | 12.38 MiB (12.98 MB) |
| MoE Router Output & Top-8 | `router_logits`, `top_indices`, `top_weights` | $[N \times 128]$ FP32 + $2 \times [N \times 8]$ | 0.28 MiB (0.30 MB) |
| Batched MoE Projections | `exp_gate_up_buf`, `exp_act_buf` | $[N \times 8 \times 1408]$ FP32 + $[N \times 8 \times 704]$ FP32 | 33.00 MiB (34.60 MB) |
| Batched MoE Down-Projection | `exp_down_buf` | $[N \times 8 \times 2816]$, FP32 | 44.00 MiB (46.14 MB) |
| Final Output Norm Buffer | `norm_hidden` | $[1 \times 2816]$, FP32 | 0.01 MiB (0.01 MB) |
| **Total Activation Workspace** | **27 buffers** | **All device buffers at $N=512$** | **201.17 MiB (210.94 MB)** |

---

## 3. Sharding Strategies: Layer Splitting vs. Tensor Parallelism

### 3.1 Strategy A: Pipeline Layer Splitting (Implemented)

For single-step and few-step System One reads ($T \in [1, 4]$), layer splitting provides minimal inter-device communication overhead compared to tensor parallelism: only **one boundary transfer per forward pass step**.

```text
[Host CPU]
  - Token embedding dequantization & RMSNorm (token_embd.weight)
  - Copies input embeddings H_0 to Device 0 [N, 2816]
         │
         ▼
[GPU 0: Device 0]
  - Blocks 0 through 14 (15 transformer layers)
  - Device-side top-8 MoE batched projections
  - Inter-device transfer: Hidden state H_15 (shape [N, 2816], FP32)
         │
         ▼  (P2P: cudaMemcpyPeerAsync + cudaStreamWaitEvent; or Host-Staged Fallback)
[GPU 1: Device 1]
  - Blocks 15 through 29 (15 transformer layers)
  - Device-side top-8 MoE batched projections
  - Final Output RMSNorm (output_norm.weight)
  - Copies normalized answer token vector to Host [1, 2816]
         │
         ▼
[Host CPU]
  - Candidate logit dot-product readout (model.output)
```

#### Boundary Transfer Mechanics:
1. When peer-to-peer (P2P) access is enabled (`cudaDeviceCanAccessPeer` returns 1 and peer access is enabled), transfer uses `cudaMemcpyPeerAsync` on Device 0's stream.
2. A CUDA event `boundary_event` is recorded on Device 0's stream.
3. Device 1's stream calls `cudaStreamWaitEvent(ws1.stream, boundary_event)`, ensuring GPU hardware scheduling without stalling the host CPU thread.
4. Fallback: If direct P2P is not supported by the hardware/PCIe topology (e.g. GeForce RTX 3060 PCIe without NVLink), transfer is staged through host memory with synchronous host stream coordination.

#### MoE Execution Mechanics:
- MoE routing computes `top_k=8` expert indices and softmax weights directly in device memory.
- Active experts are evaluated using batched GPU kernels (`zeno_cuda_moe_gemv_gate_up_batched`, `zeno_cuda_moe_geglu_batched`, `zeno_cuda_moe_gemv_down_batched`, `zeno_cuda_moe_accumulate_scaled_batched`).
- Zero host intervention within the layer pipeline: no routing decisions or intermediate activations are transferred to the host during transformer layer execution.

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

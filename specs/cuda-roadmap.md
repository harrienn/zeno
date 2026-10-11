# Zeno — CUDA & Multi-GPU Implementation Roadmap

**Status:** Draft / Active Specification  
**Governing Documents:** [`AGENTS.md`](../AGENTS.md), [`specs/UNRESOLVED.md`](UNRESOLVED.md), [`specs/multi-gpu.md`](multi-gpu.md), [`specs/transformer.md`](transformer.md)  
**Target Hardware:** Dual NVIDIA GeForce RTX 3060 12GB (Ampere `sm_86`, 24GB total VRAM, PCIe 4.0)  
**Target Model:** DiffusionGemma 26B-A4B-it Q4_K_M (16.8 GB GGUF, SHA-256 `24523b6c...`)  

---

## 1. Principles & Objectives

### 1.1 Objectives
1. **High-Speed System One Execution**: Accelerate single-step structured decision execution from CPU reference runtime (~10–20 seconds) down to sub-100 milliseconds.
2. **Exact Semantic Preservation**: CUDA execution must preserve the exact mathematical contracts established by the CPU reference implementation ([`AGENTS.md`](../AGENTS.md) §5).
3. **Explicit Memory Ownership**: Host memory and GPU VRAM allocations must have identifiable owners, deterministic lifetimes, and explicit transfer boundaries ([`AGENTS.md`](../AGENTS.md) §6).
4. **Standalone Native Deployment**: Maintain Zeno as a single standalone executable compiled by `zig build`, with zero dynamic interpreter or heavyweight framework dependencies.
5. **General CUDA Platform Scalability**: While the current local development environment is dual RTX 3060 (12GB each), the backend must support both **Single-GPU mode** (for cards with $\ge 24$ GB VRAM such as RTX 3090, RTX 4090, A100, H100) and **Multi-GPU mode** (pipeline layer splitting across $N \ge 2$ devices). Device partitioning must be configurable and topology-aware rather than hardcoded to a fixed 2-GPU layout.

### 1.2 Non-Goals
* General-purpose text generation or conversational autoregressive decoding loops.
* Multi-node cluster distribution (Zeno targets low-latency single-box single- or multi-GPU execution).
* Speculative framework layers or generic hardware abstraction plugins.

---

## 2. Hardware Topology & Memory Budget

### 2.1 Single-GPU Topology (Devices $\ge 24$ GB VRAM, e.g. RTX 3090 / 4090 / A100)
When running on a single GPU with sufficient memory:
* **Placement**: Entire model (Embeddings, all 30 transformer layers 0–29, Output Norm, Logit Readout) resides in Device 0.
* **Transfers**: Zero inter-device transfers; only initial prompt tokens sent to GPU and final candidate logits read back to host.
* **VRAM footprint**: ~17.5 GB total VRAM.

### 2.2 Multi-GPU Topology (e.g. Dual RTX 3060 12GB)
When running on devices where individual VRAM is under 24 GB, Zeno partitions layers across $N$ devices using Pipeline Layer Splitting:

```text
[Host: Zig Engine]
  │
  ├── 1. GGUF Parser: Streams tensors into GPU 0 and GPU 1 VRAM (once at startup)
  ├── 2. DecisionRequest: Tokenizes prompt + formats seeded 256-token canvas
  └── 3. Launches Pipeline on CUDA Stream 0
         │
         ▼
[GPU 0: Device 0 — 12 GB VRAM] (Allocated: ~9.9 GB)
  ├── Input Token Embeddings (token_embd.weight, Q6_K: ~2.1 GB)
  ├── Transformer Blocks 0 through 14 (15 layers: ~6.8 GB)
  │     - Attention (Q, K, V, RoPE, Unified Non-Causal Mask, Post-Norm)
  │     - Dense FFN (GeGLU) + 128-expert MoE (top-8 routing)
  └── Activations & KV workspace: ~1.0 GB
         │
         ▼  cudaMemcpyPeerAsync (P2P direct PCIe transfer: [N, 2816] activations, ~2.75–11.00 MiB / 2.88–11.53 MB in FP32 for N in [256, 1024]; estimated target latency < 1 ms over PCIe 4.0, subject to empirical measurement)
[GPU 1: Device 1 — 12 GB VRAM] (Allocated: ~8.3 GB)
  ├── Transformer Blocks 15 through 29 (15 layers: ~6.8 GB)
  │     - Attention + Dense FFN + 128-expert MoE
  ├── Final Output RMSNorm (output_norm.weight: ~11 KB)
  ├── Candidate Logit Projection (tied token_embd.weight rows: ~0.5 GB)
  └── Activations & KV workspace: ~1.0 GB
         │
         ▼  cudaMemcpyAsync (Reads 2 to 5 float32 logits back to host: negligible)
[Host: Zig Engine]
  └── Softmax, candidate probabilities, entropy calculation, JSON response
```

---

## 3. Phased Implementation Stages

### Stage 1: Build Pipeline & Device Topology Discovery
* **Goal**: Enable `zig build` to detect `nvcc`, compile native CUDA C++ sources into an object file (`zeno_cuda.o`), link it directly into the `zeno` executable, and report device topology (GPU models, compute capabilities, free/total VRAM, and P2P peer access matrix) while keeping the CPU-only build functional when CUDA is absent.
* **Deliverables**:
  * [`build.zig`](../build.zig) updated with `-Dcuda` flag (defaulting to false or auto-detecting `/usr/local/cuda/bin/nvcc`, preserving zero-dependency CPU builds).
  * `src/cuda/zeno_cuda.h` defining the C-ABI function prototypes.
  * `src/cuda/zeno_cuda.cu` implementing device query functions.
  * CLI command `zeno cuda-info` reporting GPU names, compute capabilities, free VRAM, and P2P peer access status across visible devices.
* **Exit Gate**: `zig build` compiles cleanly, CPU-only build remains fully functional, and `zeno cuda-info` successfully queries visible CUDA devices and reports their capabilities and peer-access matrix.

### Stage 2: Device Memory Management & Fused Elementwise Kernels
* **Goal**: Establish GPU memory buffers in Zig and implement core elementwise vector operations on CUDA.
* **Deliverables**:
  * `src/cuda/ops.cu`:
    * `cuda_rms_norm`: Fused RMSNorm with $\epsilon = 10^{-6}$ and learned scaling weights.
    * `cuda_unweighted_rms_norm`: RMSNorm without learned scale (used by canvas embed & router).
    * `cuda_rope`: NeoX-style Rotary Position Embedding over 512 / 256 head dimensions with frequency scaling.
    * `cuda_geglu`: Fused GeGLU activation ($\text{GELU}(a) \times b$).
  * `src/cuda_mem.zig`: Zig struct wrapping `cudaMalloc`, `cudaFree`, `cudaMemcpyAsync`.
* **Exit Gate**: Zig unit tests comparing each CUDA kernel against CPU reference implementations in `src/ops.zig` with numerical tolerance $\le 10^{-5}$.

### Stage 3: Quantized Matrix-Vector Multiplication (GEMV)
* **Goal**: Implement high-throughput matrix-vector kernels for quantized weights in the active inference path.
* **Deliverables**:
  * `src/cuda/gemv_q4k.cu`: Warp-parallel matrix-vector multiplication for Q4_K_M blocks using `__shfl_sync` warp reduction (attention Q/K/O, shared FFN gate/up, MoE `ffn_gate_up_exps.weight`).
  * `src/cuda/gemv_q8_0.cu`: Q8_0 dot product for shared FFN down (`ffn_down.weight`) and MoE expert down (`ffn_down_exps.weight`) (R10-01).
  * `src/cuda/gemv_q6k.cu`: Q6_K dot product for token embeddings (`token_embd.weight`), attention V (`blk.{i}.attn_v.weight` in SWA layers), and candidate logit projection.
  * *(Deferred to Phase 4+)* `src/cuda/gemv_q5_0.cu`: Q5_0 dot product for self-conditioning down projection (`self_cond_down.weight`), bypassed in one-step reads.
* **Exit Gate**: Numerical agreement tests comparing GPU GEMV output against CPU `ops.gemvTensor` within quantization precision tolerance.

### Stage 4: Unified Attention & MoE Expert Execution
* **Goal**: Implement full transformer layer components on GPU.
* **Deliverables**:
  * `src/cuda/attention.cu`: Bidirectional non-causal attention kernel supporting:
    * Causal masking for prompt queries ($s \le t$).
    * Full bidirectional visibility for canvas queries ($s \in [0, N)$).
    * Gemma 4 attention scale ($1.0$).
  * `src/cuda/moe.cu`:
    * MoE router projection and top-8 expert selection with softmax renormalization.
    * Batched expert execution over selected active experts.
* **Exit Gate**: Single transformer layer forward pass on GPU matches CPU layer output.

### Stage 5: 2-GPU Pipeline Layer Split
* **Goal**: Partition the 30-layer model across Device 0 (layers 0–14) and Device 1 (layers 15–29) and execute via asynchronous P2P transfer (gated on dual-GPU availability and confirmed P2P capability).
* **Deliverables**:
  * `src/infer_cuda.zig`: High-level GPU inference pipeline.
  * Enable direct P2P memory access: `cudaDeviceEnablePeerAccess(1, 0)` and `cudaDeviceEnablePeerAccess(0, 1)`.
  * Asynchronous stream coordination across Device 0 and Device 1.
  * Layer boundary transfer: `cudaMemcpyPeerAsync` transfers $H_{15}$ ($[N, 2816]$ where $N = \text{prompt\_len} + 256 \le 1024$, taking $2.75$ to $11.00$ MiB / $2.88$ to $11.53$ MB in FP32; estimated target latency < 1 ms over PCIe 4.0, subject to empirical measurement). FP16 activation transfer ($1.38$ to $5.50$ MiB) is retained as a proposed bandwidth optimization subject to empirical numerical tolerance testing (R10-02).
* **Exit Gate**: Full 30-layer model forward pass executes across both GPUs without host intervention during inference.

### Stage 6: Parity Validation & Latency Benchmarks
* **Goal**: Integrate CUDA backend into `DecisionEngine` and validate against golden fixtures.
* **Deliverables**:
  * Implement explicit backend selection in the CLI (`zeno decide-fixture --backend=[cpu|cuda]`) and add `--backend` flag support to `tools/verify_all_oracles.sh` (R10-03).
  * Run `./tools/verify_all_oracles.sh --backend=cuda`.
  * Verify 5/5 discrete decision matches (`Yes`, `Medium`, `No`, `Low`, `5`) against CPU reference engine.
  * Evaluate candidate logits and probabilities against reference oracle records.
  * Reproducible latency benchmarks measuring single-query decision latency (target: sub-100ms).
* **Exit Gate**: All golden fixtures evaluated, discrete decisions aligned, verifier reports executed engine, and reproducible performance benchmark registered in documentation.

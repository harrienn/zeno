# Zeno — Specification Index

This directory contains the normative technical specifications, empirical research analyses, and verification ledgers governing the Zeno runtime.

For high-level system architecture and execution flow, start with [README.md](../README.md) and [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 1. Normative Implemented Specifications

These documents define the authoritative contracts for the implemented v0.2.0 runtime:

- [**`decision-api.md`**](decision-api.md) `[IMPLEMENTED]`: Specification of the Jev-compatible `/v1/systemone` HTTP REST and native Zig library interfaces, query schemas, single-token candidate validation, response serialization, wire error codes, and entropy reporting.
- [**`canvas-templates.md`**](canvas-templates.md) `[IMPLEMENTED]`: Dual-region sequence layout (causal prompt prefix $0 \dots P-1$ and bidirectional canvas $P \dots P+C-1$), verified prompt template construction, answer slot pinning ($p=0$), and configurable canvas prefix widths ($C \in [16, 256]$).
- [**`structured-reads.md`**](structured-reads.md) `[IMPLEMENTED]`: Single-step candidate logit extraction, single-softcapped unsoftcap transformation, candidate probability normalization, and Shannon entropy calculation.
- [**`transformer.md`**](transformer.md) `[IMPLEMENTED]`: Full 30-layer Gemma 4 MoE model architecture: bidirectional non-causal attention, dual RoPE frequency bases (10,000 / 1,000,000), sliding-window and global attention head patterns, 128-expert top-8 routing, and unweighted pre-FFN RMSNorm.
- [**`tokenizer.md`**](tokenizer.md) `[IMPLEMENTED]`: Gemma 4 BPE vocabulary format ($V=262,144$), control token IDs (`bos=2`, `eos=1`, `mask=4`, `pad=0`), verbatim chat template integration, single-BOS invariant, and single-token continuation resolution.
- [**`gguf-mapping.md`**](gguf-mapping.md) `[IMPLEMENTED]`: Verified mapping of DiffusionGemma 26B-A4B-it Q4_K_M GGUF tensor names, dimensions, quantization formats (Q4_K, Q6_K, Q8_0, Q5_0), and memory offsets.
- [**`multi-gpu.md`**](multi-gpu.md) `[IMPLEMENTED]`: Dual-GPU pipeline layer splitting (Strategy A: Layers 0..14 on GPU 0, Layers 15..29 on GPU 1), peer-to-peer / host-staged boundary transfer mechanics, and activation workspace memory sizing.

---

## 2. Empirical Research & Evaluation

These documents record verified empirical findings from dedicated experimental studies:

- [**`denoising.md`**](denoising.md) `[IMPLEMENTED / EMPIRICAL]`: Multi-step discrete diffusion mechanics and empirical evaluation of adaptive 2-step denoising with candidate-restricted self-conditioning soft embeddings ($E_{\text{sc}}$), eliminating option-rotation bias while preserving sub-100ms median latency.
- [**`calibration-analysis.md`**](calibration-analysis.md) `[EMPIRICAL RESEARCH]`: 5-fold cross-validated probability calibration on ARC-Challenge ($N=150$); optimal operating temperatures ($T_{\text{opt}} \approx 1.60$ softcapped, $T_{\text{opt}} \approx 2.40$ raw) reducing ECE by >40%–55%.
- [**`canvas-geometry.md`**](canvas-geometry.md) `[EMPIRICAL RESEARCH]`: Answer position geometry ($k \in [0, 255]$) and canvas width ($C \in [16, 256]$) sensitivity study; proves that answer placement at $k \in \{0, 1\}$ maximizes prompt cue attention and logit margins.
- [**`autoregressive-vs-diffusion.md`**](autoregressive-vs-diffusion.md) `[EMPIRICAL RESEARCH]`: Head-to-head empirical comparison of structured discrete diffusion vs. grammar-constrained autoregressive decoding across latency, parameter streaming traffic, calibration, and positional bias.

---

## 3. Verification & Provenance Ledgers

- [**`UNRESOLVED.md`**](UNRESOLVED.md) `[RESOLVED LEDGER]`: The authoritative specification gate register. All 23 original specification questions (U-01 through U-23) are formally resolved with code, commit, and fixture evidence. Contains the `zeno.fixture/v1` schema and the `zeno.tolerance/v2` numerical comparison policy governing reference parity.
- [**`cuda-roadmap.md`**](cuda-roadmap.md) `[HISTORICAL ROADMAP]`: Original engineering roadmap for the CUDA C++ backend, documenting completed Stage 1 through Stage 6 milestones.
- [**Phase Optimization RFCs**](cuda-phase15-optimization-rfc.md) `[HISTORICAL RFCs]`: Design RFCs for completed kernel optimizations that achieved the sub-100ms milestone: [Phase 15](cuda-phase15-optimization-rfc.md), [Phase 16](cuda-phase16-moe-gemv-rfc.md), [Phase 17](cuda-phase17-dense-gemv-rfc.md), [Phase 18](cuda-phase18-sub100ms-rfc.md), and [Phase 19](cuda-phase19-prefill-rfc.md).


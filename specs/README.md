# Zeno — Specifications

This directory contains technical specifications for the Zeno runtime, Jev-style decision semantics, and execution formats.

As outlined in [ARCHITECTURE.md](../ARCHITECTURE.md) and [PLAN.md](../PLAN.md), high-level architecture and sequencing belong in the root documents, while precise mathematical, layout, and behavioral specifications belong in `specs/`.

## Authored Specifications

- [**`decision-api.md`**](decision-api.md): Specification of the Jev-compatible `/v1/systemone` HTTP REST and native Zig library interfaces, query schemas, single-token choice constraints, response formats, and entropy reporting.
- [**`canvas-templates.md`**](canvas-templates.md): Specification of the 256-token canvas layout, template construction, prompt context placement, answer-position pinning, and single-token candidate validation.
- [**`structured-reads.md`**](structured-reads.md): Mathematical and execution semantics of single-step structured reads: non-causal forward evaluation, logit slicing at pinned slots, logit softcapping, candidate softmax normalization, and Shannon entropy calculation.
- [**`gguf-mapping.md`**](gguf-mapping.md): Concrete mapping of DiffusionGemma GGUF tensor names, shapes, and quantization types (Q4_K_M) verified against the local repository artifact (`models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`), including global attention $V$-sharing and MoE expert topology.

## Planned Specifications

- **`transformer.md`**: Detailed layer mechanics: bidirectional non-causal attention, RoPE configuration, 128-expert MoE routing, and RMSNorm.
- **`denoising.md`**: Adaptive step-capping and discrete diffusion dynamics for high-entropy slots and multi-token structured infilling.
- **`multi-gpu.md`**: Investigation of multi-GPU execution, sharding, device-resident self-conditioning, and cache lifetime.
- **`tokenizer.md`**: Tokenizer vocabulary format, byte-pair/SentencePiece merges, and special token conventions (`bos=2`, `eos=1`, `mask=4`).

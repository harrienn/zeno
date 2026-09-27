# Zeno — Specifications

This directory contains technical specifications for the Zeno runtime, Jev-style decision semantics, and execution formats.

As outlined in [ARCHITECTURE.md](../ARCHITECTURE.md) and [PLAN.md](../PLAN.md), high-level architecture and sequencing belong in the root documents, while precise mathematical, layout, and behavioral specifications belong in `specs/`.

## Authored Specifications

- [**`decision-api.md`**](decision-api.md): Specification of the Jev-compatible `/v1/systemone` HTTP REST and native Zig library interfaces, query schemas, single-token choice constraints, response formats, and entropy reporting.
- [**`gguf-mapping.md`**](gguf-mapping.md): Concrete mapping of DiffusionGemma GGUF tensor names, shapes, and quantization types (Q4_K_M) verified against the local repository artifact (`models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`), including global attention $V$-sharing and MoE expert topology.

## Planned Specifications

- **`canvas-templates.md`**: Specification of seeded token canvas layout, template construction, prompt context placement, and answer-position seeding/pinning behavior.
- **`structured-reads.md`**: Mathematical and execution semantics of single-step structured reads: forward-pass logit extraction at answer positions, candidate normalization, and Shannon entropy.
- **`denoising.md`**: Adaptive step-capping and discrete diffusion dynamics for high-entropy slots and multi-token structured infilling.
- **`multi-gpu.md`**: Investigation of multi-GPU execution, sharding, device-resident self-conditioning, and cache lifetime.
- **`transformer.md`**: Detailed layer mechanics: bidirectional non-causal attention, RoPE configuration, 128-expert MoE routing, and RMSNorm.
- **`tokenizer.md`**: Tokenizer vocabulary format, byte-pair/SentencePiece merges, and special token conventions (`bos=2`, `eos=1`, `mask=4`).

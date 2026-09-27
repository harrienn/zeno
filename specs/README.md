# Zeno — Specifications

This directory contains technical specifications for the Zeno runtime, Jev-style decision semantics, and execution formats.

As outlined in [ARCHITECTURE.md](../ARCHITECTURE.md) and [PLAN.md](../PLAN.md), high-level architecture and sequencing belong in the root documents, while precise mathematical, layout, and behavioral specifications belong in `specs/`.

## Planned Specifications

- **`decision-api.md`**: Specification of the Jev-compatible `/v1/systemone` HTTP and native Zig API contract, query schemas (boolean judgments, multiple-choice classification, ordinal scores), and response formats.
- **`canvas-templates.md`**: Specification of seeded token canvas layout, template construction, prompt context placement, and pinned mask slot positioning (`mask_token_id = 4`).
- **`structured-reads.md`**: Mathematical and execution semantics of single-step structured reads: forward pass logit extraction at pinned slots, softmax probability normalization, Shannon entropy calculation, and calibrated confidence scoring.
- **`denoising.md`**: Adaptive step-capping and discrete diffusion dynamics for high-entropy slots and multi-token structured infilling.
- **`multi-gpu.md`**: Multi-GPU tensor parallelism, sharded MoE expert execution, device-resident self-conditioning, and prompt cache preservation (incorporating lessons from the Multi-GPU patch to llama.cpp PR #24423).
- **`gguf-mapping.md`**: Exact mapping of DiffusionGemma GGUF tensor names, shapes, and quantization types (Q4_K_M) to runtime model layers.
- **`transformer.md`**: Detailed layer mechanics: bidirectional non-causal attention, RoPE configuration, 128-expert MoE routing, and RMSNorm.
- **`tokenizer.md`**: Tokenizer vocabulary format, byte-pair/SentencePiece merges, and special token conventions (`bos=2`, `eos=1`, `mask=4`).

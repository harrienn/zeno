# Zeno — Specifications

This directory is the index for planned technical specifications for the Zeno runtime, Jev-style decision semantics, and execution formats. The specifications listed below have not been written yet.

As outlined in [ARCHITECTURE.md](../ARCHITECTURE.md) and [PLAN.md](../PLAN.md), high-level architecture and sequencing belong in the root documents, while precise mathematical, layout, and behavioral specifications belong in `specs/`.

## Planned Specifications

- **`decision-api.md`**: Specification of the Jev-compatible `/v1/systemone` HTTP and native Zig API contract, query schemas (boolean judgments, multiple-choice classification, ordinal scores), and response formats.
- **`canvas-templates.md`**: Specification of seeded token canvas layout, template construction, prompt context placement, and answer-position seeding/pinning behavior. Confirm token IDs and exact pinning semantics against the selected model artifact and reference implementation.
- **`structured-reads.md`**: Mathematical and execution semantics of single-step structured reads: forward-pass logit extraction at answer positions, candidate normalization, and Shannon entropy. Probability calibration is evaluated separately and must not be inferred from softmax normalization.
- **`denoising.md`**: Adaptive step-capping and discrete diffusion dynamics for high-entropy slots and multi-token structured infilling.
- **`multi-gpu.md`**: Investigation of multi-GPU execution, sharding, device-resident self-conditioning, and cache lifetime. Include the limits of the separate experimental two-GPU community patch for llama.cpp PR #24423.
- **`gguf-mapping.md`**: Exact mapping of DiffusionGemma GGUF tensor names, shapes, and quantization types (Q4_K_M) to runtime model layers.
- **`transformer.md`**: Detailed layer mechanics: bidirectional non-causal attention, RoPE configuration, 128-expert MoE routing, and RMSNorm.
- **`tokenizer.md`**: Tokenizer vocabulary format, byte-pair/SentencePiece merges, and special token conventions (`bos=2`, `eos=1`, `mask=4`).

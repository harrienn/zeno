# Zeno — Specifications

This directory contains technical specifications for the Zeno runtime, model semantics, and execution formats.

As outlined in [ARCHITECTURE.md](../ARCHITECTURE.md) and [PLAN.md](../PLAN.md), high-level architecture and sequencing belong in the root documents, while precise mathematical, layout, and behavioral specifications belong in `specs/`.

## Planned Specifications

- **`denoising.md`**: Formal specification of the discrete diffusion process, token canvas representation, noise/mask token scheduling, confidence/entropy-bounded snapping rules, and step transitions.
- **`gguf-mapping.md`**: Exact mapping of DiffusionGemma GGUF tensor names, shapes, and quantization types to runtime model layers.
- **`transformer.md`**: Detailed layer mechanics: bidirectional non-causal attention, RoPE configuration, MoE routing and expert dispatch, and RMSNorm.
- **`tokenizer.md`**: Tokenizer vocabulary format, byte-pair/SentencePiece merges, and special token conventions (`bos=2`, `eos=1`, `mask=4`).

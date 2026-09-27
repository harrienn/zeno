# Zeno — Specifications

This directory contains technical specifications for the Zeno runtime, Jev-style decision semantics, and execution formats.

As outlined in [ARCHITECTURE.md](../ARCHITECTURE.md) and [PLAN.md](../PLAN.md), high-level architecture and sequencing belong in the root documents, while precise mathematical, layout, and behavioral specifications belong in `specs/`.

## Authored Specifications

- [**`decision-api.md`**](decision-api.md): Specification of the Jev-compatible `/v1/systemone` HTTP REST and native Zig library interfaces, query schemas, single-token choice constraints, response formats, and entropy reporting.
- [**`canvas-templates.md`**](canvas-templates.md): Proposed canvas layout and prompt construction, with answer pinning and padding semantics marked for reference validation.
- [**`structured-reads.md`**](structured-reads.md): Candidate logit extraction and scoring, including the causal prefill/bidirectional decoder distinction and open parity requirements.
- [**`transformer.md`**](transformer.md): Detailed layer execution mechanics: bidirectional non-causal attention, dual RoPE frequency bases, global layer $V=K$ sharing, 128-expert MoE top-8 routing, and RMSNorm.
- [**`tokenizer.md`**](tokenizer.md): Gemma 4 BPE vocabulary format, special token conventions (`bos=2`, `eos=1`, `mask=4`), single-token candidate validation, and native Zig tokenizer contracts.
- [**`gguf-mapping.md`**](gguf-mapping.md): Concrete mapping of DiffusionGemma GGUF tensor names, shapes, and quantization types (Q4_K_M) verified against the local repository artifact (`models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`).
- [**`denoising.md`**](denoising.md): Reference denoising behavior and clearly labeled experiments for follow-up reads; proposed update equations are not implementation requirements until validated.
- [**`multi-gpu.md`**](multi-gpu.md): Multi-GPU options and verification questions. Hardware budgets and communication counts are hypotheses until measured.

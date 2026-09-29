# Zeno — Specification: GGUF Model & Tensor Mapping

## 1. Scope & Artifact Provenance

This document specifies the tensor layout, metadata parameters, and weight mapping for DiffusionGemma models in Zeno.

All values, shapes, and quantization types documented here are directly extracted and verified against the local repository artifact:

* **Artifact Filename**: `models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`
* **Format**: GGUF Version 3 (quantization version 2, file type 15)
* **Reported Architecture**: `diffusion-gemma`
* **Reported Model Name**: `Dg_Rc0P1_Patched` (Size label: `128x2.6B`)
* **Total Tensors**: 692
* **Total Metadata Key-Values**: 44

These claims are specific to that local file. Its SHA-256 digest and the exact inspection tool/version are not recorded yet; add both before using this table as reproducible provenance. Dimensions below are GGUF/GGML tensor dimensions in stored order, not an implicit row-major `[rows, columns]` convention. Any conversion to a runtime matrix layout must state the transpose/stride mapping explicitly.

> **Provenance is incomplete (U-14).** Until the SHA-256 digest and the inspection tool/version are recorded, every value in this document is a claim about an unnamed file and is not reproducible. The omission is already acknowledged above; it is tracked here so that it cannot be forgotten. Until it is closed, a golden fixture built from this artifact may not cite this document as its evidence (see [`UNRESOLVED.md`](UNRESOLVED.md) §5).

---

## 2. Model Metadata & Structural Invariants

**Reading this table.** Entries divide into two classes, and the distinction is load-bearing:

* **Invariants** — values a runtime must *validate on load* and reject the artifact if violated. Hardcoding a value the artifact contradicts produces silently wrong computation rather than a clean failure, so these belong in the loader's acceptance check.
* **Descriptive** — values that document the artifact but are not yet contractually enforced.

As of this revision the loader validates `block_count` and `embedding_length` only. The most consequential unvalidated invariant is **`diffusion-gemma.attention.causal`**, which must be `false` for Zeno's structured-read semantics to apply at all; it is currently unchecked, and every remaining `diffusion-gemma.*` structural key below is descriptive only.

| Parameter Key | Value | Description |
| :--- | :--- | :--- |
| `diffusion.canvas_length` | `256` | Default token canvas width |
| `diffusion-gemma.block_count` | `30` | Number of transformer layers ($L = 0 \dots 29$) |
| `diffusion-gemma.context_length` | `262144` | Maximum context length |
| `diffusion-gemma.embedding_length` | `2816` | Hidden state dimension ($d_{\text{model}}$) |
| `diffusion-gemma.feed_forward_length` | `2112` | Shared FFN intermediate dimension |
| `diffusion-gemma.attention.causal` | `false` | **Non-causal bidirectional attention** |
| `diffusion-gemma.attention.head_count` | `16` | Number of Query heads ($H_Q$) |
| `diffusion-gemma.attention.head_count_kv` | *array[30]* | KV heads: 8 for SWA layers, 2 for Global layers. The 30 literal values are not recorded; the pattern (5×8 then 2, repeating 6×) is asserted in [`transformer.md`](transformer.md) §3.1. Record the array so the pattern can be verified rather than assumed. |
| `diffusion-gemma.attention.key_length` | `512` | Head dimension for Global layers ($d_k$) |
| `diffusion-gemma.attention.value_length` | `512` | Head dimension for Global layers ($d_v$) |
| `diffusion-gemma.attention.key_length_swa` | `256` | Head dimension for SWA layers ($d_{k,\text{swa}}$) |
| `diffusion-gemma.attention.value_length_swa` | `256` | Head dimension for SWA layers ($d_{v,\text{swa}}$) |
| `diffusion-gemma.attention.sliding_window` | `1024` | Sliding window attention horizon. Inoperative at `canvas_length = 256`; do not implement windowing for Phase 3 (U-20). |
| `diffusion-gemma.rope.freq_base` | `1000000.0` | Global RoPE frequency base |
| `diffusion-gemma.rope.freq_base_swa` | `10000.0` | SWA RoPE frequency base |
| `diffusion-gemma.rope.dimension_count` | `512` | RoPE dimension for Global layers |
| `diffusion-gemma.rope.dimension_count_swa` | `256` | RoPE dimension for SWA layers |
| `diffusion-gemma.attention.layer_norm_rms_epsilon`| `1e-6` | RMSNorm epsilon |
| `diffusion-gemma.final_logit_softcapping` | `30.0` | Logit softcapping limit |
| `diffusion-gemma.expert_count` | `128` | Total MoE experts per layer |
| `diffusion-gemma.expert_used_count` | `8` | Active MoE experts per token (top-8 routing) |
| `diffusion-gemma.expert_feed_forward_length` | `704` | Intermediate dimension per expert FFN |
| `tokenizer.ggml.bos_token_id` | `2` | Beginning-of-sequence token |
| `tokenizer.ggml.eos_token_id` | `1` | End-of-sequence token |
| `tokenizer.ggml.mask_token_id` | `4` | Discrete diffusion mask / noise token |
| `tokenizer.ggml.padding_token_id` | `0` | Padding token |
| `tokenizer.ggml.chat_template` | **not yet extracted** | Chat template. Required for prompt construction; see [`canvas-templates.md`](canvas-templates.md) §4. Tracked as **U-05**. |

---

> **Phase scope**: the `self_cond_*` tensors below are Phase 4+ territory. Whether a one-step read
> evaluates the self-conditioning MLP at all is unresolved (U-18 in [`UNRESOLVED.md`](UNRESOLVED.md)).

## 3. Global Tensors (7 Tensors)

These tensors reside outside the repeated transformer blocks:

| Tensor Name | Shape `[dims]` | GGML Type | Description |
| :--- | :--- | :--- | :--- |
| `token_embd.weight` | `[2816, 262144]` | `Q6_K` | Token embedding table. **Tied to output projection**. |
| `output_norm.weight` | `[2816]` | `F32` | Final pre-output RMSNorm scale |
| `rope_freqs.weight` | `[256]` | `F32` | Precomputed rotary position frequencies |
| `self_cond_pre_norm.weight` | `[2816]` | `F32` | Self-conditioning input RMSNorm |
| `self_cond_down.weight` | `[2112, 2816]` | `Q5_0` | Self-conditioning down projection |
| `self_cond_gate.weight` | `[2816, 2112]` | `Q4_K` | Self-conditioning gated activation |
| `self_cond_up.weight` | `[2816, 2112]` | `Q4_K` | Self-conditioning up projection |

> **Invariant: Output Weight Tying**: There is no dedicated `output.weight` tensor in this artifact. The final logit projection uses `token_embd.weight` directly.

---

## 4. Transformer Block Architecture (30 Blocks)

DiffusionGemma alternates between two block types based on `sliding_window_pattern`:

- **Sliding-Window Attention (SWA) Layers** (25 layers): Layers $0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28$.
- **Global Attention Layers** (5 layers): Every 6th block: Layers $5, 11, 17, 23, 29$.

### 4.1 Block Tensor Inventory

Each SWA block contains **23 tensors**.
Each Global block contains **22 tensors** (omits `attn_v.weight`).

Sanity check: $7 + 25 \times 23 + 5 \times 22 = 692$, matching the tensor count in §1. A reader can therefore confirm this inventory is complete rather than trusting it.

> **Runtime mapping gap**: the table below lists 23 tensors per block, but a conforming runtime structure must expose *all* of them. If any of `post_ffw_norm`, `post_ffw_norm_1`, `post_ffw_norm_2`, `pre_ffw_norm_2`, `enc_layer_output_scale`, or `layer_output_scale` is absent from the runtime, slots 11 and 13–15 of the ordering table in [`transformer.md`](transformer.md) §4.3 cannot be executed. Six per-block tensors are commonly skipped; none of them are optional for parity.

| Component | Tensor Pattern | Shape (SWA) | Shape (Global) | Type | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Attention Norm** | `blk.{i}.attn_norm.weight` | `[2816]` | `[2816]` | `F32` | Pre-attention RMSNorm |
| **Query Projection** | `blk.{i}.attn_q.weight` | `[2816, 4096]` | `[2816, 8192]` | `Q4_K` | $16 \times 256$ (SWA) vs $16 \times 512$ (Global) |
| **Query Head Norm** | `blk.{i}.attn_q_norm.weight` | `[256]` | `[512]` | `F32` | Per-head RMSNorm for Query |
| **Key Projection** | `blk.{i}.attn_k.weight` | `[2816, 2048]` | `[2816, 1024]` | `Q4_K` | $8 \times 256$ (SWA) vs $2 \times 512$ (Global) |
| **Key Head Norm** | `blk.{i}.attn_k_norm.weight` | `[256]` | `[512]` | `F32` | Per-head RMSNorm for Key |
| **Value Projection** | `blk.{i}.attn_v.weight` | `[2816, 2048]` | *OMITTED* | `Q6_K` | In Global layers, **Key $K$ is reused as Value $V$** |
| **Attention Output** | `blk.{i}.attn_output.weight` | `[4096, 2816]` | `[8192, 2816]` | `Q4_K` | Linear projection back to $d_{\text{model}}$ |
| **Post-Attn Norm** | `blk.{i}.post_attention_norm.weight`| `[2816]` | `[2816]` | `F32` | RMSNorm post attention |
| **Shared FFN Norm** | `blk.{i}.ffn_norm.weight` | `[2816]` | `[2816]` | `F32` | Pre-FFN RMSNorm |
| **Shared FFN Gate** | `blk.{i}.ffn_gate.weight` | `[2816, 2112]` | `[2816, 2112]` | `Q4_K` | Gate projection for the configured dense MLP activation |
| **Shared FFN Up** | `blk.{i}.ffn_up.weight` | `[2816, 2112]` | `[2816, 2112]` | `Q4_K` | Up projection for the configured dense MLP activation |
| **Shared FFN Down** | `blk.{i}.ffn_down.weight` | `[2112, 2816]` | `[2112, 2816]` | `Q8_0` | FFN down projection |
| **MoE Router Gate** | `blk.{i}.ffn_gate_inp.weight` | `[2816, 128]` | `[2816, 128]` | `F32` | Top-8 expert router weights |
| **MoE Router Scale**| `blk.{i}.ffn_gate_inp.scale` | `[2816]` | `[2816]` | `F32` | Router input normalizer |
| **MoE Experts Gate/Up** | `blk.{i}.ffn_gate_up_exps.weight` | `[2816, 1408, 128]`| `[2816, 1408, 128]`| `Q4_K` | Fused Gate + Up for 128 experts |
| **MoE Experts Down**| `blk.{i}.ffn_down_exps.weight`| `[704, 2816, 128]`| `[704, 2816, 128]`| `Q8_0` | Down projection for 128 experts |
| **MoE Down Scale** | `blk.{i}.ffn_down_exps.scale` | `[128]` | `[128]` | `F32` | Per-expert output scaling factor |
| **Post-FFN Norm 0** | `blk.{i}.post_ffw_norm.weight`| `[2816]` | `[2816]` | `F32` | Post-FFN residual RMSNorm |
| **Post-FFN Norm 1** | `blk.{i}.post_ffw_norm_1.weight`| `[2816]` | `[2816]` | `F32` | Additional post-FFN norm |
| **Post-FFN Norm 2** | `blk.{i}.post_ffw_norm_2.weight`| `[2816]` | `[2816]` | `F32` | Residual norm component |
| **Pre-FFN Norm 2** | `blk.{i}.pre_ffw_norm_2.weight` | `[2816]` | `[2816]` | `F32` | Residual norm component |
| **Enc Scale** | `blk.{i}.enc_layer_output_scale.weight`| `[1]` | `[1]` | `F32` | Encoder output scaling scalar |
| **Layer Scale** | `blk.{i}.layer_output_scale.weight` | `[1]` | `[1]` | `F32` | Final layer residual scale scalar |

---

## 5. Critical Architectural Invariants for Runtime

1. **Global Layer $V$-sharing**:
   In blocks $5, 11, 17, 23, 29$, `blk.{i}.attn_v.weight` is not present in this artifact. The key projection weights are reused for the value projection. Preserve the reference's separate K and V processing after projection; do not assume the normalized/rotated key tensor is itself the value tensor.

2. **Dual RoPE Dimensions and Bases**:
   - SWA layers use frequency base `10,000.0` with RoPE dimension `256`.
   - Global layers use frequency base `1,000,000.0` with RoPE dimension `512`.

3. **Per-Head Attention Normalization**:
   Before computing attention dot-products, Queries and Keys are normalized across their head dimension using `attn_q_norm` and `attn_k_norm` RMSNorm scales.

4. **MoE Topology**:
   Every layer contains a shared dense FFN (intermediate dim `2112`) and an MoE branch with 128 experts (intermediate dim `704`, top-8 routing). Activation functions and residual/norm ordering are defined by the model configuration and reference implementation, not by tensor shapes alone.

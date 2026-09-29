# Zeno — Specification: Transformer Architecture & Layer Mechanics

## 1. Scope & Structural Parameters

This document records the target layer mechanics for the **DiffusionGemma** transformer backbone. Treat the equations below as source-derived requirements to verify against a pinned reference and the named GGUF artifact; do not implement from the tensor inventory alone.

All structural parameters, head configurations, and routing dimensions are derived from `models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (as cataloged in [`specs/gguf-mapping.md`](gguf-mapping.md)).

### Core Model Dimensions

| Parameter | Value | Mathematical Symbol |
| :--- | :--- | :--- |
| Layers ($L$) | `30` | $l \in \{0, 1, \dots, 29\}$ |
| Model Dimension | `2816` | $d_{\text{model}}$ |
| Query Heads | `16` | $H_Q$ |
| SWA KV Heads | `8` | $H_{KV,\text{swa}}$ |
| Global KV Heads | `2` | $H_{KV,\text{global}}$ |
| SWA Head Dimension | `256` | $d_{k,\text{swa}} = d_{v,\text{swa}}$ |
| Global Head Dimension | `512` | $d_{k,\text{global}} = d_{v,\text{global}}$ |
| Shared FFN Hidden Dim | `2112` | $d_{\text{ffn,shared}}$ |
| MoE Expert Hidden Dim | `704` | $d_{\text{ffn,expert}}$ |
| Total Experts | `128` | $E$ |
| Active Experts per Token | `8` | $K_{\text{exp}}$ |
| RMSNorm Epsilon | `1e-6` | $\epsilon$ |
| Final Logit Softcapping | `30.0` | $C_{\text{cap}}$ |

---

## 2. Normalization: Gemma RMSNorm

All normalization layers in DiffusionGemma utilize Root Mean Square Normalization without learned bias:

$$\text{RMSNorm}(x, \gamma) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

where $\epsilon = 10^{-6}$ and $\gamma \in \mathbb{R}^d$ is the scale parameter stored in GGUF as `F32`.

### Normalization Placement per Block
Each block $l$ applies RMSNorm at distinct stages:
1. `blk.{l}.attn_norm`: Pre-attention input normalization.
2. `blk.{l}.attn_q_norm` and `attn_k_norm`: Per-head normalization applied to Query and Key vectors after projection.
3. `blk.{l}.post_attention_norm`: Applied to the attention sub-layer output before residual addition.
4. `blk.{l}.ffn_norm`: Pre-FFN normalization applied to the attention residual output.
5. `blk.{l}.post_ffw_norm`, `post_ffw_norm_1`, `post_ffw_norm_2`, `pre_ffw_norm_2`: Residual and post-combination scaling norms.

---

## 3. Attention Sub-layer

DiffusionGemma has distinct execution modes: causal prompt prefill/encoder execution and bidirectional decoder/denoising execution. The decoder reads KV state established by prefill and uses self-conditioning on later denoising steps. Attention behavior therefore depends on the execution phase; “bidirectional attention over the canvas” is not a complete forward-pass specification.

> **Windowing is inoperative at Phase 3.** `sliding_window = 1024` against `canvas_length = 256`
> ([`gguf-mapping.md`](gguf-mapping.md) §2): the window never binds. Do not implement sliding-window
> bookkeeping for Phase 3, and do not treat its absence as a parity gap. Recorded as U-20 in
> [`UNRESOLVED.md`](UNRESOLVED.md).

### 3.1 Layer Specialization: SWA vs. Global

The 30 blocks follow a recurring 5:1 pattern:
- **Blocks $\{0..4, 6..10, 12..16, 18..22, 24..28\}$ (25 layers)**: Sliding-Window Attention (SWA).
- **Blocks $\{5, 11, 17, 23, 29\}$ (5 layers)**: Global Full Attention.

```text
[0..4 SWA] ──► [5 Global] ──► [6..10 SWA] ──► [11 Global] ──► ... ──► [29 Global]
```

### 3.2 Projections and Dimensions

#### SWA Layers ($H_Q = 16, H_{KV} = 8, d_k = 256$):
* $Q = x \cdot W_Q$ where $W_Q \in \mathbb{R}^{2816 \times 4096}$ ($16 \text{ heads} \times 256$).
* $K = x \cdot W_K$ where $W_K \in \mathbb{R}^{2816 \times 2048}$ ($8 \text{ heads} \times 256$).
* $V = x \cdot W_V$ where $W_V \in \mathbb{R}^{2816 \times 2048}$ ($8 \text{ heads} \times 256$).

#### Global Layers ($H_Q = 16, H_{KV} = 2, d_k = 512$):
* $Q = x \cdot W_Q$ where $W_Q \in \mathbb{R}^{2816 \times 8192}$ ($16 \text{ heads} \times 512$).
* $K = x \cdot W_K$ where $W_K \in \mathbb{R}^{2816 \times 1024}$ ($2 \text{ heads} \times 512$).
* **Shared Key/Value Projection**: In Global layers, `attn_v.weight` is omitted from this artifact. The value projection reuses the key projection weights. The value path and key path may apply different normalization/position operations; do not alias the post-normalization K tensor as V without matching the reference order.

### 3.3 Rotary Position Embeddings (RoPE)

RoPE is applied to Query and Key head vectors with dual frequency bases:
* **SWA Layers**: Base frequency $\theta_{\text{swa}} = 10,000.0$, dimension $d_{\text{rope}} = 256$.
* **Global Layers**: Base frequency $\theta_{\text{global}} = 1,000,000.0$, dimension $d_{\text{rope}} = 512$.
* **Coordinate pairing**: NeoX-style half-split. For rotary dimension $d$, rotate coordinate pairs $(x_i, x_{i+d/2})$ using inverse frequency $\theta^{-2i/d}$. Interleaved adjacent-pair rotation does not match the pinned reference (U-23).

### 3.4 Attention Dot-Product & Output
1. Reshape $Q$ into $[B, L, H_Q, d_k]$ and $K, V$ into $[B, L, H_{KV}, d_k]$.
2. Apply per-head normalization and RoPE in the reference-defined order. For shared K/V weights, preserve distinct K and V processing.
3. Apply the configured attention mask and KV read/write mode for the current encoder or decoder phase; do not assume full-canvas unmasked attention for every phase.
4. Expand $K, V$ via Grouped-Query Attention (GQA) factor ($2$ for SWA, $8$ for Global), as confirmed for the selected model artifact.
   > **Resolved (U-08):** use blocked repetition, $\mathrm{kv}(q) = q \div (H_Q / H_{KV})$, as in the pinned Gemma4 reference.
> **Resolved (U-09):** this artifact has no attention-logit softcap metadata; final-logit softcapping is specified separately in [`UNRESOLVED.md`](UNRESOLVED.md).

5. Compute attention scores under the phase's mask:
   $$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
6. Project back to model dimension:
   $$O = \text{Attn}(Q, K, V) \cdot W_O \quad \text{where } W_O \in \mathbb{R}^{(H_Q \cdot d_k) \times 2816}$$

---

## 4. Feed-Forward & MoE Sub-layer

The inspected artifact has a shared dense branch and a routed MoE branch. Implement their activation functions, router preprocessing, and residual/norm ordering from the selected model config and reference; the following pseudocode is not sufficient as an independent implementation contract.

### 4.1 Shared FFN Branch
The shared FFN uses the activation configured for the model's dense MLP. Do not assume SiLU/Swish from the tensor names; record and use the artifact's configured activation:
$$\text{Gate} = x \cdot W_{\text{gate}} \quad (W_{\text{gate}} \in \mathbb{R}^{2816 \times 2112})$$
$$\text{Up} = x \cdot W_{\text{up}} \quad (W_{\text{up}} \in \mathbb{R}^{2816 \times 2112})$$
$$\text{FFN}_{\text{shared}} = (\text{ConfiguredActivation}(\text{Gate}, \text{Up})) \cdot W_{\text{down}} \quad (W_{\text{down}} \in \mathbb{R}^{2112 \times 2816})$$

For the cited vLLM Gemma4 implementation, the dense branch uses the configured `hidden_activation`, resolved as `gelu_pytorch_tanh` for this model.

> **Resolved (U-07):** the dense branch uses `gelu_pytorch_tanh` (GeGLU), matching the pinned `Gemma4MLP` implementation. Do not infer this from tensor names.

### 4.2 MoE Branch (128 Experts, Top-8 Routing)

1. **Router Input and Logits**:
   Apply the reference router preprocessing (unweighted RMSNorm, root-size scaling, and the learned per-dimension scale) before projecting to expert logits. The `ffn_gate_inp.scale` tensor alone does not describe the complete router operation.

2. **Top-8 Selection**:
   Identify the 8 experts with highest routing logits:
   $$\mathcal{E} = \text{top\_8}(\text{logits}_{\text{router}})$$
   Follow the reference ordering for softmax, top-k selection, and renormalization (the vLLM Gemma4 path softmaxes across experts, selects top-k, then renormalizes):
   $$w_e = \frac{\exp(\text{logits}_{\text{router}}[e])}{\sum_{j \in \mathcal{E}} \exp(\text{logits}_{\text{router}}[j])} \quad \text{for } e \in \mathcal{E}$$

3. **Expert Evaluation**:
   For each selected expert $e \in \mathcal{E}$:
   - Retrieve expert weights from fused tensor `ffn_gate_up_exps.weight` ($2816 \times 1408 \times 128$):
     $$\text{Gate}_e = x \cdot W_{\text{gate}}^{(e)} \quad (W_{\text{gate}}^{(e)} \in \mathbb{R}^{2816 \times 704})$$
     $$\text{Up}_e = x \cdot W_{\text{up}}^{(e)} \quad (W_{\text{up}}^{(e)} \in \mathbb{R}^{2816 \times 704})$$
   - Apply the configured expert activation (the cited vLLM Gemma4 path uses `gelu_tanh`) and the checkpoint's per-expert output scaling:
     $$y_e = [\text{ExpertActivation}(\text{Gate}_e, \text{Up}_e) \cdot W_{\text{down}}^{(e)}] \times s_e$$
     where $W_{\text{down}}^{(e)} \in \mathbb{R}^{704 \times 2816}$.

4. **MoE Output Combination**:
   $$\text{FFN}_{\text{MoE}} = \sum_{e \in \mathcal{E}} w_e \cdot y_e$$

### 4.3 Norm and Residual Ordering

The verified 15-slot sequence below records the pinned decoder block order (U-06). Preserve the branch-specific norms, residuals, and layer output scale exactly.

| # | Operation | Input source | Norm applied | Residual added | Status |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | Input RMSNorm | block input $x_l$ | `attn_norm` | — | `verified` (`gemma4.py:713`) |
| 2 | Q/K/V projection | post-norm $x$ | — | — | `verified` (`gemma4.py:539-540`; Global ties $V=K$ before $K$-norm) |
| 3 | Per-head Q/K/V norm | projected Q, K, V | `attn_q_norm`, `attn_k_norm`; unweighted RMSNorm for V | — | `verified` (pinned `gemma4.py`: V norm definition and forward application) |
| 4 | RoPE | Q, K | — | — | `verified` (pinned `gemma4.py`, `is_neox_style=True`; NeoX style on Q/K; V has no RoPE) |
| 5 | Attention + $W_O$ | Q, K, V | — | — | `verified` (`gemma4.py:554-556`; blocked GQA repeat, $W_O$ projection) |
| 6 | Post-attention norm | attention output | `post_attention_norm` | — | `verified` (`gemma4.py:719`) |
| 7 | Attention residual add | slot 6 output | — | $x_l$ | `verified` (`gemma4.py:720`, produces $x_{\text{mid}}$) |
| 8 | Dense FFN input norm | slot 7 output ($x_{\text{mid}}$) | `ffn_norm` | — | `verified` (`gemma4.py:725`) |
| 9 | Dense MLP + post-norm 1 | slot 8 output | `post_ffw_norm_1` | — | `verified` (`gemma4.py:726-727`; GeGLU `gelu_pytorch_tanh`) |
| 10 | MoE router | slot 7 output ($x_{\text{mid}}$) | unweighted RMSNorm + `ffn_gate_inp.scale` | — | `verified` (`gemma4.py:360-363,732`; top-8 renormalize + `ffn_down_exps.scale`) |
| 11 | Expert branch + post-norm 2 | slot 7 output ($x_{\text{mid}}$) | `pre_ffw_norm_2` (pre), `post_ffw_norm_2` (post) | — | `verified` (`gemma4.py:730,735`; 8 experts evaluated) |
| 12 | Branch combination | slots 9 + 11 | — | — | `verified` (`gemma4.py:736`; `hidden_states_1 + hidden_states_2`) |
| 13 | Combined post-FFN norm | slot 12 output | `post_ffw_norm` | — | `verified` (`gemma4.py:740`) |
| 14 | FFN residual add | slot 13 output | — | slot 7 output ($x_{\text{mid}}$) | `verified` (`gemma4.py:741`) |
| 15 | Layer output scale | slot 14 output | — | — | `verified` (`gemma4.py:748`; `layer_output_scale.weight`) |

The exact sequence was established from [`vllm/model_executor/models/gemma4.py`](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/vllm/model_executor/models/gemma4.py) at commit `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`:

* **Slot 10** — the MoE router consumes the *attention residual* $x_{\text{mid}}$ (slot 7), not the FFN-normalized input (slot 8). Router preprocessing applies unweighted RMSNorm, scales by $d_{\text{model}}^{-0.5}$, multiplies by learned dimension scale `blk.{l}.ffn_gate_inp.scale`, projects via `blk.{l}.ffn_gate_inp.weight`, computes softmax over all 128 experts, selects top 8, renormalizes top-8 weights to sum to 1.0, and scales by per-expert `blk.{l}.ffn_down_exps.scale`.
* **Slots 8–13** — Dense MLP has `pre_feedforward_layernorm` (`ffn_norm`) before MLP and `post_feedforward_layernorm_1` (`post_ffw_norm_1`) after MLP. MoE has `pre_feedforward_layernorm_2` (`pre_ffw_norm_2`) before MoE and `post_feedforward_layernorm_2` (`post_ffw_norm_2`) after MoE. The two branches are summed (slot 12), and the sum is normalized by `post_feedforward_layernorm` (`post_ffw_norm`, slot 13).
* **Slots 14–15** — The attention residual $x_{\text{mid}}$ is added to slot 13 output (slot 14), and the layer output is scaled by `blk.{l}.layer_output_scale.weight` (slot 15). Decoder reads use `layer_output_scale`; `enc_layer_output_scale` is strictly for encoder mode.

---

> **Phase 3 behavior (U-18):** one-step reads bypass the self-conditioning MLP. This sub-layer is reserved for the multi-step path and must not be run by the single-step implementation.

## 5. Self-Conditioning Sub-layer

For multi-step discrete diffusion passes, self-conditioning uses the previous step's token probability distributions to form probability-weighted soft embeddings from the token embedding table (with the model normalizer). A gated MLP combines those soft embeddings with the current input embeddings. This is not a projection of the final hidden state directly into the next input embeddings. Follow the reference's normalization, activation, dtype, pin masking, and residual order.

The exact equation and per-layer tensor mapping must be established from the artifact and compared against a reference step before Zeno implements multi-step execution.

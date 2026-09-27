# Zeno — Specification: Transformer Architecture & Layer Mechanics

## 1. Scope & Structural Parameters

This document specifies the internal layer execution mechanics of the **DiffusionGemma** transformer backbone in Zeno.

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

DiffusionGemma operates with **non-causal bidirectional attention** over the token canvas.

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
* **Value Projection Invariant ($V = K$)**: In Global layers, `attn_v.weight` is omitted from the model file. The Value representation is explicitly tied to the Key projection:
  $$V = K$$

### 3.3 Rotary Position Embeddings (RoPE)

RoPE is applied to Query and Key head vectors with dual frequency bases:
* **SWA Layers**: Base frequency $\theta_{\text{swa}} = 10,000.0$, dimension $d_{\text{rope}} = 256$.
* **Global Layers**: Base frequency $\theta_{\text{global}} = 1,000,000.0$, dimension $d_{\text{rope}} = 512$.

### 3.4 Attention Dot-Product & Output
1. Reshape $Q$ into $[B, L, H_Q, d_k]$ and $K, V$ into $[B, L, H_{KV}, d_k]$.
2. Apply per-head RMSNorm to $Q$ and $K$.
3. Expand $K, V$ via Grouped-Query Attention (GQA) factor ($2$ for SWA, $8$ for Global).
4. Compute non-causal attention scores across all positions $i, j \in [0, L-1]$:
   $$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
5. Project back to model dimension:
   $$O = \text{Attn}(Q, K, V) \cdot W_O \quad \text{where } W_O \in \mathbb{R}^{(H_Q \cdot d_k) \times 2816}$$

---

## 4. Feed-Forward & MoE Sub-layer

Every transformer block incorporates both a **Shared FFN** and a **Routed Mixture-of-Experts (MoE)** branch.

### 4.1 Shared FFN Branch
The shared FFN provides a baseline dense representation using SwiGLU activation:
$$\text{Gate} = x \cdot W_{\text{gate}} \quad (W_{\text{gate}} \in \mathbb{R}^{2816 \times 2112})$$
$$\text{Up} = x \cdot W_{\text{up}} \quad (W_{\text{up}} \in \mathbb{R}^{2816 \times 2112})$$
$$\text{FFN}_{\text{shared}} = (\text{Swish}(\text{Gate}) \odot \text{Up}) \cdot W_{\text{down}} \quad (W_{\text{down}} \in \mathbb{R}^{2112 \times 2816})$$

where $\text{Swish}(z) = z \cdot \sigma(z)$.

### 4.2 MoE Branch (128 Experts, Top-8 Routing)

1. **Routing Logits**:
   $$\text{logits}_{\text{router}} = (x \odot \gamma_{\text{router\_scale}}) \cdot W_{\text{router\_gate}} \quad (W_{\text{router\_gate}} \in \mathbb{R}^{2816 \times 128})$$

2. **Top-8 Selection**:
   Identify the 8 experts with highest routing logits:
   $$\mathcal{E} = \text{top\_8}(\text{logits}_{\text{router}})$$
   Compute normalized softmax routing weights across the selected 8 experts:
   $$w_e = \frac{\exp(\text{logits}_{\text{router}}[e])}{\sum_{j \in \mathcal{E}} \exp(\text{logits}_{\text{router}}[j])} \quad \text{for } e \in \mathcal{E}$$

3. **Expert Evaluation**:
   For each selected expert $e \in \mathcal{E}$:
   - Retrieve expert weights from fused tensor `ffn_gate_up_exps.weight` ($2816 \times 1408 \times 128$):
     $$\text{Gate}_e = x \cdot W_{\text{gate}}^{(e)} \quad (W_{\text{gate}}^{(e)} \in \mathbb{R}^{2816 \times 704})$$
     $$\text{Up}_e = x \cdot W_{\text{up}}^{(e)} \quad (W_{\text{up}}^{(e)} \in \mathbb{R}^{2816 \times 704})$$
   - Compute SwiGLU down-projection scaled by `ffn_down_exps.scale[e]`:
     $$y_e = [(\text{Swish}(\text{Gate}_e) \odot \text{Up}_e) \cdot W_{\text{down}}^{(e)}] \times s_e$$
     where $W_{\text{down}}^{(e)} \in \mathbb{R}^{704 \times 2816}$.

4. **MoE Output Combination**:
   $$\text{FFN}_{\text{MoE}} = \sum_{e \in \mathcal{E}} w_e \cdot y_e$$

### 4.3 Total Block Residual Addition
The combined block output is computed by summing attention, shared FFN, and MoE branches with their respective layer scales:
$$x_{\text{out}} = x + O \cdot s_{\text{layer}} + (\text{FFN}_{\text{shared}} + \text{FFN}_{\text{MoE}}) \cdot s_{\text{enc}}$$

---

## 5. Self-Conditioning Sub-layer

For multi-step discrete diffusion passes, DiffusionGemma includes a dedicated global self-conditioning projection block:

$$\text{SC}(x) = [(\text{Swish}(\text{RMSNorm}(x) \cdot W_{\text{sc\_gate}}) \odot (\text{RMSNorm}(x) \cdot W_{\text{sc\_up}})) \cdot W_{\text{sc\_down}}]$$

This projects the intermediate canvas state into a conditioning vector that is injected into subsequent denoising evaluations, preserving state consistency across steps.

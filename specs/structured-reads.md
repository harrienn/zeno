# Zeno — Specification: Structured Reads & Scoring Mathematics

## 1. Purpose & Scope

This document specifies the mathematical execution and scoring pipeline for **Structured Reads** in Zeno.

A structured read is a deterministic forward evaluation of a seeded token canvas ($L = 256$) that extracts logits exclusively at a designated answer position $p$, normalizes them across a closed set of candidate tokens, and calculates Shannon entropy to evaluate decision uncertainty.

---

## 2. Forward Evaluation Pipeline

Given:
* Seeded token canvas: $X = [x_0, x_1, \dots, x_{255}] \in \{0, \dots, V-1\}^{256}$
* Pinned answer position: $p \in [0, 255]$ with $x_p = \text{MASK}$ (`4`)
* Candidate token IDs: $C = \{c_1, c_2, \dots, c_K\}$ with $c_k \in \{0, \dots, V-1\}$

### 2.1 Embedding Lookup
The token IDs are mapped to initial hidden representations:
$$H_0 = \text{Embedding}(X) \in \mathbb{R}^{256 \times 2816}$$

In accordance with Gemma scaling conventions, embeddings are scaled by $\sqrt{d_{\text{model}}} = \sqrt{2816}$:
$$H_0 = H_0 \times \sqrt{2816}$$

### 2.2 Transformer Blocks ($l = 0 \dots 29$)
For each of the 30 layers:
1. **Pre-Attention RMSNorm**:
   $$\tilde{H}_l = \text{RMSNorm}(H_l, \gamma_{\text{attn\_norm}}^{(l)})$$
   where $\text{RMSNorm}(v, \gamma) = \frac{v}{\sqrt{\frac{1}{d}\sum_{i=1}^d v_i^2 + \epsilon}} \odot \gamma$ and $\epsilon = 10^{-6}$.

2. **Bidirectional Multi-Head Attention**:
   - Compute Query $Q$, Key $K$, Value $V$ projections.
   - For Global layers ($l \in \{5, 11, 17, 23, 29\}$), $V$ is tied to $K$ ($V = K$).
   - Apply per-head RMSNorm ($\gamma_{q\_\text{norm}}$, $\gamma_{k\_\text{norm}}$) across head dimensions.
   - Apply Rotary Position Embeddings (RoPE) using base frequency $10^4$ (SWA) or $10^6$ (Global).
   - Compute non-causal full attention across all 256 positions without triangular causal masking:
     $$A_l = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   - Project output $O_l = A_l W_O$ and add residual connection:
     $$H_l' = H_l + O_l$$

3. **Feed-Forward / MoE Execution**:
   - Pre-FFN RMSNorm: $\tilde{H}_l' = \text{RMSNorm}(H_l', \gamma_{\text{ffn\_norm}}^{(l)})$.
   - Evaluate shared SwiGLU branch:
     $$\text{FFN}_{\text{shared}} = (\text{Swish}(\tilde{H}_l' W_{\text{gate}}) \odot (\tilde{H}_l' W_{\text{up}})) W_{\text{down}}$$
   - Evaluate MoE router over 128 experts:
     $$\text{logits}_{\text{router}} = \tilde{H}_l' W_{\text{router\_gate}}$$
     Select top-8 experts and compute normalized router weights.
   - Dispatch to selected experts and sum outputs: $\text{FFN}_{\text{MoE}}$.
   - Combine shared and MoE representations, apply post-FFN scaling/norms, and add residual:
     $$H_{l+1} = H_l' + \text{FFN}_{\text{shared}} + \text{FFN}_{\text{MoE}}$$

### 2.3 Final Output Normalization
After layer 29:
$$H_{\text{final}} = \text{RMSNorm}(H_{30}, \gamma_{\text{output\_norm}}) \in \mathbb{R}^{256 \times 2816}$$

---

## 3. Logit Slicing & Softcapping

Extract the hidden state vector at the pinned answer position $p$:
$$h_p = H_{\text{final}}[p, :] \in \mathbb{R}^{2816}$$

### 3.1 Candidate Logit Projection
For each candidate token ID $c_k \in C$:
$$z_k = h_p \cdot W_{\text{token\_embd}}[c_k, :]$$

### 3.2 Logit Softcapping
DiffusionGemma applies final logit softcapping with cap parameter $C_{\text{cap}} = 30.0$ (from GGUF metadata `diffusion-gemma.final_logit_softcapping`):
$$z_k' = 30.0 \times \tanh\left(\frac{z_k}{30.0}\right)$$

---

## 4. Scoring Mathematics

Given softcapped candidate logits $[z_1', z_2', \dots, z_K']$ and temperature $T$ (default: $1.0$):

### 4.1 Normalized Candidate Likelihoods
Apply softmax restricted to the candidate set:
$$\tilde{z}_k = z_k' / T$$
$$p_k = \frac{\exp(\tilde{z}_k - \max_j \tilde{z}_j)}{\sum_{m=1}^K \exp(\tilde{z}_m - \max_j \tilde{z}_j)}$$

The decision outcome is the argmax:
$$k^* = \arg\max_{k \in \{1 \dots K\}} p_k$$
$$\text{decision} = \text{label}(c_{k^*})$$

### 4.2 Shannon Entropy (Uncertainty Metric)
To assess whether the model is decisive or uncertain among the supplied choices, compute the Shannon entropy over the candidate distribution:
$$H(P) = -\sum_{k=1}^K p_k \ln(p_k)$$

Optionally, compute normalized entropy $H_{\text{norm}} \in [0, 1]$:
$$H_{\text{norm}}(P) = \frac{H(P)}{\ln(K)}$$
where $H_{\text{norm}} = 0$ indicates total certainty ($p_k = 1.0$), and $H_{\text{norm}} = 1.0$ indicates a uniform distribution across all $K$ options.

---

## 5. Statistical Policy: Normalization vs. Calibration

In adherence to project guidelines in [`AGENTS.md`](../AGENTS.md) and [`README.md`](../README.md):

1. **Normalized scores ($p_k$) are not guaranteed probabilities**:
   Softmax over candidate logits produces numbers in $[0, 1]$ that sum to $1$, but does **not** establish empirical calibration. A score of $0.95$ does not imply that the model is correct $95\%$ of the time.
2. **Empirical calibration requires labeled evaluation**:
   Calibration must be empirically measured on labeled domain datasets using metrics such as Expected Calibration Error (ECE) and Brier scores.
3. **Entropy flags dispersion, not ground truth**:
   Shannon entropy measures dispersion over the candidate options. High entropy indicates ambiguity among the provided choices, which can be used to trigger adaptive follow-up reads or human review.

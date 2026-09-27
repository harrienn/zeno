# Zeno — Specification: Structured Reads & Scoring Mathematics

## 1. Purpose & Scope

This document specifies candidate scoring for **Structured Reads**. It is not yet a complete executable model-forward specification; validate model phases and tensor operations against a pinned reference fixture before implementing the forward path.

A structured read extracts logits at selected answer positions of a seeded canvas, restricts scoring to the requested candidate token IDs, and computes candidate-normalized scores and entropy. The request must provide a reproducible seeded canvas if deterministic results are required.

---

## 2. Forward Evaluation Pipeline

Given:
* A seeded canvas $X$ of width $L$, where $L$ comes from model/request configuration.
* An answer position $p$ selected for logit readout. Its seed token is determined by the structured-read procedure; pinning means preserving that seed token across denoising updates and does not require `x_p = MASK`.
* Distinct candidate token IDs $C = \{c_1, c_2, \dots, c_K\}$ resolved at the answer boundary by the model tokenizer.

### 2.1 Model execution phases

The reference execution path is not a single undifferentiated bidirectional pass:

1. **Causal prefill/encoder phase** processes the prompt and establishes per-layer KV state.
2. **Bidirectional decoder/denoising phase** evaluates the canvas while reading the encoder KV state. Multi-step execution also carries self-conditioning state between steps.
3. **Read-only structured read** limits the requested work to the configured step cap and returns logits from the read step without the normal commit forward pass.

See [transformer.md](transformer.md) for the model layer structure. Exact attention masks, canvas seeding, embedding scaling, output norm placement, and one-step state initialization remain parity requirements; do not infer them from the abbreviated equations in this document.

---

## 3. Logit Slicing & Softcapping

Extract the hidden state vector at each answer position $p$:
$$h_p = H_{\text{final}}[p, :] \in \mathbb{R}^{2816}$$

### 3.1 Candidate Logit Projection
For each candidate token ID $c_k \in C$:
$$z_k = h_p \cdot W_{\text{token\_embd}}[c_k, :]$$

### 3.2 Logit Softcapping
When configured by the selected model artifact, apply final logit softcapping in fp32 with cap parameter $C_{\text{cap}} = 30.0$ (from GGUF metadata `diffusion-gemma.final_logit_softcapping`):
$$z_k' = 30.0 \times \tanh\left(\frac{z_k}{30.0}\right)$$

---

## 4. Scoring Mathematics

Given softcapped candidate logits $[z_1', z_2', \dots, z_K']$ and finite temperature $T > 0$ (default: $1.0$):

### 4.1 Candidate-Normalized Scores
Apply softmax restricted to the candidate set:
$$\tilde{z}_k = z_k' / T$$
$$p_k = \frac{\exp(\tilde{z}_k - \max_j \tilde{z}_j)}{\sum_{m=1}^K \exp(\tilde{z}_m - \max_j \tilde{z}_j)}$$

The decision outcome is the argmax:
$$k^* = \arg\max_{k \in \{1 \dots K\}} p_k$$
$$\text{decision} = \text{label}(c_{k^*})$$

### 4.2 Shannon Entropy (Uncertainty Metric)
To assess whether the model is decisive or uncertain among the supplied choices, compute the Shannon entropy over the candidate distribution:
$$H(P) = -\sum_{k=1}^K p_k \ln(p_k)$$

For thresholds comparable across candidate counts, compute normalized entropy $H_{\text{norm}} \in [0, 1]$:
$$H_{\text{norm}}(P) = \frac{H(P)}{\ln(K)}$$
where $H_{\text{norm}} = 0$ indicates total certainty ($p_k = 1.0$), and $H_{\text{norm}} = 1.0$ indicates a uniform distribution across all $K$ options.

---

## 5. Statistical Policy: Normalization vs. Calibration

In adherence to project guidelines in [`AGENTS.md`](../AGENTS.md) and [`README.md`](../README.md):

1. **Candidate-normalized scores are not empirically calibrated estimates**:
   Softmax over candidate logits defines a distribution over the supplied choices, but does **not** establish empirical calibration. A score of $0.95$ does not imply that the model is correct $95\%$ of the time.
2. **Empirical calibration requires labeled evaluation**:
   Calibration must be empirically measured on labeled domain datasets using metrics such as Expected Calibration Error (ECE) and Brier scores.
3. **Entropy flags dispersion, not ground truth**:
   Shannon entropy measures dispersion over the candidate options. High entropy indicates ambiguity among the provided choices, which can be used to trigger adaptive follow-up reads or human review.

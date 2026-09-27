# Zeno — Specification: Denoising & Adaptive Step Dynamics

## 1. Purpose & Scope

This document specifies the adaptive step-control policies, discrete diffusion update mechanics, and research experimental protocols for resolving ambiguous or high-entropy queries in Zeno.

While single-step structured reads ([`specs/structured-reads.md`](structured-reads.md)) serve as the primary execution path for System One decisions, certain queries produce high Shannon entropy at the decision slot, indicating uncertainty among candidate choices.

This specification defines:
1. The **entropy-thresholding gate** separating decisive from ambiguous results.
2. The three **experimental strategies** for resolving ambiguous queries.
3. The mathematical formulation of **discrete diffusion canvas updates** and **self-conditioning injection**.
4. The empirical research protocol for validating step policies against ground truth.

---

## 2. Adaptive Step Control & Entropy Thresholding

When a structured read is executed, the normalized Shannon entropy $H(P)$ over the candidate set is evaluated against an optional user-specified or engine-configured threshold $H_{\text{threshold}}$:

```text
[Seeded Canvas (step 1)]
           ↓
[Forward Evaluation & Logit Readout]
           ↓
[Compute Candidate Normalization & Entropy H(P)]
           ↓
    H(P) < H_threshold?
   ├── YES ──► Decisive: Return immediate decision (1 step executed)
   │
   └── NO  ──► Ambiguous: Route to configured ambiguity resolution policy
```

### 2.1 Default Policy: Deterministic 1-Step Reporting
By default, Zeno operates with `step_cap = 1`. If $H(P) \ge H_{\text{threshold}}$, the engine returns the candidate label with $p_{\text{max}}$, but sets `uncertain: true` in the response payload ([`specs/decision-api.md`](decision-api.md)). This provides immediate, low-latency signaling to the caller without speculative execution overhead.

---

## 3. Ambiguity Resolution Strategies (Research Protocol)

As established in [`ARCHITECTURE.md`](../ARCHITECTURE.md#L230-L245) and [`PLAN.md`](../PLAN.md#L155-L185), the project does not assume a single universal policy for high-entropy slots. Instead, Zeno establishes three distinct experimental strategies to be comparatively evaluated:

| Strategy | Description | Compute Cost | Hypothesized Benefit |
| :--- | :--- | :--- | :--- |
| **Strategy A: Stochastic Re-draw** | Re-seed the canvas with new pseudo-random noise at unpinned slots; average candidate scores over $N$ reads. | $N \times$ single read | Smoothes out noise artifacts without running full generative rollout. |
| **Strategy B: Discrete Diffusion Refinement** | Apply discrete diffusion update rule to the canvas; inject self-conditioning and run step 2. | $2 \times$ forward pass + update | Lets the model iteratively refine attention representation over the canvas. |
| **Strategy C: Schema Escalation** | Map ambiguous classification to a secondary sub-schema (or return to human/System Two reviewer). | 0 extra model steps | Avoids spending inference compute on fundamentally under-specified context. |

---

## 4. Discrete Diffusion Mechanics & Canvas Updates

When Strategy B (Discrete Diffusion Refinement) or multi-token structured infilling is executed, the canvas state evolves via discrete diffusion.

### 4.1 Confidence-Bounded Token Snapping
In discrete diffusion language modeling, positions are updated from noise/mask placeholders into concrete tokens based on categorical probability distributions:

1. For each position $i \in [0, L-1]$:
   - Compute token probabilities across vocabulary: $P_i = \text{Softmax}(z_i \in \mathbb{R}^{262144})$.
   - Calculate maximum token confidence: $c_i = \max_{v} P_i[v]$.
2. Identify token candidate: $\hat{x}_i = \arg\max_{v} P_i[v]$.
3. Apply threshold or annealing schedule:
   - If position $i$ is an unpinned mask slot and $c_i > \tau_{\text{confidence}}$, update token:
     $$x_i^{(t+1)} = \hat{x}_i$$
   - Otherwise, retain mask token:
     $$x_i^{(t+1)} = \text{MASK} \quad (4)$$

> **Invariant**: Pinned answer positions and prompt context tokens are governed by schema masking rules and are never overwritten by arbitrary generative tokens.

### 4.2 Self-Conditioning Injection
DiffusionGemma incorporates global self-conditioning tensors (`self_cond_*` as documented in [`specs/gguf-mapping.md`](gguf-mapping.md)) to carry forward latent state across denoising steps:

1. At the conclusion of step $t$, the post-norm hidden state $H_{\text{final}} \in \mathbb{R}^{256 \times 2816}$ is projected:
   $$H_{\text{norm}} = \text{RMSNorm}(H_{\text{final}}, \gamma_{\text{sc\_norm}})$$
   $$H_{\text{gate}} = H_{\text{norm}} \cdot W_{\text{sc\_gate}} \quad (W_{\text{sc\_gate}} \in \mathbb{R}^{2816 \times 2112})$$
   $$H_{\text{up}} = H_{\text{norm}} \cdot W_{\text{sc\_up}} \quad (W_{\text{sc\_up}} \in \mathbb{R}^{2816 \times 2112})$$
   $$H_{\text{sc}} = (\text{Swish}(H_{\text{gate}}) \odot H_{\text{up}}) \cdot W_{\text{sc\_down}} \quad (W_{\text{sc\_down}} \in \mathbb{R}^{2112 \times 2816})$$

2. At step $t+1$, $H_{\text{sc}}$ is injected directly into the initial token embeddings:
   $$H_0^{(t+1)} = \text{Embedding}(X^{(t+1)}) \cdot \sqrt{2816} + H_{\text{sc}}$$

This provides temporal continuity across multi-step iterations without maintaining an autoregressive KV cache.

---

## 5. Verification & Acceptance Criteria

Before any adaptive multi-step policy is enabled as default runtime behavior in Phase 4:
1. **Empirical Accuracy Evaluation**: Evaluate Strategy A vs. Strategy B on a labeled ambiguous dataset. The multi-step policy must demonstrate statistically significant accuracy improvement over deterministic 1-step readout.
2. **Latency Trade-off Verification**: Document the exact compute cost (e.g. $2\times$ latency for 2 steps).
3. **Reproducibility**: Experiments must record model artifact revision, dataset provenance, and exact entropy threshold values.

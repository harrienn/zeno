# Zeno — Specification: Denoising & Adaptive Step Dynamics

## 1. Purpose & Scope

This document separates reference DiffusionGemma denoising behavior from proposed experiments. It is not an implementation contract for Zeno until the behavior is reproduced against a pinned reference revision.

Candidate-set entropy in a System One response measures dispersion among the requested choices. The model's denoising sampler uses per-canvas-position distributions and its own confidence/entropy controls; these quantities and thresholds are distinct.

This document records:
1. The distinction between API uncertainty reporting and model denoising controls.
2. The reference sampler behavior at a high level.
3. Candidate experiments and parity requirements.

---

## 2. Adaptive Step Control & Entropy Thresholding

The API may compare normalized candidate entropy with a caller-supplied threshold to set its `uncertain` response field. This reporting decision does not define the model's denoising schedule or enable additional steps.

```text
[Seeded Canvas (step 1)]
           ↓
[Forward Evaluation & Logit Readout]
           ↓
[Compute Candidate Normalization & Entropy H(P)]
           ↓
    normalized candidate entropy < API threshold?
   ├── YES ──► Decisive: Return immediate decision (1 step executed)
   │
   └── NO  ──► Ambiguous: Route to configured ambiguity resolution policy
```

### 2.1 Initial API Policy
The initial API supports one read. If the caller supplies an entropy threshold and normalized candidate entropy meets or exceeds it, set `uncertain: true`; otherwise return the result without claiming calibrated confidence. Multi-step policy remains disabled until specified and validated.

---

## 3. Ambiguity Resolution Strategies (Research Protocol)

The following are proposed experiments, not established DiffusionGemma mechanics or committed product behavior:

| Strategy | Description | Compute Cost | Hypothesized Benefit |
| :--- | :--- | :--- | :--- |
| **Strategy A: Stochastic Re-draw** | Re-seed the canvas with new pseudo-random noise at unpinned slots; average candidate scores over $N$ reads. | $N \times$ single read | Smoothes out noise artifacts without running full generative rollout. |
| **Strategy B: Discrete Diffusion Refinement** | Apply discrete diffusion update rule to the canvas; inject self-conditioning and run step 2. | $2 \times$ forward pass + update | Lets the model iteratively refine attention representation over the canvas. |
| **Strategy C: Schema Escalation** | Map ambiguous classification to a secondary sub-schema (or return to human/System Two reviewer). | 0 extra model steps | Avoids spending inference compute on fundamentally under-specified context. |

---

## 4. Reference Sampler Behavior (High-Level; Pin a Revision)

The vLLM reference computes token-level statistics over model logits, applies an entropy-bound acceptance policy, re-noises positions that are not accepted, and tracks convergence before committing the resulting canvas. Its exact sampler configuration and stochastic choices must be copied from a pinned source revision and validated; a per-token confidence-threshold-and-argmax rule is not an equivalent substitute. See the [reference implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/diffusion_gemma.py).

### 4.1 Self-Conditioning

The reference derives soft embeddings from the previous step's token probability distributions and embedding weights, then passes them with the current input embeddings through a gated self-conditioning MLP. Pinned positions retain their seed token; the reference also zeros their self-conditioning soft embedding. This is not equivalent to projecting the final hidden state and adding that projection directly to token embeddings. The exact weight names, normalization, scaling, and dtype order must be verified against the model implementation and GGUF artifact.

### 4.2 Pinning and Read-Only Requests

The reference's pinned positions preserve their seeded token values through denoising steps. A read-only request emits from the read step and skips the ordinary commit forward pass. Neither behavior implies that answer positions must be seeded with `MASK`; see [canvas-templates.md](canvas-templates.md).

---

## 5. Verification & Acceptance Criteria

Before implementing multi-step behavior:
1. Pin model artifact and reference commits; capture the seed canvas, model outputs, per-token entropy, accepted positions, and self-conditioning inputs for a small fixture.
2. Compare candidate follow-up strategies on the same labeled examples and report uncertainty intervals, accuracy, calibration, and latency. Do not require or claim an improvement until measured.
3. Record RNG seeds, sampler parameters, dataset provenance, and whether each threshold is per-candidate or per-token entropy.

---

## 6. Phase 9 Implementation: Adaptive Denoising & Gated Self-Conditioning

### 6.1 Gated Self-Conditioning MLP Architecture ($H_{\text{sc}}$)

In multi-step structured reads ($T \in [2, 4]$), subsequent forward passes inject a self-conditioning signal $H_{\text{sc}}$ into the canvas token embedding at answer position $p$:

1. **Candidate-Weighted Soft Embedding**:
   $$E_{\text{cand}} = \sum_{c=1}^{|C|} P(c) \cdot W_{\text{emb}}[t_c] \in \mathbb{R}^{d_{\text{model}}}$$
   where $P(c)$ is the candidate softmax probability distribution from step $t-1$, and $W_{\text{emb}}[t_c]$ is the token embedding vector of candidate token $t_c$.
2. **Pre-Layer Normalization**:
   $$\tilde{E} = \text{RMSNorm}(E_{\text{cand}}, w_{\text{sc\_pre\_norm}})$$
3. **Gated Intermediate Projection & GeGLU Activation**:
   $$G = \tilde{E} \cdot W_{\text{sc\_gate}}, \quad U = \tilde{E} \cdot W_{\text{sc\_up}} \quad \in \mathbb{R}^{d_{\text{inter}}}$$
   $$M = \text{GeGLU}(G, U) = G \odot \text{GELU}_{\text{tanh}}(U)$$
   where $d_{\text{inter}} = 2112$.
4. **Down Projection & Residual Connection**:
   $$D = M \cdot W_{\text{sc\_down}} \in \mathbb{R}^{d_{\text{model}}}$$
   $$R = E_{\text{cand}} + D$$
5. **Post-Layer Normalization & Injection**:
   $$H_{\text{sc}} = \text{RMSNorm}(R) \quad (\text{unweighted unit scaling})$$
   At step $t > 1$, the canvas token embedding at answer position $p$ is modified as:
   $$E_{\text{canvas}}[p] \leftarrow E_{\text{canvas}}[p] + H_{\text{sc}}$$

### 6.2 Adaptive Early-Stopping Dynamics

The decision engine loops over steps $t \in [1, \text{step\_cap}]$:
1. Execute structured read forward pass for step $t$.
2. Compute candidate-restricted softmax distribution $P_t$, Shannon entropy $H(P_t)$, and normalized entropy $H_{\text{norm}}(P_t) = \frac{H(P_t)}{\ln(|C|)}$.
3. **Early Termination Check**:
   - If `entropy_threshold` $\tau$ is configured and $H_{\text{norm}}(P_t) < \tau$: terminate loop immediately with `uncertain: false` and `steps_executed: t`.
   - If $t = \text{step\_cap}$ and $H_{\text{norm}}(P_t) \ge \tau$: terminate loop with `uncertain: true` and `steps_executed: step_cap`.
4. If $t < \text{step\_cap}$ and $H_{\text{norm}}(P_t) \ge \tau$: compute $H_{\text{sc}}$ from $P_t$ and propagate to step $t+1$.

### 6.3 Empirical Evaluation & Parity Parity (`maintenance_priority_uncertain.json`)

On the high-entropy query `specs/fixtures/maintenance_priority_uncertain.json` with 3 candidates (Low, Medium, High), candidate entropy was evaluated across CPU and dual-GPU CUDA backends:

| Metric | Step 1 ($T=1$) Baseline | Step 2 ($T=2$) CPU Reference | Step 2 ($T=2$) Dual RTX 3060 CUDA | $\Delta_{\text{CPU-CUDA}}$ |
| :--- | :--- | :--- | :--- | :--- |
| **P(Low)** | 0.6546 | 0.9643 | 0.9645 | +0.0002 |
| **P(Medium)** | 0.3437 | 0.0357 | 0.0355 | -0.0002 |
| **P(High)** | 0.0017 | 0.0000 | 0.0000 | 0.0000 |
| **Raw Logit (Low)** | 13.4150 | 18.5881 | 18.5973 | +0.0092 |
| **Raw Logit (Medium)** | 12.6348 | 14.2045 | 14.2009 | -0.0036 |
| **Raw Logit (High)** | 9.4087 | 4.8161 | 4.8213 | +0.0052 |
| **$H_{\text{norm}}$** | 0.5964 | 0.1404 | 0.1398 | -0.0006 |
| **`uncertain` ($\tau=0.5$)** | `true` | `false` | `false` | `exact` |
| **Steps Executed** | 1 | 2 | 2 | `exact` |

**Empirical Observations**:
1. **Entropy Collapse**: Self-conditioning collapsed normalized candidate entropy by **76.6%** (from 0.5964 down to 0.1398–0.1404), resolving the classification ambiguity in favor of `Low`.
2. **CPU-CUDA Parity**: Logit divergence between CPU FP32 reference and dual RTX 3060 CUDA execution remained $< 0.01$, and candidate probabilities agreed to within $0.0002$.
3. **Adaptive Early Stopping**: When executed with `step_cap=3` and `entropy_threshold=0.5`, the engine automatically terminated at `steps_executed: 2`, avoiding the latency cost of step 3 while ensuring decision decisiveness.


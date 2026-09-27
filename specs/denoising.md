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

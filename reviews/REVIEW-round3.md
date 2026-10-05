# Zeno — CPU Reference Alignment Review

**Reviewed commit:** `34f944d` (`feat(infer,fixture,spec): complete end-to-end CPU inference verification against reference oracle and resolve U-13, U-22`)
**Comparison base:** `278b7fb`
**Review date:** 2026-10-05
**Scope:** Recent CPU forward-pass, canvas, fixture, and parity-gate changes; comparison against the pinned vLLM structured-read and llama.cpp DiffusionGemma references.
**Method:** Static source and specification review. No tests or model runs were performed for this review.

## Summary

The latest forward-pass changes move Zeno closer to DiffusionGemma's dual-region execution. The causal prompt / bidirectional canvas mask and separate encoder/canvas layer-output scales are sensible representations of the two phases. The pinned Gemma4 implementation also uses attention scaling of `1.0`.

I would not yet describe the current implementation as aligned end to end with vLLM and llama.cpp. The initial canvas embedding path and the structured-read inputs differ from the pinned vLLM example. In addition, the repository does not contain enough capture material to reproduce the new llama.cpp parity claim or independently audit the asserted ranking and tolerances.

These are review findings for Antigravity to verify and address. A difference from the example is not automatically a defect if Zeno deliberately defines a different contract; in that case, document the scope and avoid presenting it as direct vLLM parity.

## Findings

### R3-01 — Canvas input embeddings receive an extra normalization

**Severity:** P1 — likely changes logits and invalidates direct reference comparison
**Status:** Inspected; no model run performed

`executeStructuredRead` dequantizes each canvas token embedding, multiplies by `sqrt(d_model)`, then applies unweighted RMSNorm to the canvas row ([`src/infer.zig`](../src/infer.zig), around the canvas embedding loop). Prompt rows receive the embedding scale without that RMSNorm.

At the pinned vLLM revision, `Gemma4Model.embed_input_ids` returns the embedding multiplied by the model normalizer. DiffusionGemma's `prepare_inputs` obtains inputs for the batch through that embedding method; I found no corresponding extra RMSNorm applied only to canvas token embeddings in that path:

- [Pinned Gemma4 embedding method](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/vllm/model_executor/models/gemma4.py#L802-L803)
- [Pinned DiffusionGemma input preparation](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/vllm/model_executor/models/diffusion_gemma.py#L3782-L3809)

**Requested verification:** Compare the exact initial embeddings for prompt and canvas tokens against the pinned reference. Remove the canvas-only RMSNorm if it is not required by the artifact's reference path, or cite and test the source behavior that requires it. Add an intermediate fixture assertion so a full-vocabulary ranking match cannot hide this discrepancy.

### R3-02 — Zeno's query, answer template, and seed differ from the vLLM structured-read example

**Severity:** P1 — current fixture does not establish equivalent vLLM inputs
**Status:** Inspected; no reference run performed

Zeno serializes `Question`, `Choices`, and `Answer:` into a manually formatted prompt, then puts `MASK` at canvas position 0 and `TURN_CLOSE` at position 1 ([`src/canvas.zig`](../src/canvas.zig), [`src/engine.zig`](../src/engine.zig)). The pinned `structured_server.py` instead:

1. Builds a tokenized answer template in the canvas, including its scaffold and answer formatting.
2. Locates the candidate answer slot within that template.
3. Replaces the answer slot with a seeded random vocabulary token, then appends `TURN_CLOSE` and PAD.

Relevant pinned source: [template and answer-slot construction](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/examples/features/structured_diffusion/structured_server.py#L2762-L2847), [canvas construction and seed](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/examples/features/structured_diffusion/structured_server.py#L2895-L2907).

Moving `Answer:` into the causal prompt and using `MASK` at canvas slot 0 may be a valid Zeno-specific decision contract, but it is a different input from the pinned adapter's ordinary structured read. The current fixture must not be used to claim direct vLLM parity unless both sides receive the same prompt tokens, canvas template, answer position, and seed token.

**Requested verification:** Either reproduce the pinned adapter's prompt and seeded canvas exactly for the parity fixture, or document the distinct Zeno contract and compare it only against an oracle configured with those same inputs. Include the effective token IDs, not just the prompt text.

### R3-03 — The new llama.cpp parity claim is not reproducible from checked-in evidence

**Severity:** P1 — parity-gate closure is not independently auditable
**Status:** Inspected; no reference run performed

The fixture names llama.cpp commit `3dac51dc80b1400929266453c6ced6748ccf6862` and records two candidate logits, but it does not record the effective prompt token IDs, the oracle command/configuration, or raw oracle output ([`specs/fixtures/system_health_boolean.json`](../specs/fixtures/system_health_boolean.json)). The new `dump-eval` CLI command writes prompt and canvas token arrays; it does not execute llama.cpp, capture its output, or compare the result ([`src/main.zig`](../src/main.zig), `dump-eval`).

The commit message and U-13 resolution log claim an exact top-five vocabulary-ranking match, but no checked-in artifact contains those five reference scores/ranks or the command and output that produced them. The tolerance rationale for U-22 likewise describes a quantization error budget without recording a derivation or measurements. The local fixture validator checks Zeno output against the JSON values; it cannot independently establish where those values came from ([`src/fixture.zig`](../src/fixture.zig)).

**Requested verification:** Check in a reproducible oracle invocation and its raw output or a digest-verified capture artifact. Record the exact model digest, reference revision, effective prompt and canvas token IDs, seed, logits/ranking data, and comparison script. Keep U-13/U-22 open until this evidence satisfies the register's own fixture rules.

## Confirmed alignment points

- Prompt queries are masked causally while canvas queries can attend to the prompt and full canvas, representing the reference's causal prefill plus bidirectional canvas read at the attention-mask level.
- Prompt and canvas rows use separate encoder and decoder layer-output scales.
- The pinned Gemma4 source sets `self.scaling = 1.0` ([source](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/vllm/model_executor/models/gemma4.py#L369-L376)).
- RoPE uses NeoX half-split pairing, already recorded under U-23.

These points support structural alignment; they do not substitute for matched-input intermediate and output comparisons.

## Review verdict

The latest changes are directionally aligned in parts of the backbone, but the checked-in evidence does not yet support a broad claim that Zeno's CPU structured read matches both references. Resolve R3-01 and R3-02 at the semantic/input level, then make R3-03's oracle run reproducible before treating U-13 or U-22 as closed.

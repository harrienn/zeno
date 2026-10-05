#!/usr/bin/env python3
"""Zeno — Reference Oracle Parity & Regression Verifier.

Compares raw binary logit dumps from llama.cpp's llama-diffusion-gemma-eval
against Zeno's forward pass outputs and golden fixture specifications.

Usage:
    python3 tools/verify_oracle_parity.py <out_logits.bin> [fixture.json]
"""

import json
import sys
import numpy as np

VOCAB_SIZE = 262_144
LOGIT_SOFTCAP = 30.0

def softcap(x, cap=LOGIT_SOFTCAP):
    return cap * np.tanh(x / cap)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <out_logits.bin> [fixture.json]")
        sys.exit(1)

    bin_path = sys.argv[1]
    fixture_path = sys.argv[2] if len(sys.argv) > 2 else "specs/fixtures/system_health_boolean.json"

    with open(fixture_path, "r") as f:
        fixture = json.load(f)

    with open(bin_path, "rb") as f:
        # Read the first canvas position (answer slot at canvas index 0)
        row0 = np.fromfile(f, dtype=np.float32, count=VOCAB_SIZE)

    candidates = fixture["candidates"]
    cand_ids = [c["token_id"] for c in candidates]
    cand_labels = [c["label"] for c in candidates]

    raw_logits = [float(row0[tid]) for tid in cand_ids]
    sc_logits = [float(softcap(l)) for l in raw_logits]

    # Compute softmax over candidate softcapped logits
    max_sc = max(sc_logits)
    exp_sc = [np.exp(l - max_sc) for l in sc_logits]
    sum_exp = sum(exp_sc)
    probs = [float(e / sum_exp) for e in exp_sc]

    top_idx = int(np.argmax(probs))
    decision = cand_labels[top_idx]
    decision_prob = probs[top_idx]

    # Top 5 vocabulary tokens across all 262,144 tokens
    top5_indices = np.argsort(row0)[::-1][:5].tolist()

    print("=== Reference Oracle Evaluation ===")
    print(f"File:                 {bin_path}")
    print(f"Decision:             {decision}")
    print(f"Decision Probability: {decision_prob:.4f}")
    for label, tid, raw, sc, p in zip(cand_labels, cand_ids, raw_logits, sc_logits, probs):
        print(f"  Candidate '{label}' (ID {tid}): raw={raw:.4f}, softcapped={sc:.4f}, prob={p:.4f}")
    print(f"Top 5 Vocab Tokens:   {top5_indices}")
    for t in top5_indices:
        print(f"  Token {t}: {row0[t]:.4f}")

    if "oracle_reference" in fixture:
        ref = fixture["oracle_reference"]
        print("\n=== Parity Check vs Fixture Oracle Record ===")
        print(f"Expected Decision:    {ref['decision']} (Actual: {decision})")
        assert decision == ref["decision"], f"Decision mismatch: {decision} != {ref['decision']}"
        diff_prob = abs(decision_prob - ref["decision_probability"])
        print(f"Probability Diff:     {diff_prob:.4f} (Tolerance: 0.05)")
        assert diff_prob <= 0.05, f"Probability diff {diff_prob} exceeds 0.05"
        print("✓ Reference oracle output matches fixture oracle record within tolerances.")

if __name__ == "__main__":
    main()

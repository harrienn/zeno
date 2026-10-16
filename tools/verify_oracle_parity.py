#!/usr/bin/env python3
"""Zeno — Reference Oracle Parity & Regression Verifier.

Compares raw binary logit dumps from llama.cpp's llama-diffusion-gemma-eval
and Zeno CPU's forward pass against golden fixture specifications.

Conforms to zeno.fixture/v1 and zeno.tolerance/v1 defined in specs/UNRESOLVED.md.

Usage:
    python3 tools/verify_oracle_parity.py <out_logits.bin> <fixture.json> [zeno_out.json] [--strict-parity]
"""

import json
import sys
import numpy as np

VOCAB_SIZE = 262_144
LOGIT_SOFTCAP = 30.0

def softcap(x, cap=LOGIT_SOFTCAP):
    return cap * np.tanh(x / cap)

def unsoftcap(s, cap=LOGIT_SOFTCAP):
    ratio = np.clip(s / cap, -0.999999, 0.999999)
    return float(cap * np.arctanh(ratio))

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <out_logits.bin> <fixture.json> [zeno_out.json] [--strict-parity]")
        sys.exit(1)

    bin_path = sys.argv[1]
    fixture_path = sys.argv[2]
    zeno_json_path = None
    strict_parity = False

    for arg in sys.argv[3:]:
        if arg == "--strict-parity":
            strict_parity = True
        elif not arg.startswith("--"):
            zeno_json_path = arg

    with open(fixture_path, "r") as f:
        fixture = json.load(f)

    with open(bin_path, "rb") as f:
        # Read the first canvas position (answer slot at canvas index 0)
        row0 = np.fromfile(f, dtype=np.float32, count=VOCAB_SIZE)

    candidates = fixture["candidates"]
    cand_ids = [c["token_id"] for c in candidates]
    cand_labels = [c["label"] for c in candidates]

    # Note: llama-diffusion-gemma-eval outputs res->t_logits from diffusion-gemma.cpp:605-612.
    # In GGML computation graph, final logit softcapping (30.0 * tanh(x / 30.0)) has already been applied.
    # Therefore, row0 contains single-softcapped logits.
    # We recover true raw logits via inverse softcap: 30.0 * arctanh(s / 30.0).
    oracle_sc_logits = [float(row0[tid]) for tid in cand_ids]
    oracle_raw_logits = [unsoftcap(s) for s in oracle_sc_logits]

    # Compute softmax over candidate softcapped logits
    max_sc = max(oracle_sc_logits)
    exp_sc = [np.exp(l - max_sc) for l in oracle_sc_logits]
    sum_exp = sum(exp_sc)
    oracle_probs = [float(e / sum_exp) for e in exp_sc]

    top_idx = int(np.argmax(oracle_probs))
    oracle_decision = cand_labels[top_idx]
    oracle_decision_prob = oracle_probs[top_idx]

    # Top 5 vocabulary tokens across all 262,144 tokens
    oracle_top5 = np.argsort(row0)[::-1][:5].tolist()

    # Compute Shannon entropy and uncertainty for Oracle reference
    oracle_h = float(-sum(p * np.log(p) for p in oracle_probs if p > 0.0))
    oracle_h_norm = float(oracle_h / np.log(len(candidates))) if len(candidates) > 1 else 0.0
    thresh = fixture.get("entropy_threshold")
    oracle_uncertain = bool(oracle_h_norm >= thresh) if thresh is not None else False

    print("=== Reference Oracle Execution ===")
    print(f"File:                 {bin_path}")
    print(f"Decision:             {oracle_decision}")
    print(f"Decision Probability: {oracle_decision_prob:.4f}")
    for label, tid, raw, sc, p in zip(cand_labels, cand_ids, oracle_raw_logits, oracle_sc_logits, oracle_probs):
        print(f"  Candidate '{label}' (ID {tid}): raw={raw:.4f}, softcapped={sc:.4f}, prob={p:.4f}")
    print(f"Top 5 Vocab Tokens:   {oracle_top5}")
    for t in oracle_top5:
        print(f"  Token {t}: {row0[t]:.4f}")
    print(f"Entropy (nats):       {oracle_h:.4f}")
    print(f"Normalized Entropy:   {oracle_h_norm:.4f}")
    print(f"Uncertain:            {oracle_uncertain}")

    # 1. Assert reference execution reproducibility against captured oracle_reference
    ref = fixture.get("oracle_reference", fixture.get("expected"))
    if ref:
        print("\n=== Assert Reference Oracle Reproduction vs Pinned Record ===")
        assert oracle_decision == ref["decision"], f"Oracle decision mismatch: {oracle_decision} != {ref['decision']}"
        diff_prob = abs(oracle_decision_prob - ref["decision_probability"])
        assert diff_prob <= 1e-3, f"Oracle decision prob diff {diff_prob:.6f} exceeds 1e-3"

        if "top_5_vocab_tokens" in ref:
            print(f"Top-5 Vocab:          Expected {ref['top_5_vocab_tokens']} vs Actual {oracle_top5}")
            assert oracle_top5 == ref["top_5_vocab_tokens"], f"Top-5 vocab tokens mismatch: {oracle_top5} != {ref['top_5_vocab_tokens']}"

        for i, (label, tid) in enumerate(zip(cand_labels, cand_ids)):
            raw_err = abs(oracle_raw_logits[i] - ref["candidate_logits_raw"][i])
            sc_err = abs(oracle_sc_logits[i] - ref["candidate_logits_softcapped"][i])
            prob_err = abs(oracle_probs[i] - ref["probabilities"][i])
            assert raw_err <= 1e-3, f"Oracle raw logit mismatch for '{label}': {raw_err:.6f} > 1e-3"
            assert sc_err <= 1e-3, f"Oracle softcapped logit mismatch for '{label}': {sc_err:.6f} > 1e-3"
            assert prob_err <= 1e-3, f"Oracle probability mismatch for '{label}': {prob_err:.6f} > 1e-3"

        if "entropy_nats" in ref:
            h_err = abs(oracle_h - ref["entropy_nats"])
            assert h_err <= 1e-3, f"Oracle entropy mismatch: {h_err:.6f} > 1e-3"
        if "normalized_entropy" in ref:
            hn_err = abs(oracle_h_norm - ref["normalized_entropy"])
            assert hn_err <= 1e-3, f"Oracle normalized entropy mismatch: {hn_err:.6f} > 1e-3"
        if "uncertain" in ref:
            assert oracle_uncertain == ref["uncertain"], f"Oracle uncertainty mismatch: {oracle_uncertain} != {ref['uncertain']}"

        print("✓ Reference execution identically reproduces captured oracle record.")

    # 2. Compare Zeno engine against Reference Oracle
    if zeno_json_path:
        with open(zeno_json_path, "r") as f:
            zeno = json.load(f)

        print("\n=== Zeno CPU vs Reference Oracle Comparison ===")
        print(f"Zeno Decision:        {zeno['decision']} (Oracle: {oracle_decision})")
        assert zeno["decision"] == oracle_decision, f"Discrete decision mismatch: Zeno '{zeno['decision']}' != Oracle '{oracle_decision}'"

        zeno_cands = zeno["candidates"]
        assert len(zeno_cands) == len(candidates), f"Candidate count mismatch: {len(zeno_cands)} != {len(candidates)}"

        max_raw_delta = 0.0
        max_sc_delta = 0.0
        max_prob_delta = 0.0

        print("\nCandidate Comparison Table:")
        print(f"{'Label':<10} | {'Oracle Raw':<10} | {'Zeno Raw':<10} | {'Delta Raw':<10} | {'Oracle Prob':<11} | {'Zeno Prob':<11} | {'Delta Prob':<10}")
        print("-" * 85)

        for i, (c_fix, c_zeno) in enumerate(zip(candidates, zeno_cands)):
            lbl = c_fix["label"]
            o_raw = oracle_raw_logits[i]
            z_raw = c_zeno["raw_logit"]
            d_raw = abs(z_raw - o_raw)

            o_sc = oracle_sc_logits[i]
            z_sc = c_zeno["softcapped_logit"]
            d_sc = abs(z_sc - o_sc)

            o_p = oracle_probs[i]
            z_p = c_zeno["probability"]
            d_p = abs(z_p - o_p)

            max_raw_delta = max(max_raw_delta, d_raw)
            max_sc_delta = max(max_sc_delta, d_sc)
            max_prob_delta = max(max_prob_delta, d_p)

            print(f"{lbl:<10} | {o_raw:<10.4f} | {z_raw:<10.4f} | {d_raw:<10.4f} | {o_p:<11.4f} | {z_p:<11.4f} | {d_p:<10.4f}")

        d_dec_prob = abs(zeno["decision_probability"] - oracle_decision_prob)
        d_entropy = abs(zeno.get("entropy_nats", 0.0) - oracle_h)
        d_norm_entropy = abs(zeno.get("normalized_entropy", 0.0) - oracle_h_norm)
        unc_match = (zeno.get("uncertain", False) == oracle_uncertain)

        print("-" * 85)
        print(f"Summary Deltas:")
        print(f"  Discrete Decision:       MATCH ('{zeno['decision']}')")
        print(f"  Decision Prob Delta:     {d_dec_prob:.4f} (tolerance <= 0.05)")
        print(f"  Max Raw Logit Delta:     {max_raw_delta:.4f} (tolerance <= 0.5)")
        print(f"  Max Softcapped Delta:    {max_sc_delta:.4f} (tolerance <= 0.5)")
        print(f"  Max Probability Delta:   {max_prob_delta:.4f} (tolerance <= 0.05)")
        print(f"  Entropy Delta:           {d_entropy:.4f} (tolerance <= 0.05)")
        print(f"  Norm Entropy Delta:      {d_norm_entropy:.4f} (tolerance <= 0.05)")
        print(f"  Uncertain Match:         {'MATCH' if unc_match else 'MISMATCH'} (Oracle: {oracle_uncertain}, Zeno: {zeno.get('uncertain')})")

        # Verify regression against local engine baseline if present
        if "local_engine_baseline" in fixture:
            base = fixture["local_engine_baseline"]
            print("\n=== Regression Check vs Zeno Baseline ===")
            assert zeno["decision"] == base["decision"], "Baseline decision mismatch"
            assert abs(zeno["decision_probability"] - base["decision_probability"]) <= 0.05, "Baseline prob mismatch"
            for i, c_zeno in enumerate(zeno_cands):
                b_raw = base["candidate_logits_raw"][i]
                b_p = base["probabilities"][i]
                assert abs(c_zeno["raw_logit"] - b_raw) <= 0.5, f"Baseline raw logit drift on cand {i}"
                assert abs(c_zeno["probability"] - b_p) <= 0.05, f"Baseline prob drift on cand {i}"
            print("✓ Zeno CPU output matches local engine baseline within regression tolerances.")

        # U-22 tolerance evaluation
        passed_u22 = (
            (zeno["decision"] == oracle_decision) and
            (max_raw_delta <= 0.5) and
            (max_sc_delta <= 0.5) and
            (max_prob_delta <= 0.05) and
            (d_dec_prob <= 0.05) and
            (d_entropy <= 0.05) and
            (d_norm_entropy <= 0.05) and
            unc_match
        )
        if passed_u22:
            print("\n✓ Full numerical parity PASS under U-22 tolerance policy.")
        else:
            print(f"\n! Numerical cross-engine divergence exceeds U-22 tolerance budget (max raw delta={max_raw_delta:.4f} > 0.5, max prob delta={max_prob_delta:.4f}, max H delta={d_entropy:.4f}).")
            print("  Status: Discrete decision PASSED, but cross-engine numerical parity gates U-13/U-22 remain OPEN.")
            if strict_parity:
                print("Error: Strict parity required but tolerances exceeded.")
                sys.exit(1)
    elif strict_parity:
        print("Error: --strict-parity requires a <zeno_decision_json> path argument.")
        sys.exit(1)

if __name__ == "__main__":
    main()

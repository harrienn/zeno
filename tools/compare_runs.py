#!/usr/bin/env python3
"""
Zeno — Choice-Order Robustness & Paired Run Analyzer
Evaluates positional invariance, content consistency, letter distribution,
and paired decision dynamics (McNemar's test, wrong-to-correct transitions).
"""

import argparse
import json
import sys
from collections import Counter
import numpy as np

def extract_content_text(decision_str: str) -> str:
    """Extracts raw content text by stripping 'A: ', 'B: ', etc."""
    if len(decision_str) >= 3 and decision_str[1] in (':', ')') and decision_str[2] == ' ':
        return decision_str[3:].strip()
    return decision_str.strip()

def extract_letter(decision_str: str) -> str:
    """Extracts option letter from decision string."""
    if len(decision_str) >= 2 and decision_str[1] in (':', ')'):
        return decision_str[0].upper()
    return decision_str[:1].upper()

def analyze_paired_runs(base_path: str, alt_path: str):
    with open(base_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    with open(alt_path, "r", encoding="utf-8") as f:
        alt_data = json.load(f)

    base_raw_ids = [r["id"] for r in base_data.get("results", [])]
    alt_raw_ids = [r["id"] for r in alt_data.get("results", [])]

    # Integrity check 1: Duplicate query IDs within either run
    base_dups = [item for item, count in Counter(base_raw_ids).items() if count > 1]
    if base_dups:
        print(f"Error: Base run contains duplicate query IDs ({len(base_dups)} found): {base_dups[:5]}", file=sys.stderr)
        sys.exit(1)

    alt_dups = [item for item, count in Counter(alt_raw_ids).items() if count > 1]
    if alt_dups:
        print(f"Error: Compared run contains duplicate query IDs ({len(alt_dups)} found): {alt_dups[:5]}", file=sys.stderr)
        sys.exit(1)

    # Integrity check 2: Mismatched ID sets
    base_id_set = set(base_raw_ids)
    alt_id_set = set(alt_raw_ids)
    if base_id_set != alt_id_set:
        only_base = base_id_set - alt_id_set
        only_alt = alt_id_set - base_id_set
        print(f"Error: Mismatched query ID sets between runs.", file=sys.stderr)
        if only_base:
            print(f"  Present only in base run ({len(only_base)} items): {sorted(list(only_base))[:5]}...", file=sys.stderr)
        if only_alt:
            print(f"  Present only in compared run ({len(only_alt)} items): {sorted(list(only_alt))[:5]}...", file=sys.stderr)
        sys.exit(1)

    common_ids = base_raw_ids
    n = len(common_ids)
    if n == 0:
        print("Error: No query IDs found in evaluation results.", file=sys.stderr)
        sys.exit(1)

    base_results = {r["id"]: r for r in base_data.get("results", [])}
    alt_results = {r["id"]: r for r in alt_data.get("results", [])}

    base_correct = 0
    alt_correct = 0
    content_consistent = 0
    letter_consistent = 0

    # Contingency table for McNemar's test:
    # n00: both wrong, n01: base wrong, alt correct (flipped to correct)
    # n10: base correct, alt wrong (flipped to wrong), n11: both correct
    n00, n01, n10, n11 = 0, 0, 0, 0

    base_letters = Counter()
    alt_letters = Counter()

    for qid in common_ids:
        b = base_results[qid]
        a = alt_results[qid]

        b_corr = b.get("is_correct", False)
        a_corr = a.get("is_correct", False)

        if b_corr: base_correct += 1
        if a_corr: alt_correct += 1

        if not b_corr and not a_corr: n00 += 1
        elif not b_corr and a_corr:   n01 += 1
        elif b_corr and not a_corr:   n10 += 1
        elif b_corr and a_corr:       n11 += 1

        b_content = extract_content_text(b["decision"])
        a_content = extract_content_text(a["decision"])
        if b_content == a_content:
            content_consistent += 1

        b_let = extract_letter(b["decision"])
        a_let = extract_letter(a["decision"])
        base_letters[b_let] += 1
        alt_letters[a_let] += 1
        if b_let == a_let:
            letter_consistent += 1

    # McNemar's chi-squared with clamped continuity correction
    b_disc = n01  # wrong -> correct
    c_disc = n10  # correct -> wrong
    disc_total = b_disc + c_disc
    if disc_total > 0:
        mcnemar_stat = (max(0.0, abs(b_disc - c_disc) - 1.0) ** 2) / disc_total
    else:
        mcnemar_stat = 0.0

    # P-value computation via scipy.stats
    try:
        from scipy.stats import binomtest, chi2
        has_scipy = True
        asymp_p_value = float(chi2.sf(mcnemar_stat, df=1))
        if disc_total > 0:
            exact_p_value = float(binomtest(b_disc, disc_total, p=0.5, alternative="two-sided").pvalue)
        else:
            exact_p_value = 1.0
    except ImportError:
        has_scipy = False
        asymp_p_value = None
        exact_p_value = None

    print("=" * 80)
    print(" ZENO PAIRED RUN & CHOICE-ORDER ROBUSTNESS ANALYSIS")
    print("=" * 80)
    print(f"Base Run:     {base_path}")
    print(f"Compared Run: {alt_path}")
    print(f"Common Items: {n}")
    print("-" * 80)
    print("--- Accuracy & Agreement ---")
    print(f"Base Accuracy:      {base_correct:>3}/{n} ({base_correct/n*100:6.2f}%)")
    print(f"Alt Accuracy:       {alt_correct:>3}/{n} ({alt_correct/n*100:6.2f}%)")
    print(f"Net Accuracy Delta: {(alt_correct - base_correct)/n*100:+6.2f}%")
    print(f"Content Invariance: {content_consistent:>3}/{n} ({content_consistent/n*100:6.2f}%) [picked exact same answer text]")
    print(f"Letter Invariance:  {letter_consistent:>3}/{n} ({letter_consistent/n*100:6.2f}%) [picked exact same option letter]")
    print("-" * 80)
    print("--- Paired Decision Transition Contingency Table ---")
    print(f"Both Correct (n11):              {n11:>3} ({n11/n*100:.1f}%)")
    print(f"Both Incorrect (n00):            {n00:>3} ({n00/n*100:.1f}%)")
    print(f"Flipped to Correct (n01):        {n01:>3} ({n01/n*100:.1f}%) [wrong -> correct]")
    print(f"Flipped to Incorrect (n10):      {n10:>3} ({n10/n*100:.1f}%) [correct -> wrong]")
    print(f"Discordant Pairs (b, c):         b={b_disc} (wrong->correct), c={c_disc} (correct->wrong), total={disc_total}")
    print(f"McNemar chi2 stat (continuity):  {mcnemar_stat:.4f}")
    if has_scipy:
        print(f"Asymptotic chi2 p-value:         {asymp_p_value:.6g} (df=1)")
        print(f"Exact Binomial p-value (2-sided):{exact_p_value:.6g} (recommended for small b+c)")
    else:
        print("Note: scipy not installed; install scipy to compute p-values.")
    print("-" * 80)
    print("--- Option Letter Selection Distribution ---")
    letters = sorted(list(set(list(base_letters.keys()) + list(alt_letters.keys()))))
    print(f"{'Option':<8} | {'Base Count':<12} | {'Base Rate (%)':<14} | {'Alt Count':<12} | {'Alt Rate (%)':<14}")
    print(f"{'-'*8}-|-{'-'*12}-|-{'-'*14}-|-{'-'*12}-|-{'-'*14}")
    for let in letters:
        bc = base_letters[let]
        ac = alt_letters[let]
        print(f"Option {let:<1} | {bc:>10}   | {bc/n*100:>12.1f}% | {ac:>10}   | {ac/n*100:>12.1f}%")
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze paired runs and choice-order robustness")
    parser.add_argument("base_json", help="Path to base results JSON")
    parser.add_argument("alt_json", help="Path to compared (e.g. rotated or multi-step) results JSON")
    args = parser.parse_args()

    analyze_paired_runs(args.base_json, args.alt_json)

if __name__ == "__main__":
    main()

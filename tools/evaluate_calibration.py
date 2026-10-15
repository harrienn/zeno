#!/usr/bin/env python3
"""Zeno — Empirical Calibration & Decision Evaluation Analysis Tool

Computes Expected Calibration Error (ECE), Brier score, Negative Log-Likelihood,
Reliability Diagrams, and Selective Risk curves from evaluation results JSON files.

Usage:
  python3 tools/evaluate_calibration.py results_t1.json [results_t2.json ...]
"""

import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

NUM_BINS = 10

def load_evaluation(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_ece_and_bins(results: List[Dict[str, Any]], num_bins: int = NUM_BINS) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and per-bin reliability statistics.
    """
    bins = [[] for _ in range(num_bins)]
    
    for r in results:
        conf = float(r["decision_probability"])
        is_correct = bool(r["is_correct"])
        # Map [0, 1] to bin index 0..num_bins-1
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bins[bin_idx].append((conf, 1.0 if is_correct else 0.0))
        
    total_samples = len(results)
    ece = 0.0
    mce = 0.0
    bin_stats = []
    
    for b_idx in range(num_bins):
        b_low = b_idx / num_bins
        b_high = (b_idx + 1) / num_bins
        b_samples = bins[b_idx]
        count = len(b_samples)
        
        if count > 0:
            avg_conf = sum(s[0] for s in b_samples) / count
            avg_acc = sum(s[1] for s in b_samples) / count
            gap = abs(avg_acc - avg_conf)
            weight = count / total_samples
            ece += weight * gap
            if gap > mce:
                mce = gap
        else:
            avg_conf = (b_low + b_high) / 2.0
            avg_acc = 0.0
            gap = 0.0
            
        bin_label = f"[{b_low:.1f}, {b_high:.1f}]" if b_idx == num_bins - 1 else f"[{b_low:.1f}, {b_high:.1f})"
        bin_stats.append({
            "bin": bin_label,
            "count": count,
            "avg_conf": avg_conf,
            "avg_acc": avg_acc,
            "gap": gap
        })
        
    return ece, mce, bin_stats

def compute_brier_score(results: List[Dict[str, Any]]) -> float:
    """Computes multi-class Brier score: (1/M) * sum_i sum_c (P_i(c) - y*_i(c))^2"""
    total_sq_err = 0.0
    for r in results:
        gt = r["ground_truth"]
        cands = r["candidates"]
        for c in cands:
            prob = float(c["probability"])
            target = 1.0 if c["label"] == gt else 0.0
            total_sq_err += (prob - target) ** 2
    return total_sq_err / len(results) if results else 0.0

def compute_nll(results: List[Dict[str, Any]]) -> float:
    """Computes Negative Log-Likelihood (log loss)."""
    total_nll = 0.0
    for r in results:
        gt = r["ground_truth"]
        cands = r["candidates"]
        prob_gt = 1e-12
        for c in cands:
            if c["label"] == gt:
                prob_gt = max(float(c["probability"]), 1e-12)
                break
        total_nll += -math.log(prob_gt)
    return total_nll / len(results) if results else 0.0

def compute_selective_risk(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluates coverage and selective accuracy across varying entropy thresholds tau."""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    curve = []
    total = len(results)
    
    for tau in thresholds:
        covered = [r for r in results if float(r["normalized_entropy"]) < tau]
        cov_count = len(covered)
        cov_pct = (cov_count / total) * 100.0 if total > 0 else 0.0
        correct_cov = sum(1 for r in covered if r["is_correct"])
        acc_cov = (correct_cov / cov_count) * 100.0 if cov_count > 0 else 100.0
        risk_cov = 100.0 - acc_cov
        curve.append({
            "threshold": tau,
            "coverage_pct": cov_pct,
            "covered_count": cov_count,
            "selective_acc": acc_cov,
            "selective_risk": risk_cov
        })
    return curve

def analyze_run(data: Dict[str, Any], filepath: str) -> Dict[str, Any]:
    cfg = data.get("config", {})
    summary = data.get("summary", {})
    results = data.get("results", [])
    
    ece, mce, bin_stats = compute_ece_and_bins(results)
    brier = compute_brier_score(results)
    nll = compute_nll(results)
    selective = compute_selective_risk(results)
    
    # Category breakdown
    cat_stats = {}
    for r in results:
        cat = r.get("category", "default")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0, "conf_sum": 0.0, "ent_sum": 0.0}
        cat_stats[cat]["total"] += 1
        if r["is_correct"]:
            cat_stats[cat]["correct"] += 1
        cat_stats[cat]["conf_sum"] += float(r["decision_probability"])
        cat_stats[cat]["ent_sum"] += float(r["normalized_entropy"])
        
    return {
        "file": filepath,
        "config": cfg,
        "summary": summary,
        "results_count": len(results),
        "accuracy": summary.get("accuracy", 0.0) * 100.0 if "accuracy" in summary else (sum(1 for r in results if r["is_correct"]) / len(results) * 100.0),
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "nll": nll,
        "mean_latency": summary.get("mean_latency_ms", 0.0),
        "median_latency": summary.get("median_latency_ms", 0.0),
        "mean_norm_ent": summary.get("mean_normalized_entropy", 0.0),
        "step_dist": summary.get("step_distribution", {}),
        "bin_stats": bin_stats,
        "selective": selective,
        "cat_stats": cat_stats
    }

def print_single_report(analysis: Dict[str, Any]):
    print("\n" + "=" * 80)
    print(f" Zeno Decision & Calibration Report: {analysis['file']}")
    print("=" * 80)
    
    cfg = analysis["config"]
    print(f"Backend:           {cfg.get('backend', 'unknown').upper()}")
    print(f"Step Cap:          {cfg.get('step_cap', 1)}")
    print(f"Entropy Threshold: {cfg.get('entropy_threshold', 'none')}")
    print(f"Canvas Length:     {cfg.get('canvas_length', 256)}")
    print(f"Evaluated Queries: {analysis['results_count']}")
    
    print("\n--- Core Statistical Performance ---")
    print(f"Overall Accuracy:  {analysis['accuracy']:.2f}%")
    print(f"Expected Calib Error (ECE):  {analysis['ece']:.4f}  ({analysis['ece']*100:.2f}%)")
    print(f"Maximum Calib Error (MCE):   {analysis['mce']:.4f}  ({analysis['mce']*100:.2f}%)")
    print(f"Brier Score:                 {analysis['brier']:.4f}  (lower is better, 0.0 = perfect)")
    print(f"Negative Log-Likelihood (NLL): {analysis['nll']:.4f}")
    print(f"Mean Latency:                {analysis['mean_latency']:.1f} ms (Median: {analysis['median_latency']:.1f} ms)")
    print(f"Mean Normalized Entropy:     {analysis['mean_norm_ent']:.4f}")
    
    step_d = analysis["step_dist"]
    if step_d:
        print(f"Step Distribution: T=1: {step_d.get('step_1', 0)} | T=2: {step_d.get('step_2', 0)} | T=3: {step_d.get('step_3', 0)} | T=4: {step_d.get('step_4', 0)}")

    print("\n--- Reliability Diagram (ECE Bins) ---")
    print("| Confidence Bin | Count | Avg Confidence | Observed Accuracy | Calibration Gap |")
    print("|:---------------|------:|---------------:|------------------:|----------------:|")
    for b in analysis["bin_stats"]:
        if b["count"] > 0:
            print(f"| {b['bin']:<14} | {b['count']:>5} | {b['avg_conf']:>13.4f}  | {b['avg_acc']:>16.4f}  | {b['gap']:>14.4f}  |")
        else:
            print(f"| {b['bin']:<14} |     0 |         (empty) |           (empty) |          0.0000 |")

    print("\n--- Uncertainty-Gated Selective Classification ---")
    print("| Entropy Cutoff (tau) | Coverage (%) | Covered Queries | Selective Acc (%) | Selective Risk (%) |")
    print("|:---------------------|-------------:|----------------:|------------------:|-------------------:|")
    for s in analysis["selective"]:
        print(f"| < {s['threshold']:<17.1f} | {s['coverage_pct']:>11.1f}% | {s['covered_count']:>15} | {s['selective_acc']:>16.2f}% | {s['selective_risk']:>17.2f}% |")

    print("\n--- Category Breakdown ---")
    print("| Category                 | Count | Accuracy (%) | Mean Conf | Mean H_norm |")
    print("|:-------------------------|------:|-------------:|----------:|------------:|")
    for cat, cs in sorted(analysis["cat_stats"].items()):
        acc = (cs["correct"] / cs["total"]) * 100.0 if cs["total"] > 0 else 0.0
        m_conf = cs["conf_sum"] / cs["total"] if cs["total"] > 0 else 0.0
        m_ent = cs["ent_sum"] / cs["total"] if cs["total"] > 0 else 0.0
        print(f"| {cat:<24} | {cs['total']:>5} | {acc:>11.2f}% | {m_conf:>9.4f} | {m_ent:>11.4f} |")
    print("=" * 80 + "\n")

def print_comparison_table(analyses: List[Dict[str, Any]]):
    print("\n" + "=" * 90)
    print(" Comparative Step Dynamics Evaluation Matrix")
    print("=" * 90)
    print("| Run Name / Config | Mode | Steps | Acc (%) | ECE (Gap) | Brier | NLL   | Mean Latency | Multi-Step Rate |")
    print("|:-------------------|:-----|:-----:|:-------:|:---------:|:-----:|:-----:|:------------:|:---------------:|")
    for a in analyses:
        cfg = a["config"]
        name = Path(a["file"]).stem
        mode = f"step_cap={cfg.get('step_cap', 1)}, tau={cfg.get('entropy_threshold', 'none')}"
        steps = f"T={cfg.get('step_cap', 1)}"
        acc = f"{a['accuracy']:.2f}%"
        ece = f"{a['ece']:.4f}"
        brier = f"{a['brier']:.4f}"
        nll = f"{a['nll']:.4f}"
        lat = f"{a['mean_latency']:.1f}ms"
        
        # Multi-step rate: fraction that needed extra steps (steps_executed > 1)
        s_dist = a.get("step_dist", {})
        t2 = s_dist.get("step_2", 0) + s_dist.get("step_3", 0) + s_dist.get("step_4", 0)
        multi_step_rate = f"{t2}/{a['results_count']} ({t2/a['results_count']*100:.1f}%)" if a['results_count'] > 0 else "N/A"
        
        print(f"| {name:<18} | {mode:<20} | {steps:<5} | {acc:>7} | {ece:>9} | {brier:>5} | {nll:>5} | {lat:>12} | {multi_step_rate:>15} |")
    print("=" * 90 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/evaluate_calibration.py results1.json [results2.json ...]")
        sys.exit(1)
        
    analyses = []
    for filepath in sys.argv[1:]:
        data = load_evaluation(filepath)
        analysis = analyze_run(data, filepath)
        analyses.append(analysis)
        print_single_report(analysis)
        
    if len(analyses) > 1:
        print_comparison_table(analyses)

if __name__ == "__main__":
    main()

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
import random
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
        if not any(c["label"] == gt for c in cands):
            raise ValueError(f"ground_truth '{gt}' not found in candidate labels for id '{r.get('id', 'unknown')}'")
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
        found = False
        prob_gt = 1e-12
        for c in cands:
            if c["label"] == gt:
                prob_gt = max(float(c["probability"]), 1e-12)
                found = True
                break
        if not found:
            raise ValueError(f"ground_truth '{gt}' not found in candidate labels for id '{r.get('id', 'unknown')}'")
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
        acc_cov = (correct_cov / cov_count) * 100.0 if cov_count > 0 else None
        risk_cov = (100.0 - acc_cov) if acc_cov is not None else None
        curve.append({
            "threshold": tau,
            "coverage_pct": cov_pct,
            "covered_count": cov_count,
            "selective_acc": acc_cov,
            "selective_risk": risk_cov
        })
    return curve

def optimize_temperature(
    results: List[Dict[str, Any]],
    use_softcapped: bool = True,
    t_min: float = 0.05,
    t_max: float = 5.0,
    steps: int = 1000
) -> Dict[str, Any]:
    """Finds optimal temperature T that minimizes Negative Log-Likelihood (NLL)."""
    data = []
    logit_key = "softcapped_logit" if use_softcapped else "raw_logit"
    for r in results:
        gt = r.get("ground_truth")
        cands = r.get("candidates", [])
        if not cands or not gt or logit_key not in cands[0]:
            continue
        logits = [float(c[logit_key]) for c in cands]
        target_idx = None
        for idx, c in enumerate(cands):
            if c["label"] == gt:
                target_idx = idx
                break
        if target_idx is not None:
            data.append((logits, target_idx))

    if not data:
        return {}

    def eval_t(T: float):
        total_nll = 0.0
        total_sq_err = 0.0
        scaled_results = []
        for logits, target_idx in data:
            max_l = max(logits)
            exp_l = [math.exp((l - max_l) / T) for l in logits]
            sum_exp = sum(exp_l)
            probs = [e / sum_exp for e in exp_l]
            gt_prob = max(probs[target_idx], 1e-15)
            total_nll += -math.log(gt_prob)
            for idx, p in enumerate(probs):
                target = 1.0 if idx == target_idx else 0.0
                total_sq_err += (p - target) ** 2
            
            top_prob = max(probs)
            top_idx = probs.index(top_prob)
            is_correct = (top_idx == target_idx)
            scaled_results.append({
                "decision_probability": top_prob,
                "is_correct": is_correct
            })
            
        nll = total_nll / len(data)
        brier = total_sq_err / len(data)
        ece, mce, bin_stats = compute_ece_and_bins(scaled_results)
        return nll, brier, ece, mce, bin_stats

    best_t = 1.0
    best_nll = float('inf')
    dt = (t_max - t_min) / steps
    for step in range(steps + 1):
        T = t_min + step * dt
        nll, _, _, _, _ = eval_t(T)
        if nll < best_nll:
            best_nll = nll
            best_t = T

    base_nll, base_brier, base_ece, base_mce, _ = eval_t(1.0)
    opt_nll, opt_brier, opt_ece, opt_mce, opt_bins = eval_t(best_t)

    return {
        "logit_type": "softcapped" if use_softcapped else "raw",
        "optimal_temperature": best_t,
        "base_nll": base_nll,
        "calibrated_nll": opt_nll,
        "base_brier": base_brier,
        "calibrated_brier": opt_brier,
        "base_ece": base_ece,
        "calibrated_ece": opt_ece,
        "base_mce": base_mce,
        "calibrated_mce": opt_mce,
        "calibrated_bins": opt_bins
    }

def compute_cross_validated_temperature(
    results: List[Dict[str, Any]],
    use_softcapped: bool = True,
    k_folds: int = 5,
    t_min: float = 0.05,
    t_max: float = 5.0,
    steps: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates temperature scaling using seeded stratified k-fold cross-validation on held-out splits (out-of-sample, R46-03)."""
    logit_key = "softcapped_logit" if use_softcapped else "raw_logit"
    valid_items = []
    for r in results:
        gt = r.get("ground_truth")
        cands = r.get("candidates", [])
        if not cands or not gt or logit_key not in cands[0]:
            continue
        logits = [float(c[logit_key]) for c in cands]
        target_idx = None
        for idx, c in enumerate(cands):
            if c["label"] == gt:
                target_idx = idx
                break
        if target_idx is not None:
            valid_items.append((logits, target_idx))

    if len(valid_items) < k_folds:
        return {}

    # Deterministic stratified k-fold partitioning by target class index with reproducible shuffle (R46-03)
    strata: Dict[int, List[Tuple[List[float], int]]] = {}
    for logits, target_idx in valid_items:
        strata.setdefault(target_idx, []).append((logits, target_idx))

    rng = random.Random(seed)
    folds: List[List[Tuple[List[float], int]]] = [[] for _ in range(k_folds)]
    for target_idx in sorted(strata.keys()):
        stratum_items = list(strata[target_idx])
        rng.shuffle(stratum_items)
        for idx, item in enumerate(stratum_items):
            folds[idx % k_folds].append(item)

    oos_scaled_results = []
    fold_optimal_temps = []
    total_oos_nll = 0.0
    total_oos_sq_err = 0.0

    for fold in range(k_folds):
        val_data = folds[fold]
        train_data = []
        for f_idx in range(k_folds):
            if f_idx != fold:
                train_data.extend(folds[f_idx])

        # Fit T on train_data
        best_t = 1.0
        best_nll = float('inf')
        dt = (t_max - t_min) / steps
        for step in range(steps + 1):
            T = t_min + step * dt
            train_nll = 0.0
            for logits, target_idx in train_data:
                max_l = max(logits)
                exp_l = [math.exp((l - max_l) / T) for l in logits]
                gt_prob = max(exp_l[target_idx] / sum(exp_l), 1e-15)
                train_nll += -math.log(gt_prob)
            train_nll /= len(train_data)
            if train_nll < best_nll:
                best_nll = train_nll
                best_t = T
        fold_optimal_temps.append(best_t)

        # Evaluate on held-out val_data with best_t
        for logits, target_idx in val_data:
            max_l = max(logits)
            exp_l = [math.exp((l - max_l) / best_t) for l in logits]
            sum_exp = sum(exp_l)
            probs = [e / sum_exp for e in exp_l]
            gt_prob = max(probs[target_idx], 1e-15)
            total_oos_nll += -math.log(gt_prob)
            for idx, p in enumerate(probs):
                target = 1.0 if idx == target_idx else 0.0
                total_oos_sq_err += (p - target) ** 2
            top_prob = max(probs)
            top_idx = probs.index(top_prob)
            is_correct = (top_idx == target_idx)
            oos_scaled_results.append({
                "decision_probability": top_prob,
                "is_correct": is_correct
            })

    mean_opt_t = sum(fold_optimal_temps) / len(fold_optimal_temps)
    min_opt_t = min(fold_optimal_temps)
    max_opt_t = max(fold_optimal_temps)
    std_opt_t = math.sqrt(sum((t - mean_opt_t) ** 2 for t in fold_optimal_temps) / len(fold_optimal_temps))
    oos_nll = total_oos_nll / len(valid_items)
    oos_brier = total_oos_sq_err / len(valid_items)
    oos_ece, oos_mce, _ = compute_ece_and_bins(oos_scaled_results)

    return {
        "stratified_seed": seed,
        "mean_fold_temperature": mean_opt_t,
        "min_fold_temperature": min_opt_t,
        "max_fold_temperature": max_opt_t,
        "std_fold_temperature": std_opt_t,
        "fold_temperatures": fold_optimal_temps,
        "oos_nll": oos_nll,
        "oos_brier": oos_brier,
        "oos_ece": oos_ece,
        "oos_mce": oos_mce,
    }

def evaluate_dataset_at_temperature(results: List[Dict[str, Any]], T: float, use_softcapped: bool = True) -> Tuple[float, float, float]:
    """Evaluates NLL, Brier score, and ECE for a dataset at a specific temperature T."""
    logit_key = "softcapped_logit" if use_softcapped else "raw_logit"
    total_nll = 0.0
    total_sq_err = 0.0
    scaled = []
    valid_count = 0
    for r in results:
        gt = r.get("ground_truth")
        cands = r.get("candidates", [])
        if not cands or not gt or logit_key not in cands[0]:
            continue
        logits = [float(c[logit_key]) for c in cands]
        target_idx = None
        for idx, c in enumerate(cands):
            if c["label"] == gt:
                target_idx = idx
                break
        if target_idx is None:
            continue
        max_l = max(logits)
        exp_l = [math.exp((l - max_l) / T) for l in logits]
        sum_exp = sum(exp_l)
        probs = [e / sum_exp for e in exp_l]
        total_nll += -math.log(max(probs[target_idx], 1e-15))
        for idx, p in enumerate(probs):
            target = 1.0 if idx == target_idx else 0.0
            total_sq_err += (p - target) ** 2
        top_prob = max(probs)
        top_idx = probs.index(top_prob)
        scaled.append({
            "decision_probability": top_prob,
            "is_correct": (top_idx == target_idx)
        })
        valid_count += 1
    if valid_count == 0:
        return 0.0, 0.0, 0.0
    nll = total_nll / valid_count
    brier = total_sq_err / valid_count
    ece, _, _ = compute_ece_and_bins(scaled)
    return nll, brier, ece

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

    temp_softcapped = optimize_temperature(results, use_softcapped=True)
    temp_raw = optimize_temperature(results, use_softcapped=False)
    cv_softcapped = compute_cross_validated_temperature(results, use_softcapped=True)
    cv_raw = compute_cross_validated_temperature(results, use_softcapped=False)
        
    return {
        "file": filepath,
        "config": cfg,
        "summary": summary,
        "raw_results": results,
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
        "cat_stats": cat_stats,
        "temp_softcapped": temp_softcapped,
        "temp_raw": temp_raw,
        "cv_softcapped": cv_softcapped,
        "cv_raw": cv_raw,
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
        acc_str = f"{s['selective_acc']:>16.2f}%" if s["selective_acc"] is not None else f"{'N/A':>17}"
        risk_str = f"{s['selective_risk']:>17.2f}%" if s["selective_risk"] is not None else f"{'N/A':>18}"
        print(f"| < {s['threshold']:<17.1f} | {s['coverage_pct']:>11.1f}% | {s['covered_count']:>15} | {acc_str} | {risk_str} |")

    print("\n--- Category Breakdown ---")
    print("| Category                 | Count | Accuracy (%) | Mean Conf | Mean H_norm |")
    print("|:-------------------------|------:|-------------:|----------:|------------:|")
    for cat, cs in sorted(analysis["cat_stats"].items()):
        acc = (cs["correct"] / cs["total"]) * 100.0 if cs["total"] > 0 else 0.0
        m_conf = cs["conf_sum"] / cs["total"] if cs["total"] > 0 else 0.0
        m_ent = cs["ent_sum"] / cs["total"] if cs["total"] > 0 else 0.0
        print(f"| {cat:<24} | {cs['total']:>5} | {acc:>11.2f}% | {m_conf:>9.4f} | {m_ent:>11.4f} |")

    if analysis.get("temp_softcapped"):
        ts = analysis["temp_softcapped"]
        print("\n--- Temperature Scaling Optimization (Post-Hoc Calibration) ---")
        print(f"Optimal Temperature (Softcapped, In-Sample): T = {ts['optimal_temperature']:.3f}")
        print(f"  NLL:   {ts['base_nll']:.4f} -> {ts['calibrated_nll']:.4f} (delta = {ts['calibrated_nll'] - ts['base_nll']:+.4f})")
        print(f"  Brier: {ts['base_brier']:.4f} -> {ts['calibrated_brier']:.4f} (delta = {ts['calibrated_brier'] - ts['base_brier']:+.4f})")
        print(f"  ECE:   {ts['base_ece']:.4f} -> {ts['calibrated_ece']:.4f} (delta = {ts['calibrated_ece'] - ts['base_ece']:+.4f})")
        print(f"  MCE:   {ts['base_mce']:.4f} -> {ts['calibrated_mce']:.4f} (delta = {ts['calibrated_mce'] - ts['base_mce']:+.4f})")
        if analysis.get("cv_softcapped"):
            cv = analysis["cv_softcapped"]
            print(f"  5-Fold Stratified Out-of-Sample CV (Softcapped, Seed = {cv['stratified_seed']}, Mean Fold T = {cv['mean_fold_temperature']:.3f}, Range = [{cv['min_fold_temperature']:.3f}, {cv['max_fold_temperature']:.3f}], std = {cv['std_fold_temperature']:.3f}):")
            print(f"    Held-Out NLL:   {cv['oos_nll']:.4f} (delta = {cv['oos_nll'] - ts['base_nll']:+.4f})")
            print(f"    Held-Out Brier: {cv['oos_brier']:.4f} (delta = {cv['oos_brier'] - ts['base_brier']:+.4f})")
            print(f"    Held-Out ECE:   {cv['oos_ece']:.4f} (delta = {cv['oos_ece'] - ts['base_ece']:+.4f})")
            print(f"    Held-Out MCE:   {cv['oos_mce']:.4f} (delta = {cv['oos_mce'] - ts['base_mce']:+.4f})")
    if analysis.get("temp_raw"):
        tr = analysis["temp_raw"]
        print(f"\nOptimal Temperature (Raw Logits, In-Sample):  T = {tr['optimal_temperature']:.3f}")
        print(f"  NLL:   {tr['base_nll']:.4f} -> {tr['calibrated_nll']:.4f} (delta = {tr['calibrated_nll'] - tr['base_nll']:+.4f})")
        print(f"  Brier: {tr['base_brier']:.4f} -> {tr['calibrated_brier']:.4f} (delta = {tr['calibrated_brier'] - tr['base_brier']:+.4f})")
        print(f"  ECE:   {tr['base_ece']:.4f} -> {tr['calibrated_ece']:.4f} (delta = {tr['calibrated_ece'] - tr['base_ece']:+.4f})")
        print(f"  MCE:   {tr['base_mce']:.4f} -> {tr['calibrated_mce']:.4f} (delta = {tr['calibrated_mce'] - tr['base_mce']:+.4f})")
        if analysis.get("cv_raw"):
            cvr = analysis["cv_raw"]
            print(f"  5-Fold Stratified Out-of-Sample CV (Raw Logits, Seed = {cvr['stratified_seed']}, Mean Fold T = {cvr['mean_fold_temperature']:.3f}, Range = [{cvr['min_fold_temperature']:.3f}, {cvr['max_fold_temperature']:.3f}], std = {cvr['std_fold_temperature']:.3f}):")
            print(f"    Held-Out NLL:   {cvr['oos_nll']:.4f} (delta = {cvr['oos_nll'] - tr['base_nll']:+.4f})")
            print(f"    Held-Out Brier: {cvr['oos_brier']:.4f} (delta = {cvr['oos_brier'] - tr['base_brier']:+.4f})")
            print(f"    Held-Out ECE:   {cvr['oos_ece']:.4f} (delta = {cvr['oos_ece'] - tr['base_ece']:+.4f})")
            print(f"    Held-Out MCE:   {cvr['oos_mce']:.4f} (delta = {cvr['oos_mce'] - tr['base_mce']:+.4f})")

    print("=" * 80 + "\n")

def print_comparison_table(analyses: List[Dict[str, Any]]):
    print("\n" + "=" * 118)
    print(" Comparative Step Dynamics & Calibration Matrix")
    print("=" * 118)
    print("| Run Name / Config | Mode | Steps | Acc (%) | ECE (Base) | ECE (Fit) | ECE (CV) | T_opt | Brier | NLL   | Latency |")
    print("|:-------------------|:-----|:-----:|:-------:|:----------:|:---------:|:--------:|:-----:|:-----:|:-----:|:-------:|")
    for a in analyses:
        cfg = a["config"]
        name = Path(a["file"]).stem
        mode = f"step={cfg.get('step_cap', 1)}, tau={cfg.get('entropy_threshold', 'none')}"
        steps = f"T={cfg.get('step_cap', 1)}"
        acc = f"{a['accuracy']:.2f}%"
        ece_base = f"{a['ece']:.4f}"
        ts = a.get("temp_softcapped", {})
        ece_calib = f"{ts.get('calibrated_ece', 0.0):.4f}" if ts else "N/A"
        cv = a.get("cv_softcapped", {})
        ece_cv = f"{cv.get('oos_ece', 0.0):.4f}" if cv else "N/A"
        t_opt = f"{ts.get('optimal_temperature', 1.0):.2f}" if ts else "N/A"
        brier = f"{a['brier']:.4f}"
        nll = f"{a['nll']:.4f}"
        lat = f"{a['mean_latency']:.1f}ms"
        
        print(f"| {name:<18} | {mode:<18} | {steps:<5} | {acc:>7} | {ece_base:>10} | {ece_calib:>9} | {ece_cv:>8} | {t_opt:>5} | {brier:>5} | {nll:>5} | {lat:>7} |")
    print("=" * 118 + "\n")

def print_cross_calibration_matrix(analyses: List[Dict[str, Any]], use_softcapped: bool = True):
    key = "temp_softcapped" if use_softcapped else "temp_raw"
    valid = [a for a in analyses if "raw_results" in a and a.get(key)]
    if len(valid) < 2:
        return

    logit_label = "Softcapped" if use_softcapped else "Raw Logits"
    print("\n" + "=" * 120)
    print(f" Temperature Cross-Calibration Transfer Matrix ({logit_label})")
    print(" Format in cells: ECE (%) / NLL (lower is better; * indicates target evaluated at its own T_opt)")
    print("=" * 120)

    # Column names
    col_headers = []
    for a in valid:
        short_stem = Path(a["file"]).stem.replace("results_", "").replace("_phase18", "")
        t_opt = a[key]["optimal_temperature"]
        col_headers.append((short_stem, t_opt))

    header_line = f"| {'Target Run':<22} | {'Base (T=1.00)':<18} |"
    sep_line = f"|:{'-'*22}-|:{'-'*18}-|"
    for short_stem, t_opt in col_headers:
        hdr = f"{short_stem} (T={t_opt:.2f})"
        header_line += f" {hdr:<20} |"
        sep_line += f":{'-'*20}-|"
    print(header_line)
    print(sep_line)

    for target in valid:
        tgt_name = Path(target["file"]).stem.replace("results_", "").replace("_phase18", "")
        tgt_raw = target["raw_results"]

        base_nll, base_brier, base_ece = evaluate_dataset_at_temperature(tgt_raw, 1.0, use_softcapped=use_softcapped)
        base_cell = f"{base_ece*100:>5.2f}% / {base_nll:.3f}"
        row = f"| {tgt_name:<22} | {base_cell:<18} |"

        for src in valid:
            src_t = src[key]["optimal_temperature"]
            is_self = (src["file"] == target["file"])
            marker = "*" if is_self else " "
            s_nll, s_brier, s_ece = evaluate_dataset_at_temperature(tgt_raw, src_t, use_softcapped=use_softcapped)
            cell = f"{s_ece*100:>5.2f}% / {s_nll:.3f}{marker}"
            row += f" {cell:<20} |"
        print(row)

    print("=" * 120 + "\n")

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
        print_cross_calibration_matrix(analyses, use_softcapped=True)
        print_cross_calibration_matrix(analyses, use_softcapped=False)

if __name__ == "__main__":
    main()

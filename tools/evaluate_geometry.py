#!/usr/bin/env python3
"""Zeno — Canvas Geometry & Positional Attention Evaluation Harness.

Sweeps answer-position placement and canvas length configurations across
golden oracle fixtures and dataset queries, measuring decision stability,
logits, probabilities, normalized entropy, and execution latency.

Usage:
  python3 tools/evaluate_geometry.py [--model models/...gguf] [--cuda]
"""

import subprocess
import json
import os
import sys
from pathlib import Path

MODEL_DEFAULT = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
FIXTURES = [
    "specs/fixtures/system_health_boolean.json",
    "specs/fixtures/incident_severity_multiclass.json",
    "specs/fixtures/boundary_256_incident_triage.json",
    "specs/fixtures/maintenance_priority_uncertain.json",
    "specs/fixtures/satisfaction_score_5class.json",
]

def run_zeno_decide(model_path: str, fixture_path: str, canvas_len: int, answer_pos: int, use_cuda: bool = True):
    cmd = [
        "./zig-out/bin/zeno",
        "decide-fixture",
        model_path,
        fixture_path,
        "--canvas", str(canvas_len),
        "--answer-pos", str(answer_pos),
    ]
    if use_cuda:
        cmd.append("--cuda")
    
    # We output json to a temporary file
    out_tmp = f"/tmp/geo_out_{os.getpid()}_{canvas_len}_{answer_pos}.json"
    cmd.append(out_tmp)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        with open(out_tmp, "r") as f:
            data = json.load(f)
        return data
    finally:
        if os.path.exists(out_tmp):
            os.remove(out_tmp)

def main():
    model_path = MODEL_DEFAULT
    use_cuda = True
    for arg in sys.argv[1:]:
        if arg == "--cpu":
            use_cuda = False
        elif arg.endswith(".gguf"):
            model_path = arg

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print("=" * 105)
    print(" Zeno Canvas Geometry & Positional Attention Study")
    print(f" Model:   {model_path}")
    print(f" Backend: {'CUDA' if use_cuda else 'CPU'}")
    print("=" * 105)

    # 1. Answer Position Sweep (Fixed C=16)
    answer_positions = [0, 1, 2, 4, 7, 15]
    print("\n--- Part 1: Answer Position Sensitivity (Canvas Length C = 16) ---")
    print("Evaluates the impact of shifting the answer slot MASK within the evaluated canvas prefix.")
    print("-" * 105)
    print(f"| {'Fixture Name':<28} | {'Pos':>3} | {'Decision':<10} | {'Prob':>7} | {'Norm Ent':>8} | {'Raw Marg':>8} | {'Step1 Lat':>9} |")
    print("|:" + "-" * 28 + "-|:" + "-" * 3 + "-|:" + "-" * 10 + "-|:" + "-" * 7 + "-|:" + "-" * 8 + "-|:" + "-" * 8 + "-|:" + "-" * 9 + "-|")

    for fix in FIXTURES:
        fix_name = Path(fix).stem
        for pos in answer_positions:
            try:
                res = run_zeno_decide(model_path, fix, canvas_len=16, answer_pos=pos, use_cuda=use_cuda)
                dec = res["decision"]
                prob = res["decision_probability"]
                h_norm = res["normalized_entropy"]
                cands = res["candidates"]
                raws = [c["raw_logit"] for c in cands]
                raws_sorted = sorted(raws, reverse=True)
                raw_margin = (raws_sorted[0] - raws_sorted[1]) if len(raws_sorted) > 1 else 0.0
                lat = res.get("timings", {}).get("canvas_step1_ms", 0.0)
                print(f"| {fix_name:<28} | {pos:>3} | {dec:<10} | {prob:>7.4f} | {h_norm:>8.4f} | {raw_margin:>8.4f} | {lat:>7.1f}ms |")
            except Exception as e:
                print(f"| {fix_name:<28} | {pos:>3} | ERROR: {str(e)[:30]} |")
        print("|" + "-" * 91 + "|")

    # 2. Canvas Length Sweep (Fixed answer_pos = 0)
    canvas_lengths = [16, 32, 64, 128, 256]
    print("\n--- Part 2: Canvas Length Sweep (Answer Position = 0) ---")
    print("Evaluates the impact of trailing PAD tokens in non-causal attention across canvas lengths C.")
    print("-" * 105)
    print(f"| {'Fixture Name':<28} | {'Canvas C':>8} | {'Decision':<10} | {'Prob':>7} | {'Norm Ent':>8} | {'Raw Marg':>8} | {'Step1 Lat':>9} |")
    print("|:" + "-" * 28 + "-|:" + "-" * 8 + "-|:" + "-" * 10 + "-|:" + "-" * 7 + "-|:" + "-" * 8 + "-|:" + "-" * 8 + "-|:" + "-" * 9 + "-|")

    for fix in FIXTURES:
        fix_name = Path(fix).stem
        for c_len in canvas_lengths:
            try:
                res = run_zeno_decide(model_path, fix, canvas_len=c_len, answer_pos=0, use_cuda=use_cuda)
                dec = res["decision"]
                prob = res["decision_probability"]
                h_norm = res["normalized_entropy"]
                cands = res["candidates"]
                raws = [c["raw_logit"] for c in cands]
                raws_sorted = sorted(raws, reverse=True)
                raw_margin = (raws_sorted[0] - raws_sorted[1]) if len(raws_sorted) > 1 else 0.0
                lat = res.get("timings", {}).get("canvas_step1_ms", 0.0)
                print(f"| {fix_name:<28} | {c_len:>8} | {dec:<10} | {prob:>7.4f} | {h_norm:>8.4f} | {raw_margin:>8.4f} | {lat:>7.1f}ms |")
            except Exception as e:
                print(f"| {fix_name:<28} | {c_len:>8} | ERROR: {str(e)[:30]} |")
        print("|" + "-" * 96 + "|")

    print("\n" + "=" * 105)
    print(" Positional Study Completed Successfully.")
    print("=" * 105)

if __name__ == "__main__":
    main()

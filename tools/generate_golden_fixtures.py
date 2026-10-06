#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import numpy as np

MODEL_PATH = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
MODEL_SHA256 = "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5"
REF_REPO = "https://github.com/ggml-org/llama.cpp"
REF_COMMIT = "3dac51dc80b1400929266453c6ced6748ccf6862"
LLAMA_EVAL = "/hdd/Projects/llama.cpp/build/bin/llama-diffusion-gemma-eval"
ZENO_BIN = "./zig-out/bin/zeno"
VOCAB_SIZE = 262_144

def softcap(x, cap=30.0):
    return cap * np.tanh(x / cap)

def generate_fixture(fixture_name, base_def):
    target_path = f"specs/fixtures/{fixture_name}.json"
    p_i32 = f"/tmp/{fixture_name}_p.i32"
    c_i32 = f"/tmp/{fixture_name}_c.i32"
    out_bin = f"/tmp/{fixture_name}_gpu.bin"

    print(f"\n=======================================================")
    print(f"Generating Golden Fixture: {fixture_name}")
    print(f"=======================================================")

    # Initial write with dummy expected
    with open(target_path, "w") as f:
        json.dump(base_def, f, indent=2)

    # 1. Dump prompt and canvas via Zeno
    print("1. Running Zeno dump-fixture...")
    subprocess.run([ZENO_BIN, "dump-fixture", MODEL_PATH, target_path, p_i32, c_i32], check=True)

    prompt_tokens = np.fromfile(p_i32, dtype=np.int32).tolist()
    print(f"   Prompt tokens count: {len(prompt_tokens)}")

    # 2. Run reference oracle via llama.cpp (GPU NGL=99)
    print("2. Running reference oracle (llama-diffusion-gemma-eval)...")
    env = os.environ.copy()
    env["NGL"] = "99"
    subprocess.run([LLAMA_EVAL, MODEL_PATH, p_i32, c_i32, out_bin], env=env, check=True)

    # Parse oracle output
    with open(out_bin, "rb") as f:
        row0 = np.fromfile(f, dtype=np.float32, count=VOCAB_SIZE)

    cand_ids = [c["token_id"] for c in base_def["candidates"]]
    cand_labels = [c["label"] for c in base_def["candidates"]]

    oracle_raw = [round(float(row0[tid]), 4) for tid in cand_ids]
    oracle_sc = [round(float(softcap(r)), 4) for r in oracle_raw]

    max_sc = max(oracle_sc)
    exp_sc = [np.exp(s - max_sc) for s in oracle_sc]
    sum_exp = sum(exp_sc)
    oracle_probs = [round(float(e / sum_exp), 4) for e in exp_sc]

    top_idx = int(np.argmax(oracle_probs))
    oracle_decision = cand_labels[top_idx]
    oracle_dec_prob = oracle_probs[top_idx]
    top5_tokens = [int(x) for x in np.argsort(row0)[::-1][:5].tolist()]

    oracle_reference = {
        "decision": oracle_decision,
        "decision_probability": oracle_dec_prob,
        "candidate_logits_raw": oracle_raw,
        "candidate_logits_softcapped": oracle_sc,
        "probabilities": oracle_probs,
        "top_5_vocab_tokens": top5_tokens
    }

    base_def["prompt_tokens"] = prompt_tokens
    base_def["oracle_command"] = f"NGL=99 /hdd/Projects/llama.cpp/build/bin/llama-diffusion-gemma-eval models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf {p_i32} {c_i32} {out_bin}"
    base_def["oracle_reference"] = oracle_reference

    with open(target_path, "w") as f:
        json.dump(base_def, f, indent=2)

    # 3. Run Zeno decide-fixture to capture expected local engine outputs
    print("3. Running Zeno decide-fixture (CPU engine forward pass)...")
    zeno_res = subprocess.run([ZENO_BIN, "decide-fixture", MODEL_PATH, target_path], capture_output=True, text=True, check=True)
    zeno_lines = zeno_res.stdout.splitlines()

    zeno_decision = None
    zeno_dec_prob = None
    zeno_h_nats = None
    zeno_h_norm = None
    zeno_uncertain = None
    zeno_raw = []
    zeno_sc = []
    zeno_probs = []

    for line in zeno_lines:
        line = line.strip()
        if line.startswith("Decision:"):
            zeno_decision = line.split("Decision:")[1].strip()
        elif line.startswith("Decision Prob:"):
            zeno_dec_prob = float(line.split("Decision Prob:")[1].strip())
        elif line.startswith("Entropy (nats):"):
            zeno_h_nats = float(line.split("Entropy (nats):")[1].strip())
        elif line.startswith("Normalized Entropy:"):
            zeno_h_norm = float(line.split("Normalized Entropy:")[1].strip())
        elif line.startswith("Uncertain:"):
            zeno_uncertain = line.split("Uncertain:")[1].strip() == "true"
        elif line.startswith("- "):
            parts = line.split(":")
            metrics = parts[1].split(",")
            r_val = float(metrics[0].split("raw=")[1])
            s_val = float(metrics[1].split("sc=")[1])
            p_val = float(metrics[2].split("prob=")[1])
            zeno_raw.append(round(r_val, 4))
            zeno_sc.append(round(s_val, 4))
            zeno_probs.append(round(p_val, 4))

    # Derive reference expected from oracle_reference
    h_nats = 0.0
    for p in oracle_probs:
        if p > 1e-12:
            h_nats -= p * np.log(p)
    k = len(oracle_probs)
    h_norm = float(np.clip(h_nats / np.log(k), 0.0, 1.0)) if k > 1 else 0.0
    th = base_def.get("entropy_threshold")
    ref_uncertain = bool(h_norm >= th) if th is not None else False

    expected = {
        "decision": oracle_decision,
        "decision_probability": oracle_dec_prob,
        "candidate_logits_raw": oracle_raw,
        "candidate_logits_softcapped": oracle_sc,
        "probabilities": oracle_probs,
        "entropy_nats": round(h_nats, 4),
        "normalized_entropy": round(h_norm, 4),
        "uncertain": ref_uncertain
    }
    base_def["expected"] = expected

    local_baseline = {
        "decision": zeno_decision,
        "decision_probability": round(zeno_dec_prob, 4),
        "candidate_logits_raw": zeno_raw,
        "candidate_logits_softcapped": zeno_sc,
        "probabilities": zeno_probs,
        "entropy_nats": round(zeno_h_nats, 4),
        "normalized_entropy": round(zeno_h_norm, 4),
        "uncertain": zeno_uncertain
    }
    base_def["local_engine_baseline"] = local_baseline

    # 4. Save completed golden fixture
    with open(target_path, "w") as f:
        json.dump(base_def, f, indent=2)

    print("4. Verification & Validation:")
    print(f"   Oracle Decision: {oracle_decision} (prob={oracle_dec_prob:.4f})")
    print(f"   Zeno Decision:   {zeno_decision} (prob={zeno_dec_prob:.4f})")
    print(f"   Uncertain flag:  {zeno_uncertain}")
    assert oracle_decision == zeno_decision, f"Decision mismatch: {oracle_decision} != {zeno_decision}"
    print(f"✓ Fixture '{fixture_name}' generated and verified successfully at {target_path}")

def main():
    # Fixture 4: Maintenance Priority with High Entropy / Uncertainty Gating
    fix4_def = {
        "schema": "zeno.fixture/v1",
        "policy": "zeno.tolerance/v1",
        "description": "Maintenance event priority triage with high-entropy uncertainty gating on DiffusionGemma 26B-A4B-it Q4_K_M",
        "model": {
            "path": MODEL_PATH,
            "sha256": MODEL_SHA256
        },
        "reference": {
            "repo": REF_REPO,
            "commit": REF_COMMIT
        },
        "query": "What is the priority level of this maintenance event?",
        "context": "Routine index reorganization scheduled for customer analytics warehouse. Query latency expected to elevate by 10% for analytics queries only during off-peak window. Production OLTP databases are entirely unaffected.\n",
        "schema_type": "choice",
        "canvas_length": 256,
        "answer_pos": 0,
        "answer_seed_token": 4,
        "entropy_threshold": 0.5,
        "candidates": [
            { "label": "Low", "token": "Low", "token_id": 26834 },
            { "label": "Medium", "token": "Medium", "token_id": 49074 },
            { "label": "High", "token": "High", "token_id": 13238 }
        ],
        "expected": {
            "decision": "",
            "decision_probability": 0.0,
            "candidate_logits_raw": [],
            "candidate_logits_softcapped": [],
            "probabilities": [],
            "entropy_nats": 0.0,
            "normalized_entropy": 0.0,
            "uncertain": True
        }
    }
    generate_fixture("maintenance_priority_uncertain", fix4_def)

    # Fixture 5: 5-class numeric score query
    fix5_def = {
        "schema": "zeno.fixture/v1",
        "policy": "zeno.tolerance/v1",
        "description": "5-point customer satisfaction rating score decision query on DiffusionGemma 26B-A4B-it Q4_K_M",
        "model": {
            "path": MODEL_PATH,
            "sha256": MODEL_SHA256
        },
        "reference": {
            "repo": REF_REPO,
            "commit": REF_COMMIT
        },
        "query": "What is the satisfaction rating from 1 to 5?",
        "context": "Customer Review: Absolutely outstanding service and fantastic product quality! It arrived two days early, was packaged perfectly, and worked flawlessly out of the box. Extremely delighted with this purchase!\n",
        "schema_type": "score",
        "canvas_length": 256,
        "answer_pos": 0,
        "answer_seed_token": 4,
        "entropy_threshold": 0.8,
        "candidates": [
            { "label": "1", "token": "1", "token_id": 236770 },
            { "label": "2", "token": "2", "token_id": 236778 },
            { "label": "3", "token": "3", "token_id": 236800 },
            { "label": "4", "token": "4", "token_id": 236812 },
            { "label": "5", "token": "5", "token_id": 236810 }
        ],
        "expected": {
            "decision": "",
            "decision_probability": 0.0,
            "candidate_logits_raw": [],
            "candidate_logits_softcapped": [],
            "probabilities": [],
            "entropy_nats": 0.0,
            "normalized_entropy": 0.0,
            "uncertain": False
        }
    }
    generate_fixture("satisfaction_score_5class", fix5_def)

if __name__ == "__main__":
    main()

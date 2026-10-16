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
LLAMA_CPP_DIR = os.getenv("LLAMA_CPP_DIR", "/hdd/Projects/llama.cpp")
LLAMA_EVAL = os.getenv("LLAMA_EVAL_BIN", os.path.join(LLAMA_CPP_DIR, "build/bin/llama-diffusion-gemma-eval"))
ZENO_BIN = "./zig-out/bin/zeno"
VOCAB_SIZE = 262_144

def softcap(x, cap=30.0):
    return cap * np.tanh(x / cap)

def unsoftcap(s, cap=30.0):
    ratio = np.clip(s / cap, -0.999999, 0.999999)
    return float(cap * np.arctanh(ratio))

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
    # Note: llama-diffusion-gemma-eval outputs res->t_logits from diffusion-gemma.cpp:605-612.
    # In GGML computation graph, final logit softcapping (30.0 * tanh(x / 30.0)) has already been applied.
    # Therefore, row0 contains single-softcapped logits.
    # We recover true raw logits via inverse softcap: 30.0 * arctanh(s / 30.0).
    with open(out_bin, "rb") as f:
        row0 = np.fromfile(f, dtype=np.float32, count=VOCAB_SIZE)

    cand_ids = [c["token_id"] for c in base_def["candidates"]]
    cand_labels = [c["label"] for c in base_def["candidates"]]

    oracle_sc = [round(float(row0[tid]), 4) for tid in cand_ids]
    oracle_raw = [round(unsoftcap(s), 4) for s in oracle_sc]

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

def update_fixture_file(fixture_path):
    with open(fixture_path, "r") as f:
        base_def = json.load(f)
    fixture_name = os.path.splitext(os.path.basename(fixture_path))[0]
    generate_fixture(fixture_name, base_def)

def main():
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = [
            "specs/fixtures/system_health_boolean.json",
            "specs/fixtures/incident_severity_multiclass.json",
            "specs/fixtures/boundary_256_incident_triage.json",
            "specs/fixtures/maintenance_priority_uncertain.json",
            "specs/fixtures/satisfaction_score_5class.json",
        ]
    for target in targets:
        update_fixture_file(target)

if __name__ == "__main__":
    main()

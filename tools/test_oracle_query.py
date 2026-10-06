#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import numpy as np

MODEL_PATH = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
LLAMA_EVAL = "/hdd/Projects/llama.cpp/build/bin/llama-diffusion-gemma-eval"
ZENO_BIN = "./zig-out/bin/zeno"
VOCAB_SIZE = 262_144

def softcap(x, cap=30.0):
    return cap * np.tanh(x / cap)

def evaluate_query(query, context, candidates, schema_type="choice", entropy_threshold=0.5):
    # candidates: list of (label, token_str)
    # 1. create dummy fixture
    cand_specs = []
    # we need token IDs, but zeno dump-fixture looks up candidates by token string
    for label, token in candidates:
        cand_specs.append({
            "label": label,
            "token": token,
            "token_id": 0 # placeholder
        })
    
    fixture_data = {
        "schema": "zeno.fixture/v1",
        "policy": "zeno.tolerance/v1",
        "description": "test query",
        "model": {"path": MODEL_PATH, "sha256": ""},
        "reference": {"repo": "", "commit": ""},
        "query": query,
        "context": context,
        "schema_type": schema_type,
        "canvas_length": 256,
        "answer_pos": 0,
        "answer_seed_token": 4,
        "entropy_threshold": entropy_threshold,
        "prompt_tokens": [],
        "candidates": cand_specs,
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

    tmp_json = "/tmp/test_eval.json"
    p_i32 = "/tmp/test_eval_p.i32"
    c_i32 = "/tmp/test_eval_c.i32"
    out_bin = "/tmp/test_eval_out.bin"

    with open(tmp_json, "w") as f:
        json.dump(fixture_data, f, indent=2)

    # Dump fixture
    dump_res = subprocess.run([ZENO_BIN, "dump-fixture", MODEL_PATH, tmp_json, p_i32, c_i32], capture_output=True, text=True)
    if dump_res.returncode != 0:
        print("dump-fixture failed:", dump_res.stderr)
        return None

    # Load prompt tokens
    p_tokens = np.fromfile(p_i32, dtype=np.int32).tolist()
    print(f"Prompt tokens count: {len(p_tokens)}")

    # Run eval
    env = os.environ.copy()
    env["NGL"] = "99"
    eval_res = subprocess.run([LLAMA_EVAL, MODEL_PATH, p_i32, c_i32, out_bin], env=env, capture_output=True, text=True)
    if eval_res.returncode != 0:
        print("llama-eval failed:", eval_res.stderr)
        return None

    # Read logits
    with open(out_bin, "rb") as f:
        row0 = np.fromfile(f, dtype=np.float32, count=VOCAB_SIZE)

    # To get candidate token IDs, we can inspect what Zeno resolved
    # Let's run zeno tokenize on each candidate
    cand_ids = []
    for label, token in candidates:
        tok_res = subprocess.run([ZENO_BIN, "tokenize", MODEL_PATH, token], capture_output=True, text=True)
        # Parse candidate resolution
        for line in tok_res.stdout.splitlines():
            if "Candidate resolution: ID " in line:
                tid = int(line.split("ID ")[1].split(" ")[0])
                cand_ids.append(tid)
                break
        else:
            raise RuntimeError(f"Could not resolve candidate {token}")

    raw_logits = [float(row0[tid]) for tid in cand_ids]
    sc_logits = [float(softcap(l)) for l in raw_logits]

    # Candidate softmax
    max_sc = max(sc_logits)
    exp_sc = [np.exp(l - max_sc) for l in sc_logits]
    sum_exp = sum(exp_sc)
    probs = [float(e / sum_exp) for e in exp_sc]

    # Entropy
    h_nats = 0.0
    for p in probs:
        if p > 1e-12:
            h_nats -= p * np.log(p)
    k = len(probs)
    h_norm = float(np.clip(h_nats / np.log(k), 0.0, 1.0)) if k > 1 else 0.0
    uncertain = bool(h_norm >= entropy_threshold)

    top_idx = int(np.argmax(probs))
    top_cand = candidates[top_idx][0]

    print("--------------------------------------------------")
    print(f"Decision:            {top_cand} (prob={probs[top_idx]:.4f})")
    print(f"Entropy (nats):      {h_nats:.4f}")
    print(f"Normalized Entropy:  {h_norm:.4f}")
    print(f"Uncertain:           {uncertain} (threshold={entropy_threshold})")
    for (label, _), tid, raw, sc, p in zip(candidates, cand_ids, raw_logits, sc_logits, probs):
        print(f"  {label} (ID {tid}): raw={raw:.4f}, sc={sc:.4f}, prob={p:.4f}")
    print("--------------------------------------------------")

    return {
        "p_tokens": p_tokens,
        "cand_ids": cand_ids,
        "decision": top_cand,
        "probs": probs,
        "raw_logits": raw_logits,
        "sc_logits": sc_logits,
        "h_nats": h_nats,
        "h_norm": h_norm,
        "uncertain": uncertain
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # custom query
        pass

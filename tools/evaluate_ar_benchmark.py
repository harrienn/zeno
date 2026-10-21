#!/usr/bin/env python3
"""Zeno — Autoregressive Baseline Evaluation Tool

Evaluates autoregressive GGUF models via llama-server using grammar-constrained decoding
on zeno.dataset/v1 datasets (e.g. ARC-Challenge, BoolQ).

Computes decision accuracy, per-candidate logprobs, latency, normalized entropy,
and exports results in zeno.dataset/v1 format compatible with tools/evaluate_calibration.py.

Usage:
  python3 tools/evaluate_ar_benchmark.py --dataset specs/fixtures/arc_challenge_150.json \
                                        --server http://127.0.0.1:8088 \
                                        --out specs/fixtures/arc_results/results_ar_qwen08b.json
"""

import sys
import json
import time
import math
import argparse
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple


def format_ar_prompt(example: Dict[str, Any], schema_type: str) -> Tuple[str, str]:
    """Formats prompt and grammar for autoregressive chat model."""
    query = example["query"]
    context = example.get("context")
    candidates = example["candidates"]
    
    if schema_type == "boolean":
        prompt_content = ""
        if context:
            prompt_content += f"Context: {context}\n\n"
        prompt_content += f"Question: {query}\nAnswer with Yes or No."
        tokens = [c["token"] for c in candidates]
        grammar_rules = " | ".join(f'"{t}"' for t in tokens)
        grammar = f"root ::= {grammar_rules}"
    elif schema_type == "choice":
        prompt_content = f"Question: {query}\n\nChoices:\n"
        for c in candidates:
            prompt_content += f"{c['label']}\n"
        prompt_content += "\nAnswer with only the letter of the correct choice."
        tokens = [c["token"] for c in candidates]
        grammar_rules = " | ".join(f'"{t}"' for t in tokens)
        grammar = f"root ::= {grammar_rules}"
    else:
        tokens = [c["token"] for c in candidates]
        prompt_content = f"{query}\nOptions: {', '.join(tokens)}"
        grammar_rules = " | ".join(f'"{t}"' for t in tokens)
        grammar = f"root ::= {grammar_rules}"

    full_prompt = (
        f"<|im_start|>system\nYou are a helpful assistant. Choose the best answer from the given choices.<|im_end|>\n"
        f"<|im_start|>user\n{prompt_content}<|im_end|>\n"
        f"<|im_start|>assistant\nAnswer: "
    )
    return full_prompt, grammar


def query_llama_server(
    server_url: str,
    prompt: str,
    grammar: str,
    candidates: List[Dict[str, Any]],
    n_predict: int = 1,
    temperature: float = 0.0,
    timeout_sec: float = 30.0
) -> Dict[str, Any]:
    url = f"{server_url.rstrip('/')}/completion"
    payload = {
        "prompt": prompt,
        "grammar": grammar,
        "n_predict": n_predict,
        "temperature": temperature,
        "n_probs": 10,
        "stream": False,
        "cache_prompt": True
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        t1 = time.perf_counter()
        raw_body = resp.read().decode("utf-8")
        result = json.loads(raw_body)
    latency_ms = (t1 - t0) * 1000.0

    content = result.get("content", "").strip()
    completion_probs = result.get("completion_probabilities", [])
    
    # Extract candidate logprobs
    # If the first token probabilities are returned:
    cand_logprobs = {}
    if completion_probs:
        first_token = completion_probs[0]
        chosen_token = first_token.get("token", "").strip()
        chosen_logprob = first_token.get("logprob", 0.0)
        cand_logprobs[chosen_token] = chosen_logprob
        for alt in first_token.get("top_logprobs", []):
            t_str = alt.get("token", "").strip()
            cand_logprobs[t_str] = alt.get("logprob", -99.0)

    # Normalize candidate probabilities over the candidate set
    raw_probs = []
    for c in candidates:
        token = c["token"]
        lp = cand_logprobs.get(token, None)
        if lp is not None:
            raw_probs.append(math.exp(lp))
        else:
            # Fallback if grammar suppressed alternative or not in top_logprobs
            raw_probs.append(1e-6 if token != content else 1.0)
            
    sum_p = sum(raw_probs)
    if sum_p > 0:
        norm_probs = [p / sum_p for p in raw_probs]
    else:
        norm_probs = [1.0 / len(candidates)] * len(candidates)

    # Calculate entropy
    entropy_nats = 0.0
    for p in norm_probs:
        if p > 1e-12:
            entropy_nats -= p * math.log(p)
    max_entropy = math.log(len(candidates)) if len(candidates) > 1 else 1.0
    norm_entropy = entropy_nats / max_entropy

    # Match decision with candidate label
    decision_label = content
    decision_prob = 1.0
    cand_results = []
    for i, c in enumerate(candidates):
        p = norm_probs[i]
        c_res = {
            "token": c["token"],
            "label": c["label"],
            "probability": p,
            "raw_logit": math.log(max(p, 1e-12))
        }
        cand_results.append(c_res)
        if c["token"] == content:
            decision_label = c["label"]
            decision_prob = p

    return {
        "decision": decision_label,
        "decision_token": content,
        "decision_probability": decision_prob,
        "candidates": cand_results,
        "entropy_nats": entropy_nats,
        "normalized_entropy": norm_entropy,
        "latency_ms": latency_ms,
        "timings": result.get("timings", {})
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AR baseline on zeno benchmark")
    parser.add_argument("--dataset", required=True, help="Path to zeno.dataset/v1 json")
    parser.add_argument("--server", default="http://127.0.0.1:8088", help="llama-server URL")
    parser.add_argument("--out", required=True, help="Path to output json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data["examples"]
    if args.limit:
        examples = examples[:args.limit]

    total = len(examples)
    correct_count = 0
    results = []
    latencies = []

    print(f"Evaluating {total} examples against AR server at {args.server}...")
    for idx, ex in enumerate(examples):
        qid = ex["id"]
        gt = ex["ground_truth"]
        schema_type = ex.get("schema_type", "choice")
        candidates = ex["candidates"]

        prompt, grammar = format_ar_prompt(ex, schema_type)
        res = query_llama_server(args.server, prompt, grammar, candidates)

        # Check correctness: match either decision label or token
        is_correct = (res["decision"] == gt or res["decision_token"] == gt or 
                      (schema_type == "choice" and gt.startswith(res["decision_token"] + ":")))
        if is_correct:
            correct_count += 1

        latencies.append(res["latency_ms"])
        
        eval_item = {
            "id": qid,
            "category": ex.get("category", "unknown"),
            "ground_truth": gt,
            "decision": res["decision"],
            "is_correct": is_correct,
            "decision_probability": res["decision_probability"],
            "entropy_nats": res["entropy_nats"],
            "normalized_entropy": res["normalized_entropy"],
            "uncertain": False,
            "steps_executed": 1,
            "latency_ms": res["latency_ms"],
            "candidates": res["candidates"],
            "stage_timings": res["timings"]
        }
        results.append(eval_item)

        if (idx + 1) % 25 == 0 or idx == total - 1:
            cur_acc = (correct_count / (idx + 1)) * 100.0
            print(f"[{idx+1}/{total}] Correct: {correct_count} | Accuracy: {cur_acc:.2f}% | Latency: {res['latency_ms']:.1f}ms")

    acc = (correct_count / total) * 100.0
    mean_lat = sum(latencies) / len(latencies)
    sorted_lat = sorted(latencies)
    median_lat = sorted_lat[len(sorted_lat) // 2]

    output_doc = {
        "schema": "zeno.dataset/v1",
        "benchmark": data.get("name", "unknown"),
        "model": "Qwen3.5-0.8B-UD-Q3_K_XL (Autoregressive GBNF)",
        "backend": "Autoregressive (llama.cpp)",
        "summary": {
            "total_examples": total,
            "correct_count": correct_count,
            "accuracy": acc / 100.0,
            "mean_latency_ms": mean_lat,
            "median_latency_ms": median_lat,
            "mean_normalized_entropy": sum(r["normalized_entropy"] for r in results) / total,
            "step_counts": [0, total, 0, 0, 0],
            "uncertain_count": 0
        },
        "results": results
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2)

    print(f"\nEvaluation Complete! Results saved to {args.out}")
    print(f"Total: {total} | Accuracy: {acc:.2f}% | Mean Latency: {mean_lat:.1f}ms | Median: {median_lat:.1f}ms")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Zeno — External Benchmark Dataset Importer
Converts standard multiple-choice and decision benchmarks (e.g. ARC-Challenge, ARC-Easy, SST-2)
into zeno.dataset/v1 schema for empirical calibration and decision evaluation.
"""

import argparse
import json
import sys
import requests

HF_SERVER_URL = "https://datasets-server.huggingface.co/rows"

LETTER_MAP = ["A", "B", "C", "D", "E", "F", "G", "H"]

def fetch_hf_rows(dataset_name: str, config: str, split: str, offset: int, limit: int):
    """Fetches rows from HuggingFace dataset server API in batches of up to 100."""
    all_rows = []
    current_offset = offset
    remaining = limit

    while remaining > 0:
        batch_size = min(100, remaining)
        params = {
            "dataset": dataset_name,
            "config": config,
            "split": split,
            "offset": current_offset,
            "limit": batch_size,
        }
        res = requests.get(HF_SERVER_URL, params=params, timeout=15)
        if res.status_code != 200:
            raise RuntimeError(f"HF API request failed (HTTP {res.status_code}): {res.text}")
        
        data = res.json()
        rows = [r["row"] for r in data.get("rows", [])]
        if not rows:
            break
        all_rows.extend(rows)
        current_offset += len(rows)
        remaining -= len(rows)
        print(f"Fetched {len(all_rows)} rows from {dataset_name}...")

    return all_rows[:limit]

def convert_arc(rows, dataset_name: str, split: str):
    """Converts ARC rows into zeno.dataset/v1 structure."""
    examples = []
    for idx, row in enumerate(rows):
        qid = str(row.get("id", f"arc_{idx}"))
        question = row.get("question", "").strip()
        raw_choices = row.get("choices", {})
        texts = raw_choices.get("text", [])
        raw_labels = raw_choices.get("label", [])
        raw_answer = str(row.get("answerKey", "")).strip()

        if len(texts) != len(raw_labels) or len(texts) < 2:
            continue

        # Standardize labels to A, B, C, D...
        candidates = []
        ground_truth = None

        # Build map from original label to standardized letter
        label_to_std = {}
        for i, raw_lbl in enumerate(raw_labels):
            std_letter = LETTER_MAP[i]
            label_to_std[str(raw_lbl).strip()] = std_letter
            choice_text = texts[i].strip()
            candidates.append({
                "token": std_letter,
                "label": f"{std_letter}: {choice_text}",
            })

        if raw_answer in label_to_std:
            ground_truth = label_to_std[raw_answer]
        elif raw_answer in LETTER_MAP[:len(texts)]:
            ground_truth = raw_answer
        else:
            # Fallback if numeric or 1-based index
            try:
                ans_idx = int(raw_answer) - 1
                if 0 <= ans_idx < len(texts):
                    ground_truth = LETTER_MAP[ans_idx]
            except ValueError:
                pass

        if not ground_truth:
            print(f"Warning: could not resolve answerKey '{raw_answer}' for query {qid}, skipping")
            continue

        # Ground truth matches the candidate's full label which engine.decide returns
        correct_label = None
        for cand in candidates:
            if cand["token"] == ground_truth:
                correct_label = cand["label"]
                break

        if not correct_label:
            continue

        examples.append({
            "id": qid,
            "category": "science_reasoning",
            "query": question,
            "context": None,
            "schema_type": "choice",
            "candidates": candidates,
            "ground_truth": correct_label,
        })

    return {
        "schema": "zeno.dataset/v1",
        "name": f"{dataset_name}_{split}_{len(examples)}",
        "description": f"Standardized {len(examples)} items from {dataset_name} ({split} split) in zeno.dataset/v1 schema",
        "examples": examples,
    }

def convert_sst2(rows, dataset_name: str, split: str):
    """Converts SST-2 sentiment rows into zeno.dataset/v1 structure."""
    examples = []
    for idx, row in enumerate(rows):
        qid = f"sst2_{idx}"
        sentence = row.get("sentence", "").strip()
        label_idx = row.get("label")
        if label_idx not in (0, 1):
            continue

        ground_truth = "Positive" if label_idx == 1 else "Negative"

        examples.append({
            "id": qid,
            "category": "sentiment_analysis",
            "query": "Is the sentiment of this text positive or negative?",
            "context": sentence,
            "schema_type": "choice",
            "candidates": [
                {"token": "Positive", "label": "Positive"},
                {"token": "Negative", "label": "Negative"},
            ],
            "ground_truth": ground_truth,
        })

    return {
        "schema": "zeno.dataset/v1",
        "name": f"sst2_{split}_{len(examples)}",
        "description": f"Standardized {len(examples)} items from SST-2 ({split} split) in zeno.dataset/v1 schema",
        "examples": examples,
    }

def main():
    parser = argparse.ArgumentParser(description="Import external benchmark into zeno.dataset/v1 JSON")
    parser.add_argument("--dataset", choices=["arc-challenge", "arc-easy", "sst2"], default="arc-challenge",
                        help="Benchmark dataset identifier")
    parser.add_argument("--split", default="test", help="Dataset split (test, validation, train)")
    parser.add_argument("--limit", type=int, default=150, help="Number of examples to import")
    parser.add_argument("--offset", type=int, default=0, help="Starting row offset")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    if args.dataset in ("arc-challenge", "arc-easy"):
        config = "ARC-Challenge" if args.dataset == "arc-challenge" else "ARC-Easy"
        print(f"Fetching {args.limit} rows from allenai/ai2_arc ({config}, {args.split})...")
        rows = fetch_hf_rows("allenai/ai2_arc", config, args.split, args.offset, args.limit)
        dataset_obj = convert_arc(rows, args.dataset, args.split)
    elif args.dataset == "sst2":
        split = "validation" if args.split == "test" else args.split
        print(f"Fetching {args.limit} rows from stanfordnlp/sst2 (default, {split})...")
        rows = fetch_hf_rows("stanfordnlp/sst2", "default", split, args.offset, args.limit)
        dataset_obj = convert_sst2(rows, "sst2", split)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"Successfully converted {len(dataset_obj['examples'])} items to zeno.dataset/v1.")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset_obj, f, indent=2)
    print(f"Saved dataset to {args.out}")

if __name__ == "__main__":
    main()

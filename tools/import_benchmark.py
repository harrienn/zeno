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

def convert_arc(rows, dataset_name: str, split: str, rotate: int = 0):
    """Converts ARC rows into zeno.dataset/v1 structure, optionally applying option rotation."""
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

        # Resolve original target index for ground truth
        orig_target_idx = None
        for i, raw_lbl in enumerate(raw_labels):
            if str(raw_lbl).strip() == raw_answer:
                orig_target_idx = i
                break
        if orig_target_idx is None:
            if raw_answer in LETTER_MAP[:len(texts)]:
                orig_target_idx = LETTER_MAP.index(raw_answer)
            else:
                try:
                    ans_num = int(raw_answer) - 1
                    if 0 <= ans_num < len(texts):
                        orig_target_idx = ans_num
                except ValueError:
                    pass

        if orig_target_idx is None:
            print(f"Warning: could not resolve answerKey '{raw_answer}' for query {qid}, skipping")
            continue

        # Apply cyclic rotation if requested (Review Round 33 §5: Option Permutation Control)
        n = len(texts)
        shift = rotate % n
        rotated_texts = [None] * n
        for i in range(n):
            new_idx = (i + shift) % n
            rotated_texts[new_idx] = texts[i].strip()

        new_target_idx = (orig_target_idx + shift) % n

        candidates = []
        for i in range(n):
            letter = LETTER_MAP[i]
            candidates.append({
                "token": letter,
                "label": f"{letter}: {rotated_texts[i]}",
            })

        correct_label = candidates[new_target_idx]["label"]

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
        "name": f"{dataset_name}_{split}_{len(examples)}" + (f"_rot{rotate}" if rotate else ""),
        "description": f"Standardized {len(examples)} items from {dataset_name} ({split} split) in zeno.dataset/v1 schema" + (f" (options cyclically shifted by {rotate})" if rotate else ""),
        "rotation": rotate,
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

def convert_boolq(rows, dataset_name: str, split: str, max_words: int = 300, offset: int = 0):
    """Converts BoolQ reading comprehension rows into zeno.dataset/v1 structure."""
    examples = []
    for idx, row in enumerate(rows):
        qid = f"boolq_{idx}"
        question = row.get("question", "").strip()
        passage = row.get("passage", "").strip()
        ans = row.get("answer")
        if ans not in (True, False):
            continue

        # Format question with proper capitalization and terminal question mark
        if question and not question.endswith("?"):
            question = question + "?"
        if question:
            question = question[0].upper() + question[1:]

        # Length guard to keep prompt comfortably within CUDA workspace sequence length (512)
        words = passage.split()
        orig_word_count = len(words)
        is_truncated = False
        if orig_word_count > max_words:
            passage = " ".join(words[:max_words]) + "..."
            is_truncated = True

        ground_truth = "Yes" if ans is True else "No"

        ex_dict = {
            "id": qid,
            "source_index": offset + idx,
            "category": "reading_comprehension",
            "query": question,
            "context": passage,
            "schema_type": "boolean",
            "candidates": [
                {"token": "Yes", "label": "Yes"},
                {"token": "No", "label": "No"},
            ],
            "ground_truth": ground_truth,
        }
        if is_truncated:
            ex_dict["source_truncated"] = True
            ex_dict["original_word_count"] = orig_word_count
            ex_dict["truncation_word_limit"] = max_words

        examples.append(ex_dict)

    return {
        "schema": "zeno.dataset/v1",
        "name": f"boolq_{split}_{len(examples)}",
        "description": f"Standardized {len(examples)} items from google/boolq ({split} split) in zeno.dataset/v1 schema",
        "examples": examples,
    }

def main():
    parser = argparse.ArgumentParser(description="Import external benchmark into zeno.dataset/v1 JSON")
    parser.add_argument("--dataset", choices=["arc-challenge", "arc-easy", "sst2", "boolq"], default="arc-challenge",
                        help="Benchmark dataset identifier")
    parser.add_argument("--split", default="test", help="Dataset split (test, validation, train)")
    parser.add_argument("--limit", type=int, default=150, help="Number of examples to import")
    parser.add_argument("--offset", type=int, default=0, help="Starting row offset")
    parser.add_argument("--rotate", type=int, default=0, help="Cyclically shift option order by N positions (choice-order control)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    if args.dataset in ("arc-challenge", "arc-easy"):
        config = "ARC-Challenge" if args.dataset == "arc-challenge" else "ARC-Easy"
        print(f"Fetching {args.limit} rows from allenai/ai2_arc ({config}, {args.split})...")
        rows = fetch_hf_rows("allenai/ai2_arc", config, args.split, args.offset, args.limit)
        dataset_obj = convert_arc(rows, args.dataset, args.split, rotate=args.rotate)
    elif args.dataset == "sst2":
        split = "validation" if args.split == "test" else args.split
        print(f"Fetching {args.limit} rows from stanfordnlp/sst2 (default, {split})...")
        rows = fetch_hf_rows("stanfordnlp/sst2", "default", split, args.offset, args.limit)
        dataset_obj = convert_sst2(rows, "sst2", split)
    elif args.dataset == "boolq":
        split = "validation" if args.split == "test" else args.split
        print(f"Fetching {args.limit} rows from google/boolq (default, {split})...")
        rows = fetch_hf_rows("google/boolq", "default", split, args.offset, args.limit)
        dataset_obj = convert_boolq(rows, "boolq", split, offset=args.offset)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"Successfully converted {len(dataset_obj['examples'])} items to zeno.dataset/v1.")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset_obj, f, indent=2)
    print(f"Saved dataset to {args.out}")

if __name__ == "__main__":
    main()

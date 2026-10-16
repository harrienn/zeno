#!/usr/bin/env python3
"""Zeno — Layer-by-Layer Activation Drift Analyzer.

Compares intermediate hidden activation vectors dumped by Zeno and llama.cpp
at the answer position across all 30 transformer layers.

Usage:
    python3 tools/compare_layers.py <zeno_dump.bin> <llama_dump.bin>
"""

import sys
import struct
import numpy as np

RECORD_NAME_SIZE = 32
EMBEDDING_DIM = 2816
RECORD_BYTES = RECORD_NAME_SIZE + EMBEDDING_DIM * 4

def load_dump_file(path):
    records = {}
    with open(path, "rb") as f:
        while True:
            chunk = f.read(RECORD_BYTES)
            if len(chunk) < RECORD_BYTES:
                break
            name_raw = chunk[:RECORD_NAME_SIZE]
            name = name_raw.split(b"\x00")[0].decode("ascii", errors="replace")
            vec = np.frombuffer(chunk[RECORD_NAME_SIZE:], dtype=np.float32)
            records[name] = vec.copy()
    return records

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <zeno_dump.bin> <llama_dump.bin>")
        sys.exit(1)

    zeno_path = sys.argv[1]
    llama_path = sys.argv[2]

    zeno_records = load_dump_file(zeno_path)
    llama_records = load_dump_file(llama_path)

    print("=" * 115)
    print(f" Zeno vs llama.cpp Layer-by-Layer Activation Drift Analysis")
    print(f" Zeno dump:  {zeno_path} ({len(zeno_records)} records)")
    print(f" Llama dump: {llama_path} ({len(llama_records)} records)")
    print("=" * 115)

    headers = ["Layer / Tensor", "Zeno L2", "Llama L2", "L2 Delta", "Cos Sim", "Max Abs Diff", "Rel L2 Error"]
    print(f"| {headers[0]:<16} | {headers[1]:>10} | {headers[2]:>10} | {headers[3]:>10} | {headers[4]:>10} | {headers[5]:>12} | {headers[6]:>12} |")
    print(f"|:{'-'*16}-|-{'-'*10}:|-{'-'*10}:|-{'-'*10}:|-{'-'*10}:|-{'-'*12}:|-{'-'*12}:|")

    ordered_keys = []
    if "inp_region" in zeno_records:
        ordered_keys.append("inp_region")
    for i in range(30):
        k = f"l_out-{i}"
        if k in zeno_records:
            ordered_keys.append(k)
    if "result_norm" in zeno_records:
        ordered_keys.append("result_norm")

    # Add any extra keys
    for k in zeno_records:
        if k not in ordered_keys:
            ordered_keys.append(k)

    min_cos_sim = 1.0
    max_rel_err = 0.0

    for name in ordered_keys:
        if name not in llama_records:
            print(f"| {name:<16} | {'(present)':>10} | {'(missing)':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>12} | {'N/A':>12} |")
            continue

        z_vec = zeno_records[name]
        l_vec = llama_records[name]

        z_l2 = float(np.linalg.norm(z_vec))
        l_l2 = float(np.linalg.norm(l_vec))
        delta_l2 = abs(z_l2 - l_l2)

        cos_sim = cosine_similarity(z_vec, l_vec)
        max_abs = float(np.max(np.abs(z_vec - l_vec)))
        diff_l2 = float(np.linalg.norm(z_vec - l_vec))
        rel_err = (diff_l2 / l_l2) * 100.0 if l_l2 > 0 else 0.0

        min_cos_sim = min(min_cos_sim, cos_sim)
        max_rel_err = max(max_rel_err, rel_err)

        print(f"| {name:<16} | {z_l2:>10.4f} | {l_l2:>10.4f} | {delta_l2:>10.4f} | {cos_sim:>10.6f} | {max_abs:>12.4f} | {rel_err:>11.2f}% |")

    print("-" * 115)
    print(f"Summary Metrics across {len(ordered_keys)} layers:")
    print(f"  Minimum Cosine Similarity: {min_cos_sim:.6f}")
    print(f"  Maximum Relative L2 Error:  {max_rel_err:.2f}%")
    print("=" * 115)

if __name__ == "__main__":
    main()

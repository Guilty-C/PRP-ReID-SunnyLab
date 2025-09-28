#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CLIP retrieval results
- Input: JSON file (from retrieve.py)
- Output: Average similarity, Rank-1, mAP
"""

import argparse
import json
import numpy as np
from tqdm import tqdm

def compute_map(results):
    """Compute mean Average Precision"""
    aps = []
    for qid, retrieved in results.items():
        # ground truth is the same image_id
        gt = qid
        y_true = [1 if rid == gt else 0 for rid, _ in retrieved]
        if sum(y_true) == 0:
            aps.append(0)
            continue
        # precision at each hit
        precisions = []
        correct = 0
        for idx, val in enumerate(y_true, start=1):
            if val == 1:
                correct += 1
                precisions.append(correct / idx)
        aps.append(np.mean(precisions))
    return np.mean(aps)

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP retrieval results")
    parser.add_argument("--results", required=True, help="JSON file from retrieve.py")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    sims = []
    correct_top1 = 0
    total = len(results)

    for qid, retrieved in tqdm(results.items(), desc="Evaluating"):
        if not retrieved:
            continue
        top1_id, top1_sim = retrieved[0]
        sims.append(top1_sim)
        if top1_id == qid:
            correct_top1 += 1

    avg_sim = np.mean(sims)
    rank1 = correct_top1 / total
    mAP = compute_map(results)

    print("===== Evaluation Results =====")
    print(f"Queries: {total}")
    print(f"Average Top-1 Similarity: {avg_sim:.4f}")
    print(f"Rank-1 Accuracy: {rank1*100:.2f}%")
    print(f"mAP: {mAP*100:.2f}%")

if __name__ == "__main__":
    main()

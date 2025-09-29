#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CLIP retrieval results (Market1501 style, ignoring same-camera matches)
- Input: JSON file (from retrieve.py)
- Output: Average similarity, Rank-1, mAP
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm


def parse_pid_cam(fname: str):
    """
    Market1501 文件名格式: 0001_c1s1_001051_00.jpg
    - 前4位: person ID
    - 第二段 'cX': camera ID
    """
    parts = os.path.basename(fname).split("_")
    pid = parts[0]
    cam = parts[1]  # e.g. 'c1s1' → camera = 'c1'
    cam_id = cam[:2]  # 'c1'
    return pid, cam_id


def compute_map(results):
    """Compute mean Average Precision (ignore same-camera matches)"""
    aps = []
    for qid, retrieved in results.items():
        q_pid, q_cam = parse_pid_cam(qid)

        # 生成标签：同 PID 且不同摄像头才算正样本
        y_true = []
        for rid, _ in retrieved:
            r_pid, r_cam = parse_pid_cam(rid)
            if r_pid == q_pid and r_cam != q_cam:
                y_true.append(1)
            else:
                y_true.append(0)

        if sum(y_true) == 0:  # 没有正样本
            aps.append(0)
            continue

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

        q_pid, q_cam = parse_pid_cam(qid)
        t_pid, t_cam = parse_pid_cam(top1_id)

        # Rank-1 只算不同摄像头的同 PID
        if q_pid == t_pid and q_cam != t_cam:
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

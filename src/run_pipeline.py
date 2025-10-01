#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-stop pipeline for prompt evaluation in Person ReID:
1. Prepare data and generate index file
2. Generate captions with prompt (using batch parallel processing)
3. Compute CLIP-based retrieval
4. Evaluate (mAP, Rank-1)
5. Save logs (auto exp001, exp002…)
"""

import os, sys, argparse, subprocess, datetime, json

def get_next_exp_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("exp")]
    if not existing:
        return os.path.join(base_dir, "exp001")
    nums = [int(d[3:]) for d in existing if d[3:].isdigit()]
    next_num = max(nums) + 1
    return os.path.join(base_dir, f"exp{next_num:03d}")

def run(cmd, log_file):
    full_cmd = [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd
    print(f"[Pipeline] Running: {' '.join(full_cmd)}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[CMD] {' '.join(full_cmd)}\n")
        # 捕捉 stdout 和 stderr
        result = subprocess.run(full_cmd, capture_output=True, text=True)
        f.write(result.stdout)
        f.write(result.stderr)

def main(args):
    exp_dir = get_next_exp_dir(args.out_dir)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "pipeline.log")

    # 1. prepare_data → 生成索引
    index_file = os.path.join(exp_dir, "index.json")
    run([
        "python", "src/prepare_data.py",
        "--data_root", args.data_root,
        "--split", "query",
        "--out_index", index_file
    ], log_file)
    
    # 如果限制数量 → 截断 index.json
    if args.num != "all":
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = data[:int(args.num)]
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Pipeline] 截取前 {args.num} 张图片作为实验数据")

    # 2. gen_caption_batch → 批量生成 captions
    captions_file = os.path.join(exp_dir, "captions.csv")
    run([
        "python", "src/gen_caption_batch.py", index_file,
        "--prompt_file", args.prompt_file,
        "--out", captions_file,
        "--batch_size", "8",
        "--num_workers", "4"
    ], log_file)

    # 3. retrieve → 检索
    results_file = os.path.join(exp_dir, "results.json")
    run([
        "python", "src/retrieve.py",
        "--captions", captions_file,
        "--gallery", os.path.join(args.data_root, "bounding_box_test"),
        "--out", results_file,
        "--topk", "5"
    ], log_file)

    # 4. eval_clip_results → 评估
    run([
        "python", "src/eval_clip_results.py",
        "--results", results_file
    ], log_file)

    # ===== 在这里记录 Prompt 和分数 =====
    # 读取 Prompt
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    # 从 results.json 里读取结果（需要 eval_clip_results.py 输出 JSON 格式的结果）
    avg_sim, rank1, mAP = "N/A", "N/A", "N/A"
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        avg_sim = results_data.get("avg_similarity", "N/A")
        rank1   = results_data.get("rank1", "N/A")
        mAP     = results_data.get("mAP", "N/A")
    except Exception:
        pass

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n===== Experiment Summary =====\n")
        f.write(f"Prompt file: {args.prompt_file}\n")
        f.write(f"Prompt text:\n{prompt_text}\n\n")
        f.write(f"Average Top-1 Similarity: {avg_sim}\n")
        f.write(f"Rank-1 Accuracy: {rank1}\n")
        f.write(f"mAP: {mAP}\n")
        f.write("=============================\n\n")

    print(f"✅ Pipeline finished. Results saved in {exp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/market1501", help="数据集根目录")
    parser.add_argument("--prompt_file", type=str, default="prompts/base.txt", help="Prompt 文件")
    parser.add_argument("--num", type=str, default="all", help="使用的图片数量 (int 或 'all')")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="输出目录")
    args = parser.parse_args()
    main(args)

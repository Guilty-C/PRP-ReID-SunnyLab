# tools/subset_embeds_by_pid.py
from __future__ import annotations
import argparse, os
import numpy as np

def read_lines(p):
    # 兼容 UTF-8 / UTF-8-SIG
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")
    return lines

def main():
    ap = argparse.ArgumentParser("Subset image embeddings by a given pid list")
    ap.add_argument("--emb-image", required=True, help="输入：图像向量 .npy  (N, D)")
    ap.add_argument("--labels-image", required=True, help="输入：图像标签 .txt  (N 行，对应上面 .npy)")
    ap.add_argument("--keep-pids", required=True, help="输入：保留的 pid 列表 .txt（每行一个 pid）")
    ap.add_argument("--out-emb", required=True, help="输出：子集图像向量 .npy")
    ap.add_argument("--out-labels", required=True, help="输出：子集图像标签 .txt")
    args = ap.parse_args()

    I = np.load(args.emb_image)                     # (N, D)
    Li = read_lines(args.labels_image)              # len N
    keep = set(read_lines(args.keep_pids))          # pid set

    assert len(Li) == I.shape[0], "labels-image 与 emb-image 数量不一致"
    mask = np.array([pid in keep for pid in Li], dtype=bool)
    n_keep = int(mask.sum())
    if n_keep == 0:
        raise RuntimeError("没有任何图片的 pid 命中 keep-pids，请检查 pid 格式是否一致（零填等）")

    I_sub = I[mask]
    Li_sub = [pid for pid, m in zip(Li, mask) if m]

    os.makedirs(os.path.dirname(args.out_emb), exist_ok=True)
    np.save(args.out_emb, I_sub.astype(np.float32))
    os.makedirs(os.path.dirname(args.out_labels), exist_ok=True)
    with open(args.out_labels, "w", encoding="utf-8") as f:
        f.write("\n".join(Li_sub) + "\n")

    uniq = len(set(Li_sub))
    print(f"[subset] in={I.shape} -> out={I_sub.shape}  kept_images={n_keep}  kept_unique_pids={uniq}")
    print(f"[subset] saved: {args.out_emb}")
    print(f"[subset] saved: {args.out_labels}")

if __name__ == "__main__":
    main()

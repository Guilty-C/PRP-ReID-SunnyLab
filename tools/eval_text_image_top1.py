# tools/eval_text_image_top1.py
from __future__ import annotations
import argparse
import numpy as np

def _load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser("Evaluate text-image similarity (Top-1 sim; optional Rank-1)")
    ap.add_argument("--text", required=True, help=".npy 文本嵌入 (N_text, D)")
    ap.add_argument("--image", required=True, help=".npy 图像嵌入 (N_img, D)")
    ap.add_argument("--labels-text", default=None, help="文本标签 .txt（每行一个），用于 Rank-1（可选）")
    ap.add_argument("--labels-image", default=None, help="图像标签 .txt（每行一个），用于 Rank-1（可选）")
    args = ap.parse_args()

    T = np.load(args.text)      # (Nt, D)
    I = np.load(args.image)     # (Ni, D)

    # 防御性归一化（若已归一化则影响可忽略）
    eps = 1e-8
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + eps)
    I = I / (np.linalg.norm(I, axis=1, keepdims=True) + eps)

    # 余弦相似度 (Nt, Ni)
    S = T @ I.T

    # 每条文本的 top-1 相似度与索引
    top1_idx = np.argmax(S, axis=1)
    top1_sim = S[np.arange(S.shape[0]), top1_idx]
    avg_top1 = float(np.mean(top1_sim))

    print(f"Texts: {T.shape[0]}  Images: {I.shape[0]}")
    print(f"Average Top-1 Similarity: {avg_top1:.4f}")

    # 可选 Rank-1（需要标签对齐）
    if args.labels_text and args.labels_image:
        lt = _load_labels(args.labels_text)
        li = _load_labels(args.labels_image)
        if len(lt) != T.shape[0]:
            raise SystemExit(f"labels-text 数量 {len(lt)} 与文本嵌入 {T.shape[0]} 不一致")
        if len(li) != I.shape[0]:
            raise SystemExit(f"labels-image 数量 {len(li)} 与图像嵌入 {I.shape[0]} 不一致")

        # 命中：top-1 对应的图像标签 == 文本标签
        hits = 0
        for i, idx in enumerate(top1_idx):
            if lt[i] == li[idx]:
                hits += 1
        rank1 = hits / len(lt)
        print(f"Rank-1 (optional): {rank1:.4f}")

if __name__ == "__main__":
    main()

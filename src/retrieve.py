import argparse
import json
import os
import numpy as np
import clip
import torch
from PIL import Image
import csv
from tqdm import tqdm


# 初始化 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def encode_text(text: str) -> np.ndarray:
    """使用 CLIP 编码文本"""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # 归一化
    return feats.cpu().numpy()[0]


def encode_image(img_path: str) -> np.ndarray:
    """使用 CLIP 编码图像"""
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(image)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # 归一化
    return feats.cpu().numpy()[0]


def load_captions(path: str):
    """支持 CSV/JSONL 格式的描述文件"""
    records = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    elif path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 假设 CSV 至少有 "image_id" 和 "caption" 两列
                records.append({
    "image_id": row.get("path", ""),   # 用 path 当作唯一ID
    "caption": row.get("caption", "")
})
    else:
        raise ValueError(f"不支持的文件格式: {path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str, required=True, help="描述文件 captions.csv/jsonl")
    parser.add_argument("--gallery", type=str, required=False, default=None, help="图库路径 (若提供则加载实际图像，否则随机模拟)")
    parser.add_argument("--out", type=str, required=True, help="输出结果文件路径")
    parser.add_argument("--topk", type=int, default=5, help="返回前K个结果")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 读取描述文件
    records = load_captions(args.captions)

    # 构建图库
    gallery = {}
    files = [f for f in os.listdir(args.gallery) if f.lower().endswith((".jpg",".png"))]
    for fname in tqdm(files, desc="Encoding gallery images"):
        img_path = os.path.join(args.gallery, fname)
        img_id = fname
        gallery[img_id] = encode_image(img_path)


    # 执行检索
    results = {}
    for rec in tqdm(records, desc="Processing queries"):
        q_text = rec["caption"]
        q_vec = encode_text(q_text)

        sims = {}
        for gid, gvec in gallery.items():
            sims[gid] = float(np.dot(q_vec, gvec) /
                            (np.linalg.norm(q_vec) * np.linalg.norm(gvec)))

        topk = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:args.topk]
        results[rec["image_id"]] = topk

    # 保存结果
    with open(args.out, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"[retrieve] 完成检索，共处理 {len(records)} 个查询。")
    print(f"[retrieve] 结果写入: {args.out}")

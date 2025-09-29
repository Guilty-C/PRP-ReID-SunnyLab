import argparse
import json
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def load_captions(path: str):
    """支持 CSV/JSONL 格式的描述文件"""
    records = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    elif path.endswith(".csv"):
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "image_id": row.get("image_id", row.get("path", "")),
                    "caption": row.get("caption", "")
                })
    else:
        raise ValueError(f"不支持的文件格式: {path}")
    return records


def encode_image(model, processor, img_path, device):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def encode_text(model, processor, text, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str, required=True, help="描述文件 captions.csv/jsonl")
    parser.add_argument("--gallery", type=str, required=True, help="图库路径 (文件夹，包含图像)")
    parser.add_argument("--out", type=str, required=True, help="输出结果文件路径")
    parser.add_argument("--topk", type=int, default=5, help="返回前K个结果")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 加载 CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[retrieve] Using device: {device}")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True   # ✅ 强制使用 safetensors 权重，避免 torch.load 漏洞
    ).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 读取描述文件
    records = load_captions(args.captions)
    print(f"[retrieve] 加载 {len(records)} 条查询")

    # 构建图库
    gallery = {}
    files = [f for f in os.listdir(args.gallery) if f.lower().endswith((".jpg", ".png"))]
    for fname in tqdm(files, desc="Encoding gallery images"):
        img_path = os.path.join(args.gallery, fname)
        gallery[fname] = encode_image(model, processor, img_path, device)

    # 执行检索
    results = {}
    for rec in tqdm(records, desc="Processing queries"):
        q_text = rec["caption"]
        q_vec = encode_text(model, processor, q_text, device)

        sims = {gid: float(np.dot(q_vec, gvec)) for gid, gvec in gallery.items()}
        topk = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:args.topk]
        results[rec["image_id"]] = topk

    # 保存结果
    with open(args.out, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"[retrieve] 完成检索，共处理 {len(records)} 个查询。")
    print(f"[retrieve] 结果写入: {args.out}")


if __name__ == "__main__":
    main()

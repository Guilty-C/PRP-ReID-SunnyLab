import argparse, os, os.path as op
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
import csv, json


def load_captions(file_path):
    """
    支持 CSV 和 JSONL 两种输入格式
    """
    caps = []
    if file_path.endswith(".csv"):
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "caption" not in reader.fieldnames:
                raise ValueError(f"CSV 文件 {file_path} 中没有 'caption' 列，实际字段: {reader.fieldnames}")
            for row in reader:
                caps.append(row["caption"])
    else:  # JSONL
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        caps.append(json.loads(line)["caption"])
                    except Exception as e:
                        raise ValueError(f"解析 JSONL 出错: {e}, 行内容: {line[:100]}...")
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="./outputs/captions/captions.jsonl")
    ap.add_argument("--out_feats", default="./outputs/feats/text.npy")
    args = ap.parse_args()

    # 读取 caption
    caps = load_captions(args.captions)
    print(f"[encode_clip] 加载 {len(caps)} 条 captions")

    # 加载 CLIP 模型（强制使用 safetensors 避免 torch.load 报错）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[encode_clip] Using device: {device}")

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True   # ✅ 关键：强制用 safetensors
    ).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 编码文本
    all_feats = []
    batch_size = 64
    for i in range(0, len(caps), batch_size):
        batch_caps = caps[i:i+batch_size]
        inputs = processor(
            text=batch_caps,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        all_feats.append(feats.cpu().numpy())

        if (i // batch_size + 1) % 10 == 0:
            print(f"[encode_clip] 已完成 {i+len(batch_caps)}/{len(caps)}")

    feats = np.vstack(all_feats).astype("float32")

    # 保存结果
    os.makedirs(op.dirname(args.out_feats), exist_ok=True)
    np.save(args.out_feats, feats)
    print(f"[encode_clip] encoded {len(caps)} -> {args.out_feats} (dim={feats.shape[1]})")


if __name__ == "__main__":
    main()

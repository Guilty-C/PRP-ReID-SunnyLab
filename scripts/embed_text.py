# scripts/embed_text.py
from __future__ import annotations
import argparse
import json
import os
from typing import List

import numpy as np
from tqdm import tqdm

# 本地包
from providers import create_provider

def _read_lines_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def _read_lines_jsonl(path: str) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 兼容键名：优先 text/caption；否则尝试 value 为字符串的唯一字段
            if "text" in obj and isinstance(obj["text"], str):
                texts.append(obj["text"])
            elif "caption" in obj and isinstance(obj["caption"], str):
                texts.append(obj["caption"])
            else:
                # fallback: 取第一个 string 值
                cand = None
                for v in obj.values():
                    if isinstance(v, str):
                        cand = v
                        break
                if cand is None:
                    raise ValueError(f"JSONL 行无法解析为文本: {obj}")
                texts.append(cand)
    return texts

def _read_input(path: str) -> List[str]:
    if path.lower().endswith(".jsonl"):
        return _read_lines_jsonl(path)
    return _read_lines_txt(path)

def main():
    ap = argparse.ArgumentParser("Embed texts into CLIP (or compatible) embeddings")
    ap.add_argument("--input", required=True, help="输入文件（.jsonl 或 .txt，每行一条）")
    ap.add_argument("--out", required=True, help="输出 .npy 路径")
    ap.add_argument("--provider", default="clip-text-local", help="provider 名称（如 clip-text-local）")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    texts = _read_input(args.input)
    if len(texts) == 0:
        raise SystemExit("输入为空。")

    print(f"[embed_text] provider = {args.provider}")
    print(f"[embed_text] num_texts = {len(texts)}")
    print(f"[embed_text] batch_size = {args.batch_size}")

    provider = create_provider(args.provider)

    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i : i + args.batch_size]
        arr = provider.embed(batch, batch_size=len(batch))
        embs.append(arr)

    embs = np.concatenate(embs, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, embs)
    print(f"[embed_text] saved: {args.out}  shape={embs.shape}")

if __name__ == "__main__":
    main()

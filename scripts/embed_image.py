# scripts/embed_image.py
from __future__ import annotations
import argparse, os, math, re
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
import open_clip  # pip install open-clip-torch
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _iter_images(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files

_pid_re = re.compile(r"^(-?\d+)_")  # Market-1501: "0002_c1s1_XXXX.jpg"；-1 为 junk

def _parse_pid(path: str) -> Optional[str]:
    bn = os.path.basename(path)
    m = _pid_re.match(bn)
    if not m:
        return None
    pid = m.group(1)
    if pid == "-1":
        return None  # 默认丢弃 junk
    return pid

def _resolve_split(root: str, split: str, glob: Optional[str]) -> str:
    if glob:
        # 允许自定义子目录/通配
        base = os.path.join(root, glob)
        if not os.path.exists(os.path.dirname(base)) and not os.path.exists(base):
            raise SystemExit(f"[embed_image] glob 不存在: {base}")
        return base
    split = split.lower()
    if split == "gallery":
        sub = "bounding_box_test"
    elif split == "query":
        sub = "query"
    elif split == "train":
        sub = "bounding_box_train"
    else:
        raise SystemExit(f"[embed_image] 不支持的 split: {split}")
    d = os.path.join(root, sub)
    if not os.path.isdir(d):
        raise SystemExit(f"[embed_image] 目录不存在: {d}")
    return d

def main():
    ap = argparse.ArgumentParser("Embed images with local CLIP and export labels")
    ap.add_argument("--root", required=True, help="Market-1501 根目录（或自定义数据根）")
    ap.add_argument("--split", default="gallery", choices=["gallery", "query", "train"],
                    help="使用的标准子集：gallery=bounding_box_test, query=query, train=bounding_box_train")
    ap.add_argument("--glob", default=None, help="可选：相对 root 的自定义子路径/通配，优先于 --split")
    ap.add_argument("--out", required=True, help="输出 .npy 路径 (N, D)")
    ap.add_argument("--labels-out", default=None, help="可选：输出标签 .txt（每行一个 pid）")
    ap.add_argument("--paths-out", default=None, help="可选：输出图片路径清单 .txt（每行一个路径）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model", default=os.getenv("CLIP_LOCAL_MODEL", "ViT-B-16"))
    ap.add_argument("--pretrained", default=os.getenv("CLIP_LOCAL_PRETRAIN", "laion2b_s34b_b88k"))
    args = ap.parse_args()

    device = _device()
    print(f"[embed_image] model={args.model}  pretrained={args.pretrained}  device={device}")

    # 模型与图像预处理
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model,
        pretrained=args.pretrained,
        device=device,
    )
    model.eval()

    # 选定数据子集
    data_dir = _resolve_split(args.root, args.split, args.glob)

    # 收集图片路径
    all_imgs = _iter_images(data_dir)
    if len(all_imgs) == 0:
        raise SystemExit(f"[embed_image] 未找到图片: {data_dir}")

    # 解析 pid（丢弃 -1 junk）
    paths: List[str] = []
    labels: List[str] = []
    for p in all_imgs:
        pid = _parse_pid(p)
        if pid is None:
            continue
        paths.append(p)
        labels.append(pid)

    if len(paths) == 0:
        raise SystemExit("[embed_image] 过滤 junk 后没有图片。")

    print(f"[embed_image] images={len(paths)}  from={data_dir}")

    # 批量编码
    embs: List[np.ndarray] = []
    steps = math.ceil(len(paths) / args.batch_size)
    with torch.no_grad():
        for i in tqdm(range(steps)):
            batch_paths = paths[i*args.batch_size : (i+1)*args.batch_size]
            ims = []
            for bp in batch_paths:
                img = Image.open(bp).convert("RGB")
                ims.append(preprocess(img))
            ims = torch.stack(ims, dim=0).to(device)

            feats = model.encode_image(ims).float()
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)  # 归一化
            embs.append(feats.cpu().numpy())

    E = np.concatenate(embs, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, E)
    print(f"[embed_image] saved: {args.out}  shape={E.shape}")

    # 可选：输出 label 与路径清单
    def _dump_list(lst: List[str], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for x in lst:
                f.write(str(x).strip() + "\n")
        print(f"[embed_image] saved: {path}  lines={len(lst)}")

    if args.labels_out:
        _dump_list(labels, args.labels_out)
    if args.paths_out:
        _dump_list(paths, args.paths_out)

if __name__ == "__main__":
    main()

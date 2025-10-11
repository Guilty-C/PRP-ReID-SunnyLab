"""Extract embeddings and evaluate on the official Market-1501 test split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.market1501_clean import build_market1501_transforms
from train_reid_baseline import (
    ReIDBaseline,
    cosine_distance_matrix,
    make_tqdm,
    market1501_metrics,
    pick_device,
    progress_write,
)


MARKET_RE = re.compile(
    r"^(?P<pid>-?\d{1,5})_c(?P<cam>\d)s(?P<seq>\d)_(?P<frame>\d{3,7})_(?P<idx>\d{2})\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


def parse_market_name(path: Path):
    match = MARKET_RE.fullmatch(path.name)
    if not match:
        return None
    groups = match.groupdict()
    return {k: int(groups[k]) for k in ("pid", "cam", "seq", "frame", "idx")}


class ImageListDataset(Dataset):
    def __init__(self, paths: Sequence[Path], transform) -> None:
        self.items: List[tuple[Path, dict]] = []
        self.transform = transform
        for path in paths:
            meta = parse_market_name(path)
            if meta is None:
                continue
            self.items.append((path, meta))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, meta = self.items[index]
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img) if self.transform else img
        return tensor, int(meta["pid"]), int(meta["cam"]), str(path)


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def extract_features(
    model: ReIDBaseline,
    loader: DataLoader,
    device: torch.device,
    *,
    args: argparse.Namespace,
    desc: str,
    position: int,
    use_amp: bool,
    autocast_dtype: torch.dtype | None,
    channels_last: bool,
):
    feats: List[torch.Tensor] = []
    pids: List[int] = []
    camids: List[int] = []
    paths: List[str] = []

    model.eval()
    refresh = max(1, getattr(args, "progress_refresh", 10))
    start_time = time.time()
    processed = 0

    feature_bar = make_tqdm(
        loader,
        desc=desc,
        position=position,
        leave=True,
        _args=args,
    )

    autocast_kwargs = {
        "device_type": device.type,
        "enabled": use_amp,
        "dtype": autocast_dtype,
    }

    with torch.no_grad():
        for idx, (images, pid, cam, path) in enumerate(feature_bar, 1):
            images = images.to(device, non_blocking=True)
            if channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(**autocast_kwargs):
                _, bn_feat, _ = model(images)
            norm_feat = F.normalize(bn_feat, dim=1)
            feats.append(norm_feat.cpu())
            pids.extend(int(x) for x in pid)
            camids.extend(int(x) for x in cam)
            paths.extend(path)
            processed += images.size(0)

            if idx % refresh == 0 and hasattr(feature_bar, "set_postfix"):
                elapsed = max(time.time() - start_time, 1e-6)
                feature_bar.set_postfix({"img/s": f"{processed / elapsed:.1f}"})

    features = torch.cat(feats, dim=0) if feats else torch.empty(0, device="cpu")
    return features, pids, camids, paths


def write_topk(
    out_csv: Path,
    distmat,
    query_paths: Sequence[str],
    gallery_paths: Sequence[str],
    query_pids: Sequence[int],
    gallery_pids: Sequence[int],
    query_camids: Sequence[int],
    gallery_camids: Sequence[int],
    topk: int = 10,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    gallery_pids_arr = np.asarray(gallery_pids)
    gallery_camids_arr = np.asarray(gallery_camids)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["q_path", "g_path", "dist", "pid_match", "cam_match"])
        indices = distmat.argsort(axis=1)
        for i, q_path in enumerate(query_paths):
            q_pid = query_pids[i]
            q_cam = query_camids[i]
            order = indices[i]
            mask = ~((gallery_pids_arr[order] == q_pid) & (gallery_camids_arr[order] == q_cam))
            order = order[mask][:topk]
            for j in order:
                g_path = gallery_paths[j]
                dist = float(distmat[i, j])
                pid_match = bool(gallery_pids_arr[j] == q_pid)
                cam_match = bool(gallery_camids_arr[j] == q_cam)
                writer.writerow([q_path, g_path, f"{dist:.6f}", int(pid_match), int(cam_match)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features and evaluate on Market-1501 test")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to trained checkpoint")
    parser.add_argument("--test_root", type=Path, required=True, help="Path to Market-1501 root")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory to store evaluation outputs")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto|cuda|cuda:0|mps|cpu (default: auto detects CUDA/MPS)",
    )
    parser.add_argument(
        "--precision",
        default="amp",
        choices=["amp", "fp32", "bf16"],
        help="Training precision (default: amp)",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help="Use channels_last memory format on GPU",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model if supported",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度条")
    parser.add_argument("--force_progress", action="store_true", help="即使非TTY也显示进度条")
    parser.add_argument("--progress_ascii", action="store_true", help="用ASCII渲染进度条（Windows/日志更稳）")
    parser.add_argument(
        "--progress_refresh",
        type=int,
        default=10,
        help="batch 间隔多少步刷新一次后缀信息",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    args.ckpt = args.ckpt.expanduser().resolve()
    args.test_root = args.test_root.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.test_root.exists():
        raise FileNotFoundError(f"Test root not found: {args.test_root}")

    device_arg = args.device or "auto"
    device = pick_device(device_arg)
    print(
        f"INFO torch: {torch.__version__} | cuda_available: {torch.cuda.is_available()} | torch.cuda: {getattr(torch.version, 'cuda', None)}"
    )
    print(f"INFO device: {device}")
    if device.type == "cuda":
        try:
            print(f"INFO cuda: {torch.cuda.get_device_name(device)}")
        except Exception:
            print("INFO cuda: <unknown device>")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif device_arg.lower().startswith("cuda") and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available.", file=sys.stderr)
        print("Hint: Install a CUDA-enabled PyTorch build, e.g.:", file=sys.stderr)
        print(
            "  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision",
            file=sys.stderr,
        )
        print(
            "Also verify NVIDIA driver + CUDA toolkit compatible with your PyTorch build.",
            file=sys.stderr,
        )
        return 2

    precision_mode = args.precision
    autocast_dtype: torch.dtype | None = None
    use_amp = False
    if precision_mode == "amp":
        use_amp = device.type in ("cuda", "mps")
    elif precision_mode == "bf16":
        if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
            use_amp = True
        else:
            progress_write("WARN: bf16 requested but not supported. Falling back to fp32.")
            precision_mode = "fp32"
    else:
        use_amp = False

    if device.type == "mps" and precision_mode == "amp":
        progress_write("INFO: Using MPS autocast mixed precision.")
    elif precision_mode == "amp" and not use_amp:
        progress_write("WARN: AMP requested but not supported on selected device. Using fp32.")

    if args.grad_accum_steps > 1:
        progress_write("INFO: --grad_accum_steps has no effect during evaluation")

    checkpoint = torch.load(args.ckpt, map_location=device)
    num_classes = int(checkpoint.get("num_classes", 0))
    if num_classes <= 0:
        raise ValueError("Checkpoint is missing 'num_classes'")

    model = ReIDBaseline(num_classes=num_classes)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            progress_write("INFO: model compiled with torch.compile")
        except Exception as exc:
            progress_write(f"WARN: torch.compile failed -> {exc}")

    transform = build_market1501_transforms(height=args.height, width=args.width, is_train=False)

    gallery_dir = args.test_root / "bounding_box_test"
    query_dir = args.test_root / "query"
    if not gallery_dir.is_dir() or not query_dir.is_dir():
        raise FileNotFoundError("Test root must contain 'bounding_box_test' and 'query' directories")

    gallery_paths = list_images(gallery_dir)
    query_paths = list_images(query_dir)

    gallery_dataset = ImageListDataset(gallery_paths, transform)
    query_dataset = ImageListDataset(query_paths, transform)

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    loader_kwargs = {
        "batch_size": args.batch,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    gallery_loader = DataLoader(
        gallery_dataset,
        **loader_kwargs,
    )
    query_loader = DataLoader(
        query_dataset,
        **loader_kwargs,
    )

    if args.num_workers > 0 and os.name == "nt":
        progress_write("Hint: Windows 下如遇 DataLoader 卡顿，可尝试 --num_workers 0")

    progress_write("Warming up dataloaders ...")
    for name, loader in (("gallery", gallery_loader), ("query", query_loader)):
        try:
            iterator = iter(loader)
            next(iterator)
            progress_write(f"{name.capitalize()} loader ready.")
        except StopIteration:
            progress_write(f"{name.capitalize()} loader is empty.")
        except Exception as exc:  # pragma: no cover - dataloader backend issues
            progress_write(f"Failed to warm up {name} loader: {exc}")
    progress_write("Dataloaders ready.")

    gallery_feats, gallery_pids, gallery_camids, gallery_strs = extract_features(
        model,
        gallery_loader,
        device,
        args=args,
        desc="Extract[gallery]",
        position=0,
        use_amp=use_amp,
        autocast_dtype=autocast_dtype,
        channels_last=args.channels_last,
    )
    query_feats, query_pids, query_camids, query_strs = extract_features(
        model,
        query_loader,
        device,
        args=args,
        desc="Extract[query]",
        position=0,
        use_amp=use_amp,
        autocast_dtype=autocast_dtype,
        channels_last=args.channels_last,
    )

    if gallery_feats.numel() == 0 or query_feats.numel() == 0:
        raise RuntimeError("No features extracted. Please ensure the dataset folders contain images.")

    distmat = cosine_distance_matrix(query_feats, gallery_feats)
    map_score, cmc_dict = market1501_metrics(
        distmat,
        query_pids,
        gallery_pids,
        query_camids,
        gallery_camids,
        progress_args=args,
        progress_position=1,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results.json").write_text(
        json.dumps({"mAP": map_score, "CMC": {f"R{k}": v for k, v in cmc_dict.items()}}, indent=2),
        encoding="utf-8",
    )
    write_topk(
        args.outdir / "topk.csv",
        distmat,
        query_strs,
        gallery_strs,
        query_pids,
        gallery_pids,
        query_camids,
        gallery_camids,
        topk=10,
    )

    progress_write("Evaluation complete")
    progress_write(f"mAP: {map_score:.4f}")
    progress_write("CMC:")
    for rank in sorted(cmc_dict):
        progress_write(f"  Rank-{rank}: {cmc_dict[rank]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

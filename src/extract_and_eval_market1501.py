"""Extract embeddings and evaluate on the official Market-1501 test split."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.market1501_clean import build_market1501_transforms
from train_reid_baseline import ReIDBaseline, cosine_distance_matrix, market1501_metrics


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


def extract_features(model: ReIDBaseline, loader: DataLoader, device: torch.device):
    feats: List[torch.Tensor] = []
    pids: List[int] = []
    camids: List[int] = []
    paths: List[str] = []

    model.eval()
    with torch.no_grad():
        for images, pid, cam, path in loader:
            images = images.to(device, non_blocking=True)
            _, bn_feat, _ = model(images)
            norm_feat = F.normalize(bn_feat, dim=1)
            feats.append(norm_feat.cpu())
            pids.extend(int(x) for x in pid)
            camids.extend(int(x) for x in cam)
            paths.extend(path)

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
    parser.add_argument("--device", type=str, default=None, help="Computation device (e.g., cuda, cpu)")
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

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt, map_location=device)
    num_classes = int(checkpoint.get("num_classes", 0))
    if num_classes <= 0:
        raise ValueError("Checkpoint is missing 'num_classes'")

    model = ReIDBaseline(num_classes=num_classes)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    transform = build_market1501_transforms(height=args.height, width=args.width, is_train=False)

    gallery_dir = args.test_root / "bounding_box_test"
    query_dir = args.test_root / "query"
    if not gallery_dir.is_dir() or not query_dir.is_dir():
        raise FileNotFoundError("Test root must contain 'bounding_box_test' and 'query' directories")

    gallery_paths = list_images(gallery_dir)
    query_paths = list_images(query_dir)

    gallery_dataset = ImageListDataset(gallery_paths, transform)
    query_dataset = ImageListDataset(query_paths, transform)

    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    gallery_feats, gallery_pids, gallery_camids, gallery_strs = extract_features(model, gallery_loader, device)
    query_feats, query_pids, query_camids, query_strs = extract_features(model, query_loader, device)

    if gallery_feats.numel() == 0 or query_feats.numel() == 0:
        raise RuntimeError("No features extracted. Please ensure the dataset folders contain images.")

    distmat = cosine_distance_matrix(query_feats, gallery_feats)
    map_score, cmc_dict = market1501_metrics(distmat, query_pids, gallery_pids, query_camids, gallery_camids)

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

    print("Evaluation complete")
    print(f"mAP: {map_score:.4f}")
    print("CMC:")
    for rank in sorted(cmc_dict):
        print(f"  Rank-{rank}: {cmc_dict[rank]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

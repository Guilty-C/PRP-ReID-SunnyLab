"""Train a supervised Market-1501 baseline on washed data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing
    tqdm = None


def make_tqdm(iterable, **kw):
    args = kw.pop("_args", None)
    if tqdm is None:
        if not getattr(make_tqdm, "_warned", False):
            print("tqdm is not installed. Run `pip install tqdm>=4.66` to enable progress bars.")
            make_tqdm._warned = True
        return iterable
    if getattr(args, "no_progress", False) and not getattr(args, "force_progress", False):
        return iterable
    disable = kw.pop("disable", False)
    if not sys.stdout.isatty() and not getattr(args, "force_progress", False):
        disable = True
    ascii_opt = getattr(args, "progress_ascii", False) or None
    return tqdm(iterable, disable=disable, ascii=ascii_opt, **kw)


make_tqdm._warned = False  # type: ignore[attr-defined]


def progress_write(message: str) -> None:
    if tqdm is not None:
        try:
            tqdm.write(message)
            return
        except Exception:  # pragma: no cover - tqdm edge case
            pass
    print(message)

from data.market1501_clean import (
    Market1501CleanDataset,
    PKSampler,
    build_market1501_transforms,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReIDBaseline(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = backbone
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_feat = self.backbone(x)
        if global_feat.ndim == 4:
            global_feat = torch.flatten(global_feat, 1)
        bn_feat = self.bnneck(global_feat)
        cls_logits = self.classifier(bn_feat)
        return global_feat, bn_feat, cls_logits


def hard_triplet_loss(features: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("Features must be 2-D (batch_size, dim)")
    if labels.ndim != 1:
        raise ValueError("Labels must be 1-D")

    dist_mat = torch.cdist(features, features, p=2)
    N = labels.size(0)
    mask = labels.expand(N, N).eq(labels.expand(N, N).t())
    dist_pos = dist_mat.clone()
    dist_pos[~mask] = -1.0
    dist_pos[range(N), range(N)] = -1.0
    hardest_pos, _ = dist_pos.max(dim=1)

    dist_neg = dist_mat.clone()
    dist_neg[mask] = float("inf")
    hardest_neg, _ = dist_neg.min(dim=1)

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()


def cosine_distance_matrix(query: torch.Tensor, gallery: torch.Tensor) -> np.ndarray:
    q = F.normalize(query, dim=1)
    g = F.normalize(gallery, dim=1)
    dist = 1.0 - torch.mm(q, g.t())
    return dist.cpu().numpy()


def market1501_metrics(
    distmat: np.ndarray,
    query_pids: Sequence[int],
    gallery_pids: Sequence[int],
    query_camids: Sequence[int],
    gallery_camids: Sequence[int],
    ranks: Tuple[int, ...] = (1, 5, 10),
    *,
    progress_args=None,
    progress_position: int | None = None,
) -> Tuple[float, Dict[int, float]]:
    q_pids = np.asarray(query_pids)
    g_pids = np.asarray(gallery_pids)
    q_camids = np.asarray(query_camids)
    g_camids = np.asarray(gallery_camids)

    num_q, num_g = distmat.shape
    if num_g == 0:
        return 0.0, {r: 0.0 for r in ranks}

    indices = np.argsort(distmat, axis=1)
    matches = g_pids[indices] == q_pids[:, np.newaxis]

    cmc = np.zeros(num_g)
    aps: List[float] = []
    num_valid_q = 0

    tqdm_kwargs = {"desc": "Match", "leave": False}
    if progress_position is not None:
        tqdm_kwargs["position"] = progress_position

    query_iter = make_tqdm(range(num_q), _args=progress_args, **tqdm_kwargs)

    for q_idx in query_iter:
        q_pid = q_pids[q_idx]
        q_cam = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
        keep = ~remove
        order = order[keep]
        if order.size == 0:
            continue
        match = matches[q_idx, keep]
        if not np.any(match):
            continue

        num_valid_q += 1
        first_index = np.nonzero(match)[0][0]
        cmc[first_index:] += 1

        precision = np.cumsum(match) / (np.arange(match.size) + 1)
        aps.append(float((precision * match).sum() / match.sum()))

    if num_valid_q == 0:
        return 0.0, {r: 0.0 for r in ranks}

    cmc = cmc / num_valid_q
    mAP = float(np.mean(aps)) if aps else 0.0
    cmc_dict = {r: float(cmc[r - 1]) if r - 1 < len(cmc) else float(cmc[-1]) for r in ranks}
    return mAP, cmc_dict


def build_dataloaders(
    clean_root: Path,
    manifest: Path,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    height: int,
    width: int,
    batch_size: int,
    pk_p: int,
    pk_k: int,
    seed: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Dict[int, int]]:
    train_transform = build_market1501_transforms(height=height, width=width, is_train=True)
    val_transform = build_market1501_transforms(height=height, width=width, is_train=False)

    train_dataset = Market1501CleanDataset(
        root=clean_root,
        manifest_csv=manifest,
        ids=train_ids,
        transform=train_transform,
        is_train=True,
    )
    val_dataset = Market1501CleanDataset(
        root=clean_root,
        manifest_csv=manifest,
        ids=val_ids,
        transform=val_transform,
        is_train=False,
    )

    pid_to_label = {pid: idx for idx, pid in enumerate(sorted(set(train_ids)))}

    train_sampler = PKSampler(train_dataset.labels, P=pk_p, K=pk_k, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, pid_to_label


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    args: argparse.Namespace | None = None,
    desc: str = "Extract",
    position: int | None = None,
    ce_criterion: nn.Module | None = None,
    triplet_margin: float | None = None,
    pid_to_label: Dict[int, int] | None = None,
) -> Tuple[torch.Tensor, List[int], List[int], List[str], TrainStats | None]:
    model.eval()
    feats: List[torch.Tensor] = []
    pids: List[int] = []
    camids: List[int] = []
    paths: List[str] = []

    compute_loss = ce_criterion is not None and triplet_margin is not None and pid_to_label is not None
    stats = TrainStats() if compute_loss else None
    refresh = max(1, getattr(args, "progress_refresh", 10)) if args is not None else 10
    start_time = time.time()
    processed = 0

    tqdm_kwargs = {"desc": desc, "leave": False}
    if position is not None:
        tqdm_kwargs["position"] = position

    feature_bar = make_tqdm(loader, _args=args, **tqdm_kwargs)

    with torch.no_grad():
        for idx, (images, pid, cam, path) in enumerate(feature_bar, 1):
            images = images.to(device, non_blocking=True)
            pid_dev = pid.to(device)
            global_feat, bn_feat, logits = model(images)
            norm_feat = F.normalize(bn_feat, dim=1)
            feats.append(norm_feat.cpu())
            pids.extend(int(x) for x in pid)
            camids.extend(int(x) for x in cam)
            paths.extend(path)
            processed += images.size(0)

            if compute_loss and stats is not None:
                triplet_feat = F.normalize(global_feat, dim=1)
                labels = (
                    [pid_to_label.get(int(x), -1) for x in pid.tolist()]
                    if pid_to_label
                    else [int(x) for x in pid.tolist()]
                )
                targets = torch.tensor(labels, device=device)
                ce_loss = ce_criterion(logits, targets)
                tri_loss = hard_triplet_loss(triplet_feat, pid_dev, margin=triplet_margin)
                total_loss = ce_loss + tri_loss
                stats.loss += total_loss.item() * images.size(0)
                stats.ce_loss += ce_loss.item() * images.size(0)
                stats.triplet_loss += tri_loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                stats.acc += (preds == targets).float().sum().item()
                stats.count += images.size(0)

            if idx % refresh == 0 and hasattr(feature_bar, "set_postfix"):
                elapsed = max(time.time() - start_time, 1e-6)
                postfix = {"img/s": f"{processed / elapsed:.1f}"}
                if compute_loss and stats is not None and stats.count:
                    postfix["loss"] = f"{stats.loss / stats.count:.4f}"
                feature_bar.set_postfix(postfix)

    features = torch.cat(feats, dim=0) if feats else torch.empty(0, device="cpu")
    if stats is not None and stats.count:
        stats.loss /= stats.count
        stats.ce_loss /= stats.count
        stats.triplet_loss /= stats.count
        stats.acc /= stats.count
    return features, pids, camids, paths, stats


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    args: argparse.Namespace,
    ce_criterion: nn.Module,
    triplet_margin: float,
    pid_to_label: Dict[int, int] | None = None,
) -> Tuple[float, Dict[int, float], TrainStats | None]:
    progress_write("INFO eval: using raw PIDs (no mapping).")
    feats, pids, camids, _, stats = extract_features(
        model,
        loader,
        device,
        args=args,
        desc="Val",
        position=1,
        ce_criterion=ce_criterion,
        triplet_margin=triplet_margin,
        pid_to_label=None,
    )
    if feats.numel() == 0:
        return 0.0, {1: 0.0, 5: 0.0, 10: 0.0}, stats
    distmat = cosine_distance_matrix(feats, feats)
    metrics = market1501_metrics(
        distmat,
        pids,
        pids,
        camids,
        camids,
        progress_args=args,
        progress_position=2,
    )
    return metrics[0], metrics[1], stats


def build_scheduler(optimizer: AdamW, epochs: int, warmup_epochs: int) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@dataclass
class TrainStats:
    loss: float = 0.0
    ce_loss: float = 0.0
    triplet_loss: float = 0.0
    acc: float = 0.0
    count: int = 0


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    ce_criterion: nn.Module,
    triplet_margin: float,
    pid_to_label: Dict[int, int],
    print_freq: int,
    epoch: int,
    *,
    args: argparse.Namespace,
) -> TrainStats:
    model.train()
    stats = TrainStats()
    refresh = max(1, getattr(args, "progress_refresh", 10))
    start_time = time.time()
    processed = 0
    loader_len = len(loader)
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:  # pragma: no cover - CPU or unsupported devices
            pass

    train_bar = make_tqdm(
        loader,
        desc="Train",
        position=1,
        leave=False,
        _args=args,
    )

    _autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for iteration, (images, pids, _camids, _paths) in enumerate(train_bar, 1):
        images = images.to(device, non_blocking=True)
        pids = pids.to(device)
        targets = torch.tensor([pid_to_label[int(pid)] for pid in pids.tolist()], device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(_autocast_device, enabled=scaler.is_enabled()):
            global_feat, bn_feat, logits = model(images)
            norm_feat = F.normalize(global_feat, dim=1)
            ce_loss = ce_criterion(logits, targets)
            tri_loss = hard_triplet_loss(norm_feat, pids, margin=triplet_margin)
            loss = ce_loss + tri_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            stats.loss += loss.item() * images.size(0)
            stats.ce_loss += ce_loss.item() * images.size(0)
            stats.triplet_loss += tri_loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            stats.acc += (preds == targets).float().sum().item()
            stats.count += images.size(0)
            processed += images.size(0)

        if iteration % refresh == 0:
            avg_loss = stats.loss / max(1, stats.count)
            avg_ce = stats.ce_loss / max(1, stats.count)
            avg_tri = stats.triplet_loss / max(1, stats.count)
            avg_acc = stats.acc / max(1, stats.count)
            if hasattr(train_bar, "set_postfix"):
                lr = optimizer.param_groups[0]["lr"]
                elapsed = max(time.time() - start_time, 1e-6)
                imgs_per_s = processed / elapsed
                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "ce": f"{avg_ce:.4f}",
                    "tri": f"{avg_tri:.4f}",
                    "acc": f"{avg_acc:.3f}",
                    "lr": f"{lr:.2e}",
                    "img/s": f"{imgs_per_s:.1f}",
                }
                if torch.cuda.is_available():
                    try:
                        mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                        postfix["mem"] = f"{mem:.0f}MB"
                    except Exception:  # pragma: no cover - CPU fallback
                        pass
                train_bar.set_postfix(postfix)
            elif (iteration % max(1, print_freq) == 0) or (iteration == loader_len):
                progress_write(
                    "Epoch {epoch} Iter {iter_idx}/{total} | Loss: {loss:.4f} (CE: {ce:.4f}, Tri: {tri:.4f}) | Acc: {acc:.3f}".format(
                        epoch=epoch,
                        iter_idx=iteration,
                        total=loader_len,
                        loss=avg_loss,
                        ce=avg_ce,
                        tri=avg_tri,
                        acc=avg_acc,
                    )
                )

    stats.loss /= max(1, stats.count)
    stats.ce_loss /= max(1, stats.count)
    stats.triplet_loss /= max(1, stats.count)
    stats.acc /= max(1, stats.count)
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet-50 BNNeck baseline on Market-1501")
    parser.add_argument("--clean_root", type=Path, required=True, help="Path to cleaned Market-1501 root")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to train_manifest.csv")
    parser.add_argument("--splits", type=Path, required=True, help="Path to splits.json")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory to store checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=64, help="Batch size (should match P*K)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pk_p", type=int, default=16)
    parser.add_argument("--pk_k", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_freq", type=int, default=20, help="Logging frequency when progress bars are disabled")
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

    args.clean_root = args.clean_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.splits = args.splits.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()

    if args.batch != args.pk_p * args.pk_k:
        progress_write(
            f"Warning: --batch ({args.batch}) does not equal --pk_p * --pk_k ({args.pk_p * args.pk_k}). Using PK sampler batch size."
        )

    set_seed(args.seed)

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    train_ids = [int(x) for x in splits.get("train_ids", [])]
    val_ids = [int(x) for x in splits.get("val_ids", [])]
    if not train_ids:
        raise ValueError("No train IDs found in splits.json")
    if not val_ids:
        raise ValueError("No val IDs found in splits.json")

    args.outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, pid_to_label = build_dataloaders(
        clean_root=args.clean_root,
        manifest=args.manifest,
        train_ids=train_ids,
        val_ids=val_ids,
        height=args.height,
        width=args.width,
        batch_size=args.batch,
        pk_p=args.pk_p,
        pk_k=args.pk_k,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    if args.num_workers > 0 and os.name == "nt":
        progress_write("Hint: Windows 下如遇 DataLoader 卡顿，可尝试 --num_workers 0")

    progress_write("Warming up first batch ...")
    for loader_name, loader in (("train", train_loader), ("val", val_loader)):
        try:
            iterator = iter(loader)
            next(iterator)
            progress_write(f"{loader_name.capitalize()} loader ready.")
        except StopIteration:
            progress_write(f"{loader_name.capitalize()} loader is empty.")
        except Exception as exc:  # pragma: no cover - dataloader backend issues
            progress_write(f"Failed to warm up {loader_name} loader: {exc}")
    progress_write("Dataloaders ready.")

    model = ReIDBaseline(num_classes=len(pid_to_label))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=args.epochs, warmup_epochs=10)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_margin = 0.3

    best_map = -1.0
    best_epoch = -1

    epoch_bar = make_tqdm(
        range(1, args.epochs + 1),
        desc="Epoch",
        position=0,
        leave=True,
        _args=args,
    )

    for epoch in epoch_bar:
        train_sampler = train_loader.batch_sampler
        if isinstance(train_sampler, PKSampler):
            train_sampler.set_epoch(epoch)

        stats = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            ce_criterion,
            triplet_margin=triplet_margin,
            pid_to_label=pid_to_label,
            print_freq=args.print_freq,
            epoch=epoch,
            args=args,
        )

        val_map, val_cmc, val_stats = evaluate_split(
            model,
            val_loader,
            device,
            args=args,
            ce_criterion=ce_criterion,
            triplet_margin=triplet_margin,
            pid_to_label=None,
        )
        scheduler.step()

        summary = (
            f"Epoch {epoch:03d}/{args.epochs} | Loss {stats.loss:.4f} | "
            f"CE {stats.ce_loss:.4f} | Triplet {stats.triplet_loss:.4f} | Acc {stats.acc:.3f}"
        )
        progress_write(summary)
        if val_stats is not None:
            progress_write(
                f"Val Loss {val_stats.loss:.4f} | CE {val_stats.ce_loss:.4f} | Triplet {val_stats.triplet_loss:.4f} | Acc {val_stats.acc:.3f}"
            )
        progress_write(
            f"Validation mAP: {val_map:.4f} | "
            f"CMC@1: {val_cmc.get(1, 0.0):.4f} | CMC@5: {val_cmc.get(5, 0.0):.4f} | CMC@10: {val_cmc.get(10, 0.0):.4f}"
        )

        if hasattr(epoch_bar, "set_postfix"):
            postfix = {"train_loss": f"{stats.loss:.4f}", "val_mAP": f"{val_map:.4f}"}
            if val_stats is not None:
                postfix["val_loss"] = f"{val_stats.loss:.4f}"
            epoch_bar.set_postfix(postfix)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "num_classes": len(pid_to_label),
            "pid_to_label": pid_to_label,
            "val_map": val_map,
        }
        torch.save(ckpt, args.outdir / "last.pth")

        if val_map > best_map:
            best_map = val_map
            best_epoch = epoch
            torch.save(ckpt, args.outdir / "best.pth")
            progress_write(f"New best model at epoch {epoch} with mAP {val_map:.4f}")

    progress_write(f"Training complete. Best mAP {best_map:.4f} at epoch {best_epoch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

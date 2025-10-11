"""Train a supervised Market-1501 baseline on washed data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50


def pick_device(dev_arg: str) -> torch.device:
    dev_arg = (dev_arg or "auto").lower()
    if dev_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if dev_arg.startswith("cuda"):
        return torch.device(dev_arg if torch.cuda.is_available() else "cpu")
    if dev_arg == "mps":
        has_mps = getattr(torch.backends, "mps", None)
        return torch.device("mps" if has_mps and torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")

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


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in sd.items():
        if key.startswith("module."):
            out[key[len("module."):]] = value
        else:
            out[key] = value
    return out


def load_checkpoint_finetune(
    model: nn.Module,
    ckpt_path: os.PathLike[str] | str,
    *,
    strict: bool = False,
    skip_keys_prefix: Tuple[str, ...] = ("classifier.", "bnneck."),
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format from {ckpt_path!r}")
    state_dict = _strip_module_prefix(state_dict)

    model_sd = model.state_dict()
    loadable: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    skipped: List[Tuple[str, str]] = []
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in skip_keys_prefix):
            skipped.append((key, "prefix-skip"))
            continue
        if key in model_sd and model_sd[key].shape == value.shape:
            loadable[key] = value
        else:
            target_tensor = model_sd.get(key)
            target_shape = tuple(target_tensor.shape) if target_tensor is not None else ()
            skipped.append((key, f"shape {tuple(value.shape)} != {target_shape}"))

    progress_write(
        f"INFO resume: Loaded {len(loadable)}/{len(model_sd)} params from {ckpt_path}."
    )
    if skipped:
        preview = [f"{k} ({reason})" for k, reason in skipped[:8]]
        more = " ..." if len(skipped) > 8 else ""
        progress_write(f"INFO resume: skipped keys = {preview}{more}")

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(loadable, strict=False)

    return ckpt if isinstance(ckpt, dict) else {"state_dict": ckpt}


def set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("classifier.") or name.startswith("bnneck."):
            continue
        param.requires_grad = requires_grad


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
    *,
    device: torch.device,
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
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
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
    use_amp: bool = False,
    autocast_dtype: torch.dtype | None = None,
    channels_last: bool = False,
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
            pid_dev = pid.to(device, non_blocking=True)
            with torch.amp.autocast(**autocast_kwargs):
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
                targets = torch.as_tensor(labels, device=device)
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
    use_amp: bool = False,
    autocast_dtype: torch.dtype | None = None,
    channels_last: bool = False,
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
        use_amp=use_amp,
        autocast_dtype=autocast_dtype,
        channels_last=channels_last,
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
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    ce_criterion: nn.Module,
    triplet_margin: float,
    pid_to_label: Dict[int, int],
    print_freq: int,
    epoch: int,
    *,
    args: argparse.Namespace,
    use_amp: bool,
    autocast_dtype: torch.dtype | None,
    grad_accum_steps: int,
    channels_last: bool,
) -> TrainStats:
    model.train()
    stats = TrainStats()
    refresh = max(1, getattr(args, "progress_refresh", 10))
    start_time = time.time()
    processed = 0
    loader_len = len(loader)
    if device.type == "cuda":
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

    optimizer.zero_grad(set_to_none=True)

    for iteration, (images, pids, _camids, _paths) in enumerate(train_bar, 1):
        batch_size = images.size(0)
        try:
            images = images.to(device, non_blocking=True)
            if channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            pids = pids.to(device, non_blocking=True)
            pid_list = pids.detach().cpu().tolist()
            targets = torch.as_tensor(
                [pid_to_label[int(pid)] for pid in pid_list], device=device
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=autocast_dtype,
            ):
                global_feat, bn_feat, logits = model(images)
                norm_feat = F.normalize(global_feat, dim=1)
                ce_loss = ce_criterion(logits, targets)
                tri_loss = hard_triplet_loss(norm_feat, pids, margin=triplet_margin)
                total_loss = ce_loss + tri_loss

            loss = total_loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (iteration % grad_accum_steps == 0) or (iteration == loader_len)
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                stats.loss += total_loss.item() * batch_size
                stats.ce_loss += ce_loss.item() * batch_size
                stats.triplet_loss += tri_loss.item() * batch_size
                preds = logits.argmax(dim=1)
                stats.acc += (preds == targets).float().sum().item()
                stats.count += batch_size
                processed += batch_size
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress_write(
                "WARN: CUDA OOM; consider reducing batch size or increase --grad_accum_steps"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

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
                if device.type == "cuda":
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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for finetune/resume")
    parser.add_argument(
        "--resume_strict",
        action="store_true",
        help="Strict load all layers (default: non-strict)",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Freeze backbone for first N epochs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    args.clean_root = args.clean_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.splits = args.splits.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    if args.resume:
        args.resume = Path(args.resume).expanduser().resolve()

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
        print(
            "Hint: Install a CUDA-enabled PyTorch build, e.g.:",
            file=sys.stderr,
        )
        print(
            "  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision",
            file=sys.stderr,
        )
        print(
            "Also verify NVIDIA driver + CUDA toolkit compatible with your PyTorch build.",
            file=sys.stderr,
        )
        return 2

    grad_accum_steps = max(1, args.grad_accum_steps)
    if grad_accum_steps != args.grad_accum_steps:
        progress_write("INFO: --grad_accum_steps < 1. Using 1 instead.")

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

    scaler: torch.amp.GradScaler | None = None
    if precision_mode == "amp" and device.type == "cuda" and use_amp:
        try:
            scaler = torch.amp.GradScaler(device.type)
        except Exception:
            scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

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
        device=device,
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
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            progress_write("INFO: model compiled with torch.compile")
        except Exception as exc:
            progress_write(f"WARN: torch.compile failed -> {exc}")

    resume_ckpt: Dict[str, Any] | None = None
    start_epoch = 0
    if args.resume:
        resume_ckpt = load_checkpoint_finetune(
            model,
            args.resume,
            strict=args.resume_strict,
            skip_keys_prefix=("classifier.", "bnneck."),
        )
        start_epoch = int(resume_ckpt.get("epoch", 0))
        progress_write(f"INFO resume: starting from epoch {start_epoch}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=args.epochs, warmup_epochs=10)

    if resume_ckpt is not None:
        opt_state = resume_ckpt.get("optimizer")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
                progress_write("INFO resume: optimizer state loaded")
            except Exception as exc:  # pragma: no cover - state mismatch
                progress_write(f"WARN resume: optimizer state not loaded: {exc}")
        sch_state = resume_ckpt.get("scheduler")
        if sch_state is not None and scheduler is not None:
            try:
                scheduler.load_state_dict(sch_state)
                progress_write("INFO resume: scheduler state loaded")
            except Exception as exc:  # pragma: no cover - state mismatch
                progress_write(f"WARN resume: scheduler state not loaded: {exc}")
        scaler_state = resume_ckpt.get("scaler")
        if scaler_state is not None and scaler is not None:
            try:
                scaler.load_state_dict(scaler_state)
                progress_write("INFO resume: scaler state loaded")
            except Exception as exc:  # pragma: no cover - scaler mismatch
                progress_write(f"WARN resume: scaler state not loaded: {exc}")
        elif scaler_state is not None and scaler is None:
            progress_write("WARN resume: scaler state present but AMP disabled; skipping load")

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_margin = 0.3

    if resume_ckpt is not None:
        best_map = float(resume_ckpt.get("val_map", -1.0))
        best_epoch = int(resume_ckpt.get("best_epoch", resume_ckpt.get("epoch", -1)))
    else:
        best_map = -1.0
        best_epoch = -1

    epoch_iter = range(start_epoch + 1, args.epochs + 1)
    epoch_bar = make_tqdm(
        epoch_iter,
        desc="Epoch",
        position=0,
        leave=True,
        _args=args,
        total=args.epochs,
        initial=start_epoch,
    )

    backbone_frozen: bool | None = None

    for epoch in epoch_bar:
        freeze_active = args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        set_backbone_requires_grad(model, not freeze_active)
        if freeze_active and backbone_frozen is not True:
            progress_write(
                f"INFO finetune: freezing backbone for first {args.freeze_backbone_epochs} epochs"
            )
            backbone_frozen = True
        elif not freeze_active and backbone_frozen is True:
            progress_write("INFO finetune: unfreezing backbone parameters")
            backbone_frozen = False
        elif backbone_frozen is None:
            backbone_frozen = freeze_active

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
            use_amp=use_amp,
            autocast_dtype=autocast_dtype,
            grad_accum_steps=grad_accum_steps,
            channels_last=args.channels_last,
        )

        val_map, val_cmc, val_stats = evaluate_split(
            model,
            val_loader,
            device,
            args=args,
            ce_criterion=ce_criterion,
            triplet_margin=triplet_margin,
            pid_to_label=None,
            use_amp=use_amp,
            autocast_dtype=autocast_dtype,
            channels_last=args.channels_last,
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
            "scaler": scaler.state_dict() if scaler is not None else None,
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

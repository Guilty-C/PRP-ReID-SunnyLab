"""CLI tool to clean the Market-1501 training split.

The script indexes images, applies quality filters, removes near duplicates,
and writes cleaned subsets for full classification training and a balanced core set.
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image


LOGGER = logging.getLogger("market1501_wash")
FILENAME_PATTERN = re.compile(r"([\-\d]{1,5})_c(\d)s\d+_.*\\.jpg", re.IGNORECASE)
RNG = random.Random(42)


@dataclass(frozen=True)
class ImageRecord:
    """Container for metadata collected for each image."""

    path: Path
    rel_path: str
    pid: str
    pid_int: int
    camid: int
    height: int
    width: int
    blur_score: float
    phash: imagehash.ImageHash


@dataclass
class WashStats:
    """Counters for quality-control statistics."""

    rejected_blur: int = 0
    rejected_tiny: int = 0
    rejected_dupes: int = 0


def configure_logging() -> None:
    """Configure logging to emit structured messages to stdout."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s event=%(message)s",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Wash Market-1501 training images.")
    parser.add_argument("--src", required=True, type=Path, help="Path to Market-1501 root")
    parser.add_argument("--dst", required=True, type=Path, help="Path to output directory")
    parser.add_argument("--min_side", type=int, default=32, help="Minimum side length to keep")
    parser.add_argument("--blur_th", type=float, default=80.0, help="Variance of Laplacian threshold")
    parser.add_argument(
        "--dup_hamming",
        type=int,
        default=6,
        help="Maximum Hamming distance for duplicates within a pid/cam pair",
    )
    parser.add_argument(
        "--core_per_id",
        type=int,
        default=6,
        help="Number of images to sample per identity for the core subset",
    )
    parser.add_argument(
        "--min_imgs_per_id",
        type=int,
        default=4,
        help="Minimum number of images per identity after filtering",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and report dataset statistics without writing outputs",
    )
    return parser.parse_args(argv)


def validate_source(root: Path) -> Path:
    """Validate source directory structure and return training directory."""
    root = Path(root).expanduser().resolve()
    candidates = [
        root / "bounding_box_train",
        root / "Market-1501" / "bounding_box_train",
        root / "Market-1501-v15.09.15" / "bounding_box_train",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    search_patterns = [
        "bounding_box_train",
        "*/bounding_box_train",
        "*/*/bounding_box_train",
    ]
    for pattern in search_patterns:
        for found in root.glob(pattern):
            if found.is_dir():
                return found.resolve()

    tried_paths = "\n  - " + "\n  - ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(
        "Could not locate 'bounding_box_train' under source root.\n"
        f"Source root checked: {root.as_posix()}\n"
        f"Tried:{tried_paths}\n\n"
        "Tips:\n"
        "  • Ensure --src points to the Market-1501 root (the folder that contains 'bounding_box_train').\n"
        "  • See tools/market1501_wash/README.md for examples on Windows/macOS/Linux."
    )


def parse_metadata(image_path: Path) -> Tuple[str, int]:
    """Extract person id and camera id from a Market-1501 filename."""

    match = FILENAME_PATTERN.match(image_path.name)
    if not match:
        raise ValueError(f"Filename does not match Market-1501 pattern: {image_path.name}")
    pid_str, camid_str = match.groups()
    pid_int = int(pid_str)
    camid = int(camid_str)
    if pid_int <= 0:
        raise ValueError(f"Invalid pid {pid_int} for file {image_path.name}")
    return pid_str, camid


def compute_image_metrics(image_path: Path) -> Tuple[int, int, float, imagehash.ImageHash]:
    """Load an image and compute dimensions, blur score, and perceptual hash."""

    with Image.open(image_path) as pil_img:
        rgb_img = pil_img.convert("RGB")
        width, height = rgb_img.width, rgb_img.height
        gray_array = np.array(rgb_img.convert("L"))
        laplacian = cv2.Laplacian(gray_array, cv2.CV_64F)
        blur_score = float(laplacian.var())
        phash = imagehash.phash(rgb_img)
    return height, width, blur_score, phash


def index_dataset(train_dir: Path, dst_dir: Optional[Path]) -> List[ImageRecord]:
    """Index images, run basic filters, and write metadata CSV."""

    records: List[ImageRecord] = []
    writer: Optional[csv.writer] = None
    csv_file = None
    if dst_dir is not None:
        csv_path = dst_dir / "market1501_index.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["path", "pid", "camid", "h", "w"])

    image_paths = sorted(train_dir.glob("*.jpg"))
    LOGGER.info("phase=index total_files=%d", len(image_paths))
    try:
        for img_path in image_paths:
            try:
                pid_str, camid = parse_metadata(img_path)
                pid_int = int(pid_str)
                height, width, blur_score, phash = compute_image_metrics(img_path)
            except (ValueError, OSError, IOError) as exc:
                LOGGER.warning("event=skip reason=invalid_file path=%s error=%s", img_path, exc)
                continue

            rel_path = str(img_path.relative_to(train_dir.parent))
            if writer is not None:
                writer.writerow([rel_path, pid_str, camid, height, width])
            record = ImageRecord(
                path=img_path,
                rel_path=rel_path,
                pid=pid_str,
                pid_int=pid_int,
                camid=camid,
                height=height,
                width=width,
                blur_score=blur_score,
                phash=phash,
            )
            records.append(record)
    finally:
        if csv_file is not None:
            csv_file.close()
    LOGGER.info("phase=index indexed=%d", len(records))
    return records


def apply_quality_filters(
    records: Sequence[ImageRecord], cfg: argparse.Namespace, stats: WashStats
) -> List[ImageRecord]:
    """Apply blur and size filters to the indexed images."""

    kept: List[ImageRecord] = []
    for record in records:
        if record.blur_score < cfg.blur_th:
            stats.rejected_blur += 1
            continue
        if min(record.height, record.width) < cfg.min_side:
            stats.rejected_tiny += 1
            continue
        kept.append(record)
    LOGGER.info(
        "phase=quality total=%d kept=%d rejected_blur=%d rejected_tiny=%d",
        len(records),
        len(kept),
        stats.rejected_blur,
        stats.rejected_tiny,
    )
    return kept


def deduplicate(records: Sequence[ImageRecord], cfg: argparse.Namespace, stats: WashStats) -> List[ImageRecord]:
    """Remove near-duplicate images per (pid, camid)."""

    grouped: Dict[Tuple[str, int], List[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.pid, record.camid)].append(record)

    deduped: List[ImageRecord] = []
    for (pid, camid), group in grouped.items():
        group_sorted = sorted(group, key=lambda rec: rec.path.name)
        kept_group: List[ImageRecord] = []
        for record in group_sorted:
            is_duplicate = False
            for kept_record in kept_group:
                if (record.phash - kept_record.phash) <= cfg.dup_hamming:
                    stats.rejected_dupes += 1
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_group.append(record)
        deduped.extend(kept_group)
        LOGGER.info(
            "phase=dedup pid=%s camid=%d before=%d after=%d",
            pid,
            camid,
            len(group_sorted),
            len(kept_group),
        )
    LOGGER.info(
        "phase=dedup total=%d kept=%d rejected_dupes=%d",
        len(records),
        len(deduped),
        stats.rejected_dupes,
    )
    return deduped


def filter_identities(records: Sequence[ImageRecord], min_imgs_per_id: int) -> Dict[str, List[ImageRecord]]:
    """Group by person id and drop identities with too few images."""

    grouped: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.pid].append(record)

    filtered: Dict[str, List[ImageRecord]] = {}
    for pid, items in grouped.items():
        if len(items) >= min_imgs_per_id:
            filtered[pid] = sorted(items, key=lambda rec: (rec.camid, rec.path.name))
        else:
            LOGGER.info("phase=filter_ids pid=%s reason=too_few_images count=%d", pid, len(items))
    LOGGER.info(
        "phase=filter_ids total_ids=%d kept_ids=%d min_imgs=%d",
        len(grouped),
        len(filtered),
        min_imgs_per_id,
    )
    return filtered


def copy_images(records: Iterable[ImageRecord], dst_dir: Path) -> None:
    """Copy images into the specified destination directory."""

    for record in records:
        target_dir = dst_dir / record.pid
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / record.path.name
        shutil.copy2(record.path, target_path)


def select_core_subset(records: Sequence[ImageRecord], core_per_id: int) -> List[ImageRecord]:
    """Select a camera-balanced subset of records for one identity."""

    if core_per_id <= 0:
        return []

    cam_groups: Dict[int, List[ImageRecord]] = defaultdict(list)
    for record in records:
        cam_groups[record.camid].append(record)
    for group in cam_groups.values():
        group.sort(key=lambda rec: rec.path.name)

    selected: List[ImageRecord] = []
    camids = sorted(cam_groups.keys())
    while len(selected) < core_per_id and any(cam_groups[cam] for cam in camids):
        progress = False
        for cam in camids:
            if not cam_groups[cam]:
                continue
            selected.append(cam_groups[cam].pop(0))
            progress = True
            if len(selected) >= core_per_id:
                break
        if not progress:
            break

    if len(selected) >= core_per_id:
        return selected[:core_per_id]

    remaining = [rec for rec in sorted(records, key=lambda rec: rec.path.name) if rec not in selected]
    RNG.shuffle(remaining)
    for record in remaining:
        if len(selected) >= core_per_id:
            break
        selected.append(record)
    return selected


def write_stats_file(stats_path: Path, stats: WashStats, kept_ids: int, kept_images: int) -> None:
    """Write washing statistics to a text file."""

    with stats_path.open("w", encoding="utf-8") as handle:
        handle.write(f"IDs kept: {kept_ids}\n")
        handle.write(f"Images kept (full_clean): {kept_images}\n")
        handle.write(f"Rejected blur: {stats.rejected_blur}\n")
        handle.write(f"Rejected tiny: {stats.rejected_tiny}\n")
        handle.write(f"Rejected dupes: {stats.rejected_dupes}\n")


def run(cfg: argparse.Namespace) -> None:
    """Main execution pipeline."""

    configure_logging()
    src = Path(cfg.src).expanduser().resolve()
    dst = Path(cfg.dst).expanduser().resolve()
    dry_run = bool(getattr(cfg, "dry_run", False))
    LOGGER.info(
        "phase=start src=%s dst=%s min_side=%d blur_th=%.2f dup_hamming=%d core_per_id=%d min_imgs_per_id=%d dry_run=%s",
        src.as_posix(),
        dst.as_posix(),
        cfg.min_side,
        cfg.blur_th,
        cfg.dup_hamming,
        cfg.core_per_id,
        cfg.min_imgs_per_id,
        dry_run,
    )

    train_dir = validate_source(src)
    LOGGER.info("phase=validate train_dir=%s", train_dir.as_posix())
    dst.mkdir(parents=True, exist_ok=True)

    stats = WashStats()

    indexed_records = index_dataset(train_dir, None if dry_run else dst)
    filtered_records = apply_quality_filters(indexed_records, cfg, stats)
    deduped_records = deduplicate(filtered_records, cfg, stats)
    grouped_records = filter_identities(deduped_records, cfg.min_imgs_per_id)

    if dry_run:
        LOGGER.info(
            "phase=dry_run_result ids=%d images=%d",
            len(grouped_records),
            sum(len(records) for records in grouped_records.values()),
        )
        return

    full_clean_dir = dst / "train_full_clean"
    core_dir = dst / "train_core"
    full_clean_dir.mkdir(parents=True, exist_ok=True)
    core_dir.mkdir(parents=True, exist_ok=True)

    total_kept_images = 0
    for pid, records in grouped_records.items():
        copy_images(records, full_clean_dir)
        core_subset = select_core_subset(records, cfg.core_per_id)
        copy_images(core_subset, core_dir)
        total_kept_images += len(records)
        LOGGER.info(
            "phase=copy pid=%s full_clean=%d core=%d",
            pid,
            len(records),
            len(core_subset),
        )

    stats_path = dst / "wash_stats.txt"
    write_stats_file(stats_path, stats, len(grouped_records), total_kept_images)
    LOGGER.info(
        "phase=done ids=%d images=%d stats_path=%s",
        len(grouped_records),
        total_kept_images,
        stats_path.as_posix(),
    )


if __name__ == "__main__":
    run(parse_args())

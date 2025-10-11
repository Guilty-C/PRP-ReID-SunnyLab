"""Export manifests for washed Market-1501 data."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Iterable, List, Tuple

_PATTERN = re.compile(
    r"^(?P<pid>-?\d{1,5})_c(?P<cam>\d)s(?P<seq>\d)_(?P<frame>\d{3,7})_(?P<idx>\d{2})\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export manifest CSV for washed Market-1501 data")
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing the cleaned Market-1501 images",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Destination CSV path (default: <root>/train_manifest.csv)",
    )
    parser.add_argument(
        "--splits_json",
        type=Path,
        default=None,
        help="Destination JSON path for ID splits (default: <root>/splits.json)",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Store paths relative to the root directory",
    )
    parser.add_argument(
        "--posix_paths",
        action="store_true",
        help="Use POSIX style (forward slash) paths in the manifest",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of person IDs to reserve for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split (default: 42)",
    )
    return parser.parse_args()


def iter_image_paths(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            suffix = Path(name).suffix.lower()
            if suffix in SUPPORTED_EXTS:
                yield Path(dirpath) / name


def format_path(path: Path, root: Path, use_relative: bool, use_posix: bool) -> str:
    final_path = path
    if use_relative:
        final_path = path.relative_to(root)
    if use_posix:
        return final_path.as_posix()
    return str(final_path)


def build_rows(
    root: Path,
    use_relative: bool,
    use_posix: bool,
) -> Tuple[List[List[str]], int, List[int]]:
    rows: List[List[str]] = []
    bad_names = 0
    pids: List[int] = []
    for path in iter_image_paths(root):
        match = _PATTERN.match(path.name)
        if not match:
            bad_names += 1
            continue
        pid = int(match.group("pid"))
        cam = int(match.group("cam"))
        seq = int(match.group("seq"))
        frame = int(match.group("frame"))
        idx = int(match.group("idx"))

        path_str = format_path(path, root, use_relative, use_posix)
        rows.append([path_str, str(pid), str(cam), str(seq), str(frame), str(idx)])
        pids.append(pid)
    return rows, bad_names, pids


def write_manifest(csv_path: Path, rows: List[List[str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "pid", "cam", "seq", "frame", "idx"])
        writer.writerows(rows)


def build_splits(pids: Iterable[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    unique_ids = sorted({pid for pid in pids if pid != -1})
    if not unique_ids:
        return [], []
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError("--val_ratio must be between 0 and 1")
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    val_count = int(round(len(unique_ids) * val_ratio))
    if val_ratio > 0 and val_count == 0:
        val_count = 1
    val_ids = sorted(unique_ids[:val_count])
    train_ids = sorted(unique_ids[val_count:])
    return train_ids, val_ids


def write_splits(json_path: Path, train_ids: List[int], val_ids: List[int]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")
    csv_path = (args.output_csv or (root / "train_manifest.csv")).resolve()
    splits_path = (args.splits_json or (root / "splits.json")).resolve()

    rows, bad_names, pids = build_rows(root, args.relative, args.posix_paths)
    rows.sort(key=lambda row: row[0])
    write_manifest(csv_path, rows)

    train_ids, val_ids = build_splits(pids, args.val_ratio, args.seed)
    write_splits(splits_path, train_ids, val_ids)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.info("manifest: %s rows=%d bad_names_skipped=%d", csv_path, len(rows), bad_names)
    logging.info(
        "splits: %s train_ids=%d val_ids=%d seed=%d val_ratio=%s",
        splits_path,
        len(train_ids),
        len(val_ids),
        args.seed,
        args.val_ratio,
    )


if __name__ == "__main__":
    main()

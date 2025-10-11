"""Export manifest and train/val splits for cleaned Market-1501 data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


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


def iter_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Market-1501 cleaned manifest and splits")
    parser.add_argument("--root", required=True, type=Path, help="Path to cleaned Market-1501 root")
    parser.add_argument("--out_csv", required=True, type=Path, help="Path to output manifest CSV")
    parser.add_argument("--out_splits", required=True, type=Path, help="Path to output splits JSON")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio by identity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling IDs")
    return parser


def write_manifest(rows: Sequence[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "pid", "cam", "seq", "frame", "idx"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_ids(ids: Sequence[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    ids_list = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids_list)
    if not ids_list:
        return [], []

    val_count = int(round(len(ids_list) * val_ratio)) if val_ratio > 0 else 0
    if val_ratio > 0 and val_count == 0:
        val_count = 1
    if val_count >= len(ids_list):
        val_count = max(len(ids_list) - 1, 0)
    train_ids = ids_list[:-val_count] if val_count else ids_list
    val_ids = ids_list[-val_count:] if val_count else []
    return train_ids, val_ids


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    if not root.exists():
        parser.error(f"Root directory does not exist: {root}")

    records: List[dict] = []
    for img_path in iter_images(root):
        parsed = parse_market_name(img_path)
        if not parsed:
            continue
        try:
            rel_path = img_path.relative_to(root)
            rel_str = rel_path.as_posix()
        except ValueError:
            rel_str = img_path.as_posix()
        records.append({"path": rel_str, **parsed})

    records.sort(key=lambda r: (r["pid"], r["cam"], r["seq"], r["frame"], r["idx"]))

    write_manifest(records, args.out_csv.expanduser())

    unique_ids = sorted({r["pid"] for r in records if r["pid"] != -1})
    train_ids, val_ids = split_ids(unique_ids, args.val_ratio, args.seed)

    splits = {"train_ids": train_ids, "val_ids": val_ids}
    out_splits = args.out_splits.expanduser()
    out_splits.parent.mkdir(parents=True, exist_ok=True)
    out_splits.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    total_images = len(records)
    total_ids = len(unique_ids)
    print(f"Indexed {total_images} images under {root}")
    print(f"Unique IDs (excluding -1): {total_ids}")
    print(f"Train IDs: {len(train_ids)} | Val IDs: {len(val_ids)}")
    print(f"Manifest written to: {args.out_csv}")
    print(f"Splits written to: {args.out_splits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

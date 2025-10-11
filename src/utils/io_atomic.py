from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _atomic_replace(tmp_path: Path, dst_path: Path) -> None:
    """Replace ``dst_path`` with ``tmp_path`` atomically where possible."""
    if dst_path.exists():
        dst_path.unlink()
    tmp_path.replace(dst_path)


def _make_temp_path(dst_path: Path, suffix: str) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=str(dst_path.parent))
    os.close(fd)
    return Path(tmp_name)


def atomic_write_json(dst_path: Path, data: Mapping[str, object]) -> None:
    tmp_path = _make_temp_path(dst_path, ".json.tmp")
    try:
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        _atomic_replace(tmp_path, dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_csv(dst_path: Path, rows: Iterable[Mapping[str, object]], *, fieldnames: Sequence[str]) -> None:
    tmp_path = _make_temp_path(dst_path, ".csv.tmp")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        _atomic_replace(tmp_path, dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_npy(dst_path: Path, array: np.ndarray) -> None:
    tmp_path = _make_temp_path(dst_path, ".npy.tmp")
    try:
        with tmp_path.open("wb") as f:
            np.save(f, array)
        _atomic_replace(tmp_path, dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_bytes(dst_path: Path, data: bytes, suffix: str = ".tmp") -> None:
    tmp_path = _make_temp_path(dst_path, suffix)
    try:
        with tmp_path.open("wb") as f:
            f.write(data)
        _atomic_replace(tmp_path, dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

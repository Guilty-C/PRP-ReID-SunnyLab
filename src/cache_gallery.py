from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.io_atomic import atomic_write_csv, atomic_write_json, atomic_write_npy

IMG_EXTS = {".jpg", ".jpeg", ".png"}
CACHE_VERSION = 1


@dataclass
class ManifestEntry:
    id: str
    path: str
    etag: str
    mtime_ns: int
    size: int
    sha1: str = ""

    def to_row(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "path": self.path,
            "etag": self.etag,
            "mtime_ns": str(self.mtime_ns),
            "size": str(self.size),
            "sha1": self.sha1 or "",
        }

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "ManifestEntry":
        return cls(
            id=row.get("id", ""),
            path=row.get("path", ""),
            etag=row.get("etag", ""),
            mtime_ns=int(row.get("mtime_ns", "0") or 0),
            size=int(row.get("size", "0") or 0),
            sha1=row.get("sha1", ""),
        )


@dataclass
class KeepInfo:
    index: int
    entry: ManifestEntry


class GalleryCache:
    def __init__(
        self,
        gallery_dir: Path,
        cache_dir: Path,
        *,
        model_id: str,
        image_size: int,
        normalize: str,
        dim: int,
        dtype: str,
        etag_method: str,
        repo_version: str,
    ) -> None:
        self.gallery_dir = gallery_dir
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.image_size = image_size
        self.normalize = normalize
        self.dim = dim
        self.dtype = dtype
        self.etag_method = etag_method
        self.repo_version = repo_version

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "meta.json"
        self.manifest_path = self.cache_dir / "manifest.csv"
        self.embeddings_path = self.cache_dir / "embeddings.npy"

    # ------------------------------------------------------------------
    # Metadata helpers
    def _load_meta(self) -> Optional[Dict[str, object]]:
        if not self.meta_path.exists():
            return None
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _meta_matches(self, meta: Dict[str, object]) -> bool:
        return bool(
            meta.get("cache_version") == CACHE_VERSION
            and meta.get("model_id") == self.model_id
            and meta.get("image_size") == self.image_size
            and meta.get("normalize") == self.normalize
            and meta.get("dim") == self.dim
            and meta.get("dtype") == self.dtype
            and meta.get("repo_version") == self.repo_version
        )

    # ------------------------------------------------------------------
    def load_existing(
        self, *, mmap: bool = False
    ) -> Tuple[Optional[Dict[str, object]], List[ManifestEntry], Optional[np.ndarray]]:
        meta = self._load_meta()
        if not meta or not self._meta_matches(meta):
            return None, [], None
        if not (self.manifest_path.exists() and self.embeddings_path.exists()):
            return None, [], None

        entries: List[ManifestEntry] = []
        try:
            with self.manifest_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entries.append(ManifestEntry.from_row(row))
        except Exception:
            return None, [], None

        try:
            embeddings = np.load(
                self.embeddings_path,
                mmap_mode="r" if mmap else None,
                allow_pickle=False,
            )
        except Exception:
            return None, [], None

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            return None, [], None

        if embeddings.dtype != np.float32:
            embeddings = np.asarray(embeddings, dtype=np.float32)
        return meta, entries, embeddings

    # ------------------------------------------------------------------
    def gather_file_infos(self) -> List[Dict[str, object]]:
        if not self.gallery_dir.exists():
            raise FileNotFoundError(f"Gallery directory not found: {self.gallery_dir}")

        files = [
            p
            for p in sorted(self.gallery_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        infos: List[Dict[str, object]] = []
        for path in files:
            infos.append(self._build_file_info(path))
        return infos

    def _build_file_info(self, path: Path) -> Dict[str, object]:
        stat = path.stat()
        rel_path = path.relative_to(self.gallery_dir).as_posix()
        sha1_hash = ""
        if self.etag_method == "sha1":
            sha1 = hashlib.sha1()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    if not chunk:
                        break
                    sha1.update(chunk)
            sha1_hash = sha1.hexdigest()
            etag = f"{stat.st_size}:{stat.st_mtime_ns}:{sha1_hash}"
        else:
            etag = f"{stat.st_size}:{stat.st_mtime_ns}"
        return {
            "rel_path": rel_path,
            "abs_path": path,
            "etag": etag,
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "sha1": sha1_hash,
        }

    # ------------------------------------------------------------------
    def compute_diff(
        self,
        file_infos: Sequence[Dict[str, object]],
        manifest_entries: Sequence[ManifestEntry],
    ) -> Tuple[List[KeepInfo], List[Dict[str, object]], List[ManifestEntry]]:
        manifest_map: Dict[str, Tuple[int, ManifestEntry]] = {
            entry.path: (idx, entry) for idx, entry in enumerate(manifest_entries)
        }
        keep: List[KeepInfo] = []
        to_encode: List[Dict[str, object]] = []

        for info in file_infos:
            rel_path = info["rel_path"]
            if rel_path in manifest_map:
                idx, entry = manifest_map[rel_path]
                if entry.etag == info["etag"]:
                    keep.append(KeepInfo(index=idx, entry=entry))
                else:
                    to_encode.append(info)
            else:
                to_encode.append(info)

        keep_indices = {item.index for item in keep}
        dropped = [
            entry
            for idx, entry in enumerate(manifest_entries)
            if idx not in keep_indices
        ]
        return keep, to_encode, dropped

    # ------------------------------------------------------------------
    def merge_embeddings(
        self,
        existing_embeddings: Optional[np.ndarray],
        file_infos: Sequence[Dict[str, object]],
        keep_items: Sequence[KeepInfo],
        to_encode_infos: Sequence[Dict[str, object]],
        new_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, List[ManifestEntry]]:
        if existing_embeddings is None:
            existing_embeddings = np.zeros((0, self.dim), dtype=np.float32)

        keep_map: Dict[str, Tuple[np.ndarray, ManifestEntry]] = {}
        for item in keep_items:
            keep_map[item.entry.path] = (
                np.asarray(existing_embeddings[item.index], dtype=np.float32),
                item.entry,
            )

        new_lookup: Dict[str, Tuple[np.ndarray, Dict[str, object]]] = {}
        if len(to_encode_infos) != len(new_embeddings):
            raise ValueError("Number of new embeddings does not match files to encode")
        for idx, info in enumerate(to_encode_infos):
            new_lookup[info["rel_path"]] = (
                np.asarray(new_embeddings[idx], dtype=np.float32),
                info,
            )

        final_vectors: List[np.ndarray] = []
        final_entries: List[ManifestEntry] = []

        for info in file_infos:
            rel_path = info["rel_path"]
            if rel_path in keep_map:
                vector, entry = keep_map[rel_path]
                final_vectors.append(vector)
                final_entries.append(replace(entry))
            elif rel_path in new_lookup:
                vector, new_info = new_lookup[rel_path]
                final_vectors.append(vector)
                final_entries.append(
                    ManifestEntry(
                        id="",
                        path=rel_path,
                        etag=new_info["etag"],
                        mtime_ns=int(new_info["mtime_ns"]),
                        size=int(new_info["size"]),
                        sha1=str(new_info.get("sha1", "")),
                    )
                )
            else:
                raise KeyError(f"Missing embedding for gallery image: {rel_path}")

        for idx, entry in enumerate(final_entries):
            entry.id = str(idx)

        if final_vectors:
            final_matrix = np.vstack(final_vectors).astype(np.float32)
        else:
            final_matrix = np.zeros((0, self.dim), dtype=np.float32)
        return final_matrix, final_entries

    # ------------------------------------------------------------------
    def write_cache(self, embeddings: np.ndarray, entries: Sequence[ManifestEntry]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError("Embeddings shape mismatch")

        target_dtype = np.float16 if self.dtype == "fp16" else np.float32
        serialized_embeddings = embeddings.astype(target_dtype)

        meta = {
            "cache_version": CACHE_VERSION,
            "model_id": self.model_id,
            "image_size": self.image_size,
            "normalize": self.normalize,
            "dim": self.dim,
            "dtype": self.dtype,
            "repo_version": self.repo_version,
            "created_at": time.time(),
        }

        atomic_write_json(self.meta_path, meta)
        atomic_write_csv(
            self.manifest_path,
            (entry.to_row() for entry in entries),
            fieldnames=("id", "path", "etag", "mtime_ns", "size", "sha1"),
        )
        atomic_write_npy(self.embeddings_path, serialized_embeddings)


def get_repo_version(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"

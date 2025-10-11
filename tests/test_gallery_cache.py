from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cache_gallery import GalleryCache, ManifestEntry


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), color)
    image.save(path, format="PNG")


def _build_cache(gallery: Path, cache_dir: Path, *, etag_method: str = "mtime") -> GalleryCache:
    return GalleryCache(
        gallery_dir=gallery,
        cache_dir=cache_dir,
        model_id="test-model",
        image_size=224,
        normalize="l2",
        dim=4,
        dtype="fp32",
        etag_method=etag_method,
        repo_version="test",
    )


def test_gallery_cache_incremental_updates(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery"
    _make_image(gallery / "a.png", (255, 0, 0))
    _make_image(gallery / "b.png", (0, 255, 0))
    _make_image(gallery / "c.png", (0, 0, 255))

    cache_dir = tmp_path / "cache"
    cache = _build_cache(gallery, cache_dir)

    file_infos = cache.gather_file_infos()
    keep, to_encode, dropped = cache.compute_diff(file_infos, [])
    assert not keep
    assert len(to_encode) == 3
    assert not dropped

    embeddings = np.stack([np.full(4, idx, dtype=np.float32) for idx in range(len(to_encode))])
    merged_embeddings, merged_entries = cache.merge_embeddings(None, file_infos, keep, to_encode, embeddings)
    cache.write_cache(merged_embeddings, merged_entries)

    _, manifest, stored = cache.load_existing()
    assert manifest and isinstance(manifest[0], ManifestEntry)
    assert stored is not None and stored.shape == (3, 4)
    assert np.allclose(stored, embeddings)

    # Second scan without changes should keep everything
    file_infos_2 = cache.gather_file_infos()
    keep2, to_encode2, dropped2 = cache.compute_diff(file_infos_2, manifest)
    assert len(keep2) == 3
    assert not to_encode2
    assert not dropped2

    # Modify one image -> only that entry re-encoded
    _make_image(gallery / "a.png", (128, 128, 0))
    file_infos_3 = cache.gather_file_infos()
    keep3, to_encode3, dropped3 = cache.compute_diff(file_infos_3, manifest)
    assert len(keep3) == 2
    assert len(to_encode3) == 1
    assert len(dropped3) == 1

    new_embedding = np.full((1, 4), 99, dtype=np.float32)
    updated_embeddings, updated_entries = cache.merge_embeddings(stored, file_infos_3, keep3, to_encode3, new_embedding)
    cache.write_cache(updated_embeddings, updated_entries)
    _, manifest3, stored3 = cache.load_existing()
    assert stored3.shape == (3, 4)
    changed_path = to_encode3[0]["rel_path"]
    changed_entry = next(entry for entry in manifest3 if entry.path == changed_path)
    assert changed_entry.etag == to_encode3[0]["etag"]

    # Delete one image -> cache shrinks
    (gallery / "b.png").unlink()
    file_infos_4 = cache.gather_file_infos()
    keep4, to_encode4, dropped4 = cache.compute_diff(file_infos_4, manifest3)
    assert len(keep4) == 2
    assert not to_encode4
    assert len(dropped4) == 1

    final_embeddings, final_entries = cache.merge_embeddings(stored3, file_infos_4, keep4, to_encode4, np.zeros((0, 4), dtype=np.float32))
    cache.write_cache(final_embeddings, final_entries)
    _, manifest4, stored4 = cache.load_existing()
    assert stored4.shape == (2, 4)
    remaining_paths = {entry.path for entry in manifest4}
    assert "b.png" not in remaining_paths


def test_gallery_cache_sha1_etag(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery_sha1"
    _make_image(gallery / "sample.png", (12, 34, 56))

    cache_dir = tmp_path / "cache_sha1"
    cache = _build_cache(gallery, cache_dir, etag_method="sha1")

    file_infos = cache.gather_file_infos()
    keep, to_encode, _ = cache.compute_diff(file_infos, [])
    embeddings = np.ones((len(to_encode), 4), dtype=np.float32)
    merged_embeddings, merged_entries = cache.merge_embeddings(None, file_infos, keep, to_encode, embeddings)
    cache.write_cache(merged_embeddings, merged_entries)

    _, manifest, _ = cache.load_existing()
    assert manifest
    entry = manifest[0]
    info = file_infos[0]

    with (gallery / entry.path).open("rb") as f:
        expected_sha1 = hashlib.sha1(f.read()).hexdigest()

    assert entry.sha1 == expected_sha1
    assert entry.sha1 and entry.sha1 in entry.etag
    assert info["sha1"] == expected_sha1

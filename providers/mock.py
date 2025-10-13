from __future__ import annotations
import hashlib
import numpy as np
from .base import EmbeddingProvider

_DEF_DIM = 512

class MockProvider(EmbeddingProvider):
    """Deterministic, dependency-free mock embeddings.

    - Text: seed from SHA1(text)
    - Image: seed from SHA1(file bytes)
    Useful for plumbing tests without external APIs / heavy models.
    """
    name = "mock"

    def __init__(self, dim: int = _DEF_DIM):
        self.dim = int(dim)

    def _vec_by_seed(self, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        v = rng.randn(self.dim).astype("float32")
        return v

    def embed_text(self, texts: list[str]) -> np.ndarray:
        mats = []
        for t in texts:
            h = int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8], 16)
            mats.append(self._vec_by_seed(h))
        X = np.stack(mats, axis=0)
        return self.l2_normalize(X)

    def embed_images(self, paths: list[str]) -> np.ndarray:
        mats = []
        for p in paths:
            with open(p, "rb") as f:
                h = int(hashlib.sha1(f.read()).hexdigest()[:8], 16)
            mats.append(self._vec_by_seed(h))
        X = np.stack(mats, axis=0)
        return self.l2_normalize(X)

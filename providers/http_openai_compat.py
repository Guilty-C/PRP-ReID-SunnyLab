
from __future__ import annotations
import os, time, json
from typing import List, Optional
import numpy as np, requests
from .base import EmbeddingProvider

def _l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

class OpenAICompatEmbedTextProvider(EmbeddingProvider):
    name = "openai-compat"
    prefix: str = "OPENAI"  # override

    def __init__(self, prefix: Optional[str] = None):
        self.prefix = prefix or self.prefix
        self.api_key  = self._env("API_KEY", required=True)
        self.api_base = self._env("API_BASE", default="https://api.openai.com/v1").rstrip("/")
        self.model    = self._env("EMBED_MODEL", required=True)
        self.batch    = int(self._env("BATCH", default="64"))
        self.timeout  = int(self._env("TIMEOUT", default="60"))
        import requests  # ensure present
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        self.embed_url = f"{self.api_base}/embeddings"

    def _env(self, key: str, default: str|None=None, required: bool=False) -> str:
        env_key = f"{self.prefix}_{key}"
        v = os.environ.get(env_key, default)
        if required and (v is None or v == ""):
            raise SystemExit(f"[{self.__class__.__name__}] Missing env: {env_key}")
        return str(v)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.empty((0,0), dtype="float32")
        out = []
        for i in range(0, len(texts), self.batch):
            out.append(self._post(texts[i:i+self.batch]))
        X = np.concatenate(out, axis=0)
        return _l2n(X.astype("float32"))

    def embed_images(self, paths: List[str]) -> np.ndarray:
        raise SystemExit(f"{self.__class__.__name__} supports text embeddings only.")

    def _post(self, texts: List[str]) -> np.ndarray:
        payload = {"input": texts, "model": self.model}
        backoff = 1.0
        for _ in range(6):
            try:
                r = self.session.post(self.embed_url, data=json.dumps(payload), timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()["data"]
                    vecs = [np.array(d["embedding"], dtype="float32") for d in data]
                    return np.stack(vecs, axis=0)
                if r.status_code in (429,500,502,503,504):
                    time.sleep(backoff); backoff = min(backoff*2, 8); continue
                raise SystemExit(f"HTTP {r.status_code}: {r.text}")
            except Exception:
                time.sleep(backoff); backoff = min(backoff*2, 8)
        raise SystemExit("Failed after retries")

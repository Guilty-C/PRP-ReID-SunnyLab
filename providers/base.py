# providers/base.py
from __future__ import annotations

import os
from typing import Any, Dict, Sequence, Tuple
import requests


class ProviderBase:
    """通用 Provider 接口：文本/图像嵌入由各子类按需实现。"""

    # 文本嵌入：返回 (N, D) 的 np.ndarray（float32）
    def embed_text(self, texts: Sequence[str]):
        raise NotImplementedError("This provider does not implement embed_text().")

    # 图像嵌入：有的实现会返回 (Xq, Xg) 两个矩阵
    def embed_images(self, *args, **kwargs):
        raise NotImplementedError("This provider does not implement embed_images().")


class HTTPProviderBase(ProviderBase):
    """HTTP(OpenAI兼容风格) 提供者的基类，子类可直接 self._post() 调用。"""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.api_base = os.environ.get(f"{prefix}_API_BASE", "").rstrip("/")
        self.api_key = os.environ.get(f"{prefix}_API_KEY", "")
        self.timeout = float(os.environ.get(f"{prefix}_TIMEOUT", "60"))

        self.session = requests.Session()
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def _post(self, path: str, json: Dict[str, Any]):
        if not self.api_base:
            raise RuntimeError(f"{self.prefix}_API_BASE not set")
        url = f"{self.api_base}{path if path.startswith('/') else '/' + path}"
        r = self.session.post(url, json=json, timeout=self.timeout)
        r.raise_for_status()
        return r


# 兼容别名
APIProviderBase = HTTPProviderBase

__all__ = ["ProviderBase", "HTTPProviderBase", "APIProviderBase", "EmbeddingProvider"]


# Back-compat alias for older providers
EmbeddingProvider = ProviderBase

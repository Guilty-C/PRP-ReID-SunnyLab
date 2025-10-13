# providers/clip_text_local.py
from __future__ import annotations
import os
import math
from typing import List, Optional

import numpy as np
import torch
import open_clip  # pip install open-clip-torch

from . import register

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Mac MPS 可选
        return torch.device("mps")
    return torch.device("cpu")

@register("clip-text-local")
class ClipTextLocalProvider:
    """
    使用 open_clip 在本地把文本编码为嵌入向量。
    - 支持批处理、自动设备、L2 归一化（与图像侧对齐更稳）。
    """
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        device: Optional[torch.device] = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or _device()
        self.normalize = normalize

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    @classmethod
    def from_env(cls, **kwargs):
        """优先从环境变量读取配置。"""
        model = os.getenv("CLIP_LOCAL_MODEL", kwargs.pop("model_name", "ViT-B-16"))
        pretrain = os.getenv("CLIP_LOCAL_PRETRAIN", kwargs.pop("pretrained", "laion2b_s34b_b88k"))
        return cls(model_name=model, pretrained=pretrain, **kwargs)

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        返回 shape = (N, D) 的 float32 numpy 数组。
        """
        if not texts:
            return np.zeros((0, self.model.text_projection.shape[1]), dtype=np.float32)

        embs: List[np.ndarray] = []
        total = len(texts)
        steps = math.ceil(total / batch_size)

        for i in range(steps):
            batch = texts[i * batch_size : (i + 1) * batch_size]
            tokens = self.tokenizer(batch)  # 已含截断
            tokens = tokens.to(self.device)

            feats = self.model.encode_text(tokens)
            feats = feats.float()

            if self.normalize:
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)

            embs.append(feats.detach().cpu().numpy())

        return np.concatenate(embs, axis=0)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

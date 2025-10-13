import os
from pathlib import Path
from typing import List
import numpy as np
import torch
import open_clip
from PIL import Image
from .base import EmbeddingProvider

class ClipLocalProvider(EmbeddingProvider):
    """
    Local CLIP encoder using open-clip-torch.
    - Model: ViT-B/32 (laion2b_s34b_b79k)
    - Works on CPU/GPU; batch size via env CLIP_LOCAL_BATCH (default 32)
    """
    name = "clip-local"

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        # dimension inferred from model
        with torch.no_grad():
            self._dim = int(self.model.text_projection.shape[1]) if hasattr(self.model, "text_projection") else 512
        try:
            self.batch = int(os.environ.get("CLIP_LOCAL_BATCH", "32"))
        except ValueError:
            self.batch = 32

    def _encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            feats = self.model.encode_image(imgs)
            feats = feats.float()
            feats = torch.nn.functional.normalize(feats, dim=1)
        return feats

    def _encode_texts(self, toks: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            feats = self.model.encode_text(toks)
            feats = feats.float()
            feats = torch.nn.functional.normalize(feats, dim=1)
        return feats

    def embed_images(self, paths: List[str]) -> np.ndarray:
        out = []
        b = self.batch
        N = len(paths)
        for i in range(0, N, b):
            chunk = paths[i:i+b]
            ims = []
            for p in chunk:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                ims.append(self.preprocess(im))
            batch = torch.stack(ims, dim=0).to(self.device)
            feats = self._encode_images(batch).cpu().numpy().astype("float32")
            out.append(feats)
        X = np.concatenate(out, axis=0) if out else np.empty((0, self._dim), dtype="float32")
        return X  # 已归一化，无需再 l2_normalize

    def embed_text(self, texts: List[str]) -> np.ndarray:
        # 不是本次评测必需，但补上以便未来做跨模态
        out = []
        b = self.batch
        N = len(texts)
        for i in range(0, N, b):
            chunk = texts[i:i+b]
            toks = self.tokenizer(chunk).to(self.device)
            feats = self._encode_texts(toks).cpu().numpy().astype("float32")
            out.append(feats)
        X = np.concatenate(out, axis=0) if out else np.empty((0, self._dim), dtype="float32")
        return X


import os
import numpy as np
import torch, open_clip
from .base import ProviderBase

class CLIPLocalProvider(ProviderBase):
    name = "clip-local"

    def __init__(self):
        super().__init__(prefix="CLIP_LOCAL")
        self.model_name = os.environ.get("CLIP_LOCAL_MODEL", "ViT-B-32")
        self.pretrained = os.environ.get("CLIP_LOCAL_PRETRAIN", "laion2b_s34b_b79k")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("CLIP local ->", self.model_name, "/", self.pretrained)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.device = device
        self.model.eval()

    def embed_images(self, pil_images):
        # 输入: List[PIL.Image]
        imgs = [self.preprocess(img).unsqueeze(0) for img in pil_images]
        x = torch.cat(imgs, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feats.detach().cpu().numpy().astype("float32")

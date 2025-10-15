"""Prompt augmentation utilities for the tuner CLI."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

NEGATIVE_CLAUSES: Sequence[str] = (
    "no occlusion",
    "not blurry",
    "no heavy shadow",
    "no back-view if uncertain",
    "no accessories unless clearly visible",
)


@dataclass(slots=True)
class AugmentConfig:
    """Configuration flags for prompt augmentation."""

    ensemble: int = 1
    neg_rate: float = 0.0
    seed: Optional[int] = None
    deterministic: bool = True


def augment_prompts(base_prompts: Sequence[str], cfg: AugmentConfig) -> List[str]:
    """Return augmented prompts according to *cfg*."""

    if cfg.deterministic and cfg.seed is not None:
        random.seed(cfg.seed)

    augmented: List[str] = []
    for prompt in base_prompts:
        cleaned = prompt.strip()
        if not cleaned:
            continue
        for _ in range(max(cfg.ensemble, 0)):
            variant = cleaned
            if cfg.neg_rate > 0.0 and random.random() < cfg.neg_rate:
                clause = random.choice(NEGATIVE_CLAUSES)
                variant = f"{variant} ({clause})"
            augmented.append(variant)
    return augmented

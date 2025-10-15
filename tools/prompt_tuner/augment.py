"""Prompt augmentation utilities for the tuner CLI."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Sequence

NEGATIVE_PHRASES: list[str] = [
    "no occlusion",
    "not blurry",
    "no heavy shadow",
    "no back-view if uncertain",
    "no accessories unless clearly visible",
]

LOGGER = logging.getLogger(__name__)


@dataclass
class AugmentConfig:
    """Configuration flags for prompt augmentation."""

    ensemble: int = 1
    neg_rate: float = 0.0
    seed: int | None = None
    deterministic: bool = True


def augment_prompts(base_prompts: Sequence[str], cfg: AugmentConfig) -> list[str]:
    """Return augmented prompts according to *cfg* without mutating the input."""

    if cfg.deterministic and cfg.seed is not None:
        random.seed(cfg.seed)
        LOGGER.debug("Seeded random with %d for deterministic augmentation", cfg.seed)

    augmented: list[str] = []
    for prompt in base_prompts:
        for _ in range(cfg.ensemble):
            variant = prompt
            if cfg.neg_rate > 0.0 and random.random() < cfg.neg_rate:
                clause = random.choice(NEGATIVE_PHRASES)
                variant = f"{variant} ({clause})"
            augmented.append(variant)

    LOGGER.debug(
        "Augmented %d prompts from %d base entries (ensemble=%d, neg_rate=%.3f)",
        len(augmented),
        len(base_prompts),
        cfg.ensemble,
        cfg.neg_rate,
    )
    return augmented

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prompt_tuner.augment import (
    NEGATIVE_PHRASES,
    AugmentConfig,
    augment_prompts,
)


def test_augment_length_and_input_integrity() -> None:
    base = ["prompt one", "prompt two"]
    cfg = AugmentConfig(ensemble=3, neg_rate=0.2, seed=42, deterministic=True)
    original = tuple(base)
    augmented = augment_prompts(base, cfg)

    assert len(augmented) == len(base) * cfg.ensemble
    assert tuple(base) == original


def test_augment_injects_negative_phrase_when_always_triggered() -> None:
    base = ["observe shoes"]
    cfg = AugmentConfig(ensemble=2, neg_rate=1.0, seed=123, deterministic=True)

    augmented = augment_prompts(base, cfg)

    assert any(
        any(phrase in variant for phrase in NEGATIVE_PHRASES)
        for variant in augmented
    )
    assert any(variant != base[0] for variant in augmented)


def test_augment_respects_zero_negative_rate() -> None:
    base = ["keep baseline", "keep two"]
    cfg = AugmentConfig(ensemble=2, neg_rate=0.0, seed=777, deterministic=True)

    augmented = augment_prompts(base, cfg)

    expected = [prompt for prompt in base for _ in range(cfg.ensemble)]
    assert augmented == expected

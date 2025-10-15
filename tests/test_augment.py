from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prompt_tuner.augment import AugmentConfig, augment_prompts


def test_augment_length_scales_with_ensemble() -> None:
    base = ["prompt one", "prompt two"]
    cfg = AugmentConfig(ensemble=3, neg_rate=0.0)
    augmented = augment_prompts(base, cfg)
    assert len(augmented) == len(base) * cfg.ensemble
    assert augmented.count("prompt one") == cfg.ensemble
    assert augmented.count("prompt two") == cfg.ensemble


def test_augment_is_deterministic_with_seed() -> None:
    base = ["observe shoes"]
    cfg = AugmentConfig(ensemble=2, neg_rate=1.0, seed=99)
    first = augment_prompts(base, cfg)
    second = augment_prompts(base, cfg)
    assert first == second
    assert any("observe shoes" != variant for variant in first)


def test_augment_does_not_mutate_inputs() -> None:
    base = ["keep baseline", "keep two"]
    cfg = AugmentConfig(ensemble=1, neg_rate=0.0)
    original = tuple(base)
    augment_prompts(base, cfg)
    assert tuple(base) == original

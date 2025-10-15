"""Smoke test for the prompt tuner agent."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.prompt_tuner.augment import AugmentConfig, augment_prompts


def _run_cli() -> None:
    script = Path(__file__).resolve().parent / "agent.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--rounds",
            "1",
            "--ensemble",
            "1",
            "--neg-rate",
            "0",
        ],
        check=True,
    )


def _check_augmentation_contract() -> None:
    base_prompts = ["red jacket", "blue shirt"]
    cfg = AugmentConfig(ensemble=2, neg_rate=0.1, seed=123, deterministic=True)
    augmented = augment_prompts(base_prompts, cfg)
    assert len(augmented) == len(base_prompts) * cfg.ensemble
    assert base_prompts == ["red jacket", "blue shirt"]


def run_smoke() -> None:
    _run_cli()
    _check_augmentation_contract()


if __name__ == "__main__":
    run_smoke()

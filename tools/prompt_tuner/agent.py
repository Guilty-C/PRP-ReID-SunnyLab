"""Prompt tuning agent CLI."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import List

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.prompt_tuner.augment import AugmentConfig, augment_prompts

DEFAULT_PROMPTS: List[str] = [
    "Describe the person succinctly while focusing on clearly visible cues.",
    "Highlight colors, clothing types, and obvious accessories in the description.",
]

LOGGER = logging.getLogger(__name__)


def _float_in_unit_interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:  # pragma: no cover - defensive against argparse
        raise argparse.ArgumentTypeError(str(exc)) from exc

    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1 inclusive")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - defensive against argparse
        raise argparse.ArgumentTypeError(str(exc)) from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _build_base_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt:
        return list(args.prompt)
    return DEFAULT_PROMPTS.copy()


def _resolve_config(args: argparse.Namespace) -> AugmentConfig:
    return AugmentConfig(
        ensemble=args.ensemble,
        neg_rate=args.neg_rate,
        seed=args.seed,
        deterministic=args.deterministic_augment,
    )


def run_agent(args: argparse.Namespace) -> List[List[str]]:
    base_prompts = _build_base_prompts(args)
    cfg = _resolve_config(args)
    augmented_prompts = augment_prompts(base_prompts, cfg)

    LOGGER.info(
        "Prepared %d prompts across %d base entries (ensemble=%d, neg_rate=%.2f)",
        len(augmented_prompts),
        len(base_prompts),
        cfg.ensemble,
        cfg.neg_rate,
    )

    rounds: List[List[str]] = []
    for _ in range(args.rounds):
        rounds.append(list(augmented_prompts))
    return rounds


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prompt tuner agent")
    parser.add_argument(
        "--rounds",
        type=_positive_int,
        default=1,
        help="Number of dialogue rounds to emit prompts for (default: 1)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Base prompt string; repeatable. Defaults to two internal prompts if omitted.",
    )
    parser.add_argument(
        "--ensemble",
        type=_positive_int,
        default=1,
        help=(
            "Number of prompt variants to produce per base prompt. "
            "Use 1 to keep existing behaviour (no replication).",
        ),
    )
    parser.add_argument(
        "--neg-rate",
        dest="neg_rate",
        type=_float_in_unit_interval,
        default=0.0,
        help=(
            "Probability for appending a safety-focused negative clause to each variant. "
            "Set to 0 to preserve current prompts.",
        ),
    )
    parser.add_argument(
        "--neg_rate",
        dest="neg_rate",
        type=_float_in_unit_interval,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible augmentation.",
    )
    parser.add_argument(
        "--no-deterministic-augment",
        dest="deterministic_augment",
        action="store_false",
        default=True,
        help="Disable deterministic augmentation even when a seed is provided.",
    )
    parser.add_argument(
        "--log-level",
        choices=("INFO", "DEBUG", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity for the agent (default: INFO).",
    )

    parsed = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, parsed.log_level))

    rounds = run_agent(parsed)
    for round_idx, prompts in enumerate(rounds, start=1):
        print(f"Round {round_idx}:")
        for prompt in prompts:
            print(f"- {prompt}")


if __name__ == "__main__":
    main()

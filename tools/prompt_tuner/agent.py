"""Prompt tuning agent CLI.

This module exposes a simple loop that prints tuned prompts across multiple
rounds. The agent can optionally duplicate prompts via an ensemble factor and
sprinkle negative safety clauses at random.
"""
from __future__ import annotations

import argparse
import random
from typing import Iterable, List

NEGATIVE_CLAUSES: List[str] = [
    "no occlusion",
    "not blurry",
    "no heavy shadow",
    "no back-view if uncertain",
    "no accessories unless clearly visible",
]

DEFAULT_PROMPTS: List[str] = [
    "Describe the person succinctly while focusing on clearly visible cues.",
    "Highlight colors, clothing types, and obvious accessories in the description.",
]


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


def _augment_prompts(base_prompts: Iterable[str], ensemble: int, neg_rate: float) -> List[str]:
    """Return an augmented prompt list.

    Args:
        base_prompts: The raw prompts to replicate.
        ensemble: Number of variants to generate per base prompt.
        neg_rate: Probability of appending a negative clause to a variant.
    """

    prompts: List[str] = []
    base_list = list(base_prompts)

    if not base_list:
        return prompts

    for prompt in base_list:
        stripped = prompt.strip()
        if not stripped:
            continue

        for replica_idx in range(ensemble):
            variant = stripped
            if ensemble > 1:
                variant = f"{variant} (variant {replica_idx + 1})"

            if neg_rate > 0.0 and random.random() < neg_rate:
                clause = random.choice(NEGATIVE_CLAUSES)
                sanitized = variant.rstrip()
                if sanitized and sanitized[-1] in ".!?":
                    variant = f"{sanitized} {clause}."
                else:
                    variant = f"{sanitized}, {clause}."

            prompts.append(variant)

    return prompts


def _build_base_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt:
        return args.prompt
    return DEFAULT_PROMPTS.copy()


def run_agent(args: argparse.Namespace) -> List[List[str]]:
    base_prompts = _build_base_prompts(args)
    augmented_prompts = _augment_prompts(base_prompts, args.ensemble, args.neg_rate)

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
            "Use 1 to keep existing behaviour (no replication)."
        ),
    )
    parser.add_argument(
        "--neg_rate",
        type=_float_in_unit_interval,
        default=0.0,
        help=(
            "Probability for appending a safety-focused negative clause to each variant. "
            "Set to 0 to preserve current prompts."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible augmentation.",
    )

    parsed = parser.parse_args(argv)

    if parsed.seed is not None:
        random.seed(parsed.seed)

    rounds = run_agent(parsed)
    for round_idx, prompts in enumerate(rounds, start=1):
        print(f"Round {round_idx}:")
        for prompt in prompts:
            print(f"- {prompt}")


if __name__ == "__main__":
    main()

"""Inference helpers including JSONL logging for query metrics."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

Arm = Literal["baseline", "prompt"]
Domain = Literal["in_domain", "unseen"]


@dataclass
class QueryLog:
    """Serializable per-query inference record."""

    qid: str
    arm: Arm
    AP: float
    R1: int
    latency_ms: float
    online_tokens: int
    gpu_time_s: float
    domain: Domain
    occluded: bool
    top1_score: float

    def to_json(self) -> str:
        """Return the JSON string for this log entry."""

        payload = asdict(self)
        payload["R1"] = int(self.R1)
        payload["online_tokens"] = int(self.online_tokens)
        payload["occluded"] = bool(self.occluded)
        return json.dumps(payload, sort_keys=True)


class QueryLogger:
    """Append-only JSONL logger for inference metrics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: QueryLog) -> None:
        """Write a query log entry to disk."""

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.to_json())
            handle.write("\n")


def route_query(
    top1_score: float,
    tau: float,
    pws_mean: Optional[float] = None,
    pws_low95: Optional[float] = None,
) -> bool:
    """Decide whether to use the prompt arm for a query."""

    if tau <= 0:
        raise ValueError("tau must be positive")
    if pws_mean is not None and pws_low95 is not None:
        from reid.pws.core import route

        return route(top1_score, tau, pws_mean, pws_low95)
    return top1_score < tau


__all__ = ["QueryLog", "QueryLogger", "route_query"]

"""I/O helpers for Prompt Worth Score workflows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import pandas as pd
import yaml
from pydantic import BaseModel, Field, validator


class LogRow(BaseModel):
    """Schema for a single inference log row."""

    qid: str
    arm: str
    AP: float
    R1: int
    latency_ms: float
    online_tokens: int
    gpu_time_s: float
    domain: str
    occluded: bool
    top1_score: float

    @validator("arm")
    def validate_arm(cls, value: str) -> str:
        if value not in {"baseline", "prompt"}:
            raise ValueError("arm must be 'baseline' or 'prompt'")
        return value

    @validator("R1")
    def validate_r1(cls, value: int) -> int:
        if value not in {0, 1}:
            raise ValueError("R1 must be 0 or 1")
        return value

    @validator("domain")
    def validate_domain(cls, value: str) -> str:
        if value not in {"in_domain", "unseen"}:
            raise ValueError("domain must be 'in_domain' or 'unseen'")
        return value

    class Config:
        extra = "ignore"


class WeightsConfig(BaseModel):
    """Weights configuration schema."""

    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.7
    delta: float = 0.5
    kappa: float = 1.0
    lambda_: float = Field(100.0, alias="lambda")
    mu: float = 0.2
    nu: float = 0.5

    def as_dict(self) -> Mapping[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "mu": self.mu,
            "nu": self.nu,
        }


class CostsConfig(BaseModel):
    """Cost configuration schema."""

    token_price_per_1k: float
    GPU_cost_per_hr: float
    monthly_offline_tokens: float = 0.0
    monthly_queries: int = 1

    @validator("monthly_queries")
    def validate_queries(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("monthly_queries must be positive")
        return value

    def as_dict(self) -> Mapping[str, float]:
        return {
            "token_price_per_1k": self.token_price_per_1k,
            "GPU_cost_per_hr": self.GPU_cost_per_hr,
            "monthly_offline_tokens": self.monthly_offline_tokens,
            "monthly_queries": self.monthly_queries,
        }


def read_jsonl(path: str | Path) -> pd.DataFrame:
    """Read a JSONL log file into a validated dataframe."""

    path = Path(path)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                row = LogRow.model_validate(payload)
            except Exception as exc:  # pragma: no cover - pydantic message includes context
                raise ValueError(f"Invalid log row at line {line_number}: {exc}") from exc
            rows.append(row.model_dump())
    return pd.DataFrame(rows)


def merge_logs(baseline: pd.DataFrame, prompt: pd.DataFrame) -> pd.DataFrame:
    """Merge baseline and prompt logs on query identifier."""

    if baseline.empty or prompt.empty:
        raise ValueError("Both baseline and prompt logs must contain rows")
    if baseline["qid"].duplicated().any():
        raise ValueError("Duplicate qids found in baseline logs")
    if prompt["qid"].duplicated().any():
        raise ValueError("Duplicate qids found in prompt logs")
    merged = pd.merge(baseline, prompt, on="qid", suffixes=("_baseline", "_prompt"))

    if not (merged["arm_baseline"] == "baseline").all():
        raise ValueError("baseline logs must have arm='baseline'")
    if not (merged["arm_prompt"] == "prompt").all():
        raise ValueError("prompt logs must have arm='prompt'")

    if (merged["domain_baseline"] != merged["domain_prompt"]).any():
        raise ValueError("domain mismatch between baseline and prompt logs")
    if (merged["occluded_baseline"] != merged["occluded_prompt"]).any():
        raise ValueError("occlusion flags mismatch between baseline and prompt logs")

    merged = merged.drop(columns=["arm_baseline", "arm_prompt", "domain_prompt", "occluded_prompt"])
    merged = merged.rename(columns={"domain_baseline": "domain", "occluded_baseline": "occluded"})
    return merged


def load_weights(path: str | Path | None) -> Mapping[str, float]:
    """Load weight configuration from YAML."""

    if path is None:
        return WeightsConfig().as_dict()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return WeightsConfig.model_validate(data).as_dict()


def load_costs(path: str | Path | None) -> Mapping[str, float]:
    """Load cost configuration from YAML."""

    if path is None:
        raise ValueError("Cost configuration path is required")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return CostsConfig.model_validate(data).as_dict()


def write_json(data: Mapping[str, object], path: str | Path) -> None:
    """Write a JSON summary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    """Write a CSV decision matrix."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def write_markdown_report(
    summary: Mapping[str, float],
    path: str | Path,
    prompt_type: str,
    tau: float,
    deploy: bool,
) -> None:
    """Write a concise markdown report summarizing PWS results."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    decision_text = "Deploy" if deploy else "Hold"
    lines = [
        f"# Prompt Worth Score Report — Prompt {prompt_type}",
        "",
        f"**Decision:** {decision_text} (lower 95% CI = {summary['PWS_lower95']:.2f})",
        "",
        "## Key Metrics",
        f"- ΔmAP: {summary['ΔmAP']:.2f} pp",
        f"- ΔR1: {summary['ΔR1']:.2f} pp",
        f"- ΔDG: {summary['ΔDG']:.2f} pp",
        f"- ΔOcc: {summary['ΔOcc']:.2f} pp",
        f"- Latency Penalty: {summary['LatencyPenalty']:.4f}",
        f"- Token $/q: {summary['TokenCost']:.6f}",
        f"- GPU $/q: {summary['GPUCost']:.6f}",
        f"- Offline Token $/q: {summary['OfflineTokenCost']:.6f}",
        "",
        "## PWS Distribution",
        f"Mean PWS: {summary['PWS_mean']:.2f} (95% CI: {summary['PWS_lower95']:.2f} — {summary['PWS_upper95']:.2f}).",
        "",
        f"Routing guidance: route to prompt when top-1 score < τ ({tau}) and the above PWS conditions hold.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Core computations for the Prompt Worth Score (PWS)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PWSWeights:
    """Container for Prompt Worth Score weights."""

    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.7
    delta: float = 0.5
    kappa: float = 1.0
    lambda_: float = 100.0
    mu: float = 0.2
    nu: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "PWSWeights":
        """Create an instance from a generic mapping."""

        return cls(
            alpha=float(values.get("alpha", cls.alpha)),
            beta=float(values.get("beta", cls.beta)),
            gamma=float(values.get("gamma", cls.gamma)),
            delta=float(values.get("delta", cls.delta)),
            kappa=float(values.get("kappa", cls.kappa)),
            lambda_=float(values.get("lambda", cls.lambda_)),
            mu=float(values.get("mu", cls.mu)),
            nu=float(values.get("nu", cls.nu)),
        )

    def as_dict(self) -> Dict[str, float]:
        """Return the weights as a standard dictionary."""

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


def compute_latency_penalty(p95_ms: float, sla_ms: float) -> float:
    """Compute the latency penalty given a 95th percentile latency and SLA."""

    if sla_ms <= 0:
        raise ValueError("SLA must be positive")
    return float(max(0.0, (p95_ms - sla_ms) / sla_ms))


def compute_costs(
    online_tokens: float,
    token_price_per_1k: float,
    monthly_offline_tokens: float,
    monthly_queries: int,
    gpu_time_s: float,
    gpu_cost_per_hr: float,
) -> Dict[str, float]:
    """Compute per-query token and GPU costs."""

    if monthly_queries <= 0:
        raise ValueError("monthly_queries must be positive")
    token_cost = (online_tokens / 1000.0) * token_price_per_1k
    offline_cost = (monthly_offline_tokens / 1000.0 * token_price_per_1k) / monthly_queries
    gpu_cost = (gpu_time_s / 3600.0) * gpu_cost_per_hr
    return {
        "TokenCost": float(token_cost),
        "OfflineTokenCost": float(offline_cost),
        "GPUCost": float(gpu_cost),
    }


def compute_risk(
    drift: float,
    bias: float,
    privacy: float,
    repro: float,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute a weighted risk score in [0, 1]."""

    if weights is None:
        weights = {"drift": 0.25, "bias": 0.25, "privacy": 0.25, "repro": 0.25}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Risk weights must sum to a positive value")
    normalized = {k: v / total_weight for k, v in weights.items()}
    score = (
        drift * normalized.get("drift", 0.0)
        + bias * normalized.get("bias", 0.0)
        + privacy * normalized.get("privacy", 0.0)
        + repro * normalized.get("repro", 0.0)
    )
    return float(score)


def compute_deltas(df: pd.DataFrame) -> Dict[str, float]:
    """Compute delta metrics between prompt and baseline arms."""

    required_columns = {
        "AP_baseline",
        "AP_prompt",
        "R1_baseline",
        "R1_prompt",
        "latency_ms_prompt",
        "domain",
        "occluded",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")

    delta_map = (df["AP_prompt"] - df["AP_baseline"]).mean() * 100.0
    delta_r1 = (df["R1_prompt"] - df["R1_baseline"]).mean() * 100.0

    unseen_mask = df["domain"] == "unseen"
    if unseen_mask.any():
        delta_dg = (df.loc[unseen_mask, "AP_prompt"] - df.loc[unseen_mask, "AP_baseline"]).mean() * 100.0
    else:
        delta_dg = 0.0

    occ_mask = df["occluded"].astype(bool)
    if occ_mask.any():
        delta_occ = (df.loc[occ_mask, "AP_prompt"] - df.loc[occ_mask, "AP_baseline"]).mean() * 100.0
    else:
        delta_occ = 0.0

    latency_p95 = float(np.percentile(df["latency_ms_prompt"], 95))

    return {
        "ΔmAP": float(delta_map),
        "ΔR1": float(delta_r1),
        "ΔDG": float(delta_dg),
        "ΔOcc": float(delta_occ),
        "Latency_p95": latency_p95,
    }


def compute_pws(
    deltas: Mapping[str, float],
    costs: Mapping[str, float],
    complexity: int,
    risk: float,
    weights: Mapping[str, float],
) -> float:
    """Compute the Prompt Worth Score from component metrics."""

    if complexity < 1 or complexity > 5:
        raise ValueError("complexity must be in [1, 5]")
    comp_norm = (complexity - 1) / 4.0
    lambda_weight = weights.get("lambda") if "lambda" in weights else weights.get("lambda_", 100.0)
    score = (
        weights.get("alpha", 1.0) * deltas["ΔmAP"]
        + weights.get("beta", 0.5) * deltas["ΔR1"]
        + weights.get("gamma", 0.7) * deltas["ΔDG"]
        + weights.get("delta", 0.5) * deltas["ΔOcc"]
        - weights.get("kappa", 1.0) * deltas["LatencyPenalty"]
        - lambda_weight * (
            costs["TokenCost"] + costs["OfflineTokenCost"] + costs["GPUCost"]
        )
        - weights.get("mu", 0.2) * comp_norm
        - weights.get("nu", 0.5) * risk
    )
    return float(score)


def bootstrap_pws(
    df: pd.DataFrame,
    static_inputs: Mapping[str, object],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap Prompt Worth Score estimates."""

    if df.empty:
        raise ValueError("Input dataframe must not be empty")
    sla_ms = float(static_inputs["sla_ms"])
    weights_map = static_inputs["weights"]
    costs_cfg = static_inputs["costs"]
    complexity = int(static_inputs["complexity"])
    risk_cfg = static_inputs.get("risk", {})
    risk = compute_risk(
        float(risk_cfg.get("drift", 0.0)),
        float(risk_cfg.get("bias", 0.0)),
        float(risk_cfg.get("privacy", 0.0)),
        float(risk_cfg.get("repro", 0.0)),
        risk_cfg.get("weights"),
    )

    rng = np.random.default_rng(seed)
    n = len(df)

    pws_samples: list[float] = []
    latency_penalties: list[float] = []
    delta_map_samples: list[float] = []
    delta_r1_samples: list[float] = []
    delta_dg_samples: list[float] = []
    delta_occ_samples: list[float] = []
    token_costs: list[float] = []
    offline_costs: list[float] = []
    gpu_costs: list[float] = []

    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        sample = df.iloc[indices]
        deltas = compute_deltas(sample)
        latency_penalty = compute_latency_penalty(deltas.pop("Latency_p95"), sla_ms)
        deltas["LatencyPenalty"] = latency_penalty
        mean_online_tokens = float(sample["online_tokens_prompt"].mean())
        mean_gpu_time = float(sample["gpu_time_s_prompt"].mean())
        costs = compute_costs(
            online_tokens=mean_online_tokens,
            token_price_per_1k=float(costs_cfg["token_price_per_1k"]),
            monthly_offline_tokens=float(costs_cfg.get("monthly_offline_tokens", 0.0)),
            monthly_queries=int(costs_cfg.get("monthly_queries", 1)),
            gpu_time_s=mean_gpu_time,
            gpu_cost_per_hr=float(costs_cfg["GPU_cost_per_hr"]),
        )
        score = compute_pws(deltas, costs, complexity, risk, weights_map)

        pws_samples.append(score)
        latency_penalties.append(latency_penalty)
        delta_map_samples.append(deltas["ΔmAP"])
        delta_r1_samples.append(deltas["ΔR1"])
        delta_dg_samples.append(deltas["ΔDG"])
        delta_occ_samples.append(deltas["ΔOcc"])
        token_costs.append(costs["TokenCost"])
        offline_costs.append(costs["OfflineTokenCost"])
        gpu_costs.append(costs["GPUCost"])

    summary = {
        "PWS_mean": float(np.mean(pws_samples)),
        "PWS_lower95": float(np.percentile(pws_samples, 2.5)),
        "PWS_upper95": float(np.percentile(pws_samples, 97.5)),
        "ΔmAP": float(np.mean(delta_map_samples)),
        "ΔR1": float(np.mean(delta_r1_samples)),
        "ΔDG": float(np.mean(delta_dg_samples)),
        "ΔOcc": float(np.mean(delta_occ_samples)),
        "LatencyPenalty": float(np.mean(latency_penalties)),
        "TokenCost": float(np.mean(token_costs)),
        "OfflineTokenCost": float(np.mean(offline_costs)),
        "GPUCost": float(np.mean(gpu_costs)),
    }
    return summary


def decision(pws_ci: Mapping[str, float]) -> bool:
    """Return True if the lower 95%% CI exceeds zero."""

    return float(pws_ci.get("PWS_lower95", 0.0)) > 0.0


def route(top1_score: float, tau: float, pws_mean: float, pws_low95: float) -> bool:
    """Routing helper for online prompt selection."""

    if tau <= 0:
        raise ValueError("tau must be positive")
    return top1_score < tau and pws_mean > 0.0 and pws_low95 > 0.0

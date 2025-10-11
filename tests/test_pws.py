from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from reid.pws.core import (
    PWSWeights,
    bootstrap_pws,
    compute_costs,
    compute_latency_penalty,
    compute_risk,
    decision,
    route,
)


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    rows = []
    for i in range(40):
        qid = f"q{i:03d}"
        base_ap = 0.45 + 0.01 * (i % 5)
        prompt_ap = base_ap + 0.06
        base_r1 = 1 if i % 3 == 0 else 0
        prompt_r1 = min(1, base_r1 + (1 if i % 4 == 0 else 0))
        domain = "unseen" if i % 4 == 0 else "in_domain"
        occluded = i % 5 == 0
        latency_prompt = 280.0 + (i % 6) * 12.0
        latency_base = 250.0 + (i % 6) * 8.0
        online_tokens_prompt = 210 + (i % 3) * 10
        gpu_time_prompt = 0.18 + (i % 2) * 0.02
        rows.append(
            {
                "qid": qid,
                "AP_baseline": base_ap,
                "AP_prompt": prompt_ap,
                "R1_baseline": base_r1,
                "R1_prompt": prompt_r1,
                "latency_ms_baseline": latency_base,
                "latency_ms_prompt": latency_prompt,
                "online_tokens_baseline": 0,
                "online_tokens_prompt": online_tokens_prompt,
                "gpu_time_s_baseline": 0.12,
                "gpu_time_s_prompt": gpu_time_prompt,
                "domain": domain,
                "occluded": occluded,
            }
        )
    return pd.DataFrame(rows)


def test_compute_latency_penalty() -> None:
    assert compute_latency_penalty(350.0, 300.0) == pytest.approx(50.0 / 300.0)
    assert compute_latency_penalty(250.0, 300.0) == 0.0


def test_compute_costs() -> None:
    costs = compute_costs(online_tokens=200, token_price_per_1k=0.002, monthly_offline_tokens=0, monthly_queries=1000, gpu_time_s=0.2, gpu_cost_per_hr=0.5)
    assert costs["TokenCost"] == pytest.approx(0.0004)
    assert costs["OfflineTokenCost"] == 0.0
    assert costs["GPUCost"] == pytest.approx((0.2 / 3600.0) * 0.5)


def test_compute_risk_defaults() -> None:
    score = compute_risk(0.1, 0.2, 0.3, 0.4)
    assert score == pytest.approx((0.1 + 0.2 + 0.3 + 0.4) / 4)

    weights = {"drift": 0.4, "bias": 0.2, "privacy": 0.2, "repro": 0.2}
    score_weighted = compute_risk(0.2, 0.1, 0.3, 0.4, weights)
    expected = 0.2 * 0.4 + 0.1 * 0.2 + 0.3 * 0.2 + 0.4 * 0.2
    assert score_weighted == pytest.approx(expected)


def test_bootstrap_and_decision(synthetic_df: pd.DataFrame) -> None:
    weights = PWSWeights().as_dict()
    static_inputs = {
        "sla_ms": 300.0,
        "weights": weights,
        "costs": {
            "token_price_per_1k": 0.002,
            "GPU_cost_per_hr": 0.5,
            "monthly_offline_tokens": 0.0,
            "monthly_queries": 100000,
        },
        "complexity": 2,
        "risk": {"drift": 0.05, "bias": 0.0, "privacy": 0.0, "repro": 0.0},
    }

    summary = bootstrap_pws(synthetic_df, static_inputs, n_boot=256, seed=42)
    assert summary["PWS_lower95"] < summary["PWS_mean"] < summary["PWS_upper95"]

    deploy = decision(summary)
    assert isinstance(deploy, bool)
    assert deploy is True

    routed = route(0.2, 0.5, summary["PWS_mean"], summary["PWS_lower95"])
    assert routed is True

    with pytest.raises(ValueError):
        route(0.2, 0.0, summary["PWS_mean"], summary["PWS_lower95"])

    # High score should not route even when PWS positive
    assert route(0.9, 0.5, summary["PWS_mean"], summary["PWS_lower95"]) is False



"""CLI for evaluating Prompt Worth Score (PWS)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from reid.pws.core import bootstrap_pws, decision
from reid.pws.io import (
    load_costs,
    load_weights,
    merge_logs,
    read_jsonl,
    write_csv,
    write_json,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Prompt Worth Score (PWS)")
    parser.add_argument("--logs", required=True, help="JSONL logs for the prompt arm")
    parser.add_argument("--baseline-logs", required=True, help="JSONL logs for the baseline arm")
    parser.add_argument("--prompt-type", required=True, help="Prompt strategy identifier")
    parser.add_argument("--sla-ms", type=float, required=True, help="Latency SLA in milliseconds")
    parser.add_argument("--weights", help="YAML file with PWS weights")
    parser.add_argument("--costs", required=True, help="YAML file with cost parameters")
    parser.add_argument("--complexity", type=int, default=1, help="Engineering complexity (1-5)")
    parser.add_argument("--risk-drift", type=float, default=0.0)
    parser.add_argument("--risk-bias", type=float, default=0.0)
    parser.add_argument("--risk-privacy", type=float, default=0.0)
    parser.add_argument("--risk-repro", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.5, help="Routing threshold for top-1 score")
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed")
    parser.add_argument("--out-dir", required=True, help="Output directory for artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt_df = read_jsonl(args.logs)
    baseline_df = read_jsonl(args.baseline_logs)

    merged = merge_logs(baseline_df, prompt_df)

    weights = load_weights(args.weights)
    costs = load_costs(args.costs)

    static_inputs: Dict[str, Any] = {
        "sla_ms": args.sla_ms,
        "weights": weights,
        "costs": costs,
        "complexity": args.complexity,
        "risk": {
            "drift": args.risk_drift,
            "bias": args.risk_bias,
            "privacy": args.risk_privacy,
            "repro": args.risk_repro,
        },
    }

    summary = bootstrap_pws(merged, static_inputs, n_boot=args.n_boot, seed=args.seed)
    deploy = decision(summary)

    out_dir = Path(args.out_dir)
    report_path = out_dir / "report.md"
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "decision_matrix.csv"

    write_markdown_report(summary, report_path, args.prompt_type, args.tau, deploy)
    write_json({**summary, "prompt_type": args.prompt_type, "deploy": deploy}, json_path)

    decision_rows = [{
        "PromptType": args.prompt_type,
        "ΔmAP": summary["ΔmAP"],
        "ΔR1": summary["ΔR1"],
        "ΔDG": summary["ΔDG"],
        "ΔOcc": summary["ΔOcc"],
        "LatencyPenalty": summary["LatencyPenalty"],
        "Token$/q": summary["TokenCost"],
        "GPU$/q": summary["GPUCost"],
        "OfflineToken$/q": summary["OfflineTokenCost"],
        "PWS_mean": summary["PWS_mean"],
        "PWS_lower95": summary["PWS_lower95"],
        "PWS_upper95": summary["PWS_upper95"],
        "Deploy": deploy,
    }]
    write_csv(decision_rows, csv_path)

    print(f"Prompt {args.prompt_type} → PWS_mean={summary['PWS_mean']:.2f}, deploy={deploy}")


if __name__ == "__main__":
    main()

"""Smoke test for the prompt tuner agent."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_smoke() -> None:
    script = Path(__file__).resolve().parent / "agent.py"
    subprocess.run(
        [sys.executable, str(script), "--rounds", "1", "--ensemble", "1", "--neg_rate", "0"],
        check=True,
    )


if __name__ == "__main__":
    run_smoke()

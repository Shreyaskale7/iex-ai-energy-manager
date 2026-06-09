#!/usr/bin/env python3
"""Train all ML models on RTM_Market Snapshot Excel files in the project root."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    def run_cmd(name: str, script: str, extra_args: list[str] | None = None) -> None:
        cmd = [PY, str(ROOT / "scripts" / script)]
        if extra_args:
            cmd.extend(extra_args)
        print(f"\n{'=' * 72}\n>>> {name}\n{'=' * 72}")
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        if result.returncode != 0:
            raise SystemExit(f"Step failed: {name} (exit {result.returncode})")

    steps = [
        ("1/6 Ingest RTM Market Excel files", "build_rtm_master.py", []),
        ("2/6 Build features", "build_features.py", []),
        ("3/6 Train XGBoost regressor", "train_xgboost.py", ["--trials", "25"]),
        ("4/6 Train LightGBM regressor", "train_lightgbm.py", ["--trials", "25"]),
        ("5/6 Train CatBoost regressor", "train_catboost.py", ["--trials", "20"]),
        ("6/6 Train ensemble + spike classifier", "_train_final", []),
    ]

    for name, script, args in steps:
        if script == "_train_final":
            run_cmd("6a Ensemble weights", "train_ensemble.py")
            run_cmd("6b Spike classifier", "train_spike_classifier.py")
            continue
        run_cmd(name, script, args)

    print(f"\n{'=' * 72}\nTraining pipeline complete.\n{'=' * 72}")
    print("Artifacts:")
    print("  data/processed/rtm_master.parquet")
    print("  data/features/features.parquet")
    print("  models/xgboost.pkl, lightgbm.pkl, catboost.pkl, ensemble.pkl, spike_classifier.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

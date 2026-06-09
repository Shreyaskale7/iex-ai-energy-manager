#!/usr/bin/env python3
"""Train XGBoost RTM MCP spike classifier."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.spike_classifier import RTMSpikeClassifierTrainer, configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train MCP spike classifier")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "features" / "features.parquet",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "spike_classifier.pkl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "spike_classifier_report.html",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=75.0,
        help="MCP percentile threshold for spike label",
    )
    args = parser.parse_args()

    result = RTMSpikeClassifierTrainer(
        features_path=args.features,
        model_path=args.model,
        report_path=args.report,
        spike_percentile=args.percentile,
    ).run()

    test = result.metrics["test"]
    print(f"Model saved:  {result.model_path}")
    print(f"Threshold:    {result.spike_threshold:.2f} Rs/MWh")
    print(f"Precision:    {test['precision']:.4f}")
    print(f"Recall:       {test['recall']:.4f}")
    print(f"F1:           {test['f1']:.4f}")
    print(f"ROC AUC:      {test['roc_auc']:.4f}")
    print(f"Confusion matrix (test): {test['confusion_matrix']}")
    print(f"Report:       {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

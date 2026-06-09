"""MAPE / sMAPE evaluation and HTML report for trained MCP models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.dataset import chronological_split, load_forecast_dataset
from models.ensemble import WeightedEnsemble
from models.metrics import mape, smape
from models.report import _fig_to_base64

# IEX hour 1–24: evening peak 18:00–22:59 IST (hours 18–22 inclusive)
PEAK_HOURS_IEX = frozenset(range(18, 23))
MODEL_ORDER = ("xgboost", "lightgbm", "catboost", "ensemble")


@dataclass
class MapeReportResult:
    json_path: Path
    html_path: Path
    metrics: dict[str, Any]


def _hour_from_timestamps(ts: pd.Series) -> pd.Series:
    """IEX-style hour 1–24 from block timestamps."""
    parsed = pd.to_datetime(ts, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return parsed.dt.hour + 1


def _predict_base_model(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    features = bundle["feature_names"]
    return np.asarray(model.predict(X[features]), dtype=float)


def collect_test_predictions(
    features_path: Path,
    model_dir: Path,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build test-set frame with actuals, hour labels, and model predictions."""
    X, y, timestamps, feature_names = load_forecast_dataset(features_path)
    split = chronological_split(X, y, timestamps, val_ratio=val_ratio, test_ratio=test_ratio)

    X_test = split["X_test"]
    y_test = np.asarray(split["y_test"], dtype=float)
    ts_test = split["ts_test"].reset_index(drop=True)

    frame = pd.DataFrame(
        {
            "block_timestamp": ts_test,
            "hour": _hour_from_timestamps(ts_test),
            "actual_mcp": y_test,
        }
    )
    frame["segment"] = np.where(
        frame["hour"].isin(PEAK_HOURS_IEX),
        "peak",
        "off_peak",
    )

    xgb_bundle = joblib.load(model_dir / "xgboost.pkl")
    lgbm_bundle = joblib.load(model_dir / "lightgbm.pkl")
    cat_bundle = joblib.load(model_dir / "catboost.pkl")
    ens_bundle = joblib.load(model_dir / "ensemble.pkl")

    frame["xgboost"] = _predict_base_model(xgb_bundle, X_test)
    frame["lightgbm"] = _predict_base_model(lgbm_bundle, X_test)
    frame["catboost"] = _predict_base_model(cat_bundle, X_test)

    ensemble = WeightedEnsemble(
        weights=ens_bundle["weights"],
        model_paths=ens_bundle["model_paths"],
        feature_names=ens_bundle["feature_names"],
    )
    frame["ensemble"] = ensemble.predict(X_test)

    return frame, y_test, feature_names


def _metrics_for_mask(
    frame: pd.DataFrame,
    y_col: str = "actual_mcp",
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        out[model] = {
            "mape": mape(frame[y_col].values, frame[model].values),
            "smape": smape(frame[y_col].values, frame[model].values),
            "n": int(len(frame)),
        }
    return out


def compute_mape_breakdown(frame: pd.DataFrame) -> dict[str, Any]:
    """Overall, hourly, peak, and off-peak MAPE/sMAPE for all models."""
    overall = _metrics_for_mask(frame)

    hourly: dict[str, dict[str, dict[str, float]]] = {}
    for hour, group in frame.groupby("hour", sort=True):
        hourly[str(int(hour))] = _metrics_for_mask(group)

    peak = _metrics_for_mask(frame.loc[frame["segment"] == "peak"])
    off_peak = _metrics_for_mask(frame.loc[frame["segment"] == "off_peak"])

    return {
        "overall": overall,
        "hourly": hourly,
        "peak_hour": peak,
        "off_peak": off_peak,
        "definitions": {
            "peak_hours_iex": sorted(PEAK_HOURS_IEX),
            "peak_description": "IEX hours 18–22 (18:00–22:59 IST)",
            "off_peak_description": "All other IEX hours (1–17, 23–24)",
            "hourly_description": "MAPE/sMAPE computed per IEX hour (1–24) on the holdout test set",
            "mape_formula": "mean(|actual - pred| / max(|actual|, 1)) * 100",
            "smape_formula": "mean(2|actual - pred| / max(|actual| + |pred|, 1)) * 100",
        },
    }


def _plot_overall_comparison(overall: dict[str, dict[str, float]]) -> str:
    models = list(MODEL_ORDER)
    mape_vals = [overall[m]["mape"] for m in models]
    smape_vals = [overall[m]["smape"] for m in models]
    labels = [m.replace("_", " ").title() for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0f1419")
    colors = ["#f59e0b", "#38bdf8", "#34d399", "#a78bfa"]
    for ax, vals, title in zip(
        axes,
        [mape_vals, smape_vals],
        ["Overall MAPE (%)", "Overall sMAPE (%)"],
    ):
        ax.set_facecolor("#1a2332")
        ax.bar(labels, vals, color=colors, alpha=0.9)
        ax.set_title(title, color="#e6edf3")
        ax.tick_params(colors="#c9d1d9", axis="x", rotation=15)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", color="#c9d1d9", fontsize=9)
    fig.suptitle("Test Set — Percentage Error by Model", color="#e6edf3", y=1.02)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_hourly_heatmap(hourly: dict[str, dict[str, dict[str, float]]], metric: str) -> str:
    hours = sorted(int(h) for h in hourly.keys())
    data = np.array([[hourly[str(h)][m][metric] for h in hours] for m in MODEL_ORDER])

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([str(h) for h in hours], color="#c9d1d9")
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([m.title() for m in MODEL_ORDER], color="#c9d1d9")
    ax.set_xlabel("IEX Hour (1–24)", color="#c9d1d9")
    ax.set_title(f"Hourly {metric.upper()} (%) — Test Set", color="#e6edf3")
    plt.colorbar(im, ax=ax, label=f"{metric.upper()} %")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_segment_bars(
    peak: dict[str, dict[str, float]],
    off_peak: dict[str, dict[str, float]],
    metric: str,
) -> str:
    models = list(MODEL_ORDER)
    x = np.arange(len(models))
    width = 0.35
    peak_vals = [peak[m][metric] for m in models]
    off_vals = [off_peak[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")
    ax.bar(x - width / 2, peak_vals, width, label="Peak (18–22)", color="#fb7185", alpha=0.9)
    ax.bar(x + width / 2, off_vals, width, label="Off-peak", color="#38bdf8", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in models], color="#c9d1d9")
    ax.set_ylabel(f"{metric.upper()} (%)", color="#c9d1d9")
    ax.set_title(f"Peak vs Off-Peak {metric.upper()} — Test Set", color="#e6edf3")
    ax.legend(facecolor="#1a2332", labelcolor="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _metrics_table_rows(metrics: dict[str, dict[str, float]]) -> str:
    rows = []
    for model in MODEL_ORDER:
        m = metrics[model]
        rows.append(
            f"<tr><td>{model}</td><td>{m['mape']:.2f}%</td><td>{m['smape']:.2f}%</td><td>{m['n']:,}</td></tr>"
        )
    return "\n".join(rows)


def _hourly_table_html(hourly: dict[str, dict[str, dict[str, float]]]) -> str:
    hours = sorted(int(h) for h in hourly.keys())
    header = "<tr><th>Hour</th>" + "".join(
        f"<th colspan='2'>{m.title()}</th>" for m in MODEL_ORDER
    ) + "</tr>"
    sub = "<tr><th></th>" + "".join("<th>MAPE</th><th>sMAPE</th>" for _ in MODEL_ORDER) + "</tr>"
    body = []
    for h in hours:
        cells = [f"<td>{h}</td>"]
        for model in MODEL_ORDER:
            cells.append(f"<td>{hourly[str(h)][model]['mape']:.2f}%</td>")
            cells.append(f"<td>{hourly[str(h)][model]['smape']:.2f}%</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='metrics'>{header}{sub}{''.join(body)}</table>"


def generate_mape_html_report(
    breakdown: dict[str, Any],
    test_rows: int,
    output_path: Path,
    charts: dict[str, str],
) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    defs = breakdown["definitions"]
    overall = breakdown["overall"]
    peak = breakdown["peak_hour"]
    off_peak = breakdown["off_peak"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>IEX RTM MCP — MAPE / sMAPE Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0f1419; color: #e6edf3; margin: 2rem; }}
    h1, h2 {{ color: #e6edf3; }}
    p, li {{ color: #c9d1d9; line-height: 1.5; }}
    table.metrics {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
    table.metrics th, table.metrics td {{ border: 1px solid #30363d; padding: 0.5rem 0.75rem; text-align: right; }}
    table.metrics th {{ background: #1a2332; color: #8b949e; }}
    table.metrics td:first-child, table.metrics th:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border-radius: 8px; margin: 1rem 0; }}
    .note {{ background: #1a2332; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b; }}
  </style>
</head>
<body>
  <h1>MCP Forecast — MAPE &amp; sMAPE Report</h1>
  <p>Generated: {generated} | Holdout test rows: {test_rows:,}</p>

  <div class="note">
    <strong>Definitions</strong>
    <ul>
      <li><strong>MAPE:</strong> {defs['mape_formula']}</li>
      <li><strong>sMAPE:</strong> {defs['smape_formula']}</li>
      <li><strong>Peak hours:</strong> {defs['peak_description']} (IEX hours {defs['peak_hours_iex']})</li>
      <li><strong>Off-peak:</strong> {defs['off_peak_description']}</li>
      <li><strong>Hourly:</strong> {defs['hourly_description']}</li>
    </ul>
  </div>

  <h2>Overall (Test Set)</h2>
  <table class="metrics">
    <tr><th>Model</th><th>MAPE</th><th>sMAPE</th><th>N</th></tr>
    {_metrics_table_rows(overall)}
  </table>
  <img src="data:image/png;base64,{charts['overall']}" alt="Overall comparison"/>

  <h2>Peak vs Off-Peak</h2>
  <h3>Peak hours (18:00–22:59 IST)</h3>
  <table class="metrics">
    <tr><th>Model</th><th>MAPE</th><th>sMAPE</th><th>N</th></tr>
    {_metrics_table_rows(peak)}
  </table>
  <h3>Off-peak</h3>
  <table class="metrics">
    <tr><th>Model</th><th>MAPE</th><th>sMAPE</th><th>N</th></tr>
    {_metrics_table_rows(off_peak)}
  </table>
  <img src="data:image/png;base64,{charts['segment_mape']}" alt="Peak vs off-peak MAPE"/>
  <img src="data:image/png;base64,{charts['segment_smape']}" alt="Peak vs off-peak sMAPE"/>

  <h2>Hourly Breakdown</h2>
  {_hourly_table_html(breakdown['hourly'])}
  <img src="data:image/png;base64,{charts['hourly_mape']}" alt="Hourly MAPE heatmap"/>
  <img src="data:image/png;base64,{charts['hourly_smape']}" alt="Hourly sMAPE heatmap"/>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


class MapeReportGenerator:
    def __init__(
        self,
        features_path: Path | str = "data/features/features.parquet",
        model_dir: Path | str = "models",
        json_path: Path | str = "reports/mape_metrics.json",
        html_path: Path | str = "reports/mape_report.html",
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ) -> None:
        self.features_path = Path(features_path)
        self.model_dir = Path(model_dir)
        self.json_path = Path(json_path)
        self.html_path = Path(html_path)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def run(self) -> MapeReportResult:
        frame, _, _ = collect_test_predictions(
            self.features_path,
            self.model_dir,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )
        breakdown = compute_mape_breakdown(frame)
        breakdown["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        breakdown["test_rows"] = int(len(frame))

        charts = {
            "overall": _plot_overall_comparison(breakdown["overall"]),
            "segment_mape": _plot_segment_bars(breakdown["peak_hour"], breakdown["off_peak"], "mape"),
            "segment_smape": _plot_segment_bars(breakdown["peak_hour"], breakdown["off_peak"], "smape"),
            "hourly_mape": _plot_hourly_heatmap(breakdown["hourly"], "mape"),
            "hourly_smape": _plot_hourly_heatmap(breakdown["hourly"], "smape"),
        }

        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(json.dumps(breakdown, indent=2), encoding="utf-8")
        generate_mape_html_report(breakdown, len(frame), self.html_path, charts)

        return MapeReportResult(
            json_path=self.json_path,
            html_path=self.html_path,
            metrics=breakdown,
        )

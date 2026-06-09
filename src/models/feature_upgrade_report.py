"""Before/after report for microstructure feature upgrade and model retrain."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from models.compare import load_model_metrics
from models.report import _fig_to_base64, plot_feature_importance

MODEL_NAMES = ("xgboost", "lightgbm", "catboost", "ensemble")
METRIC_KEYS = ("mae", "rmse", "mape", "r2")


def extract_importance(model_path: Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["feature_names"]
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        scores = model.get_feature_importance()
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    imp = pd.DataFrame({"feature": features, "importance": scores})
    total = imp["importance"].sum()
    if total > 0:
        imp["importance"] = imp["importance"] / total
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def snapshot_baseline(
    model_dir: Path,
    snapshot_path: Path,
    label: str = "baseline_before_microstructure",
    source_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist metrics and feature importance from current models on disk."""
    baseline_dir = model_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "label": label,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {},
        "importance": {},
        "feature_counts": {},
    }

    read_dir = source_dir or model_dir
    for name in ("xgboost", "lightgbm", "catboost"):
        src = read_dir / f"{name}.pkl"
        if not src.exists():
            continue
        shutil.copy2(src, baseline_dir / f"{name}.pkl")
        json_path = read_dir / f"{name}_metrics.json"
        metrics = _flatten_model_metrics(load_model_metrics(src, json_path))
        if metrics:
            payload["metrics"][name] = metrics
        imp = extract_importance(src)
        payload["importance"][name] = imp.to_dict(orient="records")
        payload["feature_counts"][name] = len(imp)

    ens_path = read_dir / "ensemble.pkl"
    if ens_path.exists():
        shutil.copy2(ens_path, baseline_dir / "ensemble.pkl")
        metrics = _flatten_model_metrics(load_model_metrics(ens_path, read_dir / "ensemble_metrics.json"))
        if metrics:
            payload["metrics"]["ensemble"] = metrics

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_snapshot_metrics(snapshot: dict[str, Any], model: str, split: str = "test") -> dict[str, float]:
    return snapshot["metrics"][model][split]


def _metrics_comparison_table(
    before: dict[str, Any],
    after: dict[str, Any],
    split: str = "test",
) -> str:
    rows = []
    for model in MODEL_NAMES:
        if model not in before.get("metrics", {}) or model not in after.get("metrics", {}):
            continue
        b = before["metrics"][model][split]
        a = after["metrics"][model][split]
        for key in METRIC_KEYS:
            bv, av = b[key], a[key]
            if key == "r2":
                delta = av - bv
                better = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            else:
                delta = av - bv
                better = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
            rows.append(
                f"<tr><td>{model}</td><td>{key.upper()}</td>"
                f"<td>{bv:.4f}</td><td>{av:.4f}</td><td>{delta:+.4f}</td><td>{better}</td></tr>"
            )
    return "\n".join(rows)


def plot_metrics_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    split: str = "test",
) -> str:
    models = [m for m in MODEL_NAMES if m in before.get("metrics", {}) and m in after.get("metrics", {})]
    metrics = ["mae", "rmse", "r2"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#0f1419")
    colors_before, colors_after = "#64748b", "#38bdf8"

    for ax, metric in zip(axes, metrics):
        ax.set_facecolor("#1a2332")
        b_vals = [before["metrics"][m][split][metric] for m in models]
        a_vals = [after["metrics"][m][split][metric] for m in models]
        x = range(len(models))
        w = 0.35
        ax.bar([i - w / 2 for i in x], b_vals, w, label="Before", color=colors_before, alpha=0.9)
        ax.bar([i + w / 2 for i in x], a_vals, w, label="After", color=colors_after, alpha=0.9)
        ax.set_xticks(list(x))
        ax.set_xticklabels([m.title() for m in models], color="#c9d1d9", rotation=15)
        ax.set_title(metric.upper(), color="#e6edf3")
        ax.tick_params(colors="#c9d1d9")
        ax.legend(facecolor="#1a2332", labelcolor="#c9d1d9", fontsize=8)

    fig.suptitle(f"Model metrics — {split} set (before vs after microstructure features)", color="#e6edf3")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_importance_comparison(
    before_imp: pd.DataFrame,
    after_imp: pd.DataFrame,
    title: str,
    top_n: int = 20,
) -> str:
    """Side-by-side top feature importance (normalized)."""
    b_top = before_imp.head(top_n).iloc[::-1]
    a_top = after_imp.head(top_n).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#0f1419")
    for ax, data, subtitle, color in zip(
        axes,
        [b_top, a_top],
        ["Before", "After"],
        ["#64748b", "#f59e0b"],
    ):
        ax.set_facecolor("#1a2332")
        ax.barh(data["feature"], data["importance"], color=color, alpha=0.9)
        ax.set_title(f"{subtitle} — top {top_n}", color="#e6edf3")
        ax.tick_params(colors="#c9d1d9")
    fig.suptitle(title, color="#e6edf3", fontsize=12)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_new_feature_ranking(after_imp: pd.DataFrame, new_features: list[str]) -> str:
    subset = after_imp[after_imp["feature"].isin(new_features)].sort_values("importance", ascending=True)
    if subset.empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")
    ax.barh(subset["feature"], subset["importance"], color="#34d399", alpha=0.9)
    ax.set_title("New microstructure features — XGBoost importance (after)", color="#e6edf3")
    ax.tick_params(colors="#c9d1d9")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _flatten_model_metrics(raw: dict[str, Any] | None) -> dict[str, dict[str, float]] | None:
    """Normalize bundle/json metrics to split -> scalar metrics."""
    if not raw:
        return None
    if "test" in raw and isinstance(raw["test"], dict):
        if "mae" in raw["test"]:
            return raw
        if "ensemble" in raw["test"]:
            return {
                "test": raw["test"]["ensemble"],
                "validation": raw["validation"]["ensemble"],
                "train": raw.get("train", {}).get("ensemble") if isinstance(raw.get("train"), dict) else {},
            }
    return raw


def build_after_snapshot(model_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": "after_microstructure_features",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {},
        "importance": {},
        "feature_counts": {},
    }
    for name in ("xgboost", "lightgbm", "catboost"):
        path = model_dir / f"{name}.pkl"
        if not path.exists():
            continue
        json_path = model_dir / f"{name}_metrics.json"
        metrics = _flatten_model_metrics(load_model_metrics(path, json_path))
        if metrics:
            payload["metrics"][name] = metrics
        imp = extract_importance(path)
        payload["importance"][name] = imp.to_dict(orient="records")
        payload["feature_counts"][name] = len(imp)
    ens_metrics = _flatten_model_metrics(
        load_model_metrics(model_dir / "ensemble.pkl", model_dir / "ensemble_metrics.json")
    )
    if ens_metrics:
        payload["metrics"]["ensemble"] = ens_metrics
    return payload


def _normalize_snapshot_metrics(snapshot: dict[str, Any]) -> dict[str, Any]:
    out = dict(snapshot)
    metrics = {}
    for model, raw in snapshot.get("metrics", {}).items():
        flat = _flatten_model_metrics(raw)
        if flat:
            metrics[model] = flat
    out["metrics"] = metrics
    return out


def generate_feature_upgrade_report(
    before: dict[str, Any],
    after: dict[str, Any],
    output_path: Path,
    new_features: list[str],
) -> Path:
    before = _normalize_snapshot_metrics(before)
    after = _normalize_snapshot_metrics(after)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metrics_chart = plot_metrics_delta(before, after, split="test")

    xgb_before = pd.DataFrame(before["importance"]["xgboost"])
    xgb_after = pd.DataFrame(after["importance"]["xgboost"])
    imp_chart = plot_importance_comparison(
        xgb_before,
        xgb_after,
        "XGBoost feature importance (normalized)",
    )
    new_feat_chart = plot_new_feature_ranking(xgb_after, new_features)
    lgbm_imp_chart = ""
    if "lightgbm" in before["importance"] and "lightgbm" in after["importance"]:
        lgbm_imp_chart = plot_importance_comparison(
            pd.DataFrame(before["importance"]["lightgbm"]),
            pd.DataFrame(after["importance"]["lightgbm"]),
            "LightGBM feature importance (normalized)",
        )

    fc_before = before.get("feature_counts", {}).get("xgboost", "?")
    fc_after = after.get("feature_counts", {}).get("xgboost", "?")

    new_rows = ""
    for feat in new_features:
        row = xgb_after[xgb_after["feature"] == feat]
        rank = int(xgb_after.index.get_loc(row.index[0]) + 1) if not row.empty else "-"
        imp = float(row["importance"].iloc[0]) if not row.empty else 0.0
        new_rows += f"<tr><td>{feat}</td><td>{imp:.4f}</td><td>{rank}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Microstructure Feature Upgrade Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0f1419; color: #e6edf3; margin: 2rem; max-width: 1200px; }}
    h1, h2 {{ color: #e6edf3; }}
    p, li, td, th {{ color: #c9d1d9; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #30363d; padding: 0.5rem 0.75rem; text-align: right; }}
    th {{ background: #1a2332; text-align: left; }}
    img {{ max-width: 100%; border-radius: 8px; margin: 1rem 0; }}
  </style>
</head>
<body>
  <h1>RTM Microstructure Feature Upgrade</h1>
  <p>Generated: {generated}</p>
  <p>Features: {fc_before} (before) → {fc_after} (after)</p>

  <h2>New microstructure features</h2>
  <ul>
    <li><strong>demand_supply_ratio</strong> — purchase_bid / sell_bid</li>
    <li><strong>market_imbalance</strong> — (purchase − sell) / (purchase + sell)</li>
    <li><strong>bid_spread</strong> — purchase_bid − sell_bid</li>
    <li><strong>relative_volume</strong> — volume / rolling_mean(volume, 96)</li>
    <li><strong>mcp_momentum</strong> — mcp_lag_1 − mcp_lag_4</li>
    <li><strong>mcp_velocity</strong> — mcp_lag_1 − mcp_lag_2</li>
    <li><strong>mcp_acceleration</strong> — (mcp_lag_1 − mcp_lag_2) − (mcp_lag_2 − mcp_lag_4)</li>
    <li><strong>rolling_bid_ratio_4/8/96</strong> — rolling sum(purchase) / rolling sum(sell)</li>
  </ul>

  <h2>Test metrics — before vs after</h2>
  <table>
    <tr><th>Model</th><th>Metric</th><th>Before</th><th>After</th><th>Delta</th><th></th></tr>
    {_metrics_comparison_table(before, after)}
  </table>
  <img src="data:image/png;base64,{metrics_chart}" alt="Metrics comparison"/>

  <h2>Feature importance — XGBoost</h2>
  <img src="data:image/png;base64,{imp_chart}" alt="XGBoost importance"/>
  {f'<img src="data:image/png;base64,{new_feat_chart}" alt="New features"/>' if new_feat_chart else ''}

  <h2>New features in XGBoost (after)</h2>
  <table>
    <tr><th>Feature</th><th>Importance</th><th>Rank</th></tr>
    {new_rows}
  </table>

  <h2>Feature importance — LightGBM</h2>
  {f'<img src="data:image/png;base64,{lgbm_imp_chart}" alt="LightGBM importance"/>' if lgbm_imp_chart else '<p>Not available</p>'}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path

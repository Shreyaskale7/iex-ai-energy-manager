"""HTML training report for XGBoost MCP model."""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0f1419")
    for ax in axes:
        ax.set_facecolor("#1a2332")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")

    sample = min(500, len(y_true))
    axes[0].plot(y_true[:sample], label="Actual", color="#38bdf8", linewidth=1.2)
    axes[0].plot(y_pred[:sample], label="Predicted", color="#f59e0b", linewidth=1.2, alpha=0.85)
    axes[0].set_title(f"{title} — Time Series (sample)")
    axes[0].set_xlabel("Observation index")
    axes[0].set_ylabel("MCP (Rs/MWh)")
    axes[0].legend()

    axes[1].scatter(y_true, y_pred, alpha=0.35, s=12, color="#34d399", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "--", color="#fb7185", linewidth=1)
    axes[1].set_title(f"{title} — Actual vs Predicted")
    axes[1].set_xlabel("Actual MCP")
    axes[1].set_ylabel("Predicted MCP")

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_feature_importance(importance: pd.DataFrame, title: str, color: str = "#f59e0b") -> str:
    top = importance.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")
    ax.barh(top["feature"], top["importance"], color=color, alpha=0.9)
    ax.set_title(title, color="#e6edf3")
    ax.tick_params(colors="#c9d1d9")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_metric_comparison(
    xgb_metrics: dict[str, dict[str, float]],
    lgbm_metrics: dict[str, dict[str, float]],
    split: str = "test",
) -> str:
    names = ["MAE", "RMSE", "MAPE", "R2"]
    keys = ["mae", "rmse", "mape", "r2"]
    x_vals = [xgb_metrics[split][k] for k in keys]
    l_vals = [lgbm_metrics[split][k] for k in keys]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.patch.set_facecolor("#0f1419")
    for ax, name, x_v, l_v in zip(axes, names, x_vals, l_vals):
        ax.set_facecolor("#1a2332")
        ax.bar(["XGBoost", "LightGBM"], [x_v, l_v], color=["#f59e0b", "#38bdf8"], alpha=0.9)
        ax.set_title(name, color="#e6edf3")
        ax.tick_params(colors="#c9d1d9")
    fig.suptitle(f"Test Set Metric Comparison ({split})", color="#e6edf3", y=1.02)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_model_predictions_overlay(
    y_true: np.ndarray,
    xgb_pred: np.ndarray,
    lgbm_pred: np.ndarray,
) -> str:
    sample = min(400, len(y_true))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")
    ax.plot(y_true[:sample], label="Actual", color="#e6edf3", linewidth=1.2)
    ax.plot(xgb_pred[:sample], label="XGBoost", color="#f59e0b", linewidth=1.0, alpha=0.85)
    ax.plot(lgbm_pred[:sample], label="LightGBM", color="#38bdf8", linewidth=1.0, alpha=0.85)
    ax.set_title("Holdout Predictions — XGBoost vs LightGBM", color="#e6edf3")
    ax.set_xlabel("Index")
    ax.set_ylabel("MCP (Rs/MWh)")
    ax.legend()
    ax.tick_params(colors="#c9d1d9")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_shap_summary(shap_values: np.ndarray, feature_names: list[str], title: str | None = None) -> str:
    import shap

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1419")
    shap.summary_plot(
        shap_values,
        features=pd.DataFrame(shap_values, columns=feature_names),
        show=False,
        max_display=20,
    )
    plt.title(title or "SHAP Summary (next 15-min MCP)", color="#e6edf3")
    fig = plt.gcf()
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_xgb_html_report(
    output_path: Path,
    metrics: dict[str, dict[str, float]],
    best_params: dict[str, Any],
    optuna_summary: dict[str, Any],
    importance: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
    split_info: dict[str, Any],
) -> Path:
    pred_img = plot_predictions(y_true, y_pred, "Holdout Test")
    fi_img = plot_feature_importance(importance, "XGBoost Feature Importance", "#f59e0b")
    shap_img = plot_shap_summary(shap_values, feature_names, "SHAP — XGBoost (next 15-min MCP)")

    metrics_rows = ""
    for split_name, split_metrics in metrics.items():
        metrics_rows += (
            f"<tr><td>{split_name}</td>"
            f"<td>{split_metrics['mae']:.3f}</td>"
            f"<td>{split_metrics['rmse']:.3f}</td>"
            f"<td>{split_metrics['mape']:.3f}%</td>"
            f"<td>{split_metrics['r2']:.4f}</td></tr>"
        )

    fi_table_rows = "".join(
        f"<tr><td>{row.feature}</td><td>{row.importance:.6f}</td></tr>"
        for row in importance.head(15).itertuples()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>XGBoost RTM MCP Forecast Report</title>
  <style>
    body {{ font-family:Segoe UI,Arial,sans-serif; background:#0f1419; color:#e6edf3; margin:24px; }}
    h1,h2 {{ color:#f59e0b; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0 24px; }}
    th,td {{ border:1px solid #2d3a4a; padding:8px 12px; text-align:left; }}
    th {{ background:#1a2332; }}
    tr:nth-child(even) {{ background:#151d28; }}
    .card {{ background:#1a2332; border-radius:8px; padding:16px; margin-bottom:20px; }}
    img {{ max-width:100%; border-radius:6px; margin-top:8px; }}
    pre {{ background:#151d28; padding:12px; overflow:auto; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>XGBoost — Next 15-Minute MCP Forecast</h1>
  <p>Generated (UTC): {datetime.now(timezone.utc).isoformat()}</p>

  <div class="card">
    <h2>Objective</h2>
    <p>Predict <strong>next block MCP (Rs/MWh)</strong> — one 15-minute step ahead.</p>
    <p>Training rows: {split_info.get("train_rows", 0):,} |
       Validation: {split_info.get("val_rows", 0):,} |
       Holdout test: {split_info.get("test_rows", 0):,}</p>
  </div>

  <div class="card">
    <h2>Metrics</h2>
    <table>
      <tr><th>Split</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>
      {metrics_rows}
    </table>
  </div>

  <div class="card">
    <h2>Optuna Tuning</h2>
    <pre>{json.dumps(optuna_summary, indent=2)}</pre>
    <h3>Best Hyperparameters</h3>
    <pre>{json.dumps(best_params, indent=2)}</pre>
  </div>

  <div class="card">
    <h2>Holdout Predictions</h2>
    <img src="data:image/png;base64,{pred_img}" alt="Predictions"/>
  </div>

  <div class="card">
    <h2>Feature Importance</h2>
    <img src="data:image/png;base64,{fi_img}" alt="Feature Importance"/>
    <table>
      <tr><th>Feature</th><th>Importance</th></tr>
      {fi_table_rows}
    </table>
  </div>

  <div class="card">
    <h2>SHAP Explainability</h2>
    <p>Computed on a stratified sample of holdout rows (n={shap_values.shape[0]:,}).</p>
    <img src="data:image/png;base64,{shap_img}" alt="SHAP Summary"/>
  </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_lgbm_html_report(
    output_path: Path,
    metrics: dict[str, dict[str, float]],
    best_params: dict[str, Any],
    optuna_summary: dict[str, Any],
    importance: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
    split_info: dict[str, Any],
) -> Path:
    pred_img = plot_predictions(y_true, y_pred, "LightGBM Holdout Test")
    fi_img = plot_feature_importance(importance, "LightGBM Feature Importance (Split)", "#38bdf8")
    shap_img = plot_shap_summary(shap_values, feature_names, "SHAP — LightGBM (next 15-min MCP)")

    metrics_rows = ""
    for split_name, split_metrics in metrics.items():
        metrics_rows += (
            f"<tr><td>{split_name}</td>"
            f"<td>{split_metrics['mae']:.3f}</td>"
            f"<td>{split_metrics['rmse']:.3f}</td>"
            f"<td>{split_metrics['mape']:.3f}%</td>"
            f"<td>{split_metrics['r2']:.4f}</td></tr>"
        )

    fi_table_rows = "".join(
        f"<tr><td>{row.feature}</td><td>{row.importance:.6f}</td></tr>"
        for row in importance.head(15).itertuples()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>LightGBM RTM MCP Forecast Report</title>
  <style>
    body {{ font-family:Segoe UI,Arial,sans-serif; background:#0f1419; color:#e6edf3; margin:24px; }}
    h1,h2 {{ color:#38bdf8; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0 24px; }}
    th,td {{ border:1px solid #2d3a4a; padding:8px 12px; text-align:left; }}
    th {{ background:#1a2332; }}
    tr:nth-child(even) {{ background:#151d28; }}
    .card {{ background:#1a2332; border-radius:8px; padding:16px; margin-bottom:20px; }}
    img {{ max-width:100%; border-radius:6px; margin-top:8px; }}
    pre {{ background:#151d28; padding:12px; overflow:auto; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>LightGBM — Next 15-Minute MCP Forecast</h1>
  <p>Generated (UTC): {datetime.now(timezone.utc).isoformat()}</p>

  <div class="card">
    <h2>Objective</h2>
    <p>Predict <strong>next block MCP (Rs/MWh)</strong> — one 15-minute step ahead.</p>
    <p>Training rows: {split_info.get("train_rows", 0):,} |
       Validation: {split_info.get("val_rows", 0):,} |
       Holdout test: {split_info.get("test_rows", 0):,}</p>
  </div>

  <div class="card">
    <h2>Metrics</h2>
    <table>
      <tr><th>Split</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>
      {metrics_rows}
    </table>
  </div>

  <div class="card">
    <h2>Optuna Tuning</h2>
    <pre>{json.dumps(optuna_summary, indent=2)}</pre>
    <h3>Best Hyperparameters</h3>
    <pre>{json.dumps(best_params, indent=2)}</pre>
  </div>

  <div class="card">
    <h2>Holdout Predictions</h2>
    <img src="data:image/png;base64,{pred_img}" alt="Predictions"/>
  </div>

  <div class="card">
    <h2>Feature Importance</h2>
    <img src="data:image/png;base64,{fi_img}" alt="Feature Importance"/>
    <table>
      <tr><th>Feature</th><th>Importance</th></tr>
      {fi_table_rows}
    </table>
  </div>

  <div class="card">
    <h2>SHAP Explainability</h2>
    <p>Sample size: {shap_values.shape[0]:,} holdout rows.</p>
    <img src="data:image/png;base64,{shap_img}" alt="SHAP Summary"/>
  </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_comparison_report(
    output_path: Path,
    xgb_metrics: dict[str, dict[str, float]],
    lgbm_metrics: dict[str, dict[str, float]],
    comparison: dict[str, Any],
    y_true: np.ndarray | None = None,
    xgb_pred: np.ndarray | None = None,
    lgbm_pred: np.ndarray | None = None,
) -> Path:
    metric_chart = plot_metric_comparison(xgb_metrics, lgbm_metrics, split=comparison.get("split", "test"))
    overlay_img = ""
    if y_true is not None and xgb_pred is not None and lgbm_pred is not None:
        overlay_img = plot_model_predictions_overlay(y_true, xgb_pred, lgbm_pred)

    def rows_for_split(split: str) -> str:
        rows = ""
        for model_name, m in [("xgboost", xgb_metrics[split]), ("lightgbm", lgbm_metrics[split])]:
            rows += (
                f"<tr><td>{model_name}</td><td>{split}</td>"
                f"<td>{m['mae']:.3f}</td><td>{m['rmse']:.3f}</td>"
                f"<td>{m['mape']:.3f}%</td><td>{m['r2']:.4f}</td></tr>"
            )
        return rows

    comp_rows = ""
    for metric, info in comparison["metrics"].items():
        comp_rows += (
            f"<tr><td>{metric.upper()}</td>"
            f"<td>{info['xgboost']:.4f}</td>"
            f"<td>{info['lightgbm']:.4f}</td>"
            f"<td>{info['delta_lgbm_minus_xgb']:+.4f}</td>"
            f"<td><strong>{info['winner']}</strong></td></tr>"
        )

    overlay_section = ""
    if overlay_img:
        overlay_section = f"""
  <div class="card">
    <h2>Prediction Overlay (Holdout Sample)</h2>
    <img src="data:image/png;base64,{overlay_img}" alt="Prediction Overlay"/>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>XGBoost vs LightGBM — Performance Comparison</title>
  <style>
    body {{ font-family:Segoe UI,Arial,sans-serif; background:#0f1419; color:#e6edf3; margin:24px; }}
    h1 {{ color:#a78bfa; }}
    h2 {{ color:#38bdf8; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0 24px; }}
    th,td {{ border:1px solid #2d3a4a; padding:8px 12px; text-align:left; }}
    th {{ background:#1a2332; }}
    .card {{ background:#1a2332; border-radius:8px; padding:16px; margin-bottom:20px; }}
    .winner {{ font-size:1.2em; color:#34d399; }}
    img {{ max-width:100%; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>Model Performance Comparison</h1>
  <p>Generated (UTC): {datetime.now(timezone.utc).isoformat()}</p>
  <p class="winner">Overall winner (test split): <strong>{comparison['overall_winner'].upper()}</strong>
     — score XGB {comparison['score']['xgboost']} : LGBM {comparison['score']['lightgbm']}</p>

  <div class="card">
    <h2>Test Set Metrics</h2>
    <table>
      <tr><th>Model</th><th>Split</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>
      {rows_for_split("test")}
    </table>
    <img src="data:image/png;base64,{metric_chart}" alt="Metric Comparison"/>
  </div>

  <div class="card">
    <h2>Head-to-Head (Test)</h2>
    <table>
      <tr><th>Metric</th><th>XGBoost</th><th>LightGBM</th><th>Δ (LGBM−XGB)</th><th>Winner</th></tr>
      {comp_rows}
    </table>
  </div>

  <div class="card">
    <h2>Validation & Train Summary</h2>
    <table>
      <tr><th>Model</th><th>Split</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>
      {rows_for_split("validation")}
      {rows_for_split("train")}
    </table>
  </div>
  {overlay_section}
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_catboost_html_report(
    output_path: Path,
    metrics: dict[str, dict[str, float]],
    best_params: dict[str, Any],
    optuna_summary: dict[str, Any],
    importance: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
    split_info: dict[str, Any],
) -> Path:
    pred_img = plot_predictions(y_true, y_pred, "CatBoost Holdout Test")
    fi_img = plot_feature_importance(importance, "CatBoost Feature Importance", "#34d399")
    shap_img = plot_shap_summary(shap_values, feature_names, "SHAP — CatBoost (next 15-min MCP)")

    metrics_rows = "".join(
        f"<tr><td>{split_name}</td><td>{m['mae']:.3f}</td><td>{m['rmse']:.3f}</td>"
        f"<td>{m['mape']:.3f}%</td><td>{m['r2']:.4f}</td></tr>"
        for split_name, m in metrics.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"/><title>CatBoost RTM MCP Report</title>
<style>
body{{font-family:Segoe UI,sans-serif;background:#0f1419;color:#e6edf3;margin:24px}}
h1,h2{{color:#34d399}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #2d3a4a;padding:8px}}
.card{{background:#1a2332;border-radius:8px;padding:16px;margin-bottom:20px}} img{{max-width:100%}}
</style></head>
<body>
<h1>CatBoost — Next 15-Minute MCP</h1>
<p>Generated: {datetime.now(timezone.utc).isoformat()}</p>
<div class="card"><h2>Metrics</h2><table>
<tr><th>Split</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>{metrics_rows}
</table></div>
<div class="card"><h2>Optuna</h2><pre>{json.dumps(optuna_summary, indent=2)}</pre>
<pre>{json.dumps(best_params, indent=2)}</pre></div>
<div class="card"><img src="data:image/png;base64,{pred_img}"/></div>
<div class="card"><img src="data:image/png;base64,{fi_img}"/></div>
<div class="card"><img src="data:image/png;base64,{shap_img}"/></div>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def plot_ensemble_metrics_comparison(
    metrics: dict[str, dict[str, dict[str, float]]],
    split: str = "test",
) -> str:
    models = ["xgboost", "lightgbm", "catboost", "ensemble"]
    labels = ["XGBoost", "LightGBM", "CatBoost", "Ensemble"]
    keys = ["mae", "rmse", "mape", "r2"]
    titles = ["MAE", "RMSE", "MAPE", "R²"]
    colors = ["#f59e0b", "#38bdf8", "#34d399", "#a78bfa"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.patch.set_facecolor("#0f1419")
    for ax, title, key in zip(axes, titles, keys):
        ax.set_facecolor("#1a2332")
        vals = [metrics[split][m][key] for m in models]
        ax.bar(labels, vals, color=colors, alpha=0.9)
        ax.set_title(title, color="#e6edf3")
        ax.tick_params(colors="#c9d1d9", axis="x", rotation=15)
    fig.suptitle(f"All Models — {split.title()} Split", color="#e6edf3", y=1.02)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_ensemble_weights(weights: dict[str, float]) -> str:
    names = list(weights.keys())
    vals = [weights[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")
    ax.bar(names, vals, color=["#f59e0b", "#38bdf8", "#34d399"][: len(names)], alpha=0.9)
    ax.set_title("Optimized Ensemble Weights (Validation)", color="#e6edf3")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#c9d1d9")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", color="#e6edf3")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_four_model_overlay(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    ensemble_pred: np.ndarray,
) -> str:
    sample = min(400, len(y_true))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")
    ax.plot(y_true[:sample], label="Actual", color="#e6edf3", linewidth=1.3)
    ax.plot(preds["xgboost"][:sample], label="XGBoost", alpha=0.7, linewidth=0.9)
    ax.plot(preds["lightgbm"][:sample], label="LightGBM", alpha=0.7, linewidth=0.9)
    ax.plot(preds["catboost"][:sample], label="CatBoost", alpha=0.7, linewidth=0.9)
    ax.plot(ensemble_pred[:sample], label="Ensemble", color="#a78bfa", linewidth=1.4)
    ax.legend()
    ax.set_title("Holdout Predictions — Base Models vs Ensemble", color="#e6edf3")
    ax.tick_params(colors="#c9d1d9")
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_ensemble_comparison_report(
    output_path: Path,
    weights: dict[str, float],
    metrics: dict[str, dict[str, dict[str, float]]],
    y_test: np.ndarray,
    test_preds: dict[str, np.ndarray],
    test_ensemble: np.ndarray,
    overall_winner: str,
) -> Path:
    chart_img = plot_ensemble_metrics_comparison(metrics, "test")
    val_chart = plot_ensemble_metrics_comparison(metrics, "validation")
    weights_img = plot_ensemble_weights(weights)
    overlay_img = plot_four_model_overlay(y_test, test_preds, test_ensemble)

    def metric_rows(split: str) -> str:
        rows = ""
        for model in ("xgboost", "lightgbm", "catboost", "ensemble"):
            m = metrics[split][model]
            rows += (
                f"<tr><td>{model}</td><td>{m['mae']:.3f}</td><td>{m['rmse']:.3f}</td>"
                f"<td>{m['mape']:.3f}%</td><td>{m['r2']:.4f}</td></tr>"
            )
        return rows

    best_base = min(
        ("xgboost", "lightgbm", "catboost"),
        key=lambda n: metrics["test"][n]["rmse"],
    )
    ens_rmse = metrics["test"]["ensemble"]["rmse"]
    base_rmse = metrics["test"][best_base]["rmse"]
    pct = (base_rmse - ens_rmse) / base_rmse * 100
    improvement = (
        f"Ensemble RMSE vs best base ({best_base}): {ens_rmse:.3f} vs {base_rmse:.3f} ({pct:+.2f}%)"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Ensemble vs Base Models — MCP Forecast</title>
  <style>
    body {{ font-family:Segoe UI,Arial,sans-serif; background:#0f1419; color:#e6edf3; margin:24px; }}
    h1 {{ color:#a78bfa; }} h2 {{ color:#38bdf8; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0 20px; }}
    th,td {{ border:1px solid #2d3a4a; padding:8px 12px; }}
    th {{ background:#1a2332; }}
    .card {{ background:#1a2332; border-radius:8px; padding:16px; margin-bottom:20px; }}
    .winner {{ color:#34d399; font-size:1.15em; }}
    img {{ max-width:100%; border-radius:6px; margin-top:8px; }}
  </style>
</head>
<body>
  <h1>Weighted Ensemble Performance Report</h1>
  <p>Generated (UTC): {datetime.now(timezone.utc).isoformat()}</p>
  <p class="winner">Overall test winner: <strong>{overall_winner.upper()}</strong></p>
  <p>{improvement}</p>

  <div class="card">
    <h2>Optimized Weights (fitted on validation)</h2>
    <table>
      <tr><th>Model</th><th>Weight</th></tr>
      {''.join(f'<tr><td>{k}</td><td>{v:.4f}</td></tr>' for k, v in weights.items())}
    </table>
    <img src="data:image/png;base64,{weights_img}" alt="Weights"/>
  </div>

  <div class="card">
    <h2>Test Metrics — All Models</h2>
    <table>
      <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>
      {metric_rows("test")}
    </table>
    <img src="data:image/png;base64,{chart_img}" alt="Test metrics"/>
  </div>

  <div class="card">
    <h2>Validation Metrics</h2>
    <table>
      <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>
      {metric_rows("validation")}
    </table>
    <img src="data:image/png;base64,{val_chart}" alt="Validation metrics"/>
  </div>

  <div class="card">
    <h2>Holdout Prediction Overlay</h2>
    <img src="data:image/png;base64,{overlay_img}" alt="Overlay"/>
  </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def plot_confusion_matrix(cm: list[list[int]]) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")
    arr = np.array(cm)
    im = ax.imshow(arr, cmap="Blues", alpha=0.85)
    ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Spike"])
    ax.set_yticks([0, 1], labels=["Actual Normal", "Actual Spike"])
    ax.set_title("Confusion Matrix (Test)", color="#e6edf3")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="#e6edf3", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")
    ax.plot(fpr, tpr, color="#f59e0b", linewidth=2, label="ROC")
    ax.plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1)
    ax.set_xlabel("False Positive Rate", color="#c9d1d9")
    ax.set_ylabel("True Positive Rate", color="#c9d1d9")
    ax.set_title("ROC Curve (Test)", color="#e6edf3")
    ax.legend()
    ax.tick_params(colors="#c9d1d9")
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_spike_classifier_report(
    output_path: Path,
    metrics: dict[str, dict[str, Any]],
    spike_threshold: float,
    percentile: float,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    test = metrics["test"]
    cm_img = plot_confusion_matrix(test["confusion_matrix"])
    roc_img = plot_roc_curve(y_test, y_prob)

    def split_rows(name: str) -> str:
        m = metrics[name]
        return (
            f"<tr><td>{name}</td><td>{m['precision']:.4f}</td><td>{m['recall']:.4f}</td>"
            f"<td>{m['f1']:.4f}</td><td>{m['roc_auc']:.4f}</td><td>{m['accuracy']:.4f}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>RTM MCP Spike Classifier Report</title>
  <style>
    body {{ font-family:Segoe UI,sans-serif; background:#0f1419; color:#e6edf3; margin:24px; }}
    h1,h2 {{ color:#fb7185; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0 20px; }}
    th,td {{ border:1px solid #2d3a4a; padding:8px 12px; }}
    th {{ background:#1a2332; }}
    .card {{ background:#1a2332; border-radius:8px; padding:16px; margin-bottom:20px; }}
    img {{ max-width:100%; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>RTM MCP Spike Prediction (XGBoost)</h1>
  <p>Generated (UTC): {datetime.now(timezone.utc).isoformat()}</p>
  <p>Spike rule: next-block MCP &gt; <strong>{spike_threshold:.2f}</strong> Rs/MWh
     (P{percentile:.0f} on training split)</p>
  <p>Output: <code>spike_probability</code> = P(spike) from <code>predict_proba</code></p>

  <div class="card">
    <h2>Classification Metrics</h2>
    <table>
      <tr><th>Split</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC AUC</th><th>Accuracy</th></tr>
      {split_rows("train")}
      {split_rows("validation")}
      {split_rows("test")}
    </table>
  </div>

  <div class="card">
    <h2>Test Confusion Matrix</h2>
    <table>
      <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
      <tr><th>Actual 0</th><td>{test['confusion_matrix'][0][0]}</td><td>{test['confusion_matrix'][0][1]}</td></tr>
      <tr><th>Actual 1</th><td>{test['confusion_matrix'][1][0]}</td><td>{test['confusion_matrix'][1][1]}</td></tr>
    </table>
    <img src="data:image/png;base64,{cm_img}" alt="Confusion Matrix"/>
  </div>

  <div class="card">
    <h2>ROC Curve (Test)</h2>
    <img src="data:image/png;base64,{roc_img}" alt="ROC"/>
  </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path

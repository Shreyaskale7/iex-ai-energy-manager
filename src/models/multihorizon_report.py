"""HTML report generator for multi-horizon analysis."""

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def plot_error_accumulation(comp_df: pd.DataFrame) -> str:
    """Plot recursive vs direct error accumulation (RMSE)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")

    # Filter out horizons where we don't have recursive data (e.g., > 96)
    rec_df = comp_df.dropna(subset=["recursive_rmse"])
    
    ax.plot(
        rec_df["horizon"], rec_df["recursive_rmse"], 
        marker="o", color="#f59e0b", label="Recursive Ensemble", linewidth=2
    )
    ax.plot(
        comp_df["horizon"], comp_df["direct_rmse"], 
        marker="s", color="#38bdf8", label="Direct LightGBM", linewidth=2
    )

    ax.set_title("Error Accumulation (RMSE) vs Horizon", color="#e6edf3")
    ax.set_xlabel("Horizon (blocks)", color="#c9d1d9")
    ax.set_ylabel("RMSE (Rs/MWh)", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    ax.set_xscale("log")
    ax.set_xticks(comp_df["horizon"])
    ax.set_xticklabels(comp_df["label"], rotation=45)
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_r2_decay(comp_df: pd.DataFrame) -> str:
    """Plot recursive vs direct R2 decay."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")

    rec_df = comp_df.dropna(subset=["recursive_r2"])

    ax.plot(
        rec_df["horizon"], rec_df["recursive_r2"], 
        marker="o", color="#f59e0b", label="Recursive Ensemble", linewidth=2
    )
    ax.plot(
        comp_df["horizon"], comp_df["direct_r2"], 
        marker="s", color="#38bdf8", label="Direct LightGBM", linewidth=2
    )

    ax.axhline(0.9, color="#34d399", linestyle="--", alpha=0.5, label="R² = 0.90")
    ax.axhline(0.8, color="#fb7185", linestyle="--", alpha=0.5, label="R² = 0.80")

    ax.set_title("Explained Variance (R²) Decay vs Horizon", color="#e6edf3")
    ax.set_xlabel("Horizon (blocks)", color="#c9d1d9")
    ax.set_ylabel("R²", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    ax.set_xscale("log")
    ax.set_xticks(comp_df["horizon"])
    ax.set_xticklabels(comp_df["label"], rotation=45)
    ax.set_ylim(-0.2, 1.05)
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_multihorizon_report(
    output_path: Path,
    comp_df: pd.DataFrame,
    recommendation: dict[str, Any],
) -> Path:
    error_img = plot_error_accumulation(comp_df)
    r2_img = plot_r2_decay(comp_df)

    rows = ""
    for _, r in comp_df.iterrows():
        rec_rmse = f"{r['recursive_rmse']:.2f}" if pd.notna(r["recursive_rmse"]) else "—"
        dir_rmse = f"{r['direct_rmse']:.2f}" if pd.notna(r["direct_rmse"]) else "—"
        rec_r2 = f"{r['recursive_r2']:.4f}" if pd.notna(r["recursive_r2"]) else "—"
        dir_r2 = f"{r['direct_r2']:.4f}" if pd.notna(r["direct_r2"]) else "—"
        
        winner_class = 'style="color:#34d399;font-weight:bold"' if r['winner'] != "—" else ""

        rows += f"""
        <tr>
            <td>{r['horizon']} ({r['label']})</td>
            <td>{rec_rmse}</td>
            <td>{dir_rmse}</td>
            <td>{rec_r2}</td>
            <td>{dir_r2}</td>
            <td {winner_class}>{r['winner'].upper()}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Multi-Horizon Forecasting Analysis</title>
  <style>
    body {{ font-family:Segoe UI,Arial,sans-serif; background:#0f1419; color:#e6edf3; margin:24px; }}
    h1 {{ color:#a78bfa; }} h2 {{ color:#38bdf8; }} h3 {{ color:#f59e0b; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0 20px; }}
    th,td {{ border:1px solid #2d3a4a; padding:8px 12px; text-align:center; }}
    th {{ background:#1a2332; }}
    tr:nth-child(even) {{ background:#151d28; }}
    .card {{ background:#1a2332; border-radius:8px; padding:16px; margin-bottom:20px; }}
    img {{ max-width:100%; border-radius:6px; margin-top:8px; }}
    .diagram {{ background:#0f1419; padding:16px; font-family:monospace; color:#34d399; border-radius:4px; }}
  </style>
</head>
<body>
  <h1>Phase 3: Multi-Horizon Forecasting Analysis</h1>
  <p>Generated (UTC): {datetime.now(timezone.utc).isoformat()}</p>
  
  <div class="card">
    <h2>Architecture Recommendation</h2>
    <p><strong>Conclusion:</strong> {recommendation['summary']}</p>
    <ul>
        <li>R² drops below 0.90 at horizon: {recommendation.get('r2_drops_below_90_at', 'N/A')}</li>
        <li>R² drops below 0.80 at horizon: {recommendation.get('r2_drops_below_80_at', 'N/A')}</li>
        <li>Direct models outperform recursive starting at horizon: {recommendation.get('crossover_horizon', 'N/A')}</li>
    </ul>

    <h3>Proposed Production Pipeline</h3>
    <div class="diagram">
┌─────────────────────────────────────────────────┐<br/>
│           30-Day Forecast Pipeline              │<br/>
├─────────────────────────────────────────────────┤<br/>
│ Short-term ({recommendation['short_term']['horizons']}):<br/>
│   → {recommendation['short_term']['strategy'].title()} Approach<br/>
│                                                 │<br/>
│ Medium-term ({recommendation['medium_term']['horizons']}):<br/>
│   → {recommendation['medium_term']['strategy'].title()} Approach<br/>
│                                                 │<br/>
│ Long-term ({recommendation['long_term']['horizons']}):<br/>
│   → {recommendation['long_term']['strategy'].title()} Approach<br/>
└─────────────────────────────────────────────────┘
    </div>
  </div>

  <div class="card">
    <h2>Error Accumulation Profile</h2>
    <img src="data:image/png;base64,{error_img}" alt="Error Accumulation"/>
    <img src="data:image/png;base64,{r2_img}" alt="R2 Decay"/>
  </div>

  <div class="card">
    <h2>Horizon-by-Horizon Comparison</h2>
    <table>
      <tr>
        <th>Horizon</th>
        <th colspan="2">RMSE (Rs/MWh)</th>
        <th colspan="2">R²</th>
        <th>Winner</th>
      </tr>
      <tr>
        <th></th>
        <th>Recursive</th>
        <th>Direct</th>
        <th>Recursive</th>
        <th>Direct</th>
        <th></th>
      </tr>
      {rows}
    </table>
  </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path

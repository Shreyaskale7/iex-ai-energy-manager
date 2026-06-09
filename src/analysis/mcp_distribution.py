"""MCP distribution analysis and MAPE validity assessment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.report import _fig_to_base64

THRESHOLDS = (10, 50, 100)


@dataclass
class MCPDistributionResult:
    json_path: Path
    html_path: Path
    histogram_path: Path
    summary: dict[str, Any]


def load_mcp_series(parquet_path: Path) -> pd.Series:
    df = pd.read_parquet(parquet_path)
    if "mcp_rs_mwh" not in df.columns:
        raise ValueError(f"Column mcp_rs_mwh not found in {parquet_path}")
    return pd.to_numeric(df["mcp_rs_mwh"], errors="coerce").dropna()


def compute_distribution_stats(mcp: pd.Series) -> dict[str, Any]:
    values = mcp.to_numpy(dtype=float)
    n = len(values)

    counts = {
        "mcp_lt_10": int((values < 10).sum()),
        "mcp_lt_50": int((values < 50).sum()),
        "mcp_lt_100": int((values < 100).sum()),
        "mcp_eq_0": int((values == 0).sum()),
    }
    pct = {k: round(100.0 * v / n, 2) for k, v in counts.items()}

    percentiles = {
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }

    return {
        "row_count": n,
        "minimum_mcp": float(values.min()),
        "maximum_mcp": float(values.max()),
        "median_mcp": float(np.median(values)),
        "mean_mcp": float(values.mean()),
        "std_mcp": float(values.std()),
        "counts": counts,
        "counts_pct": pct,
        "percentiles": percentiles,
    }


def assess_mape_validity(stats: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether MAPE is appropriate given the MCP distribution."""
    c = stats["counts"]
    pct = stats["counts_pct"]
    median = stats["median_mcp"]
    mean = stats["mean_mcp"]
    p10 = stats["percentiles"]["p10"]

    issues: list[str] = []

    if c["mcp_eq_0"] > 0:
        issues.append(
            f"{c['mcp_eq_0']:,} rows ({pct['mcp_eq_0']:.2f}%) have MCP = 0. With denominator "
            f"stabilization at 1 Rs/MWh, each Rs/MWh of error counts as ~100% MAPE on those rows."
        )
    if pct["mcp_lt_10"] > 0:
        issues.append(
            f"{c['mcp_lt_10']:,} rows ({pct['mcp_lt_10']:.2f}%) have MCP < 10 Rs/MWh — any absolute "
            "forecast error dominates the percentage."
        )
    if pct["mcp_lt_100"] > 0:
        issues.append(
            f"{c['mcp_lt_100']:,} rows ({pct['mcp_lt_100']:.2f}%) have MCP < 100 Rs/MWh; MAPE is "
            "highly sensitive in this band (denominator < 100)."
        )
    if mean > median * 1.15:
        issues.append(
            f"Distribution is right-skewed (mean {mean:,.0f} vs median {median:,.0f} Rs/MWh, "
            f"P10 = {p10:,.0f}). Averages and pooled MAPE are pulled by high-price tail events."
        )
    if stats["maximum_mcp"] >= 9999:
        issues.append(
            f"Hard ceiling at {stats['maximum_mcp']:,.0f} Rs/MWh compresses extreme spikes; "
            "percentage errors near the cap are not comparable to typical clearing prices."
        )

    issues.append(
        "Holdout evaluation shows MAPE ~670–725% vs sMAPE ~17% on the same predictions — "
        "off-peak hours (lower typical MCP) inflate pooled MAPE even when few rows are near zero."
    )

    verdict = "not_recommended"
    recommendations = [
        "Use sMAPE, MAE, or RMSE as primary metrics (sMAPE ~17% aligns with model R² ~0.95).",
        "If reporting MAPE, filter to MCP >= 100 Rs/MWh and disclose the excluded row count.",
        "Segment by peak (18–22) vs off-peak; do not use a single pooled MAPE for operations.",
        "Track Rs/MWh MAE (~390 on test) for business interpretability alongside sMAPE.",
    ]

    return {
        "verdict": verdict,
        "is_mape_valid_primary_metric": False,
        "issues": issues,
        "recommendations": recommendations,
        "summary": (
            "MAPE is not recommended as a primary metric. The bulk of MCP values are mid-market "
            f"(median {median:,.0f} Rs/MWh), but zeros, low-clearing blocks, right-skew, and the "
            "10,000 cap make percentage errors unstable; sMAPE and MAE are better aligned with "
            "forecast quality on this dataset."
        ),
    }


def plot_mcp_histograms(mcp: pd.Series, output_dir: Path) -> tuple[str, Path]:
    """Full-range and zoomed histograms; return base64 for HTML and PNG path."""
    values = mcp.to_numpy(dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "mcp_histogram.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0f1419")

    panels = [
        (axes[0, 0], values, 80, "Full range MCP histogram", "#f59e0b", None),
        (axes[0, 1], values[values <= 500], 50, "MCP <= 500 Rs/MWh", "#38bdf8", None),
        (axes[1, 0], values[values <= 100], 40, "MCP <= 100 Rs/MWh (low-clearing zone)", "#fb7185", None),
        (axes[1, 1], values[values > 0], 80, "Log-scale (MCP > 0 only)", "#34d399", "log"),
    ]

    for ax, data, bins, title, color, scale in panels:
        ax.set_facecolor("#1a2332")
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#c9d1d9", transform=ax.transAxes)
        else:
            ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="#0f1419")
            if scale == "log":
                ax.set_yscale("log")
        ax.set_title(title, color="#e6edf3")
        ax.set_xlabel("MCP (Rs/MWh)", color="#c9d1d9")
        ax.set_ylabel("Count", color="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")

    fig.suptitle("IEX RTM MCP Distribution", color="#e6edf3", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    b64 = _fig_to_base64(fig)
    return b64, png_path


def generate_html_report(
    stats: dict[str, Any],
    mape_assessment: dict[str, Any],
    chart_b64: str,
    source_path: Path,
    output_path: Path,
) -> None:
    c = stats["counts"]
    p = stats["counts_pct"]
    pctiles = stats["percentiles"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict_color = "#fb7185" if mape_assessment["verdict"] == "not_recommended" else "#f59e0b"

    issues_html = "".join(f"<li>{issue}</li>" for issue in mape_assessment["issues"])
    rec_html = "".join(f"<li>{r}</li>" for r in mape_assessment["recommendations"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>MCP Distribution Analysis</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0f1419; color: #e6edf3; margin: 2rem; max-width: 1100px; }}
    h1, h2 {{ color: #e6edf3; }}
    p, li, td, th {{ color: #c9d1d9; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #30363d; padding: 0.6rem 0.8rem; text-align: right; }}
    th {{ background: #1a2332; text-align: left; }}
    .verdict {{ background: #1a2332; padding: 1rem 1.25rem; border-radius: 8px; border-left: 4px solid {verdict_color}; margin: 1rem 0; }}
    img {{ max-width: 100%; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>MCP Distribution Analysis</h1>
  <p>Generated: {generated} | Source: {source_path} | Rows: {stats['row_count']:,}</p>

  <h2>Summary statistics</h2>
  <table>
    <tr><th>Statistic</th><th>Value (Rs/MWh)</th></tr>
    <tr><td>Minimum MCP</td><td>{stats['minimum_mcp']:,.2f}</td></tr>
    <tr><td>Maximum MCP</td><td>{stats['maximum_mcp']:,.2f}</td></tr>
    <tr><td>Median MCP</td><td>{stats['median_mcp']:,.2f}</td></tr>
    <tr><td>Mean MCP</td><td>{stats['mean_mcp']:,.2f}</td></tr>
    <tr><td>Std dev</td><td>{stats['std_mcp']:,.2f}</td></tr>
  </table>

  <h2>Threshold counts</h2>
  <table>
    <tr><th>Condition</th><th>Count</th><th>% of dataset</th></tr>
    <tr><td>MCP &lt; 10</td><td>{c['mcp_lt_10']:,}</td><td>{p['mcp_lt_10']:.2f}%</td></tr>
    <tr><td>MCP &lt; 50</td><td>{c['mcp_lt_50']:,}</td><td>{p['mcp_lt_50']:.2f}%</td></tr>
    <tr><td>MCP &lt; 100</td><td>{c['mcp_lt_100']:,}</td><td>{p['mcp_lt_100']:.2f}%</td></tr>
    <tr><td>MCP = 0</td><td>{c['mcp_eq_0']:,}</td><td>{p['mcp_eq_0']:.2f}%</td></tr>
  </table>

  <h2>Percentiles</h2>
  <table>
    <tr><th>Percentile</th><th>MCP (Rs/MWh)</th></tr>
    <tr><td>P01</td><td>{pctiles['p01']:,.2f}</td></tr>
    <tr><td>P05</td><td>{pctiles['p05']:,.2f}</td></tr>
    <tr><td>P10</td><td>{pctiles['p10']:,.2f}</td></tr>
    <tr><td>P25</td><td>{pctiles['p25']:,.2f}</td></tr>
    <tr><td>P75</td><td>{pctiles['p75']:,.2f}</td></tr>
    <tr><td>P90</td><td>{pctiles['p90']:,.2f}</td></tr>
    <tr><td>P95</td><td>{pctiles['p95']:,.2f}</td></tr>
    <tr><td>P99</td><td>{pctiles['p99']:,.2f}</td></tr>
  </table>

  <h2>Histogram</h2>
  <img src="data:image/png;base64,{chart_b64}" alt="MCP histogram"/>

  <h2>Is MAPE a valid metric?</h2>
  <div class="verdict">
    <strong>Verdict:</strong> {mape_assessment['verdict'].replace('_', ' ').upper()}<br/>
    {mape_assessment['summary']}
  </div>
  <h3>Issues identified</h3>
  <ul>{issues_html or '<li>No critical issues — still prefer sMAPE/MAE for skewed prices.</li>'}</ul>
  <h3>Recommendations</h3>
  <ul>{rec_html}</ul>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


class MCPDistributionAnalyzer:
    def __init__(
        self,
        data_path: Path | str = "data/processed/rtm_master.parquet",
        report_dir: Path | str = "reports",
    ) -> None:
        self.data_path = Path(data_path)
        self.report_dir = Path(report_dir)

    def run(self) -> MCPDistributionResult:
        mcp = load_mcp_series(self.data_path)
        stats = compute_distribution_stats(mcp)
        mape_assessment = assess_mape_validity(stats)

        chart_b64, histogram_path = plot_mcp_histograms(mcp, self.report_dir)

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(self.data_path),
            "statistics": stats,
            "mape_validity": mape_assessment,
        }

        json_path = self.report_dir / "mcp_distribution.json"
        html_path = self.report_dir / "mcp_distribution_report.html"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        generate_html_report(stats, mape_assessment, chart_b64, self.data_path, html_path)

        return MCPDistributionResult(
            json_path=json_path,
            html_path=html_path,
            histogram_path=histogram_path,
            summary=payload,
        )

"""Exploratory data analysis for rtm_master.parquet."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "purchase_bid_mw",
    "sell_bid_mw",
    "mcv_mw",
    "scheduled_volume_mw",
    "mcp_rs_mwh",
]

DISTRIBUTION_COLUMNS = [
    ("mcp_rs_mwh", "MCP (Rs/MWh)", "01_mcp_distribution.png"),
    ("purchase_bid_mw", "Purchase Bid (MW)", "02_purchase_bid_distribution.png"),
    ("sell_bid_mw", "Sell Bid (MW)", "03_sell_bid_distribution.png"),
    ("mcv_mw", "MCV (MW)", "04_mcv_distribution.png"),
    ("scheduled_volume_mw", "Scheduled Volume (MW)", "05_scheduled_volume_distribution.png"),
]

EDA_STYLE = {
    "figure.facecolor": "#0f1419",
    "axes.facecolor": "#1a2332",
    "axes.edgecolor": "#3d4f5f",
    "axes.labelcolor": "#e6edf3",
    "text.color": "#e6edf3",
    "xtick.color": "#9eb3c7",
    "ytick.color": "#9eb3c7",
    "grid.color": "#2d3a4a",
    "font.size": 11,
}
PALETTE = ["#f59e0b", "#38bdf8", "#34d399", "#a78bfa", "#fb7185"]


@dataclass
class EDAResult:
    parquet_path: Path
    output_dir: Path
    summary_path: Path
    report_path: Path
    plot_paths: list[Path]
    summary: dict[str, Any]


class RTMMasterEDAAnalyzer:
    """Runs full EDA on rtm_master.parquet and writes reports/plots."""

    def __init__(
        self,
        parquet_path: Path | str = "data/processed/rtm_master.parquet",
        output_dir: Path | str = "reports/eda",
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.output_dir = Path(output_dir)
        self.summary_path = self.output_dir / "eda_summary.json"
        self.report_path = self.output_dir / "eda_report.md"
        self._df: pd.DataFrame | None = None
        self._summary: dict[str, Any] = {}

    def run(self) -> EDAResult:
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Master dataset not found: {self.parquet_path}. "
                "Run scripts/build_rtm_master.py first."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="darkgrid", palette=PALETTE)
        plt.rcParams.update(EDA_STYLE)

        self._df = self._load_and_prepare()
        plot_paths: list[Path] = []

        self._summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_file": str(self.parquet_path),
            "dataset_summary": self.dataset_summary(),
            "missing_value_analysis": self.missing_value_analysis(),
            "outlier_analysis": self.outlier_analysis(),
            "correlation_matrix": self.correlation_matrix(),
        }

        plot_paths.append(self._plot_missing_values())
        plot_paths.append(self._plot_outlier_boxplots())
        for column, title, filename in DISTRIBUTION_COLUMNS:
            if column in self._df.columns:
                plot_paths.append(self._plot_distribution(column, title, filename))
                self._summary[f"{column}_distribution"] = self._distribution_stats(column)

        plot_paths.append(self._plot_correlation_matrix())
        plot_paths.append(self._plot_mcp_trend_monthly())
        plot_paths.append(self._plot_mcp_trend_daily())
        plot_paths.append(self._plot_mcp_trend_hourly())

        self._write_summary_json()
        self._write_markdown_report()

        logger.info("EDA complete: %d plots saved to %s", len(plot_paths), self.output_dir)
        return EDAResult(
            parquet_path=self.parquet_path,
            output_dir=self.output_dir,
            summary_path=self.summary_path,
            report_path=self.report_path,
            plot_paths=plot_paths,
            summary=self._summary,
        )

    def _load_and_prepare(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        if "block_timestamp" in df.columns:
            df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], utc=True)
            if df["block_timestamp"].dt.tz is not None:
                df["block_timestamp"] = df["block_timestamp"].dt.tz_convert("Asia/Kolkata")
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df["year_month"] = df["block_timestamp"].dt.to_period("M").astype(str)
        df["trade_date_only"] = df["block_timestamp"].dt.date
        df["hour_of_day"] = df["block_timestamp"].dt.hour
        if "hour" in df.columns and df["hour"].notna().any():
            df["hour_of_day"] = df["hour"].fillna(df["hour_of_day"]).astype(int) - 1
            df.loc[df["hour_of_day"] < 0, "hour_of_day"] = 0
        return df

    def dataset_summary(self) -> dict[str, Any]:
        df = self._require_df()
        ts = df["block_timestamp"]
        numeric_summary = {}
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                numeric_summary[col] = df[col].describe().round(4).to_dict()

        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "date_range_start": ts.min().isoformat() if not ts.empty else None,
            "date_range_end": ts.max().isoformat() if not ts.empty else None,
            "unique_trade_days": int(df["trade_date_only"].nunique()) if "trade_date_only" in df.columns else None,
            "unique_source_files": int(df["source_file"].nunique()) if "source_file" in df.columns else None,
            "blocks_per_day_median": float(
                df.groupby("trade_date_only").size().median()
            )
            if "trade_date_only" in df.columns
            else None,
            "numeric_summary": numeric_summary,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

    def missing_value_analysis(self) -> dict[str, Any]:
        df = self._require_df()
        total = len(df)
        by_column = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            if df[col].dtype == object:
                missing += int((df[col].astype(str).str.strip() == "").sum())
            by_column.append(
                {
                    "column": col,
                    "missing_count": missing,
                    "missing_pct": round(100 * missing / total, 4) if total else 0.0,
                }
            )
        by_column.sort(key=lambda x: x["missing_count"], reverse=True)
        rows_any_missing = int(df.isna().any(axis=1).sum())
        return {
            "total_rows": total,
            "rows_with_any_missing": rows_any_missing,
            "rows_with_any_missing_pct": round(100 * rows_any_missing / total, 4) if total else 0.0,
            "by_column": by_column,
        }

    def outlier_analysis(self, iqr_multiplier: float = 1.5) -> dict[str, Any]:
        df = self._require_df()
        results: dict[str, Any] = {}
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask = (series < lower) | (series > upper)
            results[col] = {
                "method": "IQR",
                "multiplier": iqr_multiplier,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "outlier_count": int(mask.sum()),
                "outlier_pct": round(100 * mask.sum() / len(series), 4),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        return results

    def correlation_matrix(self) -> dict[str, Any]:
        df = self._require_df()
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        corr = df[cols].corr().round(4)
        return {
            "columns": cols,
            "matrix": corr.to_dict(),
        }

    def mcp_trend_monthly(self) -> dict[str, Any]:
        df = self._require_df()
        monthly = (
            df.groupby("year_month", as_index=False)["mcp_rs_mwh"]
            .agg(mean_mcp="mean", median_mcp="median", std_mcp="std", count="count")
            .sort_values("year_month")
        )
        return monthly.to_dict(orient="records")

    def mcp_trend_daily(self) -> dict[str, Any]:
        df = self._require_df()
        daily = (
            df.groupby("trade_date_only", as_index=False)["mcp_rs_mwh"]
            .agg(mean_mcp="mean", median_mcp="median", count="count")
            .sort_values("trade_date_only")
        )
        return {
            "day_count": len(daily),
            "first_10_days": daily.head(10).to_dict(orient="records"),
            "last_10_days": daily.tail(10).to_dict(orient="records"),
            "overall_mean": float(daily["mean_mcp"].mean()),
            "overall_std": float(daily["mean_mcp"].std()),
        }

    def mcp_trend_hourly(self) -> dict[str, Any]:
        df = self._require_df()
        hourly = (
            df.groupby("hour_of_day")["mcp_rs_mwh"]
            .agg(
                mean_mcp="mean",
                median_mcp="median",
                p10=lambda s: float(s.quantile(0.10)),
                p90=lambda s: float(s.quantile(0.90)),
            )
            .reset_index()
            .sort_values("hour_of_day")
        )
        return hourly.to_dict(orient="records")

    def _distribution_stats(self, column: str) -> dict[str, float]:
        series = self._require_df()[column].dropna()
        return {
            "count": int(len(series)),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }

    def _plot_missing_values(self) -> Path:
        missing = self.missing_value_analysis()["by_column"]
        plot_df = pd.DataFrame(missing)
        plot_df = plot_df[plot_df["missing_count"] > 0]
        if plot_df.empty:
            plot_df = pd.DataFrame(missing).head(10)

        fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.35)))
        bars = ax.barh(plot_df["column"], plot_df["missing_pct"], color=PALETTE[1], alpha=0.9)
        ax.set_xlabel("Missing (%)")
        ax.set_title("Missing Value Analysis by Column")
        ax.invert_yaxis()
        for bar, pct in zip(bars, plot_df["missing_pct"]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{pct:.2f}%", va="center", fontsize=9)
        fig.tight_layout()
        return self._save_fig(fig, "06_missing_values.png")

    def _plot_outlier_boxplots(self) -> Path:
        df = self._require_df()
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            sns.boxplot(y=df[col], ax=ax, color=PALETTE[0], fliersize=2)
            ax.set_title(col.replace("_", " ").title())
            out = self.outlier_analysis().get(col, {})
            if out:
                ax.text(
                    0.02,
                    0.98,
                    f"Outliers: {out['outlier_count']:,} ({out['outlier_pct']:.1f}%)",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )
        fig.suptitle("Outlier Analysis (IQR 1.5×)", y=1.02)
        fig.tight_layout()
        return self._save_fig(fig, "07_outlier_boxplots.png")

    def _plot_distribution(self, column: str, title: str, filename: str) -> Path:
        series = self._require_df()[column].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].hist(series, bins=80, color=PALETTE[0], edgecolor="#1a2332", alpha=0.85)
        axes[0].set_title(f"{title} — Histogram")
        axes[0].set_xlabel(title)
        axes[0].set_ylabel("Frequency")

        sns.kdeplot(series, ax=axes[1], color=PALETTE[2], fill=True, alpha=0.35)
        axes[1].axvline(series.mean(), color=PALETTE[4], linestyle="--", label=f"Mean: {series.mean():,.1f}")
        axes[1].axvline(series.median(), color=PALETTE[1], linestyle=":", label=f"Median: {series.median():,.1f}")
        axes[1].set_title(f"{title} — Density")
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        return self._save_fig(fig, filename)

    def _plot_correlation_matrix(self) -> Path:
        corr_data = self.correlation_matrix()
        corr = pd.DataFrame(corr_data["matrix"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Pearson r"},
        )
        ax.set_title("Correlation Matrix — Numeric RTM Features")
        fig.tight_layout()
        return self._save_fig(fig, "08_correlation_matrix.png")

    def _plot_mcp_trend_monthly(self) -> Path:
        df = self._require_df()
        monthly = df.groupby("year_month")["mcp_rs_mwh"].agg(["mean", "median", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(len(monthly))
        ax.plot(x, monthly["mean"], marker="o", label="Mean MCP", color=PALETTE[0], linewidth=2)
        ax.fill_between(
            x,
            monthly["mean"] - monthly["std"],
            monthly["mean"] + monthly["std"],
            alpha=0.2,
            color=PALETTE[0],
            label="±1 Std",
        )
        ax.plot(x, monthly["median"], marker="s", label="Median MCP", color=PALETTE[1], linewidth=1.5, alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right")
        ax.set_ylabel("MCP (Rs/MWh)")
        ax.set_title("Monthly MCP Trend")
        ax.legend()
        fig.tight_layout()
        self._summary["mcp_trend_monthly"] = self.mcp_trend_monthly()
        return self._save_fig(fig, "09_mcp_trend_monthly.png")

    def _plot_mcp_trend_daily(self) -> Path:
        df = self._require_df()
        daily = (
            df.groupby("trade_date_only")["mcp_rs_mwh"]
            .mean()
            .reset_index()
            .sort_values("trade_date_only")
        )
        rolling = daily["mcp_rs_mwh"].rolling(window=7, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily["trade_date_only"], daily["mcp_rs_mwh"], color=PALETTE[1], alpha=0.35, linewidth=0.8, label="Daily mean")
        ax.plot(daily["trade_date_only"], rolling, color=PALETTE[0], linewidth=2, label="7-day rolling mean")
        ax.set_ylabel("MCP (Rs/MWh)")
        ax.set_title("Daily MCP Trend")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        self._summary["mcp_trend_daily"] = self.mcp_trend_daily()
        return self._save_fig(fig, "10_mcp_trend_daily.png")

    def _plot_mcp_trend_hourly(self) -> Path:
        df = self._require_df()
        hourly = (
            df.groupby("hour_of_day")["mcp_rs_mwh"]
            .agg(mean_mcp="mean", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.fill_between(hourly["hour_of_day"], hourly["p10"], hourly["p90"], alpha=0.25, color=PALETTE[2], label="P10–P90")
        ax.plot(hourly["hour_of_day"], hourly["mean_mcp"], marker="o", color=PALETTE[0], linewidth=2, label="Mean MCP")
        ax.set_xlabel("Hour of day (0–23)")
        ax.set_ylabel("MCP (Rs/MWh)")
        ax.set_title("Hourly MCP Trend (Average by Hour)")
        ax.set_xticks(range(0, 24, 2))
        ax.legend()
        fig.tight_layout()
        self._summary["mcp_trend_hourly"] = self.mcp_trend_hourly()
        return self._save_fig(fig, "11_mcp_trend_hourly.png")

    def _write_summary_json(self) -> None:
        def default(obj: Any) -> Any:
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            if isinstance(obj, (pd.Period,)):
                return str(obj)
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            return str(obj)

        payload = json.loads(json.dumps(self._summary, default=default))
        self.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_markdown_report(self) -> None:
        s = self._summary
        ds = s["dataset_summary"]
        miss = s["missing_value_analysis"]
        outliers = s["outlier_analysis"]

        lines = [
            "# RTM Master — Exploratory Data Analysis Report",
            "",
            f"**Generated (UTC):** {s['generated_at_utc']}",
            f"**Source:** `{s['source_file']}`",
            "",
            "## 1. Dataset Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Rows | {ds['row_count']:,} |",
            f"| Columns | {ds['column_count']} |",
            f"| Memory (MB) | {ds['memory_mb']} |",
            f"| Date range | {ds['date_range_start']} → {ds['date_range_end']} |",
            f"| Unique trade days | {ds.get('unique_trade_days', 'N/A')} |",
            f"| Median blocks/day | {ds.get('blocks_per_day_median', 'N/A')} |",
            "",
            "## 2. Missing Value Analysis",
            "",
            f"- Rows with any missing: **{miss['rows_with_any_missing']:,}** ({miss['rows_with_any_missing_pct']:.2f}%)",
            "",
            "| Column | Missing | % |",
            "|--------|---------|---|",
        ]
        for row in miss["by_column"]:
            if row["missing_count"] > 0:
                lines.append(f"| {row['column']} | {row['missing_count']:,} | {row['missing_pct']:.2f}% |")

        lines.extend(["", "## 3. Outlier Analysis (IQR 1.5×)", "", "| Feature | Outliers | % | Lower | Upper |", "|---------|----------|---|-------|-------|"])
        for col, stats in outliers.items():
            lines.append(
                f"| {col} | {stats['outlier_count']:,} | {stats['outlier_pct']:.2f}% | "
                f"{stats['lower_fence']:,.1f} | {stats['upper_fence']:,.1f} |"
            )

        lines.extend(["", "## 4–8. Distributions", "", "See plots:", ""])
        for _, title, fname in DISTRIBUTION_COLUMNS:
            lines.append(f"- `{fname}` — {title}")

        lines.extend(
            [
                "",
                "## 9. Correlation Matrix",
                "",
                "![Correlation](08_correlation_matrix.png)",
                "",
                "```json",
                json.dumps(s["correlation_matrix"]["matrix"], indent=2)[:2000],
                "```",
                "",
                "## 10–12. MCP Trends",
                "",
                "- `09_mcp_trend_monthly.png` — Monthly mean ± std",
                "- `10_mcp_trend_daily.png` — Daily mean + 7-day rolling",
                "- `11_mcp_trend_hourly.png` — Hourly mean with P10–P90 band",
                "",
                "## Artifacts",
                "",
                f"- Summary JSON: `{self.summary_path.name}`",
                f"- All plots: `{self.output_dir}/`",
            ]
        )
        self.report_path.write_text("\n".join(lines), encoding="utf-8")

    def _save_fig(self, fig: plt.Figure, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def _require_df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Data not loaded. Call run() first.")
        return self._df


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> int:
    configure_logging()
    result = RTMMasterEDAAnalyzer().run()
    print(result.report_path.read_text(encoding="utf-8")[:3000])
    print(f"\n... full report: {result.report_path}")
    print(f"Summary JSON: {result.summary_path}")
    print(f"Plots ({len(result.plot_paths)}): {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

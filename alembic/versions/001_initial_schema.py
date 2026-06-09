"""Consolidated schema — RTM blocks, forecast storage (Phase 5), training runs.

Revision ID: 001
Revises: None
Create Date: 2026-06-05

This single migration creates the full Phase 5 schema from scratch.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── rtm_blocks ────────────────────────────────────────────────────
    op.create_table(
        "rtm_blocks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("hour", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("time_block", sa.Integer(), nullable=False),
        sa.Column("purchase_bid_mw", sa.Float(), nullable=False),
        sa.Column("sell_bid_mw", sa.Float(), nullable=False),
        sa.Column("mcv_mw", sa.Float(), nullable=False),
        sa.Column("scheduled_volume_mw", sa.Float(), nullable=False),
        sa.Column("mcp_rs_mwh", sa.Float(), nullable=False),
        sa.Column("block_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_file", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trade_date", "hour", "session_id", "time_block", name="uq_rtm_block_identity"),
    )
    op.create_index("ix_rtm_blocks_block_timestamp", "rtm_blocks", ["block_timestamp"])

    # ── forecast_runs (Phase 5) ───────────────────────────────────────
    op.create_table(
        "forecast_runs",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("feature_version", sa.String(64), nullable=False, server_default="v1"),
        sa.Column("forecast_type", sa.String(32), nullable=False, server_default="30-Day"),
        sa.Column("status", sa.String(32), nullable=False, server_default="completed"),
        sa.Column("csv_path", sa.String(512), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_forecast_runs_generated_at", "forecast_runs", ["generated_at"])

    # ── forecasts (Phase 5) ───────────────────────────────────────────
    op.create_table(
        "forecasts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("forecast_run_id", sa.String(36), nullable=False),
        sa.Column("forecast_horizon", sa.Integer(), nullable=False),
        sa.Column("forecast_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("block_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("predicted_mcp", sa.Float(), nullable=False),
        sa.Column("zone", sa.String(10), nullable=False, server_default="GREEN"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("spike_probability", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("lower_bound", sa.Float(), nullable=True),
        sa.Column("upper_bound", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("forecast_run_id", "forecast_horizon", name="uq_forecast_horizon"),
    )
    op.create_index("ix_forecasts_forecast_run_id", "forecasts", ["forecast_run_id"])
    op.create_index("ix_forecasts_timestamp", "forecasts", ["forecast_timestamp"])
    op.create_index("ix_forecasts_zone", "forecasts", ["zone"])

    # ── forecast_accuracy ─────────────────────────────────────────────
    op.create_table(
        "forecast_accuracy",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("forecast_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_mcp", sa.Float(), nullable=False),
        sa.Column("actual_mcp", sa.Float(), nullable=False),
        sa.Column("predicted_zone", sa.String(10), nullable=False),
        sa.Column("actual_zone", sa.String(10), nullable=False),
        sa.Column("absolute_error", sa.Float(), nullable=False),
        sa.Column("percentage_error", sa.Float(), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "horizon", name="uq_accuracy_run_horizon"),
    )
    op.create_index("ix_accuracy_run_id", "forecast_accuracy", ["run_id"])

    # ── training_runs ─────────────────────────────────────────────────
    op.create_table(
        "training_runs",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("horizons_trained", sa.Integer(), nullable=False),
        sa.Column("train_rows", sa.Integer(), nullable=False),
        sa.Column("test_rows", sa.Integer(), nullable=False),
        sa.Column("mean_mae", sa.Float(), nullable=False),
        sa.Column("mean_rmse", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("training_runs")
    op.drop_index("ix_accuracy_run_id", table_name="forecast_accuracy")
    op.drop_table("forecast_accuracy")
    op.drop_index("ix_forecasts_zone", table_name="forecasts")
    op.drop_index("ix_forecasts_timestamp", table_name="forecasts")
    op.drop_index("ix_forecasts_forecast_run_id", table_name="forecasts")
    op.drop_table("forecasts")
    op.drop_index("ix_forecast_runs_generated_at", table_name="forecast_runs")
    op.drop_table("forecast_runs")
    op.drop_index("ix_rtm_blocks_block_timestamp", table_name="rtm_blocks")
    op.drop_table("rtm_blocks")

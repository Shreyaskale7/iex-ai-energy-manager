"""IEX RTM Excel schema definitions."""

from __future__ import annotations

BLOCKS_PER_DAY = 96
BLOCK_MINUTES = 15
TIMEZONE = "Asia/Kolkata"

RAW_COLUMNS = [
    "Date",
    "Hour",
    "Session ID",
    "Time Block",
    "Purchase Bid (MW)",
    "Sell Bid (MW)",
    "MCV (MW)",
    "Scheduled Volume (MW)",
    "MCP (Rs/MWh)",
]

COLUMN_ALIASES = {
    "Final Scheduled Volume (MW)": "Scheduled Volume (MW)",
    "Scheduled Volume (MW)": "Scheduled Volume (MW)",
    "MCP (Rs/MWh) *": "MCP (Rs/MWh)",
    "MCP (Rs/MWh)": "MCP (Rs/MWh)",
}

COLUMN_RENAME_MAP = {
    "Date": "trade_date",
    "Hour": "hour",
    "Session ID": "session_id",
    "Time Block": "time_block",
    "Purchase Bid (MW)": "purchase_bid_mw",
    "Sell Bid (MW)": "sell_bid_mw",
    "MCV (MW)": "mcv_mw",
    "Scheduled Volume (MW)": "scheduled_volume_mw",
    "MCP (Rs/MWh)": "mcp_rs_mwh",
}

RTM_MARKET_GLOB_PATTERNS = ("RTM_Market*.xlsx", "RTM_Market*.xls", "*.xlsx", "*.xls")

STANDARD_COLUMNS = list(COLUMN_RENAME_MAP.values()) + ["source_file", "block_timestamp", "daily_block_index"]

REQUIRED_FOR_TIMESTAMP = ["trade_date", "time_block"]
REQUIRED_FOR_MODEL = ["trade_date", "time_block", "mcp_rs_mwh"]

NUMERIC_COLUMNS = [
    "hour",
    "session_id",
    "time_block",
    "purchase_bid_mw",
    "sell_bid_mw",
    "mcv_mw",
    "scheduled_volume_mw",
    "mcp_rs_mwh",
]

DEDUP_KEYS = ["trade_date", "hour", "session_id", "time_block"]

"""RTM market and schema constants."""

BLOCKS_PER_DAY = 96
BLOCK_MINUTES = 15
FORECAST_HORIZON = 1440

# ── Zone classification thresholds (Rs/MWh) ──────────────────────────
ZONE_GREEN_MAX = 3000.0
ZONE_YELLOW_MAX = 6000.0

ZONE_GREEN = "GREEN"
ZONE_YELLOW = "YELLOW"
ZONE_RED = "RED"


def classify_zone(mcp: float) -> str:
    """Classify an MCP value into GREEN / YELLOW / RED."""
    if mcp < ZONE_GREEN_MAX:
        return ZONE_GREEN
    if mcp < ZONE_YELLOW_MAX:
        return ZONE_YELLOW
    return ZONE_RED


EXPECTED_COLUMNS = [
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

COLUMN_MAP = {
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

FEATURE_COLUMNS = [
    "hour",
    "session_id",
    "time_block",
    "purchase_bid_mw",
    "sell_bid_mw",
    "mcv_mw",
    "scheduled_volume_mw",
    "bid_imbalance_mw",
    "bid_ratio",
    "bid_spread",
    "bid_ratio_ma4",
    "bid_ratio_ma16",
    "volume_fill_ratio",
    "sell_bid_ma4",
    "sell_bid_drop",
    "purchase_bid_ma4",
    "purchase_surge",
    "mcp_lag_1",
    "mcp_lag_4",
    "mcp_lag_96",
    "mcp_roll_mean_4",
    "mcp_roll_mean_96",
    "mcp_roll_std_4",
    "hour_sin",
    "hour_cos",
    "block_sin",
    "block_cos",
    "day_of_week",
    "is_weekend",
    "horizon",
]

TARGET_COLUMN = "mcp_rs_mwh"

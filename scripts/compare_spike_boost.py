import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iex_forecast.application.forecast_engine import ForecastGenerationService
import pandas as pd

svc = ForecastGenerationService()
svc._ensure_loaded()
history = svc._get_market_state()
origin_ts = history["block_timestamp"].iloc[-1]
if not hasattr(origin_ts, 'tzinfo') or origin_ts.tzinfo is None:
    origin_ts = pd.to_datetime(origin_ts).tz_localize('Asia/Kolkata')
blocks = svc._recursive_forecast(history, origin_ts, 96)
df = pd.DataFrame(blocks)
df["block_timestamp"] = pd.to_datetime(df["forecast_timestamp"])
df["mcp"] = df["predicted_mcp"]
df["block_number"] = df["block_timestamp"].dt.hour * 4 + df["block_timestamp"].dt.minute // 15 + 1

# Zone breakdown
zones = {
    "Solar   (blocks 33-60, 8am-3pm)":  df[df["block_number"].between(33, 60)],
    "Evening (blocks 68-88, 5pm-10pm)": df[df["block_number"].between(68, 88)],
    "Morning (blocks 1-32, midnight-8am)": df[df["block_number"].between(1, 32)],
}

print("=== 24h Forecast Zone Summary ===")
for label, zone_df in zones.items():
    print(f"\n{label}")
    print(f"  mean : {zone_df['mcp'].mean():>8.1f} Rs")
    print(f"  max  : {zone_df['mcp'].max():>8.1f} Rs")
    print(f"  min  : {zone_df['mcp'].min():>8.1f} Rs")

# Spike classifier activity
if "spike_probability" in df.columns:
    spiking = df[df["spike_probability"] > 0.70]
    print(f"\n=== Spike Classifier Activity ===")
    print(f"Blocks with spike_prob > 0.70 : {len(spiking)}")
    print(f"Their block numbers           : {sorted(spiking['block_number'].tolist())}")
    print(f"Their MCP range               : {spiking['mcp'].min():.0f} – {spiking['mcp'].max():.0f} Rs")
else:
    print("\nWARNING: spike_probability not in output — check routes.py schema")

# Peak block
peak = df.loc[df['mcp'].idxmax()]
print(f"\n=== Peak Block ===")
print(f"Block {int(peak['block_number'])} at {peak['block_timestamp']} → {peak['mcp']:.1f} Rs")

import pandas as pd
import numpy as np

df = pd.read_csv("demo/demo_7d_rtm_forecast_v3.csv")
df["date"] = pd.to_datetime(df["block_timestamp"]).dt.date
df["block_number"] = pd.to_numeric(df["block_number"])

print("=== 7-Day RTM Forecast Validation ===")
print(f"{'Date':<12} {'Day':<4} {'Solar':>7} {'EveMax':>8} {'PeakBlk':>8} {'Peak':>7} {'Min':>7}")
print("-" * 60)

for date, day_df in df.groupby("date"):
    solar   = day_df[day_df["block_number"].between(33, 60)]
    evening = day_df[day_df["block_number"].between(68, 96)]
    peak    = day_df.loc[day_df["mcp"].idxmax()]
    dow     = pd.to_datetime(str(date)).strftime("%a")
    print(f"{str(date):<12} {dow:<4} "
          f"{solar['mcp'].mean():>7.0f} "
          f"{evening['mcp'].max():>8.0f} "
          f"{int(peak['block_number']):>8} "
          f"{day_df['mcp'].max():>7.0f} "
          f"{day_df['mcp'].min():>7.0f}")

print(f"\nOverall: max={df['mcp'].max():.0f}  min={df['mcp'].min():.0f}  mean={df['mcp'].mean():.0f}")
print(f"Zones: {df['zone'].value_counts().to_dict()}")

# Pass/Fail checks
solar_ok   = df[df["block_number"].between(33,60)]["mcp"].mean() < 1500
evening_ok = df[df["block_number"].between(68,96)]["mcp"].max() > 7000
peak_ok    = df.loc[df["mcp"].idxmax(), "block_number"] in range(80, 97)
min_ok     = df["mcp"].min() < 1200
weekend_ok = (
    df[pd.to_datetime(df["block_timestamp"]).dt.dayofweek >= 5]["mcp"].mean() >
    df[pd.to_datetime(df["block_timestamp"]).dt.dayofweek < 5]["mcp"].mean()
)

print(f"\n=== Pass/Fail ===")
print(f"Solar mean < 1500 Rs:          {'PASS' if solar_ok   else 'FAIL'}")
print(f"Evening max > 7000 Rs:         {'PASS' if evening_ok else 'FAIL'}")
print(f"Peak block in 80-96:           {'PASS' if peak_ok    else 'FAIL'}")
print(f"Min < 1200 Rs (solar floor):   {'PASS' if min_ok     else 'FAIL'}")
print(f"Weekend > Weekday price:       {'PASS' if weekend_ok else 'FAIL'}")

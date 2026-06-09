import requests
import pandas as pd
from pathlib import Path

LAT, LON = 12.9716, 77.5946
OUT = Path("data/external/weather_bengaluru.parquet")
Out = Path("data/external")
Out.mkdir(parents=True, exist_ok=True)

# Historical
hist = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": LAT, "longitude": LON,
    "hourly": "temperature_2m,cloudcover,windspeed_10m,precipitation",
    "timezone": "Asia/Kolkata",
    "start_date": "2024-01-01",
    "end_date": "2026-06-06",
}).json()

# 7-day forecast (Deterministic)
fcast = requests.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": LAT, "longitude": LON,
    "hourly": "temperature_2m,cloudcover,windspeed_10m,precipitation",
    "timezone": "Asia/Kolkata",
    "forecast_days": 14,
}).json()

# 7-day forecast (Ensemble - for spread)
ensemble_fcast = requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={
    "latitude": LAT, "longitude": LON,
    "hourly": "temperature_2m",
    "models": "icon_seamless",
    "timezone": "Asia/Kolkata",
    "forecast_days": 14,
}).json()

# Combine
df_h = pd.DataFrame(hist["hourly"])
df_f = pd.DataFrame(fcast["hourly"])
df = pd.concat([df_h, df_f]).drop_duplicates("time").reset_index(drop=True)

# Calculate Ensemble Spread (if available)
if "hourly" in ensemble_fcast and "temperature_2m_member01" in ensemble_fcast["hourly"]:
    ens_df = pd.DataFrame(ensemble_fcast["hourly"])
    member_cols = [c for c in ens_df.columns if "member" in c]
    if member_cols:
        ens_df["temp_ensemble_spread"] = ens_df[member_cols].std(axis=1)
        df = df.merge(ens_df[["time", "temp_ensemble_spread"]], on="time", how="left")

if "temp_ensemble_spread" not in df.columns:
    df["temp_ensemble_spread"] = 0.0  # Fallback
    
df["temp_ensemble_spread"] = df["temp_ensemble_spread"].fillna(0.0)

df["time"] = pd.to_datetime(df["time"])
df.to_parquet(OUT)
print(f"Saved {len(df)} hourly rows to {OUT}")
print(df.tail(3))

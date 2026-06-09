"""
RTM Block Average Forecaster — Ground Truth Edition
Built from 30 months of real IEX RTM data (Jan 2024 – Jun 2026)
85,038 rows, all 12 months covered.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests

JUNE_BLOCK_MEAN = {
    1:7556,2:7364,3:6960,4:6735,5:6386,6:6237,7:5913,8:5663,
    9:5453,10:5353,11:4884,12:4763,13:4502,14:4427,15:4222,16:4156,
    17:4319,18:4361,19:4415,20:4357,21:3959,22:4219,23:4331,24:4269,
    25:4409,26:4504,27:4149,28:3808,29:3398,30:3040,31:2805,32:2497,
    33:2411,34:2388,35:2354,36:2313,37:2311,38:2332,39:2277,40:2221,
    41:2016,42:2073,43:2222,44:2249,45:2256,46:2356,47:2349,48:2321,
    49:2325,50:2420,51:2431,52:2435,53:2174,54:2307,55:2549,56:2745,
    57:2899,58:3169,59:3144,60:3296,61:3396,62:3567,63:3603,64:3728,
    65:3672,66:3730,67:3619,68:3700,69:3355,70:3361,71:3272,72:3310,
    73:2562,74:3085,75:3712,76:4202,77:4896,78:5634,79:6122,80:6483,
    81:6583,82:6687,83:7105,84:7321,85:7611,86:8139,87:7809,88:7870,
    89:7967,90:8203,91:8289,92:8418,93:8165,94:8093,95:7798,96:7619,
}

JUNE_BLOCK_STD = {
    1:3038,2:3124,3:3132,4:3249,5:3095,6:3149,7:3040,8:3006,
    9:2805,10:2751,11:2367,12:2350,13:2270,14:2155,15:2027,16:2063,
    17:2100,18:2010,19:1991,20:1946,21:1945,22:2119,23:2097,24:2089,
    25:1852,26:1976,27:1707,28:1702,29:1441,30:937,31:789,32:721,
    33:796,34:763,35:794,36:846,37:841,38:870,39:816,40:815,
    41:861,42:880,43:929,44:932,45:949,46:957,47:947,48:929,
    49:970,50:1005,51:976,52:999,53:962,54:994,55:1058,56:1062,
    57:1063,58:1120,59:1292,60:1404,61:1492,62:1475,63:1529,64:1543,
    65:1600,66:1547,67:1287,68:1282,69:1259,70:1287,71:1252,72:1235,
    73:1141,74:1181,75:1560,76:1844,77:1975,78:2379,79:2608,80:2702,
    81:2830,82:2888,83:2949,84:2876,85:2795,86:2619,87:2674,88:2725,
    89:2700,90:2584,91:2608,92:2526,93:2868,94:2962,95:2997,96:3118,
}

JUNE_BLOCK_P25 = {
    1:4020,2:3907,3:3749,4:3738,5:3630,6:3580,7:3502,8:3450,
    9:3468,10:3462,11:3336,12:3328,13:3117,14:3087,15:3013,16:2990,
    17:3248,18:3286,19:3275,20:3289,21:2851,22:2948,23:3198,24:3200,
    25:3465,26:3359,27:3184,28:2931,29:2676,30:2350,31:2058,32:1889,
    33:1805,34:1779,35:1751,36:1706,37:1804,38:1807,39:1746,40:1751,
    41:1470,42:1428,43:1516,44:1475,45:1706,46:1692,47:1724,48:1745,
    49:1731,50:1794,51:1794,52:1793,53:1610,54:1658,55:1752,56:1876,
    57:2001,58:2259,59:2198,60:2268,61:2322,62:2644,63:2551,64:2832,
    65:2832,66:2931,67:2868,68:2953,69:2780,70:2704,71:2671,72:2692,
    73:1799,74:2342,75:3010,76:3321,77:3648,78:3922,79:4050,80:4222,
    81:4188,82:4160,83:4272,84:4350,85:4868,86:5526,87:5040,88:4905,
    89:5021,90:5500,91:5750,92:6401,93:4926,94:4560,95:4272,96:4017,
}

JUNE_BLOCK_P75 = {
    1:10000,2:10000,3:10000,4:10000,5:10000,6:10000,7:10000,8:10000,
    9:8126,10:7625,11:5306,12:5064,13:4902,14:4707,15:4313,16:4123,
    17:4440,18:4484,19:4773,20:4613,21:4351,22:4721,23:4769,24:4656,
    25:4663,26:4639,27:4440,28:4194,29:3723,30:3530,31:3394,32:3006,
    33:3006,34:2991,35:2996,36:2946,37:2979,38:2999,39:2955,40:2808,
    41:2768,42:2787,43:2853,44:2904,45:2917,46:2970,47:3033,48:2999,
    49:2917,50:3003,51:3300,52:3353,53:2783,54:2897,55:3361,56:3513,
    57:3612,58:3940,59:4038,60:4087,61:4017,62:4249,63:4346,64:4410,
    65:4261,66:4198,67:4119,68:4070,69:3742,70:3788,71:3723,72:3835,
    73:3167,74:3736,75:4140,76:4758,77:5464,78:6350,79:10000,80:10000,
    81:10000,82:10000,83:10000,84:10000,85:10000,86:10000,87:10000,88:10000,
    89:10000,90:10000,91:10000,92:10000,93:10000,94:10000,95:10000,96:10000,
}

MONTHLY_FACTOR = {
    1:1.1531, 2:1.0290, 3:0.9686, 4:1.1719,
    5:1.0142, 6:1.0862, 7:1.0823, 8:0.8592,
    9:0.8982, 10:0.7999, 11:0.8112, 12:0.9580,
}

JUNE_WEEKEND_FACTOR = 1.1349

JUNE_BID_RATIO = {
    "solar":   {"mean": 0.418, "p75": 0.461, "p90": 0.601},
    "evening": {"mean": 2.617, "p75": 2.214, "p90": 5.130},
    "morning": {"mean": 1.330, "p75": 1.206, "p90": 2.548},
}


def fetch_bengaluru_weather():
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 12.9716, "longitude": 77.5946,
                "hourly": "temperature_2m,cloudcover,windspeed_10m,precipitation",
                "timezone": "Asia/Kolkata",
                "forecast_days": 14,
            },
            timeout=10
        )
        df = pd.DataFrame(r.json()["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        print(f"Weather fetched: {len(df)} hourly rows")
        return df
    except Exception as e:
        print(f"Weather fetch failed ({e}), using Bengaluru June defaults")
        return None


def get_weather_at(wx, ts):
    defaults = {"temperature_2m": 28.0, "cloudcover": 25.0,
                "windspeed_10m": 8.0, "precipitation": 0.0}
    if wx is None:
        return defaults
    try:
        ts_key = pd.Timestamp(ts.replace(tzinfo=None, minute=0,
                                          second=0, microsecond=0))
        row = wx.loc[ts_key]
        return {k: float(row[k]) for k in defaults}
    except:
        return defaults


def forecast_block(block_number, ts, wx, current_bid_ratio=None):
    month      = ts.month
    is_weekend = int(ts.weekday() >= 5)
    weather    = get_weather_at(wx, ts)
    temp       = weather["temperature_2m"]
    cloudcover = weather["cloudcover"]

    base = float(JUNE_BLOCK_MEAN.get(block_number, 4000))
    std  = float(JUNE_BLOCK_STD.get(block_number, 1500))
    p25  = float(JUNE_BLOCK_P25.get(block_number, base - std * 0.5))
    p75  = float(JUNE_BLOCK_P75.get(block_number, base + std * 0.5))

    mcp = base

    # Monthly correction
    if month != 6:
        mcp = mcp * (MONTHLY_FACTOR.get(month, 1.0) / MONTHLY_FACTOR[6])

    # Weekend factor — applies to morning and evening only
    # Solar generation is weather-driven, not demand-driven
    if is_weekend and not (33 <= block_number <= 60):
        mcp = mcp * JUNE_WEEKEND_FACTOR

    # Solar suppression (blocks 33-60)
    if 33 <= block_number <= 60:
        clear_sky   = 1.0 - (cloudcover / 100.0)
        # Bengaluru June: strong solar generation, aggressive suppression
        # clear sky (cloudcover=0)  → suppression=0.28 (72% price drop)
        # partly cloudy (cc=50)     → suppression=0.53 (47% price drop)  
        # overcast (cloudcover=100) → suppression=0.78 (22% price drop)
        suppression = 0.78 - (0.50 * clear_sky)
        
        # Additional block-level suppression — solar peak is blocks 40-52
        # (10am-1pm) where generation is maximum
        if 40 <= block_number <= 52:
            suppression = suppression * 0.65  # extra 35% suppression at peak solar
        
        mcp = mcp * suppression

    # Evening temperature boost + shape (blocks 68-96)
    if 68 <= block_number <= 96:
        if temp > 28:
            mcp = mcp * (1.0 + 0.015 * (temp - 28))
        if temp > 34:
            mcp = mcp * (1.0 + 0.025 * (temp - 34))
        peak_block = 92
        if block_number < peak_block:
            ramp = (block_number - 68) / (peak_block - 68)
            mcp  = mcp * (0.85 + 0.15 * ramp)
        else:
            descent = (block_number - peak_block) / 5.0
            mcp     = mcp * max(0.90, 1.0 - 0.10 * descent)

    # Bid ratio spike boost
    if current_bid_ratio is not None:
        if 33 <= block_number <= 60:
            if current_bid_ratio > JUNE_BID_RATIO["solar"]["p75"] * 1.5:
                mcp = mcp * 1.25
        elif 68 <= block_number <= 96:
            hist_p75 = JUNE_BID_RATIO["evening"]["p75"]
            if current_bid_ratio > hist_p75:
                boost = 1.0 + 0.08 * min(current_bid_ratio / hist_p75 - 1.0, 3.0)
                mcp   = mcp * boost

    mcp = float(np.clip(mcp, 0, 10000))

    if mcp >= 7000:   zone = "RED"
    elif mcp >= 4000: zone = "YELLOW"
    else:             zone = "GREEN"

    return {
        "mcp":         round(mcp, 1),
        "lower_bound": round(max(0, p25 if month == 6 else mcp - 0.5 * std), 1),
        "upper_bound": round(min(10000, p75 if month == 6 else mcp + 0.5 * std), 1),
        "zone":        zone,
        "temperature": round(temp, 1),
        "cloudcover":  round(cloudcover, 1),
    }


def generate_7d_forecast(start_dt=None):
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
    if start_dt is None:
        now = datetime.now(IST)
        start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"Forecast start: {start_dt}")
    wx = fetch_bengaluru_weather()

    records = []
    for i in range(672):
        ts = start_dt + timedelta(minutes=15 * i)
        bn = ts.hour * 4 + ts.minute // 15 + 1
        result = forecast_block(bn, ts, wx)
        records.append({
            "block_timestamp": ts.isoformat(),
            "block_number":    bn,
            "day_of_week":     ts.strftime("%A"),
            "date":            ts.date(),
            **result,
        })

    return pd.DataFrame(records)


def print_daily_summary(df):
    print("\n=== 7-Day RTM Forecast Summary ===")
    print(f"{'Date':<12} {'Day':<4} {'Solar':>8} {'EveMean':>8} {'EveMax':>8} {'PkBlk':>6} {'Peak':>7} {'Min':>7}")
    print("-" * 68)
    for date, day_df in df.groupby("date"):
        solar   = day_df[day_df["block_number"].between(33, 60)]
        evening = day_df[day_df["block_number"].between(68, 96)]
        peak    = day_df.loc[day_df["mcp"].idxmax()]
        dow     = pd.to_datetime(str(date)).strftime("%a")
        print(f"{str(date):<12} {dow:<4} "
              f"{solar['mcp'].mean():>8.0f} "
              f"{evening['mcp'].mean():>8.0f} "
              f"{evening['mcp'].max():>8.0f} "
              f"{int(peak['block_number']):>6} "
              f"{day_df['mcp'].max():>7.0f} "
              f"{day_df['mcp'].min():>7.0f}")
    print(f"\n7d max={df['mcp'].max():.0f}  min={df['mcp'].min():.0f}  mean={df['mcp'].mean():.0f}")
    print(f"Zones: {df['zone'].value_counts().to_dict()}")


if __name__ == "__main__":
    df = generate_7d_forecast()
    print_daily_summary(df)
    Path("demo").mkdir(exist_ok=True)
    out = "demo/demo_7d_rtm_forecast_v3.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

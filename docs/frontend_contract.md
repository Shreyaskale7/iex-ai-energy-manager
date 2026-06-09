# Frontend API Contract

This document provides schemas and endpoint specifications for the Frontend team connecting to the FastAPI backend.

## Base URL
`http://<SERVER_IP>:8000`

## Authentication Requirements
All data endpoints require an API key passed in the headers:
- **Header**: `X-API-Key`
- **Value**: `dev-api-key` (Configure via `.env`)

## Error Responses
Standard 4xx/5xx responses return JSON:
```json
{
  "detail": "No 24-Hour forecast available."
}
```

---

## 1. Market Status
**Endpoint**: `GET /market-status`
**Query Params**: `limit` (int, default=96)
**Response**:
```json
{
  "latest_blocks": [
    {
      "trade_date": "2026-06-05",
      "time_block": 96,
      "block_timestamp": "2026-06-05T23:45:00Z",
      "purchase_bid_mw": 12000.5,
      "sell_bid_mw": 14000.0,
      "mcv_mw": 11500.0,
      "mcp_rs_mwh": 3100.5
    }
  ]
}
```

## 2. Forecast Latest (Dynamic Horizon)
**Endpoint**: `GET /forecast/latest`
**Query Params**: `forecast_type` (String: "24-Hour", "7-Day", "30-Day". Default="24-Hour")
**Response**: `ForecastResponse`

## 3. Forecast Aliases (Day, Week, Month, Date)
- `GET /forecast/day` (Returns 96 blocks)
- `GET /forecast/week` (Returns 672 blocks)
- `GET /forecast/month` (Returns 2880 blocks with `skip`/`limit` pagination support)
- `GET /forecast/date/{date}` (e.g. `2026-06-05`)

**ForecastResponse Schema Example**:
```json
{
  "run_id": "7290565b-b9ea-4d5b-91fa-da10db8f9ba3",
  "origin_timestamp": "2026-06-04T14:15:00Z",
  "model_version": "ensemble-v2",
  "horizon_blocks": 96,
  "total_points": 96,
  "zone_summary": { "GREEN": 22, "YELLOW": 74, "RED": 0 },
  "points": [
    {
      "horizon": 1,
      "forecast_timestamp": "2026-06-04T14:30:00Z",
      "predicted_mcp": 3196.17,
      "zone": "YELLOW",
      "confidence": 0.7898,
      "spike_probability": 0.0997,
      "lower_bound": 2716.74,
      "upper_bound": 3675.59
    }
  ]
}
```

## 4. Alerts
**Endpoint**: `GET /alerts`
Returns paginated forecast points strictly where `zone == 'RED'` OR `spike_probability > 0.5`.

## 5. Smart Decision Engine
**Endpoint**: `POST /decision/schedule`
**Request Schema**:
```json
{
  "forecast_type": "24-Hour",
  "devices": [
    {
      "device_id": "HVAC_1",
      "name": "Central HVAC",
      "category": "Flexible",
      "power_kw": 150.0,
      "priority_level": 1
    }
  ],
  "thresholds": {
    "spike_prob_red_threshold": 0.5
  }
}
```

**Response Schema**:
```json
{
  "baseline_cost_rs": 12500.50,
  "optimized_cost_rs": 9800.00,
  "expected_savings_rs": 2700.50,
  "savings_percentage": 21.6,
  "recommended_schedule": [
    {
      "forecast_timestamp": "2026-06-04T14:30:00Z",
      "predicted_mcp": 3196.17,
      "effective_zone": "YELLOW",
      "total_load_kw": 150.0,
      "device_states": [
        {
          "device_id": "HVAC_1",
          "name": "Central HVAC",
          "category": "Flexible",
          "recommended_state": "ON"
        }
      ]
    }
  ]
}
```

# IEX RTM Electricity Price Forecasting System

Production-grade forecasting pipeline for Indian Energy Exchange (IEX) Real Time Market (RTM) **Market Clearing Price (MCP)** in Rs/MWh. Predicts the next **96 blocks** (24 hours at 15-minute resolution).

## Stack

- **Python** · Pandas · XGBoost · LightGBM · CatBoost
- **FastAPI** (inference API) · **Streamlit** (operations dashboard)
- **PostgreSQL** (canonical store) · **Docker Compose**

## Project layout

```
src/iex_forecast/
  config/          Settings & environment
  core/            Logging, exceptions
  domain/          Entities, constants, interfaces
  infrastructure/  PostgreSQL, file I/O
  data/            Ingestion, cleaning, persistence
  features/        Feature engineering for RTM blocks
  models/          Training, ensemble, registry
  inference/       96-block horizon predictor
  api/             FastAPI application
  dashboard/       Streamlit UI
scripts/           CLI: ingest, train, migrate
alembic/           Database migrations
tests/
data/raw/          Monthly RTM Excel files
data/processed/
data/models/       Serialized ensembles per horizon
```

## Quick start

1. Copy environment file and place Excel files under `data/raw/`:

   ```bash
   cp .env.example .env
   ```

2. Start services:

   ```bash
   docker compose up -d --build
   ```

3. Build master dataset, then ingest to DB and train:

   ```bash
   docker compose exec api alembic upgrade head
   python scripts/build_rtm_master.py
   docker compose exec api python scripts/ingest_raw.py
   docker compose exec api python scripts/train_models.py
   ```

   `build_rtm_master.py` reads `data/raw/*.xlsx`, validates, and writes:
   `data/processed/rtm_master.parquet`, `rtm_master.csv`, and `data_quality_report.json`.

4. Run exploratory analysis:

   ```bash
   pip install matplotlib seaborn pyarrow
   python scripts/run_eda.py
   ```

   Outputs plots and reports under `reports/eda/` (summary JSON, markdown report, 11 PNG charts).

5. Build forecasting features:

   ```bash
   python scripts/build_features.py
   ```

   Reads `data/processed/rtm_master.parquet` and writes `data/features/features.parquet`
   (lags, rolling stats, cyclical encodings, market-derived signals).

6. Train XGBoost (next 15-min MCP):

   ```bash
   python scripts/train_xgboost.py
   ```

   Uses TimeSeriesSplit + Optuna tuning, early stopping, SHAP explainability.
   Outputs `models/xgboost.pkl` and `reports/xgb_report.html`.

7. Train LightGBM and compare to XGBoost:

   ```bash
   python scripts/train_xgboost.py   # required first for comparison
   python scripts/train_lightgbm.py
   ```

   Outputs `models/lightgbm.pkl`, `reports/lgbm_report.html`, and
   `reports/model_comparison_report.html`.

8. Train CatBoost and weighted ensemble:

   ```bash
   python scripts/train_catboost.py
   python scripts/train_ensemble.py
   ```

   Ensemble optimizes weights on the **validation** split (non-negative, sum to 1).
   Outputs `models/ensemble.pkl` and `reports/ensemble_comparison_report.html`.

9. Recursive 96-block forecast (24h ahead):

   ```bash
   python scripts/run_forecast_96.py
   ```

   Uses current market state from `rtm_master.parquet` and writes `forecasts/forecast_96.csv`.

10. Train MCP spike classifier:

   ```bash
   python scripts/train_spike_classifier.py
   ```

   Spike = next-block MCP above 90th percentile (train split). Outputs `models/spike_classifier.pkl`
   and `reports/spike_classifier_report.html` (confusion matrix, precision, recall, F1, ROC AUC).

4. Open dashboard: http://localhost:8501  
   API docs: http://localhost:8000/docs

## Forecast horizon

| Parameter | Value |
|-----------|-------|
| Block length | 15 minutes |
| Blocks per day | 96 |
| Forecast horizon | 96 blocks (24h ahead) |
| Target | MCP (Rs/MWh) |

## API

- `GET /health` — liveness
- `GET /forecast/latest` — latest 96-block MCP forecast
- `POST /forecast/run` — trigger inference from DB state
- `GET /metrics/model` — last training run metadata

All mutating routes require header `X-API-Key` matching `API_KEY` in `.env`.

## License

Proprietary — internal use.

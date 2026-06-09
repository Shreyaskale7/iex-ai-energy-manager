# IEX AI Energy Manager - Handoff Guide

Welcome! This repository contains the AI-driven forecast engine for predicting the Indian Energy Exchange (IEX) Real-Time Market (RTM) Market Clearing Prices (MCP). 

This document explains the architecture, how to run the pipeline, and where everything lives.

## 📁 Repository Structure

You will find the following core directories in this handoff package:

- **`src/iex_forecast/`**: The core application logic. This includes the `ForecastGenerationService`, feature pipelines, and domain entities.
- **`scripts/`**: Orchestration scripts for running the system (ingestion, training, forecasting).
- **`models/`**: The trained AI models. **Do not delete this directory!** It contains:
  - `ensemble.pkl`: The base short-term predictive weights.
  - `spike_classifier.pkl`: The model predicting the probability of extreme price spikes.
  - `direct/`: A subdirectory containing 21 specific direct-horizon models (LightGBM) trained specifically to capture diurnal realism (morning/evening peaks).
- **`data/`**: The local database.
  - `data/processed/rtm_master.parquet`: The unified historical market data.
  - `data/features/features.parquet`: The engineered features ready for inference.
  - `data/processed/residual_profile.parquet`: The intraday residual profile used to ensure the forecasts exhibit realistic daily cyclic behavior.
- **`forecasts/`**: Where the final generated forecasts are saved (as CSV files).
- **`demo/`**: Contains demonstration CSVs (like the `demo_7d_forecast_june7_14.csv`).
- **`reports/`**: HTML analysis reports generated during R&D detailing multi-horizon performance and forecast realism improvements.

## 🚀 How to Run the System

**Important:** The project uses the `src/` directory as its root package. Whenever running scripts, you must set the `PYTHONPATH`.

### 1. Generating a Forecast
To generate a new forecast (e.g., 24-hour, 7-day, or 30-day), run:

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="src"; python scripts/run_forecast.py --type 7d
```

**Linux/Mac:**
```bash
PYTHONPATH="src" python scripts/run_forecast.py --type 7d
```

Valid `--type` arguments are: `24h`, `7d`, `30d`, or `all`.
The output will be saved in the `forecasts/` directory.

### 2. Retraining Models
If you acquire new historical data and need to retrain the models:

**Train base ensemble models (1-96 horizons):**
```powershell
$env:PYTHONPATH="src"; python scripts/train_models.py
```

**Train direct interpolation models (Required for accurate evening peaks):**
```powershell
$env:PYTHONPATH="src"; python scripts/train_models.py --direct-only
```

### 3. Rebuilding the Residual Profile
If you retrain models, you should rebuild the residual profile so the forecast realism stays accurate:
```powershell
$env:PYTHONPATH="src"; python reports/forecast_realism_improvement.py
```

## 🧠 Architecture Overview
The forecast engine uses a **Hybrid Multi-Horizon Architecture**:
1. **Short-Term (Blocks 1-96)**: Uses a recursive ensemble of LightGBM, XGBoost, and CatBoost.
2. **Medium/Long-Term**: Uses Direct-Horizon forecasting. We explicitly train models at critical anchor blocks (like block 8, 78, 80, 82, 88, 92) where morning and evening peaks occur. 
3. **Reconstruction**: The `ForecastGenerationService` predicts values at the anchors, interpolates between them, and then applies a residual profile to reconstruct realistic daily market volatility.

Good luck!

# System Architecture

This outlines the high-level infrastructure of the AI-Powered Smart Energy Management System.

## Architecture Layers

### 1. Data & Persistence Layer
- **PostgreSQL Database**: Serves as the central state machine and historic ledger. 
  - Tracks actual market execution.
  - Archives AI forecast runs.
  - Maintains accuracy evaluation logs.
- **Local Storage (CSV/Parquet)**: Retained for rapid feature engineering, model training loops, and cold backups.

### 2. Analytical Engine
- **Recursive Forecast Engine (24-Hour)**: Steps forward block-by-block using an Ensemble of LightGBM, XGBoost, and CatBoost models. Includes dynamic feature reconstruction.
- **Direct-Horizon Engine (7-Day / 30-Day)**: Uses independent LightGBM models acting as anchor points, with linear interpolation spanning the 2,880 monthly blocks for high-speed generation.
- **Spike Classifier**: XGBoost Binary Classifier injecting probability metadata for volatility.

### 3. Application Layer (FastAPI)
- **FastAPI Core**: Standardized routing, validation (Pydantic), dependency injection, and automatic OpenAPI generation.
- **Decision Engine Service**: Business logic mapping forecast data onto device profiles (Critical, Flexible, Deferrable) to synthesize an optimal ON/OFF operation schedule.

### 4. Edge & IoT Layer
- **ESP32 Microcontrollers**: Listen to the API (or webhooks/MQTT in future iterations) mapping the `recommended_state` (ON/OFF) array to physical GPIO pins.
- **Relay Boards**: Physically connect/disconnect electrical loads based on ESP32 signaling.

### 5. Frontend Dashboard
- **React/Next.js (TBD)**: Pulls market status, zones, alerts, and schedules to visualize daily operations and estimated financial savings.

# Deployment Guide

Instructions for standing up the backend and database infrastructure.

## 1. Environment Requirements
- Python 3.11+
- Docker & Docker Compose
- `uv` or `pip` package manager

## 2. Database Setup

1. Start the PostgreSQL Docker container:
```bash
docker compose up -d postgres
```
*(Alternatively, run manually)*:
```bash
docker run --name iex-postgres -e POSTGRES_USER=iex_user -e POSTGRES_PASSWORD=change_me_secure_password -e POSTGRES_DB=iex_rtm -p 5432:5432 -d postgres:15
```

2. Run Alembic Database Migrations:
```bash
alembic upgrade head
```

## 3. Environment Variables

Create a `.env` file in the project root:
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=iex_rtm
POSTGRES_USER=iex_user
POSTGRES_PASSWORD=change_me_secure_password

APP_ENV=production
LOG_LEVEL=INFO
API_KEY=dev-api-key
```

## 4. FastAPI Startup

Launch the production web server:
```bash
uvicorn iex_forecast.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Check the health endpoint to verify connectivity:
`http://localhost:8000/health`

## 5. Forecast Generation Jobs

Forecasts are generated via the CLI. In a production environment, you should schedule this via CRON or a task scheduler (like Celery or APScheduler) to run daily at a specific time (e.g., 23:00).

```bash
# Generate all horizons (24h, 7d, 30d)
python scripts/run_forecast.py --type all
```

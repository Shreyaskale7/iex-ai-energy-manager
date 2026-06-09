FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml README.md alembic.ini ./
COPY alembic ./alembic
COPY src ./src
COPY scripts ./scripts

RUN mkdir -p data/raw data/processed data/models

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "iex_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS dashboard
EXPOSE 8501
CMD ["streamlit", "run", "src/iex_forecast/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

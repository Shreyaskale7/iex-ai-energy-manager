"""Streamlit dashboard for RTM MCP forecasting."""

import os
from datetime import datetime

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.getenv("FORECAST_API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", os.getenv("api_key", "dev-api-key-change-in-production"))

st.set_page_config(
    page_title="IEX RTM Forecast",
    page_icon="⚡",
    layout="wide",
)

st.title("IEX RTM Electricity Price Forecast")
st.caption("96-block horizon · 15-minute resolution · MCP (Rs/MWh)")


@st.cache_data(ttl=60)
def fetch_health() -> dict:
    response = httpx.get(f"{API_BASE}/health", timeout=30.0)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=120)
def fetch_latest_forecast(forecast_type: str = "24-Hour") -> dict | None:
    response = httpx.get(f"{API_BASE}/forecast/latest?forecast_type={forecast_type}", timeout=60.0)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def fetch_training_metrics() -> dict | None:
    response = httpx.get(f"{API_BASE}/metrics/model", timeout=30.0)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def trigger_forecast(forecast_type: str = "24h") -> dict:
    response = httpx.post(
        f"{API_BASE}/forecast/generate",
        headers={"X-API-Key": API_KEY},
        json={"forecast_type": forecast_type},
        timeout=600.0,
    )
    response.raise_for_status()
    results = response.json()
    return results[0] if results else {}


col_status, col_action = st.columns([3, 1])

with col_status:
    try:
        health = fetch_health()
        st.success(f"API online · {health['app_name']} · {health['environment']}")
    except httpx.HTTPError as exc:
        st.error(f"API unreachable at {API_BASE}: {exc}")

with col_action:
    button_text = f"Run {forecast_type} forecast"
    if st.button(button_text, type="primary"):
        with st.spinner(f"Generating {forecast_type} forecast…"):
            try:
                # Map Streamlit dropdown to API schema
                type_map = {"24-Hour": "24h", "7-Day": "7d", "30-Day": "30d"}
                api_type = type_map.get(forecast_type, "24h")
                
                result = trigger_forecast(api_type)
                st.cache_data.clear()
                st.session_state["last_forecast"] = result
                st.success(f"Forecast complete · {result.get('total_blocks', 0)} blocks")
            except httpx.HTTPError as exc:
                st.error(f"Forecast failed: {exc}")

metrics = fetch_training_metrics()
if metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean MAE", f"{metrics.get('mean_mae', 0):.2f}")
    m2.metric("Mean RMSE", f"{metrics.get('mean_rmse', 0):.2f}")
    m3.metric("Train rows", metrics.get("train_rows", "—"))
    m4.metric("Model version", metrics.get("model_version", "—"))

st.sidebar.header("Configuration")
forecast_type = st.sidebar.selectbox(
    "Forecast Horizon",
    ["24-Hour", "7-Day", "30-Day"]
)
st.sidebar.write(f"API: `{API_BASE}`")
st.sidebar.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

forecast_payload = fetch_latest_forecast(forecast_type)

if not forecast_payload or not forecast_payload.get("points"):
    st.info("No forecast in database. Ingest data, train models, then run a forecast.")
    st.stop()

df = pd.DataFrame(forecast_payload["points"])
df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"])

tab_chart, tab_table, tab_profile, tab_spikes = st.tabs(["Price curve", "Data table", "Block profile", "Spike Risk"])

with tab_chart:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["forecast_timestamp"],
            y=df["predicted_mcp"] if "predicted_mcp" in df.columns else df.get("mcp_forecast_rs_mwh", []),
            mode="lines+markers",
            name="MCP Forecast",
            line=dict(color="#f59e0b", width=2),
        )
    )
    if "lower_bound" in df.columns and "upper_bound" in df.columns and df["lower_bound"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=pd.concat([df["forecast_timestamp"], df["forecast_timestamp"][::-1]]),
                y=pd.concat([df["upper_bound"], df["lower_bound"][::-1]]),
                fill="toself",
                fillcolor="rgba(245,158,11,0.15)",
                line=dict(width=0),
                name="Uncertainty band",
                showlegend=True,
            )
        )
    
    if "spike_probability" in df.columns:
        spikes = df[df["spike_probability"] > 0.5]
        if not spikes.empty:
            fig.add_trace(
                go.Scatter(
                    x=spikes["forecast_timestamp"],
                    y=spikes["predicted_mcp"] if "predicted_mcp" in df.columns else spikes.get("mcp_forecast_rs_mwh", []),
                    mode="markers",
                    name="High Risk Spikes",
                    marker=dict(color="red", size=10, symbol="triangle-up"),
                )
            )

    fig.update_layout(
        title=f"{forecast_type} MCP forecast",
        xaxis_title="Time (IST)",
        yaxis_title="MCP (Rs/MWh)",
        height=480,
        template="plotly_dark",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_table:
    display = df.copy()
    display["forecast_timestamp"] = display["forecast_timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(display, use_container_width=True, hide_index=True)

with tab_profile:
    st.subheader("Hourly average forecast profile")
    hourly = df.copy()
    hourly["hour"] = hourly["forecast_timestamp"].dt.hour
    mcp_col = "predicted_mcp" if "predicted_mcp" in df.columns else "mcp_forecast_rs_mwh"
    profile = hourly.groupby("hour")[mcp_col].mean().reset_index()
    st.bar_chart(profile, x="hour", y=mcp_col)

with tab_spikes:
    st.subheader("High Risk Spike Periods")
    if "spike_probability" in df.columns:
        spikes = df[df["spike_probability"] > 0.5].copy()
        if spikes.empty:
            st.success("No high risk spikes detected.")
        else:
            spikes["forecast_timestamp"] = spikes["forecast_timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(spikes[["forecast_timestamp", "predicted_mcp", "spike_probability"]].sort_values("spike_probability", ascending=False), hide_index=True)
    else:
        st.info("Spike probability data not available.")

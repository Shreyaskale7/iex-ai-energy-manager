"""Unit tests for the Smart Load Decision Engine."""

import pytest
from datetime import datetime

from iex_forecast.domain.devices import DeviceProfile, DeviceCategory, DeviceState
from iex_forecast.application.decision_engine import DecisionEngineService, CustomThresholds

@pytest.fixture
def service():
    return DecisionEngineService()

@pytest.fixture
def devices():
    return [
        DeviceProfile(device_id="d1", name="Life Support", category=DeviceCategory.CRITICAL, power_kw=5.0),
        DeviceProfile(device_id="d2", name="HVAC", category=DeviceCategory.FLEXIBLE, power_kw=20.0),
        DeviceProfile(device_id="d3", name="EV Charger", category=DeviceCategory.DEFERRABLE, power_kw=15.0),
    ]

def test_evaluate_state_green(service, devices):
    thresholds = CustomThresholds()
    for d in devices:
        state = service.evaluate_state(predicted_mcp=2000.0, forecast_zone="GREEN", spike_prob=0.01, device=d, thresholds=thresholds)
        assert state == DeviceState.ON

def test_evaluate_state_yellow(service, devices):
    thresholds = CustomThresholds()
    # In yellow: Critical ON, Flexible ON, Deferrable OFF
    res = {d.category: service.evaluate_state(3500.0, "YELLOW", 0.05, d, thresholds) for d in devices}
    assert res[DeviceCategory.CRITICAL] == DeviceState.ON
    assert res[DeviceCategory.FLEXIBLE] == DeviceState.ON
    assert res[DeviceCategory.DEFERRABLE] == DeviceState.OFF

def test_evaluate_state_red(service, devices):
    thresholds = CustomThresholds()
    # In red: Critical ON, Flexible OFF, Deferrable OFF
    res = {d.category: service.evaluate_state(6500.0, "RED", 0.1, d, thresholds) for d in devices}
    assert res[DeviceCategory.CRITICAL] == DeviceState.ON
    assert res[DeviceCategory.FLEXIBLE] == DeviceState.OFF
    assert res[DeviceCategory.DEFERRABLE] == DeviceState.OFF

def test_evaluate_state_spike_override(service, devices):
    # Zone is GREEN, but spike probability is 0.9 (above 0.5 threshold) -> behaves as RED
    thresholds = CustomThresholds(spike_prob_red_threshold=0.5)
    res = {d.category: service.evaluate_state(2500.0, "GREEN", 0.9, d, thresholds) for d in devices}
    assert res[DeviceCategory.CRITICAL] == DeviceState.ON
    assert res[DeviceCategory.FLEXIBLE] == DeviceState.OFF
    assert res[DeviceCategory.DEFERRABLE] == DeviceState.OFF

def test_evaluate_state_manual_override(service):
    # Device is deferrable and manual_override is ON in a RED zone
    d = DeviceProfile(device_id="d3", name="EV Charger", category=DeviceCategory.DEFERRABLE, power_kw=15.0, manual_override=DeviceState.ON)
    state = service.evaluate_state(7000.0, "RED", 0.1, d, CustomThresholds())
    assert state == DeviceState.ON

def test_calculate_schedule(service, devices):
    points = [
        {"forecast_timestamp": datetime(2026, 6, 5, 10, 0), "predicted_mcp": 2000.0, "zone": "GREEN", "spike_probability": 0.0},
        {"forecast_timestamp": datetime(2026, 6, 5, 10, 15), "predicted_mcp": 3500.0, "zone": "YELLOW", "spike_probability": 0.0},
        {"forecast_timestamp": datetime(2026, 6, 5, 10, 30), "predicted_mcp": 7000.0, "zone": "RED", "spike_probability": 0.0},
    ]
    
    result = service.calculate_schedule(points, devices)
    
    assert "recommended_schedule" in result
    assert len(result["recommended_schedule"]) == 3
    
    # GREEN block load = 5 + 20 + 15 = 40 kW
    assert result["recommended_schedule"][0]["total_load_kw"] == 40.0
    # YELLOW block load = 5 + 20 = 25 kW
    assert result["recommended_schedule"][1]["total_load_kw"] == 25.0
    # RED block load = 5 kW
    assert result["recommended_schedule"][2]["total_load_kw"] == 5.0
    
    assert result["expected_savings_rs"] > 0

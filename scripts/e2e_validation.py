import logging
from fastapi.testclient import TestClient
from iex_forecast.api.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("e2e_validation")

def run_validation():
    client = TestClient(app)
    headers = {"X-API-Key": "dev-api-key"}
    
    logger.info("1. Testing Health...")
    r = client.get("/health")
    assert r.status_code == 200
    
    logger.info("2. Fetching Market Status...")
    r = client.get("/market-status?limit=5", headers=headers)
    assert r.status_code == 200
    assert len(r.json()["latest_blocks"]) > 0
    
    logger.info("3. Generating 24-Hour Forecast...")
    r = client.post("/forecast/generate", json={"forecast_type": "24h"}, headers=headers)
    assert r.status_code == 200
    assert r.json()[0]["total_blocks"] == 96
    
    logger.info("4. Fetching Latest 24-Hour Forecast...")
    r = client.get("/forecast/latest?forecast_type=24-Hour", headers=headers)
    assert r.status_code == 200
    assert r.json()["total_points"] == 96
    
    logger.info("5. Testing Decision Engine...")
    payload = {
        "forecast_type": "24-Hour",
        "devices": [
            {"device_id": "c1", "name": "Server Rack", "category": "Critical", "power_kw": 10.0, "priority_level": 1},
            {"device_id": "f1", "name": "HVAC", "category": "Flexible", "power_kw": 50.0, "priority_level": 2},
            {"device_id": "d1", "name": "EV Fleet", "category": "Deferrable", "power_kw": 100.0, "priority_level": 5}
        ]
    }
    r = client.post("/decision/schedule", json=payload, headers=headers)
    assert r.status_code == 200
    
    data = r.json()
    assert "recommended_schedule" in data
    assert "expected_savings_rs" in data
    assert len(data["recommended_schedule"]) == 96
    
    logger.info(f"Success! Expected Savings: Rs {data['expected_savings_rs']}")
    
    import csv
    with open("demo/demo_device_schedule.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["forecast_timestamp", "predicted_mcp", "effective_zone", "total_load_kw", "devices_on"])
        for block in data["recommended_schedule"]:
            devices_on = [d["name"] for d in block["device_states"] if d["recommended_state"] == "ON"]
            writer.writerow([
                block["forecast_timestamp"],
                block["predicted_mcp"],
                block["effective_zone"],
                block["total_load_kw"],
                ", ".join(devices_on)
            ])
            
    logger.info("Validation complete and demo schedule saved.")

if __name__ == "__main__":
    run_validation()

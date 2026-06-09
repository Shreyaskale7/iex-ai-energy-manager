"""Test load control decision loop against live forecast."""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path for direct service import (if API not running)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests
import pandas as pd

# Configuration
API_URL = "http://localhost:8000/forecast/day"
PRICE_TRIGGER_HIGH = 4000
PRICE_TRIGGER_VERY_HIGH = 7000
PRICE_RESUME = 2500
TRIGGER_LEAD_BLOCKS = 4

def get_forecast_from_api():
    """Try to fetch forecast from live FastAPI server."""
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if "blocks" in data:
            return data["blocks"]
        else:
            print(f"Unexpected response format: {list(data.keys())}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"API not reachable at {API_URL}")
        return None
    except Exception as e:
        print(f"API error: {e}")
        return None


def get_forecast_from_service():
    """Fallback: generate forecast directly using service."""
    try:
        from iex_forecast.application.forecast_engine import ForecastGenerationService
        from pathlib import Path
        import pandas as pd
        
        print("Using direct forecast service (API not running)...")
        svc = ForecastGenerationService()
        result = svc.generate_24h()
        
        # Read from the CSV that was just written
        latest_csv = Path("forecasts/forecast_20260604_194500.csv")
        if latest_csv.exists():
            df = pd.read_csv(latest_csv)
        else:
            # Try archive
            archive_files = sorted(Path("forecasts/archive").glob("*24h.csv"))
            if archive_files:
                df = pd.read_csv(archive_files[-1])
            else:
                print("No forecast CSV found")
                return None
        
        blocks = []
        for _, row in df.iterrows():
            blocks.append({
                "horizon": int(row["horizon"]),
                "predicted_mcp": float(row["predicted_mcp"]),
                "forecast_timestamp": str(row["forecast_timestamp"]),
                "confidence": float(row.get("confidence", 0.0)),
                "zone": row.get("zone", "N/A")
            })
        return blocks
    except Exception as e:
        print(f"Forecast service error: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_control_decision(blocks, lead_blocks=TRIGGER_LEAD_BLOCKS):
    """Look ahead N blocks and decide load state."""
    if not blocks:
        return None, None, "NO_DATA"
    
    lookahead = blocks[:min(lead_blocks, len(blocks))]
    max_forecast_price = max(b.get("predicted_mcp", 0) for b in lookahead)
    
    if max_forecast_price >= PRICE_TRIGGER_VERY_HIGH:
        decision = "ALL_OFF"
    elif max_forecast_price >= PRICE_TRIGGER_HIGH:
        decision = "SHED"
    elif max_forecast_price >= PRICE_RESUME:
        decision = "NORMAL"
    else:
        decision = "LOW_PRICE"
    
    return decision, max_forecast_price, "OK"


def main():
    print("=" * 70)
    print("LOAD CONTROL DECISION TEST")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Lead time: {TRIGGER_LEAD_BLOCKS} blocks (1 hour)")
    print(f"Thresholds:")
    print(f"  - Resume: <= {PRICE_RESUME} Rs/MWh")
    print(f"  - Trigger (SHED): >= {PRICE_TRIGGER_HIGH} Rs/MWh")
    print(f"  - Trigger (ALL_OFF): >= {PRICE_TRIGGER_VERY_HIGH} Rs/MWh")
    print()
    
    # Try API first, fall back to service
    print("Fetching forecast...")
    blocks = get_forecast_from_api()
    
    if blocks is None:
        print("Falling back to direct forecast service...")
        blocks = get_forecast_from_service()
    
    if blocks is None:
        print("ERROR: Could not get forecast from API or service")
        return 1
    
    print(f"Got {len(blocks)} blocks")
    print()
    
    # Display first 10 blocks
    print("First 10 blocks (1hr lookahead window):")
    print("-" * 70)
    print(f"{'Block':<8} {'Time':<26} {'MCP':<12} {'Zone':<10} {'Conf':<6}")
    print("-" * 70)
    for i, b in enumerate(blocks[:10]):
        ts = b.get("forecast_timestamp", "N/A")
        mcp = b.get("predicted_mcp", 0)
        zone = b.get("zone", "N/A")
        conf = b.get("confidence", 0.0)
        print(f"{i+1:<8} {str(ts):<26} {mcp:<12.1f} {zone:<10} {conf:<6.4f}")
    print()
    
    # Make decision
    decision, price, status = make_control_decision(blocks)
    
    print("=" * 70)
    print("CONTROL DECISION")
    print("=" * 70)
    print(f"Status:         {status}")
    print(f"Max price (1h): {price:.1f} Rs/MWh")
    print(f"Decision:       {decision}")
    print()
    
    # Interpretation
    if decision == "ALL_OFF":
        print("⚠️  CRITICAL: Shed all deferrable load immediately!")
    elif decision == "SHED":
        print("⚠️  WARNING: Shed non-critical load (price spike expected)")
    elif decision == "NORMAL":
        print("✓ Normal operation (price acceptable)")
    elif decision == "LOW_PRICE":
        print("✓ Low price period (opportunity for flexible loads)")
    
    print()
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

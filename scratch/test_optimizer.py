import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iex_forecast.application.optimizer import BatteryOptimizer
from iex_forecast.application.risk_engine import RiskManager
from iex_forecast.application.backtester import InstitutionalBacktester

def run_test():
    print("--- Running Institutional Trading System Test ---")
    
    # 1. Mock a 24-hour forecast (96 blocks)
    # Let's create a clear price pattern:
    # 00:00 - 06:00 (Blocks 1-24): Low prices (Rs 3000)
    # 06:00 - 10:00 (Blocks 25-40): Morning peak (Rs 6000)
    # 10:00 - 17:00 (Blocks 41-68): Solar hours, low prices (Rs 2000)
    # 17:00 - 23:00 (Blocks 69-92): Evening peak (Rs 9000)
    # 23:00 - 00:00 (Blocks 93-96): Low prices (Rs 3500)
    
    prices = np.zeros(96)
    prices[0:24] = 3000
    prices[24:40] = 6000
    prices[40:68] = 2000
    prices[68:92] = 9000
    prices[92:96] = 3500
    
    # Add some noise
    prices += np.random.normal(0, 100, 96)
    
    timestamps = pd.date_range("2026-06-08 00:00:00", periods=96, freq="15min")
    forecast_series = pd.Series(prices, index=timestamps)
    
    print("\n[1] Testing Battery Optimizer (10MW / 40MWh BESS)...")
    optimizer = BatteryOptimizer(max_mw=10.0, capacity_mwh=40.0, efficiency=0.85)
    schedule = optimizer.optimize(forecast_series)
    
    total_charged_mwh = schedule["charge_mw"].sum() * 0.25
    total_discharged_mwh = schedule["discharge_mw"].sum() * 0.25
    gross_cashflow = schedule["cashflow_rs"].sum()
    
    print(f"Total Charged: {total_charged_mwh:.2f} MWh")
    print(f"Total Discharged: {total_discharged_mwh:.2f} MWh")
    print(f"Optimal Gross Arbitrage Profit: Rs {gross_cashflow:,.2f}")
    
    # Let's look at the evening peak
    print("\nEvening Peak Behavior (18:00 - 19:00):")
    print(schedule.loc["2026-06-08 18:00:00":"2026-06-08 19:00:00", ["forecast_price", "net_mw", "soc_mwh"]])
    
    print("\n[2] Testing Risk Engine...")
    # Mock quantiles: 10% lower/higher
    p10 = forecast_series * 0.90
    p90 = forecast_series * 1.10
    
    rm = RiskManager()
    risk_metrics = rm.calculate_var(schedule, p10, p90)
    print(f"Expected P&L: Rs {risk_metrics['expected_pnl_rs']:,.2f}")
    print(f"Worst Case Scenario (VaR 90% Base): Rs {risk_metrics['worst_case_pnl_rs']:,.2f}")
    print(f"Value at Risk (Downside): Rs {risk_metrics['var_90_rs']:,.2f}")
    
    print("\n[3] Testing Backtester Module...")
    actual_df = pd.DataFrame({"mcp_rs_mwh": prices + np.random.normal(0, 500, 96)}, index=timestamps)
    forecast_df = pd.DataFrame({"mcp_forecast_rs_mwh": prices}, index=timestamps)
    
    backtester = InstitutionalBacktester(exchange_fee_rs_mwh=20.0)
    bt_metrics = backtester.run_bess_backtest(forecast_df, actual_df, optimizer)
    
    print(f"Simulated Actual Gross Revenue: Rs {bt_metrics['total_gross_rs']:,.2f}")
    print(f"Exchange Fees Paid: Rs {bt_metrics['total_fees_rs']:,.2f}")
    print(f"Final Net Profit: Rs {bt_metrics['total_net_profit_rs']:,.2f}")
    
if __name__ == "__main__":
    run_test()

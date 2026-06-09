"""Historical Backtesting Engine for Institutional P&L Simulation."""

import pandas as pd
import logging
from .optimizer import BatteryOptimizer

logger = logging.getLogger(__name__)

class InstitutionalBacktester:
    """Simulates trading strategy against historical prices."""
    def __init__(self, exchange_fee_rs_mwh: float = 20.0):
        self.exchange_fee = exchange_fee_rs_mwh
        
    def run_bess_backtest(
        self, 
        forecast_df: pd.DataFrame, 
        actual_df: pd.DataFrame,
        optimizer: BatteryOptimizer
    ) -> dict:
        """
        Runs a backtest simulating a battery asset optimizing against forecasts
        and clearing against actual prices.
        
        Args:
            forecast_df: DataFrame with 'mcp_forecast_rs_mwh'
            actual_df: DataFrame with true 'mcp_rs_mwh'
            optimizer: Configured BatteryOptimizer instance
            
        Returns:
            Dictionary with financial metrics.
        """
        # Align indexes
        forecast_series = forecast_df["mcp_forecast_rs_mwh"].copy()
        
        # 1. Generate optimal schedule based on forecasts
        opt_results = optimizer.optimize(forecast_series)
        
        # 2. Replay against actuals
        # Join optimal schedule with actual clearing prices
        sim = opt_results.join(actual_df[["mcp_rs_mwh"]], how="inner")
        
        dt = 0.25 # 15 min block
        gross_revenue = 0.0
        total_fees = 0.0
        
        daily_pnl = []
        
        for idx, row in sim.iterrows():
            charge_mw = row["charge_mw"]
            discharge_mw = row["discharge_mw"]
            actual_price = row["mcp_rs_mwh"]
            
            # Simulated Cashflow using actual clearing price
            # When we charge (buy), we pay the actual price
            # When we discharge (sell), we receive the actual price
            cashflow = (discharge_mw - charge_mw) * actual_price * dt
            
            # Friction costs: We pay exchange fees on both buy and sell volume
            volume_mwh = (charge_mw + discharge_mw) * dt
            fees = volume_mwh * self.exchange_fee
            
            net_pnl = cashflow - fees
            
            gross_revenue += cashflow
            total_fees += fees
            
            daily_pnl.append(net_pnl)
            
        sim["net_pnl"] = daily_pnl
        sim["cumulative_pnl"] = sim["net_pnl"].cumsum()
        
        metrics = {
            "total_gross_rs": gross_revenue,
            "total_fees_rs": total_fees,
            "total_net_profit_rs": gross_revenue - total_fees,
            "total_volume_mwh_traded": sim["charge_mw"].sum() * dt + sim["discharge_mw"].sum() * dt,
            "simulation_df": sim
        }
        
        return metrics

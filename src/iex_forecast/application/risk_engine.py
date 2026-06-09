"""Risk Management Engine for Institutional Algorithmic Trading."""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """Calculates risk metrics and Value at Risk (VaR)."""
    
    def calculate_var(self, optimal_schedule: pd.DataFrame, p10_forecast: pd.Series, p90_forecast: pd.Series) -> dict:
        """
        Calculates the 90% Confidence Value at Risk (VaR) for the day.
        
        If we are LONG (Charging/Buying): 
        Our risk is that the price drops unexpectedly (we overpaid). We look at p10.
        
        If we are SHORT (Discharging/Selling):
        Our risk is that the price spikes unexpectedly (we sold too cheap, opportunity cost). We look at p90.
        
        Args:
            optimal_schedule: DataFrame output from BatteryOptimizer
            p10_forecast: Series of p10 lower bound prices
            p90_forecast: Series of p90 upper bound prices
        """
        dt = 0.25
        worst_case_pnl = []
        expected_pnl = []
        
        for idx, row in optimal_schedule.iterrows():
            net_mw = row["net_mw"] # Positive = Discharging (Short), Negative = Charging (Long)
            expected_price = row["forecast_price"]
            
            # Expected Cashflow
            expected_cashflow = net_mw * expected_price * dt
            expected_pnl.append(expected_cashflow)
            
            if net_mw > 0:
                # We are selling power.
                # Worst case: we could have sold at p90 (opportunity loss), but in strict cashflow terms, 
                # our actual worst case realized revenue is if prices crash to p10.
                # If we committed to sell at day-ahead prices, this changes. 
                # In RTM, if we discharge, we receive the clearing price. 
                # Worst case revenue = p10 price.
                worst_cashflow = net_mw * p10_forecast.loc[idx] * dt
            elif net_mw < 0:
                # We are buying power.
                # Worst case cost = p90 price. (We have to pay a massive spike)
                worst_cashflow = net_mw * p90_forecast.loc[idx] * dt
            else:
                worst_cashflow = 0.0
                
            worst_case_pnl.append(worst_cashflow)
            
        total_expected = sum(expected_pnl)
        total_worst_case = sum(worst_case_pnl)
        
        # VaR is the difference between Expected P&L and Worst Case P&L
        var_90 = total_expected - total_worst_case
        
        return {
            "expected_pnl_rs": total_expected,
            "worst_case_pnl_rs": total_worst_case,
            "var_90_rs": var_90
        }

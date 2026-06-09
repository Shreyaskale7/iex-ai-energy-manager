"""Linear Programming optimizer for Battery Energy Storage Systems (BESS)."""

import pandas as pd
import pulp
import logging

logger = logging.getLogger(__name__)

class BatteryOptimizer:
    """Optimizes charge/discharge schedule for a battery to maximize arbitrage profit."""
    def __init__(
        self,
        max_mw: float = 10.0,
        capacity_mwh: float = 40.0,
        efficiency: float = 0.85,
        initial_soc_mwh: float = 0.0,
        min_soc_mwh: float = 0.0
    ):
        self.max_mw = max_mw
        self.capacity_mwh = capacity_mwh
        self.efficiency = efficiency
        self.initial_soc = initial_soc_mwh
        self.min_soc = min_soc_mwh

    def optimize(self, forecast_prices: pd.Series) -> pd.DataFrame:
        """
        Solves for the optimal charge/discharge schedule over the forecast horizon.
        
        Args:
            forecast_prices: Series of prices (Rs/MWh) indexed by timestamp.
            
        Returns:
            DataFrame with charge_mw, discharge_mw, soc_mwh, and net_cashflow.
        """
        n_periods = len(forecast_prices)
        prices = forecast_prices.values
        timestamps = forecast_prices.index
        dt = 0.25  # 15-minute blocks = 0.25 hours

        # Create LP problem
        prob = pulp.LpProblem("Battery_Arbitrage_Optimization", pulp.LpMaximize)

        # Decision Variables
        charge = pulp.LpVariable.dicts("charge_mw", range(n_periods), lowBound=0, upBound=self.max_mw)
        discharge = pulp.LpVariable.dicts("discharge_mw", range(n_periods), lowBound=0, upBound=self.max_mw)
        soc = pulp.LpVariable.dicts("soc_mwh", range(n_periods), lowBound=self.min_soc, upBound=self.capacity_mwh)

        # Objective Function: Maximize Profit
        # Profit = Revenue from discharging - Cost of charging
        profit = pulp.lpSum([discharge[i] * prices[i] * dt - charge[i] * prices[i] * dt for i in range(n_periods)])
        prob += profit

        # Constraints
        for i in range(n_periods):
            # Cannot charge and discharge at the same time (implied by prices, but explicit is safer for some solvers, 
            # though standard LP handles it by basic economics - it won't do both if prices are positive. 
            # To be strictly linear, we allow both but economics forces one to 0).
            
            # State of Charge Update
            if i == 0:
                prob += soc[i] == self.initial_soc + (charge[i] * self.efficiency * dt) - (discharge[i] / self.efficiency * dt)
            else:
                prob += soc[i] == soc[i-1] + (charge[i] * self.efficiency * dt) - (discharge[i] / self.efficiency * dt)

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != 'Optimal':
            logger.warning("Solver could not find optimal solution.")
            
        results = []
        for i in range(n_periods):
            c_val = charge[i].varValue or 0.0
            d_val = discharge[i].varValue or 0.0
            soc_val = soc[i].varValue or 0.0
            net_mw = d_val - c_val
            cashflow = net_mw * prices[i] * dt
            
            results.append({
                "timestamp": timestamps[i],
                "forecast_price": prices[i],
                "charge_mw": c_val,
                "discharge_mw": d_val,
                "net_mw": net_mw,
                "soc_mwh": soc_val,
                "cashflow_rs": cashflow
            })

        df_results = pd.DataFrame(results).set_index("timestamp")
        return df_results

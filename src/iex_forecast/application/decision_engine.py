"""Smart Load Decision Engine Service."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from iex_forecast.domain.devices import DeviceCategory, DeviceProfile, DeviceState

class CustomThresholds(BaseModel):
    # Optional manual overrides for the boundaries, e.g., to treat MCP > 4000 as RED
    force_yellow_mcp_min: Optional[float] = None
    force_red_mcp_min: Optional[float] = None
    # If spike probability is above this, act as if it's RED regardless of MCP
    spike_prob_red_threshold: float = 0.5


class DecisionEngineService:
    """Evaluates forecast blocks to generate an optimal load schedule."""

    def evaluate_state(
        self,
        predicted_mcp: float,
        forecast_zone: str,
        spike_prob: float,
        device: DeviceProfile,
        thresholds: CustomThresholds
    ) -> DeviceState:
        """Evaluate whether a specific device should be ON or OFF in a given block."""
        
        # 1. Manual Override takes precedence
        if device.manual_override is not None:
            return device.manual_override

        # 2. Determine Effective Zone based on custom thresholds & spike probability
        effective_zone = forecast_zone.upper()
        
        if thresholds.force_yellow_mcp_min is not None and predicted_mcp >= thresholds.force_yellow_mcp_min:
            effective_zone = "YELLOW"
        if thresholds.force_red_mcp_min is not None and predicted_mcp >= thresholds.force_red_mcp_min:
            effective_zone = "RED"
            
        if spike_prob >= thresholds.spike_prob_red_threshold:
            effective_zone = "RED"

        # 3. Base Rules
        # GREEN: All ON
        # YELLOW: Critical ON, Flexible ON, Deferrable OFF
        # RED: Critical ON, Flexible OFF, Deferrable OFF
        
        if device.category == DeviceCategory.CRITICAL:
            return DeviceState.ON
            
        if effective_zone == "GREEN":
            return DeviceState.ON
            
        if effective_zone == "YELLOW":
            if device.category == DeviceCategory.FLEXIBLE:
                # Basic priority handling: if priority is very low (>5), maybe turn off even in YELLOW
                # But strict rules say Flexible ON in YELLOW. 
                # We'll respect strict rules by default unless we implement priority shedding.
                return DeviceState.ON
            elif device.category == DeviceCategory.DEFERRABLE:
                return DeviceState.OFF
                
        if effective_zone == "RED":
            if device.category in (DeviceCategory.FLEXIBLE, DeviceCategory.DEFERRABLE):
                return DeviceState.OFF

        # Default fallback
        return DeviceState.ON

    def calculate_schedule(
        self,
        forecast_points: List[dict],
        devices: List[DeviceProfile],
        thresholds: Optional[CustomThresholds] = None
    ) -> Dict[str, Any]:
        """
        Generate schedule and calculate savings against an 'always on' baseline.
        
        forecast_points should contain keys:
        - forecast_timestamp
        - predicted_mcp
        - zone
        - spike_probability
        """
        if thresholds is None:
            thresholds = CustomThresholds()

        schedule = []
        
        baseline_cost_rs = 0.0
        optimized_cost_rs = 0.0

        for point in forecast_points:
            mcp = float(point.get("predicted_mcp", 0.0))
            zone = str(point.get("zone", "GREEN"))
            spike_prob = float(point.get("spike_probability", 0.0))
            ts = point.get("forecast_timestamp")

            block_recommendations = []
            
            block_baseline_kw = 0.0
            block_optimized_kw = 0.0

            for device in devices:
                state = self.evaluate_state(mcp, zone, spike_prob, device, thresholds)
                block_recommendations.append({
                    "device_id": device.device_id,
                    "name": device.name,
                    "category": device.category.value,
                    "recommended_state": state.value
                })
                
                # Baseline assumes ALL devices are always ON (except if manual_override == OFF)
                # But conventionally baseline is just running normally (always ON)
                baseline_on = True
                if device.manual_override == DeviceState.OFF:
                    baseline_on = False
                    
                if baseline_on:
                    block_baseline_kw += device.power_kw
                    
                if state == DeviceState.ON:
                    block_optimized_kw += device.power_kw

            # Cost per block = (kW / 1000) * MCP_per_MWh / 4 (since 15-min blocks)
            # Actually, Power_kW is already kW. 
            # Energy in MWh for 15 mins = (Power_kW / 1000) * 0.25
            mwh_multiplier = 0.25 / 1000.0
            
            baseline_cost_rs += (block_baseline_kw * mwh_multiplier * mcp)
            optimized_cost_rs += (block_optimized_kw * mwh_multiplier * mcp)

            schedule.append({
                "forecast_timestamp": ts,
                "predicted_mcp": mcp,
                "effective_zone": "RED" if spike_prob >= thresholds.spike_prob_red_threshold else zone,
                "total_load_kw": block_optimized_kw,
                "device_states": block_recommendations
            })

        expected_savings = baseline_cost_rs - optimized_cost_rs

        return {
            "recommended_schedule": schedule,
            "baseline_cost_rs": round(baseline_cost_rs, 2),
            "optimized_cost_rs": round(optimized_cost_rs, 2),
            "expected_savings_rs": round(expected_savings, 2),
            "savings_percentage": round((expected_savings / max(baseline_cost_rs, 1.0)) * 100, 2)
        }

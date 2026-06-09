"""Domain entities for the Smart Load Decision Engine."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class DeviceCategory(str, Enum):
    CRITICAL = "Critical"
    FLEXIBLE = "Flexible"
    DEFERRABLE = "Deferrable"

class DeviceState(str, Enum):
    ON = "ON"
    OFF = "OFF"

class DeviceProfile(BaseModel):
    device_id: str
    name: str
    category: DeviceCategory
    power_kw: float = Field(..., gt=0.0)
    priority_level: int = Field(1, ge=1, le=10, description="1 is highest priority")
    manual_override: Optional[DeviceState] = None

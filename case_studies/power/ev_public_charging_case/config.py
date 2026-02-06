from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class ArrivalConfig:
    hourly_rate: List[float]
    scale: float = 1.0
    rate_is_system_total: bool = True

@dataclass(frozen=True)
class EVConfig:
    battery_capacity_list: List[float] = (55.0, 75.0, 80.0, 100.0)
    soc_ini_range: Tuple[float, float] = (0.1, 0.5)
    demand_soc_min_range: Tuple[float, float] = (0.2, 0.4)
    time_cost_per_hour: float = 5.0
    alpha_scale_range: Tuple[float, float] = (0.8, 1.2)
    leave_u_diff_max: float = 10.0
    leave_prob_max: float = 0.3

    def cost_est(self, price: float = 0.25, parking_fee: float = 3.0, charging_power: float = 100.0) -> float:
        time_cost_per_kwh = self.time_cost_per_hour / charging_power
        parking_fee_per_kwh = parking_fee / charging_power
        return price + parking_fee_per_kwh + time_cost_per_kwh

    def alpha_range(self) -> Tuple[float, float]:
        base = self.cost_est()
        lo, hi = self.alpha_scale_range
        return base * lo, base * hi

@dataclass(frozen=True)
class StationConfig:
    num_chargers: int = 2
    charging_power: float = 100.0
    init_price: float = 0.25
    parking_fee: float = 3.0
    charging_efficiency: float = 0.9
    cost_kwh: float = 0.01
    cost_hour: float = 1000 / 30 / 24

@dataclass(frozen=True)
class WorldConfig:
    dt: int = 60
    action_period: int = 15 * 60
    horizon_seconds: int = 48 * 3600
    seed: int = 0

@dataclass(frozen=True)
class PriceConfig:
    price_min: float = 0.05
    price_max: float = 0.80

# ev_charging/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# ============================================================
# Arrival (demand side)
# ============================================================

@dataclass(frozen=True)
class ArrivalConfig:
    """
    Arrival rate configuration.

    hourly_rate:
        length-24 list, arrival rate per hour
    scale:
        global multiplier
    rate_is_system_total:
        True  -> hourly_rate is total system arrival rate
        False -> hourly_rate is per-station arrival rate
    """
    hourly_rate: List[float]
    scale: float = 1.0
    rate_is_system_total: bool = True


# ============================================================
# EV user behavior
# ============================================================

@dataclass(frozen=True)
class EVConfig:
    """
    EV heterogeneity and utility model parameters.
    """

    # battery & SoC
    battery_capacity_list: List[float] = (55.0, 75.0, 80.0, 100.0)
    soc_ini_range: Tuple[float, float] = (0.1, 0.5)
    demand_soc_min_range: Tuple[float, float] = (0.2, 0.4)

    # time valuation ($/hour)
    time_cost_per_hour: float = 5.0

    # α scaling relative to estimated cost
    alpha_scale_range: Tuple[float, float] = (0.8, 1.2)

    # leaving behavior
    leave_u_diff_max: float = 10.0
    leave_prob_max: float = 0.3

    # ---- helpers used by EV ---------------------------------

    def cost_est(
        self,
        price: float = 0.25,
        parking_fee: float = 3.0,
        charging_power: float = 100.0,
    ) -> float:
        """
        Estimated cost per kWh used to scale alpha.
        Mirrors your old COST_EST logic but generalized.
        """
        time_cost_per_kwh = self.time_cost_per_hour / charging_power
        parking_fee_per_kwh = parking_fee / charging_power
        return price + parking_fee_per_kwh + time_cost_per_kwh

    def alpha_range(self) -> Tuple[float, float]:
        """
        Alpha range derived from estimated cost.
        Used directly by EV.__post_init__().
        """
        base = self.cost_est()
        lo, hi = self.alpha_scale_range
        return base * lo, base * hi


# ============================================================
# Charging station (entity)
# ============================================================

@dataclass(frozen=True)
class StationConfig:
    """
    Physical and economic parameters of a charging station.
    """

    # capacity
    num_chargers: int = 2
    charging_power: float = 100.0   # kW

    # pricing
    init_price: float = 0.25        # $/kWh
    parking_fee: float = 3.0        # $/hour

    # cost model
    charging_efficiency: float = 0.9
    cost_kwh: float = 0.01          # marginal cost ($/kWh)
    cost_hour: float = 1000 / 30 / 24  # fixed hourly cost


# ============================================================
# (Optional) world-level timing
# ============================================================

@dataclass(frozen=True)
class WorldConfig:
    """
    Time discretization and coordination.
    """
    dt: int = 60                  # seconds
    action_period: int = 1800     # seconds

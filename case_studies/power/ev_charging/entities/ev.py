# ev_charging/entities/ev.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..config import EVConfig


@dataclass
class EV:
    """EV user entity (same semantics as your old EV.py, but config-driven)."""

    time_arrival: float
    cfg: EVConfig
    rng: np.random.Generator
    u_tol: float = 0.0

    # sampled / internal
    battery_capacity: float = 0.0
    SoC_ini: float = 0.0
    demand_min: float = 0.0
    alpha: float = 0.0
    leave_prob_max: float = 0.0

    # dynamic state
    SoC: float = 0.0
    SoE: float = 0.0
    demand_ini: Optional[float] = None
    demand: Optional[float] = None
    selected_station: Optional[str] = None
    utility_ini: Optional[float] = None
    price: Optional[float] = None
    parking_fee: Optional[float] = None

    def __post_init__(self) -> None:
        # sample user heterogeneity (same meaning as old code)
        self.battery_capacity = float(self.rng.choice(self.cfg.battery_capacity_list))

        lo, hi = self.cfg.soc_ini_range
        self.SoC_ini = float(self.rng.uniform(lo, hi))

        lo, hi = self.cfg.demand_soc_min_range
        self.demand_min = float(self.rng.uniform(lo, hi) * self.battery_capacity)

        # alpha range derived from config helper (keeps your COST_EST logic configurable)
        a_lo, a_hi = self.cfg.alpha_range()
        self.alpha = float(self.rng.uniform(a_lo, a_hi))

        self.leave_prob_max = float(self.cfg.leave_prob_max)

        self.SoC = self.SoC_ini
        self.SoE = self.SoC * self.battery_capacity

    # ----------------- charging & economics -----------------

    def charge_and_pay(self, charging_power: float, dt: float) -> Tuple[float, float]:
        """Same as old: integrate energy, payment based on (price + parking_fee/charging_power)."""
        SoE_pre = self.SoE
        self.SoE = min(self.SoE + float(charging_power) * float(dt) / 3600.0, self.battery_capacity)
        self.SoC = self.SoE / self.battery_capacity

        energy_delivered = self.SoE - SoE_pre
        payment = energy_delivered * (float(self.price) + float(self.parking_fee) / float(charging_power))
        return energy_delivered, payment

    def get_utility(self, x: float, price: float, parking_fee: float, charging_power: float) -> float:
        """Same as old utility."""
        time_cost = float(self.cfg.time_cost_per_hour)
        utility = (
            self.alpha * self.battery_capacity * math.log(1.0 + float(x) / max(self.SoE, 1e-9))
            - float(price) * float(x)
            - (float(parking_fee) + time_cost) * (float(x) / float(charging_power))
        )
        return float(utility)

    def get_optimal_demand(
        self, price: float, parking_fee: float, charging_power: float
    ) -> Tuple[float, float]:
        """Same as old derivative-based optimum + clamp to [demand_min, battery_capacity-SoE]."""
        time_cost = float(self.cfg.time_cost_per_hour)
        denom = float(price) + (float(parking_fee) + time_cost) / float(charging_power)

        x_pole = self.alpha * self.battery_capacity / max(denom, 1e-9) - self.SoE
        x_up = self.battery_capacity - self.SoE

        if x_pole <= self.demand_min:
            x_opt = self.demand_min
        elif x_pole >= x_up:
            x_opt = x_up
        else:
            x_opt = x_pole

        utility = self.get_utility(x_opt, price, parking_fee, charging_power)
        return float(x_opt), float(utility)

    # ----------------- station choice -----------------

    def choose_station(self, evsts_info: Dict[str, Dict[str, Any]]):
        """
        Same decision rule as your old EV.choose_station:
        1) compute (x,u) for each station
        2) find best overall and best available
        3) leave with probability based on utility gap
        4) among available stations, choose uniformly from those within tolerance
        """
        best_st = None
        best_u = -float("inf")

        best_avail_st = None
        best_avail_u = -float("inf")

        evsts_x_u = {key: {} for key in evsts_info}
        available_list = []

        for name, info in evsts_info.items():
            x, u = self.get_optimal_demand(info["price"], info["parking_fee"], info["charging_power"])
            evsts_x_u[name]["x"] = x
            evsts_x_u[name]["u"] = u

            if u > best_u:
                best_u = u
                best_st = name

            if (info["num_chargers"] - info["num_users"] > 0) and (u > best_avail_u):
                available_list.append(name)
                best_avail_u = u
                best_avail_st = name

        # no available station
        if not available_list:
            return None, None, None

        p_leave, _dist = self.get_leave_probability(best_u, best_avail_u)

        if float(self.rng.random()) < p_leave:
            return None, None, None

        # candidate within tolerance (utility difference <= u_tol)
        candidate_list = []
        for name in available_list:
            d = best_avail_u - evsts_x_u[name]["u"]
            if d <= float(self.u_tol):
                candidate_list.append(name)

        if not candidate_list:
            # fallback to best available if tolerance list becomes empty
            candidate_list = [best_avail_st] if best_avail_st is not None else available_list

        self.selected_station = str(self.rng.choice(candidate_list))
        self.demand = self.demand_ini = float(evsts_x_u[self.selected_station]["x"])
        self.utility_ini = float(evsts_x_u[self.selected_station]["u"])
        self.price = float(evsts_info[self.selected_station]["price"])
        self.parking_fee = float(evsts_info[self.selected_station]["parking_fee"])

        return self.selected_station, self.demand_ini, self.utility_ini

    def get_leave_probability(self, best_u: float, u: float) -> Tuple[float, float]:
        """Same as old: p = leave_prob_max * min((best_u-u)/LEAVE_U_DIFF_MAX, 1)."""
        distance = float(best_u - u)
        diff_r = min(distance / float(self.cfg.leave_u_diff_max), 1.0)
        p = float(self.leave_prob_max) * diff_r
        return float(p), float(distance)

    def get_demand(self) -> float:
        """Same as old: remaining demand = demand_ini + SoC_ini*cap - SoE."""
        if self.demand_ini is None:
            return 0.0
        return float(self.demand_ini + self.SoC_ini * self.battery_capacity - self.SoE)

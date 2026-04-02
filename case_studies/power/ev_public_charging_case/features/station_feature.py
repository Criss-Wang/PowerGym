"""Charging station feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence

from heron.core.feature import Feature
from case_studies.power.ev_public_charging_case.utils import safe_div, norm01

PRICE_LO = 0.0
PRICE_HI = 0.8
REVENUE_LO = 0.0
REVENUE_HI = 1000.0  # Max expected hourly revenue per station
POWER_HI = 1500.0  # Max expected station power (kW)
STEP_PROFIT_SCALE = 50.0  # $ scale for tanh normalization of step profit


@dataclass(slots=True)
class ChargingStationFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ['owner', 'upper_level', 'system']
    open_chargers: int = 5
    max_chargers: int = 5
    charging_price: float = 0.25
    station_step_revenue: float = 0.0
    station_step_energy_cost: float = 0.0
    station_step_overhead_cost: float = 0.0
    station_step_profit: float = 0.0
    station_step_profit_obs: float = 0.0
    station_cumulative_revenue: float = 0.0
    station_cumulative_profit: float = 0.0
    station_power: float = 0.0  # Current power output (kW)
    station_capacity: float = 0.0  # Total capacity (kW)
    

    @staticmethod
    def profit_to_obs(step_profit: float) -> float:
        return float(np.tanh(float(step_profit) / STEP_PROFIT_SCALE))

    @staticmethod
    def obs_to_profit(step_profit_obs: float) -> float:
        clipped = float(np.clip(step_profit_obs, -0.999999, 0.999999))
        return float(np.arctanh(clipped) * STEP_PROFIT_SCALE)

    def vector(self) -> np.ndarray:
        self.station_step_profit_obs = self.profit_to_obs(self.station_step_profit)
        return np.array(
            [
                safe_div(self.open_chargers, self.max_chargers),
                norm01(self.charging_price, PRICE_LO, PRICE_HI),
                self.station_step_profit_obs,
                safe_div(self.station_power, self.station_capacity) if self.station_capacity > 0 else 0.0,  # utilization
            ],
            dtype=np.float32,
        )

    def names(self):
        return ['open_norm', 'price_norm', 'step_profit_obs', 'utilization']

    def to_dict(self):
        return {
            'open_chargers': self.open_chargers,
            'max_chargers': self.max_chargers,
            'charging_price': self.charging_price,
            'station_step_revenue': self.station_step_revenue,
            'station_step_energy_cost': self.station_step_energy_cost,
            'station_step_overhead_cost': self.station_step_overhead_cost,
            'station_step_profit': self.station_step_profit,
            'station_step_profit_obs': self.station_step_profit_obs,
            'station_cumulative_revenue': self.station_cumulative_revenue,
            'station_cumulative_profit': self.station_cumulative_profit,
            'station_power': self.station_power,
            'station_capacity': self.station_capacity,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        allowed = {
            'open_chargers', 'max_chargers', 'charging_price',
            'station_step_revenue', 'station_step_energy_cost', 'station_step_overhead_cost',
            'station_step_profit', 'station_step_profit_obs',
            'station_cumulative_revenue', 'station_cumulative_profit',
            'station_power', 'station_capacity',
        }
        for k, v in kw.items():
            if k not in allowed:
                continue
            if k == 'charging_price':
                self.charging_price = float(v)
            elif k == 'open_chargers':
                self.open_chargers = int(v)
            elif k == 'max_chargers':
                self.max_chargers = int(v)
            elif k == 'station_step_revenue':
                self.station_step_revenue = float(v)
            elif k == 'station_step_energy_cost':
                self.station_step_energy_cost = float(v)
            elif k == 'station_step_overhead_cost':
                self.station_step_overhead_cost = float(v)
            elif k == 'station_step_profit':
                self.station_step_profit = float(v)
                self.station_step_profit_obs = self.profit_to_obs(self.station_step_profit)
            elif k == 'station_step_profit_obs':
                self.station_step_profit_obs = float(v)
            elif k == 'station_cumulative_revenue':
                self.station_cumulative_revenue = float(v)
            elif k == 'station_cumulative_profit':
                self.station_cumulative_profit = float(v)
            elif k == 'station_power':
                self.station_power = float(v)
            elif k == 'station_capacity':
                self.station_capacity = float(v)


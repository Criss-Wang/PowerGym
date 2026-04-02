"""Charger feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence

from heron.core.feature import Feature
from case_studies.power.ev_public_charging_case.utils import safe_div


@dataclass(slots=True)
class ChargerFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ['owner']
    p_kw: float = 0.0
    p_max_kw: float = 150.0
    occupied_or_not: int = 0
    charging_price: float = 0.25
    step_energy_delivered_kwh: float = 0.0
    step_revenue: float = 0.0
    step_energy_cost: float = 0.0
    step_profit: float = 0.0
    cumulative_revenue: float = 0.0
    cumulative_profit: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array(
            [safe_div(self.p_kw, self.p_max_kw),
             float(self.occupied_or_not),
             self.charging_price,
             self.step_profit],
            dtype=np.float32,
        )

    def names(self):
        return ['p_norm', 'occupied_or_not', 'price', 'step_profit']

    def to_dict(self):
        return {
            'p_kw': self.p_kw,
            'p_max_kw': self.p_max_kw,
            'occupied_or_not': self.occupied_or_not,
            'charging_price': self.charging_price,
            'step_energy_delivered_kwh': self.step_energy_delivered_kwh,
            'step_revenue': self.step_revenue,
            'step_energy_cost': self.step_energy_cost,
            'step_profit': self.step_profit,
            'cumulative_revenue': self.cumulative_revenue,
            'cumulative_profit': self.cumulative_profit,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        allowed = {
            'p_kw', 'p_max_kw', 'occupied_or_not', 'charging_price',
            'step_energy_delivered_kwh', 'step_revenue', 'step_energy_cost',
            'step_profit', 'cumulative_revenue', 'cumulative_profit',
        }
        for k, v in kw.items():
            if k in allowed:
                setattr(self, k, v)
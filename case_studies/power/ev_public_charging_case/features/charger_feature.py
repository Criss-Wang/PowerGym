"""Charger feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import Any, List

from heron.core.feature import FeatureProvider
from case_studies.power.ev_public_charging_case.utils import safe_div


@dataclass(slots=True)
class ChargerFeature(FeatureProvider):
    """Power state feature for chargers."""
    visibility = ["owner", "upper_level"]

    p_kw: float = 0.0
    p_max_kw: float = 150.0
    occupancy_flag: int = 0
    open_or_not: int = 1
    current_price: float = 0.5

    def vector(self) -> np.ndarray:
        return np.array([
            safe_div(self.p_kw, self.p_max_kw),
            float(self.occupancy_flag),
            float(self.open_or_not),
            self.current_price
        ], dtype=np.float32)

    def names(self) -> List[str]:
        return ['p_norm', 'occupied', 'open', 'price']

    def to_dict(self):
        return {
            'p_kw': self.p_kw,
            'p_max_kw': self.p_max_kw,
            'occupancy_flag': self.occupancy_flag,
            'open_or_not': self.open_or_not,
            'current_price': self.current_price
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw: Any) -> None:
        if 'p_kw' in kw: self.p_kw = float(kw['p_kw'])
        if 'occupancy_flag' in kw: self.occupancy_flag = int(np.clip(kw['occupancy_flag'], 0, 1))
        if 'current_price' in kw: self.current_price = float(kw['current_price'])
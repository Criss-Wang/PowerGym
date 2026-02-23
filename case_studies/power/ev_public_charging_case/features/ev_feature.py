"""Electric vehicle feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import Any, List

from heron.core.feature import FeatureProvider

@dataclass(slots=True)
class ElectricVehicleFeature(FeatureProvider):
    """State feature for electric vehicles."""
    visibility = ["owner"]

    soc: float = 0.2
    soc_target: float = 0.8
    arrival_time: float = 0.0
    max_wait_time: float = 3600.0
    price_sensitivity: float = 0.5
    is_present: int = 1
    accumulated_cost: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([
            float(self.soc),
            float(self.soc_target),
            float(self.price_sensitivity),
            float(self.is_present)
        ], dtype=np.float32)

    def names(self) -> List[str]:
        return ['soc', 'soc_target', 'price_sensitivity', 'is_present']

    def to_dict(self):
        return {
            'soc': self.soc,
            'soc_target': self.soc_target,
            'arrival_time': self.arrival_time,
            'max_wait_time': self.max_wait_time,
            'price_sensitivity': self.price_sensitivity,
            'is_present': self.is_present,
            'accumulated_cost': self.accumulated_cost
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw: Any) -> None:
        if 'soc' in kw: self.soc = float(kw['soc'])
        if 'is_present' in kw: self.is_present = int(kw['is_present'])
        if 'accumulated_cost' in kw: self.accumulated_cost = float(kw['accumulated_cost'])
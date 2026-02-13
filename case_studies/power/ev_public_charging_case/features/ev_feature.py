"""Electric vehicle feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from heron.core.feature import FeatureProvider


@dataclass
class ElectricVehicleFeature(FeatureProvider):
    visibility = ['owner', 'upper_level']
    soc: float = 0.2
    soc_target: float = 0.8
    arrival_time: float = 0.0  # Time when EV arrived (seconds)
    max_wait_time: float = 3600.0  # Maximum willing to wait (seconds)
    price_sensitivity: float = 0.5  # 0-1, higher = more sensitive to price
    preferred_station_id: Optional[str] = None  # Preferred station (can be None)

    def vector(self) -> np.ndarray:
        return np.array([
            float(self.soc),
            float(self.soc_target),
            float(self.price_sensitivity)
        ], dtype=np.float32)

    def names(self): return ['soc', 'soc_target', 'price_sensitivity']

    def to_dict(self):
        return {
            'soc': self.soc,
            'soc_target': self.soc_target,
            'arrival_time': self.arrival_time,
            'max_wait_time': self.max_wait_time,
            'price_sensitivity': self.price_sensitivity,
            'preferred_station_id': self.preferred_station_id
        }

    @classmethod
    def from_dict(cls, d): return cls(**d)

    def set_values(self, **kw):
        if 'soc' in kw: self.soc = float(np.clip(kw['soc'], 0.0, 1.0))
        if 'arrival_time' in kw: self.arrival_time = float(kw['arrival_time'])
        if 'price_sensitivity' in kw: self.price_sensitivity = float(np.clip(kw['price_sensitivity'], 0.0, 1.0))


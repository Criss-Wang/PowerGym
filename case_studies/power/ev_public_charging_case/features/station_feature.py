"""Charging station feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence

from heron.core.feature import Feature
from case_studies.power.ev_public_charging_case.utils import safe_div, norm01

PRICE_LO = 0.0
PRICE_HI = 0.8
POWER_HI = 1500.0  # Max expected station power (kW)


@dataclass(slots=True)
class ChargingStationFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ['owner', 'upper_level', 'system']
    open_chargers: int = 5
    max_chargers: int = 5
    charging_price: float = 0.25
    station_power: float = 0.0  # Current power output (kW)
    station_capacity: float = 0.0  # Total capacity (kW)

    def vector(self) -> np.ndarray:
        return np.array(
            [
                safe_div(self.open_chargers, self.max_chargers),
                norm01(self.charging_price, PRICE_LO, PRICE_HI),
                safe_div(self.station_power, self.station_capacity) if self.station_capacity > 0 else 0.0,  # utilization
            ],
            dtype=np.float32,
        )

    def names(self):
        return ['open_norm', 'price_norm', 'utilization']

    def to_dict(self):
        return {
            'open_chargers': self.open_chargers,
            'max_chargers': self.max_chargers,
            'charging_price': self.charging_price,
            'station_power': self.station_power,
            'station_capacity': self.station_capacity,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        allowed = {
            'open_chargers', 'max_chargers', 'charging_price',
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
            elif k == 'station_power':
                self.station_power = float(v)
            elif k == 'station_capacity':
                self.station_capacity = float(v)


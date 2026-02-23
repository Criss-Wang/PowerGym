"""Charging station feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import Any, Tuple

from heron.core.feature import FeatureProvider
from case_studies.power.ev_public_charging_case.utils import safe_div, norm01


@dataclass(slots=True)
class ChargingStationFeature(FeatureProvider):
    """State feature for charging stations."""
    visibility = ['public']

    open_chargers: int = 5
    max_chargers: int = 5
    charging_price: float = 0.25
    price_range: Tuple[float, float] = (0.0, 0.8)

    def vector(self) -> np.ndarray:
        return np.array([safe_div(self.open_chargers, self.max_chargers),
                         norm01(self.charging_price, self.price_range[0], self.price_range[1])], dtype=np.float32)

    def names(self):
        return ['open_norm', 'price_norm']

    def to_dict(self):
        return {'open_chargers': self.open_chargers, 'charging_price': self.charging_price}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw: Any) -> None:
        if 'charging_price' in kw: self.charging_price = float(kw['charging_price'])
        if 'open_chargers' in kw: self.open_chargers = int(kw['open_chargers'])



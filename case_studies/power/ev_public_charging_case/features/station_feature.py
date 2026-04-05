"""Charging station feature provider."""
import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence, List
from heron.core.feature import Feature
from case_studies.power.ev_public_charging_case.utils import norm01

@dataclass(slots=True)
class ChargingStationFeature(Feature):
    """Station-level operating features for the Coordinator."""
    visibility: ClassVar[Sequence[str]] = ['owner', 'upper_level', 'system']

    # Normalization boundaries
    PROFIT_DELTA_HI: ClassVar[float] = 100.0
    NUM_CHARGERS_HI: ClassVar[float] = 50.0
    PRICE_LO: ClassVar[float] = 0.0
    PRICE_HI: ClassVar[float] = 1.0

    occupancy_ratio: float = 0.0  # rho
    current_price: float = 0.5    # p_curr
    delta_profit: float = 0.0     # Delta Pi (step profit)
    num_chargers: float = 0.0     # N_max

    def vector(self) -> np.ndarray:
        """Returns 4D vector: [rho, p_curr, Delta_Pi, N_max]."""
        return np.array([
            norm01(self.occupancy_ratio, 0.0, 1.0),
            norm01(self.current_price, self.PRICE_LO, self.PRICE_HI),
            norm01(self.delta_profit, -self.PROFIT_DELTA_HI, self.PROFIT_DELTA_HI),
            norm01(self.num_chargers, 0.0, self.NUM_CHARGERS_HI),
        ], dtype=np.float32)

    def names(self) -> List[str]:
        return ['occupancy_ratio', 'current_price', 'delta_profit', 'num_chargers']

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        for k, v in kw.items():
            if k in self.__slots__:
                setattr(self, k, float(v))
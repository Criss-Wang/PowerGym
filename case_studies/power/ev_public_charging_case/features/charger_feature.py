"""Charger feature provider."""
import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence, List
from heron.core.feature import Feature
from case_studies.power.ev_public_charging_case.utils import norm01

@dataclass(slots=True)
class ChargerFeature(Feature):
    """Micro-level state of a single physical charging unit."""
    visibility: ClassVar[Sequence[str]] = ['owner', 'upper_level', 'system']
    charging_efficiency: float = 0.95                        # eta: Efficiency
    p_max_kw: float = 150.0                  # Physical constraint

    p_kw: float = 0.0                        # P_i: Current power
    occupied_or_not: float = 0.0                 # Binary busy indicator
    step_energy_delivered_kwh: float = 0.0   # E_step: Delivered in current dt
    session_price: float = 0.5                      # p_sess: Locked session price
    elapsed_charging_time: float = 0.0                 # elapsed_charging_time: Elapsed charging time

    def vector(self) -> np.ndarray:
        """Returns 6D vector for physical tracking."""
        return np.array([
            norm01(self.p_kw, 0.0, self.p_max_kw),
            norm01(self.charging_efficiency, 0.0, 1.0),
            float(self.occupied_or_not),
            norm01(self.step_energy_delivered_kwh, 0.0, self.p_max_kw * 0.25),
            norm01(self.session_price, 0.0, 1.0),
            norm01(self.elapsed_charging_time, 0.0, 86400.0), # Normalized to 24h
        ], dtype=np.float32)

    def names(self) -> List[str]:
        return ['p_kw', 'charging_efficiency', 'occupied_or_not', 'step_energy', 'session_price', 'elapsed_charging_time']

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        for k, v in kw.items():
            if k in self.__slots__:
                setattr(self, k, float(v))
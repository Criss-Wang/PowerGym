"""Market feature provider."""

import numpy as np
from dataclasses import dataclass

from heron.core.feature import FeatureProvider


@dataclass
class MarketFeature(FeatureProvider):
    visibility = ['owner', 'upper_level', 'system']
    lmp: float = 0.20
    t_day_s: float = 0.0

    def vector(self) -> np.ndarray:
        theta = 2.0 * np.pi * (self.t_day_s % 86400.0) / 86400.0
        return np.array([self.lmp, np.sin(theta), np.cos(theta)], dtype=np.float32)

    def names(self):
        return ['lmp', 't_sin', 't_cos']

    def to_dict(self):
        return {'lmp': self.lmp, 't_day_s': self.t_day_s}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        if 'lmp' in kw: self.lmp = float(kw['lmp'])
        if 't_day_s' in kw: self.t_day_s = float(kw['t_day_s'])
"""Market feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence, List

from heron.core.feature import Feature

@dataclass(slots=True)
class MarketFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ['owner', 'upper_level', 'system', 'public']
    t_day_s: float = 0.0
    lmp: float = 0.25  # Kept in dict for bookkeeping, but removed from vector

    def vector(self) -> np.ndarray:
        """Returns 2D cyclic time encoding: [sin(theta), cos(theta)]."""
        # theta represents the time of day mapped to [0, 2pi]
        theta = 2.0 * np.pi * (self.t_day_s % 86400.0) / 86400.0
        return np.array([np.sin(theta), np.cos(theta)], dtype=np.float32)

    def names(self) -> List[str]:
        return ['t_sin', 't_cos']

    def to_dict(self):
        return {'t_day_s': self.t_day_s, 'lmp': self.lmp}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        for k, v in kw.items():
            if k in self.__slots__:
                setattr(self, k, float(v))
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
from heron.core.feature import FeatureProvider

@dataclass(slots=True)
class PublicMarketSignal(FeatureProvider):
    """
    Public Market signals
    """
    visibility: List[str] = ("public",)

    elec_price: float = 0.0
    time_sin: float = 0.0
    time_cos: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([self.elec_price, self.time_sin, self.time_cos], dtype=np.float32)

    def names(self) -> List[str]:
        return ["elec_price", "time_sin", "time_cos"]

    def set_values(self, **kwargs: Any):
        if "elec_price" in kwargs:
            self.elec_price = float(kwargs["elec_price"])
        if "hour" in kwargs:
            rad = 2 * np.pi * float(kwargs["hour"]) / 24
            self.time_sin = np.sin(rad)
            self.time_cos = np.cos(rad)
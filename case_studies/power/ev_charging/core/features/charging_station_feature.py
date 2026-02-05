from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
from heron.core.feature import FeatureProvider


@dataclass(slots=True)
class ChargingStationStatus(FeatureProvider):
    """
    visibility = ['owner', 'upper_level']
    """
    visibility: List[str] = ("owner", "upper_level")

    occupancy: float = 0.0
    price: float = 0.35

    def vector(self) -> np.ndarray:
        return np.array([self.occupancy, self.price], dtype=np.float32)

    def names(self) -> List[str]:
        return ["occupancy", "price"]

    def to_dict(self) -> Dict[str, Any]:
        return {"occupancy": self.occupancy, "price": self.price}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChargingStationStatus":
        return cls(occupancy=d.get("occupancy", 0.0), price=d.get("price", 0.35))

    def set_values(self, **kwargs: Any):
        if "occupancy" in kwargs:
            self.occupancy = np.clip(float(kwargs["occupancy"]), 0.0, 1.0)
        if "price" in kwargs:
            self.price = float(kwargs["price"])
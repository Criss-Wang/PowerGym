from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from ..config import EVConfig

StationsInfo = Dict[str, Dict[str, Any]]

@dataclass(slots=True)
class EV:
    time_arrival: float
    cfg: EVConfig
    rng: np.random.Generator
    u_tol: float = 0.0

    battery_kwh: float = 75.0
    soc_ini: float = 0.2
    demand_soc_min: float = 0.3
    alpha: float = 1.0

    def __post_init__(self):
        self.battery_kwh = float(self.rng.choice(self.cfg.battery_capacity_list))
        self.soc_ini = float(self.rng.uniform(*self.cfg.soc_ini_range))
        self.demand_soc_min = float(self.rng.uniform(*self.cfg.demand_soc_min_range))
        lo, hi = self.cfg.alpha_range()
        self.alpha = float(self.rng.uniform(lo, hi))

    @property
    def energy_need_kwh(self) -> float:
        need = max(0.0, (self.demand_soc_min - self.soc_ini) * self.battery_kwh)
        return float(max(need, 5.0))

    def choose_station(self, evsts_info: StationsInfo) -> Tuple[Optional[str], float, float]:
        best = None
        best_u = -1e18
        for name, info in evsts_info.items():
            price = float(info.get("price", 0.25))
            occ = float(info.get("occupancy", 0.0))
            u = - self.alpha * price - 0.5 * occ + float(self.rng.normal(0.0, 0.05))
            if u > best_u:
                best_u = u
                best = name
        if best is None:
            return None, 0.0, -1e18
        return best, 0.0, best_u

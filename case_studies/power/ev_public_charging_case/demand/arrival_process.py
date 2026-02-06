from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..config import ArrivalConfig

@dataclass(slots=True)
class HourlyPoissonArrival:
    num_stations: int
    rng: np.random.Generator
    cfg: ArrivalConfig

    def rate_per_second(self, t: float) -> float:
        hour_index = int((t % 86400) // 3600)
        rate_per_hour = float(self.cfg.hourly_rate[hour_index]) * float(self.cfg.scale)
        if not getattr(self.cfg, "rate_is_system_total", True):
            rate_per_hour *= int(self.num_stations)
        return rate_per_hour / 3600.0

    def sample(self, t: float, dt: float) -> int:
        lam = self.rate_per_second(t) * float(dt)
        return int(self.rng.poisson(lam))

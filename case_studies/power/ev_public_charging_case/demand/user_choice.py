from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from ..config import EVConfig
from ..entities.ev import EV

StationInfo = Dict[str, Any]
StationsInfo = Dict[str, StationInfo]

@dataclass(slots=True)
class UtilityBasedUserChoice:
    ev_cfg: EVConfig
    rng: np.random.Generator
    u_tol: float = 0.0

    def allocate(
        self,
        t: float,
        arrivals: int,
        evsts_info: StationsInfo,
        num_chargers_total: int,
    ) -> Tuple[Dict[str, List[EV]], Dict[str, int]]:
        remaining = int(num_chargers_total - sum(int(info["num_users"]) for info in evsts_info.values()))
        allocation: Dict[str, List[EV]] = {name: [] for name in evsts_info}
        blocked = giveup = assigned = 0

        for _ in range(int(arrivals)):
            if remaining <= 0:
                blocked += 1
                continue
            ev = EV(time_arrival=t, cfg=self.ev_cfg, rng=self.rng, u_tol=self.u_tol)
            st_name, _, _ = ev.choose_station(evsts_info)
            if st_name is None or st_name not in evsts_info:
                giveup += 1
                continue
            allocation[st_name].append(ev)
            evsts_info[st_name]["num_users"] += 1
            remaining -= 1
            assigned += 1

        return allocation, {"blocked": blocked, "giveup": giveup, "assigned": assigned}

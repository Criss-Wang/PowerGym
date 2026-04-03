"""Exogenous disturbance events for event-driven scheduling (Class 4).

A Disturbance represents an external state change (line fault, load spike, etc.)
that arrives at a stochastic time and modifies environment state discontinuously.
A DisturbanceSchedule collects disturbances for an episode and enqueues them
as ENV_UPDATE events on the scheduler timeline at reset time.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from heron.agents.constants import SYSTEM_AGENT_ID
from heron.scheduling.event import Event, EventType

if TYPE_CHECKING:
    from heron.scheduling.scheduler import EventScheduler


@dataclass
class Disturbance:
    """A single exogenous disturbance event.

    Attributes:
        timestamp: When the disturbance occurs (simulation time).
        disturbance_type: Domain-specific type string
            (e.g., ``"line_fault"``, ``"load_spike"``, ``"generation_drop"``).
        payload: Domain-specific parameters
            (e.g., ``{"element": "line_7_8", "action": "disconnect"}``).
        requires_physics: If True, an immediate SIMULATION is triggered
            after the disturbance is applied (e.g., topology change
            requires re-solving power flow).  Default True.
    """

    timestamp: float
    disturbance_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    requires_physics: bool = True


class DisturbanceSchedule:
    """A schedule of exogenous disturbances for one episode.

    Can be deterministic (scripted scenario) or stochastic (sampled per
    episode).  Enqueued onto the event scheduler at episode reset.

    Examples::

        # Deterministic
        DisturbanceSchedule([
            Disturbance(12.3, "line_fault", {"element": "line_7_8"}),
            Disturbance(45.0, "load_spike", {"bus": 5, "delta_kw": 500}),
        ])

        # From dicts
        DisturbanceSchedule.from_list([
            {"t": 12.3, "type": "line_fault", "element": "line_7_8"},
        ])

        # Stochastic (Poisson process)
        DisturbanceSchedule.poisson(rate=0.1, disturbance_types=["line_fault"],
                                    t_end=100.0, rng=np.random.default_rng(42))
    """

    def __init__(self, disturbances: Optional[List[Disturbance]] = None):
        self.disturbances: List[Disturbance] = sorted(
            disturbances or [], key=lambda d: d.timestamp
        )

    @classmethod
    def from_list(cls, specs: List[Dict[str, Any]]) -> "DisturbanceSchedule":
        """Create from a list of dicts.

        Each dict must have ``"t"`` (timestamp) and ``"type"``
        (disturbance_type).  All other keys become payload entries.
        Optional ``"requires_physics"`` key (defaults to True).

        Args:
            specs: List of disturbance specification dicts.
        """
        return cls([
            Disturbance(
                timestamp=s["t"],
                disturbance_type=s["type"],
                payload={
                    k: v for k, v in s.items()
                    if k not in ("t", "type", "requires_physics")
                },
                requires_physics=s.get("requires_physics", True),
            )
            for s in specs
        ])

    @classmethod
    def poisson(
        cls,
        rate: float,
        disturbance_types: List[str],
        t_end: float,
        rng: Optional[np.random.Generator] = None,
        default_payload: Optional[Dict[str, Any]] = None,
    ) -> "DisturbanceSchedule":
        """Generate a stochastic disturbance schedule via Poisson process.

        Args:
            rate: Average number of disturbances per unit time.
            disturbance_types: Pool of disturbance types to sample from.
            t_end: End of episode time (exclusive).
            rng: NumPy random Generator (seeded for reproducibility).
            default_payload: Default payload for generated disturbances.
        """
        if rng is None:
            rng = np.random.default_rng()
        disturbances: List[Disturbance] = []
        t = 0.0
        while t < t_end:
            t += rng.exponential(1.0 / rate)
            if t < t_end:
                dtype = str(rng.choice(disturbance_types))
                disturbances.append(Disturbance(
                    timestamp=t,
                    disturbance_type=dtype,
                    payload=dict(default_payload or {}),
                ))
        return cls(disturbances)

    def enqueue(self, scheduler: "EventScheduler") -> None:
        """Place all disturbances as ENV_UPDATE events on the scheduler.

        Each event targets the SystemAgent with priority=0 (same as
        ACTION_EFFECT) so disturbances settle before periodic physics.
        """
        for d in self.disturbances:
            scheduler.schedule(Event(
                timestamp=d.timestamp,
                event_type=EventType.ENV_UPDATE,
                agent_id=SYSTEM_AGENT_ID,
                priority=0,
                payload={"disturbance": d},
            ))

    def __len__(self) -> int:
        return len(self.disturbances)

    def __repr__(self) -> str:
        return f"DisturbanceSchedule(n={len(self.disturbances)})"

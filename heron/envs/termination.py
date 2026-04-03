"""Episode termination and truncation configuration.

Provides ``TerminationConfig`` for environment-level episode limits and
``compute_all_done`` — a shared utility that both training and event-driven
modes use to derive the ``__all__`` flag from per-agent flags.

Example — via ``EnvBuilder``::

    env = (
        EnvBuilder("my_env")
        .add_agents("agent", MyAgent, count=3, features=[MyFeature()])
        .simulation(my_sim)
        .termination(max_steps=100, all_semantics="any")
        .build()
    )

Example — direct construction::

    env = DefaultHeronEnv(
        agents=agents,
        hierarchy=hierarchy,
        simulation_func=my_sim,
        termination_config=TerminationConfig.with_max_steps(50),
    )
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AllSemantics(Enum):
    """How per-agent terminated/truncated flags combine into ``__all__``.

    Attributes:
        ALL: ``__all__`` is ``True`` when **every** agent is done (default).
        ANY: ``__all__`` is ``True`` when **any** single agent is done.
    """

    ALL = "all"
    ANY = "any"


@dataclass
class TerminationConfig:
    """Episode termination and truncation configuration.

    Attributes:
        max_steps: Maximum env steps before truncation (training mode).
            ``None`` means no step limit — only agent-initiated termination.
        max_sim_time: Maximum simulation time before truncation (event-driven).
            ``None`` means no time limit — relies on ``t_end`` from
            ``run_event_driven()``.
        all_semantics: How per-agent flags combine into ``__all__``.

    Raises:
        ValueError: If ``max_steps < 1`` or ``max_sim_time <= 0``.
    """

    max_steps: Optional[int] = None
    max_sim_time: Optional[float] = None
    all_semantics: AllSemantics = AllSemantics.ALL

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 or None, got {self.max_steps}")
        if self.max_sim_time is not None and self.max_sim_time <= 0.0:
            raise ValueError(f"max_sim_time must be > 0 or None, got {self.max_sim_time}")

    # ------------------------------------------------------------------
    # Factory class methods (follow ScheduleConfig pattern)
    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "TerminationConfig":
        """No limits, all-agents semantics."""
        return cls()

    @classmethod
    def with_max_steps(
        cls,
        max_steps: int,
        all_semantics: AllSemantics = AllSemantics.ALL,
    ) -> "TerminationConfig":
        """Training-mode truncation at *max_steps*."""
        return cls(max_steps=max_steps, all_semantics=all_semantics)

    @classmethod
    def with_max_sim_time(
        cls,
        max_sim_time: float,
        all_semantics: AllSemantics = AllSemantics.ALL,
    ) -> "TerminationConfig":
        """Event-driven truncation at *max_sim_time*."""
        return cls(max_sim_time=max_sim_time, all_semantics=all_semantics)


# Re-export from heron.core for convenience
from heron.core.env_context import compute_all_done  # noqa: F401

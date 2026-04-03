"""Environment context for agent termination decisions.

``EnvContext`` is a frozen dataclass providing a read-only snapshot of
environment-level state.  It lives in ``heron.core`` (not ``heron.envs``)
because it is consumed by both agents and environments — placing it in
``heron.envs`` would introduce a cross-layer import from agents → envs.

Training mode
    A fresh ``EnvContext`` is created each ``BaseEnv.step()`` with an
    up-to-date ``step_count`` and ``sim_time``.

Event-driven mode
    ``BaseEnv`` publishes a *template* context on the ``Proxy`` at the
    start of ``run_event_driven()``.  The ``Proxy`` creates a fresh
    snapshot per ``LOCAL_STATE`` response, filling in the live
    ``scheduler.current_time`` so that agents always see an accurate
    ``sim_time``.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Set


@dataclass(frozen=True)
class EnvContext:
    """Read-only snapshot of environment-level state for agent termination decisions.

    Attributes:
        step_count: Current env step (1-indexed after first step, training mode).
        sim_time: Current simulation time (event-driven mode; 0.0 at reset).
        max_steps: Configured max steps before truncation, or ``None``.
        max_sim_time: Configured max simulation time before truncation, or ``None``.
        all_semantics: How per-agent flags combine into ``__all__``.
            ``"all"`` (default) — every agent must be done.
            ``"any"`` — any single agent triggers ``__all__``.

    Example::

        def is_terminated(self, local_state, env_context=None):
            if env_context and env_context.step_count > 100:
                return True
            soc = local_state.get("BatteryFeature", [1.0])[0]
            return soc <= 0.0
    """

    step_count: int = 0
    sim_time: float = 0.0
    max_steps: Optional[int] = None
    max_sim_time: Optional[float] = None
    all_semantics: str = "all"


def compute_all_done(
    agent_flags: Dict[str, bool],
    expected_agents: Set[str],
    all_semantics: str = "all",
) -> bool:
    """Compute ``__all__`` from per-agent terminated/truncated flags.

    Shared by both ``SystemAgent.execute()`` (training) and
    ``BaseEnv._check_terminated_event_driven()`` (event-driven) to
    prevent logic drift between modes.

    Lives in ``heron.core`` so both agents and envs can import it
    without cross-layer dependencies.

    Args:
        agent_flags: ``{agent_id: bool}`` — latest flag per agent.
        expected_agents: Set of agent IDs that should be considered.
            Agents not in *agent_flags* default to ``False``.
        all_semantics: ``"all"`` or ``"any"``.

    Returns:
        ``True`` if the ``__all__`` condition is met.
        ``False`` if *expected_agents* is empty (guards ``all([]) == True``).
    """
    if not expected_agents:
        return False
    full = {aid: agent_flags.get(aid, False) for aid in expected_agents}
    combiner = all if all_semantics == "all" else any
    return combiner(full.values())

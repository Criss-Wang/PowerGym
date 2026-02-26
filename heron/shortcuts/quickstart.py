"""Convenience helper for creating a SimpleEnv in one call.

Usage::

    import heron

    env = heron.quickstart.make_env(agent_1, agent_2, simulation_fn=my_sim)
    obs, _ = env.reset()
"""

from typing import Callable, Optional

from heron.agents.field_agent import FieldAgent
from heron.shortcuts.env_builder import EnvBuilder
from heron.shortcuts.simulation_bridge import SimpleEnv


def make_env(
    *agents: FieldAgent,
    simulation_fn: Optional[Callable] = None,
) -> SimpleEnv:
    """Create a ``SimpleEnv`` from pre-built agents.

    A thin convenience wrapper around ``EnvBuilder`` for the common case
    where no custom coordinator wiring is needed.

    Args:
        *agents: Pre-built ``FieldAgent`` instances.
        simulation_fn: Optional ``(flat_states) -> flat_states`` function.

    Returns:
        A ``SimpleEnv`` instance ready for ``reset()`` / ``step()``.
    """
    builder = EnvBuilder().add_agents(*agents)
    if simulation_fn is not None:
        builder = builder.with_simulation(simulation_fn)
    return builder.build()

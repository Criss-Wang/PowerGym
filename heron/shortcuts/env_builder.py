"""Fluent factory for wiring HERON agent hierarchies.

Replaces the manual creation of coordinators, system agents,
and env subclass with a concise builder API.  Users construct
their own agent instances and pass them in.

Usage::

    env = (
        EnvBuilder()
        .add_agent(DeviceAgent(agent_id="d1", features=[PowerFeature()]))
        .add_agent(DeviceAgent(agent_id="d2", features=[PowerFeature()]))
        .add_coordinator("zone", subordinates=["d*"])
        .with_simulation(simulate)
        .build()
    )
"""

from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Optional

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.agents.system_agent import SystemAgent
from heron.shortcuts.simulation_bridge import SimpleEnv


class EnvBuilder:
    """Fluent builder for constructing a ``SimpleEnv``.

    Example::

        env = (
            EnvBuilder()
            .add_agent(DeviceAgent(agent_id="d1", features=[PowerFeature()]))
            .add_agent(DeviceAgent(agent_id="d2", features=[PowerFeature()]))
            .add_coordinator("zone", subordinates=["d*"])
            .with_simulation(my_sim_fn)
            .build()
        )
    """

    def __init__(self) -> None:
        self._agents: Dict[str, FieldAgent] = {}
        self._coordinator_specs: List[Dict[str, Any]] = []
        self._simulate_fn: Optional[Callable] = None
        self._scheduler_config: Optional[Dict[str, Any]] = None
        self._broker_config: Optional[Dict[str, Any]] = None
        self._simulation_wait: Optional[float] = None

    # -- Agent registration ---------------------------------------------------

    def add_agent(self, agent: FieldAgent) -> "EnvBuilder":
        """Register a pre-built field agent.

        Args:
            agent: A fully constructed ``FieldAgent`` instance.
        """
        self._agents[agent.agent_id] = agent
        return self

    def add_agents(self, *agents: FieldAgent) -> "EnvBuilder":
        """Register multiple pre-built field agents.

        Args:
            *agents: ``FieldAgent`` instances to register.
        """
        for agent in agents:
            self._agents[agent.agent_id] = agent
        return self

    # -- Coordinator registration ---------------------------------------------

    def add_coordinator(
        self,
        coordinator_id: str,
        subordinates: Optional[List[str]] = None,
        **coord_kwargs: Any,
    ) -> "EnvBuilder":
        """Add a coordinator with glob-pattern subordinate matching.

        Args:
            coordinator_id: Coordinator agent ID.
            subordinates: List of agent ID patterns (e.g. ``["device_*"]``).
                If ``None``, all registered field agents are included.
            **coord_kwargs: Forwarded to ``CoordinatorAgent`` constructor
                (e.g. ``protocol=VerticalProtocol()``).
        """
        self._coordinator_specs.append({
            "id": coordinator_id,
            "subordinates": subordinates,
            **coord_kwargs,
        })
        return self

    # -- Configuration --------------------------------------------------------

    def with_simulation(self, fn: Callable) -> "EnvBuilder":
        """Set the simulation function ``(flat_states) -> flat_states``."""
        self._simulate_fn = fn
        return self

    def with_scheduler(self, config: Dict[str, Any]) -> "EnvBuilder":
        """Set scheduler config."""
        self._scheduler_config = config
        return self

    def with_broker(self, config: Dict[str, Any]) -> "EnvBuilder":
        """Set message broker config."""
        self._broker_config = config
        return self

    def with_simulation_wait(self, interval: float) -> "EnvBuilder":
        """Set simulation wait interval."""
        self._simulation_wait = interval
        return self

    # -- Build ----------------------------------------------------------------

    def build(self) -> SimpleEnv:
        """Construct the ``SimpleEnv`` from the accumulated configuration.

        If no coordinators were added, a default coordinator is created that
        manages all field agents. A ``SystemAgent`` is always created
        automatically.
        """
        coordinators: Dict[str, CoordinatorAgent] = {}

        if not self._coordinator_specs:
            coord = CoordinatorAgent(
                agent_id="coordinator",
                subordinates=dict(self._agents),
            )
            coordinators["coordinator"] = coord
        else:
            for spec in self._coordinator_specs:
                cid = spec.pop("id")
                sub_patterns = spec.pop("subordinates", None)

                if sub_patterns is None:
                    matched = dict(self._agents)
                else:
                    matched = {}
                    for pattern in sub_patterns:
                        for aid, agent in self._agents.items():
                            if fnmatch(aid, pattern):
                                matched[aid] = agent

                coord = CoordinatorAgent(
                    agent_id=cid,
                    subordinates=matched,
                    **spec,
                )
                coordinators[cid] = coord

        system = SystemAgent(
            agent_id="system_agent",
            subordinates=coordinators,
        )

        return SimpleEnv(
            simulate_fn=self._simulate_fn,
            system_agent=system,
            scheduler_config=self._scheduler_config or {"start_time": 0.0, "time_step": 1.0},
            message_broker_config=self._broker_config or {"buffer_size": 1000, "max_queue_size": 100},
            simulation_wait_interval=self._simulation_wait or 0.01,
        )

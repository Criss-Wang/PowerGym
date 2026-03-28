"""Fluent factory for constructing HERON environments.

``EnvBuilder`` eliminates manual agent-hierarchy wiring by providing a
chainable API that resolves subordinate assignments via glob patterns
and builds a ready-to-train environment.

Example::

    from heron.envs.builder import EnvBuilder

    env = (
        EnvBuilder("my_env")
        .add_agents("battery", BatteryAgent, count=3, features=[BatteryFeature()])
        .add_coordinator("zone", subordinates=["battery_*"])
        .simulation(my_sim_func)
        .build()
    )

With ``RLlibBasedHeronEnv``, pass agent specs as plain dicts — the adapter
builds the ``EnvBuilder`` internally::

    config = PPOConfig().environment(
        env=RLlibBasedHeronEnv,
        env_config={
            "agents": [{"agent_id": "b0", "agent_cls": BatteryAgent,
                        "features": [BatteryFeature()]}],
            "simulation": my_sim_func,
            "max_steps": 100,
        },
    )

If no coordinator is specified, field agents are attached directly to the
system agent.
"""

import copy
import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.agents.system_agent import SystemAgent
from heron.core.feature import Feature
from heron.scheduling.schedule_config import ScheduleConfig
from heron.envs.base import HeronEnv
from heron.protocols.base import Protocol
from heron.envs.simple import SimpleEnv


@dataclass
class _AgentSpec:
    agent_cls: Type[FieldAgent]
    agent_id: str
    features: List[Feature] = field(default_factory=list)
    coordinator_id: Optional[str] = None
    schedule_config: Optional[ScheduleConfig] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _CoordinatorSpec:
    agent_id: str
    agent_cls: Type[CoordinatorAgent] = CoordinatorAgent
    features: List[Feature] = field(default_factory=list)
    protocol: Optional[Protocol] = None
    subordinate_patterns: List[str] = field(default_factory=list)
    subordinate_ids: List[str] = field(default_factory=list)
    schedule_config: Optional[ScheduleConfig] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class _SystemSpec:
    features: List[Feature] = field(default_factory=list)
    schedule_config: Optional[ScheduleConfig] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


class EnvBuilder:
    """Fluent factory for constructing HERON environments.

    Parameters
    ----------
    env_id : str
        Environment identifier (default ``"default_env"``).
    """

    def __init__(self, env_id: str = "default_env") -> None:
        self._env_id = env_id
        self._agent_specs: List[_AgentSpec] = []
        self._coordinator_specs: List[_CoordinatorSpec] = []
        self._system_spec: Optional[_SystemSpec] = None
        self._simulation_func: Optional[Callable] = None
        self._env_cls: Optional[Type[HeronEnv]] = None
        self._env_kwargs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    #  Agent registration
    # ------------------------------------------------------------------

    def add_agents(
        self,
        prefix: str,
        agent_cls: Type[FieldAgent],
        count: int = 1,
        features: Optional[List[Feature]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        coordinator: Optional[str] = None,
        **kwargs: Any,
    ) -> "EnvBuilder":
        """Register *count* field agents with auto-generated IDs.

        IDs follow the pattern ``{prefix}_0``, ``{prefix}_1``, ...
        When *count* is 1 the ID is just *prefix*.
        """
        for i in range(count):
            agent_id = f"{prefix}_{i}" if count > 1 else prefix
            self._agent_specs.append(_AgentSpec(
                agent_cls=agent_cls,
                agent_id=agent_id,
                features=list(features or []),
                coordinator_id=coordinator,
                schedule_config=schedule_config,
                kwargs=dict(kwargs),
            ))
        return self

    def add_agent(
        self,
        agent_id: str,
        agent_cls: Type[FieldAgent],
        features: Optional[List[Feature]] = None,
        coordinator: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> "EnvBuilder":
        """Register a single named field agent."""
        self._agent_specs.append(_AgentSpec(
            agent_cls=agent_cls,
            agent_id=agent_id,
            features=list(features or []),
            coordinator_id=coordinator,
            schedule_config=schedule_config,
            kwargs=dict(kwargs),
        ))
        return self

    # ------------------------------------------------------------------
    #  Coordinator registration
    # ------------------------------------------------------------------

    def add_coordinator(
        self,
        coordinator_id: str,
        agent_cls: Type[CoordinatorAgent] = CoordinatorAgent,
        features: Optional[List[Feature]] = None,
        protocol: Optional[Protocol] = None,
        subordinates: Optional[List[str]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> "EnvBuilder":
        """Register a coordinator agent.

        *subordinates* can contain exact agent IDs or glob patterns
        (e.g. ``"battery_*"``).
        """
        patterns: List[str] = []
        explicit: List[str] = []
        for s in (subordinates or []):
            if "*" in s or "?" in s:
                patterns.append(s)
            else:
                explicit.append(s)

        self._coordinator_specs.append(_CoordinatorSpec(
            agent_id=coordinator_id,
            agent_cls=agent_cls,
            features=list(features or []),
            protocol=protocol,
            subordinate_patterns=patterns,
            subordinate_ids=explicit,
            schedule_config=schedule_config,
            kwargs=dict(kwargs),
        ))
        return self
    
    def add_system_agent(
        self,
        features: Optional[List[Feature]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> "EnvBuilder":
        """Configure the SystemAgent (auto-created if not specified)."""
        if features or schedule_config or kwargs:
            self._system_spec = _SystemSpec(
                features=list(features or []),
                schedule_config=schedule_config,
                kwargs=dict(kwargs),
            )
        return self

    # ------------------------------------------------------------------
    #  Environment configuration
    # ------------------------------------------------------------------

    def simulation(self, func: Callable) -> "EnvBuilder":
        """Set the simulation function (``SimpleEnv`` auto-bridge)."""
        self._simulation_func = func
        return self

    def env_class(self, cls: Type[HeronEnv], **kwargs: Any) -> "EnvBuilder":
        """Use a specific ``HeronEnv`` subclass instead of ``SimpleEnv``."""
        self._env_cls = cls
        self._env_kwargs = kwargs
        return self

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------

    def build(self) -> HeronEnv:
        """Construct and return the configured environment."""
        field_agents = self._instantiate_agents()
        coordinators, coord_hierarchy, unassigned = self._resolve_coordinators(field_agents)
        system_agent, sys_hierarchy = self._resolve_system_agent(coordinators, unassigned)

        all_agents = list(field_agents.values()) + coordinators + [system_agent]
        hierarchy = {**coord_hierarchy, **sys_hierarchy}

        return self._build_env(all_agents, hierarchy)

    def __call__(self, config: Any = None) -> HeronEnv:
        """Build the environment (callable shorthand for ``build()``).

        Kept for backward compatibility with code that uses the builder
        as a callable factory.
        """
        return self.build()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _instantiate_agents(self) -> Dict[str, Agent]:
        agents: Dict[str, Agent] = {}
        for spec in self._agent_specs:
            ctor_kwargs = dict(agent_id=spec.agent_id, **spec.kwargs)
            if spec.features:
                ctor_kwargs["features"] = [copy.deepcopy(f) for f in spec.features]
            if spec.schedule_config is not None:
                ctor_kwargs["schedule_config"] = spec.schedule_config
            agents[spec.agent_id] = spec.agent_cls(**ctor_kwargs)
        return agents

    def _resolve_coordinators(
        self, agents: Dict[str, Agent],
    ) -> tuple[List[Agent], Dict[str, List[str]], List[str]]:
        """Create coordinators (without subordinates) and return hierarchy edges.

        Returns:
            Tuple of (coordinator_list, hierarchy_dict, unassigned_ids) where
            hierarchy_dict maps coordinator_id -> [subordinate_agent_ids] and
            unassigned_ids are agents not assigned to any coordinator.
        """
        coordinators: List[Agent] = []
        hierarchy: Dict[str, List[str]] = {}
        assigned: set = set()

        for cspec in self._coordinator_specs:
            sub_ids: List[str] = list(cspec.subordinate_ids)

            # Resolve glob patterns
            for pattern in cspec.subordinate_patterns:
                for aid in agents:
                    if fnmatch.fnmatch(aid, pattern) and aid not in sub_ids:
                        sub_ids.append(aid)

            # Include agents that declared this coordinator
            for aspec in self._agent_specs:
                if aspec.coordinator_id == cspec.agent_id and aspec.agent_id not in sub_ids:
                    sub_ids.append(aspec.agent_id)

            # Filter to agents that actually exist
            sub_ids = [aid for aid in sub_ids if aid in agents]
            assigned.update(sub_ids)
            hierarchy[cspec.agent_id] = sub_ids

            features = [copy.deepcopy(f) for f in cspec.features]
            ctor_kwargs = dict(
                agent_id=cspec.agent_id,
                features=features,
                **cspec.kwargs,
            )
            if cspec.schedule_config is not None:
                ctor_kwargs["schedule_config"] = cspec.schedule_config
            if cspec.protocol is not None:
                ctor_kwargs["protocol"] = cspec.protocol
            coordinators.append(cspec.agent_cls(**ctor_kwargs))

        unassigned = [aid for aid in agents if aid not in assigned]
        return coordinators, hierarchy, unassigned
    
    def _resolve_system_agent(
        self,
        coordinators: List[Agent],
        unassigned_ids: List[str],
    ) -> tuple[SystemAgent, Dict[str, List[str]]]:
        """Create system agent (without subordinates) and return hierarchy edges.

        Unassigned field agents are attached directly under the system agent
        alongside any coordinators.
        """
        child_ids = [c.agent_id for c in coordinators] + unassigned_ids

        if self._system_spec:
            features = list(self._system_spec.features)
            schedule_config = self._system_spec.schedule_config
            kwargs = dict(self._system_spec.kwargs)
            system_agent = SystemAgent(
                features=features,
                schedule_config=schedule_config,
                **kwargs,
            )
        else:
            system_agent = SystemAgent()

        hierarchy = {system_agent.agent_id: child_ids}
        return system_agent, hierarchy

    def _build_env(
        self,
        agents: List,
        hierarchy: Dict[str, List[str]],
    ) -> HeronEnv:
        if self._env_cls is not None:
            return self._env_cls(
                agents=agents,
                hierarchy=hierarchy,
                env_id=self._env_id,
                **self._env_kwargs,
            )

        if self._simulation_func is not None:
            return SimpleEnv(
                agents=agents,
                hierarchy=hierarchy,
                env_id=self._env_id,
                simulation_func=self._simulation_func,
            )

        raise ValueError(
            "EnvBuilder requires either a simulation function (.simulation()) "
            "or a custom env class (.env_class()) to build an environment."
        )


import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import uuid

import gymnasium as gym

from heron.agents.base import Agent
from heron.core.action import Action
from heron.core.policies import Policy
from heron.messaging import MessageBroker, ChannelManager, Message, MessageType
from heron.utils.typing import AgentID, MultiAgentDict
from heron.scheduling import EventScheduler, Event, EpisodeAnalyzer, EpisodeStats
from heron.agents.system_agent import SystemAgent
from heron.agents.proxy_agent import Proxy
from heron.agents.constants import PROXY_AGENT_ID

logger = logging.getLogger(__name__)


class BaseEnv(ABC):
    def __init__(
        self,
        agents: List[Agent],
        hierarchy: Dict[AgentID, List[AgentID]],
        env_id: Optional[str] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        message_broker_config: Optional[Dict[str, Any]] = None,
        # simulation-related params
        simulation_wait_interval: Optional[float] = None,
        # disturbance-related params (Class 4)
        disturbance_schedule: Optional[Any] = None,
    ) -> None:
        # environment attributes
        self.env_id = env_id or f"env_{uuid.uuid4().hex[:8]}"
        self.simulation_wait_interval = simulation_wait_interval
        self.disturbance_schedule = disturbance_schedule

        # agent-specific fields
        self.registered_agents: Dict[AgentID, Agent] = {}
        self._register_agents(agents, hierarchy)

        # initialize proxy agent (singleton) for state access and action dispatch
        self.proxy = Proxy(agent_id=PROXY_AGENT_ID)
        self._register_agent(self.proxy)

        # setup message broker (before proxy attach - proxy needs it for channels)
        self.message_broker = MessageBroker.init(message_broker_config)
        self.message_broker.attach(self.registered_agents)

        # attach message broker to proxy agent for communication
        self.proxy.set_message_broker(self.message_broker)
        # establish direction link between registered agents and proxy for state access
        self.proxy.attach(self.registered_agents)

        # setup scheduler (before initialization - agents need it)
        self.scheduler = EventScheduler.init(scheduler_config)
        self.scheduler.attach(self.registered_agents)

    # ============================================
    # Agent Management Methods
    # ============================================
    def _register_agents(
        self,
        agents: List[Agent],
        hierarchy: Dict[AgentID, List[AgentID]],
    ) -> None:
        """Validate, wire, configure, and register all agents."""
        agent_map = {a.agent_id: a for a in agents}
        root_id = self._validate_hierarchy(agent_map, hierarchy)
        self._wire_hierarchy(agent_map, hierarchy, root_id)
        self._system_agent = agent_map[root_id]
        self._configure_simulation(self._system_agent)
        self._register_agent(self._system_agent)

    @staticmethod
    def _validate_hierarchy(
        agent_map: Dict[AgentID, Agent],
        hierarchy: Dict[AgentID, List[AgentID]],
    ) -> AgentID:
        """Validate hierarchy and return the root agent ID.

        Checks:
            - All parent/child IDs in hierarchy exist in agent_map
            - Exactly one root (parent that never appears as a child)
            - Root is a SystemAgent
            - No orphaned agents (in agent_map but absent from hierarchy)
        """
        all_children: set = set()
        for parent_id, child_ids in hierarchy.items():
            if parent_id not in agent_map:
                raise ValueError(f"Parent '{parent_id}' in hierarchy not found in agents list.")
            for cid in child_ids:
                if cid not in agent_map:
                    raise ValueError(f"Child '{cid}' in hierarchy not found in agents list.")
                all_children.add(cid)

        parents = set(hierarchy.keys())
        roots = parents - all_children
        if len(roots) != 1:
            raise ValueError(f"Hierarchy must have exactly one root agent, found: {roots}")
        root_id = roots.pop()
        if not isinstance(agent_map[root_id], SystemAgent):
            raise TypeError(f"Root agent '{root_id}' must be a SystemAgent, got {type(agent_map[root_id]).__name__}.")

        orphans = set(agent_map.keys()) - (parents | all_children)
        if orphans:
            raise ValueError(f"Agents not in hierarchy (orphaned): {orphans}")

        return root_id

    @staticmethod
    def _wire_hierarchy(
        agent_map: Dict[AgentID, Agent],
        hierarchy: Dict[AgentID, List[AgentID]],
        root_id: AgentID,
    ) -> None:
        """Set subordinates on each parent according to the hierarchy.

        After all subordinates are wired, re-resolves periodic children on
        the root SystemAgent (which caches the result internally).
        """
        for parent_id, child_ids in hierarchy.items():
            parent = agent_map[parent_id]
            subs = {cid: agent_map[cid] for cid in child_ids}
            parent.subordinates = parent.build_subordinates(subs)

        # Re-resolve periodic children now that hierarchy is fully wired
        # (resolve_periodic_children runs in __init__ when subordinates may be empty)
        agent_map[root_id].refresh_periodic_agents()

    def _configure_simulation(self, system_agent: SystemAgent) -> None:
        """Bind env simulation callables to the system agent."""
        system_agent.set_simulation(
            self.run_simulation,
            self.env_state_to_global_state,
            self.global_state_to_env_state,
            self.simulation_wait_interval,
            self.pre_step,
            self.apply_disturbance,
        )

    def get_agent(self, agent_id: AgentID) -> Optional[Agent]:
        """Get a registered agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent instance or None if not found
        """
        return self.registered_agents.get(agent_id)

    def _register_agent(self, agent: Agent) -> None:
        """Register an agent with the environment.

        Args:
            agent: Agent to register
        """
        agent.env_id = self.env_id
        self.registered_agents[agent.agent_id] = agent
        for subordinate in agent.subordinates.values():
            self._register_agent(subordinate)

    # ===========================================
    # Environment Interaction Methods
    # ==========================================
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        jitter_seed: Optional[int] = None,
        disturbance_schedule: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset all registered agents.

        Args:
            seed: Random seed for agent/env reset.
            jitter_seed: If provided, enables jitter on all agents with
                this seed (for reproducible event-driven evaluation).
                Tick configs should be set at agent construction time
                (e.g. via ``schedule_config`` in env_config).
            disturbance_schedule: If provided, overrides
                ``self.disturbance_schedule`` for this episode only.
                If ``None``, uses the schedule configured at construction.
            **kwargs: Additional reset parameters
        """
        # Re-seed jitter RNG per episode for reproducible event-driven eval
        if jitter_seed is not None:
            for agent in self.registered_agents.values():
                agent.enable_jitter(seed=jitter_seed)

        # Sync tick configs in case agents were reconfigured after construction
        self.scheduler.sync_schedule_configs(self.registered_agents)
        # reset scheduler and clear messages before resetting agents to ensure a clean slate
        self.scheduler.reset(start_time=0.0)  # Always reset to time 0
        self.clear_broker_environment()

        # Reset agents — SystemAgent.reset() returns (obs_vectorized, {})
        self.proxy.reset(seed=seed)
        obs = self._system_agent.reset(seed=seed, proxy=self.proxy)
        self.proxy.init_global_state()  # Cache initial state in proxy after reset

        # Enqueue disturbances: per-call override > env-level default
        schedule = disturbance_schedule if disturbance_schedule is not None else self.disturbance_schedule
        if schedule is not None:
            schedule.enqueue(self.scheduler)

        return obs
    
    def step(self, actions: Dict[AgentID, Any]) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict],
    ]:
        """Execute one environment step.

        The system_agent in the environment is responsible for entire 
        simulation step

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
            - observations: Dict mapping agent IDs to observation arrays
            - rewards: Dict mapping agent IDs to reward floats
            - terminated: Dict with agent IDs and "__all__" key
            - truncated: Dict with agent IDs and "__all__" key
            - infos: Dict mapping agent IDs to info dicts
        """
        self._system_agent.execute(actions, self.proxy)
        return self.proxy.get_step_results()
    
    def run_event_driven(
        self,
        t_end: float,
        episode_analyzer: Optional[EpisodeAnalyzer] = None,
        max_events: Optional[int] = None,
    ) -> EpisodeStats:
        """Run event-driven simulation until time limit.

        Args:
            t_end: Stop when simulation time exceeds this
            episode_analyzer: EpisodeAnalyzer to parse events during simulation.
                If None, a default EpisodeAnalyzer() is used.
            max_events: Optional maximum number of events to process

        Returns:
            EpisodeStats containing all event analyses from the simulation
        """
        if episode_analyzer is None:
            episode_analyzer = EpisodeAnalyzer()
        result = EpisodeStats()
        for event in self.scheduler.run_until(t_end=t_end, max_events=max_events):
            result.add_event_analysis(episode_analyzer.parse_event(event))
        return result

    # ============================================
    # Simulation-related Methods
    # ============================================
    def pre_step(self) -> None:
        """Hook called at the start of each step before agent actions.

        Override this method in subclasses to perform environment-specific
        setup at the beginning of each step (e.g., updating profiles, loading
        time-series data for current timestep).

        Default implementation is a no-op.
        """
        pass

    def apply_disturbance(self, disturbance: Any) -> None:
        """Apply an exogenous disturbance to the environment state (Class 4).

        Override in subclasses to handle domain-specific disturbance types.
        Examples:
          - Power systems: disconnect a line, change a load value
          - Traffic: block a road segment, change signal timing

        Called by SystemAgent's env_update_handler during event-driven execution.

        Args:
            disturbance: A ``Disturbance`` object with ``disturbance_type``,
                ``payload``, and ``requires_physics`` fields.
        """
        raise NotImplementedError(
            f"apply_disturbance() not implemented for {type(self).__name__}. "
            f"Override this method to handle disturbance type "
            f"'{getattr(disturbance, 'disturbance_type', '?')}'."
        )

    @abstractmethod
    def run_simulation(self, env_state: Any, *args, **kwargs) -> Any:
        """ Custom simulation logic post-system_agent.act and before system_agent.update_from_environment().

        In the long run, this can be eventually turned into a static SimulatorAgent.
        """
        pass

    @abstractmethod
    def env_state_to_global_state(self, env_state: Any) -> Dict[str, Any]:
        """Convert custom environment state to HERON global state dict format.

        This method is called after run_simulation() to convert the updated
        environment state back into the global state dict structure that will
        be stored in proxy.state_cache["global"].

        Args:
            env_state: Custom environment state after simulation

        Returns:
            Dict that will be merged into proxy.state_cache["global"] via .update()
            Typically includes "agent_states" dict with updated agent state dicts.

        Example:
            return {
                "agent_states": {
                    "agent_1": {"FeatureName": {"field": value}},
                    ...
                }
            }
        """
        pass

    @abstractmethod
    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> Any:
        """Convert HERON global state dict to custom environment state format.

        This method is called before run_simulation() to extract relevant info
        from proxy.state_cache["global"] and convert it to the custom format
        your simulation function expects.

        Args:
            global_state: Dict from proxy.state_cache["global"] with structure:
                {
                    "agent_states": {agent_id: state_dict, ...},
                    ... other global fields ...
                }
                Note: state_dict is the dict representation from State.to_dict()

        Returns:
            Custom environment state object for your simulation

        Example:
            agent_states = global_state.get("agent_states", {})
            battery_soc = agent_states["battery_1"]["BatteryFeature"]["soc"]
            return CustomEnvState(battery_soc=battery_soc)
        """
        pass


    # ============================================
    # Utility Methods
    # ============================================
    def get_all_policies(self) -> Dict[AgentID, Policy]:
        return {
            agent_id: agent.policy
            for agent_id, agent in self.registered_agents.items()
            if agent.policy
        }
    
    def set_agent_policies(self, policies: Dict[AgentID, Policy]) -> None:
        for agent_id, policy in policies.items():
            agent = self.registered_agents.get(agent_id)
            if agent:
                agent.policy = policy

    def clear_broker_environment(self) -> None:
        """Clear all messages for this environment from the broker.

        Useful for resetting the environment.
        """
        if self.message_broker is not None:
            self.message_broker.clear_environment(self.env_id)

    def close(self) -> None:
        """Clean up environment resources."""
        if self.message_broker is not None:
            self.message_broker.close()

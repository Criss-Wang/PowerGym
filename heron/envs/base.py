"""Abstract environment interfaces for the HERON framework.

This module defines the base environment classes that domain-specific
implementations should extend.

Architecture:
    HeronEnvCore (Mixin) - Core HERON functionality (agent mgmt, messaging)
    │
    ├── BaseEnv (gym.Env + HeronEnvCore) - Single-agent Gymnasium interface
    │
    ├── MultiAgentEnv (HeronEnvCore) - Abstract multi-agent base
    │   ├── PettingZooParallelEnv - PettingZoo ParallelEnv interface
    │   └── RLlibMultiAgentEnv - RLlib MultiAgentEnv interface

Execution Modes:
    1. Synchronous (default): All agents step together via step() method
    2. Distributed (async): Hierarchical message-based via step_distributed()
    3. Event-driven: Priority-queue scheduling via run_event_driven()
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import uuid
import asyncio

import gymnasium as gym
import numpy as np

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.messaging.base import MessageBroker
from heron.utils.typing import AgentID

if TYPE_CHECKING:
    from heron.scheduling import EventScheduler, Event


class HeronEnvCore:
    """Core mixin providing HERON-specific functionality.

    This mixin provides agent management, message broker integration,
    and distributed execution support. It does NOT inherit from any
    environment interface, allowing composition with different backends.

    Note: Internal attributes use underscore prefix (_heron_agents, _heron_coordinators)
    to avoid conflicts with framework-specific properties like PettingZoo's `agents`.
    Access via heron_agents/heron_coordinators properties or get_heron_agent() method.

    Attributes:
        env_id: Unique environment identifier
        message_broker: Optional message broker for distributed execution
        distributed: Whether to use distributed execution mode
        scheduler: Optional EventScheduler for event-driven execution
        heron_agents: Dictionary mapping agent IDs to Agent instances
        heron_coordinators: Dictionary mapping coordinator IDs to CoordinatorAgent instances
    """

    def _init_heron_core(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
        distributed: bool = False,
        scheduler: Optional["EventScheduler"] = None,
    ) -> None:
        """Initialize HERON core functionality.

        Call this in your __init__ after calling super().__init__().

        Args:
            env_id: Environment identifier (auto-generated if not provided)
            message_broker: Optional message broker for distributed mode
            distributed: If True, use distributed execution mode
            scheduler: Optional EventScheduler for event-driven execution
        """
        self.env_id = env_id or f"env_{uuid.uuid4().hex[:8]}"
        self.message_broker = message_broker
        self.distributed = distributed
        self.scheduler = scheduler
        self._timestep = 0

        # Use underscore prefix to avoid conflicts with framework properties
        # (e.g., PettingZoo's `agents` property)
        self._heron_agents: Dict[AgentID, Agent] = {}
        self._heron_coordinators: Dict[AgentID, CoordinatorAgent] = {}

    @property
    def heron_agents(self) -> Dict[AgentID, Agent]:
        """Dictionary of registered HERON agents."""
        return self._heron_agents

    @property
    def heron_coordinators(self) -> Dict[AgentID, CoordinatorAgent]:
        """Dictionary of registered coordinator agents."""
        return self._heron_coordinators

    def get_heron_agent(self, agent_id: AgentID) -> Optional[Agent]:
        """Get a registered agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent instance or None if not found
        """
        return self._heron_agents.get(agent_id)

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the environment.

        Args:
            agent: Agent to register
        """
        self._heron_agents[agent.agent_id] = agent
        if isinstance(agent, CoordinatorAgent):
            self._heron_coordinators[agent.agent_id] = agent

    def register_agents(self, agents: List[Agent]) -> None:
        """Register multiple agents.

        Args:
            agents: List of agents to register
        """
        for agent in agents:
            self.register_agent(agent)

    def get_observations(
        self, global_state: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Observation]:
        """Collect observations from all agents.

        Args:
            global_state: Optional global state to pass to agents

        Returns:
            Dictionary mapping agent IDs to observations
        """
        observations = {}
        for agent_id, agent in self._heron_agents.items():
            observations[agent_id] = agent.observe(global_state)
        return observations

    def apply_actions(
        self,
        actions: Dict[AgentID, Any],
        observations: Optional[Dict[AgentID, Observation]] = None,
    ) -> None:
        """Apply actions to agents.

        Args:
            actions: Dictionary mapping agent IDs to actions
            observations: Optional observations to pass to agents
        """
        for agent_id, action in actions.items():
            if agent_id in self._heron_agents:
                obs = observations.get(agent_id) if observations else None
                self._heron_agents[agent_id].act(obs, upstream_action=action)

    def get_agent_action_spaces(self) -> Dict[AgentID, gym.Space]:
        """Get action spaces for all agents.

        Returns:
            Dictionary mapping agent IDs to their action spaces
        """
        return {
            agent_id: agent.action_space
            for agent_id, agent in self._heron_agents.items()
            if agent.action_space is not None
        }

    def get_agent_observation_spaces(self) -> Dict[AgentID, gym.Space]:
        """Get observation spaces for all agents.

        Returns:
            Dictionary mapping agent IDs to their observation spaces
        """
        return {
            agent_id: agent.observation_space
            for agent_id, agent in self._heron_agents.items()
            if agent.observation_space is not None
        }

    def reset_agents(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset all registered agents.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        for agent in self._heron_agents.values():
            agent.reset(seed=seed, **kwargs)

    async def step_distributed(self) -> None:
        """Execute distributed step using message broker.

        This triggers the hierarchical execution flow where agents
        communicate via the message broker.
        """
        if not self.distributed or not self.message_broker:
            raise RuntimeError(
                "Distributed step requires distributed=True and message_broker"
            )

        # Execute all coordinators (they will recursively execute subordinates)
        await asyncio.gather(
            *[coord.step_distributed() for coord in self._heron_coordinators.values()]
        )

    def get_total_reward(self) -> Dict[str, float]:
        """Get total reward aggregated from all agents.

        Returns:
            Dictionary with total cost and safety
        """
        total_cost = sum(
            agent.cost if hasattr(agent, "cost") else 0
            for agent in self._heron_agents.values()
        )
        total_safety = sum(
            agent.safety if hasattr(agent, "safety") else 0
            for agent in self._heron_agents.values()
        )
        return {"cost": total_cost, "safety": total_safety}

    def close_heron(self) -> None:
        """Clean up HERON-specific resources."""
        if self.message_broker:
            self.message_broker.clear_environment(self.env_id)

    # ============================================
    # Event-Driven Execution Support
    # ============================================

    def setup_event_driven(
        self,
        scheduler: Optional["EventScheduler"] = None,
    ) -> "EventScheduler":
        """Setup event-driven execution with scheduler.

        Registers all agents with the scheduler using their timing parameters.
        Creates a new scheduler if none provided.

        Args:
            scheduler: Optional existing scheduler (creates new if None)

        Returns:
            The configured EventScheduler
        """
        from heron.scheduling import EventScheduler

        if scheduler is None:
            scheduler = EventScheduler(start_time=0.0)

        self.scheduler = scheduler

        # Register all agents with their timing parameters
        for agent_id, agent in self._heron_agents.items():
            scheduler.register_agent(
                agent_id=agent_id,
                tick_interval=getattr(agent, 'tick_interval', 1.0),
                obs_delay=getattr(agent, 'obs_delay', 0.0),
                act_delay=getattr(agent, 'act_delay', 0.0),
            )

        return scheduler

    def set_event_handlers(
        self,
        on_agent_tick: Optional[Callable[["Event", "EventScheduler"], None]] = None,
        on_action_effect: Optional[Callable[["Event", "EventScheduler"], None]] = None,
        on_message_delivery: Optional[Callable[["Event", "EventScheduler"], None]] = None,
    ) -> None:
        """Set event handlers for event-driven execution.

        Args:
            on_agent_tick: Handler for AGENT_TICK events
            on_action_effect: Handler for ACTION_EFFECT events
            on_message_delivery: Handler for MESSAGE_DELIVERY events
        """
        if self.scheduler is None:
            raise RuntimeError("Call setup_event_driven() first")

        from heron.scheduling import EventType

        if on_agent_tick:
            self.scheduler.set_handler(EventType.AGENT_TICK, on_agent_tick)
        if on_action_effect:
            self.scheduler.set_handler(EventType.ACTION_EFFECT, on_action_effect)
        if on_message_delivery:
            self.scheduler.set_handler(EventType.MESSAGE_DELIVERY, on_message_delivery)

    def run_event_driven(
        self,
        t_end: float,
        max_events: Optional[int] = None,
    ) -> int:
        """Run event-driven simulation until time limit.

        Args:
            t_end: Stop when simulation time exceeds this
            max_events: Optional maximum number of events to process

        Returns:
            Number of events processed

        Raises:
            RuntimeError: If scheduler not configured
        """
        if self.scheduler is None:
            raise RuntimeError("Call setup_event_driven() first")

        return self.scheduler.run_until(t_end=t_end, max_events=max_events)

    @property
    def simulation_time(self) -> float:
        """Current simulation time (from scheduler or timestep)."""
        if self.scheduler:
            return self.scheduler.current_time
        return float(self._timestep)


class BaseEnv(gym.Env, HeronEnvCore, ABC):
    """Abstract base environment for HERON with Gymnasium interface.

    Extends Gymnasium's Env interface with HERON-specific functionality
    for hierarchical multi-agent systems.

    This is suitable for single-agent environments or environments
    where you want a simple Gymnasium-compatible interface.
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
    ):
        """Initialize base environment.

        Args:
            env_id: Environment identifier (auto-generated if not provided)
            message_broker: Optional message broker for distributed mode
        """
        gym.Env.__init__(self)
        self._init_heron_core(
            env_id=env_id, message_broker=message_broker, distributed=False
        )

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (observation, info)
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        self.close_heron()


class MultiAgentEnv(HeronEnvCore, ABC):
    """Abstract base class for multi-agent environments.

    Provides common functionality for environments with multiple agents,
    including agent registration, observation collection, and action distribution.

    This class does NOT inherit from any specific environment interface.
    Use one of the interface adapters (PettingZooParallelEnv, RLlibMultiAgentEnv)
    for compatibility with specific frameworks.

    Subclasses must implement:
        - reset()
        - step()
        - get_joint_observation_space()
        - get_joint_action_space()
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
        distributed: bool = False,
        scheduler: Optional["EventScheduler"] = None,
    ):
        """Initialize multi-agent environment.

        Args:
            env_id: Environment identifier
            message_broker: Message broker for distributed mode
            distributed: If True, use distributed execution mode
            scheduler: Optional EventScheduler for event-driven mode
        """
        self._init_heron_core(
            env_id=env_id,
            message_broker=message_broker,
            distributed=distributed,
            scheduler=scheduler,
        )

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[AgentID, Any], Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (observations_dict, info)
        """
        pass

    @abstractmethod
    def step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[AgentID, Dict],
    ]:
        """Execute one step in the environment.

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
        """
        pass

    @abstractmethod
    def get_joint_observation_space(self) -> gym.Space:
        """Get the joint observation space for all agents.

        Returns:
            Gymnasium space representing joint observations
        """
        pass

    @abstractmethod
    def get_joint_action_space(self) -> gym.Space:
        """Get the joint action space for all agents.

        Returns:
            Gymnasium space representing joint actions
        """
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        self.close_heron()

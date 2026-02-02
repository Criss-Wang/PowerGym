"""Abstract environment interfaces for the HERON framework.

This module defines the base environment classes that domain-specific
implementations should extend.

Architecture:
    HeronEnvCore (Mixin) - Core HERON functionality (agent mgmt, event scheduling)
    │
    ├── BaseEnv (gym.Env + HeronEnvCore) - Single-agent Gymnasium interface
    │
    ├── MultiAgentEnv (HeronEnvCore) - Abstract multi-agent base
    │   ├── PettingZooParallelEnv - PettingZoo ParallelEnv interface
    │   └── RLlibMultiAgentEnv - RLlib MultiAgentEnv interface

Execution Modes:
    1. Synchronous (Option A - Training): All agents step together via step()
       - CTDE pattern: centralized training, decentralized execution
       - Coordinator aggregates observations, computes joint action, distributes

    2. Event-driven (Option B - Testing): Priority-queue scheduling via run_event_driven()
       - Each agent ticks independently at its own interval
       - Configurable observation/action/message delays
       - Tests policy robustness to realistic timing constraints
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import uuid

import gymnasium as gym

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.messaging.base import MessageBroker, ChannelManager, Message, MessageType
from heron.utils.typing import AgentID

if TYPE_CHECKING:
    from heron.scheduling import EventScheduler, Event
    from heron.agents.system_agent import SystemAgent
    from heron.agents.proxy_agent import ProxyAgent


class HeronEnvCore:
    """Core mixin providing HERON-specific functionality.

    This mixin provides agent management, event-driven execution, and message-based
    communication support. It does NOT inherit from any environment interface,
    allowing composition with different backends (Gymnasium, PettingZoo, RLlib, etc.).

    Note: Internal attributes use underscore prefix (_heron_agents, _heron_coordinators)
    to avoid conflicts with framework-specific properties like PettingZoo's `agents`.
    Access via heron_agents/heron_coordinators properties or get_heron_agent() method.

    Execution Modes:
        - Option A (Training): Synchronous step() with direct method calls
        - Option B (Testing): Event-driven via EventScheduler with timing delays

    Messaging:
        Message broker is always available (defaults to InMemoryBroker).
        Agents can use send_message/receive_messages for pub/sub communication.

    Attributes:
        env_id: Unique environment identifier
        scheduler: Optional EventScheduler for event-driven execution
        message_broker: MessageBroker for agent communication (always available)
        heron_agents: Dictionary mapping agent IDs to Agent instances
        heron_coordinators: Dictionary mapping coordinator IDs to CoordinatorAgent instances
    """

    def _init_heron_core(
        self,
        env_id: Optional[str] = None,
        scheduler: Optional["EventScheduler"] = None,
        message_broker: Optional["MessageBroker"] = None,
    ) -> None:
        """Initialize HERON core functionality.

        Call this in your __init__ after calling super().__init__().

        Args:
            env_id: Environment identifier (auto-generated if not provided)
            scheduler: Optional EventScheduler for event-driven execution
            message_broker: Optional MessageBroker (defaults to InMemoryBroker)
        """
        self.env_id = env_id or f"env_{uuid.uuid4().hex[:8]}"
        self.scheduler = scheduler
        self._timestep = 0

        # Message broker (always available, defaults to InMemoryBroker)
        if message_broker is None:
            from heron.messaging.in_memory_broker import InMemoryBroker
            self.message_broker = InMemoryBroker()
        else:
            self.message_broker = message_broker

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

    def configure_agents_for_distributed(self) -> None:
        """Configure all registered agents with the message broker. [Distributed Mode]

        Sets the message broker reference on all agents so they can use
        message-based communication. Call this after registering agents
        and before starting the simulation.

        Raises:
            RuntimeError: If message broker is not configured
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker not configured.")

        for agent in self._heron_agents.values():
            agent.set_message_broker(self.message_broker)
            # Also set env_id if not already set
            if agent.env_id is None:
                agent.env_id = self.env_id

    # ============================================
    # Synchronous Execution (Option A - Training)
    # ============================================

    def get_observations(
        self, global_state: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Observation]:
        """Collect observations from all agents. [Training Only]

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
        """Apply actions to agents. [Training Only]

        Args:
            actions: Dictionary mapping agent IDs to actions
            observations: Optional observations to pass to agents
        """
        for agent_id, action in actions.items():
            if agent_id in self._heron_agents:
                obs = observations.get(agent_id) if observations else None
                self._heron_agents[agent_id].act(obs, upstream_action=action)

    def get_agent_action_spaces(self) -> Dict[AgentID, gym.Space]:
        """Get action spaces for all agents. [Both Modes]

        Returns:
            Dictionary mapping agent IDs to their action spaces
        """
        return {
            agent_id: agent.action_space
            for agent_id, agent in self._heron_agents.items()
            if agent.action_space is not None
        }

    def get_agent_observation_spaces(self) -> Dict[AgentID, gym.Space]:
        """Get observation spaces for all agents. [Both Modes]

        Returns:
            Dictionary mapping agent IDs to their observation spaces
        """
        return {
            agent_id: agent.observation_space
            for agent_id, agent in self._heron_agents.items()
            if agent.observation_space is not None
        }

    def reset_agents(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset all registered agents. [Both Modes]

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        for agent in self._heron_agents.values():
            agent.reset(seed=seed, **kwargs)

    # ============================================
    # Event-Driven Execution (Option B - Testing)
    # ============================================

    def setup_event_driven(
        self,
        scheduler: Optional["EventScheduler"] = None,
    ) -> "EventScheduler":
        """Setup event-driven execution with scheduler. [Testing Only]

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
        """Set event handlers for event-driven execution. [Testing Only]

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

    def setup_default_handlers(
        self,
        global_state_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        on_action_effect: Optional[Callable[[AgentID, Any], None]] = None,
    ) -> None:
        """Setup default event handlers for event-driven execution. [Testing Only]

        This convenience method sets up standard handlers that:
        - AGENT_TICK: Calls agent.tick() with scheduler and current time
        - ACTION_EFFECT: Calls the provided callback to apply actions
        - MESSAGE_DELIVERY: Publishes messages via message broker

        Args:
            global_state_fn: Optional function returning current global state
                            for agent.tick(). If None, passes None to tick().
            on_action_effect: Optional callback(agent_id, action) to apply actions.
                            Override this to implement domain-specific action application.
        """
        if self.scheduler is None:
            raise RuntimeError("Call setup_event_driven() first")

        from heron.scheduling import EventType

        # Create closures that capture self and callbacks
        def agent_tick_handler(event: "Event", scheduler: "EventScheduler") -> None:
            agent = self._heron_agents.get(event.agent_id)
            if agent is not None:
                global_state = global_state_fn() if global_state_fn else None
                # Pass proxy_agent to enable delayed observations (Option B)
                proxy = getattr(self, '_proxy_agent', None)
                agent.tick(scheduler, event.timestamp, global_state, proxy)

        def action_effect_handler(event: "Event", scheduler: "EventScheduler") -> None:
            agent_id = event.agent_id
            action = event.payload.get("action")
            if on_action_effect and action is not None:
                on_action_effect(agent_id, action)

        def message_delivery_handler(event: "Event", scheduler: "EventScheduler") -> None:
            """Deliver message via message broker."""
            recipient_id = event.agent_id
            sender_id = event.payload.get("sender")
            message_content = event.payload.get("message", {})

            # Publish message via message broker
            if self.message_broker is not None and sender_id is not None:
                self.publish_action(
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    action=message_content.get("action"),
                ) if "action" in message_content else self.publish_info(
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    info=message_content,
                )

        self.scheduler.set_handler(EventType.AGENT_TICK, agent_tick_handler)
        self.scheduler.set_handler(EventType.ACTION_EFFECT, action_effect_handler)
        self.scheduler.set_handler(EventType.MESSAGE_DELIVERY, message_delivery_handler)

    def run_event_driven(
        self,
        t_end: float,
        max_events: Optional[int] = None,
    ) -> int:
        """Run event-driven simulation until time limit. [Testing Only]

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
        """Current simulation time (from scheduler or timestep). [Both Modes]"""
        if self.scheduler:
            return self.scheduler.current_time
        return float(self._timestep)

    # ============================================
    # Distributed Mode (Message Broker)
    # ============================================

    def setup_broker_channels(self) -> None:
        """Setup message broker channels for all registered agents. [Distributed Mode]

        Creates action and info channels for each agent based on their hierarchy.
        Should be called after all agents are registered.

        Raises:
            RuntimeError: If message broker is not configured
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker not configured.")

        for agent_id, agent in self._heron_agents.items():
            # Get agent's hierarchy info
            upstream_id = getattr(agent, 'upstream_id', None)
            subordinate_ids = list(getattr(agent, 'subordinates', {}).keys())

            # Get channels for this agent
            channels = ChannelManager.agent_channels(
                agent_id=agent_id,
                upstream_id=upstream_id,
                subordinate_ids=subordinate_ids,
                env_id=self.env_id,
            )

            # Create all channels
            for channel in channels['subscribe'] + channels['publish']:
                self.message_broker.create_channel(channel)

    def publish_action(
        self,
        sender_id: AgentID,
        recipient_id: AgentID,
        action: Any,
    ) -> None:
        """Publish an action from sender to recipient via message broker. [Distributed Mode]

        Args:
            sender_id: ID of the agent sending the action
            recipient_id: ID of the agent receiving the action
            action: Action data to send

        Raises:
            RuntimeError: If message broker is not configured
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker not configured.")

        channel = ChannelManager.action_channel(sender_id, recipient_id, self.env_id)
        msg = Message(
            env_id=self.env_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            timestamp=float(self._timestep),
            message_type=MessageType.ACTION,
            payload={"action": action},
        )
        self.message_broker.publish(channel, msg)

    def publish_info(
        self,
        sender_id: AgentID,
        recipient_id: AgentID,
        info: Dict[str, Any],
    ) -> None:
        """Publish info from sender to recipient via message broker. [Distributed Mode]

        Args:
            sender_id: ID of the agent sending the info
            recipient_id: ID of the agent receiving the info
            info: Information data to send

        Raises:
            RuntimeError: If message broker is not configured
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker not configured.")

        channel = ChannelManager.info_channel(sender_id, recipient_id, self.env_id)
        msg = Message(
            env_id=self.env_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            timestamp=float(self._timestep),
            message_type=MessageType.INFO,
            payload=info,
        )
        self.message_broker.publish(channel, msg)

    def publish_state_update(
        self,
        state: Dict[str, Any],
    ) -> None:
        """Publish state update to the state update channel. [Distributed Mode]

        Used by the environment to broadcast state updates to all interested agents.

        Args:
            state: State data to broadcast

        Raises:
            RuntimeError: If message broker is not configured
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker not configured.")

        channel = ChannelManager.state_update_channel(self.env_id)
        msg = Message(
            env_id=self.env_id,
            sender_id="environment",
            recipient_id="broadcast",
            timestamp=float(self._timestep),
            message_type=MessageType.STATE_UPDATE,
            payload=state,
        )
        self.message_broker.publish(channel, msg)

    def broadcast_to_agents(
        self,
        sender_id: str,
        payload: Dict[str, Any],
        agent_ids: Optional[List[AgentID]] = None,
    ) -> None:
        """Broadcast a message to multiple agents. [Distributed Mode]

        Args:
            sender_id: ID of the broadcasting agent/entity
            payload: Message payload
            agent_ids: List of recipient agent IDs (default: all agents)

        Raises:
            RuntimeError: If message broker is not configured
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker not configured.")

        if agent_ids is None:
            agent_ids = list(self._heron_agents.keys())

        for agent_id in agent_ids:
            channel = ChannelManager.broadcast_channel(sender_id, self.env_id)
            msg = Message(
                env_id=self.env_id,
                sender_id=sender_id,
                recipient_id=agent_id,
                timestamp=float(self._timestep),
                message_type=MessageType.BROADCAST,
                payload=payload,
            )
            self.message_broker.publish(channel, msg)

    def consume_actions_for_agent(
        self,
        agent_id: AgentID,
        upstream_id: Optional[AgentID] = None,
    ) -> List[Message]:
        """Consume action messages for an agent from its upstream. [Distributed Mode]

        Args:
            agent_id: ID of the agent consuming actions
            upstream_id: ID of the upstream agent (auto-detected if not provided)

        Returns:
            List of action messages

        Raises:
            RuntimeError: If message broker is not configured
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker not configured.")

        if upstream_id is None:
            agent = self._heron_agents.get(agent_id)
            if agent:
                upstream_id = getattr(agent, 'upstream_id', None)

        if upstream_id is None:
            return []

        channel = ChannelManager.action_channel(upstream_id, agent_id, self.env_id)
        return self.message_broker.consume(channel, agent_id, self.env_id)

    def consume_info_for_agent(
        self,
        agent_id: AgentID,
        subordinate_ids: Optional[List[AgentID]] = None,
    ) -> Dict[AgentID, List[Message]]:
        """Consume info messages for an agent from its subordinates. [Distributed Mode]

        Args:
            agent_id: ID of the agent consuming info
            subordinate_ids: IDs of subordinates (auto-detected if not provided)

        Returns:
            Dict mapping subordinate IDs to their info messages

        Raises:
            RuntimeError: If message broker is not configured
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker not configured.")

        if subordinate_ids is None:
            agent = self._heron_agents.get(agent_id)
            if agent:
                subordinate_ids = list(getattr(agent, 'subordinates', {}).keys())

        if not subordinate_ids:
            return {}

        result = {}
        for sub_id in subordinate_ids:
            channel = ChannelManager.info_channel(sub_id, agent_id, self.env_id)
            messages = self.message_broker.consume(channel, agent_id, self.env_id)
            if messages:
                result[sub_id] = messages

        return result

    def clear_broker_environment(self) -> None:
        """Clear all messages for this environment from the broker. [Distributed Mode]

        Useful for resetting the environment.
        """
        if self.message_broker is not None:
            self.message_broker.clear_environment(self.env_id)

    def close_heron(self) -> None:
        """Clean up HERON-specific resources. [Both Modes]"""
        if self.message_broker is not None:
            self.message_broker.close()

    # ============================================
    # SystemAgent Integration (Both Modes)
    # ============================================

    def set_system_agent(self, system_agent: "SystemAgent") -> None:
        """Set the SystemAgent for this environment. [Both Modes]

        The SystemAgent serves as the interface between the environment and
        the agent hierarchy. When set, the environment can use:
        - system_agent.update_from_environment() to push state
        - system_agent.get_state_for_environment() to get actions

        Args:
            system_agent: SystemAgent instance to manage agent hierarchy
        """
        self._system_agent = system_agent

        # Register all coordinators from the system agent
        for coord_id, coordinator in system_agent.coordinators.items():
            self.register_agent(coordinator)

        # Configure message broker for system agent
        if self.message_broker:
            system_agent.set_message_broker(self.message_broker)
            system_agent.env_id = self.env_id

    @property
    def system_agent(self) -> Optional["SystemAgent"]:
        """Get the SystemAgent for this environment. [Both Modes]"""
        return getattr(self, '_system_agent', None)

    def set_proxy_agent(self, proxy_agent: "ProxyAgent") -> None:
        """Set the ProxyAgent for state distribution. [Both Modes]

        The ProxyAgent manages state distribution to agents with visibility
        filtering. When set, agents can request state through the proxy
        instead of accessing the environment directly.

        Args:
            proxy_agent: ProxyAgent instance for state distribution
        """
        self._proxy_agent = proxy_agent
        self.register_agent(proxy_agent)

    @property
    def proxy_agent(self) -> Optional["ProxyAgent"]:
        """Get the ProxyAgent for this environment. [Both Modes]"""
        return getattr(self, '_proxy_agent', None)

    def update_proxy_state(self, state: Dict[str, Any]) -> None:
        """Update the ProxyAgent's cached state. [Both Modes]

        Convenience method to update the proxy agent's state cache.
        Should be called after physics/simulation updates.

        Args:
            state: Current environment state to cache

        Raises:
            RuntimeError: If no proxy agent is configured
        """
        if self._proxy_agent is None:
            raise RuntimeError("No proxy agent configured. Call set_proxy_agent() first.")
        self._proxy_agent.update_state(state)

    def step_with_system_agent(
        self,
        actions: Dict[AgentID, Any],
        global_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Execute step using SystemAgent pattern. [Training - Option A]

        This convenience method implements the standard CTDE flow:
        1. SystemAgent observes (aggregates from coordinators)
        2. SystemAgent acts (distributes actions to coordinators)
        3. Environment applies physics/simulation
        4. SystemAgent receives updated state

        Args:
            actions: Dictionary mapping coordinator IDs to actions
            global_state: Optional global state for observation

        Raises:
            RuntimeError: If no system agent is configured
        """
        if self._system_agent is None:
            raise RuntimeError("No system agent configured. Call set_system_agent() first.")

        # 1. Observe
        observation = self._system_agent.observe(global_state)

        # 2. Act (distribute actions to coordinators)
        self._system_agent.act(observation, upstream_action=actions)

    def run_event_driven_with_system_agent(
        self,
        t_end: float,
        get_global_state: Optional[Callable[[], Dict[str, Any]]] = None,
        on_action_effect: Optional[Callable[[AgentID, Any], None]] = None,
        max_events: Optional[int] = None,
    ) -> int:
        """Run event-driven simulation with SystemAgent. [Testing - Option B]

        This convenience method sets up and runs event-driven execution
        with the SystemAgent hierarchy. It:
        1. Sets up the scheduler and registers all agents
        2. Configures default event handlers
        3. Runs the simulation until t_end

        Args:
            t_end: Stop when simulation time exceeds this
            get_global_state: Optional function returning current global state
            on_action_effect: Optional callback(agent_id, action) for actions
            max_events: Optional maximum number of events to process

        Returns:
            Number of events processed

        Raises:
            RuntimeError: If no system agent is configured
        """
        if self._system_agent is None:
            raise RuntimeError("No system agent configured. Call set_system_agent() first.")

        # Setup scheduler if not already done
        if self.scheduler is None:
            self.setup_event_driven()

        # Register system agent with scheduler
        self.scheduler.register_agent(
            agent_id=self._system_agent.agent_id,
            tick_interval=self._system_agent.tick_interval,
            obs_delay=self._system_agent.obs_delay,
            act_delay=self._system_agent.act_delay,
        )

        # Setup default handlers
        self.setup_default_handlers(
            global_state_fn=get_global_state,
            on_action_effect=on_action_effect,
        )

        # Run simulation
        return self.run_event_driven(t_end=t_end, max_events=max_events)


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
    ):
        """Initialize base environment.

        Args:
            env_id: Environment identifier (auto-generated if not provided)
        """
        gym.Env.__init__(self)
        self._init_heron_core(env_id=env_id)

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

    Execution Modes:
        - Option A (Training): Synchronous step() with CTDE pattern
        - Option B (Testing): Event-driven via EventScheduler with timing delays

    Subclasses must implement:
        - reset()
        - step()
        - get_joint_observation_space()
        - get_joint_action_space()

    Attributes:
        system_agent: Optional SystemAgent for hierarchical agent management
        proxy_agent: Optional ProxyAgent for state distribution
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        scheduler: Optional["EventScheduler"] = None,
        message_broker: Optional[MessageBroker] = None,
    ):
        """Initialize multi-agent environment.

        Args:
            env_id: Environment identifier
            scheduler: Optional EventScheduler for event-driven mode
            message_broker: Optional MessageBroker (defaults to InMemoryBroker)
        """
        self._init_heron_core(
            env_id=env_id,
            scheduler=scheduler,
            message_broker=message_broker,
        )

        # SystemAgent integration (set via set_system_agent)
        self._system_agent: Optional["SystemAgent"] = None
        self._proxy_agent: Optional["ProxyAgent"] = None

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

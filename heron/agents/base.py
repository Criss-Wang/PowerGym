"""Base agent abstraction for hierarchical multi-agent control.

This module provides the core abstractions for agents in the HERON framework,
supporting hierarchical control, agent-to-agent communication, and flexible
observation/action interfaces.

Execution Modes:
    1. Synchronous (Option A - Training): observe() -> act() with CTDE pattern
       - Coordinator collects observations from subordinates
       - Centralized policy computes joint action
       - Coordinator distributes actions to subordinates
       - All operations are synchronous within env.step()

    2. Event-Driven (Option B - Testing): EventScheduler with independent agent ticks
       - Each agent ticks independently at its own interval (tick_interval)
       - Different hierarchy levels can have different tick rates
       - Observations/actions have configurable delays (obs_delay, act_delay)
       - Hierarchical communication via async messages with msg_delay:
         * Subordinates send info to coordinators (may arrive at different times)
         * Coordinators send actions to subordinates (arrive after msg_delay)
       - Coordinators act on whatever info has arrived - no waiting for all subordinates
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import gymnasium as gym

from heron.core.action import Action
from heron.core.observation import Observation
from heron.messaging.base import ChannelManager, Message as BrokerMessage, MessageType
from heron.utils.typing import AgentID

from heron.messaging.base import MessageBroker
from heron.scheduling.tick_config import TickConfig, JitterType
from heron.scheduling.scheduler import EventScheduler


class Agent(ABC):
    """Abstract base class for all agents in the hierarchy.

    Agents can operate at different levels of the control hierarchy:
    - Field level (L1): Individual units, sensors, actuators
    - Coordinator level (L2): Regional controllers, aggregators
    - System level (L3): Central controller, market operator

    Attributes:
        agent_id: Unique identifier for this agent
        level: Hierarchy level (1=field, 2=coordinator, 3=system)
        observation_space: Gymnasium space for observations
        action_space: Gymnasium space for actions
        upstream_id: Optional upstream agent ID for hierarchy structure
        subordinates: Dict of subordinate agents for hierarchy structure
        env_id: Environment ID for multi-environment isolation
        tick_config: Timing configuration for event-driven scheduling (tick_interval,
            obs_delay, act_delay, msg_delay, jitter settings)
    """

    def __init__(
        self,
        agent_id: AgentID,
        level: int = 1,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        subordinates: Optional[Dict[AgentID, "Agent"]] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
    ):
        """Initialize agent.

        Args:
            agent_id: Unique identifier
            level: Hierarchy level (1=field, 2=coordinator, 3=system)
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            upstream_id: Optional upstream agent ID for hierarchy structure
            subordinates: Optional dict of subordinate agents
            env_id: Optional environment ID for multi-environment isolation
            tick_config: Timing configuration (defaults to deterministic with
                tick_interval=1.0, no delays). Use TickConfig.deterministic() or
                TickConfig.with_jitter() to create custom configs.
        """
        self.agent_id = agent_id
        self.level = level
        self.observation_space = observation_space
        self.action_space = action_space

        # Execution state
        self._timestep: float = 0.0
        self._last_observation: Optional[Observation] = None  # Cached obs for tick()

        # Message broker reference (set by environment in distributed mode)
        self._message_broker: Optional[MessageBroker] = None

        # Timing configuration (via TickConfig)
        self._tick_config = tick_config or TickConfig.deterministic()

        # Hierarchy structure (used by coordinators)
        self.upstream_id = upstream_id
        self.subordinates = subordinates or {}
        self.env_id = env_id
        self.subordinates_info: Dict[AgentID, Dict[str, Any]] = {}

    # ============================================
    # Tick Configuration (Event-Driven Mode)
    # ============================================

    @property
    def tick_config(self) -> TickConfig:
        """Get the tick configuration for this agent."""
        return self._tick_config

    @tick_config.setter
    def tick_config(self, config: TickConfig) -> None:
        """Set the tick configuration for this agent."""
        self._tick_config = config

    def enable_jitter(
        self,
        jitter_type: Optional[JitterType] = None,
        jitter_ratio: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """Enable jitter for testing mode.

        Converts current tick_config to use jitter with same base values.

        Args:
            jitter_type: Distribution type for jitter (default: GAUSSIAN)
            jitter_ratio: Jitter magnitude as fraction of base
            seed: Optional RNG seed for reproducibility
        """
        if jitter_type is None:
            jitter_type = JitterType.GAUSSIAN

        self._tick_config = TickConfig.with_jitter(
            tick_interval=self._tick_config.tick_interval,
            obs_delay=self._tick_config.obs_delay,
            act_delay=self._tick_config.act_delay,
            msg_delay=self._tick_config.msg_delay,
            jitter_type=jitter_type,
            jitter_ratio=jitter_ratio,
            seed=seed,
        )

    def disable_jitter(self) -> None:
        """Disable jitter (switch to deterministic mode)."""
        self._tick_config = TickConfig.deterministic(
            tick_interval=self._tick_config.tick_interval,
            obs_delay=self._tick_config.obs_delay,
            act_delay=self._tick_config.act_delay,
            msg_delay=self._tick_config.msg_delay,
        )

    # ============================================
    # Core Lifecycle Methods (Both Modes)
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent to initial state. [Both Modes]

        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional reset parameters
        """
        self._timestep = 0.0
        self._last_observation = None

    # ============================================
    # Synchronous Execution (Option A - Training)
    # ============================================

    @abstractmethod
    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
        """Extract relevant observations from global state. [Both Modes]

        - Training (Option A): Called directly by coordinator to collect observations
        - Testing (Option B): Called internally by tick() to get current observation

        Args:
            global_state: Complete environment state

        Returns:
            Structured observation for this agent
        """
        pass

    @abstractmethod
    def act(self, observation: Observation, *args, **kwargs) -> Optional[Action]:
        """Compute action from observation. [Both Modes]

        - Training (Option A): Called directly by coordinator to apply actions
        - Testing (Option B): Called internally by tick() to compute action

        Args:
            observation: Structured observation from observe()

        Returns:
            Action object, or None if action is stored internally / agent doesn't act
        """
        pass

    def _update_action_features(self, action: Any, observation: Observation) -> None:
        """Update features that depend on the action taken. [Phase 1 - Both Modes]

        Called immediately after action is computed/set in act() and tick().
        Override in subclasses to update action-dependent features that should
        be included when environment collects agent states via get_state_for_environment().

        This enables the two-phase update flow:
        1. Phase 1: Agent takes action → _update_action_features() updates features
           → Environment collects states (with updated features) → runs simulation
        2. Phase 2: Environment returns results → update_from_environment() updates
           result-dependent features

        Args:
            action: The action just taken (may be None if no action)
            observation: The observation used to compute the action

        Example:
            def _update_action_features(self, action, observation):
                if action is not None:
                    # Update features based on action
                    self.state.features.power_setpoint = action[0]
                    self.state.features.coordination_signal = action[1]
        """
        pass  # Override in subclasses

    # ============================================
    # Event-Driven Execution (Option B - Testing)
    # ============================================

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
        global_state: Optional[Dict[str, Any]] = None,
        proxy: Optional["Agent"] = None,
    ) -> None:
        """Execute one tick in event-driven mode. [Testing Only]

        Called by EventScheduler when AGENT_TICK event fires.
        Override in subclasses (FieldAgent, HierarchicalAgent) to implement
        the agent-specific tick behavior.

        Args:
            scheduler: EventScheduler instance for scheduling future events
            current_time: Current simulation time
            global_state: Optional global state for observation
            proxy: Optional ProxyAgent for delayed observations
        """
        pass  # Override in subclasses

    # ============================================
    # Messaging via Message Broker (Both Modes)
    # ============================================

    def set_message_broker(self, broker: MessageBroker) -> None:
        """Set the message broker for this agent. [Both Modes]

        Called by the environment to configure distributed messaging.

        Args:
            broker: MessageBroker instance
        """
        self._message_broker = broker

    @property
    def message_broker(self) -> Optional[MessageBroker]:
        """Get the message broker for this agent. [Both Modes]"""
        return self._message_broker

    def send_message(
        self,
        content: Dict[str, Any],
        recipient_id: str,
        message_type: str = "INFO",
    ) -> None:
        """Send a message to another agent via message broker. [Both Modes]

        Args:
            content: Message payload
            recipient_id: Target agent ID
            message_type: Type of message (ACTION, INFO, BROADCAST, etc.)

        Raises:
            RuntimeError: If no message broker is configured
        """
        if self._message_broker is None:
            raise RuntimeError(
                f"Agent {self.agent_id} has no message broker configured. "
                "Call set_message_broker() first."
            )

        channel = ChannelManager.info_channel(
            self.agent_id, recipient_id, self.env_id or "default"
        )
        self.publish_to_broker(
            broker=self._message_broker,
            channel=channel,
            payload=content,
            recipient_id=recipient_id,
            message_type=message_type,
        )

    def receive_messages(
        self,
        sender_id: Optional[str] = None,
        clear: bool = True,
    ) -> List[Dict[str, Any]]:
        """Receive messages from the message broker. [Both Modes]

        Args:
            sender_id: Optional sender ID to filter messages from (uses upstream_id if not provided)
            clear: If True, remove consumed messages

        Returns:
            List of message payloads

        Raises:
            RuntimeError: If no message broker is configured
        """
        if self._message_broker is None:
            return []  # No broker, no messages

        if sender_id is None:
            sender_id = self.upstream_id
        if sender_id is None:
            return []

        channel = ChannelManager.info_channel(
            sender_id, self.agent_id, self.env_id or "default"
        )
        messages = self.consume_from_broker(self._message_broker, channel, clear=clear)
        return [msg.payload for msg in messages]

    def receive_action_messages(
        self,
        sender_id: Optional[str] = None,
        clear: bool = True,
    ) -> List[Any]:
        """Receive action messages from the message broker. [Both Modes]

        Convenience method for receiving action messages from upstream.

        Args:
            sender_id: Optional sender ID (uses upstream_id if not provided)
            clear: If True, remove consumed messages

        Returns:
            List of actions received
        """
        if self._message_broker is None:
            return []

        return self.receive_actions_from_broker(
            broker=self._message_broker,
            upstream_id=sender_id,
            clear=clear,
        )

    def send_action_to_subordinate(
        self,
        recipient_id: str,
        action: Any,
    ) -> None:
        """Send an action to a subordinate agent. [Both Modes]

        Convenience method for sending actions to subordinates.

        Args:
            recipient_id: ID of the subordinate agent
            action: Action to send
        """
        if self._message_broker is None:
            raise RuntimeError(
                f"Agent {self.agent_id} has no message broker configured."
            )

        self.send_action_via_broker(
            broker=self._message_broker,
            recipient_id=recipient_id,
            action=action,
        )

    # ============================================
    # Utility Methods (Both Modes)
    # ============================================

    def update_timestep(self, timestep: float) -> None:
        """Update the internal timestep. [Both Modes]

        Args:
            timestep: New timestep value
        """
        self._timestep = timestep

    def request_state_from_proxy(
        self,
        proxy: "Agent",
        owner_id: Optional[AgentID] = None,
        at_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Request filtered state from ProxyAgent. [Both Modes]

        This is the recommended pattern for agents to access state in
        information-constrained environments. The ProxyAgent applies
        visibility rules before returning state.

        In Option B (Testing), use at_time to get delayed observations:
            state = self.request_state_from_proxy(
                proxy, at_time=current_time - self.tick_config.obs_delay
            )

        Args:
            proxy: ProxyAgent instance to request state from
            owner_id: ID of agent whose state to request (defaults to self)
            at_time: Optional timestamp for delayed observations (Option B)

        Returns:
            Filtered state dict based on visibility rules

        Example:
            # In agent's observe() or act() method:
            state = self.request_state_from_proxy(self.proxy_agent)

            # With observation delay (Option B):
            state = self.request_state_from_proxy(
                self.proxy_agent, at_time=current_time - self.tick_config.obs_delay
            )
        """
        # Import here to avoid circular dependency
        from heron.agents.proxy_agent import ProxyAgent

        if not isinstance(proxy, ProxyAgent):
            raise TypeError(f"Expected ProxyAgent, got {type(proxy).__name__}")

        if owner_id is None:
            owner_id = self.agent_id

        return proxy.get_state_for_agent(
            agent_id=self.agent_id,
            requestor_level=self.level,
            owner_id=owner_id,
            at_time=at_time,
        )

    # ============================================
    # Distributed Mode (Message Broker)
    # ============================================

    def publish_to_broker(
        self,
        broker: MessageBroker,
        channel: str,
        payload: Dict[str, Any],
        recipient_id: str = "broadcast",
        message_type: str = "INFO",
    ) -> None:
        """Publish a message via the message broker. [Distributed Mode]

        Args:
            broker: MessageBroker instance
            channel: Channel name to publish to
            payload: Message payload
            recipient_id: Recipient agent ID (default: broadcast)
            message_type: Type of message (ACTION, INFO, BROADCAST, etc.)
        """
        msg = BrokerMessage(
            env_id=self.env_id or "default",
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            timestamp=self._timestep,
            message_type=MessageType[message_type],
            payload=payload,
        )
        broker.publish(channel, msg)

    def consume_from_broker(
        self,
        broker: MessageBroker,
        channel: str,
        clear: bool = True,
    ) -> List[BrokerMessage]:
        """Consume messages from the message broker. [Distributed Mode]

        Args:
            broker: MessageBroker instance
            channel: Channel name to consume from
            clear: If True, remove consumed messages

        Returns:
            List of messages for this agent
        """
        return broker.consume(
            channel=channel,
            recipient_id=self.agent_id,
            env_id=self.env_id or "default",
            clear=clear,
        )

    def send_action_via_broker(
        self,
        broker: MessageBroker,
        recipient_id: str,
        action: Any,
    ) -> None:
        """Send an action to a subordinate via the message broker. [Distributed Mode]

        Args:
            broker: MessageBroker instance
            recipient_id: ID of the recipient agent
            action: Action to send
        """
        channel = ChannelManager.action_channel(
            self.agent_id, recipient_id, self.env_id or "default"
        )
        self.publish_to_broker(
            broker=broker,
            channel=channel,
            payload={"action": action},
            recipient_id=recipient_id,
            message_type="ACTION",
        )

    def send_info_via_broker(
        self,
        broker: MessageBroker,
        recipient_id: str,
        info: Dict[str, Any],
    ) -> None:
        """Send info to an upstream agent via the message broker. [Distributed Mode]

        Args:
            broker: MessageBroker instance
            recipient_id: ID of the recipient agent (typically upstream)
            info: Information payload
        """
        channel = ChannelManager.info_channel(
            self.agent_id, recipient_id, self.env_id or "default"
        )
        self.publish_to_broker(
            broker=broker,
            channel=channel,
            payload=info,
            recipient_id=recipient_id,
            message_type="INFO",
        )

    def receive_actions_from_broker(
        self,
        broker: MessageBroker,
        upstream_id: Optional[str] = None,
        clear: bool = True,
    ) -> List[Any]:
        """Receive actions from upstream via the message broker. [Distributed Mode]

        Args:
            broker: MessageBroker instance
            upstream_id: ID of the upstream agent (uses self.upstream_id if not provided)
            clear: If True, remove consumed messages

        Returns:
            List of actions received
        """
        if upstream_id is None:
            upstream_id = self.upstream_id
        if upstream_id is None:
            return []

        channel = ChannelManager.action_channel(
            upstream_id, self.agent_id, self.env_id or "default"
        )
        messages = self.consume_from_broker(broker, channel, clear=clear)
        return [msg.payload.get("action") for msg in messages if "action" in msg.payload]

    def receive_info_from_broker(
        self,
        broker: MessageBroker,
        subordinate_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Receive info from subordinates via the message broker. [Distributed Mode]

        Args:
            broker: MessageBroker instance
            subordinate_ids: IDs of subordinate agents (uses self.subordinates if not provided)

        Returns:
            Dict mapping subordinate IDs to their info payloads
        """
        if subordinate_ids is None:
            subordinate_ids = list(self.subordinates.keys())

        result = {}
        for sub_id in subordinate_ids:
            channel = ChannelManager.info_channel(
                sub_id, self.agent_id, self.env_id or "default"
            )
            messages = self.consume_from_broker(broker, channel)
            if messages:
                result[sub_id] = [msg.payload for msg in messages]

        return result

    # ============================================
    # Async Observation Methods (Option B - Fully Async)
    # ============================================

    def send_observation_to_upstream(
        self,
        observation: Observation,
        scheduler: Optional["EventScheduler"] = None,
    ) -> None:
        """Send observation to upstream agent via message broker. [Testing Only - Async Mode]

        Used in fully async event-driven mode where subordinates push observations
        to coordinators instead of coordinators pulling them via direct method calls.

        **Tricky Part - Timing**:
        Observations are sent with msg_delay, so they arrive at the coordinator
        after some latency. If the coordinator ticks before the observation arrives,
        it will use stale/cached data. This is realistic but may surprise users.

        **Tricky Part - Message Delivery vs Direct Send**:
        If scheduler is provided, observation is scheduled as MESSAGE_DELIVERY event
        with msg_delay. Otherwise, it's published directly to the broker (no delay).

        Args:
            observation: Observation to send
            scheduler: Optional EventScheduler for delayed delivery
        """
        if self._message_broker is None or self.upstream_id is None:
            return

        payload = {"observation": observation.to_dict()}

        if scheduler is not None and self._tick_config.msg_delay > 0:
            # Schedule delayed delivery
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=self.upstream_id,
                message=payload,
                delay=self._tick_config.msg_delay,
            )
        else:
            # Direct publish (no delay)
            channel = ChannelManager.observation_channel(
                self.agent_id, self.upstream_id, self.env_id or "default"
            )
            self.publish_to_broker(
                broker=self._message_broker,
                channel=channel,
                payload=payload,
                recipient_id=self.upstream_id,
                message_type="INFO",
            )

    def receive_observations_from_subordinates(
        self,
        subordinate_ids: Optional[List[str]] = None,
        clear: bool = True,
    ) -> Dict[str, Observation]:
        """Receive observations from subordinates via message broker. [Testing Only - Async Mode]

        Returns whatever observations have arrived from subordinates. May be partial
        (some subordinates haven't sent yet) or stale (old observations).

        **Tricky Part - Partial Information**:
        Unlike synchronous mode where coordinator waits for all subordinates,
        async mode returns only what has arrived. The coordinator must handle:
        - Missing subordinates (use cached observation or skip)
        - Stale observations (subordinate sent before its latest tick)
        - Out-of-order delivery (older observation arrives after newer one)

        **Tricky Part - Observation Space Mismatch**:
        If the policy expects a fixed observation shape but some subordinates
        haven't reported, the observation vector will be different. Options:
        1. Pad missing observations with zeros (requires knowing expected shape)
        2. Use cached observations for missing subordinates
        3. Train policy to handle variable-length observations

        **Follow-up Work Needed**:
        - Add observation caching with timestamps to detect staleness
        - Add padding option for fixed-shape policy compatibility
        - Add metrics for observation freshness/completeness

        Args:
            subordinate_ids: IDs of subordinates to receive from (defaults to all)
            clear: If True, remove consumed messages from broker

        Returns:
            Dict mapping subordinate IDs to their Observations (only those received)
        """
        if self._message_broker is None:
            return {}

        if subordinate_ids is None:
            subordinate_ids = list(self.subordinates.keys())

        result = {}
        for sub_id in subordinate_ids:
            channel = ChannelManager.observation_channel(
                sub_id, self.agent_id, self.env_id or "default"
            )
            messages = self.consume_from_broker(self._message_broker, channel, clear=clear)
            if messages:
                # Use most recent observation (last message)
                obs_dict = messages[-1].payload.get("observation", {})
                result[sub_id] = Observation.from_dict(obs_dict)

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, level={self.level})"

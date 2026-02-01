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
       - Each agent ticks at its own interval (tick_interval)
       - Observations/actions have configurable delays (obs_delay, act_delay)
       - Messages between agents have delays (msg_delay)
       - No hierarchical waiting - agents act on potentially stale information
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import gymnasium as gym

from heron.core.action import Action
from heron.core.observation import Observation
from heron.messaging.base import ChannelManager, Message as BrokerMessage, MessageType
from heron.utils.typing import AgentID

if TYPE_CHECKING:
    from heron.messaging.base import MessageBroker
    from heron.scheduling.tick_config import TickConfig, JitterType


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
        tick_interval: Time between agent steps (for event-driven scheduling)
        obs_delay: Observation delay - agent sees state from t - obs_delay
        act_delay: Action delay - action takes effect at t + act_delay
        msg_delay: Message delay - messages arrive after msg_delay
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

        # timing/latency params (for event-driven scheduling)
        tick_interval: float = 1.0,
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,

        # NEW: TickConfig for full control (overrides individual timing params)
        tick_config: Optional["TickConfig"] = None,
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
            tick_interval: Time between agent steps (ignored if tick_config provided)
            obs_delay: Observation delay (ignored if tick_config provided)
            act_delay: Action delay (ignored if tick_config provided)
            msg_delay: Message delay (ignored if tick_config provided)
            tick_config: Optional TickConfig for full control including jitter
        """
        self.agent_id = agent_id
        self.level = level
        self.observation_space = observation_space
        self.action_space = action_space

        # Execution state
        self._timestep: float = 0.0
        self._last_observation: Optional[Observation] = None  # Cached obs for tick()

        # Message broker reference (set by environment in distributed mode)
        self._message_broker: Optional["MessageBroker"] = None

        # Timing configuration (via TickConfig)
        if tick_config is not None:
            self._tick_config = tick_config
        else:
            # Create deterministic config from legacy params
            from heron.scheduling.tick_config import TickConfig as TC

            self._tick_config = TC.deterministic(
                tick_interval=tick_interval,
                obs_delay=obs_delay,
                act_delay=act_delay,
                msg_delay=msg_delay,
            )

        # Hierarchy structure (used by coordinators)
        self.upstream_id = upstream_id
        self.subordinates = subordinates or {}
        self.env_id = env_id
        self.subordinates_info: Dict[AgentID, Dict[str, Any]] = {}

    # ============================================
    # Tick Configuration (Event-Driven Mode)
    # ============================================

    @property
    def tick_config(self) -> "TickConfig":
        """Get the tick configuration for this agent."""
        return self._tick_config

    @tick_config.setter
    def tick_config(self, config: "TickConfig") -> None:
        """Set the tick configuration for this agent."""
        self._tick_config = config

    # Backward-compatible properties (read base values from config)
    @property
    def tick_interval(self) -> float:
        """Base tick interval (use tick_config.get_tick_interval() for jittered)."""
        return self._tick_config.tick_interval

    @property
    def obs_delay(self) -> float:
        """Base observation delay (use tick_config.get_obs_delay() for jittered)."""
        return self._tick_config.obs_delay

    @property
    def act_delay(self) -> float:
        """Base action delay (use tick_config.get_act_delay() for jittered)."""
        return self._tick_config.act_delay

    @property
    def msg_delay(self) -> float:
        """Base message delay (use tick_config.get_msg_delay() for jittered)."""
        return self._tick_config.msg_delay

    def enable_jitter(
        self,
        jitter_type: "JitterType" = None,
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
        from heron.scheduling.tick_config import TickConfig as TC, JitterType

        if jitter_type is None:
            jitter_type = JitterType.GAUSSIAN

        self._tick_config = TC.with_jitter(
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
        from heron.scheduling.tick_config import TickConfig as TC

        self._tick_config = TC.deterministic(
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

    # ============================================
    # Event-Driven Execution (Option B - Testing)
    # ============================================

    def tick(
        self,
        scheduler: "EventScheduler",
        current_time: float,
        global_state: Optional[Dict[str, Any]] = None,
        proxy: Optional["Agent"] = None,
    ) -> None:
        """Execute one tick in event-driven mode. [Testing Only]

        Called by EventScheduler when AGENT_TICK event fires. The default
        implementation provides a simple observe-act cycle. Override for
        custom behavior.

        Default behavior:
        1. Update internal timestep
        2. Get observation (using cached or fresh)
        3. Compute action via act()
        4. Schedule ACTION_EFFECT event with act_delay

        Args:
            scheduler: EventScheduler instance for scheduling future events
            current_time: Current simulation time
            global_state: Optional global state for observation
            proxy: Optional ProxyAgent for delayed observations
        """
        self._timestep = current_time

        # Get observation (may use delayed state via ProxyAgent in subclasses)
        observation = self.observe(global_state)
        self._last_observation = observation

        # Compute action
        self.act(observation)

        # Schedule delayed action effect if act_delay > 0
        if self.act_delay > 0 and hasattr(self, 'action'):
            scheduler.schedule_action_effect(
                agent_id=self.agent_id,
                action=getattr(self, 'action', None),
                delay=self.act_delay,
            )

    # ============================================
    # Messaging via Message Broker (Both Modes)
    # ============================================

    def set_message_broker(self, broker: "MessageBroker") -> None:
        """Set the message broker for this agent. [Both Modes]

        Called by the environment to configure distributed messaging.

        Args:
            broker: MessageBroker instance
        """
        self._message_broker = broker

    @property
    def message_broker(self) -> Optional["MessageBroker"]:
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
                proxy, at_time=current_time - self.obs_delay
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
                self.proxy_agent, at_time=current_time - self.obs_delay
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
        broker: "MessageBroker",
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
        broker: "MessageBroker",
        channel: str,
        clear: bool = True,
    ) -> List["BrokerMessage"]:
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
        broker: "MessageBroker",
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
        broker: "MessageBroker",
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
        broker: "MessageBroker",
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
        broker: "MessageBroker",
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, level={self.level})"

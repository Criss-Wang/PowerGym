"""Proxy Agent for managing state distribution in MARL.

The ProxyAgent acts as an intermediary between the environment and other agents,
managing state updates and controlling information visibility.

Key responsibilities:
1. Cache state from the environment
2. Apply visibility rules to filter state for each agent
3. Provide state on-demand to requesting agents (with optional time delay)

Usage Pattern (Option A - Synchronous):
    proxy = ProxyAgent(
        env_id="env_1",
        registered_agents=["sensor_1", "controller_1"],
        visibility_rules={
            "sensor_1": ["reading", "status"],
            "controller_1": ["measurement"],
        }
    )

    # In env.step():
    proxy.update_state(env_state)  # Cache latest state

    # Agents request state through proxy
    state = proxy.get_state_for_agent("sensor_1", requestor_level=1)

Usage Pattern (Option B - Event-Driven with Message Broker):
    from heron.messaging.base import InMemoryBroker

    broker = InMemoryBroker()
    proxy = ProxyAgent(
        env_id="env_1",
        registered_agents=["grid_1", "grid_2"],
        message_broker=broker,
        result_channel_type="power_flow",  # Custom channel type
    )

    # Environment publishes state to proxy via broker
    # proxy.receive_state_from_environment()  # Pull from broker
    # proxy.distribute_state_to_agents()      # Push to agents
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.messaging.base import ChannelManager, Message, MessageBroker, MessageType
from heron.utils.typing import AgentID

if TYPE_CHECKING:
    from heron.core.state import State


PROXY_LEVEL = 0  # Proxy is not part of the agent hierarchy (L1-L3)


class ProxyAgent(Agent):
    """Proxy agent that manages state distribution.

    The ProxyAgent sits between the environment and other agents, acting as the
    single source of truth for state information. All agents should retrieve
    state through the ProxyAgent rather than directly accessing the environment.

    This enables:
    - Visibility filtering based on agent level and configuration
    - Historical state access for observation delays (Option B)
    - Centralized state management

    Attributes:
        state_cache: Latest state received from environment
        state_history: Historical states for delayed observations (Option B)
        visibility_rules: Dict mapping agent IDs to allowed state keys
        registered_agents: List of agent IDs that can request state from this proxy
    """

    def __init__(
        self,
        agent_id: AgentID = "proxy_agent",
        env_id: Optional[str] = None,
        registered_agents: Optional[List[AgentID]] = None,
        visibility_rules: Optional[Dict[AgentID, List[str]]] = None,
        history_length: int = 100,
        message_broker: Optional[MessageBroker] = None,
        result_channel_type: str = "result",
    ):
        """Initialize proxy agent.

        Args:
            agent_id: Unique identifier for this proxy agent
            env_id: Environment ID for multi-environment isolation
            registered_agents: List of agent IDs that can request state
            visibility_rules: Dict mapping agent IDs to allowed state keys.
                If None, all agents see all state by default.
            history_length: Number of timesteps of history to maintain
            message_broker: Optional message broker for distributed communication.
                If provided, enables receive_state_from_environment() and
                distribute_state_to_agents() methods.
            result_channel_type: Channel type for receiving state from environment.
                Default is "result". Domain-specific implementations may use
                custom types like "power_flow".
        """
        super().__init__(
            agent_id=agent_id,
            level=PROXY_LEVEL,
            upstream_id=None,
            env_id=env_id,
            subordinates={},
        )

        self.registered_agents: List[AgentID] = registered_agents or []
        self.visibility_rules: Dict[AgentID, List[str]] = visibility_rules or {}
        self.state_cache: Dict[str, Any] = {}
        self.state_history: List[Dict[str, Any]] = []
        self.history_length = history_length

        # Message broker support (Option B - Event-Driven)
        self._result_channel_type = result_channel_type
        if message_broker is not None:
            self.set_message_broker(message_broker)
            self._setup_channels()

    # ============================================
    # State Management
    # ============================================

    def update_state(self, state: Dict[str, Any]) -> None:
        """Update cached state from environment.

        Should be called by the environment after each step.

        Args:
            state: Current environment state
        """
        self.state_cache = state.copy()

        # Add to history for delayed observations
        self.state_history.append({
            'timestamp': self._timestep,
            'state': state.copy()
        })

        # Trim history if exceeds limit
        if len(self.state_history) > self.history_length:
            self.state_history = self.state_history[-self.history_length:]

    def get_state_at_time(self, target_time: float) -> Dict[str, Any]:
        """Get state from a specific time for delayed observations.

        Used in Option B to simulate observation delays.

        Args:
            target_time: Timestamp to retrieve state for

        Returns:
            State at or before target_time, or current cache if not available
        """
        if not self.state_history:
            return self.state_cache

        # Find the most recent state at or before target_time
        for entry in reversed(self.state_history):
            if entry['timestamp'] <= target_time:
                return entry['state']

        # If target_time is before all history, return oldest
        return self.state_history[0]['state'] if self.state_history else self.state_cache

    # ============================================
    # State Filtering
    # ============================================
    #
    # Two visibility mechanisms are supported:
    # 1. Key-based: visibility_rules dict maps agent_id -> allowed state keys
    # 2. FeatureProvider-based: State.features with is_observable_by() for level-based rules
    #
    # Use key-based for simple scenarios. Use FeatureProvider for hierarchy-aware visibility.

    def _filter_state_by_keys(
        self,
        agent_id: AgentID,
        agent_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Filter state based on visibility rules for a specific agent.

        Args:
            agent_id: Agent requesting the state
            agent_state: Agent-specific state to filter

        Returns:
            Filtered state dict containing only allowed keys
        """
        if agent_id not in self.visibility_rules:
            # No specific rules, return full agent state
            return agent_state.copy() if agent_state else {}

        allowed_keys = self.visibility_rules[agent_id]
        return {key: agent_state[key] for key in allowed_keys if key in agent_state}

    def get_state_for_agent(
        self,
        agent_id: AgentID,
        requestor_level: int,
        owner_id: Optional[AgentID] = None,
        owner_level: Optional[int] = None,
        state: Optional["State"] = None,
        at_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get filtered state respecting visibility rules.

        This is the primary method for agents to access state. It integrates
        with FeatureProvider.is_observable_by() for fine-grained visibility control.

        Args:
            agent_id: ID of the agent requesting state
            requestor_level: Hierarchy level of requesting agent (1=field, 2=coord, 3=system)
            owner_id: ID of the agent whose state is being requested (defaults to agent_id)
            owner_level: Hierarchy level of owner (defaults to requestor_level)
            state: Optional State object with FeatureProviders for visibility checking
            at_time: Optional timestamp for delayed observations (Option B)

        Returns:
            Filtered state dict based on visibility rules
        """
        owner_id = owner_id or agent_id
        owner_level = owner_level or requestor_level

        # Get base state (from history if time specified, otherwise current)
        base_state = self.get_state_at_time(at_time) if at_time is not None else self.state_cache

        agents_state = base_state.get('agents', {})
        agent_state = agents_state.get(owner_id, {})

        # Apply key-based visibility rules
        filtered = self._filter_state_by_keys(agent_id, agent_state)

        # If State object provided, apply FeatureProvider visibility rules
        if state is not None:
            return {
                feature.feature_name: filtered[feature.feature_name]
                for feature in state.features
                if feature.is_observable_by(agent_id, requestor_level, owner_id, owner_level)
                and feature.feature_name in filtered
            }

        return filtered

    def get_observable_features(
        self,
        requestor_id: AgentID,
        requestor_level: int,
        owner_id: AgentID,
        owner_level: int,
        state: "State",
    ) -> List[str]:
        """Get list of feature names observable by the requesting agent.

        Args:
            requestor_id: ID of agent requesting observation
            requestor_level: Hierarchy level of requestor
            owner_id: ID of agent that owns the state
            owner_level: Hierarchy level of owner
            state: State object containing FeatureProviders

        Returns:
            List of feature names the requestor can observe
        """
        return [
            feature.feature_name
            for feature in state.features
            if feature.is_observable_by(requestor_id, requestor_level, owner_id, owner_level)
        ]

    # ============================================
    # Agent Registration
    # ============================================

    def register_agent(self, agent_id: AgentID) -> None:
        """Register a new agent that can request state.

        Args:
            agent_id: Agent ID to register
        """
        if agent_id not in self.registered_agents:
            self.registered_agents.append(agent_id)

    def set_visibility_rules(
        self,
        agent_id: AgentID,
        allowed_keys: List[str],
    ) -> None:
        """Set visibility rules for an agent.

        Args:
            agent_id: Agent to set rules for
            allowed_keys: List of state keys the agent can access
        """
        self.visibility_rules[agent_id] = allowed_keys

    # ============================================
    # Required Agent Interface Methods
    # ============================================

    def observe(
        self,
        global_state: Optional[Dict[str, Any]] = None,
        proxy: Optional[Agent] = None,
        **kwargs,
    ) -> Observation:
        """ProxyAgent doesn't observe in the traditional sense.

        It receives state updates via update_state() instead.

        Returns:
            Empty observation with current timestamp
        """
        return Observation(timestamp=self._timestep)

    def act(self, observation: Observation, upstream_action: Any = None) -> None:
        """ProxyAgent doesn't take actions.

        It manages state distribution instead.
        """
        pass

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset proxy agent state.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)
        self.state_cache = {}
        self.state_history = []

    # ============================================
    # Message Broker Support (Option B - Event-Driven)
    # ============================================

    def _setup_channels(self) -> None:
        """Setup message broker channels for communication.

        Creates channels for:
        - Environment -> Proxy (result channel)
        - Proxy -> Each registered agent (info channels)
        """
        if self._message_broker is None:
            return

        env_id = self.env_id or "default"

        # Create env->proxy channel for receiving results
        env_channel = ChannelManager.custom_channel(
            self._result_channel_type, env_id, self.agent_id
        )
        self._message_broker.create_channel(env_channel)

        # Create proxy->agent channels for distributing state
        for agent_id in self.registered_agents:
            agent_channel = ChannelManager.info_channel(
                self.agent_id, agent_id, env_id
            )
            self._message_broker.create_channel(agent_channel)

    def receive_state_from_environment(self) -> Optional[Dict[str, Any]]:
        """Receive state from environment via message broker.

        Consumes messages from the result channel and updates the
        state cache with the latest state.

        Returns:
            Latest state payload, or None if no messages or no broker configured
        """
        if self._message_broker is None:
            return None

        env_id = self.env_id or "default"
        channel = ChannelManager.custom_channel(
            self._result_channel_type, env_id, self.agent_id
        )

        messages = self._message_broker.consume(
            channel=channel,
            recipient_id=self.agent_id,
            env_id=env_id,
            clear=True,
        )

        if not messages:
            return None

        # Use the latest message
        latest_msg = messages[-1]
        self.update_state(latest_msg.payload)
        return latest_msg.payload

    def distribute_state_to_agents(self) -> None:
        """Distribute cached state to registered agents via message broker.

        Sends filtered state to each registered agent based on visibility rules.
        Only works if message_broker was provided during initialization.
        """
        if self._message_broker is None:
            return

        if not self.state_cache:
            return

        env_id = self.env_id or "default"
        agents_state = self.state_cache.get("agents", {})

        for agent_id in self.registered_agents:
            # Get agent-specific state from aggregated state
            agent_state = agents_state.get(agent_id, {})

            # Apply visibility filtering
            filtered_state = self._filter_state_by_keys(agent_id, agent_state)

            # Send to agent via info channel
            channel = ChannelManager.info_channel(self.agent_id, agent_id, env_id)
            msg = Message(
                env_id=env_id,
                sender_id=self.agent_id,
                recipient_id=agent_id,
                timestamp=self._timestep,
                message_type=MessageType.INFO,
                payload=filtered_state,
            )
            self._message_broker.publish(channel, msg)

    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        num_registered = len(self.registered_agents)
        has_broker = self._message_broker is not None
        return f"ProxyAgent(id={self.agent_id}, registered_agents={num_registered}, broker={has_broker})"

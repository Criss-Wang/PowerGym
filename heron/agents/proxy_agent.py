"""Proxy Agent for managing state distribution in distributed MARL.

The ProxyAgent acts as an intermediary between the environment and other agents,
managing state updates and controlling information visibility.

Key responsibilities:
1. Receive state updates from the environment
2. Cache the latest state
3. Distribute state to other agents based on visibility rules
4. Control what information each agent can access

Usage Pattern:
    # Setup
    proxy = ProxyAgent(
        message_broker=broker,
        env_id="env_1",
        subordinate_agents=["battery_1", "solar_1"],
        visibility_rules={
            "battery_1": ["SoC", "Power"],  # Only see these features
            "solar_1": ["Irradiance"],
        }
    )

    # In simulation loop
    proxy.receive_state_from_environment()  # Get state from env
    proxy.distribute_state_to_agents()       # Send filtered state to agents

    # Agents request state through proxy (recommended pattern)
    state = proxy.get_state_for_agent("battery_1", requestor_level=1)
"""

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from heron.agents.base import Agent, AgentID
from heron.messaging.base import ChannelManager, Message, MessageBroker, MessageType
from heron.core.observation import Observation

if TYPE_CHECKING:
    from heron.core.feature import FeatureProvider
    from heron.core.state import State


PROXY_LEVEL = 3  # Level identifier for proxy-level agents (higher than coordinator)


class ProxyAgent(Agent):
    """Proxy agent that manages state distribution.

    The ProxyAgent sits between the environment and other agents, acting as the
    single source of truth for state information. All agents must retrieve
    state through the ProxyAgent rather than directly accessing the environment.

    Attributes:
        message_broker: MessageBroker instance for communication
        env_id: Environment ID for multi-environment isolation
        state_cache: Latest state received from environment
        visibility_rules: Dict mapping agent IDs to allowed state keys
        subordinate_agents: List of agent IDs that can request state from this proxy
        result_channel_type: Channel type for receiving results from environment
    """

    def __init__(
        self,
        agent_id: AgentID = "proxy_agent",
        message_broker: Optional[MessageBroker] = None,
        env_id: Optional[str] = None,
        subordinate_agents: Optional[List[AgentID]] = None,
        visibility_rules: Optional[Dict[AgentID, List[str]]] = None,
        result_channel_type: str = "result",
    ):
        """Initialize proxy agent.

        Args:
            agent_id: Unique identifier for this proxy agent
            message_broker: Message broker for communication
            env_id: Environment ID for multi-environment isolation
            subordinate_agents: List of agent IDs that can request state
            visibility_rules: Dict mapping agent IDs to allowed state keys.
                            If None, all agents see all state by default.
            result_channel_type: Channel type for environment->proxy communication.
                               Defaults to "result". Domains can customize this
                               for domain-specific communication patterns.
        """
        if message_broker is None:
            raise ValueError("ProxyAgent requires a message broker for communication")

        super().__init__(
            agent_id=agent_id,
            level=PROXY_LEVEL,
            message_broker=message_broker,
            upstream_id=None,  # Proxy has no upstream
            env_id=env_id,
            subordinates={},  # Proxy doesn't manage subordinates hierarchically
        )

        self.subordinate_agents = subordinate_agents or []
        self.visibility_rules = visibility_rules or {}
        self.state_cache: Dict[str, Any] = {}
        self.result_channel_type = result_channel_type

        # Setup channels for communication
        self._setup_proxy_channels()

    def _setup_proxy_channels(self) -> None:
        """Setup message channels for proxy agent communication.

        Creates channels for:
        1. Receiving state from environment
        2. Sending state to subordinate agents
        """
        if not self.message_broker:
            return

        # Channel for receiving state from environment
        env_to_proxy_channel = ChannelManager.custom_channel(
            self.result_channel_type,
            self.env_id,
            self.agent_id
        )
        self.message_broker.create_channel(env_to_proxy_channel)

        # Channels for sending state to each subordinate agent
        for agent_id in self.subordinate_agents:
            proxy_to_agent_channel = ChannelManager.info_channel(
                self.agent_id,
                agent_id,
                self.env_id
            )
            self.message_broker.create_channel(proxy_to_agent_channel)

    def receive_state_from_environment(self) -> Optional[Dict[str, Any]]:
        """Receive and cache state from environment.

        Returns:
            State payload or None if no message available
        """
        if not self.message_broker or not self.env_id:
            return None

        channel = ChannelManager.custom_channel(
            self.result_channel_type,
            self.env_id,
            self.agent_id
        )
        messages = self.message_broker.consume(
            channel,
            recipient_id=self.agent_id,
            env_id=self.env_id,
            clear=True
        )

        if messages:
            # Cache the most recent state
            self.state_cache = messages[-1].payload
            return self.state_cache

        return None

    def distribute_state_to_agents(self) -> None:
        """Distribute cached state to all subordinate agents.

        Sends the appropriate state information to each agent based on
        visibility rules. This should be called after receiving updated state
        from the environment.

        The environment sends an aggregated state with structure:
        {
            'converged': bool,  # or other status fields
            'agents': {
                'agent_id_1': {...agent1_specific_state...},
                'agent_id_2': {...agent2_specific_state...},
                ...
            }
        }

        The ProxyAgent extracts each agent's specific state and sends it individually.
        """
        if not self.message_broker or not self.state_cache:
            return

        # Extract agent-specific states from aggregated state
        agents_state = self.state_cache.get('agents', {})

        for agent_id in self.subordinate_agents:
            # Get agent-specific state from aggregated state
            agent_state = agents_state.get(agent_id, {})

            # Apply visibility filtering
            filtered_state = self._filter_state_for_agent(agent_id, agent_state)

            # Send state to agent
            channel = ChannelManager.info_channel(
                self.agent_id,
                agent_id,
                self.env_id
            )
            message = Message(
                env_id=self.env_id,
                sender_id=self.agent_id,
                recipient_id=agent_id,
                timestamp=self._timestep,
                message_type=MessageType.INFO,
                payload=filtered_state
            )
            self.message_broker.publish(channel, message)

    def _filter_state_for_agent(
        self, agent_id: AgentID, agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Filter state based on visibility rules for a specific agent.

        Args:
            agent_id: Agent requesting the state
            agent_state: Agent-specific state extracted from aggregated state

        Returns:
            Filtered state dict
        """
        if agent_id not in self.visibility_rules:
            # No specific rules, return full agent state
            return agent_state.copy()

        allowed_keys = self.visibility_rules[agent_id]
        filtered_state = {}

        for key in allowed_keys:
            if key in agent_state:
                filtered_state[key] = agent_state[key]

        return filtered_state

    def get_latest_state_for_agent(self, agent_id: AgentID) -> Dict[str, Any]:
        """Get the latest cached state for a specific agent.

        This method can be called by agents to retrieve state on-demand.

        Args:
            agent_id: Agent requesting the state

        Returns:
            Filtered state dict
        """
        agents_state = self.state_cache.get('agents', {})
        agent_state = agents_state.get(agent_id, {})
        return self._filter_state_for_agent(agent_id, agent_state)

    def get_state_for_agent(
        self,
        agent_id: AgentID,
        requestor_level: int,
        owner_id: Optional[AgentID] = None,
        owner_level: Optional[int] = None,
        state: Optional["State"] = None,
    ) -> Dict[str, Any]:
        """Get filtered state respecting FeatureProvider visibility rules.

        This is the recommended method for agents to access state. It integrates
        with FeatureProvider.is_observable_by() for fine-grained visibility control.

        Args:
            agent_id: ID of the agent requesting state
            requestor_level: Hierarchy level of requesting agent (1=field, 2=coord, 3=system)
            owner_id: ID of the agent whose state is being requested (defaults to agent_id)
            owner_level: Hierarchy level of owner (defaults to requestor_level)
            state: Optional State object with FeatureProviders for visibility checking

        Returns:
            Filtered state dict based on visibility rules
        """
        if owner_id is None:
            owner_id = agent_id
        if owner_level is None:
            owner_level = requestor_level

        # Get base state from cache
        agents_state = self.state_cache.get('agents', {})
        agent_state = agents_state.get(owner_id, {})

        # Apply key-based visibility rules first
        filtered = self._filter_state_for_agent(agent_id, agent_state)

        # If State object provided, apply FeatureProvider visibility rules
        if state is not None:
            feature_filtered = {}
            for feature in state.features:
                if feature.is_observable_by(
                    agent_id, requestor_level, owner_id, owner_level
                ):
                    feature_name = feature.feature_name
                    if feature_name in filtered:
                        feature_filtered[feature_name] = filtered[feature_name]
            return feature_filtered

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
        observable = []
        for feature in state.features:
            if feature.is_observable_by(
                requestor_id, requestor_level, owner_id, owner_level
            ):
                observable.append(feature.feature_name)
        return observable

    def register_subordinate(self, agent_id: AgentID) -> None:
        """Register a new subordinate agent.

        Args:
            agent_id: Agent ID to register
        """
        if agent_id not in self.subordinate_agents:
            self.subordinate_agents.append(agent_id)
            # Create channel for this agent
            if self.message_broker:
                channel = ChannelManager.info_channel(
                    self.agent_id, agent_id, self.env_id
                )
                self.message_broker.create_channel(channel)

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
    # Required Abstract Methods (Not used by ProxyAgent)
    # ============================================

    def observe(
        self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs
    ) -> Observation:
        """ProxyAgent doesn't need to observe in the traditional sense.

        Returns:
            Empty observation
        """
        return Observation(
            timestamp=self._timestep,
            local={},
            global_info={},
            messages=[]
        )

    def act(self, observation: Observation, *args, **kwargs) -> Any:
        """ProxyAgent doesn't take actions in the traditional sense.

        Instead, it manages state distribution.
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

    def __repr__(self) -> str:
        num_subordinates = len(self.subordinate_agents)
        return f"ProxyAgent(id={self.agent_id}, subordinates={num_subordinates})"

"""Proxy Agent for managing network state distribution.

The ProxyAgent acts as an intermediary between the environment and other agents,
managing network state updates and controlling information visibility.

Key responsibilities:
1. Receive network state updates from the environment
2. Cache the latest network state
3. Distribute network state to other agents based on visibility rules
4. Control what information each agent can access
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandapower as pp

from heron.agents.base import Agent, AgentID
from heron.messaging.base import ChannelManager, Message, MessageBroker, MessageType
from heron.core.observation import Observation


PROXY_LEVEL = 3  # Level identifier for proxy-level agents (higher than grid)


class ProxyAgent(Agent):
    """Proxy agent that manages network state distribution.

    The ProxyAgent sits between the environment and other agents, acting as the
    single source of truth for network state information. All agents must retrieve
    network state through the ProxyAgent rather than directly accessing the network.

    Attributes:
        message_broker: MessageBroker instance for communication
        env_id: Environment ID for multi-environment isolation
        network_state_cache: Latest network state received from environment
        visibility_rules: Dict mapping agent IDs to allowed state keys
        subordinate_agents: List of agent IDs that can request state from this proxy
    """

    def __init__(
        self,
        agent_id: AgentID = "proxy_agent",
        message_broker: Optional[MessageBroker] = None,
        env_id: Optional[str] = None,
        subordinate_agents: Optional[List[AgentID]] = None,
        visibility_rules: Optional[Dict[AgentID, List[str]]] = None,
    ):
        """Initialize proxy agent.

        Args:
            agent_id: Unique identifier for this proxy agent
            message_broker: Message broker for communication
            env_id: Environment ID for multi-environment isolation
            subordinate_agents: List of agent IDs that can request state
            visibility_rules: Dict mapping agent IDs to allowed state keys.
                            If None, all agents see all state by default.
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
        self.network_state_cache: Dict[str, Any] = {}

        # Setup channels for communication
        self._setup_proxy_channels()

    def _setup_proxy_channels(self) -> None:
        """Setup message channels for proxy agent communication.

        Creates channels for:
        1. Receiving network state from environment
        2. Sending network state to subordinate agents
        """
        if not self.message_broker:
            return

        # Channel for receiving network state from environment
        env_to_proxy_channel = ChannelManager.power_flow_result_channel(
            self.env_id,
            self.agent_id
        )
        self.message_broker.create_channel(env_to_proxy_channel)

        # Channels for sending network state to each subordinate agent
        for agent_id in self.subordinate_agents:
            proxy_to_agent_channel = ChannelManager.info_channel(
                self.agent_id,
                agent_id,
                self.env_id
            )
            self.message_broker.create_channel(proxy_to_agent_channel)

    def receive_network_state_from_environment(self) -> Optional[Dict[str, Any]]:
        """Receive and cache network state from environment.

        Returns:
            Network state payload or None if no message available
        """
        if not self.message_broker or not self.env_id:
            return None

        channel = ChannelManager.power_flow_result_channel(self.env_id, self.agent_id)
        messages = self.message_broker.consume(
            channel,
            recipient_id=self.agent_id,
            env_id=self.env_id,
            clear=True
        )

        if messages:
            # Cache the most recent network state
            self.network_state_cache = messages[-1].payload
            return self.network_state_cache

        return None

    def distribute_network_state_to_agents(self) -> None:
        """Distribute cached network state to all subordinate agents.

        Sends the appropriate network state information to each agent based on
        visibility rules. This should be called after receiving updated state
        from the environment.

        The environment sends an aggregated state with structure:
        {
            'converged': bool,
            'agents': {
                'agent_id_1': {...agent1_specific_state...},
                'agent_id_2': {...agent2_specific_state...},
                ...
            }
        }

        The ProxyAgent extracts each agent's specific state and sends it individually.
        """
        if not self.message_broker or not self.network_state_cache:
            return

        # Extract agent-specific states from aggregated state
        agents_state = self.network_state_cache.get('agents', {})

        for agent_id in self.subordinate_agents:
            # Get agent-specific state from aggregated state
            agent_state = agents_state.get(agent_id, {})

            # Apply visibility filtering
            filtered_state = self._filter_state_for_agent(agent_id, agent_state)

            # Send network state to agent
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

    def _filter_state_for_agent(self, agent_id: AgentID, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Filter network state based on visibility rules for a specific agent.

        Args:
            agent_id: Agent requesting the state
            agent_state: Agent-specific state extracted from aggregated state

        Returns:
            Filtered network state dict
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

    def get_latest_network_state_for_agent(self, agent_id: AgentID) -> Dict[str, Any]:
        """Get the latest cached network state for a specific agent.

        This method can be called by agents to retrieve network state on-demand.

        Args:
            agent_id: Agent requesting the state

        Returns:
            Filtered network state dict
        """
        agents_state = self.network_state_cache.get('agents', {})
        agent_state = agents_state.get(agent_id, {})
        return self._filter_state_for_agent(agent_id, agent_state)

    # ============================================
    # Required Abstract Methods (Not used by ProxyAgent)
    # ============================================

    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
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

        Instead, it manages network state distribution.
        """
        pass

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset proxy agent state.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)
        self.network_state_cache = {}

    def __repr__(self) -> str:
        num_subordinates = len(self.subordinate_agents)
        return f"ProxyAgent(id={self.agent_id}, subordinates={num_subordinates})"

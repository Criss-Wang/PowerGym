"""Power Grid Proxy Agent.

This module provides a power-grid specific ProxyAgent that uses the
"power_flow" channel type for environment-to-proxy communication.
"""

from typing import Any, Dict, List, Optional

from heron.agents.proxy_agent import ProxyAgent as BaseProxyAgent, PROXY_LEVEL
from heron.messaging.base import ChannelManager, Message, MessageBroker, MessageType
from heron.utils.typing import AgentID


# Re-export PROXY_LEVEL
__all__ = ["ProxyAgent", "PROXY_LEVEL", "POWER_FLOW_CHANNEL_TYPE"]

# Power grid uses "power_flow" as the channel type for power flow results
POWER_FLOW_CHANNEL_TYPE = "power_flow"


class ProxyAgent(BaseProxyAgent):
    """Power grid proxy agent for managing network state distribution.

    This is a thin wrapper around the generic ProxyAgent that adds
    power-grid specific message broker integration for receiving
    power flow results from the environment and distributing state
    to registered agents.

    Example:
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            registered_agents=["grid_1", "grid_2"],
        )
    """

    def __init__(
        self,
        agent_id: AgentID = "proxy_agent",
        message_broker: Optional[MessageBroker] = None,
        env_id: Optional[str] = None,
        registered_agents: Optional[List[AgentID]] = None,
        visibility_rules: Optional[Dict[AgentID, List[str]]] = None,
        history_length: int = 100,
    ):
        """Initialize power grid proxy agent.

        Args:
            agent_id: Unique identifier for this proxy agent
            message_broker: Message broker for communication (required)
            env_id: Environment ID for multi-environment isolation
            registered_agents: List of agent IDs that can request state
            visibility_rules: Dict mapping agent IDs to allowed state keys.
                If None, all agents see all state by default.
            history_length: Number of timesteps of history to maintain

        Raises:
            ValueError: If message_broker is not provided
        """
        if message_broker is None:
            raise ValueError(
                "Power grid ProxyAgent requires a message broker. "
                "Pass message_broker=InMemoryBroker() or similar."
            )

        super().__init__(
            agent_id=agent_id,
            env_id=env_id,
            registered_agents=registered_agents,
            visibility_rules=visibility_rules,
            history_length=history_length,
        )

        # Set message broker for distributed communication
        self.set_message_broker(message_broker)

        # Power grid specific channel type
        self._result_channel_type = POWER_FLOW_CHANNEL_TYPE

        # Setup channels for message broker communication
        self._setup_channels()

    def _setup_channels(self) -> None:
        """Setup message broker channels for communication."""
        if self._message_broker is None:
            return

        env_id = self.env_id or "default"

        # Create env->proxy channel for receiving power flow results
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

    # ============================================
    # Message Broker Communication Methods
    # ============================================

    def receive_state_from_environment(self) -> Optional[Dict[str, Any]]:
        """Receive state from environment via message broker.

        Consumes messages from the power_flow channel and updates the
        state cache with the latest state.

        Returns:
            Latest state payload, or None if no messages
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
        self.state_cache = latest_msg.payload
        return latest_msg.payload

    def distribute_state_to_agents(self) -> None:
        """Distribute cached state to registered agents via message broker.

        Sends filtered state to each registered agent based on visibility rules.
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

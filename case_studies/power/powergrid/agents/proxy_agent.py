"""Power Grid Proxy Agent.

This module provides a power-grid specific ProxyAgent that uses the
"power_flow" channel type for environment-to-proxy communication.
"""

from typing import Dict, List, Optional

from heron.agents.base import AgentID
from heron.agents.proxy_agent import ProxyAgent as BaseProxyAgent, PROXY_LEVEL
from heron.messaging.base import MessageBroker


# Re-export PROXY_LEVEL for backwards compatibility
__all__ = ["ProxyAgent", "PROXY_LEVEL", "POWER_FLOW_CHANNEL_TYPE"]

# Power grid uses "power_flow" as the channel type for power flow results
POWER_FLOW_CHANNEL_TYPE = "power_flow"


class ProxyAgent(BaseProxyAgent):
    """Power grid proxy agent for managing network state distribution.

    This is a thin wrapper around the generic ProxyAgent that sets the
    channel type to "power_flow" for power grid specific communication.

    Example:
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["grid_1", "grid_2"],
        )
    """

    def __init__(
        self,
        agent_id: AgentID = "proxy_agent",
        message_broker: Optional[MessageBroker] = None,
        env_id: Optional[str] = None,
        subordinate_agents: Optional[List[AgentID]] = None,
        visibility_rules: Optional[Dict[AgentID, List[str]]] = None,
    ):
        """Initialize power grid proxy agent.

        Args:
            agent_id: Unique identifier for this proxy agent
            message_broker: Message broker for communication
            env_id: Environment ID for multi-environment isolation
            subordinate_agents: List of agent IDs that can request state
            visibility_rules: Dict mapping agent IDs to allowed state keys.
                            If None, all agents see all state by default.
        """
        super().__init__(
            agent_id=agent_id,
            message_broker=message_broker,
            env_id=env_id,
            subordinate_agents=subordinate_agents,
            visibility_rules=visibility_rules,
            result_channel_type=POWER_FLOW_CHANNEL_TYPE,
        )

    # Backwards-compatible aliases for power grid specific naming
    @property
    def network_state_cache(self) -> Dict:
        """Alias for state_cache (backwards compatibility)."""
        return self.state_cache

    @network_state_cache.setter
    def network_state_cache(self, value: Dict) -> None:
        """Alias setter for state_cache (backwards compatibility)."""
        self.state_cache = value

    def receive_network_state_from_environment(self):
        """Alias for receive_state_from_environment (backwards compatibility)."""
        return self.receive_state_from_environment()

    def distribute_network_state_to_agents(self):
        """Alias for distribute_state_to_agents (backwards compatibility)."""
        return self.distribute_state_to_agents()

    def get_latest_network_state_for_agent(self, agent_id: AgentID):
        """Alias for get_latest_state_for_agent (backwards compatibility)."""
        return self.get_latest_state_for_agent(agent_id)

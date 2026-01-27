"""Power grid specific messaging channels and types.

This module provides power-grid domain extensions to the HERON messaging system.
"""

from heron.messaging.base import (
    ChannelManager,
    ChannelRegistry,
    MessageTypeRegistry,
)


# Register power-grid specific message types
MessageTypeRegistry.register(
    "power_flow_result",
    "Power flow computation results from environment to proxy agent"
)
MessageTypeRegistry.register(
    "voltage_update",
    "Voltage measurements update"
)
MessageTypeRegistry.register(
    "line_flow_update",
    "Line power flow measurements update"
)

# Register power-grid specific channel types
ChannelRegistry.register(
    "power_flow",
    "Channel for power flow results from environment to proxy agent"
)


class PowerGridChannelManager:
    """Power grid specific channel management utilities.

    Provides convenience methods for creating power-grid specific channels
    using the generic ChannelManager.custom_channel() method.
    """

    @staticmethod
    def power_flow_result_channel(env_id: str, agent_id: str) -> str:
        """Generate channel for power flow results from environment to proxy.

        Args:
            env_id: Environment ID
            agent_id: Proxy agent ID

        Returns:
            Channel name string
        """
        return ChannelManager.custom_channel("power_flow", env_id, agent_id)

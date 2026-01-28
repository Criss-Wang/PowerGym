"""Device-level agents that wrap Device objects as autonomous agents.

DeviceAgent provides a bridge between the existing Device abstraction and the
new Agent abstraction, enabling devices to participate in multi-agent control.

DeviceAgent extends FieldAgent from the HERON framework, inheriting the
standard field-level agent capabilities while adding power-grid specific
functionality.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass

from heron.agents.field_agent import FieldAgent, FieldConfig
from powergrid.core.state.state import DeviceState
from heron.core.policies import Policy
from heron.protocols.base import NoProtocol, Protocol
from heron.utils.typing import AgentID
from heron.messaging.base import MessageBroker

DEVICE_LEVEL = 1  # Level identifier for device-level agents


# Alias for backward compatibility
DeviceConfig = FieldConfig


class DeviceAgent(FieldAgent):
    """Wraps a Device as an autonomous agent.

    DeviceAgent extends FieldAgent from the HERON framework, maintaining
    compatibility with the existing Device interface while adding agent
    capabilities like observation extraction, communication, and pluggable
    policies.

    DeviceAgent only observes its local device state. Global information should
    be provided by parent GridAgent through coordination protocols/messages.

    Attributes:
        device: Underlying Device object (DG, ESS, RES, etc.)
        policy: Decision-making policy (learned or rule-based)
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        protocol: Protocol = NoProtocol(),
        policy: Optional[Policy] = None,

        # communication params
        message_broker: Optional[MessageBroker] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # DeviceAgent specific params
        device_config: Dict[str, Any] = {},
    ):
        """Initialize device agent.

        Args:
            agent_id: Agent ID (defaults to device name from config)
            policy: Decision policy (optional)
            protocol: Communication protocol (defaults to NoProtocol)
            message_broker: Optional message broker for hierarchical execution
            upstream_id: Optional upstream agent ID for hierarchical execution
            env_id: Optional environment ID for multi-environment isolation
            device_config: Device configuration dict
        """
        # Convert device_config to the config format expected by FieldAgent
        config = {
            "name": device_config.get("name", "device_agent"),
            "state_config": device_config.get("device_state_config", {}),
            "discrete_action": device_config.get("discrete_action", False),
            "discrete_action_cats": device_config.get("discrete_action_cats", 2),
        }

        # Store device-specific config for backward compatibility
        self._device_config = device_config

        super().__init__(
            agent_id=agent_id or config["name"],
            protocol=protocol,
            policy=policy,
            message_broker=message_broker,
            upstream_id=upstream_id,
            env_id=env_id,
            config=config,
        )

        # Use DeviceState instead of FieldAgentState for power-grid domain
        self.state = DeviceState(
            owner_id=self.agent_id,
            owner_level=DEVICE_LEVEL
        )

        # Re-initialize state with device-specific setup
        self._init_state()

    # ============================================
    # Backward Compatibility Aliases
    # ============================================

    @property
    def config(self) -> FieldConfig:
        """Alias for field_config for backward compatibility."""
        return self.field_config

    @config.setter
    def config(self, value: FieldConfig) -> None:
        """Setter for config alias."""
        self.field_config = value

    def set_device_action(self) -> None:
        """Alias for set_action() for backward compatibility.

        Subclasses should override this method to define device-specific
        action spaces.
        """
        pass

    def set_device_state(self) -> None:
        """Alias for set_state() for backward compatibility.

        Subclasses should override this method to define device-specific
        state attributes.
        """
        pass

    def reset_device(self, *args, **kwargs) -> None:
        """Alias for reset_agent() for backward compatibility.

        Subclasses should override this method to implement device-specific
        reset logic.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        pass

    # Override FieldAgent methods to call device-specific aliases
    def set_action(self) -> None:
        """Define/initialize the device-specific action.

        Delegates to set_device_action() for backward compatibility.
        """
        self.set_device_action()

    def set_state(self) -> None:
        """Define/initialize device-specific state.

        Delegates to set_device_state() for backward compatibility.
        """
        self.set_device_state()

    def reset_agent(self, *args, **kwargs) -> None:
        """Reset device to initial state.

        Delegates to reset_device() for backward compatibility.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self.reset_device(*args, **kwargs)

    # ============================================
    # Device-Specific Methods (Abstract)
    # ============================================

    def update_state(self, *args, **kwargs) -> None:
        """Update device state (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError

    def update_cost_safety(self, *args, **kwargs) -> None:
        """Update device cost and safety metrics (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError

    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        """Return string representation of the agent.

        Returns:
            String representation
        """
        return f"DeviceAgent(id={self.agent_id}, policy={self.policy}, protocol={self.protocol})"

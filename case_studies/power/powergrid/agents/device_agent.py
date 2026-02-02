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

        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # DeviceAgent specific params
        device_config: Dict[str, Any] = {},

        # timing params (for event-driven scheduling - Option B)
        tick_interval: float = 1.0,
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,
    ):
        """Initialize device agent.

        Args:
            agent_id: Agent ID (defaults to device name from config)
            policy: Decision policy (optional)
            protocol: Communication protocol (defaults to NoProtocol)
            upstream_id: Optional upstream agent ID for hierarchy structure
            env_id: Optional environment ID for multi-environment isolation
            device_config: Device configuration dict
            tick_interval: Time between agent ticks (default 1s for devices)
            obs_delay: Observation delay
            act_delay: Action delay
            msg_delay: Message delay
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
            upstream_id=upstream_id,
            env_id=env_id,
            config=config,
            tick_interval=tick_interval,
            obs_delay=obs_delay,
            act_delay=act_delay,
            msg_delay=msg_delay,
        )

        # Use DeviceState instead of FieldAgentState for power-grid domain
        self.state = DeviceState(
            owner_id=self.agent_id,
            owner_level=DEVICE_LEVEL
        )

        # Power-grid specific cost/safety metrics
        self.cost: float = 0.0
        self.safety: float = 0.0

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

    def get_reward(self) -> Dict[str, float]:
        """Get device reward based on cost and safety metrics.

        Returns:
            Dict with cost and safety values
        """
        return {"cost": self.cost, "safety": self.safety}

    # ============================================
    # Distributed Mode Configuration
    # ============================================

    def configure_for_distributed(self, message_broker=None) -> None:
        """Configure this device for distributed execution mode.

        In distributed mode, the device communicates via message broker
        rather than direct method calls. This enables realistic async
        communication with configurable delays.

        Args:
            message_broker: MessageBroker instance for inter-agent communication.
                           If None, uses the broker from parent (upstream) agent.
        """
        if message_broker is not None:
            self.set_message_broker(message_broker)

        # Create message channel for this device (uses result channel for device state)
        if self.message_broker is not None:
            from heron.messaging.base import ChannelManager
            env_id = self.env_id or "default"
            channel = ChannelManager.result_channel(env_id, self.agent_id)
            self.message_broker.create_channel(channel)

    def publish_state_update(self) -> None:
        """Publish current state to message broker for environment sync.

        In distributed mode (Option B), devices publish their state updates
        which the environment then applies to the PandaPower network.
        Called after tick() processes action in event-driven mode.
        """
        if self.message_broker is None:
            return

        from heron.messaging.base import ChannelManager, Message, MessageType

        # Build state update payload
        state_update = {
            'agent_id': self.agent_id,
            'device_type': self._get_pandapower_device_type(),
            'P_MW': self._get_power_output(),
            'Q_MVAr': self._get_reactive_power(),
            'in_service': self._is_in_service(),
        }

        # Publish to state update channel
        channel = ChannelManager.state_update_channel(self.env_id)
        message = Message(
            env_id=self.env_id,
            sender_id=self.agent_id,
            recipient_id="environment",
            timestamp=self._timestep,
            message_type=MessageType.STATE_UPDATE,
            payload=state_update,
        )
        self.message_broker.publish(channel, message)

    def _get_pandapower_device_type(self) -> str:
        """Get the PandaPower element type for this device.

        Returns:
            String identifier for PandaPower element type (e.g., 'sgen', 'storage')
        """
        # Override in subclasses (Generator -> 'sgen', ESS -> 'storage')
        return 'sgen'

    def _get_power_output(self) -> float:
        """Get current real power output in MW.

        Returns:
            Real power output (positive = generation)
        """
        # Override in subclasses to return actual power
        return 0.0

    def _get_reactive_power(self) -> float:
        """Get current reactive power output in MVAr.

        Returns:
            Reactive power output
        """
        # Override in subclasses to return actual reactive power
        return 0.0

    def _is_in_service(self) -> bool:
        """Check if device is currently in service.

        Returns:
            True if device is operational
        """
        # Override in subclasses to return actual status
        return True

    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        """Return string representation of the agent.

        Returns:
            String representation
        """
        return f"DeviceAgent(id={self.agent_id}, policy={self.policy}, protocol={self.protocol})"

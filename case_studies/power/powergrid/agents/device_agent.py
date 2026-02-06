"""Device-level agents that wrap Device objects as autonomous agents.

DeviceAgent provides a bridge between the existing Device abstraction and the
new Agent abstraction, enabling devices to participate in multi-agent control.

DeviceAgent extends FieldAgent from the HERON framework, inheriting the
standard field-level agent capabilities while adding power-grid specific
functionality.

Timing Configuration:
    DeviceAgent accepts a `tick_config` parameter for event-driven scheduling.
    Use `TickConfig.deterministic()` for training (no jitter) or
    `TickConfig.with_jitter()` for testing (realistic timing variance).

    Example:
        # Deterministic timing (training)
        tick_config = TickConfig.deterministic(tick_interval=1.0)

        # With jitter (testing)
        tick_config = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.1,
            act_delay=0.2,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=42,
        )
"""

from typing import Any, Dict, Optional

from heron.agents.field_agent import FieldAgent
from heron.core.policies import Policy
from heron.protocols.base import NoProtocol, Protocol
from heron.scheduling.tick_config import TickConfig
from heron.utils.typing import AgentID


class DeviceAgent(FieldAgent):
    """Wraps a Device as an autonomous agent.

    DeviceAgent extends FieldAgent from the HERON framework, maintaining
    compatibility with the existing Device interface while adding agent
    capabilities like observation extraction, communication, and pluggable
    policies.

    DeviceAgent only observes its local device state. Global information should
    be provided by parent GridAgent through coordination protocols/messages.

    Subclasses must implement:
        - set_action(): Define device-specific action space
        - set_state(): Define device-specific state features
        - reset_agent(): Reset device to initial state
        - update_state(): Update device state from action/environment
        - update_cost_safety(): Calculate cost and safety metrics

    Attributes:
        state: DeviceState containing device features
        policy: Decision-making policy (learned or rule-based)
        cost: Per-step operational cost
        safety: Per-step safety violation penalty
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
        device_config: Optional[Dict[str, Any]] = None,

        # timing config (for event-driven scheduling - Option B)
        tick_config: Optional[TickConfig] = None,
    ):
        """Initialize device agent.

        Args:
            agent_id: Agent ID (defaults to device name from config)
            policy: Decision policy (optional)
            protocol: Communication protocol (defaults to NoProtocol)
            upstream_id: Optional upstream agent ID for hierarchy structure
            env_id: Optional environment ID for multi-environment isolation
            device_config: Device configuration dict
            tick_config: Timing configuration for event-driven scheduling.
                Defaults to TickConfig with tick_interval=1.0 (devices tick frequently).
                Use TickConfig.deterministic() or TickConfig.with_jitter().
        """
        # Convert device_config to the config format expected by FieldAgent
        if device_config is None:
            device_config = {}
        config = {
            "name": device_config.get("name", "device_agent"),
            "state_config": device_config.get("device_state_config", {}),
            "discrete_action": device_config.get("discrete_action", False),
            "discrete_action_cats": device_config.get("discrete_action_cats", 2),
        }

        # Store device-specific config
        self._device_config = device_config

        super().__init__(
            agent_id=agent_id or config["name"],
            protocol=protocol,
            policy=policy,
            upstream_id=upstream_id,
            env_id=env_id,
            config=config,
            tick_config=tick_config,
        )

        # Power-grid specific cost/safety metrics
        self.cost: float = 0.0
        self.safety: float = 0.0

    # ============================================
    # Device-Specific Methods (Override in Subclasses)
    # ============================================

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
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        """Return string representation of the agent.

        Returns:
            String representation
        """
        return f"DeviceAgent(id={self.agent_id}, policy={self.policy}, protocol={self.protocol})"

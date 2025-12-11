from dataclasses import dataclass
from typing import Any, Optional, Dict

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.core.policies import Policy
from powergrid.features.tap_changer import TapChangerPh
from powergrid.messaging.base import MessageBroker
from powergrid.utils.cost import tap_change_cost
from powergrid.utils.safety import loading_over_pct
from powergrid.utils.typing import AgentID


@dataclass
class TransformerConfig:
    """Configuration for Transformer device."""
    name: str
    sn_mva: Optional[float] = None
    tap_max: Optional[int] = None
    tap_min: Optional[int] = None
    dt: float = 1.0
    tap_change_cost: float = 0.0


class Transformer(DeviceAgent):
    """On-load tap changer (OLTC) transformer.

    Discrete action selects tap index in [tap_min, tap_max]. Optional
    tap_change_cost applies per step moved to account for wear/operations.
    """

    def __init__(
        self,
        *,
        agent_id: Optional[AgentID] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        message_broker: Optional[MessageBroker] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        device_config: Dict[str, Any] = {},
    ) -> None:
        """Initialize Transformer agent.

        Args:
            agent_id: Agent ID (defaults to name from config)
            policy: Decision policy (optional)
            protocol: Communication protocol (defaults to NoProtocol)
            message_broker: Optional message broker for hierarchical execution
            upstream_id: Optional upstream agent ID for hierarchical execution
            env_id: Optional environment ID for multi-environment isolation
            device_config: Device configuration dict
        """
        state_config = device_config.get("device_state_config", {})

        self._transformer_config = TransformerConfig(
            name=state_config.get("name", "transformer"),
            sn_mva=state_config.get("sn_mva", None),
            tap_max=state_config.get("tap_max", None),
            tap_min=state_config.get("tap_min", None),
            dt=state_config.get("dt", 1.0),
            tap_change_cost=state_config.get("tap_change_cost", 0.0),
        )

        self._last_tap_position = 0

        super().__init__(
            agent_id=agent_id or self._transformer_config.name,
            policy=policy,
            protocol=protocol,
            message_broker=message_broker,
            upstream_id=upstream_id,
            env_id=env_id,
            device_config=device_config,
        )

    def set_device_action(self) -> None:
        """Define discrete action for tap position selection."""
        cfg = self._transformer_config
        if cfg.tap_max is not None and cfg.tap_min is not None:
            self.action.set_specs(
                dim_c=0,
                dim_d=1,
                ncats=cfg.tap_max - cfg.tap_min + 1,
            )

    def set_device_state(self) -> None:
        """Define device state with tap changer block."""
        cfg = self._transformer_config
        tap_changer = TapChangerPh(
            tap_position=cfg.tap_min if cfg.tap_min is not None else 0,
            tap_min=cfg.tap_min or 0,
            tap_max=cfg.tap_max or 0,
        )
        self.state.features = [tap_changer]
        self.state.owner_id = self.agent_id
        self.state.owner_level = self.level

    def reset_device(self, **kwargs) -> None:
        """Reset transformer to initial tap position.

        Args:
            **kwargs: Optional keyword arguments (unused)
        """
        self.state.reset()
        self.action.reset()

        cfg = self._transformer_config
        self._last_tap_position = cfg.tap_min if cfg.tap_min is not None else 0
        self.cost = 0.0
        self.safety = 0.0

    def update_state(self, **kwargs) -> None:
        """Update tap position from action.

        Args:
            **kwargs: Optional keyword arguments for feature updates
        """
        cfg = self._transformer_config
        if cfg.tap_max is not None and cfg.tap_min is not None and self.action.d.size:
            new_tap = int(self.action.d[0]) + int(cfg.tap_min)
            self.tap_changer.set_values(tap_position=new_tap)

        # Apply any additional kwargs to tap changer
        if kwargs:
            self.tap_changer.set_values(**kwargs)

    def update_cost_safety(self, **kwargs) -> None:
        """Update cost from tap changes and safety from loading.

        Args:
            **kwargs: Optional keyword arguments:
                loading_percentage: Transformer loading percentage
        """
        loading_percentage = kwargs.get("loading_percentage", 0.0)

        # Safety: loading-derived penalty
        self.safety = loading_over_pct(loading_percentage)

        # Cost: tap change operations
        delta = abs(self.tap_changer.tap_position - self._last_tap_position)
        self.cost = tap_change_cost(delta, self._transformer_config.tap_change_cost)
        self._last_tap_position = self.tap_changer.tap_position

    @property
    def tap_changer(self) -> TapChangerPh:
        """Get tap changer feature block."""
        for f in self.state.features:
            if isinstance(f, TapChangerPh):
                return f

    @property
    def name(self) -> str:
        """Get transformer name."""
        return self.agent_id

    def __repr__(self) -> str:
        """Return string representation of the transformer."""
        cfg = self._transformer_config
        return f"Transformer(name={self.name}, S={cfg.sn_mva}MVA, tap∈[{cfg.tap_min},{cfg.tap_max}])"
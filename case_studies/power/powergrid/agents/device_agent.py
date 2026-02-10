from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import FeatureProvider
from heron.core.policies import Policy
from heron.protocols.base import NoProtocol, Protocol
from heron.scheduling.tick_config import TickConfig
from heron.utils.typing import AgentID


@dataclass
class CostSafetyMetrics(FeatureProvider):
    """Cost and safety metrics for device agents.

    These values are updated by update_cost_safety() and exposed
    as observable features so they can be included in local_state
    for compute_local_reward().
    """
    visibility = ["owner"]  # Only the device itself can observe these

    cost: float = 0.0       # Operating cost for this timestep
    safety: float = 0.0     # Safety penalty/violation metric


class DeviceAgent(FieldAgent):
    def __init__(
        self,
        agent_id: AgentID,
        features: List[FeatureProvider] = [],
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
            protocol=protocol,
            policy=policy,
        )


    # ============================================
    # Property Accessors for Features
    # ============================================

    @property
    def cost(self) -> float:
        """Get current operating cost from features."""
        for f in self.state.features:
            if isinstance(f, CostSafetyMetrics):
                return f.cost
        return 0.0

    @property
    def safety(self) -> float:
        """Get current safety penalty from features."""
        for f in self.state.features:
            if isinstance(f, CostSafetyMetrics):
                return f.safety
        return 0.0


    # ============================================
    # Lifecycle Methods (Following BatteryAgent Pattern)
    # ============================================

    @abstractmethod
    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """Initialize action space (abstract - must override in subclasses).

        Subclasses define:
        - Continuous action dimensions (dim_c) and ranges
        - Discrete action dimensions (dim_d) and categories (ncats)

        Returns:
            Initialized Action object with specs and default values
        """
        pass

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward for this agent from local state.

        Default implementation: reward = -cost - safety
        (minimize operating cost and safety violations)

        Args:
            local_state: State dict from proxy.get_local_state() with structure:
                {"FeatureName": np.array([...]), ...}

                For DeviceAgent, this includes:
                - "CostSafetyMetrics": np.array([cost, safety])
                - Plus any device-specific features (ElectricalBasePh, StorageBlock, etc.)

        Returns:
            Reward value (higher is better)

        Note:
            Subclasses can override to implement custom reward functions.
            For example, an ESS might reward high SOC: reward = soc - cost - safety
        """
        if "CostSafetyMetrics" not in local_state:
            raise ValueError("Local state must include 'CostSafetyMetrics' for reward computation")
        metrics_vec = local_state["CostSafetyMetrics"]  # array([cost, safety])
        cost = float(metrics_vec[0])
        safety = float(metrics_vec[1])
        return -cost - safety

    @abstractmethod
    def set_action(self, action: Any) -> None:
        """Set action from Action object or compatible format.

        Args:
            action: Action object or numpy array/dict with action values
        """
        pass

    def apply_action(self):
        """Apply the current action to update state.

        Default implementation calls set_state() for backward compatibility.
        Subclasses should override either apply_action() or set_state().
        """
        self.set_state()
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from powergrid.agents.device_agent import DeviceAgent, CostSafetyMetrics
from heron.core.action import Action
from heron.core.feature import FeatureProvider
from heron.protocols.base import Protocol
from heron.core.policies import Policy
from powergrid.core.features.tap_changer import TapChangerPh
from powergrid.utils.cost import tap_change_cost
from powergrid.utils.safety import loading_over_pct
from heron.utils.typing import AgentID
from heron.scheduling.tick_config import TickConfig


@dataclass
class TransformerConfig:
    """Configuration for Transformer device."""
    name: str

    # Phase model configuration
    phase_model: str = "balanced_1ph"
    phase_spec: Optional[Dict[str, Any]] = None

    # Capacity
    sn_mva: Optional[float] = None

    # Tap limits
    tap_max: Optional[int] = None
    tap_min: Optional[int] = None

    # Time step
    dt: float = 1.0

    # Economic parameters
    tap_change_cost: float = 0.0


class Transformer(DeviceAgent):
    """On-load tap changer (OLTC) transformer.

    Discrete action selects tap index in [tap_min, tap_max]. Optional
    tap_change_cost applies per step moved to account for wear/operations.
    """

    def __init__(
        self,
        agent_id: AgentID,
        transformer_config: TransformerConfig,
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
        # Extract config fields into instance variables
        self._initialize_from_config(transformer_config)

        super().__init__(
            agent_id=agent_id,
            features=features,
            policy=policy,
            protocol=protocol,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
        )

    def _initialize_from_config(self, config: TransformerConfig) -> None:
        """Extract and store needed config fields as instance variables.

        This avoids persisting the entire config object and makes explicit
        which fields are actually used by the transformer.

        Args:
            config: TransformerConfig object to extract fields from
        """
        # Capacity
        self._sn_mva = config.sn_mva

        # Action space bounds
        self._tap_max = config.tap_max
        self._tap_min = config.tap_min

        # Time step
        self._dt = config.dt

        # Economic parameters
        self._tap_change_cost = config.tap_change_cost

        # Phase model & spec
        from powergrid.utils.phase import PhaseModel, PhaseSpec, check_phase_model_consistency
        self.phase_model = PhaseModel(config.phase_model)
        self.phase_spec = PhaseSpec().from_dict(config.phase_spec or {})
        check_phase_model_consistency(self.phase_model, self.phase_spec)

        # Tap position tracking for cost computation
        self._last_tap_position = 0

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """Initialize discrete action space for tap position selection.

        Returns:
            Initialized Action object with discrete tap selection specs
        """
        action = Action()

        if self._tap_max is not None and self._tap_min is not None:
            action.set_specs(
                dim_c=0,
                dim_d=1,
                ncats=self._tap_max - self._tap_min + 1,
            )

            # Initialize with neutral tap position (typically 0 offset)
            action.set_values(d=np.array([0], dtype=np.int32))

        return action

    def set_action(self, action: Any) -> None:
        """Set action from Action object or compatible format.

        Args:
            action: Action object or numpy array with action values
        """
        if isinstance(action, Action):
            # Extract action vector from Action object
            if action.d.size > 0:
                self.action.set_values(d=action.d)
        else:
            # Direct value (numpy array or dict)
            self.action.set_values(action)

    def set_state(self) -> None:
        """Apply action to update transformer state.

        Modern HERON pattern: called by apply_action() after action is set.
        Updates tap position and cost/safety metrics.
        """
        self._update_tap_position()
        # Update tap change cost (loading_percentage=0 as it comes from observations)
        self.update_cost_safety(loading_percentage=0.0)

    def apply_action(self) -> None:
        """Apply current action to update state.

        Overrides DeviceAgent.apply_action() to follow modern HERON pattern.
        """
        self.set_state()

    def sync_state_from_observed(self, observed_state: Any) -> None:
        """Synchronize state from external observations, then update cost/safety.

        Called after state features are updated from observations (e.g., from power flow).
        Recalculates cost and safety metrics including loading from observations.

        Args:
            observed_state: External observations (may include loading_percentage)
        """
        # Call parent to sync state features
        super().sync_state_from_observed(observed_state)

        # Extract loading percentage from observations if available
        loading_pct = 0.0
        if isinstance(observed_state, dict):
            loading_pct = observed_state.get("loading_percentage", 0.0)

        # Update cost and safety based on new state and loading
        self.update_cost_safety(loading_percentage=loading_pct)

    def reset_agent(self, **kwargs) -> None:
        """Reset transformer to initial tap position.

        Args:
            **kwargs: Optional keyword arguments (unused)
        """
        self.state.reset()
        self.action.reset()

        self._last_tap_position = self._tap_min if self._tap_min is not None else 0

        # Reset cost/safety metrics
        self.state.update_feature(
            CostSafetyMetrics.feature_name,
            cost=0.0,
            safety=0.0
        )

    def _update_tap_position(self) -> None:
        """Update tap position from discrete action.

        Extracts the discrete action value and updates the tap changer feature.
        """
        if self._tap_max is not None and self._tap_min is not None and self.action.d.size > 0:
            new_tap = int(self.action.d[0]) + int(self._tap_min)
            self.state.update_feature(
                TapChangerPh.feature_name,
                tap_position=new_tap
            )

    def update_cost_safety(self, **kwargs) -> None:
        """Update cost from tap changes and safety from loading.

        Args:
            **kwargs: Optional keyword arguments:
                loading_percentage: Transformer loading percentage
        """
        loading_percentage = kwargs.get("loading_percentage", 0.0)

        # Safety: loading-derived penalty
        safety = loading_over_pct(loading_percentage)

        # Cost: tap change operations
        delta = abs(self.tap_changer.tap_position - self._last_tap_position)
        cost = tap_change_cost(delta, self._tap_change_cost)
        self._last_tap_position = self.tap_changer.tap_position

        # Sync cost and safety to the CostSafetyMetrics feature
        self.state.update_feature(
            CostSafetyMetrics.feature_name,
            cost=cost,
            safety=safety
        )

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward for Transformer based on cost and safety.

        Reward = -cost - safety
        (minimize tap change operations and loading violations)

        Args:
            local_state: State dict from proxy.get_local_state() with structure:
                {"TapChangerPh": np.array([tap_position, ...]),
                 "CostSafetyMetrics": np.array([cost, safety]), ...}

        Returns:
            Reward value (higher is better)
        """
        cost_penalty = 0.0
        safety_penalty = 0.0

        # Extract cost/safety from CostSafetyMetrics
        if "CostSafetyMetrics" in local_state:
            metrics_vec = local_state["CostSafetyMetrics"]
            cost_penalty = float(metrics_vec[0])
            safety_penalty = float(metrics_vec[1])

        return -cost_penalty - safety_penalty

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
        return f"Transformer(name={self.name}, S={self._sn_mva}MVA, tap∈[{self._tap_min},{self._tap_max}])"
"""Charger agent - field level agent that receives price actions from coordinator."""

from typing import Any
import numpy as np
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from case_studies.power.ev_public_charging_case.features import ChargerFeature


class ChargerAgent(FieldAgent):
    """Field agent representing a charging station unit.

    Receives price actions from coordinator via protocol, updates power state
    based on price-demand response model, computes local reward (revenue).
    """

    def __init__(self, agent_id: str, p_max: float = 150.0, **kwargs):
        self._p_max = p_max
        self.current_price = 0.5
        self._upstream_action = None  # Initialize for step-mode upstream actions
        self._reward_history = []
        self._last_reward_timestamp = None
        self._last_reward = 0.0
        super().__init__(agent_id=agent_id, **kwargs)

        # Ensure feature exists (Feature/State is the single source of truth)
        if not hasattr(self, "state") or not getattr(self.state, "features", None):
            self.state.features = {}
        self.state.features.setdefault(
            "ChargerFeature",
            ChargerFeature(p_max_kw=float(p_max)),
        )

    def init_action(self, features=None):
        """Initialize action (price signal) expected from protocol."""
        action = Action()
        # price in [0.0, 0.8]
        action.set_specs(dim_c=1, range=(np.array([0.0], dtype=np.float32), np.array([0.8], dtype=np.float32)))
        action.set_values(np.array([self.current_price], dtype=np.float32))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Set action from protocol."""
        if isinstance(action, Action):
            if hasattr(self, "action") and hasattr(self.action, "dim_c") and len(action.c) != self.action.dim_c:
                self.action.set_values(action.c[: self.action.dim_c])
            else:
                self.action.set_values(c=action.c)
        elif isinstance(action, np.ndarray):
            self.action.set_values(action)
        else:
            self.action.set_values(np.array([float(action)]))

        self.current_price = float(self.action.c[0])

    def handle_self_action(self, action: Any, proxy=None):
        """Override to check for upstream actions in step mode.

        In step mode, the StationCoordinator broadcasts price actions to chargers
        by setting _upstream_action. This method checks for that before falling
        back to the default policy-based action.
        """
        import sys
        has_upstream = self._upstream_action is not None
        print(f"[handle_self_action] {self.agent_id}: action={action}, upstream={has_upstream}", file=sys.stderr, flush=True)

        if action:
            # Explicit action provided (rare for chargers)
            self.set_action(action)
        elif has_upstream:
            # Upstream action from coordinator (typical case - price signal)
            self.set_action(self._upstream_action)
            self._upstream_action = None  # Clear after use
        elif self.policy:
            # Policy-based action (if charger has its own policy)
            local_obs = proxy.get_observation(self.agent_id)
            self.set_action(self.policy.forward(observation=local_obs))
        # If none of the above, charger keeps its current price (initialized to 0.5)

        self.apply_action()

        if not self.state:
            raise ValueError("Charger state is not initialized")
        proxy.set_local_state(self.agent_id, self.state)

    def apply_action(self):
        """Update charger state.

        Charger outputs constant max power if an EV is assigned to it, 0 otherwise.
        Power is NOT affected by price action - it's determined by whether EV is present.
        The occupancy_flag is set by the environment based on EV assignment.
        """
        charger_feature = self.state.features["ChargerFeature"]

        # Read occupancy from charger's own state (should be updated by env via global_state)
        occupancy = int(getattr(charger_feature, 'occupancy_flag', 0))

        # Output max power if EV is assigned, 0 otherwise
        target_p_kw = self._p_max if occupancy > 0 else 0.0

        # Update all relevant fields
        charger_feature.set_values(
            p_kw=target_p_kw,
            occupancy_flag=occupancy,
            current_price=self.current_price
        )

        # Debug output
        import sys
        print(f"[apply_action] {self.agent_id}: occupancy={occupancy}, p_kw={target_p_kw:.2f}, price={self.current_price}", file=sys.stderr, flush=True)

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute local reward: revenue = price × power."""
        # Support both flattened local_state and nested feature dicts
        if "ChargerFeature" in local_state and isinstance(local_state["ChargerFeature"], dict):
            ls = local_state["ChargerFeature"]
        else:
            ls = local_state

        p_kw = float(ls.get("p_kw", 75.0))
        # Use price from feature if present (env might overwrite)
        price = float(ls.get("current_price", self.current_price))
        revenue = price * p_kw

        if hasattr(self, '_timestep') and hasattr(self, '_reward_history'):
            current_ts = self._timestep
            if self._last_reward_timestamp != current_ts:
                self._reward_history.append((current_ts, revenue))
                self._last_reward_timestamp = current_ts
                self._last_reward = revenue

        return float(revenue)

    def is_terminated(self, local_state: dict) -> bool:
        return False

    def is_truncated(self, local_state: dict) -> bool:
        return False

    def get_local_info(self, local_state: dict) -> dict:
        """Return debug info."""
        if "ChargerFeature" in local_state and isinstance(local_state["ChargerFeature"], dict):
            ls = local_state["ChargerFeature"]
        else:
            ls = local_state
        return {"price": float(ls.get("current_price", self.current_price)), "p_kw": float(ls.get("p_kw", 0.0))}
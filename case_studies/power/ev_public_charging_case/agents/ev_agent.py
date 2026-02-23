"""Electric vehicle agent - passive demand-side agent.

Design choice (by request): EVAgent is an *agent* (has state/features/reward/termination)
but it is *decoupled* from station control:
  - It does NOT consume coordinator actions (set_action is a no-op)
  - It does NOT make price-based decisions in apply_action

The environment (charging_env) should be responsible for assignment/queueing/charging
physics and for updating EVFeature each step.
This agent therefore only maintains a minimal autonomous lifecycle fallback:
  - If the env does not update SOC, it will drift SOC upward slowly so EVs can finish.
"""

from typing import Any

import numpy as np
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from case_studies.power.ev_public_charging_case.features import ElectricVehicleFeature


class EVAgent(FieldAgent):
    """Field agent representing an electric vehicle.

    Passive agent: exposes EV state via ElectricVehicleFeature.
    Any charging/queueing logic should be executed by the environment.
    """

    def __init__(
        self,
        agent_id: str,
        battery_capacity: float = 75.0,
        arrival_time: float = 0.0,
        **kwargs,
    ):
        self._capacity = battery_capacity
        self._arrival_time = arrival_time
        self._reward_history = []
        self._last_reward_timestamp = None
        self._last_reward = 0.0
        super().__init__(agent_id=agent_id, **kwargs)

        # Ensure feature exists (Feature/State is the single source of truth)
        if not hasattr(self, "state") or not getattr(self.state, "features", None):
            self.state.features = {}
        self.state.features.setdefault(
            "ElectricVehicleFeature",
            ElectricVehicleFeature(arrival_time=float(arrival_time)),
        )

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """EV is decoupled from station control: ignore any incoming action."""
        return

    def apply_action(self):
        """Passive default dynamics.

        Preferred: environment updates EVFeature each step.
        Fallback: if EV is present and SOC < target, drift SOC upward slowly.
        """
        ev_feature = self.state.features["ElectricVehicleFeature"]

        if ev_feature.is_present == 0:
            return  # EV has left

        # If environment isn't updating SOC, do a minimal self-contained progression.
        if ev_feature.soc < ev_feature.soc_target:
            # Small SOC increase per tick (unitless). Keep conservative.
            drift = 0.01
            ev_feature.set_values(soc=min(ev_feature.soc_target, ev_feature.soc + drift))

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute EV utility (passive).

        The env can drive accumulated_cost; this reward reads it if present.
        Reward encourages reaching target SOC and penalizes accumulated cost.
        """

        # Support both flattened local_state and nested feature dicts
        if "ElectricVehicleFeature" in local_state and isinstance(local_state["ElectricVehicleFeature"], dict):
            ls = local_state["ElectricVehicleFeature"]
        else:
            ls = local_state

        if ls.get("is_present", 1) == 0:
            reward = -1.0
        else:
            soc = float(ls.get("soc", 0.2))
            soc_target = float(ls.get("soc_target", 0.8))
            accumulated_cost = float(ls.get("accumulated_cost", 0.0))
            # Progress term: closer to target is better
            progress = -(max(0.0, soc_target - soc))
            reward = progress - 0.01 * accumulated_cost

        # Store reward history
        if hasattr(self, '_timestep') and hasattr(self, '_reward_history'):
            current_ts = self._timestep
            if self._last_reward_timestamp != current_ts:
                self._reward_history.append((current_ts, reward))
                self._last_reward_timestamp = current_ts
                self._last_reward = reward

        return float(reward)

    def is_terminated(self, local_state: dict) -> bool:
        """Terminate when EV finishes charging or leaves."""
        if "ElectricVehicleFeature" in local_state and isinstance(local_state["ElectricVehicleFeature"], dict):
            ls = local_state["ElectricVehicleFeature"]
        else:
            ls = local_state

        is_present = int(ls.get("is_present", 1))
        soc = float(ls.get("soc", 0.2))
        soc_target = float(ls.get("soc_target", 0.8))
        return is_present == 0 or soc >= soc_target

    def is_truncated(self, local_state: dict) -> bool:
        """No truncation for individual EVs."""
        return False

    def get_local_info(self, local_state: dict) -> dict:
        """Return debug info."""
        if "ElectricVehicleFeature" in local_state and isinstance(local_state["ElectricVehicleFeature"], dict):
            ls = local_state["ElectricVehicleFeature"]
        else:
            ls = local_state
        return {
            "soc": float(ls.get("soc", 0.0)),
            "soc_target": float(ls.get("soc_target", 0.8)),
            "cost": float(ls.get("accumulated_cost", 0.0)),
            "is_present": int(ls.get("is_present", 1)),
            "arrival_time": float(ls.get("arrival_time", self._arrival_time)),
        }
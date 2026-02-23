"""Station coordinator agent - Level 2 coordinator.

Changes in this version (robust for step-mode):
- set_action() broadcasts station price to charger subordinates via `_upstream_action`
- apply_action() ALSO broadcasts (covers cases where framework bypasses set_action ordering)
- observe() returns {agent_id: Observation} (dict) as required by HERON SystemAgent.observe aggregation
- EVs are not required to be in subordinates (env can model EV as entity only)
"""

import numpy as np
from typing import Dict, Any
from gymnasium.spaces import Box

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.core.action import Action

from case_studies.power.ev_public_charging_case.protocols import PriceProtocol
from case_studies.power.ev_public_charging_case.features import ChargingStationFeature, MarketFeature
from .charger_agent import ChargerAgent
from .ev_agent import EVAgent


class StationCoordinator(CoordinatorAgent):
    """Level 2 Coordinator managing chargers at a station."""

    def __init__(self, agent_id: str, num_chargers: int = 5, init_price: float = 0.5, **kwargs):
        self._num_chargers = int(num_chargers)
        self._init_price = float(init_price)

        self._reward_history = []
        self._last_reward_timestamp = None
        self._last_reward = 0.0

        super().__init__(
            agent_id=agent_id,
            protocol=PriceProtocol(),
            **kwargs
        )

        # Ensure features exist even if base init_state created something else
        self.state.features = {
            "ChargingStationFeature": ChargingStationFeature(max_chargers=self._num_chargers),
            "MarketFeature": MarketFeature(),
        }

        # Track subordinates
        self.ev_subordinates: Dict[str, EVAgent] = {}         # optional; env can leave empty
        self.charger_subordinates: Dict[str, ChargerAgent] = {}

        # Action space: single price [0.0, 0.8]
        self.action_space = Box(0.0, 0.8, (1,), np.float32)

        # Observation space: station_feature(2) + market_feature(3) = 5
        self.observation_space = Box(-np.inf, np.inf, (5,), np.float32)

    def init_action(self, features=None):
        action = Action()
        action.set_specs(
            dim_c=1,
            range=(np.array([0.0], dtype=np.float32), np.array([0.8], dtype=np.float32)),
        )
        action.set_values(np.array([self._init_price], dtype=np.float32))
        return action

    def observe(self, global_state=None, proxy=None, *args, **kwargs):
        aggregated = []
        aggregated.extend(self.state.features["ChargingStationFeature"].vector())
        aggregated.extend(self.state.features["MarketFeature"].vector())
        aggregated_vec = np.array(aggregated, dtype=np.float32)

        obs_obj = Observation(timestamp=self._timestep, local={"state": aggregated_vec})
        return {self.agent_id: obs_obj}

    def _broadcast_price_to_chargers(self, price: float) -> None:
        """Broadcast coordinator price to charger subordinates as upstream_action.

        This is the *FieldAgent has no policy* pathway: chargers act only if they
        receive an upstream action.
        """
        price = float(np.clip(price, 0.0, 0.8))

        for cid, ch in self.charger_subordinates.items():
            a = Action()
            a.set_specs(
                dim_c=1,
                range=(np.array([0.0], dtype=np.float32), np.array([0.8], dtype=np.float32)),
            )
            a.set_values(np.array([price], dtype=np.float32))

            # Set upstream action for charger to use in handle_self_action
            ch._upstream_action = a

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Receive action from env/system agent."""

        if isinstance(action, Action):
            # compatible with HERON Action
            if hasattr(action, "c"):
                self.action.set_values(np.array(action.c, dtype=np.float32))
            else:
                self.action.set_values(action)
        elif isinstance(action, np.ndarray):
            self.action.set_values(action.astype(np.float32))
        elif action is not None:
            self.action.set_values(np.array([float(action)], dtype=np.float32))

        # Immediately push station price downstream for step-mode
        if getattr(self, "action", None) is not None and hasattr(self.action, "c") and len(self.action.c) > 0:
            self._broadcast_price_to_chargers(float(self.action.c[0]))

    def apply_action(self):
        """Apply station action to local station feature AND broadcast downstream."""
        if getattr(self, "action", None) is None or not hasattr(self.action, "c") or len(self.action.c) == 0:
            return
        price = float(self.action.c[0])
        self.state.features["ChargingStationFeature"].set_values(charging_price=price)

        # Extra safety: broadcast here too (covers ordering differences in framework)
        self._broadcast_price_to_chargers(price)

    def compute_local_reward(self, local_state: dict) -> float:
        total_reward = 0.0
        for subordinate in self.subordinates.values():
            if hasattr(subordinate, "_last_reward"):
                total_reward += subordinate._last_reward

        if hasattr(self, "_timestep") and hasattr(self, "_reward_history"):
            current_ts = self._timestep
            if self._last_reward_timestamp != current_ts:
                self._reward_history.append((current_ts, total_reward))
                self._last_reward_timestamp = current_ts
                self._last_reward = total_reward

        return float(total_reward)

    def is_terminated(self, local_state: dict) -> bool:
        return len(self.charger_subordinates) == 0

    def is_truncated(self, local_state: dict) -> bool:
        return False

    def get_local_info(self, local_state: dict) -> dict:
        return {
            "num_chargers": len(self.charger_subordinates),
            "num_evs": len(self.ev_subordinates),
            "price": float(self.action.c[0]) if getattr(self, "action", None) is not None else 0.0,
        }

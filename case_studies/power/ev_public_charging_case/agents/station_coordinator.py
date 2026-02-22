"""Station coordinator agent.

Manages a fixed pool of ChargingSlot subordinates and makes pricing decisions.
Follows the same pattern as powergrid's PowerGridAgent(CoordinatorAgent).
"""

from typing import Dict, List, Optional

import numpy as np
from gymnasium.spaces import Box

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.feature import FeatureProvider
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.protocols.vertical import BroadcastActionProtocol, VerticalProtocol
from heron.scheduling.tick_config import TickConfig
from heron.utils.typing import AgentID

from case_studies.power.ev_public_charging_case.features import ChargingStationFeature, MarketFeature
from .charging_slot import ChargingSlot


class StationCoordinator(CoordinatorAgent):
    """Coordinator for a single charging station with fixed charger slots.

    Observes: ChargingStationFeature (2D) + MarketFeature (3D) = 5D observation
    Action: 1D continuous pricing decision in [0, 0.8] $/kWh
    Reward: aggregate subordinate slot rewards
    """

    def __init__(
        self,
        agent_id: AgentID,
        subordinates: Dict[AgentID, ChargingSlot],
        features: Optional[List[FeatureProvider]] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        tick_config: Optional[TickConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        if not subordinates:
            raise ValueError(
                "StationCoordinator requires subordinates (ChargingSlot agents). "
                "Create slots externally and pass as subordinates dict."
            )

        default_features = [
            ChargingStationFeature(max_chargers=len(subordinates), open_chargers=len(subordinates)),
            MarketFeature(),
        ]
        all_features = (features or []) + default_features

        super().__init__(
            agent_id=agent_id,
            features=all_features,
            subordinates=subordinates,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
            policy=policy,
            protocol=protocol or VerticalProtocol(
                action_protocol=BroadcastActionProtocol(),
            ),
        )

        # Observation: ChargingStationFeature(2) + MarketFeature(3) = 5
        self.observation_space = Box(-np.inf, np.inf, (5,), np.float32)
        # Action: pricing in [0, 0.8] $/kWh
        self.action_space = Box(0.0, 0.8, (1,), np.float32)

    @property
    def charging_slots(self) -> Dict[AgentID, ChargingSlot]:
        """Alias for subordinates."""
        return self.subordinates

    def compute_rewards(self, proxy) -> Dict[AgentID, float]:
        """Compute rewards for coordinator and all subordinate slots.

        Overrides base class to aggregate subordinate rewards into the
        coordinator reward. This works in both training mode (synchronous
        execute()) and event-driven mode, unlike the _tick_results approach
        which only works in event-driven mode.
        """
        # First compute all subordinate rewards
        sub_rewards: Dict[AgentID, float] = {}
        for subordinate in self.subordinates.values():
            sub_rewards.update(subordinate.compute_rewards(proxy))

        # Coordinator reward = sum of subordinate rewards
        coordinator_reward = sum(sub_rewards.values())

        rewards = {self.agent_id: coordinator_reward}
        rewards.update(sub_rewards)
        return rewards

    def compute_local_reward(self, local_state: dict) -> float:
        """Fallback for event-driven mode where _tick_results is populated."""
        subordinate_rewards = local_state.get("subordinate_rewards", {})
        return sum(subordinate_rewards.values())

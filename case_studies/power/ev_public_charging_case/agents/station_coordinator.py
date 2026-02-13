"""Station coordinator agent."""

import numpy as np
from typing import Dict
from gymnasium.spaces import Box

from heron.agents.coordinator_agent import CoordinatorAgent, COORDINATOR_LEVEL
from heron.core.state import CoordinatorAgentState
from heron.protocols.vertical import SetpointProtocol
from case_studies.power.ev_public_charging_case.features import ChargingStationFeature, MarketFeature
from .charger_agent import ChargerAgent
from .ev_agent import EVAgent


class StationCoordinator(CoordinatorAgent):
    def __init__(self, agent_id: str, num_chargers: int = 5, **kwargs):
        self._num_chargers = num_chargers
        # Must initialize state before super().__init__ as per CoordinatorAgent base class
        self.state = CoordinatorAgentState(owner_id=agent_id, owner_level=COORDINATOR_LEVEL,
                                           features=[ChargingStationFeature(max_chargers=num_chargers),
                                                     MarketFeature()])
        super().__init__(agent_id=agent_id, config={'num_chargers': num_chargers}, protocol=SetpointProtocol(),
                         **kwargs)
        self.ev_subordinates: Dict[str, EVAgent] = {}
        self.action_space = Box(0.0, 0.8, (1,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (5,), np.float32)

    def _build_subordinate_agents(self, agent_configs, env_id=None, upstream_id=None):
        chargers = {}
        for i in range(self._num_chargers):
            c_id = f"{upstream_id}_c{i}"
            chargers[c_id] = ChargerAgent(c_id, upstream_id=upstream_id)
        return chargers

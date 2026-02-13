"""Electric vehicle agent."""

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.core.observation import Observation
from case_studies.power.ev_public_charging_case.features import ElectricVehicleFeature

class EVAgent(FieldAgent):
    def __init__(self, agent_id: str, battery_capacity: float = 75.0, arrival_time: float = 0.0, **kwargs):
        self._capacity = battery_capacity
        self._arrival_time = arrival_time
        super().__init__(agent_id=agent_id, **kwargs)

    def set_state(self):
        """HERON Hook: Initialize Feature before base class samples observations"""
        initial_soc = np.random.uniform(0.1, 0.3)
        price_sensitivity = np.random.uniform(0.2, 0.8)
        self.state.features = [ElectricVehicleFeature(
            soc=initial_soc,
            arrival_time=self._arrival_time,
            price_sensitivity=price_sensitivity
        )]

    def observe(self, global_state=None, *args, **kwargs) -> Observation:
        return Observation(timestamp=self._timestep, local={'state': self.state.vector()})

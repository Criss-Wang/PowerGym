"""Charger agent."""

from heron.agents.field_agent import FieldAgent
from heron.core.observation import Observation
from case_studies.power.ev_public_charging_case.features import ChargerFeature


class ChargerAgent(FieldAgent):
    def __init__(self, agent_id: str, p_max: float = 150.0, **kwargs):
        self._p_max = p_max
        super().__init__(agent_id=agent_id, **kwargs)

    def set_state(self):
        """HERON Hook: Initialize Feature before base class samples observations"""
        self.state.features = [ChargerFeature(p_max_kw=self._p_max)]

    def observe(self, global_state=None, *args, **kwargs) -> Observation:
        return Observation(timestamp=self._timestep, local={'state': self.state.vector()})
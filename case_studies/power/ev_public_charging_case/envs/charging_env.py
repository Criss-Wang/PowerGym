"""Multi-station EV charging environment using HERON BaseEnv.

Follows the same pattern as powergrid/envs/hierarchical_microgrid_env.py:
- Extends BaseEnv
- Implements the 3 abstract simulation methods
- Receives coordinator_agents, BaseEnv auto-creates SystemAgent
- CTDE training via system_agent.execute() → layer_actions → act → simulate
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from heron.envs.base import BaseEnv
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.utils.typing import AgentID, MultiAgentDict

from .common import ChargerState, EnvState
from .ev import EV
from .senario import EVTrafficDemandModel


class ChargingEnv(BaseEnv):
    """Multi-station EV public charging environment."""

    def __init__(self, coordinator_agents: Dict[AgentID, CoordinatorAgent],
                 arrival_rate: float = 10.0,
                 dt: float = 300.0,
                 episode_length: float = 86400.0,
                 env_id: str = "ev_charging_env",
                 seed: Optional[int] = None,
                 **kwargs):
        super().__init__(env_id,
                         coordinator_agents)
        self.dt = float(dt)
        self.episode_length = float(episode_length)
        self._arrival_rate = float(arrival_rate)

        self._time_s = 0.0
        self._max_wait_time_s = 3600.0
        self._charger_agent_evs = EVTrafficDemandModel(arrival_rate_scale=arrival_rate)



    # ============================================
    # Lifecycle overrides
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset environment state for a new episode."""
    pass

    def step(self, actions: Dict[AgentID, Any]) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict],
    ]:
        """Execute one step and add __all__ + episode truncation."""
        obs, rewards, terminated, truncated, infos = super().step(actions)

        any_terminated = any(v for k, v in terminated.items() if k != "__all__")
        terminated["__all__"] = any_terminated

        time_up = self._time_s >= self.episode_length
        truncated["__all__"] = time_up

        # Ensure all agent keys are present in terminated/truncated/infos for RLlib stability
        for aid in obs.keys():
            terminated.setdefault(aid, False)
            truncated.setdefault(aid, False)
            infos.setdefault(aid, {})
        infos.setdefault("__all__", {})

        return obs, rewards, terminated, truncated, infos


    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> EnvState:
        """Extract simulation inputs from proxy global state."""
        pass

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Run one step of EV charging simulation.

        ENV DYNAMICS ONLY:
        - Market price (LMP) evolution
        - EV arrivals (uncontrolled random process)
        - EV assignment to available chargers (random)
        - Charging physics: power delivery, energy transfer
        - EV departures based on physical constraints (demand met, max wait time)

        Economics are handled by agents, not by env.
        """
        pass

    def env_state_to_global_state(self, env_state: EnvState) -> Dict[str, Any]:
        """Convert simulation results back to proxy global state format.

        CRITICAL: Only serialize scalar features and dicts/lists of scalars.
        DO NOT include EV object references in global_state - those are environment-internal only.
        """
        pass
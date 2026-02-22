"""Multi-station EV charging environment using HERON MultiAgentEnv.

Follows the same pattern as powergrid/envs/hierarchical_microgrid_env.py:
- Extends MultiAgentEnv (which extends EnvCore)
- Implements the 3 abstract simulation methods
- Receives coordinator_agents, EnvCore auto-creates SystemAgent
- CTDE training via system_agent.execute() → layer_actions → act → simulate
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from heron.envs.base import MultiAgentEnv
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.utils.typing import AgentID, MultiAgentDict

from case_studies.power.ev_public_charging_case.envs.common import EnvState, SlotState
from case_studies.power.ev_public_charging_case.envs.market_scenario import MarketScenario


class ChargingEnv(MultiAgentEnv):
    """Multi-station EV public charging environment.

    Agent hierarchy (auto-built by EnvCore):
        SystemAgent (auto-created)
         ├── StationCoordinator_0 (L2) — pricing decisions
         │    ├── ChargingSlot_0_0 (L1) — charger + EV slot
         │    └── ...
         └── StationCoordinator_1 (L2)
              └── ...

    Simulation:
        Each step: coordinator sets price → charger slots set power →
        run_simulation handles EV arrivals/departures/charging → update states
    """

    def __init__(
        self,
        coordinator_agents: List[CoordinatorAgent],
        arrival_rate: float = 10.0,
        dt: float = 300.0,
        episode_length: float = 86400.0,
        env_id: str = "ev_charging_env",
        **kwargs,
    ):
        self.dt = dt
        self.episode_length = episode_length
        self._arrival_rate = arrival_rate
        self.scenario = MarketScenario(arrival_rate, 3600.0)
        self._time_s = 0.0

        # Build slot → station mapping from coordinator subordinates
        self._slot_to_station: Dict[str, str] = {}
        for coord in coordinator_agents:
            for slot_id in coord.subordinates:
                self._slot_to_station[slot_id] = coord.agent_id

        # Call parent: creates SystemAgent internally from coordinator_agents,
        # sets up proxy, message broker, scheduler
        super().__init__(
            coordinator_agents=coordinator_agents,
            env_id=env_id,
            **kwargs,
        )

    # ============================================
    # Lifecycle overrides
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset environment state for a new episode.

        Reinitializes the market scenario and simulation clock before
        delegating to the parent class (which resets agents and proxy).
        """
        self.scenario = MarketScenario(self._arrival_rate, 3600.0)
        self._time_s = 0.0
        return super().reset(seed=seed, **kwargs)

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

        return obs, rewards, terminated, truncated, infos

    # ============================================
    # Abstract simulation methods (required by EnvCore)
    # ============================================

    def pre_step(self) -> None:
        """Advance market scenario clock (called at start of each step)."""
        pass  # Market update is done inside run_simulation

    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> EnvState:
        """Extract simulation inputs from proxy global state.

        Args:
            global_state: {"agent_states": {agent_id: state_dict_with_metadata}}
                where state_dict has _owner_id, _owner_level, _state_type, features: {...}

        Returns:
            EnvState with slot states, station prices, and market info
        """
        agent_states = global_state.get("agent_states", {})
        env_state = EnvState(
            slot_to_station=dict(self._slot_to_station),
            dt=self.dt,
            time_s=self._time_s,
        )

        for agent_id, state_dict in agent_states.items():
            features = state_dict.get("features", state_dict)

            # Coordinator agent → extract pricing
            if "ChargingStationFeature" in features:
                csf = features["ChargingStationFeature"]
                env_state.station_prices[agent_id] = csf.get("charging_price", 0.25)

            # Charging slot → extract charger + EV slot state
            if "ChargerFeature" in features and "EVSlotFeature" in features:
                cf = features["ChargerFeature"]
                ef = features["EVSlotFeature"]
                env_state.slot_states[agent_id] = SlotState(
                    p_kw=cf.get("p_kw", 0.0),
                    p_max_kw=cf.get("p_max_kw", 150.0),
                    open_or_not=cf.get("open_or_not", 1),
                    occupied=ef.get("occupied", 0),
                    soc=ef.get("soc", 0.0),
                    soc_target=ef.get("soc_target", 0.8),
                    arrival_time=ef.get("arrival_time", 0.0),
                    max_wait_time=ef.get("max_wait_time", 3600.0),
                    price_sensitivity=ef.get("price_sensitivity", 0.5),
                )

        return env_state

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Run one step of EV charging simulation.

        1. Advance market (LMP, time)
        2. EV arrivals → assign to empty slots
        3. Charging physics → update SOC
        4. EV departures → free completed slots
        5. Compute per-slot revenue
        """
        # 1. Advance market
        scenario_data = self.scenario.step(self.dt)
        self._time_s = scenario_data["t"]
        env_state.lmp = scenario_data["lmp"]
        env_state.time_s = scenario_data["t"]
        env_state.new_arrivals = scenario_data["arrivals"]

        # 2. EV arrivals — assign to random empty slots
        empty_slots = [
            sid for sid, ss in env_state.slot_states.items()
            if ss.occupied == 0 and ss.open_or_not == 1
        ]
        num_to_assign = min(env_state.new_arrivals, len(empty_slots))
        if num_to_assign > 0:
            chosen = np.random.choice(empty_slots, size=num_to_assign, replace=False)
            for slot_id in chosen:
                ss = env_state.slot_states[slot_id]
                ss.occupied = 1
                ss.soc = float(np.random.uniform(0.1, 0.3))
                ss.soc_target = float(np.random.uniform(0.7, 0.95))
                ss.price_sensitivity = float(np.random.uniform(0.2, 0.8))
                ss.arrival_time = env_state.time_s

        # 3. Charging physics — update SOC for occupied slots
        # p_kw was already set by each slot's apply_action() (price-responsive).
        # The simulation trusts the slot's power setpoint and computes energy/SOC.
        for slot_id, ss in env_state.slot_states.items():
            ss.revenue = 0.0
            if ss.occupied == 0:
                ss.p_kw = 0.0
                continue

            station_id = env_state.slot_to_station.get(slot_id)
            price = env_state.station_prices.get(station_id, 0.25)

            p_charge = min(ss.p_kw, ss.p_max_kw)
            ss.p_kw = p_charge
            energy_kwh = p_charge * self.dt / 3600.0

            if energy_kwh > 0 and ss.soc < ss.soc_target:
                battery_kwh = 75.0
                delta_soc = energy_kwh / battery_kwh
                ss.soc = min(1.0, ss.soc + delta_soc)
                ss.revenue = (price - env_state.lmp) * energy_kwh

        # 4. EV departures — slots where SOC >= target or max wait exceeded
        for slot_id, ss in env_state.slot_states.items():
            if ss.occupied == 0:
                continue
            time_connected = env_state.time_s - ss.arrival_time
            if ss.soc >= ss.soc_target or time_connected > ss.max_wait_time:
                ss.occupied = 0
                ss.soc = 0.0
                ss.p_kw = 0.0

        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict[str, Any]:
        """Convert simulation results back to proxy global state format.

        Must include metadata (_owner_id, _owner_level, _state_type) so that
        State.from_dict() reconstructs proper State objects with correct
        ownership for visibility-based observation filtering.

        Returns:
            {"agent_states": {agent_id: {metadata + features: {...}}}}
        """
        agent_states: Dict[str, Any] = {}

        # Update slot agent states (FieldAgent, level 1)
        for slot_id, ss in env_state.slot_states.items():
            agent_states[slot_id] = {
                "_owner_id": slot_id,
                "_owner_level": 1,
                "_state_type": "FieldAgentState",
                "features": {
                    "ChargerFeature": {
                        "p_kw": ss.p_kw,
                        "p_max_kw": ss.p_max_kw,
                        "open_or_not": ss.open_or_not,
                    },
                    "EVSlotFeature": {
                        "occupied": ss.occupied,
                        "soc": ss.soc,
                        "soc_target": ss.soc_target,
                        "arrival_time": ss.arrival_time,
                        "max_wait_time": ss.max_wait_time,
                        "price_sensitivity": ss.price_sensitivity,
                    },
                },
            }

        # Update coordinator agent states (level 2)
        for station_id, price in env_state.station_prices.items():
            # Count open chargers for this station
            station_slots = [
                sid for sid, st in env_state.slot_to_station.items()
                if st == station_id
            ]
            open_count = sum(
                1 for sid in station_slots
                if sid in env_state.slot_states and env_state.slot_states[sid].occupied == 0
            )
            agent_states[station_id] = {
                "_owner_id": station_id,
                "_owner_level": 2,
                "_state_type": "CoordinatorAgentState",
                "features": {
                    "ChargingStationFeature": {
                        "charging_price": price,
                        "open_chargers": open_count,
                    },
                    "MarketFeature": {
                        "lmp": env_state.lmp,
                        "t_day_s": env_state.time_s,
                    },
                },
            }

        return {"agent_states": agent_states}

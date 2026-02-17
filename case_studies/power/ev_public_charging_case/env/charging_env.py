"""Multi-station EV charging environment with robust RLlib integration."""

import numpy as np
from typing import Dict, Tuple, Any, Optional

from heron.envs.adapters import PettingZooParallelEnv
from case_studies.power.ev_public_charging_case.agents import StationCoordinator, EVAgent
from case_studies.power.ev_public_charging_case.env.market_scenario import MarketScenario


class SimpleChargingEnv(PettingZooParallelEnv):
    def __init__(self, arrival_rate=10.0, dt=300.0, num_stations=2):
        super().__init__(env_id="multi_station_charging_game")
        self.dt, self.scenario = dt, MarketScenario(arrival_rate, 3600.0)

        self.stations = {}
        station_ids = [f"station_{i}" for i in range(num_stations)]

        for s_id in station_ids:
            station = StationCoordinator(s_id, num_chargers=5)
            self.stations[s_id] = station

            # Register agents to HERON core
            self.register_agent(station)
            for c in station.subordinate_agents.values():
                # Although an L1 agent is registered, it will not be included in the RL's agent list.
                self.register_agent(c)

        # Configure only the station as an agent controlled by the RL algorithm.
        self._set_agent_ids(station_ids)
        self.init_spaces()
        self.total_ev_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[
        Dict[str, np.ndarray], Dict[str, dict]]:
        """Overridden reset - clean and simple."""
        obs_dict_processed, infos = super().reset(seed=seed, options=options)

        obs_dict = {}
        for aid in self.agents:
            obs = obs_dict_processed.get(aid)

            if obs is None or (isinstance(obs, np.ndarray) and obs.size == 0):
                obs_dict[aid] = np.zeros(5, dtype=np.float32)
            else:
                obs_dict[aid] = obs

        return obs_dict, {aid: {} for aid in self.agents}


    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict]
    ]:
        """Multi-station step with robust observation filtering."""

        observations_before = self.core.get_observations()

        self.core.apply_actions(actions, observations=observations_before)

        scenario_data = self.scenario.step(self.dt)
        lmp = scenario_data["lmp"]

        new_arrivals = scenario_data["arrivals"]
        for _ in range(new_arrivals):
            ev_id = f"ev_{self.total_ev_count}"
            self.total_ev_count += 1
            target_station_id = np.random.choice(list(self.stations.keys()))
            new_ev = EVAgent(ev_id, upstream_id=target_station_id)
            self.stations[target_station_id].ev_subordinates[ev_id] = new_ev

        total_rewards = {}
        for s_id, station in self.stations.items():
            s_action = actions.get(s_id)
            price = float(s_action[0]) if s_action is not None else 0.25

            station.state.update_feature("ChargingStationFeature", charging_price=price)
            station.state.update_feature("MarketFeature", lmp=lmp, t_day_s=scenario_data["t"])

            current_profit = 0.0
            for ev in station.ev_subordinates.values():
                ev_feat = next((f for f in ev.state.features if f.feature_name == "ElectricVehicleFeature"), None)
                if ev_feat and price < 0.6:
                    p_charge = 50.0
                    energy_kwh = (p_charge * self.dt / 3600.0)
                    ev_feat.soc = min(1.0, ev_feat.soc + energy_kwh / ev._capacity)
                    current_profit += (price - lmp) * energy_kwh

            total_rewards[s_id] = current_profit

            station.ev_subordinates = {
                k: v for k, v in station.ev_subordinates.items()
                if next((f.soc for f in v.state.features if f.feature_name == "ElectricVehicleFeature"), 0.0) < 0.95
            }

        new_raw_obs = self.core.get_observations()
        obs_dict = {}
        for aid in self.agents:
            if aid in new_raw_obs:
                obs = self._to_np_obs(new_raw_obs[aid].local)
                obs_dict[aid] = obs if obs.size > 0 else np.zeros(5, dtype=np.float32)
            else:
                obs_dict[aid] = np.zeros(5, dtype=np.float32)

        terminations = {aid: False for aid in self.agents}
        truncations = {aid: scenario_data["t"] > 86400 for aid in self.agents}

        infos = {
            aid: {
                "lmp": lmp,
                "price": float(actions[aid][0]) if aid in actions else 0.25,
                "evs_count": len(self.stations[aid].ev_subordinates)
            }
            for aid in self.agents
        }

        return obs_dict, total_rewards, terminations, truncations, infos
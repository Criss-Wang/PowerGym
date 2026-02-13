"""Multi-station EV charging environment."""

import numpy as np
from typing import Dict, Tuple

from heron.envs.adapters import PettingZooParallelEnv
from case_studies.power.ev_public_charging_case.agents import StationCoordinator, EVAgent
from case_studies.power.ev_public_charging_case.env.market_scenario import MarketScenario


class SimpleChargingEnv(PettingZooParallelEnv):
    def __init__(self, arrival_rate=10.0, dt=300.0, num_stations=2):
        super().__init__(env_id="multi_station_charging_game")
        self.dt, self.scenario = dt, MarketScenario(arrival_rate, 3600.0)

        # 1. Create multiple stations
        self.stations = {}
        station_ids = [f"station_{i}" for i in range(num_stations)]

        for s_id in station_ids:
            station = StationCoordinator(s_id, num_chargers=5)
            self.stations[s_id] = station

            # 2. Register the station and its fixed chargers
            self.register_agent(station)
            for c in station.subordinate_agents.values():
                self.register_agent(c)

        # 3. Inform PettingZoo that all stations are RL agents
        self._set_agent_ids(station_ids)
        self.init_spaces()
        self.total_ev_count = 0

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict]
    ]:
        """Multi-station step following HERON CTDE pattern."""

        # HERON CTDE Pattern Step 1: Collect observations BEFORE applying actions
        observations = self.core.get_observations()

        # HERON CTDE Pattern Step 2: Apply actions WITH observations
        self.core.apply_actions(actions, observations=observations)

        # A. Update Market Scenario (shared by all stations)
        scenario_data = self.scenario.step(self.dt)
        lmp = scenario_data["lmp"]

        # B. Distribute new arrivals between stations
        new_arrivals = scenario_data["arrivals"]
        for _ in range(new_arrivals):
            ev_id = f"ev_{self.total_ev_count}"
            self.total_ev_count += 1

            # Randomly assign EV to one of the stations
            target_station_id = np.random.choice(list(self.stations.keys()))
            target_station = self.stations[target_station_id]

            new_ev = EVAgent(ev_id, upstream_id=target_station_id)
            target_station.ev_subordinates[ev_id] = new_ev

        # C. Process each station's logic
        total_rewards = {}
        for s_id, station in self.stations.items():
            # Get action for this specific station
            s_action = actions.get(s_id)
            price = float(s_action[0]) if s_action is not None else 0.25

            # Update local features
            station.state.update_feature("ChargingStationFeature", charging_price=price)
            station.state.update_feature("MarketFeature", lmp=lmp, t_day_s=scenario_data["t"])

            # Run the local charging game for this station
            current_profit = 0.0
            for ev in station.ev_subordinates.values():
                ev_feat = next((f for f in ev.state.features if f.feature_name == "ElectricVehicleFeature"), None)
                if ev_feat and price < 0.6:
                    p_charge = 50.0
                    energy_kwh = (p_charge * self.dt / 3600.0)
                    ev_feat.soc = min(1.0, ev_feat.soc + energy_kwh / ev._capacity)
                    current_profit += (price - lmp) * energy_kwh

            total_rewards[s_id] = current_profit

            # Cleanup finished EVs for this station
            station.ev_subordinates = {
                k: v for k, v in station.ev_subordinates.items()
                if next((f.soc for f in v.state.features if f.feature_name == "ElectricVehicleFeature"), 0.0) < 0.95
            }

        # HERON CTDE Pattern Step 3: Collect NEW observations after state changes
        observations = self.core.get_observations()
        obs_dict = {aid: self._to_np_obs(observations[aid].local) for aid in self.agents}

        # D. Standard PettingZoo Return
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: scenario_data["t"] > 86400 for aid in self.agents}
        infos = {
            aid: {
                "lmp": lmp,
                "price": float(actions[aid][0]) if aid in actions else 0.25,
                "evs": len(self.stations[aid].ev_subordinates)
            }
            for aid in self.agents
        }

        return obs_dict, total_rewards, terminations, truncations, infos

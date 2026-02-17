import warnings

warnings.filterwarnings('ignore')

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from gymnasium.spaces import Box

from heron.core.feature import FeatureProvider
from heron.core.state import FieldAgentState, CoordinatorAgentState
from heron.core.observation import Observation
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent, COORDINATOR_LEVEL
from heron.protocols.vertical import SetpointProtocol
from heron.envs.adapters import PettingZooParallelEnv


# # ============================================
# # helper function
# # ============================================
# def _safe_div(a: float, b: float) -> float:
#     return float(a) / float(max(b, 1e-6))
#
#
# def _norm01(x: float, lo: float, hi: float) -> float:
#     if hi - lo <= 1e-6: return 0.0
#     return float(np.clip((float(x) - lo) / (hi - lo), 0.0, 1.0))
#
#
# # ============================================
# # Features
# # ============================================
# @dataclass
# class ChargerFeature(FeatureProvider):
#     visibility = ['owner', 'upper_level']
#     p_kw: float = 0.0
#     p_max_kw: float = 150.0
#     occupancy_flag: int = 0
#     open_or_not: int = 1
#
#     def vector(self) -> np.ndarray:
#         return np.array([_safe_div(self.p_kw, self.p_max_kw), float(self.occupancy_flag), float(self.open_or_not)],
#                         dtype=np.float32)
#
#     def names(self):
#         return ['p_norm', 'occupied', 'open']
#
#     def to_dict(self):
#         return {'p_kw': self.p_kw, 'p_max_kw': self.p_max_kw, 'occupancy_flag': self.occupancy_flag,
#                 'open_or_not': self.open_or_not}
#
#     @classmethod
#     def from_dict(cls, d):
#         return cls(**d)
#
#     def set_values(self, **kw):
#         if 'p_kw' in kw: self.p_kw = float(kw['p_kw'])
#         if 'occupancy_flag' in kw: self.occupancy_flag = int(np.clip(kw['occupancy_flag'], 0, 1))
#
#
# @dataclass
# class ChargingStationFeature(FeatureProvider):
#     visibility = ['owner', 'upper_level', 'system']
#     open_chargers: int = 5
#     max_chargers: int = 5
#     charging_price: float = 0.25
#     price_range: Tuple[float, float] = (0.0, 0.8)
#
#     def vector(self) -> np.ndarray:
#         return np.array([_safe_div(self.open_chargers, self.max_chargers),
#                          _norm01(self.charging_price, self.price_range[0], self.price_range[1])], dtype=np.float32)
#
#     def names(self): return ['open_norm', 'price_norm']
#
#     def to_dict(self): return {'open_chargers': self.open_chargers, 'charging_price': self.charging_price}
#
#     @classmethod
#     def from_dict(cls, d): return cls(**d)
#
#     def set_values(self, **kw):
#         if 'charging_price' in kw: self.charging_price = float(kw['charging_price'])
#         if 'open_chargers' in kw: self.open_chargers = int(kw['open_chargers'])
#         if 'current_load_kw' in kw: self.current_load_kw = float(kw['current_load_kw'])
#         if 'evs_waiting' in kw: self.evs_waiting = int(kw['evs_waiting'])
#
#
# @dataclass
# class ElectricVehicleFeature(FeatureProvider):
#     visibility = ['owner', 'upper_level']
#     soc: float = 0.2
#     soc_target: float = 0.8
#     arrival_time: float = 0.0  # Time when EV arrived (seconds)
#     max_wait_time: float = 3600.0  # Maximum willing to wait (seconds)
#     price_sensitivity: float = 0.5  # 0-1, higher = more sensitive to price
#     preferred_station_id: Optional[str] = None  # Preferred station (can be None)
#
#     def vector(self) -> np.ndarray:
#         return np.array([
#             float(self.soc),
#             float(self.soc_target),
#             float(self.price_sensitivity)
#         ], dtype=np.float32)
#
#     def names(self): return ['soc', 'soc_target', 'price_sensitivity']
#
#     def to_dict(self):
#         return {
#             'soc': self.soc,
#             'soc_target': self.soc_target,
#             'arrival_time': self.arrival_time,
#             'max_wait_time': self.max_wait_time,
#             'price_sensitivity': self.price_sensitivity,
#             'preferred_station_id': self.preferred_station_id
#         }
#
#     @classmethod
#     def from_dict(cls, d): return cls(**d)
#
#     def set_values(self, **kw):
#         if 'soc' in kw: self.soc = float(np.clip(kw['soc'], 0.0, 1.0))
#         if 'arrival_time' in kw: self.arrival_time = float(kw['arrival_time'])
#         if 'price_sensitivity' in kw: self.price_sensitivity = float(np.clip(kw['price_sensitivity'], 0.0, 1.0))
#
#
# @dataclass
# class MarketFeature(FeatureProvider):
#     visibility = ['owner', 'upper_level', 'system']
#     lmp: float = 0.20
#     t_day_s: float = 0.0
#
#     def vector(self) -> np.ndarray:
#         theta = 2.0 * np.pi * (self.t_day_s % 86400.0) / 86400.0
#         return np.array([self.lmp, np.sin(theta), np.cos(theta)], dtype=np.float32)
#
#     def names(self):
#         return ['lmp', 't_sin', 't_cos']
#
#     def to_dict(self):
#         return {'lmp': self.lmp, 't_day_s': self.t_day_s}
#
#     @classmethod
#     def from_dict(cls, d):
#         return cls(**d)
#
#     def set_values(self, **kw):
#         if 'lmp' in kw: self.lmp = float(kw['lmp'])
#         if 't_day_s' in kw: self.t_day_s = float(kw['t_day_s'])
#
#
# # ============================================
# # Agents
# # ============================================
# class EVAgent(FieldAgent):
#     def __init__(self, agent_id: str, battery_capacity: float = 75.0, arrival_time: float = 0.0, **kwargs):
#         self._capacity = battery_capacity
#         self._arrival_time = arrival_time
#         super().__init__(agent_id=agent_id, **kwargs)
#
#     def set_state(self):
#         """HERON Hook: Initialize Feature before base class samples observations"""
#         initial_soc = np.random.uniform(0.1, 0.3)
#         price_sensitivity = np.random.uniform(0.2, 0.8)
#         self.state.features = [ElectricVehicleFeature(
#             soc=initial_soc,
#             arrival_time=self._arrival_time,
#             price_sensitivity=price_sensitivity
#         )]
#
#     def observe(self, global_state=None, *args, **kwargs) -> Observation:
#         return Observation(timestamp=self._timestep, local={'state': self.state.vector()})
#
#
# class ChargerAgent(FieldAgent):
#     def __init__(self, agent_id: str, p_max: float = 150.0, **kwargs):
#         self._p_max = p_max
#         super().__init__(agent_id=agent_id, **kwargs)
#
#     def set_state(self):
#         """HERON Hook: Initialize Feature before base class samples observations"""
#         self.state.features = [ChargerFeature(p_max_kw=self._p_max)]
#
#     def observe(self, global_state=None, *args, **kwargs) -> Observation:
#         return Observation(timestamp=self._timestep, local={'state': self.state.vector()})
#
#
# class StationCoordinator(CoordinatorAgent):
#     def __init__(self, agent_id: str, num_chargers: int = 5, **kwargs):
#         self._num_chargers = num_chargers
#         # Must initialize state before super().__init__ as per CoordinatorAgent base class
#         self.state = CoordinatorAgentState(owner_id=agent_id, owner_level=COORDINATOR_LEVEL,
#                                            features=[ChargingStationFeature(max_chargers=num_chargers),
#                                                      MarketFeature()])
#         super().__init__(agent_id=agent_id, config={'num_chargers': num_chargers}, protocol=SetpointProtocol(),
#                          **kwargs)
#         self.ev_subordinates: Dict[str, EVAgent] = {}
#         self.action_space = Box(0.0, 0.8, (1,), np.float32)
#         self.observation_space = Box(-np.inf, np.inf, (5,), np.float32)
#
#     def _build_subordinate_agents(self, agent_configs, env_id=None, upstream_id=None):
#         chargers = {}
#         for i in range(self._num_chargers):
#             c_id = f"{upstream_id}_c{i}"
#             chargers[c_id] = ChargerAgent(c_id, upstream_id=upstream_id)
#         return chargers
#
#
#
# # ============================================
# # Environment
# # ============================================
# class MarketScenario:
#     def __init__(self, arrival_rate: float, price_freq: float):
#         self.arrival_rate, self.price_freq = arrival_rate, price_freq
#         self.time_seconds, self.last_price_update = 0.0, -price_freq
#         self.current_lmp = 0.20
#
#     def step(self, dt: float):
#         self.time_seconds += dt
#         if self.time_seconds - self.last_price_update >= self.price_freq:
#             self.current_lmp = 0.2 + 0.1 * np.sin(2 * np.pi * self.time_seconds / 86400)
#             self.last_price_update = self.time_seconds
#         return {"lmp": self.current_lmp, "t": self.time_seconds,
#                 "arrivals": np.random.poisson(self.arrival_rate * dt / 3600.0)}
#
#
# class SimpleChargingEnv(PettingZooParallelEnv):
#     def __init__(self, arrival_rate=10.0, dt=300.0, num_stations=2):
#         super().__init__(env_id="multi_station_charging_game")
#         self.dt, self.scenario = dt, MarketScenario(arrival_rate, 3600.0)
#
#         # 1. Create multiple stations
#         self.stations = {}
#         station_ids = [f"station_{i}" for i in range(num_stations)]
#
#         for s_id in station_ids:
#             station = StationCoordinator(s_id, num_chargers=5)
#             self.stations[s_id] = station
#
#             # 2. Register the station and its fixed chargers
#             self.register_agent(station)
#             for c in station.subordinate_agents.values():
#                 self.register_agent(c)
#
#         # 3. Inform PettingZoo that all stations are RL agents
#         self._set_agent_ids(station_ids)
#         self.init_spaces()
#         self.total_ev_count = 0
#
#     def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
#         Dict[str, np.ndarray],
#         Dict[str, float],
#         Dict[str, bool],
#         Dict[str, bool],
#         Dict[str, dict]
#     ]:
#         """Multi-station step following HERON CTDE pattern."""
#
#         # HERON CTDE Pattern Step 1: Collect observations BEFORE applying actions
#         observations = self.core.get_observations()
#
#         # HERON CTDE Pattern Step 2: Apply actions WITH observations
#         self.core.apply_actions(actions, observations=observations)
#
#         # A. Update Market Scenario (shared by all stations)
#         scenario_data = self.scenario.step(self.dt)
#         lmp = scenario_data["lmp"]
#
#         # B. Distribute new arrivals between stations
#         new_arrivals = scenario_data["arrivals"]
#         for _ in range(new_arrivals):
#             ev_id = f"ev_{self.total_ev_count}"
#             self.total_ev_count += 1
#
#             # Randomly assign EV to one of the stations
#             target_station_id = np.random.choice(list(self.stations.keys()))
#             target_station = self.stations[target_station_id]
#
#             new_ev = EVAgent(ev_id, upstream_id=target_station_id)
#             target_station.ev_subordinates[ev_id] = new_ev
#
#         # C. Process each station's logic
#         total_rewards = {}
#         for s_id, station in self.stations.items():
#             # Get action for this specific station
#             s_action = actions.get(s_id)
#             price = float(s_action[0]) if s_action is not None else 0.25
#
#             # Update local features
#             station.state.update_feature("ChargingStationFeature", charging_price=price)
#             station.state.update_feature("MarketFeature", lmp=lmp, t_day_s=scenario_data["t"])
#
#             # Run the local charging game for this station
#             current_profit = 0.0
#             for ev in station.ev_subordinates.values():
#                 ev_feat = next((f for f in ev.state.features if f.feature_name == "ElectricVehicleFeature"), None)
#                 if ev_feat and price < 0.6:
#                     p_charge = 50.0
#                     energy_kwh = (p_charge * self.dt / 3600.0)
#                     ev_feat.soc = min(1.0, ev_feat.soc + energy_kwh / ev._capacity)
#                     current_profit += (price - lmp) * energy_kwh
#
#             total_rewards[s_id] = current_profit
#
#             # Cleanup finished EVs for this station
#             station.ev_subordinates = {
#                 k: v for k, v in station.ev_subordinates.items()
#                 if next((f.soc for f in v.state.features if f.feature_name == "ElectricVehicleFeature"), 0.0) < 0.95
#             }
#
#         # HERON CTDE Pattern Step 3: Collect NEW observations after state changes
#         observations = self.core.get_observations()
#         obs_dict = {aid: self._to_np_obs(observations[aid].local) for aid in self.agents}
#
#         # D. Standard PettingZoo Return
#         terminations = {aid: False for aid in self.agents}
#         truncations = {aid: scenario_data["t"] > 86400 for aid in self.agents}
#         infos = {
#             aid: {
#                 "lmp": lmp,
#                 "price": float(actions[aid][0]) if aid in actions else 0.25,
#                 "evs": len(self.stations[aid].ev_subordinates)
#             }
#             for aid in self.agents
#         }
#
#         return obs_dict, total_rewards, terminations, truncations, infos


# ============================================
# Main: Rollout
# ============================================
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SimpleChargingEnv

if __name__ == "__main__":
    print("Creating environment...")
    env = SimpleChargingEnv()
    print("Environment created, resetting...")
    obs, _ = env.reset()
    print(f"Reset complete, initial obs keys: {list(obs.keys())}")
    total_reward = 0
    for i in range(50):
        print(f"Step {i} starting...")
        actions = {a: env.action_space(a).sample() for a in env.agents}
        print(f"  Actions sampled: {actions}")
        obs, rewards, term, trunc, _ = env.step(actions)
        print(f"  Step complete, reward: {sum(rewards.values()):.4f}")
        total_reward += sum(rewards.values())
        if i % 10 == 0: print(f"Step {i}: Profit = {sum(rewards.values()):.4f}")
        if any(term.values()) or any(trunc.values()): break
    print(f"Total Profit: {total_reward:.4f}")
    env.close()

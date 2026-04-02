"""Multi-station EV charging environment using HERON BaseEnv.

Follows the same pattern as powergrid/envs/hierarchical_microgrid_env.py:
- Extends BaseEnv
- Implements the 3 abstract simulation methods
- Receives coordinator_agents, BaseEnv auto-creates SystemAgent
- CTDE training via system_agent.execute() → layer_actions → act → simulate
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from heron.envs.base import BaseEnv
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.utils.typing import AgentID, MultiAgentDict

from case_studies.power.ev_public_charging_case.envs.common import ChargerState, EnvState
from case_studies.power.ev_public_charging_case.envs.ev import EV
from case_studies.power.ev_public_charging_case.envs.market_scenario import MarketScenario


class ChargingEnv(BaseEnv):
    """Multi-station EV public charging environment."""

    def __init__(
        self,
        coordinator_agents: List[CoordinatorAgent],
        arrival_rate: float = 10.0,
        dt: float = 300.0,
        episode_length: float = 86400.0,
        hourly_overhead_cost: float = 3.0,
        operational_cost_per_kwh: float = 0.03,
        charging_efficiency: float = 0.95,
        env_id: str = "ev_charging_env",
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.dt = float(dt)
        self.episode_length = float(episode_length)
        self._arrival_rate = float(arrival_rate)
        self.hourly_overhead_cost = float(hourly_overhead_cost)
        self.operational_cost_per_kwh = float(operational_cost_per_kwh)
        self.charging_efficiency = float(charging_efficiency)
        if self.charging_efficiency <= 0.0:
            raise ValueError("charging_efficiency must be positive")

        self._time_s = 0.0
        self._max_wait_time_s = 3600.0

        # RNG for reproducibility (affects arrivals assignment + SOC sampling)
        self._rng = np.random.default_rng(seed)

        # External scenarios
        self.scenario = MarketScenario(self._arrival_rate, 3600.0)

        # Build charger-agent -> station mapping from coordinator subordinates.
        self._charger_agent_to_station: Dict[str, str] = {}
        # Keep EV objects across steps; global_state only carries scalar features.
        self._charger_agent_evs: Dict[str, EV] = {}
        # Keep cumulative metrics across steps for logging / diagnostics.
        self._charger_agent_cumulative_revenue: Dict[str, float] = {}
        self._charger_agent_cumulative_profit: Dict[str, float] = {}
        self._station_cumulative_revenue: Dict[str, float] = {}
        self._station_cumulative_profit: Dict[str, float] = {}
        for coord in coordinator_agents:
            self._station_cumulative_revenue[str(coord.agent_id)] = 0.0
            self._station_cumulative_profit[str(coord.agent_id)] = 0.0
            for charger_agent_id in coord.subordinates:
                self._charger_agent_to_station[str(charger_agent_id)] = str(coord.agent_id)
                self._charger_agent_cumulative_revenue[str(charger_agent_id)] = 0.0
                self._charger_agent_cumulative_profit[str(charger_agent_id)] = 0.0

        super().__init__(
            coordinator_agents=coordinator_agents,
            env_id=env_id,
            **kwargs,
        )

    # ============================================
    # Lifecycle overrides
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset environment state for a new episode."""

        self.scenario = MarketScenario(self._arrival_rate, 3600.0)

        self._time_s = 0.0
        self._charger_agent_evs.clear()
        for charger_id in self._charger_agent_cumulative_revenue:
            self._charger_agent_cumulative_revenue[charger_id] = 0.0
            self._charger_agent_cumulative_profit[charger_id] = 0.0
        for station_id in self._station_cumulative_revenue:
            self._station_cumulative_revenue[station_id] = 0.0
            self._station_cumulative_profit[station_id] = 0.0
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

        # Ensure all agent keys are present in terminated/truncated/infos for RLlib stability
        for aid in obs.keys():
            terminated.setdefault(aid, False)
            truncated.setdefault(aid, False)
            infos.setdefault(aid, {})
        infos.setdefault("__all__", {})

        return obs, rewards, terminated, truncated, infos

    # ============================================
    # Abstract simulation methods (required by BaseEnv)
    # ============================================

    def pre_step(self) -> None:
        """Advance market scenario clock (called at start of each step)."""
        return

    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> EnvState:
        """Extract simulation inputs from proxy global state."""
        agent_states = global_state.get("agent_states", {})
        env_state = EnvState(
            charger_agent_to_station=dict(self._charger_agent_to_station),
            dt=self.dt,
            time_s=self._time_s,
        )

        for agent_id, state_dict in agent_states.items():
            features = state_dict.get("features", state_dict)

            # Coordinator agent → extract pricing
            if "ChargingStationFeature" in features:
                csf = features["ChargingStationFeature"]
                env_state.station_prices[str(agent_id)] = float(csf.get("charging_price", 0.25))

            # Charger agent -> extract charger state (incl. last received/broadcast price)
            if "ChargerFeature" in features:
                cf = features["ChargerFeature"]
                charger_agent_id = str(agent_id)
                station_id = env_state.charger_agent_to_station.get(charger_agent_id)
                default_price = env_state.station_prices.get(station_id, 0.25)
                env_state.charger_states[charger_agent_id] = ChargerState(
                    p_kw=float(cf.get("p_kw", 0.0)),
                    p_max_kw=float(cf.get("p_max_kw", 150.0)),
                    occupied_or_not=int(cf.get("occupied_or_not", 0)),
                    last_received_price=float(cf.get("charging_price", default_price)),
                    ev=self._charger_agent_evs.get(charger_agent_id),
                    step_energy_delivered_kwh=0.0,
                    step_revenue=0.0,
                    step_energy_cost=0.0,
                    step_profit=0.0,
                    cumulative_revenue=self._charger_agent_cumulative_revenue.get(charger_agent_id, 0.0),
                    cumulative_profit=self._charger_agent_cumulative_profit.get(charger_agent_id, 0.0),
                )

        return env_state

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
        # 1) DYNAMICS: Advance market scenario (LMP evolution)
        scenario_data = self.scenario.step(self.dt)
        self._time_s = float(scenario_data["t"])
        env_state.lmp = float(scenario_data["lmp"])
        env_state.time_s = float(scenario_data["t"])
        env_state.new_arrivals = int(scenario_data["arrivals"])

        # 2) DYNAMICS: EV arrivals and random assignment to available chargers
        # Do NOT use pricing for EV choice — that's agent economics.
        # Simply count available chargers by station.
        if env_state.new_arrivals > 0:
            for _ in range(env_state.new_arrivals):
                ev = EV(time_arrival=env_state.time_s)
                
                # Find a random available charger across all stations (simple random matching)
                all_available_chargers = [
                    aid for aid, st in env_state.charger_agent_to_station.items()
                    if aid in env_state.charger_states
                    and env_state.charger_states[aid].ev is None
                    and env_state.charger_states[aid].occupied_or_not == 0
                ]
                
                if all_available_chargers:
                    # Use EV's choose_station function for price-responsive station selection
                    # Build station info for the EV to choose from
                    evsts_info = {}
                    for charger_id in all_available_chargers:
                        station_id = env_state.charger_agent_to_station.get(charger_id)
                        if station_id not in evsts_info:
                            # Get price and power from station/charger state
                            station_price = env_state.station_prices.get(station_id, 0.25)
                            # Get charger power capacity (assume all chargers at a station have same power)
                            station_chargers = [aid for aid, st in env_state.charger_agent_to_station.items() 
                                                if st == station_id]
                            num_chargers = len(station_chargers)
                            num_users = sum(1 for aid in station_chargers 
                                          if aid in env_state.charger_states and env_state.charger_states[aid].ev is not None)
                            charger_power = env_state.charger_states[charger_id].p_max_kw
                            
                            evsts_info[station_id] = {
                                "price": station_price,
                                "parking_fee": 3.0,  # Default parking fee ($/h)
                                "charging_power": charger_power,
                                "num_chargers": num_chargers,
                                "num_users": num_users,
                            }
                    
                    # EV chooses station based on price-responsiveness
                    selected_station, x_opt, u_opt = ev.choose_station(evsts_info)
                    
                    if selected_station is not None:
                        # Find an available charger at the selected station
                        station_chargers = [aid for aid, st in env_state.charger_agent_to_station.items() 
                                          if st == selected_station and aid in all_available_chargers]
                        if station_chargers:
                            charger_agent_id = station_chargers[0]  # Pick first available at selected station
                            
                            # Assign EV to charger and update charger agent state
                            # (EV's demand is already computed by choose_station based on utility function)
                            env_state.charger_states[charger_agent_id].ev = ev
                            self._charger_agent_evs[charger_agent_id] = ev
                            env_state.charger_states[charger_agent_id].occupied_or_not = 1



        # 3) DYNAMICS: Charging physics — power delivery and energy transfer with precise time tracking
        # Track per-step revenue/cost/profit from actual payments, with accurate time-based accounting
        for charger_agent_id, cs in env_state.charger_states.items():
            
            cs.step_energy_delivered_kwh = 0.0
            cs.step_revenue = 0.0
            cs.step_energy_cost = 0.0
            cs.step_profit = 0.0

            if cs.ev is None or cs.occupied_or_not == 0:
                cs.p_kw = 0.0
                continue

            ev = cs.ev
            
            # DYNAMICS: Charger operates at max power when EV is present
            # Agent only controls pricing, not power output
            cs.p_kw = cs.p_max_kw
            
            # DYNAMICS ONLY: If EV is present and has demand, allow power delivery.
            demand_remaining = ev.get_demand()
            
            if demand_remaining > 1e-6:
                # EV has demand and charger is at max power.
                # Execute physics of charging with precise time tracking.
                energy_kwh, payment = ev.charge_and_accumulate_revenue(cs.p_kw, env_state.time_s, self.dt)
                step_energy_cost = 0.0
                if energy_kwh > 0.0:
                    step_energy_cost = (
                        (env_state.lmp + self.operational_cost_per_kwh)
                        * float(energy_kwh)
                        / self.charging_efficiency
                    )

                cs.step_energy_delivered_kwh = float(energy_kwh)
                cs.step_revenue = float(payment)
                cs.step_energy_cost = float(step_energy_cost)
                cs.step_profit = float(payment - step_energy_cost)

                cs.cumulative_revenue += cs.step_revenue
                cs.cumulative_profit += cs.step_profit

                if cs.step_energy_delivered_kwh <= 0.0:
                    assert cs.step_energy_cost == 0.0, "energy cost must be zero when no energy is delivered"
            else:
                cs.p_kw = 0.0

        # Step accounting sanity check: no step metric should be carried across a fresh step.
        for cs in env_state.charger_states.values():
            assert cs.step_profit == cs.step_revenue - cs.step_energy_cost

        # 4) DYNAMICS: EV departures — physical constraints only (demand met or max wait time)
        for charger_agent_id, cs in env_state.charger_states.items():
            if cs.ev is None:
                continue
            ev = cs.ev
            time_connected = env_state.time_s - ev.time_arrival
            demand_done = ev.get_demand() <= 1e-6
            if demand_done or time_connected > self._max_wait_time_s:
                # Record departure time for accurate revenue tracking
                ev.time_departed = env_state.time_s
                cs.ev = None
                cs.p_kw = 0.0
                cs.occupied_or_not = 0  # Mark charger as available again
                self._charger_agent_evs.pop(charger_agent_id, None)

        # 5) DYNAMICS: Aggregate station power/capacity/profit
        # For each station: sum current step metrics and update cumulative totals.
        station_power: Dict[str, float] = {}
        station_capacity: Dict[str, float] = {}
        station_step_revenue: Dict[str, float] = {}
        station_step_energy_cost: Dict[str, float] = {}
        station_step_overhead_cost: Dict[str, float] = {}
        station_step_profit: Dict[str, float] = {}
        station_cumulative_revenue: Dict[str, float] = {}
        station_cumulative_profit: Dict[str, float] = {}
        step_overhead_cost = self.hourly_overhead_cost * (self.dt / 3600.0)
        for charger_agent_id, cs in env_state.charger_states.items():
            st = env_state.charger_agent_to_station.get(charger_agent_id)
            if st is None:
                continue
            
            # Initialize station aggregates
            if st not in station_power:
                station_power[st] = 0.0
                station_capacity[st] = 0.0
                station_step_revenue[st] = 0.0
                station_step_energy_cost[st] = 0.0
                station_step_overhead_cost[st] = step_overhead_cost
                station_step_profit[st] = 0.0
                station_cumulative_revenue[st] = self._station_cumulative_revenue.get(st, 0.0)
                station_cumulative_profit[st] = self._station_cumulative_profit.get(st, 0.0)
            
            # Add charger capacity (always available for aggregation)
            station_capacity[st] += float(cs.p_max_kw)
            
            # Add charger power (only when occupied - EV is charging)
            if cs.occupied_or_not == 1:
                station_power[st] += float(cs.p_kw)
            
            station_step_revenue[st] += float(cs.step_revenue)
            station_step_energy_cost[st] += float(cs.step_energy_cost)

        for st in station_step_revenue:
            station_step_profit[st] = station_step_revenue[st] - station_step_energy_cost[st] - station_step_overhead_cost[st]
            station_cumulative_revenue[st] = station_cumulative_revenue.get(st, 0.0) + station_step_revenue[st]
            station_cumulative_profit[st] = station_cumulative_profit.get(st, 0.0) + station_step_profit[st]

            expected_profit = station_step_revenue[st] - station_step_energy_cost[st] - station_step_overhead_cost[st]
            assert math.isclose(station_step_profit[st], expected_profit, rel_tol=1e-9, abs_tol=1e-9)

        env_state.station_power = station_power
        env_state.station_capacity = station_capacity
        env_state.station_step_revenue = station_step_revenue
        env_state.station_step_energy_cost = station_step_energy_cost
        env_state.station_step_overhead_cost = station_step_overhead_cost
        env_state.station_step_profit = station_step_profit
        env_state.station_cumulative_revenue = station_cumulative_revenue
        env_state.station_cumulative_profit = station_cumulative_profit

        # Save cumulative history for next step
        for charger_agent_id, cs in env_state.charger_states.items():
            self._charger_agent_cumulative_revenue[charger_agent_id] = cs.cumulative_revenue
            self._charger_agent_cumulative_profit[charger_agent_id] = cs.cumulative_profit

        for station_id in station_cumulative_revenue:
            self._station_cumulative_revenue[station_id] = station_cumulative_revenue[station_id]
            self._station_cumulative_profit[station_id] = station_cumulative_profit[station_id]

        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict[str, Any]:
        """Convert simulation results back to proxy global state format.
        
        CRITICAL: Only serialize scalar features and dicts/lists of scalars.
        DO NOT include EV object references in global_state - those are environment-internal only.
        """
        agent_states: Dict[str, Any] = {}

        # Update charger-agent states (FieldAgent, level 1)
        for charger_agent_id, cs in env_state.charger_states.items():
            agent_states[charger_agent_id] = {
                "_owner_id": charger_agent_id,
                "_owner_level": 1,
                "_state_type": "FieldAgentState",
                "features": {
                    "ChargerFeature": {
                        "p_kw": cs.p_kw,
                        "p_max_kw": cs.p_max_kw,
                        "occupied_or_not": cs.occupied_or_not,
                        "charging_price": float(cs.last_received_price),
                        "step_energy_delivered_kwh": float(cs.step_energy_delivered_kwh),
                        "step_revenue": float(cs.step_revenue),
                        "step_energy_cost": float(cs.step_energy_cost),
                        "step_profit": float(cs.step_profit),
                        "cumulative_revenue": float(cs.cumulative_revenue),
                        "cumulative_profit": float(cs.cumulative_profit),
                        # NOTE: cs.ev is NOT serialized (env-internal reference)
                    },
                },
            }

        # Update coordinator agent states (level 2)
        for station_id, price in env_state.station_prices.items():
            # Count open chargers for this station using internal EV occupancy.
            station_charger_agents = [aid for aid, st in env_state.charger_agent_to_station.items() if st == station_id]
            open_count = sum(
                1 for aid in station_charger_agents
                if aid in env_state.charger_states
                and env_state.charger_states[aid].ev is None
                and env_state.charger_states[aid].occupied_or_not == 0
            )
            
            station_step_revenue = env_state.station_step_revenue.get(station_id, 0.0)
            station_step_energy_cost = env_state.station_step_energy_cost.get(station_id, 0.0)
            station_step_overhead_cost = env_state.station_step_overhead_cost.get(station_id, 0.0)
            station_step_profit = env_state.station_step_profit.get(station_id, 0.0)
            station_cum_rev = env_state.station_cumulative_revenue.get(station_id, 0.0)
            station_cum_profit = env_state.station_cumulative_profit.get(station_id, 0.0)
            station_pw = env_state.station_power.get(station_id, 0.0)
            station_cap = env_state.station_capacity.get(station_id, 0.0)

            agent_states[station_id] = {
                "_owner_id": station_id,
                "_owner_level": 2,
                "_state_type": "CoordinatorAgentState",
                "features": {
                    "ChargingStationFeature": {
                        "charging_price": float(price),
                        "open_chargers": int(open_count),
                        "max_chargers": int(len(station_charger_agents)),
                        "station_step_revenue": float(station_step_revenue),
                        "station_step_energy_cost": float(station_step_energy_cost),
                        "station_step_overhead_cost": float(station_step_overhead_cost),
                        "station_step_profit": float(station_step_profit),
                        "station_cumulative_revenue": float(station_cum_rev),
                        "station_cumulative_profit": float(station_cum_profit),
                        "station_power": float(station_pw),
                        "station_capacity": float(station_cap),
                    },
                    "MarketFeature": {
                        "lmp": float(env_state.lmp),
                        "t_day_s": float(env_state.time_s),
                    }
                },
            }

        return {"agent_states": agent_states}
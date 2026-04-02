"""Station coordinator agent.

Manages a fixed pool of ChargerAgent subordinates and makes pricing decisions.
Follows the same pattern as powergrid's PowerGridAgent(CoordinatorAgent).

AGENT ECONOMICS:
- Optimizes pricing based on market conditions (LMP) and demand
- Computes EV demand/utility based on price
- Tracks station profit internally from physical observations
- Emits reward only at tick boundaries
"""

from typing import Dict, List, Optional

import numpy as np
from gymnasium.spaces import Box

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.protocols.vertical import BroadcastActionProtocol, VerticalProtocol
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID

# 引用定义的特征类
from case_studies.power.ev_public_charging_case.features import ChargingStationFeature, MarketFeature
# 引用子代理类
from case_studies.power.ev_public_charging_case.agents.charger_field_agent import ChargerAgent


class StationCoordinator(CoordinatorAgent):
    """站点协调员：管理固定数量的充电桩代理，并做出定价决策。

    观测空间: ChargingStationFeature (3D) + MarketFeature (3D) = 6D 观测
    动作空间: [0, 0.8] $/kWh 的 1D 连续定价决策
    奖励机制: tick-level profit delta / charger 数
    
    AGENT ECONOMICS:
    - Computes optimal demand for arriving EVs based on price
    - Determines utility function for EV station choice
    - Optimizes pricing strategy to maximize revenue
    - Tracks and aggregates charger revenues
    """

    def __init__(
        self,
        agent_id: AgentID,
        subordinates: Dict[AgentID, ChargerAgent],
        features: Optional[List[Feature]] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
        hourly_overhead_cost: float = 3.0,
        operational_cost_per_kwh: float = 0.03,
        charging_efficiency: float = 0.95,
    ):
        if not subordinates:
            raise ValueError(
                "StationCoordinator requires subordinates (ChargerAgent agents). "
                "Create charger agents externally and pass as subordinates dict."
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
            schedule_config=schedule_config,
            policy=policy,
            # 默认使用垂直广播协议，将价格下发给所有充电桩代理
            protocol=protocol or VerticalProtocol(
                action_protocol=BroadcastActionProtocol(),
            ),
        )

        self.hourly_overhead_cost = float(hourly_overhead_cost)
        self.operational_cost_per_kwh = float(operational_cost_per_kwh)
        self.charging_efficiency = float(charging_efficiency)
        if self.charging_efficiency <= 0.0:
            raise ValueError("charging_efficiency must be positive")

        self.current_price = 0.25
        self.cumulative_profit = 0.0
        self.last_tick_profit_anchor = 0.0
        self.tick_accumulated_profit = 0.0
        self.tick_counter = 0
        self.last_step_revenue = 0.0
        self.last_step_energy_cost = 0.0
        self.last_step_overhead_cost = 0.0
        self.last_step_profit = 0.0
        self.last_reward = 0.0
        self._last_accounting_time_s = 0.0
        self._last_reward_time_s = 0.0
        self._last_accounting_emitted_time_s: Optional[float] = None
        self._last_reward_debug: Dict[str, object] = {}
        self._proxy_ref = None

        # 观测空间定义
        # ChargingStationFeature.vector(): [open_norm, price_norm, utilization] (3)
        # MarketFeature.vector(): [lmp, sin(theta), cos(theta)] (3)
        self.observation_space = Box(-np.inf, np.inf, (6,), np.float32)
        expected_obs_dim = sum(feature.vector().shape[0] for feature in all_features)
        if expected_obs_dim != self.observation_space.shape[0]:
            raise ValueError(
                f"StationCoordinator observation_space mismatch: expected {expected_obs_dim}, "
                f"declared {self.observation_space.shape[0]}"
            )

        # 动作空间：定价范围 [0, 0.8] $/kWh
        self.action_space = Box(0.0, 0.8, (1,), np.float32)

    def reset(self, *, seed: Optional[int] = None, proxy=None, **kwargs):
        self._proxy_ref = proxy
        self.current_price = 0.25
        self.cumulative_profit = 0.0
        self.last_tick_profit_anchor = 0.0
        self.tick_accumulated_profit = 0.0
        self.tick_counter = 0
        self.last_step_revenue = 0.0
        self.last_step_energy_cost = 0.0
        self.last_step_overhead_cost = 0.0
        self.last_step_profit = 0.0
        self.last_reward = 0.0
        self._last_accounting_time_s = 0.0
        self._last_reward_time_s = 0.0
        self._last_accounting_emitted_time_s = None
        self._last_reward_debug = {}
        return super().reset(seed=seed, proxy=proxy, **kwargs)

    def post_proxy_attach(self, proxy) -> None:
        self._proxy_ref = proxy
        super().post_proxy_attach(proxy)

    @property
    def charger_agents(self) -> Dict[AgentID, ChargerAgent]:
        """subordinates 的别名，方便调用。"""
        return self.subordinates

    def set_action(self, action, *args, **kwargs) -> None:
        super().set_action(action, *args, **kwargs)
        if self.action is not None and self.action.is_valid():
            self.current_price = float(self.action.c[0])
            self.state.update_feature("ChargingStationFeature", charging_price=self.current_price)

    def compute_ev_demand_utility(self, ev_battery_capacity: float, ev_soc: float, 
                                 price: float, lmp: float, p_max_kw: float) -> tuple:
        """AGENT ECONOMICS: Compute optimal EV demand and utility based on price.
        
        This represents the agent's economic decision model for EV charging optimization.
        
        Args:
            ev_battery_capacity: Battery capacity (kWh)
            ev_soc: Initial state of charge (fraction)
            price: Charging price ($/kWh)
            lmp: Locational marginal price ($/kWh)
            p_max_kw: Charger power (kW)
        
        Returns:
            Tuple of (optimal_demand_kwh, utility)
        """
        # Simple linear demand model: lower price -> higher demand
        # This is a placeholder; replace with actual EV utility model
        demand_min = 0.2 * ev_battery_capacity
        max_demand = (1.0 - ev_soc) * ev_battery_capacity
        
        # Price elasticity: demand decreases with price
        price_factor = max(0.0, 1.0 - (price / 0.8))  # 0 at max price, 1 at zero price
        demand = demand_min + (max_demand - demand_min) * price_factor
        
        # Simple utility: higher demand at lower prices
        utility = max(0.0, (0.5 - price) * demand)
        
        return demand, utility

    def compute_pricing_strategy(self, lmp: float, occupancy_ratio: float) -> float:
        """AGENT ECONOMICS: Determine optimal pricing based on market and station state.
        
        Args:
            lmp: Locational marginal price ($/kWh)
            occupancy_ratio: Current occupancy ratio (0-1)
        
        Returns:
            Recommended price in [0, 0.8] range
        """
        # Simple strategy: price between LMP and reasonable markup
        base_price = lmp + 0.1  # 10 cent markup over LMP
        
        # Adjust based on occupancy: higher price when high occupancy
        occupancy_adjustment = occupancy_ratio * 0.2  # Up to 20 cent additional markup
        
        price = base_price + occupancy_adjustment
        price = np.clip(price, 0.0, 0.8)
        
        return float(price)

    def compute_rewards(self, proxy) -> Dict[AgentID, float]:
        self._proxy_ref = proxy
        reward = self._compute_tick_reward_from_proxy(proxy)
        rewards: Dict[AgentID, float] = {self.agent_id: float(reward)}
        for subordinate_id in self.subordinates:
            rewards[subordinate_id] = 0.0
        return rewards

    def _tick_interval_s(self) -> float:
        schedule = getattr(self, "schedule_config", None)
        return float(getattr(schedule, "tick_interval", 0.0) or 0.0)

    def _get_sim_agent_states(self, proxy) -> Dict[str, dict]:
        if proxy is None:
            return {}
        global_state = proxy.get_global_states(self.agent_id, self.protocol, for_simulation=True)
        if not isinstance(global_state, dict):
            return {}
        nested = global_state.get("agent_states")
        if isinstance(nested, dict):
            return nested
        return global_state

    def _get_agent_features(self, agent_states: Dict[str, dict], agent_id: str) -> Dict[str, dict]:
        state_dict = agent_states.get(agent_id, {})
        if not isinstance(state_dict, dict):
            return {}
        features = state_dict.get("features", {})
        return features if isinstance(features, dict) else {}

    def _current_time_s_from_proxy(self, agent_states: Dict[str, dict]) -> float:
        station_features = self._get_agent_features(agent_states, str(self.agent_id))
        market_feature = station_features.get("MarketFeature", {}) if isinstance(station_features, dict) else {}
        if isinstance(market_feature, dict):
            return float(market_feature.get("t_day_s", 0.0))
        return 0.0

    def _update_profit_bookkeeping_from_proxy(self, agent_states: Dict[str, dict], current_time_s: float) -> None:
        if self._last_accounting_emitted_time_s is not None and current_time_s <= self._last_accounting_emitted_time_s + 1e-9:
            return

        station_features = self._get_agent_features(agent_states, str(self.agent_id))
        station_feature = station_features.get("ChargingStationFeature", {}) if isinstance(station_features, dict) else {}
        market_feature = station_features.get("MarketFeature", {}) if isinstance(station_features, dict) else {}

        price = self.current_price
        if isinstance(station_feature, dict):
            price = float(station_feature.get("charging_price", price))

        lmp = 0.0
        if isinstance(market_feature, dict):
            lmp = float(market_feature.get("lmp", 0.0))

        delta_t = max(0.0, float(current_time_s) - float(self._last_accounting_time_s)) if self._last_accounting_time_s is not None else self._tick_interval_s()
        if self._last_accounting_time_s == 0.0 and current_time_s == 0.0:
            delta_t = 0.0

        step_revenue = 0.0
        step_energy_cost = 0.0
        step_energy_delivered = 0.0
        per_charger_energy_kwh: Dict[str, float] = {}
        for subordinate_id in self.subordinates:
            charger_features = self._get_agent_features(agent_states, str(subordinate_id))
            charger_feature = charger_features.get("ChargerFeature", {}) if isinstance(charger_features, dict) else {}
            if not isinstance(charger_feature, dict):
                continue
            energy_kwh = float(charger_feature.get("step_energy_delivered_kwh", 0.0))
            per_charger_energy_kwh[str(subordinate_id)] = energy_kwh
            if energy_kwh <= 0.0:
                continue
            step_energy_delivered += energy_kwh
            step_revenue += energy_kwh * price
            step_energy_cost += (lmp + self.operational_cost_per_kwh) * energy_kwh / self.charging_efficiency

        step_overhead_cost = self.hourly_overhead_cost * (delta_t / 3600.0)
        step_profit = step_revenue - step_energy_cost - step_overhead_cost

        self.current_price = price
        self.last_step_revenue = step_revenue
        self.last_step_energy_cost = step_energy_cost
        self.last_step_overhead_cost = step_overhead_cost
        self.last_step_profit = step_profit
        self.cumulative_profit += step_profit
        self.tick_accumulated_profit += step_profit
        self._last_accounting_time_s = float(current_time_s)
        self._last_accounting_emitted_time_s = float(current_time_s)

        # Keep a cheap internal check that profit accumulation is numerically stable.
        assert np.isfinite(self.cumulative_profit)
        if step_energy_delivered > 1e-9 and price > 1e-9:
            assert step_revenue > 0.0, "Positive delivered energy with positive price must produce positive revenue"

        self._last_reward_debug = {
            "current_time_s": float(current_time_s),
            "last_reward_time_s": float(self._last_reward_time_s),
            "tick_interval_s": float(self._tick_interval_s()),
            "current_price": float(price),
            "lmp": float(lmp),
            "per_charger_step_energy_delivered_kwh": per_charger_energy_kwh,
            "step_energy_delivered_kwh_total": float(step_energy_delivered),
            "step_revenue": float(step_revenue),
            "step_energy_cost": float(step_energy_cost),
            "step_overhead_cost": float(step_overhead_cost),
            "step_profit": float(step_profit),
            "cumulative_profit": float(self.cumulative_profit),
        }

    def _is_tick_boundary(self, current_time_s: float) -> bool:
        tick_interval = self._tick_interval_s()
        if tick_interval <= 0.0:
            return True
        return (current_time_s - self._last_reward_time_s) >= (tick_interval - 1e-9)

    def _compute_tick_reward_from_proxy(self, proxy) -> float:
        if proxy is None:
            self.last_reward = 0.0
            self._last_reward_debug = {
                "tick_boundary": False,
                "emitted_reward": 0.0,
                "warning": "proxy_unavailable",
            }
            return 0.0

        agent_states = self._get_sim_agent_states(proxy)
        current_time_s = self._current_time_s_from_proxy(agent_states)
        self._update_profit_bookkeeping_from_proxy(agent_states, current_time_s)

        num_chargers = max(1, len(self.subordinates))
        if not self._is_tick_boundary(current_time_s):
            self.last_reward = 0.0
            self._last_reward_debug["emitted_reward"] = 0.0
            self._last_reward_debug["tick_boundary"] = False
            return 0.0

        profit_delta = self.cumulative_profit - self.last_tick_profit_anchor
        reward = profit_delta / float(num_chargers)
        assert np.isclose(reward * num_chargers, profit_delta, rtol=1e-9, atol=1e-9), (
            "Station reward must equal cumulative profit delta divided by charger count"
        )

        self.last_tick_profit_anchor = self.cumulative_profit
        self.tick_accumulated_profit = 0.0
        self.tick_counter += 1
        self.last_reward = float(reward)
        self._last_reward_time_s = current_time_s
        self._last_reward_debug["emitted_reward"] = float(reward)
        self._last_reward_debug["tick_boundary"] = True
        return float(reward)

    def compute_local_reward(self, local_state: dict) -> float:
        return self._compute_tick_reward_from_proxy(self._proxy_ref)

    def get_local_info(self, local_state: dict) -> dict:
        info = super().get_local_info(local_state)
        csf = local_state.get("ChargingStationFeature")
        mf = local_state.get("MarketFeature")
        num_chargers = max(1, len(self.subordinates))

        if isinstance(csf, np.ndarray) and len(csf) >= 3:
            open_norm, price_norm, utilization = [float(x) for x in csf[:3]]
            info.update({
                "charging_price": price_norm * 0.8,
                "open_chargers": int(round(open_norm * num_chargers)),
                "num_chargers": num_chargers,
                "utilization": utilization,
                "current_price": self.current_price,
                "last_step_revenue": self.last_step_revenue,
                "last_step_energy_cost": self.last_step_energy_cost,
                "last_step_overhead_cost": self.last_step_overhead_cost,
                "last_step_profit": self.last_step_profit,
                "cumulative_profit": self.cumulative_profit,
                "tick_accumulated_profit": self.tick_accumulated_profit,
                "tick_counter": self.tick_counter,
                "reward_debug": dict(self._last_reward_debug),
            })
        else:
            feature = self.state.features.get("ChargingStationFeature")
            if feature is not None:
                utilization = 0.0
                if feature.station_capacity > 0:
                    utilization = float(feature.station_power / feature.station_capacity)
                info.update({
                    "charging_price": float(feature.charging_price),
                    "open_chargers": int(feature.open_chargers),
                    "num_chargers": num_chargers,
                    "utilization": utilization,
                    "current_price": self.current_price,
                    "last_step_revenue": self.last_step_revenue,
                    "last_step_energy_cost": self.last_step_energy_cost,
                    "last_step_overhead_cost": self.last_step_overhead_cost,
                    "last_step_profit": self.last_step_profit,
                    "cumulative_profit": self.cumulative_profit,
                    "tick_accumulated_profit": self.tick_accumulated_profit,
                    "tick_counter": self.tick_counter,
                    "reward_debug": dict(self._last_reward_debug),
                })

        if isinstance(mf, np.ndarray) and len(mf) >= 1:
            info["lmp"] = float(mf[0])
        else:
            market = self.state.features.get("MarketFeature")
            if market is not None:
                info.update({
                    "lmp": float(market.lmp),
                    "time_s": float(market.t_day_s),
                })
        return info

    def get_info(self, proxy) -> dict:
        agent_states = self._get_sim_agent_states(proxy)
        state_dict = agent_states.get(self.agent_id, {})
        features = state_dict.get("features", {}) if isinstance(state_dict, dict) else {}
        station_feature = features.get("ChargingStationFeature", {}) if isinstance(features, dict) else {}
        market_feature = features.get("MarketFeature", {}) if isinstance(features, dict) else {}

        info = self.get_local_info({})
        if isinstance(station_feature, dict):
            info.update({
                "charging_price": float(station_feature.get("charging_price", info.get("charging_price", 0.0))),
                "open_chargers": int(station_feature.get("open_chargers", info.get("open_chargers", 0))),
                "num_chargers": max(1, len(self.subordinates)),
                "current_price": self.current_price,
                "last_step_revenue": self.last_step_revenue,
                "last_step_energy_cost": self.last_step_energy_cost,
                "last_step_overhead_cost": self.last_step_overhead_cost,
                "last_step_profit": self.last_step_profit,
                "cumulative_profit": self.cumulative_profit,
                "tick_accumulated_profit": self.tick_accumulated_profit,
                "tick_counter": self.tick_counter,
                "reward_used_for_rl": self.last_reward,
                "reward_debug": dict(self._last_reward_debug),
            })
        if isinstance(market_feature, dict):
            info.update({
                "lmp": float(market_feature.get("lmp", 0.0)),
                "time_s": float(market_feature.get("t_day_s", 0.0)),
            })
        return info


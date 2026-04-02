"""Station coordinator agent.

Manages a fixed pool of ChargerAgent subordinates and makes pricing decisions.
Follows the same pattern as powergrid's PowerGridAgent(CoordinatorAgent).

AGENT ECONOMICS:
- Optimizes pricing based on market conditions (LMP) and demand
- Computes EV demand/utility based on price
- Determines optimal power delivery
- Aggregates subordinate charger revenue
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

    观测空间: ChargingStationFeature (4D) + MarketFeature (3D) = 7D 观测
    动作空间: [0, 0.8] $/kWh 的 1D 连续定价决策
    奖励机制: 站点 step profit / charger 数
    
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

        # 观测空间定义
        # ChargingStationFeature.vector(): [open_norm, price_norm, step_profit, utilization] (4)
        # MarketFeature.vector(): [lmp, sin(theta), cos(theta)] (3)
        self.observation_space = Box(-np.inf, np.inf, (7,), np.float32)
        expected_obs_dim = sum(feature.vector().shape[0] for feature in all_features)
        if expected_obs_dim != self.observation_space.shape[0]:
            raise ValueError(
                f"StationCoordinator observation_space mismatch: expected {expected_obs_dim}, "
                f"declared {self.observation_space.shape[0]}"
            )

        # 动作空间：定价范围 [0, 0.8] $/kWh
        self.action_space = Box(0.0, 0.8, (1,), np.float32)

    @property
    def charger_agents(self) -> Dict[AgentID, ChargerAgent]:
        """subordinates 的别名，方便调用。"""
        return self.subordinates

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
        """Return station reward plus subordinate rewards.

        Coordinator reward is computed from the station's current step profit,
        normalized by charger count. Subordinates keep their own per-step profit.
        """
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        global_state = proxy.get_global_states(self.agent_id, self.protocol, for_simulation=True)
        state_dict = global_state.get(self.agent_id, {}) if isinstance(global_state, dict) else {}
        features = state_dict.get("features", {}) if isinstance(state_dict, dict) else {}
        station_feature = features.get("ChargingStationFeature", {}) if isinstance(features, dict) else {}

        num_chargers = max(1, len(self.subordinates))
        raw_step_profit = float(station_feature.get("station_step_profit", 0.0)) if isinstance(station_feature, dict) else 0.0
        coordinator_reward = raw_step_profit / float(num_chargers)

        # Sanity check: local decoded reward should match raw state reward.
        local_reward = self.compute_local_reward(local_state)
        if np.isfinite(local_reward):
            assert np.isclose(local_reward, coordinator_reward, rtol=1e-5, atol=1e-6), (
                "Coordinator reward mismatch: local observation-derived reward differs from raw station_step_profit"
            )

        sub_rewards: Dict[AgentID, float] = {}
        for subordinate in self.subordinates.values():
            sub_rewards.update(subordinate.compute_rewards(proxy))

        rewards = {self.agent_id: coordinator_reward}
        rewards.update(sub_rewards)
        return rewards

    def compute_local_reward(self, local_state: dict) -> float:
        """Station reward = current step profit / number of chargers."""
        csf = local_state.get("ChargingStationFeature")
        if isinstance(csf, np.ndarray) and len(csf) >= 3:
            step_profit = ChargingStationFeature.obs_to_profit(float(csf[2]))
        else:
            feature = self.state.features.get("ChargingStationFeature")
            if feature is None:
                return 0.0
            step_profit = float(getattr(feature, "station_step_profit", 0.0))

        num_chargers = max(1, len(self.subordinates))
        reward = step_profit / float(num_chargers)
        return float(reward)

    def get_local_info(self, local_state: dict) -> dict:
        info = super().get_local_info(local_state)
        csf = local_state.get("ChargingStationFeature")
        mf = local_state.get("MarketFeature")
        num_chargers = max(1, len(self.subordinates))

        if isinstance(csf, np.ndarray) and len(csf) >= 4:
            open_norm, price_norm, step_profit_obs, utilization = [float(x) for x in csf[:4]]
            step_profit = ChargingStationFeature.obs_to_profit(step_profit_obs)
            info.update({
                "charging_price": price_norm * 0.8,
                "open_chargers": int(round(open_norm * num_chargers)),
                "num_chargers": num_chargers,
                "utilization": utilization,
                "step_profit": step_profit,
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
                    "step_revenue": float(feature.station_step_revenue),
                    "step_energy_cost": float(feature.station_step_energy_cost),
                    "step_overhead_cost": float(feature.station_step_overhead_cost),
                    "step_profit": float(feature.station_step_profit),
                    "cumulative_revenue": float(feature.station_cumulative_revenue),
                    "cumulative_profit": float(feature.station_cumulative_profit),
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
        global_state = proxy.get_global_states(self.agent_id, self.protocol, for_simulation=True)
        state_dict = global_state.get(self.agent_id, {})
        features = state_dict.get("features", {}) if isinstance(state_dict, dict) else {}
        station_feature = features.get("ChargingStationFeature", {}) if isinstance(features, dict) else {}
        market_feature = features.get("MarketFeature", {}) if isinstance(features, dict) else {}

        info = self.get_local_info({})
        if isinstance(station_feature, dict):
            info.update({
                "charging_price": float(station_feature.get("charging_price", info.get("charging_price", 0.0))),
                "open_chargers": int(station_feature.get("open_chargers", info.get("open_chargers", 0))),
                "num_chargers": max(1, len(self.subordinates)),
                "step_revenue": float(station_feature.get("station_step_revenue", 0.0)),
                "step_energy_cost": float(station_feature.get("station_step_energy_cost", 0.0)),
                "step_overhead_cost": float(station_feature.get("station_step_overhead_cost", 0.0)),
                "step_profit": float(station_feature.get("station_step_profit", 0.0)),
                "cumulative_revenue": float(station_feature.get("station_cumulative_revenue", 0.0)),
                "cumulative_profit": float(station_feature.get("station_cumulative_profit", 0.0)),
                "reward_used_for_rl": float(station_feature.get("station_step_profit", 0.0)) / max(1, len(self.subordinates)),
            })
        if isinstance(market_feature, dict):
            info.update({
                "lmp": float(market_feature.get("lmp", 0.0)),
                "time_s": float(market_feature.get("t_day_s", 0.0)),
            })
        return info


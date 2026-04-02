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

    观测空间: ChargingStationFeature (2D) + MarketFeature (3D) = 5D 观测
    动作空间: [0, 0.8] $/kWh 的 1D 连续定价决策
    奖励机制: 聚合所有下属充电桩代理（ChargerAgent）的奖励
    
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
        # ChargingStationFeature.vector(): [open_norm, price_norm] (2)
        # MarketFeature.vector(): [lmp, sin(theta), cos(theta)] (3)
        self.observation_space = Box(-np.inf, np.inf, (5,), np.float32)

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
        """聚合协调员及其所有下属充电桩代理的奖励。

        此方法在事件驱动模式下至关重要。它会轮询每个 ChargerAgent，
        获取它们基于实际接收到的价格（受通信延迟影响）计算出的奖励。
        """
        # 1. 首先计算所有下属充电桩代理的奖励
        sub_rewards: Dict[AgentID, float] = {}
        for subordinate in self.subordinates.values():
            # 这里会调用 ChargerAgent.compute_local_reward
            sub_rewards.update(subordinate.compute_rewards(proxy))

        # 2. 协调员奖励 = 所有下属充电桩代理奖励之和（即站点总收益）
        coordinator_reward = sum(sub_rewards.values())

        rewards = {self.agent_id: coordinator_reward}
        rewards.update(sub_rewards)
        return rewards

    def compute_local_reward(self, local_state: dict) -> float:
        """基于自身特征计算站点本地奖励（通常作为辅助参考）。

        奖励公式 = 占用比例 * 利润边际 (价格 - 实时电价 LMP)。
        """
        # 获取站点特征
        csf = local_state.get("ChargingStationFeature")
        if csf is None:
            return 0.0

        # open_norm 是空闲比例，则占用比例为 1 - open_norm
        open_norm = float(csf[0])
        occupied_fraction = 1.0 - open_norm

        # 获取当前设置的价格
        price_norm = float(csf[1])
        price = price_norm * 0.8

        # 获取市场特征（LMP）
        mf = local_state.get("MarketFeature")
        if mf is not None:
            lmp = float(mf[0])
        else:
            lmp = 0.2

        # 计算边际利润，最低为 0
        margin = max(0.0, price - lmp)
        return occupied_fraction * margin


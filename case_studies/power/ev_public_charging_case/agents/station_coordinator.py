"""
Station Coordinator implementation for the EV Public Charging Case Study.
Handles pricing decisions and profit-based reward calculation.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from gymnasium.spaces import Box

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.protocols.vertical import BroadcastActionProtocol, VerticalProtocol
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID

# 引用你定义的特征类与子代理
from case_studies.power.ev_public_charging_case.features import ChargingStationFeature, MarketFeature
from case_studies.power.ev_public_charging_case.agents.charger_field_agent import ChargerAgent


class StationCoordinator(CoordinatorAgent):
    """
    站点协调员：管理固定数量的充电桩代理，并做出统一的市场定价决策。

    核心逻辑：
    - 决策：输出归一化挂牌价 [-1, 1]。
    - 核算：通过 Proxy 汇总所有从属桩的实时电量与 Session 价格。
    - 奖励：以站点步长利润 (Revenue - LMP Cost) 作为强化学习奖励。
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
        # 1. 验证从属代理是否存在
        if not subordinates:
            raise ValueError("StationCoordinator requires subordinates (ChargerAgent).")

        # 2. 注入默认特征 (Station 状态 + 市场状态)
        num_chargers = len(subordinates)
        default_features = [
            ChargingStationFeature(num_chargers=float(num_chargers)),
            MarketFeature(),
        ]
        all_features = (features or []) + default_features

        # 3. 初始化父类：建立层级关系与通信协议
        super().__init__(
            agent_id=agent_id,
            features=all_features,
            subordinates=subordinates,
            upstream_id=upstream_id,
            env_id=env_id,
            schedule_config=schedule_config,
            policy=policy,
            # 使用广播协议：将单一价格动作同步给所有从属桩
            protocol=protocol or VerticalProtocol(
                action_protocol=BroadcastActionProtocol(),
            ),
        )

        # 4. 内部状态初始化
        self.current_price = 0.5
        self.cumulative_profit = 0.0

        # 5. 定义观测与动作空间 (符合特征向量维度)
        # 观测: Station (4D) + Market (2D) = 6D
        self.observation_space = Box(-np.inf, np.inf, (6,), np.float32)
        # 动作: 归一化定价信号
        self.action_space = Box(-1.0, 1.0, (1,), np.float32)

    @property
    def chargers(self) -> Dict[AgentID, ChargerAgent]:
        """别名：更直观地访问子代理。"""
        return self.subordinates

    def init_action(self, features: List[Feature] = []) -> Action:
        """初始化协调员的动作空间。"""
        action = Action()
        action.set_specs(
            dim_c=1,
            range=(np.array([-1.0]), np.array([1.0])),
        )
        action.set_values(c=np.array([0.0], dtype=np.float32))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """接收 Policy 输出的动作，映射为物理挂牌价并更新特征。"""
        super().set_action(action, *args, **kwargs)
        if self.action is not None and self.action.is_valid():
            price_norm = float(self.action.c[0])
            # 将 [-1, 1] 映射到物理价格 [0, 1]
            self.current_price = float(np.clip(0.5 * (price_norm + 1.0), 0.0, 1.0))
            # 更新特征，以便下一轮观察到决策后的价格
            self.state.update_feature("ChargingStationFeature", current_price=self.current_price)

    def _update_station_features(self, proxy) -> float:
        """
        核心账本逻辑：
        1. 遍历所有子桩，获取物理电量和入场时锁定的 session_price。
        2. 获取市场 LMP 成本。
        3. 更新 StationFeature 的占用率和利润指标。
        """
        # 获取所有代理的全局状态快照
        agent_states = proxy.get_global_states()

        # 1. 获取当前电网成本 (LMP)
        market_feat = self.state.features.get("MarketFeature")
        current_lmp = market_feat.lmp

        # 2. 汇总子代理数据
        total_revenue = 0.0
        total_energy = 0.0
        occupied_count = 0

        for sub_id in self.subordinates:
            # 从 Proxy 获取子代理的实时局部状态
            sub_state_dict = agent_states.get(str(sub_id), {})
            charger_data = sub_state_dict.get("ChargerFeature", {})

            if not charger_data:
                continue

            # 提取物理量
            energy = float(charger_data.get("step_energy_delivered_kwh", 0.0))
            is_occupied = float(charger_data.get("occupied_or_not", 0.0))
            # 核心：使用该桩入场时锁定的价格计算收入，而非当前的挂牌价
            locked_session_price = float(charger_data.get("session_price", self.current_price))

            total_revenue += energy * locked_session_price
            total_energy += energy
            if is_occupied > 0:
                occupied_count += 1

        # 3. 计算步长利润：(收入 - 电费成本)
        # 备注：此处可根据需要减去固定运营成本 (hourly_overhead_cost)
        step_profit = total_revenue - (total_energy * current_lmp)

        # 4. 更新 ChargingStationFeature 状态
        num_total = len(self.subordinates)
        self.state.update_feature(
            "ChargingStationFeature",
            occupancy_ratio=float(occupied_count / num_total) if num_total > 0 else 0.0,
            delta_profit=float(step_profit),
            num_chargers=float(num_total)
        )

        return step_profit

    def compute_rewards(self, proxy) -> Dict[AgentID, float]:
        """
        分发奖励：以单桩平均利润 (Total Profit / Num Chargers) 作为奖励信号。
        """
        # 1. 触发账本更新，获取该步长的总利润增量
        total_step_profit = self._update_station_features(proxy)

        # 2. 统计累计利润（记录总额用于监控）
        self.cumulative_profit += total_step_profit

        # 3. 归一化奖励计算：利润 / 桩数
        num_chargers = len(self.subordinates)
        # 避免除以零的防御性编程
        normalized_reward = float(total_step_profit / num_chargers) if num_chargers > 0 else 0.0

        # 4. 构造返回字典
        # 协调员获得归一化后的奖励
        rewards = {self.agent_id: normalized_reward}

        # 子代理奖励保持为 0
        for sub_id in self.subordinates:
            rewards[sub_id] = 0.0

        return rewards

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        feat = self.state.features.get("ChargingStationFeature")
        return float(feat.delta_profit)

    def get_local_info(self, local_state: dict) -> dict:
        feat = self.state.features.get("ChargingStationFeature")
        market = self.state.features.get("MarketFeature")
        return {
            "posted_price": self.current_price,
            "occupancy_rate": feat.occupancy_ratio,
            "step_profit": feat.delta_profit,
            "cumulative_profit": self.cumulative_profit,
            "market_lmp": market.lmp
        }
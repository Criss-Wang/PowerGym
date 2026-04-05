"""
Charger Field Agent implementation.
与最新的 ChargerFeature 字段完全对齐。
"""

from typing import Any, List, Optional
import numpy as np

from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.scheduler import Event, EventScheduler
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID

from case_studies.power.ev_public_charging_case.features import ChargerFeature


class ChargerAgent(FieldAgent):
    def __init__(
            self,
            agent_id: AgentID,
            p_max_kw: float = 150.0,
            upstream_id: Optional[AgentID] = None,
            env_id: Optional[str] = None,
            schedule_config: Optional[ScheduleConfig] = None,
            protocol: Optional[Protocol] = None,
    ):
        self._p_max_kw = p_max_kw
        self._last_occupied: float = 0.0

        self.current_price = 0.5

        features: List[Feature] = [ChargerFeature(p_max_kw=p_max_kw)]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            schedule_config=schedule_config,
            policy=None,  # 明确没有自主 Policy
            protocol=protocol,
        )

    def init_action(self, features: Optional[List[Feature]] = None) -> Action:
        action = Action()
        action.set_specs(
            dim_c=1,
            range=(np.array([-1.0]), np.array([1.0])),
        )
        action.set_values(c=np.array([0.0], dtype=np.float32))
        return action

    def set_state(self, **kwargs) -> None:
        self.state.update_feature("ChargerFeature", **kwargs)

    def set_action(self, action: Any, *args, **kwargs) -> None:
        if isinstance(action, Action):
            self.action = action
        else:
            val = float(np.asarray(action).flatten()[0])
            self.action.set_values(c=np.array([val], dtype=np.float32))

    def apply_action(self) -> None:
        """
        处理价格锁定、功率输出控制以及充电时长累加。
        """
        # 1. 价格映射：[-1, 1] -> [0, 1]
        price_norm = float(self.action.c[0]) if self.action.c.size > 0 else 0.0
        self.current_price = float(np.clip(0.5 * (price_norm + 1.0), 0.0, 1.0))

        feat = self.state.features.get("ChargerFeature")
        current_occupied = int(feat.occupied_or_not)

        # 获取步长 dt (从配置中读取，默认为 1.0)
        dt = getattr(self.schedule_config, "dt", 1.0)

        # 2. 状态机逻辑
        # CASE A: 新 Session 开始 (0 -> 1)
        if self._last_occupied == 0 and current_occupied == 1:
            target_p_kw = float(feat.p_max_kw * feat.charging_efficiency)
            self.state.update_feature(
                "ChargerFeature",
                p_kw=target_p_kw,
                session_price=self.current_price,  # 锁定挂牌价
                elapsed_charging_time=0.0
            )

        # CASE B: Session 持续中 (1 -> 1)
        elif self._last_occupied == 1 and current_occupied == 1:
            target_p_kw = float(feat.p_max_kw * feat.charging_efficiency)
            # 累加时长
            new_time = feat.elapsed_charging_time + dt
            self.state.update_feature(
                "ChargerFeature",
                p_kw=target_p_kw,
                elapsed_charging_time=new_time
            )

        # CASE C: Session 结束 (1 -> 0)
        elif self._last_occupied == 1 and current_occupied == 0:
            self.state.update_feature(
                "ChargerFeature",
                p_kw=0.0,
                step_energy_delivered_kwh=0.0,
                session_price=0.0,
                elapsed_charging_time=0.0
            )

        self._last_occupied = current_occupied

    # 必须重写此 handler 以确保 apply_action 修改的状态能立即同步给 Proxy
    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        from heron.agents.proxy_agent import PROXY_AGENT_ID, MSG_SET_STATE, STATE_TYPE_LOCAL

        self.apply_action()

        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={
                MSG_SET_STATE: STATE_TYPE_LOCAL,
                "body": self.state.to_dict(include_metadata=True)
            },
        )

    def compute_local_reward(self, local_state: dict) -> float:
        return 0.0

    def get_local_info(self, local_state: dict) -> dict:
        info = super().get_local_info(local_state)
        feature = self.state.features.get("ChargerFeature")
        if feature:
            info.update({
                "p_kw": float(feature.p_kw),
                "occupied_or_not": int(feature.occupied_or_not),
                "step_energy_delivered_kwh": float(feature.step_energy_delivered_kwh),
                "session_price": float(feature.session_price),
                "elapsed_charging_time": float(feature.elapsed_charging_time)
            })
        return info


"""Shared agents for the TwoRoomHeating demo environments."""

from typing import Any, List, Optional

import numpy as np

from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.agents.proxy_agent import Proxy
from heron.core.action import Action
from heron.core.feature import Feature
from heron.demo_envs.two_room_heating.features import (
    VentStatusFeature,
    ZoneTemperatureFeature,
)
from heron.scheduling.event import Event
from heron.scheduling.schedule_config import ScheduleConfig
from heron.scheduling.scheduler import EventScheduler


class HeaterAgent(FieldAgent):
    """Zone heater that adjusts temperature with configurable heat gain.

    Action: continuous scalar in ``[-1, 1]`` representing the
    heating/cooling command applied at each step, scaled by ``heat_gain``.

    Reward: ``-|T - T_target|`` -- closer to target is better.

    Supports a custom ``overheat_alert`` event (v5+): when received, the
    agent clamps its next action to cooling-only (max 0.0).
    """

    def __init__(
        self,
        agent_id: str,
        target_temp: float = 22.0,
        initial_temp: float = 18.0,
        heat_gain: float = 2.0,
        features: Optional[List[Feature]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.target_temp = target_temp
        self._initial_temp = initial_temp
        self.heat_gain = heat_gain
        self._reduce_heat = False
        super().__init__(
            agent_id=agent_id,
            features=features or [ZoneTemperatureFeature(temperature=initial_temp)],
            schedule_config=schedule_config,
            **kwargs,
        )

    def init_action(self, features: Optional[List[Feature]] = None) -> Action:
        action = Action()
        action.set_specs(
            dim_c=1,
            range=(
                np.array([-1.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ),
        )
        return action

    def reset(
        self, *, seed: Optional[int] = None, proxy: Optional[Proxy] = None, **kwargs: Any
    ) -> Any:
        """Reset state, restoring temperature to the configured initial value."""
        self._reduce_heat = False
        result = super().reset(seed=seed, proxy=proxy, **kwargs)
        self.state.update_feature(
            "ZoneTemperatureFeature", temperature=self._initial_temp
        )
        return result

    def set_state(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_action(self, action: Any, *args: Any, **kwargs: Any) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        temp_feature = self.state.features.get("ZoneTemperatureFeature")
        if temp_feature is not None:
            action_val = float(self.action.c[0])
            if self._reduce_heat:
                action_val = min(action_val, 0.0)
                self._reduce_heat = False
            new_temp = temp_feature.temperature + action_val * self.heat_gain
            self.state.update_feature("ZoneTemperatureFeature", temperature=new_temp)

    def compute_local_reward(
        self,
        local_state: dict,
        prev_post_physics_state: Optional[dict] = None,
    ) -> float:
        features = local_state.get("features", {})
        temp_data = features.get("ZoneTemperatureFeature", {})
        temperature = temp_data.get("temperature", 20.0)
        return -abs(temperature - self.target_temp)

    @Agent.custom_handler("overheat_alert")
    def on_overheat_alert(self, event: Event, scheduler: EventScheduler) -> None:
        """Respond to an overheat alert by clamping the next action to cooling-only."""
        self._reduce_heat = True


class VentAgent(FieldAgent):
    """Emergency ventilation agent that controls cooling via vent opening.

    Action: continuous scalar in ``[0, 1]`` representing the vent opening
    fraction.

    Reward: ``-0.5 * is_open`` -- energy penalty for running the vent.
    """

    def __init__(
        self,
        agent_id: str = "vent",
        features: Optional[List[Feature]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            features=features or [VentStatusFeature()],
            schedule_config=schedule_config,
            **kwargs,
        )

    def init_action(self, features: Optional[List[Feature]] = None) -> Action:
        action = Action()
        action.set_specs(
            dim_c=1,
            range=(
                np.array([0.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ),
        )
        return action

    def set_state(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_action(self, action: Any, *args: Any, **kwargs: Any) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        is_open = float(self.action.c[0])
        self.state.update_feature("VentStatusFeature", is_open=is_open)

    def compute_local_reward(
        self,
        local_state: dict,
        prev_post_physics_state: Optional[dict] = None,
    ) -> float:
        features = local_state.get("features", {})
        vent_data = features.get("VentStatusFeature", {})
        is_open = vent_data.get("is_open", 0.0)
        return -0.5 * is_open

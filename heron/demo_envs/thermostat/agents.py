"""Thermostat demo agents and features."""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.proxy_agent import Proxy
from heron.core.action import Action
from heron.core.feature import Feature
from heron.scheduling.schedule_config import ScheduleConfig


@dataclass(slots=True)
class TemperatureFeature(Feature):
    """Single-value temperature feature.

    Attributes:
        temperature: Current room temperature (Celsius).
    """

    visibility: ClassVar[Sequence[str]] = ("owner", "upper_level")
    temperature: float = 20.0


class HeaterAgent(FieldAgent):
    """Single-action heater that adjusts room temperature.

    Action: continuous scalar ``heat_delta`` in ``[-1, 1]`` representing
    the heating/cooling power applied at each step.

    Reward: ``-|T - T_target|`` — closer to target is better.

    Episode truncates at ``max_steps`` (set on the environment).
    """

    def __init__(
        self,
        agent_id: str = "heater",
        target_temp: float = 22.0,
        initial_temp: float = 20.0,
        features: Optional[List[Feature]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.target_temp = target_temp
        self._initial_temp = initial_temp
        super().__init__(
            agent_id=agent_id,
            features=features or [TemperatureFeature(temperature=initial_temp)],
            schedule_config=schedule_config,
            **kwargs,
        )

    def init_action(self, features: Optional[List[Feature]] = None) -> Action:
        action = Action()
        action.set_specs(
            dim_c=1,
            range=(np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)),
        )
        return action

    def reset(
        self, *, seed: Optional[int] = None, proxy: Optional[Proxy] = None, **kwargs: Any
    ) -> Any:
        """Reset state, restoring temperature to the configured initial value."""
        result = super().reset(seed=seed, proxy=proxy, **kwargs)
        self.state.update_feature(
            "TemperatureFeature", temperature=self._initial_temp
        )
        return result

    def set_state(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_action(self, action: Any, *args: Any, **kwargs: Any) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        temp_feature = self.state.features.get("TemperatureFeature")
        if temp_feature is not None:
            new_temp = temp_feature.temperature + float(self.action.c[0])
            self.state.update_feature("TemperatureFeature", temperature=new_temp)

    def compute_local_reward(
        self,
        local_state: dict,
        prev_post_physics_state: Optional[dict] = None,
    ) -> float:
        features = local_state.get("features", {})
        temp_data = features.get("TemperatureFeature", {})
        temperature = temp_data.get("temperature", 20.0)
        return -abs(temperature - self.target_temp)

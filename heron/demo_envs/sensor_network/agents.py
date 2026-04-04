"""Sensor network demo agents and features."""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.state import FieldAgentState, State
from heron.scheduling.schedule_config import ScheduleConfig


@dataclass(slots=True)
class SensorFeature(Feature):
    """Feature for a single sensor node in the network.

    Attributes:
        signal_strength: Actual signal intensity at this node (0.0-1.0+).
        detection: Agent's binary detection output (0 or 1).
        neighbor_avg_detection: Average detection value from neighbors
            (populated via horizontal protocol gossip).
    """

    visibility: ClassVar[Sequence[str]] = ("owner", "upper_level")
    signal_strength: float = 0.0
    detection: float = 0.0
    neighbor_avg_detection: float = 0.0


class SensorAgent(FieldAgent):
    """Sensor node that detects signals propagating across a graph.

    Action: discrete binary (0=no-detect, 1=detect).

    Reward:
        +1.0 for true positive  (signal > 0.5 and detect == 1)
        -1.0 for false positive (signal <= 0.5 and detect == 1)
        +0.5 for true negative  (signal <= 0.5 and detect == 0)
         0.0 for false negative (signal > 0.5 and detect == 0)
    """

    def __init__(
        self,
        agent_id: str = "sensor_0",
        features: Optional[List[Feature]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            features=features or [SensorFeature()],
            schedule_config=schedule_config,
            **kwargs,
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(dim_d=1, ncats=[2])  # 0=no-detect, 1=detect
        return action

    def set_state(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_action(self, action: Any, *args: Any, **kwargs: Any) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        sensor_feature = self.state.features.get("SensorFeature")
        if sensor_feature is not None:
            sensor_feature.detection = float(self.action.d[0])

    def compute_local_reward(
        self,
        local_state: dict,
        prev_post_physics_state: Optional[dict] = None,
    ) -> float:
        features = local_state.get("features", {})
        sensor_data = features.get("SensorFeature", {})
        signal = sensor_data.get("signal_strength", 0.0)
        detection = sensor_data.get("detection", 0.0)

        has_signal = signal > 0.5
        detected = detection >= 1.0

        if has_signal and detected:
            return 1.0   # true positive
        if not has_signal and detected:
            return -1.0  # false positive
        if not has_signal and not detected:
            return 0.5   # true negative
        # has_signal and not detected
        return 0.0       # false negative

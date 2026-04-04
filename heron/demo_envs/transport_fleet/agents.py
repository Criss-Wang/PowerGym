"""Transport Fleet demo agents and features."""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Sequence

import numpy as np

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature


@dataclass(slots=True)
class VehicleFeature(Feature):
    """Per-vehicle state feature.

    Attributes:
        x: Horizontal position.
        y: Vertical position.
        fuel: Remaining fuel level.
        deliveries: Number of completed deliveries.
        has_package: Whether the vehicle is carrying a package.
        is_broken: Whether the vehicle is broken down.
    """

    visibility: ClassVar[Sequence[str]] = ("owner", "upper_level")
    x: float = 0.0
    y: float = 0.0
    fuel: float = 100.0
    deliveries: int = 0
    has_package: bool = False
    is_broken: bool = False


@dataclass(slots=True)
class DepotFeature(Feature):
    """Depot-level aggregate feature.

    Attributes:
        total_deliveries: Fleet-wide delivery count.
        pending_requests: Number of outstanding delivery requests.
    """

    visibility: ClassVar[Sequence[str]] = ("owner", "system")
    total_deliveries: int = 0
    pending_requests: int = 0


class VehicleAgent(FieldAgent):
    """Vehicle agent that moves in 2D and delivers packages.

    Action: continuous 2D ``[dx, dy]`` in ``[-1, 1]``.

    Reward: ``deliveries / max(fuel_consumed, 1)`` delta since last step,
    or ``-0.1`` if broken. If fuel <= 0 or broken, movement has no effect.
    """

    def __init__(
        self,
        agent_id: str = "vehicle",
        fuel_capacity: float = 100.0,
        features: Optional[List[Feature]] = None,
        **kwargs: Any,
    ) -> None:
        self.fuel_capacity = fuel_capacity
        self._prev_deliveries = 0
        self._prev_fuel_consumed = 0.0
        super().__init__(
            agent_id=agent_id,
            features=features or [VehicleFeature(fuel=fuel_capacity)],
            **kwargs,
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(
            dim_c=2,
            range=(
                np.array([-1.0, -1.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
            ),
        )
        return action

    def set_state(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_action(self, action: Any, *args: Any, **kwargs: Any) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        vf = self.state.features.get("VehicleFeature")
        if vf is None:
            return
        if vf.is_broken or vf.fuel <= 0:
            return
        dx = float(self.action.c[0])
        dy = float(self.action.c[1])
        distance = np.sqrt(dx * dx + dy * dy)
        vf.x += dx
        vf.y += dy
        vf.fuel = max(0.0, vf.fuel - distance)

    def compute_local_reward(
        self,
        local_state: dict,
        prev_post_physics_state: Optional[dict] = None,
    ) -> float:
        # Use internal state directly — avoids format differences between
        # training mode ({feature_name: numpy_array}) and event-driven mode
        # ({features: {feature_name: {field: val}}}).
        vf = self.state.features.get("VehicleFeature")
        if vf is None:
            return 0.0
        if vf.is_broken:
            return -0.1

        fuel_consumed = self.fuel_capacity - vf.fuel
        delta_deliveries = vf.deliveries - self._prev_deliveries
        delta_fuel = fuel_consumed - self._prev_fuel_consumed

        self._prev_deliveries = vf.deliveries
        self._prev_fuel_consumed = fuel_consumed

        return delta_deliveries / max(delta_fuel, 1.0)


class DepotCoordinator(CoordinatorAgent):
    """Fleet depot coordinator that tracks aggregate deliveries.

    Reward: number of new team deliveries since last physics step.
    """

    def __init__(
        self,
        agent_id: str = "depot",
        features: Optional[List[Feature]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            features=features or [DepotFeature()],
            **kwargs,
        )
        self._prev_total_deliveries = 0

    def compute_local_reward(
        self,
        local_state: dict,
        prev_post_physics_state: Optional[dict] = None,
    ) -> float:
        # Use internal state directly — avoids format differences between modes
        df = self.state.features.get("DepotFeature")
        total = df.total_deliveries if df is not None else 0
        delta = total - self._prev_total_deliveries
        self._prev_total_deliveries = total
        return float(delta)

"""Level 7: Multi-level hierarchy with floor controller.

``floor_ctrl`` is a periodic top-level coordinator using
VectorDecompositionActionProtocol. All agents underneath become
reactive (ticked by their parent coordinator).

Hierarchy::

    system_agent -> floor_ctrl (10s, VectorDecomposition)
        -> [heater_a (reactive), heater_b (reactive),
            building_ctrl (reactive) -> vent (reactive)]
"""

from typing import Optional

from heron.demo_envs.two_room_heating.agents import HeaterAgent, VentAgent
from heron.demo_envs.two_room_heating.features import (
    VentStatusFeature,
    ZoneTemperatureFeature,
)
from heron.demo_envs.two_room_heating.physics import (
    TwoRoomEnv,
    default_disturbance_schedule,
    register_overheat_monitors,
)
from heron.envs.builder import EnvBuilder
from heron.protocols.vertical import (
    BroadcastActionProtocol,
    VectorDecompositionActionProtocol,
    VerticalProtocol,
)
from heron.scheduling import DisturbanceSchedule
from heron.scheduling.schedule_config import ScheduleConfig


def build_v7(
    target_temp: float = 22.0,
    initial_temp_a: float = 18.0,
    initial_temp_b: float = 18.0,
    max_steps: int = 200,
    ambient_temp: float = 15.0,
    cooling_rate: float = 0.05,
    coupling_rate: float = 0.03,
    heater_a_schedule: Optional[ScheduleConfig] = None,
    heater_b_schedule: Optional[ScheduleConfig] = None,
    ctrl_schedule: Optional[ScheduleConfig] = None,
    vent_schedule: Optional[ScheduleConfig] = None,
    floor_ctrl_schedule: Optional[ScheduleConfig] = None,
    overheat_threshold: float = 30.0,
    enable_condition_monitor: bool = False,
    disturbance_schedule: Optional[DisturbanceSchedule] = None,
) -> TwoRoomEnv:
    """Build a v7 TwoRoomHeating environment."""
    if disturbance_schedule is None:
        disturbance_schedule = default_disturbance_schedule()

    if floor_ctrl_schedule is None:
        floor_ctrl_schedule = ScheduleConfig.deterministic(tick_interval=10.0)

    env = (
        EnvBuilder("two_room_v7")
        .add_agent(
            "heater_a",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_a, target=target_temp)],
            schedule_config=heater_a_schedule,
            target_temp=target_temp,
            initial_temp=initial_temp_a,
            heat_gain=2.0,
        )
        .add_agent(
            "heater_b",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_b, target=target_temp)],
            schedule_config=heater_b_schedule,
            target_temp=target_temp,
            initial_temp=initial_temp_b,
            heat_gain=3.0,
        )
        .add_agent(
            "vent",
            VentAgent,
            features=[VentStatusFeature()],
            schedule_config=vent_schedule,
            coordinator="building_ctrl",
        )
        .add_coordinator(
            "building_ctrl",
            subordinates=["vent"],
            protocol=VerticalProtocol(action_protocol=BroadcastActionProtocol()),
            schedule_config=ctrl_schedule,
        )
        .add_coordinator(
            "floor_ctrl",
            subordinates=["heater_a", "heater_b", "building_ctrl"],
            protocol=VerticalProtocol(
                action_protocol=VectorDecompositionActionProtocol(),
            ),
            schedule_config=floor_ctrl_schedule,
        )
        .env_class(
            TwoRoomEnv,
            base_ambient_temp=ambient_temp,
            cooling_rate=cooling_rate,
            coupling_rate=coupling_rate,
            enable_vent=True,
            disturbance_schedule=disturbance_schedule,
        )
        .termination(max_steps=max_steps)
        .build()
    )

    if enable_condition_monitor:
        register_overheat_monitors(env, overheat_threshold)

    return env

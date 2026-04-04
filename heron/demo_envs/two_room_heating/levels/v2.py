"""Level 2: Reactive vent agent + condition trigger.

A vent agent is added under building_ctrl via BroadcastActionProtocol.
Optionally enables ConditionMonitor threshold triggers that wake the
vent when any zone overheats.

Hierarchy::

    system_agent -> [heater_a (1s), heater_b (3s),
                     building_ctrl (5s) -> vent (reactive)]
"""

from typing import Optional

from heron.demo_envs.two_room_heating.agents import HeaterAgent, VentAgent
from heron.demo_envs.two_room_heating.features import (
    VentStatusFeature,
    ZoneTemperatureFeature,
)
from heron.demo_envs.two_room_heating.physics import (
    default_ctrl_schedule,
    default_heater_a_schedule,
    default_heater_b_schedule,
    make_sim,
    register_overheat_monitors,
)
from heron.envs.builder import EnvBuilder
from heron.envs.simple import DefaultHeronEnv
from heron.protocols.vertical import BroadcastActionProtocol, VerticalProtocol
from heron.scheduling.schedule_config import ScheduleConfig


def build_v2(
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
    overheat_threshold: float = 30.0,
    enable_condition_monitor: bool = False,
) -> DefaultHeronEnv:
    """Build a v2 TwoRoomHeating environment."""
    env = (
        EnvBuilder("two_room_v2")
        .add_agent(
            "heater_a",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_a, target=target_temp)],
            schedule_config=heater_a_schedule or default_heater_a_schedule(),
            target_temp=target_temp,
            initial_temp=initial_temp_a,
            heat_gain=2.0,
        )
        .add_agent(
            "heater_b",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_b, target=target_temp)],
            schedule_config=heater_b_schedule or default_heater_b_schedule(),
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
            schedule_config=ctrl_schedule or default_ctrl_schedule(),
        )
        .simulation(make_sim(
            ambient_temp=ambient_temp,
            cooling_rate=cooling_rate,
            coupling_rate=coupling_rate,
            enable_vent=True,
        ))
        .termination(max_steps=max_steps)
        .build()
    )

    if enable_condition_monitor:
        register_overheat_monitors(env, overheat_threshold)

    return env

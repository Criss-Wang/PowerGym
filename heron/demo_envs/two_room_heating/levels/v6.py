"""Level 6: Jitter and communication delays on all agents.

Same structure as v5 but all ScheduleConfigs use ``with_jitter()``
with Gaussian noise (10% ratio) for realistic distributed-system
simulation. User-supplied schedules override the jittered defaults.

Hierarchy::

    system_agent -> [heater_a (1s), heater_b (3s),
                     building_ctrl (5s) -> vent (reactive)]
"""

from typing import Optional

from heron.demo_envs.two_room_heating.levels.v4 import build_v4
from heron.demo_envs.two_room_heating.physics import TwoRoomEnv
from heron.scheduling import DisturbanceSchedule
from heron.scheduling.schedule_config import JitterType, ScheduleConfig


def build_v6(
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
    disturbance_schedule: Optional[DisturbanceSchedule] = None,
) -> TwoRoomEnv:
    """Build a v6 TwoRoomHeating environment."""
    if heater_a_schedule is None:
        heater_a_schedule = ScheduleConfig.with_jitter(
            tick_interval=1.0, obs_delay=0.05, act_delay=0.1, msg_delay=0.02,
            jitter_type=JitterType.GAUSSIAN, jitter_ratio=0.1, seed=42,
        )
    if heater_b_schedule is None:
        heater_b_schedule = ScheduleConfig.with_jitter(
            tick_interval=3.0, obs_delay=0.05, act_delay=0.1, msg_delay=0.02,
            jitter_type=JitterType.GAUSSIAN, jitter_ratio=0.1, seed=43,
        )
    if ctrl_schedule is None:
        ctrl_schedule = ScheduleConfig.with_jitter(
            tick_interval=5.0, obs_delay=0.05, act_delay=0.1, msg_delay=0.02,
            jitter_type=JitterType.GAUSSIAN, jitter_ratio=0.1, seed=44,
        )
    if vent_schedule is None:
        vent_schedule = ScheduleConfig.with_jitter(
            tick_interval=1.0, obs_delay=0.05, act_delay=0.1, msg_delay=0.02,
            jitter_type=JitterType.GAUSSIAN, jitter_ratio=0.1, seed=45,
        )

    return build_v4(
        target_temp=target_temp,
        initial_temp_a=initial_temp_a,
        initial_temp_b=initial_temp_b,
        max_steps=max_steps,
        ambient_temp=ambient_temp,
        cooling_rate=cooling_rate,
        coupling_rate=coupling_rate,
        heater_a_schedule=heater_a_schedule,
        heater_b_schedule=heater_b_schedule,
        ctrl_schedule=ctrl_schedule,
        vent_schedule=vent_schedule,
        overheat_threshold=overheat_threshold,
        enable_condition_monitor=enable_condition_monitor,
        disturbance_schedule=disturbance_schedule,
    )

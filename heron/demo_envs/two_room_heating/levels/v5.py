"""Level 5: Custom agent-to-agent events.

Same structure as v4. ``HeaterAgent`` has an
``@Agent.custom_handler("overheat_alert")`` decorator that registers
a handler at class definition time. In event-driven mode, custom events
can be dispatched via ``scheduler.schedule_custom_event()``. In
step-based training the handler exists but is not triggered.

Structurally identical to v4 -- the new capability is latent in the
agent class definition, not in the environment wiring.

Hierarchy::

    system_agent -> [heater_a (1s), heater_b (3s),
                     building_ctrl (5s) -> vent (reactive)]
"""

from typing import Optional

from heron.demo_envs.two_room_heating.levels.v4 import build_v4
from heron.demo_envs.two_room_heating.physics import TwoRoomEnv
from heron.scheduling import DisturbanceSchedule
from heron.scheduling.schedule_config import ScheduleConfig


def build_v5(
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
    """Build a v5 TwoRoomHeating environment."""
    env = build_v4(
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
    env.env_id = "two_room_v5"
    return env

"""Level 3: Horizontal protocol (peer state sharing).

Same structure as v2. The concept this level teaches is that peers can
share state directly. In step-based mode, observation scoping already
makes upper-level features visible via the coordinator. In event-driven
mode, a ``HorizontalProtocol`` would enable direct peer-to-peer message
exchange -- that wire-up is planned but not yet implemented; this level
currently serves as a structural placeholder with its own ``env_id``.

Hierarchy::

    system_agent -> [heater_a (1s), heater_b (3s),
                     building_ctrl (5s) -> vent (reactive)]
"""

from typing import Optional

from heron.demo_envs.two_room_heating.levels.v2 import build_v2
from heron.envs.simple import DefaultHeronEnv
from heron.scheduling.schedule_config import ScheduleConfig


def build_v3(
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
    """Build a v3 TwoRoomHeating environment."""
    env = build_v2(
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
    )
    env.env_id = "two_room_v3"
    return env

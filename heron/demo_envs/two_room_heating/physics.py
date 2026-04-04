"""Shared physics, env subclass, and helpers for TwoRoomHeating levels."""

import copy
from functools import partial
from typing import Any, Dict, Optional

from heron.envs.simple import DefaultHeronEnv
from heron.scheduling import ConditionMonitor, Disturbance, DisturbanceSchedule
from heron.scheduling.schedule_config import ScheduleConfig


# ---------------------------------------------------------------------------
#  Physics simulation
# ---------------------------------------------------------------------------


def two_room_simulation(
    agent_states: Dict[str, Dict],
    *,
    ambient_temp: float = 15.0,
    cooling_rate: float = 0.05,
    coupling_rate: float = 0.03,
    enable_vent: bool = False,
    vent_cooling_rate: float = 0.5,
) -> Dict[str, Dict]:
    """Shared physics for all TwoRoomHeating levels.

    Steps:
        1. Ambient cooling per zone.
        2. Thermal coupling between zones.
        3. (Optional) Vent cooling.
        4. Clip temperatures to [5.0, 40.0].

    Uses immutable dict update pattern -- never mutates the input dicts.
    """
    zone_temps: Dict[str, float] = {}
    for agent_id, features in agent_states.items():
        if "ZoneTemperatureFeature" in features:
            zone_temps[agent_id] = features["ZoneTemperatureFeature"]["temperature"]

    vent_is_open = 0.0
    if enable_vent:
        for agent_id, features in agent_states.items():
            if "VentStatusFeature" in features:
                vent_is_open = features["VentStatusFeature"].get("is_open", 0.0)
                break

    new_temps: Dict[str, float] = {}
    zone_ids = list(zone_temps.keys())
    for zone_id in zone_ids:
        t = zone_temps[zone_id]
        t = t + cooling_rate * (ambient_temp - t)
        for other_id in zone_ids:
            if other_id != zone_id:
                t = t + coupling_rate * (zone_temps[other_id] - t)
        if enable_vent:
            t = t - vent_cooling_rate * vent_is_open
        t = max(5.0, min(40.0, t))
        new_temps[zone_id] = t

    updated: Dict[str, Dict] = {}
    for agent_id, features in agent_states.items():
        if agent_id in new_temps:
            temp_data = features["ZoneTemperatureFeature"]
            updated[agent_id] = {
                **features,
                "ZoneTemperatureFeature": {
                    **temp_data,
                    "temperature": new_temps[agent_id],
                },
            }
        else:
            updated[agent_id] = features

    return updated


def make_sim(
    ambient_temp: float = 15.0,
    cooling_rate: float = 0.05,
    coupling_rate: float = 0.03,
    enable_vent: bool = False,
) -> Any:
    """Create a simulation closure with the given physics parameters."""
    return partial(
        two_room_simulation,
        ambient_temp=ambient_temp,
        cooling_rate=cooling_rate,
        coupling_rate=coupling_rate,
        enable_vent=enable_vent,
    )


# ---------------------------------------------------------------------------
#  Custom env subclass for disturbance handling (v4+)
# ---------------------------------------------------------------------------


class TwoRoomEnv(DefaultHeronEnv):
    """DefaultHeronEnv subclass that handles ambient-temperature disturbances.

    Stores a mutable ``_ambient_temp`` that can be modified by exogenous
    disturbance events (cold snaps, heat waves) during event-driven execution.
    """

    def __init__(
        self,
        *args: Any,
        base_ambient_temp: float = 15.0,
        cooling_rate: float = 0.05,
        coupling_rate: float = 0.03,
        enable_vent: bool = False,
        vent_cooling_rate: float = 0.5,
        disturbance_schedule: Optional[DisturbanceSchedule] = None,
        **kwargs: Any,
    ) -> None:
        self._base_ambient_temp = base_ambient_temp
        self._ambient_temp = base_ambient_temp
        self._cooling_rate = cooling_rate
        self._coupling_rate = coupling_rate
        self._enable_vent = enable_vent
        self._vent_cooling_rate = vent_cooling_rate
        self.disturbance_schedule = disturbance_schedule

        def _sim(agent_states: Dict[str, Dict]) -> Dict[str, Dict]:
            return two_room_simulation(
                agent_states,
                ambient_temp=self._ambient_temp,
                cooling_rate=self._cooling_rate,
                coupling_rate=self._coupling_rate,
                enable_vent=self._enable_vent,
                vent_cooling_rate=self._vent_cooling_rate,
            )

        super().__init__(*args, simulation_func=_sim, **kwargs)

    def apply_disturbance(self, disturbance: Any) -> None:
        """Handle cold_snap and heat_wave disturbances by modifying ambient temp."""
        d_type = getattr(disturbance, "disturbance_type", "")
        payload = getattr(disturbance, "payload", {})

        if d_type == "cold_snap":
            drop = payload.get("ambient_drop", 0.0)
            self._ambient_temp = self._ambient_temp - drop
        elif d_type == "heat_wave":
            rise = payload.get("ambient_rise", 0.0)
            self._ambient_temp = self._ambient_temp + rise
        else:
            raise NotImplementedError(
                f"TwoRoomEnv does not handle disturbance type '{d_type}'."
            )

    def reset(self, **kwargs: Any) -> Any:
        """Reset ambient temperature back to base before episode start."""
        self._ambient_temp = self._base_ambient_temp
        return super().reset(**kwargs)


# ---------------------------------------------------------------------------
#  Shared constants and helpers
# ---------------------------------------------------------------------------

DEFAULT_DISTURBANCE_SCHEDULE = DisturbanceSchedule([
    Disturbance(10.0, "cold_snap", {"ambient_drop": 5.0}),
    Disturbance(30.0, "cold_snap", {"ambient_drop": 8.0}),
    Disturbance(50.0, "heat_wave", {"ambient_rise": 10.0}),
])


def register_overheat_monitors(
    env: DefaultHeronEnv,
    threshold: float,
    cooldown: float = 2.0,
) -> None:
    """Register ConditionMonitor thresholds for both heater zones.

    When a zone's temperature exceeds *threshold*, the vent agent is
    woken via a CONDITION_TRIGGER event (even if it is not periodically
    scheduled).
    """
    for zone in ("heater_a", "heater_b"):
        env.scheduler.register_condition(
            ConditionMonitor.threshold(
                monitor_id=f"overheat_{zone[-1]}",
                agent_id="vent",
                key_path=[
                    "agent_states", zone, "features",
                    "ZoneTemperatureFeature", "temperature",
                ],
                threshold=threshold,
                direction="above",
                cooldown=cooldown,
            )
        )


def default_heater_a_schedule() -> ScheduleConfig:
    return ScheduleConfig.deterministic(tick_interval=1.0)


def default_heater_b_schedule() -> ScheduleConfig:
    return ScheduleConfig.deterministic(tick_interval=3.0)


def default_ctrl_schedule() -> ScheduleConfig:
    return ScheduleConfig.deterministic(tick_interval=5.0)


def default_disturbance_schedule() -> DisturbanceSchedule:
    return copy.deepcopy(DEFAULT_DISTURBANCE_SCHEDULE)

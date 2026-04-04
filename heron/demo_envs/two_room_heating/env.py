"""TwoRoomHeating demo environments (v0-v7).

A progressive 8-level demo for the Heron multi-agent framework.
Two rooms with heaters, each level adding one concept:

    v0 -- Heterogeneous tick rates
    v1 -- Coordinator (observer)
    v2 -- Reactive vent + condition trigger
    v3 -- Horizontal protocol (peer sharing)
    v4 -- Exogenous disturbances
    v5 -- Custom events
    v6 -- Jitter + delays
    v7 -- Multi-level hierarchy
"""

import copy
from functools import partial
from typing import Any, Dict, Optional

from heron.demo_envs.two_room_heating.agents import HeaterAgent, VentAgent
from heron.demo_envs.two_room_heating.features import (
    VentStatusFeature,
    ZoneTemperatureFeature,
)
from heron.envs.builder import EnvBuilder
from heron.envs.simple import DefaultHeronEnv
from heron.protocols.vertical import (
    BroadcastActionProtocol,
    VectorDecompositionActionProtocol,
    VerticalProtocol,
)
from heron.scheduling import ConditionMonitor, Disturbance, DisturbanceSchedule
from heron.scheduling.schedule_config import JitterType, ScheduleConfig


# ---------------------------------------------------------------------------
#  Physics simulation
# ---------------------------------------------------------------------------

def _two_room_simulation(
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
    # Collect zone temperatures keyed by agent_id.
    zone_temps: Dict[str, float] = {}
    for agent_id, features in agent_states.items():
        if "ZoneTemperatureFeature" in features:
            zone_temps[agent_id] = features["ZoneTemperatureFeature"]["temperature"]

    # Collect vent status if enabled.
    vent_is_open = 0.0
    if enable_vent:
        for agent_id, features in agent_states.items():
            if "VentStatusFeature" in features:
                vent_is_open = features["VentStatusFeature"].get("is_open", 0.0)
                break

    # Compute new temperatures (ambient cooling + coupling + vent).
    new_temps: Dict[str, float] = {}
    zone_ids = list(zone_temps.keys())
    for zone_id in zone_ids:
        t = zone_temps[zone_id]

        # Ambient cooling
        t = t + cooling_rate * (ambient_temp - t)

        # Thermal coupling with other zones
        for other_id in zone_ids:
            if other_id != zone_id:
                t = t + coupling_rate * (zone_temps[other_id] - t)

        # Vent cooling
        if enable_vent:
            t = t - vent_cooling_rate * vent_is_open

        # Clip
        t = max(5.0, min(40.0, t))
        new_temps[zone_id] = t

    # Build updated agent_states using immutable pattern.
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

        # Build simulation closure that reads self._ambient_temp at call time.
        def _sim(agent_states: Dict[str, Dict]) -> Dict[str, Dict]:
            return _two_room_simulation(
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
#  Build helpers
# ---------------------------------------------------------------------------

_DEFAULT_DISTURBANCE_SCHEDULE = DisturbanceSchedule([
    Disturbance(10.0, "cold_snap", {"ambient_drop": 5.0}),
    Disturbance(30.0, "cold_snap", {"ambient_drop": 8.0}),
    Disturbance(50.0, "heat_wave", {"ambient_rise": 10.0}),
])


def _register_overheat_monitors(
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


def _default_heater_a_schedule() -> ScheduleConfig:
    return ScheduleConfig.deterministic(tick_interval=1.0)


def _default_heater_b_schedule() -> ScheduleConfig:
    return ScheduleConfig.deterministic(tick_interval=3.0)


def _default_ctrl_schedule() -> ScheduleConfig:
    return ScheduleConfig.deterministic(tick_interval=5.0)


# ---------------------------------------------------------------------------
#  v0 -- Heterogeneous tick rates
# ---------------------------------------------------------------------------

def build_v0(
    target_temp: float = 22.0,
    initial_temp_a: float = 18.0,
    initial_temp_b: float = 18.0,
    max_steps: int = 200,
    ambient_temp: float = 15.0,
    cooling_rate: float = 0.05,
    coupling_rate: float = 0.03,
    heater_a_schedule: Optional[ScheduleConfig] = None,
    heater_b_schedule: Optional[ScheduleConfig] = None,
) -> DefaultHeronEnv:
    """Level 0: Two heaters with heterogeneous tick rates.

    Hierarchy::

        system_agent -> [heater_a (1s), heater_b (3s)]

    Concept: heater_a acts every 1 s, heater_b acts every 3 s.
    No coordinator.
    """
    sim = partial(
        _two_room_simulation,
        ambient_temp=ambient_temp,
        cooling_rate=cooling_rate,
        coupling_rate=coupling_rate,
    )
    env = (
        EnvBuilder("two_room_v0")
        .add_agent(
            "heater_a",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_a, target=target_temp)],
            schedule_config=heater_a_schedule or _default_heater_a_schedule(),
            target_temp=target_temp,
            initial_temp=initial_temp_a,
            heat_gain=2.0,
        )
        .add_agent(
            "heater_b",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_b, target=target_temp)],
            schedule_config=heater_b_schedule or _default_heater_b_schedule(),
            target_temp=target_temp,
            initial_temp=initial_temp_b,
            heat_gain=3.0,
        )
        .simulation(sim)
        .termination(max_steps=max_steps)
        .build()
    )
    return env


# ---------------------------------------------------------------------------
#  v1 -- Coordinator (observer)
# ---------------------------------------------------------------------------

def build_v1(
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
) -> DefaultHeronEnv:
    """Level 1: Observer coordinator added.

    Hierarchy::

        system_agent -> [heater_a (1s), heater_b (3s), building_ctrl (5s)]

    Concept: building_ctrl is a CoordinatorAgent that observes public
    features and outputs a budget signal.  It has *no* subordinates -- it
    is a sibling that passively monitors.
    """
    sim = partial(
        _two_room_simulation,
        ambient_temp=ambient_temp,
        cooling_rate=cooling_rate,
        coupling_rate=coupling_rate,
    )
    env = (
        EnvBuilder("two_room_v1")
        .add_agent(
            "heater_a",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_a, target=target_temp)],
            schedule_config=heater_a_schedule or _default_heater_a_schedule(),
            target_temp=target_temp,
            initial_temp=initial_temp_a,
            heat_gain=2.0,
        )
        .add_agent(
            "heater_b",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_b, target=target_temp)],
            schedule_config=heater_b_schedule or _default_heater_b_schedule(),
            target_temp=target_temp,
            initial_temp=initial_temp_b,
            heat_gain=3.0,
        )
        .add_coordinator(
            "building_ctrl",
            schedule_config=ctrl_schedule or _default_ctrl_schedule(),
        )
        .simulation(sim)
        .termination(max_steps=max_steps)
        .build()
    )
    return env


# ---------------------------------------------------------------------------
#  v2 -- Reactive vent + condition trigger
# ---------------------------------------------------------------------------

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
    """Level 2: Reactive vent agent under building_ctrl with condition triggers.

    Hierarchy::

        system_agent -> [heater_a (1s), heater_b (3s),
                         building_ctrl (5s) -> vent (reactive)]

    Concept: building_ctrl now has a subordinate vent agent connected via
    BroadcastActionProtocol.  Optionally enables ConditionMonitor threshold
    triggers that wake the vent when any zone overheats.
    """
    sim = partial(
        _two_room_simulation,
        ambient_temp=ambient_temp,
        cooling_rate=cooling_rate,
        coupling_rate=coupling_rate,
        enable_vent=True,
    )
    env = (
        EnvBuilder("two_room_v2")
        .add_agent(
            "heater_a",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_a, target=target_temp)],
            schedule_config=heater_a_schedule or _default_heater_a_schedule(),
            target_temp=target_temp,
            initial_temp=initial_temp_a,
            heat_gain=2.0,
        )
        .add_agent(
            "heater_b",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_b, target=target_temp)],
            schedule_config=heater_b_schedule or _default_heater_b_schedule(),
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
            schedule_config=ctrl_schedule or _default_ctrl_schedule(),
        )
        .simulation(sim)
        .termination(max_steps=max_steps)
        .build()
    )

    if enable_condition_monitor:
        _register_overheat_monitors(env, overheat_threshold)

    return env


# ---------------------------------------------------------------------------
#  v3 -- Horizontal protocol (peer sharing)
# ---------------------------------------------------------------------------

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
    enable_horizontal: bool = True,
) -> DefaultHeronEnv:
    """Level 3: Horizontal protocol for peer temperature sharing.

    Hierarchy::

        system_agent -> [heater_a (1s), heater_b (3s),
                         building_ctrl (5s) -> vent (reactive)]

    Same structure as v2.  The concept this level teaches is that peers
    can share state directly.  In step-based mode, observation scoping
    already makes upper-level features visible via the coordinator.
    In event-driven mode, a ``HorizontalProtocol`` would enable direct
    peer-to-peer message exchange — that wire-up is planned but not yet
    implemented; this level currently serves as a structural placeholder
    with its own ``env_id`` for the progression.

    .. note:: ``enable_horizontal`` is accepted for forward-compatibility
       but currently has no effect.
    """
    # Identical to v2 structurally; distinct env_id for introspection.
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


# ---------------------------------------------------------------------------
#  v4 -- Exogenous disturbances
# ---------------------------------------------------------------------------

def build_v4(
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
    """Level 4: Exogenous disturbances (cold snaps, heat waves).

    Hierarchy::

        system_agent -> [heater_a (1s), heater_b (3s),
                         building_ctrl (5s) -> vent (reactive)]

    Concept: Uses ``TwoRoomEnv`` subclass that handles ambient-temperature
    disturbances during event-driven execution.  Default disturbance
    schedule includes cold snaps at t=10, t=30, and a heat wave at t=50.
    """
    if disturbance_schedule is None:
        disturbance_schedule = copy.deepcopy(_DEFAULT_DISTURBANCE_SCHEDULE)

    env = (
        EnvBuilder("two_room_v4")
        .add_agent(
            "heater_a",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_a, target=target_temp)],
            schedule_config=heater_a_schedule or _default_heater_a_schedule(),
            target_temp=target_temp,
            initial_temp=initial_temp_a,
            heat_gain=2.0,
        )
        .add_agent(
            "heater_b",
            HeaterAgent,
            features=[ZoneTemperatureFeature(temperature=initial_temp_b, target=target_temp)],
            schedule_config=heater_b_schedule or _default_heater_b_schedule(),
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
            schedule_config=ctrl_schedule or _default_ctrl_schedule(),
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
        _register_overheat_monitors(env, overheat_threshold)

    return env


# ---------------------------------------------------------------------------
#  v5 -- Custom events
# ---------------------------------------------------------------------------

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
    """Level 5: Custom agent-to-agent events.

    Hierarchy::

        system_agent -> [heater_a (1s), heater_b (3s),
                         building_ctrl (5s) -> vent (reactive)]

    Same structure as v4.  HeaterAgent has an ``@Agent.custom_handler("overheat_alert")``
    decorator so that during event-driven execution, when one heater detects
    T > threshold it can send a custom event to the other heater, which then
    clamps its next action to cooling-only.

    The ``@Agent.custom_handler("overheat_alert")`` decorator on
    ``HeaterAgent`` registers the handler at class definition time.
    In event-driven mode, custom events can be dispatched via
    ``scheduler.schedule_custom_event()``.  In step-based training
    the handler exists but is not triggered.

    Structurally identical to v4 — the new capability is latent in
    the agent class definition, not in the environment wiring.
    """
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


# ---------------------------------------------------------------------------
#  v6 -- Jitter + delays
# ---------------------------------------------------------------------------

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
    """Level 6: Jitter and communication delays on all agents.

    Hierarchy::

        system_agent -> [heater_a (1s), heater_b (3s),
                         building_ctrl (5s) -> vent (reactive)]

    Same structure as v5 but all ScheduleConfigs use ``with_jitter()``
    with Gaussian noise (10% ratio) for realistic distributed-system
    simulation.  User-supplied schedules override the jittered defaults.
    """
    if heater_a_schedule is None:
        heater_a_schedule = ScheduleConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.05,
            act_delay=0.1,
            msg_delay=0.02,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=42,
        )
    if heater_b_schedule is None:
        heater_b_schedule = ScheduleConfig.with_jitter(
            tick_interval=3.0,
            obs_delay=0.05,
            act_delay=0.1,
            msg_delay=0.02,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=43,
        )
    if ctrl_schedule is None:
        ctrl_schedule = ScheduleConfig.with_jitter(
            tick_interval=5.0,
            obs_delay=0.05,
            act_delay=0.1,
            msg_delay=0.02,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=44,
        )
    if vent_schedule is None:
        vent_schedule = ScheduleConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.05,
            act_delay=0.1,
            msg_delay=0.02,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=45,
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


# ---------------------------------------------------------------------------
#  v7 -- Multi-level hierarchy
# ---------------------------------------------------------------------------

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
    """Level 7: Multi-level hierarchy with floor controller.

    Hierarchy::

        system_agent -> floor_ctrl (10s, VectorDecomposition)
            -> [heater_a (reactive), heater_b (reactive),
                building_ctrl (reactive) -> vent (reactive)]

    Concept: ``floor_ctrl`` is a periodic top-level coordinator using
    VectorDecompositionActionProtocol.  All agents underneath become
    reactive (ticked by their parent coordinator).
    """
    if disturbance_schedule is None:
        disturbance_schedule = copy.deepcopy(_DEFAULT_DISTURBANCE_SCHEDULE)

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
        _register_overheat_monitors(env, overheat_threshold)

    return env

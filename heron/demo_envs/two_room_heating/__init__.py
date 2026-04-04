"""TwoRoomHeating progressive demo environments (v0-v7).

Each version builds on the previous, teaching one new Heron concept:

    from heron.demo_envs.two_room_heating import LEVELS
    for lvl in LEVELS:
        print(f"v{lvl['version']}: {lvl['title']} -- {lvl['concept']}")

Usage::

    import numpy as np
    import heron
    import heron.demo_envs

    env = heron.make("TwoRoomHeating-v0")   # start here
    obs, _ = env.reset()
    obs, rewards, terminated, truncated, infos = env.step({
        "heater_a": np.array([0.5]),
        "heater_b": np.array([0.3]),
    })
"""

import heron

from heron.demo_envs.two_room_heating.features import (
    ZoneTemperatureFeature,
    VentStatusFeature,
)
from heron.demo_envs.two_room_heating.agents import HeaterAgent, VentAgent
from heron.demo_envs.two_room_heating.levels import (
    build_v0,
    build_v1,
    build_v2,
    build_v3,
    build_v4,
    build_v5,
    build_v6,
    build_v7,
)

LEVELS = [
    {
        "version": 0,
        "title": "Heterogeneous tick rates",
        "concept": "Agents tick at different intervals (1s vs 3s)",
        "entry_point": build_v0,
    },
    {
        "version": 1,
        "title": "Coordinator hierarchy",
        "concept": "A coordinator observes zones and outputs a heating budget",
        "entry_point": build_v1,
    },
    {
        "version": 2,
        "title": "Reactive agent + condition trigger",
        "concept": "A vent agent activates only when temperature exceeds threshold",
        "entry_point": build_v2,
    },
    {
        "version": 3,
        "title": "Horizontal protocol",
        "concept": "Peer-to-peer state sharing concept; event-driven HorizontalProtocol wire-up planned",
        "entry_point": build_v3,
    },
    {
        "version": 4,
        "title": "Exogenous disturbances",
        "concept": "External weather events (cold snaps, heat waves) perturb the environment",
        "entry_point": build_v4,
    },
    {
        "version": 5,
        "title": "Custom events",
        "concept": "Agents send domain-specific signals (overheat alerts) to each other",
        "entry_point": build_v5,
    },
    {
        "version": 6,
        "title": "Jitter + communication delays",
        "concept": "Realistic observation, action, and message delays with Gaussian jitter",
        "entry_point": build_v6,
    },
    {
        "version": 7,
        "title": "Multi-level hierarchy",
        "concept": "Nested coordinators (floor -> building -> agents) with cascading coordination",
        "entry_point": build_v7,
    },
]

# Register all versions
for _lvl in LEVELS:
    heron.register(
        env_id=f"TwoRoomHeating-v{_lvl['version']}",
        entry_point=_lvl["entry_point"],
        kwargs={},
    )

__all__ = [
    "LEVELS",
    "ZoneTemperatureFeature",
    "VentStatusFeature",
    "HeaterAgent",
    "VentAgent",
    "build_v0",
    "build_v1",
    "build_v2",
    "build_v3",
    "build_v4",
    "build_v5",
    "build_v6",
    "build_v7",
]

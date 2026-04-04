"""Transport Fleet demo environment -- multi-vehicle delivery coordination.

Teaches: multiple field agents, coordinator with reward, vertical protocol.

Usage::

    import heron
    import heron.demo_envs  # auto-registers demo envs

    env = heron.make("TransportFleet-v0")
    obs = env.reset()
    actions = {aid: [0.1, 0.2] for aid in obs}
    obs, rewards, terminated, truncated, infos = env.step(actions)
"""

from heron.demo_envs.transport_fleet.env import build_transport_fleet_env
from heron.demo_envs.transport_fleet.agents import (
    DepotCoordinator,
    DepotFeature,
    VehicleAgent,
    VehicleFeature,
)

import heron

heron.register(
    env_id="TransportFleet-v0",
    entry_point=build_transport_fleet_env,
    kwargs={},
)

__all__ = [
    "build_transport_fleet_env",
    "DepotCoordinator",
    "DepotFeature",
    "VehicleAgent",
    "VehicleFeature",
]

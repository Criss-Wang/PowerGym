"""Sensor network demo environment -- visibility + horizontal protocol.

Teaches: graph-based observation scoping, horizontal peer coordination,
discrete binary actions, signal detection reward.

Usage::

    import heron
    import heron.demo_envs  # auto-registers demo envs

    env = heron.make("SensorNetwork-v0")
    obs = env.reset()
    actions = {f"sensor_{i}": 1 for i in range(5)}
    obs, rewards, terminated, truncated, infos = env.step(actions)
"""

from heron.demo_envs.sensor_network.env import build_sensor_network_env
from heron.demo_envs.sensor_network.agents import SensorAgent, SensorFeature

import heron

heron.register(
    env_id="SensorNetwork-v0",
    entry_point=build_sensor_network_env,
    kwargs={},
)

__all__ = [
    "build_sensor_network_env",
    "SensorAgent",
    "SensorFeature",
]

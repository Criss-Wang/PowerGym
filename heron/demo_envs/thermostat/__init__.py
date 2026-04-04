"""Thermostat demo environment — HERON's "Hello World".

Teaches: single field agent, continuous action, simple reward, episode truncation.

Usage::

    import numpy as np
    import heron
    import heron.demo_envs  # auto-registers demo envs

    env = heron.make("Thermostat-v0")
    obs, _ = env.reset()
    obs, rewards, terminated, truncated, infos = env.step({"heater": np.array([0.5])})
"""

import heron

from heron.demo_envs.thermostat.env import build_thermostat_env
from heron.demo_envs.thermostat.agents import HeaterAgent, TemperatureFeature

heron.register(
    env_id="Thermostat-v0",
    entry_point=build_thermostat_env,
    kwargs={},
)

__all__ = [
    "build_thermostat_env",
    "HeaterAgent",
    "TemperatureFeature",
]

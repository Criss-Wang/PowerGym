"""Thermostat demo environment — HERON's "Hello World".

Teaches: single field agent, continuous action, simple reward, episode truncation.

Usage::

    import heron
    import heron.demo_envs  # auto-registers demo envs

    env = heron.make("Thermostat-v0")
    obs = env.reset()
    obs, rewards, terminated, truncated, infos = env.step({"heater": 0.5})
"""

from heron.demo_envs.thermostat.env import build_thermostat_env
from heron.demo_envs.thermostat.agents import HeaterAgent, TemperatureFeature

import heron

heron.register(
    id="Thermostat-v0",
    entry_point=build_thermostat_env,
    kwargs={},
)

__all__ = [
    "build_thermostat_env",
    "HeaterAgent",
    "TemperatureFeature",
]

"""Thermostat environment — "Hello World" demo for HERON.

A single heater agent controls room temperature via a continuous action.
Physics: ``T_next = T + heat_delta + noise - cooling_rate``.
"""

from typing import Any, Dict, Optional

import numpy as np

from heron.agents.system_agent import SystemAgent
from heron.envs.simple import DefaultHeronEnv
from heron.envs.builder import EnvBuilder
from heron.envs.termination import TerminationConfig
from heron.demo_envs.thermostat.agents import HeaterAgent


def _thermostat_simulation(
    agent_states: Dict[str, Dict],
    *,
    cooling_rate: float = 0.1,
    noise_scale: float = 0.05,
) -> Dict[str, Dict]:
    """Physics step: cool toward ambient, add noise.

    Args:
        agent_states: ``{agent_id: {feature_name: {field: val}}}``
        cooling_rate: Passive cooling per step.
        noise_scale: Standard deviation of temperature noise.

    Returns:
        Updated agent_states dict.
    """
    updated = dict(agent_states)
    for aid, features in updated.items():
        temp_data = features.get("TemperatureFeature")
        if temp_data is None:
            continue
        t = temp_data["temperature"]
        noise = np.random.normal(0, noise_scale)
        updated[aid] = {
            **features,
            "TemperatureFeature": {**temp_data, "temperature": t + noise - cooling_rate},
        }
    return updated


def build_thermostat_env(
    target_temp: float = 22.0,
    initial_temp: float = 18.0,
    max_steps: int = 100,
    cooling_rate: float = 0.1,
    noise_scale: float = 0.05,
) -> DefaultHeronEnv:
    """Build a Thermostat-v0 environment.

    Args:
        target_temp: Target temperature for the heater agent.
        initial_temp: Starting room temperature.
        max_steps: Episode length (truncation).
        cooling_rate: Passive cooling per step.
        noise_scale: Temperature noise std.

    Returns:
        Configured ``DefaultHeronEnv``.
    """
    from heron.demo_envs.thermostat.agents import TemperatureFeature

    def sim_func(agent_states: Dict[str, Dict]) -> Dict[str, Dict]:
        return _thermostat_simulation(
            agent_states,
            cooling_rate=cooling_rate,
            noise_scale=noise_scale,
        )

    env = (
        EnvBuilder("thermostat")
        .add_agent(
            "heater",
            HeaterAgent,
            features=[TemperatureFeature(temperature=initial_temp)],
            target_temp=target_temp,
            initial_temp=initial_temp,
        )
        .simulation(sim_func)
        .termination(max_steps=max_steps)
        .build()
    )
    return env

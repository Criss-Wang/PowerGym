#!/usr/bin/env python3
"""Debug script to check observation structure"""

import numpy as np
from dataclasses import dataclass
from typing import List, ClassVar, Sequence

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import FeatureProvider
from heron.envs.simple import SimpleEnv
from heron.scheduling import TickConfig


@dataclass(slots=True)
class QueueFeature(FeatureProvider):
    visibility: ClassVar[Sequence[str]] = ["public"]
    queue: float = 0.0


class Station(FieldAgent):
    def init_action(self, features: List[FeatureProvider] = None) -> Action:
        if features is None:
            features = []
        a = Action()
        a.set_specs(dim_c=1, range=(np.array([0.0]), np.array([1.0])))
        return a

    def set_action(self, action, *args, **kwargs):
        self.action.set_values(action)

    def set_state(self, **kwargs):
        if "queue" in kwargs:
            self.state.features["QueueFeature"].set_values(queue=kwargs["queue"])

    def apply_action(self):
        pass

    def compute_local_reward(self, local_state: dict) -> float:
        return 0.0


def sim_fn(agent_states):
    return agent_states


# Build env
stations = {
    f"s{i}": Station(
        agent_id=f"s{i}",
        features=[QueueFeature()],
        tick_config=TickConfig.deterministic(),
    )
    for i in range(3)
}

env = SimpleEnv(
    coordinator_agents=[CoordinatorAgent(agent_id="coord", subordinates=stations)],
    simulation_func=sim_fn,
)

# Reset and check
obs, _ = env.reset()
print(f"Observation keys: {list(obs.keys())}")

for sid in ["s0", "s1", "s2"]:
    if sid in obs:
        obs_data = obs[sid]
        print(f"\n{sid}:")
        print(f"  Type: {type(obs_data)}")
        print(f"  Has local_vector: {hasattr(obs_data, 'local_vector')}")

        if hasattr(obs_data, 'local'):
            print(f"  local dict: {obs_data.local}")

        if hasattr(obs_data, 'local_vector'):
            vec = obs_data.local_vector()
            print(f"  local_vector shape: {vec.shape}")
            print(f"  local_vector values: {vec}")

        if hasattr(obs_data, 'vector'):
            vec = obs_data.vector()
            print(f"  full vector shape: {vec.shape}")
            print(f"  full vector values: {vec}")


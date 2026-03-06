#!/usr/bin/env python3
"""
Minimal Version: Testing Basic HERON Functionality
"""

import numpy as np
from dataclasses import dataclass
from typing import List, ClassVar, Sequence

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.feature import FeatureProvider
from heron.core.policies import Policy
from heron.envs.simple import SimpleEnv


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
        queue = float(local_state.get("QueueFeature", [0.0])[0])
        return min(queue, 1.0)


class SimplePolicy(Policy):
    observation_mode = "local"

    def __init__(self):
        self.w = None
        self.returns = []
        self.last_obs = None
        self.last_action = None

    def forward(self, obs):
        # Extract vector
        if hasattr(obs, 'local_vector'):
            obs_vec = obs.local_vector()
        elif hasattr(obs, 'vector'):
            obs_vec = obs.vector()
        else:
            obs_vec = np.asarray(obs).flatten()

        # Initialize weights lazily
        if self.w is None:
            self.w = np.random.randn(len(obs_vec)) * 0.1

        # Action
        action_val = 0.5 * (1.0 + np.tanh(np.dot(obs_vec, self.w)))
        self.last_obs = obs_vec.copy()
        self.last_action = action_val

        # Create Action
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([0.0]), np.array([1.0])))
        action.set_values(c=np.array([action_val]))
        return action

    def learn(self, reward):
        self.returns.append(reward)
        if self.last_obs is not None:
            self.w += 0.01 * reward * self.last_action * self.last_obs


def sim_fn(agent_states):
    for aid in agent_states:
        if "QueueFeature" in agent_states[aid]:
            queue = float(agent_states[aid]["QueueFeature"]["queue"])
            queue = max(0.0, queue + np.random.poisson(0.5) - np.random.randint(0, 2))
            agent_states[aid]["QueueFeature"]["queue"] = min(10.0, queue)
    return agent_states


def main():
    print("\nHERON Demo: Testing Basic Sync Training")
    print("=" * 60)

    # Build environment

    # Build environment
    stations = {
        "s0": Station(agent_id="s0", features=[QueueFeature()]),
        "s1": Station(agent_id="s1", features=[QueueFeature()]),
    }

    env = SimpleEnv(
        coordinator_agents=[CoordinatorAgent(agent_id="coord", subordinates=stations)],
        simulation_func=sim_fn,
    )

    # Policies
    policies = {"s0": SimplePolicy(), "s1": SimplePolicy()}

    # Quick training
    print("\nTraining for 5 episodes...")
    for episode in range(5):
        obs, _ = env.reset()

        for step in range(5):
            actions = {}
            for sid in ["s0", "s1"]:
                if sid in obs:
                    action = policies[sid].forward(obs[sid])
                    actions[sid] = action

            obs, rewards, _, _, _ = env.step(actions)

            for sid in ["s0", "s1"]:
                if sid in rewards:
                    policies[sid].learn(rewards[sid])

        print(f"  Episode {episode + 1}: Returns = {[np.mean(p.returns[-5:]) if p.returns else 0.0 for p in policies.values()]}")

    print("\n✓ Basic training works!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()


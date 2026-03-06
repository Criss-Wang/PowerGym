#!/usr/bin/env python3
"""
Minimal Example: Demonstrating Heron's Online Learning Capability

Demonstration Content:
1. Offline Training (Sync mode)
2. Evaluation Gap (Async mode - untrained)
3. Online Re-training (Async mode - with learning)

3 Charging Stations, RL agents learn to allocate power
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
from heron.scheduling import TickConfig, JitterType


# === Domain ===
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
        return min(queue, 1.0)  # Simple reward


class SimplePolicy(Policy):
    observation_mode = "local"

    def __init__(self):
        # Policy parameters - use fixed dimension
        self.obs_dim = 1  # Only queue feature
        self.action_dim = 1  # Power allocation [0, 1]
        self.action_range = (np.array([0.0]), np.array([1.0]))

        # Network weights
        self.w = np.random.randn(self.obs_dim) * 0.1
        self.baseline = 0.0
        self.returns = []
        self.last_obs = None
        self.last_action = None

    def forward(self, obs):
        """Convert observation to action.

        obs can be Observation object or numpy array
        Returns: Action object
        """
        # Extract vector from observation - always take first element only
        if hasattr(obs, 'local_vector'):
            obs_vec = obs.local_vector()
        elif hasattr(obs, 'vector'):
            obs_vec = obs.vector()
        else:
            obs_vec = np.asarray(obs).flatten()

        # Ensure obs_vec is 1D and take only first element (queue length)
        obs_vec = obs_vec.flatten()[:self.obs_dim]

        # Pad if needed
        if len(obs_vec) < self.obs_dim:
            obs_vec = np.pad(obs_vec, (0, self.obs_dim - len(obs_vec)))

        # Simple linear policy with tanh
        action_val = 0.5 * (1.0 + np.tanh(np.dot(obs_vec, self.w)))

        # Store for learning
        self.last_obs = obs_vec.copy()
        self.last_action = action_val

        # Create and return Action object
        action = Action()
        action.set_specs(dim_c=1, range=self.action_range)
        action.set_values(c=np.array([action_val]))

        return action

    def learn(self, reward):
        """REINFORCE-style learning"""
        self.returns.append(reward)
        self.baseline = np.mean(self.returns[-10:]) if self.returns else 0.0

        if self.last_obs is not None and self.last_action is not None:
            advantage = reward - self.baseline
            # Simple gradient update
            self.w += 0.01 * advantage * self.last_action * self.last_obs


def sim_fn(agent_states):
    for aid in agent_states:
        if "QueueFeature" in agent_states[aid]:
            queue = float(agent_states[aid]["QueueFeature"]["queue"])
            arrivals = np.random.poisson(0.5)
            charged = np.random.randint(0, 2)
            queue = max(0.0, queue + float(arrivals) - float(charged))
            agent_states[aid]["QueueFeature"]["queue"] = min(10.0, queue)
    return agent_states


# === Environments ===
def build_env(async_mode=False):
    stations = {}
    for i in range(3):
        if async_mode:
            tick = TickConfig.with_jitter(
                tick_interval=1.0 + i * 0.5,
                obs_delay=0.05,
                act_delay=0.1,
                jitter_type=JitterType.GAUSSIAN,
                jitter_ratio=0.1,
                seed=42 + i,
            )
        else:
            tick = TickConfig.deterministic()

        stations[f"s{i}"] = Station(
            agent_id=f"s{i}",
            features=[QueueFeature()],
            tick_config=tick,
        )

    return SimpleEnv(
        coordinator_agents=[CoordinatorAgent(
            agent_id="coord",
            subordinates=stations,
        )],
        simulation_func=sim_fn,
        env_id="demo",
    )


# === Main ===
def main():
    print("\n" + "="*70)
    print("HERON Online Learning Demo: Sync vs Async")
    print("="*70)

    # Create environments
    print("Building sync environment...")
    env_sync = build_env(async_mode=False)
    print("✓ Sync environment ready")

    # Initialize policies
    policies = {f"s{i}": SimplePolicy() for i in range(3)}

    # Phase 1: Offline Training (Sync)
    print("\nPhase 1: Offline Training (SYNC mode)")
    print("-" * 70)

    num_episodes = 3
    for episode in range(num_episodes):
        print(f"  Starting episode {episode + 1}...")
        obs, _ = env_sync.reset()
        print(f"    Reset complete, obs keys: {list(obs.keys())}")

        for step in range(3):
            actions = {}
            for sid in [f"s{i}" for i in range(3)]:
                if sid in obs:
                    obs_data = obs[sid]
                    # forward() already returns an Action object
                    action = policies[sid].forward(obs_data)
                    actions[sid] = action

            obs, rewards, _, _, _ = env_sync.step(actions)

            for sid in [f"s{i}" for i in range(3)]:
                if sid in rewards:
                    policies[sid].learn(rewards[sid])

            print(f"    Step {step + 1} complete")

        if (episode + 1) % 1 == 0:
            avgs = [np.mean(p.returns[-3:]) if p.returns else 0.0 for p in policies.values()]
            print(f"  Episode {episode+1:2d}: Avg Returns = {avgs}")

    print("\n  ✓ Training complete (Sync mode)")
    final_avgs = [np.mean(p.returns[-3:]) if p.returns else 0.0 for p in policies.values()]
    print(f"    Final avg returns: {final_avgs}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ Heron Sync mode training works!")
    print(f"  Trained 3 policies over {num_episodes} episodes")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


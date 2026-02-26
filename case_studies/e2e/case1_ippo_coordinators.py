"""Case 1: IPPO training on 2 fleet managers, each coordinating 3 drones.

Hierarchy:
    SystemAgent
    ├── fleet_0 (FleetManager) — owns policy, action_space = Box(-1,1,(3,))
    │   ├── drone_0_0 (Drone) — receives 1D thrust command from fleet manager
    │   ├── drone_0_1 (Drone)
    │   └── drone_0_2 (Drone)
    └── fleet_1 (FleetManager)
        ├── drone_1_0 (Drone)
        ├── drone_1_1 (Drone)
        └── drone_1_2 (Drone)

Domain: Multi-drone heavy payload transport in a dynamic warehouse airspace.
    Each fleet manager assigns thrust targets to its drones. Drones must maintain
    stable flight at target altitudes while compensating for gravity drift.

Training: IPPO — each fleet manager has an independent policy.
Evaluation: event-driven with jittered tick configs.

Usage:
    python -m case_studies.e2e.case1_ippo_coordinators
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np
from gymnasium.spaces import Box

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import FeatureProvider
from heron.protocols.vertical import VerticalProtocol


# ── Features ──────────────────────────────────────────────────────

@dataclass(slots=True)
class DroneFeature(FeatureProvider):
    """Per-drone flight state."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    altitude: float = 0.5

    def set_values(self, **kwargs: Any) -> None:
        if "altitude" in kwargs:
            self.altitude = float(np.clip(kwargs["altitude"], 0.0, 1.0))


@dataclass(slots=True)
class FleetFeature(FeatureProvider):
    """Fleet-level aggregate state observed by fleet manager."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    mean_altitude: float = 0.5
    pending_orders: float = 0.3

    def set_values(self, **kwargs: Any) -> None:
        if "mean_altitude" in kwargs:
            self.mean_altitude = float(np.clip(kwargs["mean_altitude"], 0.0, 1.0))
        if "pending_orders" in kwargs:
            self.pending_orders = float(np.clip(kwargs["pending_orders"], 0.0, 1.0))


# ── Agents ────────────────────────────────────────────────────────

class Drone(FieldAgent):
    """Drone whose thrust is set by the fleet manager."""

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(c=np.zeros(1, dtype=np.float32))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        elif isinstance(action, np.ndarray):
            self.action.set_values(c=action.flatten()[:1])
        elif isinstance(action, (int, float)):
            self.action.set_values(c=np.array([float(action)], dtype=np.float32))

    def set_state(self, **kwargs) -> None:
        if "altitude" in kwargs:
            self.state.features["DroneFeature"].set_values(altitude=kwargs["altitude"])

    def apply_action(self) -> None:
        df = self.state.features["DroneFeature"]
        new_altitude = df.altitude + self.action.c[0] * 0.05
        df.set_values(altitude=new_altitude)

    def compute_local_reward(self, local_state: dict) -> float:
        df = local_state.get("DroneFeature")
        if df is None:
            return 0.0
        altitude = float(df[0])
        # Reward: proximity to target cruise altitude (0.7)
        return 1.0 - abs(altitude - 0.7)


NUM_DRONES_PER_FLEET = 3


class FleetManager(CoordinatorAgent):
    """Fleet manager that assigns thrust targets to its drones.

    Action: 3D vector in [-1, 1] — one thrust command per subordinate drone.
    VectorDecompositionActionProtocol splits this into per-drone 1D actions.
    """

    def __init__(self, agent_id, subordinates, features=None, **kwargs):
        n_subs = len(subordinates)
        default_features = [FleetFeature()]
        all_features = (features or []) + default_features

        super().__init__(
            agent_id=agent_id,
            features=all_features,
            subordinates=subordinates,
            protocol=VerticalProtocol(),
            **kwargs,
        )

        self.observation_space = Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = Box(-1.0, 1.0, (n_subs,), np.float32)

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        action = Action()
        action.set_specs(
            dim_c=NUM_DRONES_PER_FLEET,
            range=(np.full(NUM_DRONES_PER_FLEET, -1.0), np.full(NUM_DRONES_PER_FLEET, 1.0)),
        )
        action.set_values(c=np.zeros(NUM_DRONES_PER_FLEET, dtype=np.float32))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        if isinstance(action, Action):
            self.action = action
        elif isinstance(action, np.ndarray):
            self.action.set_values(c=action.astype(np.float32))
        elif isinstance(action, (int, float)):
            self.action.set_values(c=np.array([float(action)], dtype=np.float32))

    def compute_rewards(self, proxy) -> Dict[str, float]:
        sub_rewards: Dict[str, float] = {}
        for sub in self.subordinates.values():
            sub_rewards.update(sub.compute_rewards(proxy))
        coordinator_reward = sum(sub_rewards.values())
        rewards = {self.agent_id: coordinator_reward}
        rewards.update(sub_rewards)
        return rewards

    def compute_local_reward(self, local_state: dict) -> float:
        subordinate_rewards = local_state.get("subordinate_rewards", {})
        if subordinate_rewards:
            return sum(subordinate_rewards.values())
        ff = local_state.get("FleetFeature")
        if ff is None:
            return 0.0
        return float(ff[0])  # mean_altitude


# ── Simulation ────────────────────────────────────────────────────

def simulation(agent_states: dict) -> dict:
    """Apply gravity drift and update fleet features from drone altitudes."""
    fleet_altitudes: Dict[str, List[float]] = {}
    for aid, features in agent_states.items():
        if "DroneFeature" in features:
            # Gravity drift: drones slowly lose altitude
            altitude = features["DroneFeature"]["altitude"]
            altitude -= 0.01
            features["DroneFeature"]["altitude"] = float(np.clip(altitude, 0.0, 1.0))

            # drone_0_1 → fleet_0
            parts = aid.split("_")
            if len(parts) >= 2:
                fleet_id = f"fleet_{parts[1]}"
                fleet_altitudes.setdefault(fleet_id, []).append(
                    features["DroneFeature"]["altitude"]
                )

    # Update fleet features
    for fleet_id, altitudes in fleet_altitudes.items():
        if fleet_id in agent_states and "FleetFeature" in agent_states[fleet_id]:
            agent_states[fleet_id]["FleetFeature"]["mean_altitude"] = float(np.mean(altitudes))
            agent_states[fleet_id]["FleetFeature"]["pending_orders"] = float(
                np.random.uniform(0.1, 0.5)
            )

    return agent_states


# ── Training + Evaluation ─────────────────────────────────────────

def main():
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec

    from heron.adaptors.rllib import RLlibBasedHeronEnv
    from heron.adaptors.rllib_runner import HeronEnvRunner

    print("=" * 60)
    print("Case 1: IPPO — 2 fleet managers, each with 3 drones")
    print("=" * 60)

    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    try:
        fleet_ids = ["fleet_0", "fleet_1"]

        config = (
            PPOConfig()
            .environment(
                env=RLlibBasedHeronEnv,
                env_config={
                    "env_id": "ippo_fleets",
                    "agents": [
                        {"agent_id": f"drone_{f}_{d}", "agent_cls": Drone,
                         "features": [DroneFeature()], "coordinator": f"fleet_{f}"}
                        for f in range(2) for d in range(NUM_DRONES_PER_FLEET)
                    ],
                    "coordinators": [
                        {"coordinator_id": f"fleet_{f}", "agent_cls": FleetManager}
                        for f in range(2)
                    ],
                    "simulation": simulation,
                    "max_steps": 50,
                    "agent_ids": fleet_ids,
                },
            )
            .multi_agent(
                policies={
                    "fleet_0_policy": PolicySpec(),
                    "fleet_1_policy": PolicySpec(),
                },
                policy_mapping_fn=lambda agent_id, *a, **kw: f"{agent_id}_policy",
            )
            .env_runners(
                env_runner_cls=HeronEnvRunner,
                num_env_runners=1,
                num_envs_per_env_runner=1,
            )
            .evaluation(
                evaluation_interval=5,
                evaluation_num_env_runners=0,
                evaluation_duration=1,
                evaluation_duration_unit="episodes",
                evaluation_config=HeronEnvRunner.evaluation_config(t_end=50.0),
            )
            .training(
                lr=5e-4,
                gamma=0.99,
                train_batch_size=400,
                minibatch_size=64,
                num_epochs=3,
            )
            .framework("torch")
        )

        algo = config.build()
        print("\nTraining for 10 iterations...")
        for i in range(10):
            result = algo.train()
            reward = result["env_runners"]["episode_return_mean"] # we only support rllib 2.x
            if (i + 1) % 2 == 0:
                print(f"  Iter {i + 1}/10: reward={reward:.3f}")

        print("\nRunning event-driven evaluation...")
        eval_result = algo.evaluate()
        print(f"  Evaluation result: {eval_result}")

        algo.stop()
        print("\nCase 1 PASSED.")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

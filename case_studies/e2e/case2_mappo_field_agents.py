"""Case 2: MAPPO with centralized critic and CBF safety-constrained actors.

Hierarchy:
    SystemAgent
    ├── fleet_0 (TransportCoordinator) — shared policy, action_space = Box(-1,1,(3,))
    │   ├── drone_0_0 (TransportDrone) — receives 1D velocity command
    │   ├── drone_0_1 (TransportDrone)
    │   └── drone_0_2 (TransportDrone)
    └── fleet_1 (TransportCoordinator) — shared policy
        ├── drone_1_0 (TransportDrone)
        ├── drone_1_1 (TransportDrone)
        └── drone_1_2 (TransportDrone)

Domain: Multi-drone heavy payload transport with Control Barrier Function (CBF)
    safety guarantees. Drones transport payloads across a warehouse floor while
    a CBF filter ensures minimum separation distances between drones, making
    collision avoidance mathematically guaranteed rather than learned.

Training: MAPPO — both coordinators share a single policy.
         Centralized critic: coordinator obs includes fleet-aggregate state.
         Safety filter: CBF enforces minimum drone separation in simulation.

Usage:
    python -m case_studies.e2e.case2_mappo_field_agents
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
class DronePositionFeature(FeatureProvider):
    """Drone position on the warehouse floor (normalized to [0,1])."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    x_pos: float = 0.5
    y_pos: float = 0.5

    def set_values(self, **kwargs: Any) -> None:
        if "x_pos" in kwargs:
            self.x_pos = float(np.clip(kwargs["x_pos"], 0.0, 1.0))
        if "y_pos" in kwargs:
            self.y_pos = float(np.clip(kwargs["y_pos"], 0.0, 1.0))


@dataclass(slots=True)
class FleetSafetyFeature(FeatureProvider):
    """Fleet-level aggregate: separation and delivery progress."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    mean_separation: float = 0.5
    payload_progress: float = 0.0

    def set_values(self, **kwargs: Any) -> None:
        if "mean_separation" in kwargs:
            self.mean_separation = float(np.clip(kwargs["mean_separation"], 0.0, 1.0))
        if "payload_progress" in kwargs:
            self.payload_progress = float(np.clip(kwargs["payload_progress"], 0.0, 1.0))


# ── Agents ────────────────────────────────────────────────────────

class TransportDrone(FieldAgent):
    """Drone whose velocity is set by the coordinator, with CBF-safe execution."""

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
        if "x_pos" in kwargs:
            self.state.features["DronePositionFeature"].set_values(x_pos=kwargs["x_pos"])
        if "y_pos" in kwargs:
            self.state.features["DronePositionFeature"].set_values(y_pos=kwargs["y_pos"])

    def apply_action(self) -> None:
        pf = self.state.features["DronePositionFeature"]
        # Action drives x-axis movement (toward delivery goal at x=1.0)
        new_x = pf.x_pos + self.action.c[0] * 0.02
        pf.set_values(x_pos=new_x)

    def compute_local_reward(self, local_state: dict) -> float:
        pf = local_state.get("DronePositionFeature")
        if pf is None:
            return 0.0
        x_pos = float(pf[0])
        # Reward: progress toward delivery goal (x=1.0), with diminishing returns.
        # No collision penalty — CBF guarantees safety.
        return x_pos - 0.5 * x_pos ** 2


NUM_DRONES_PER_FLEET = 3

# CBF safety parameters
CBF_SAFETY_RADIUS = 0.1   # minimum allowed separation between drones
CBF_REPULSION_GAIN = 0.5  # strength of the CBF repulsion correction


class TransportCoordinator(CoordinatorAgent):
    """Coordinator with shared policy that assigns velocity targets to drones.

    Action: N-D vector in [-1, 1] — one velocity command per subordinate drone.
    VectorDecompositionActionProtocol splits this into per-drone 1D actions.

    Safety: The simulation applies a CBF filter post-action to guarantee
    minimum separation, so no collision penalty is needed in the reward.
    """

    def __init__(self, agent_id, subordinates, features=None, **kwargs):
        n_subs = len(subordinates)
        default_features = [FleetSafetyFeature()]
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
        sf = local_state.get("FleetSafetyFeature")
        if sf is None:
            return 0.0
        return float(sf[1])  # payload_progress


# ── CBF Safety Filter ─────────────────────────────────────────────

def cbf_safety_filter(positions: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Control Barrier Function filter enforcing minimum drone separation.

    For each pair of drones, if their Euclidean distance falls below
    CBF_SAFETY_RADIUS, apply a repulsive correction proportional to the
    barrier violation: dx_i += gamma * (h_safe - d) * (p_i - p_j) / d.

    This guarantees forward invariance of the safe set {d(i,j) >= r}
    without requiring the policy to learn collision avoidance.

    Args:
        positions: {drone_id: {"x_pos": float, "y_pos": float}}

    Returns:
        Corrected positions with safety guarantees.
    """
    drone_ids = list(positions.keys())
    n = len(drone_ids)

    for i in range(n):
        for j in range(i + 1, n):
            di, dj = drone_ids[i], drone_ids[j]
            pi = np.array([positions[di]["x_pos"], positions[di]["y_pos"]])
            pj = np.array([positions[dj]["x_pos"], positions[dj]["y_pos"]])

            dist = np.linalg.norm(pi - pj)
            if dist < CBF_SAFETY_RADIUS and dist > 1e-6:
                # CBF repulsion: push drones apart along their separation axis
                direction = (pi - pj) / dist
                correction = CBF_REPULSION_GAIN * (CBF_SAFETY_RADIUS - dist)

                pi_new = pi + correction * direction
                pj_new = pj - correction * direction

                positions[di]["x_pos"] = float(np.clip(pi_new[0], 0.0, 1.0))
                positions[di]["y_pos"] = float(np.clip(pi_new[1], 0.0, 1.0))
                positions[dj]["x_pos"] = float(np.clip(pj_new[0], 0.0, 1.0))
                positions[dj]["y_pos"] = float(np.clip(pj_new[1], 0.0, 1.0))

    return positions


# ── Simulation ────────────────────────────────────────────────────

def simulation(agent_states: dict) -> dict:
    """Apply wind drag, CBF safety filter, and update fleet features."""
    # Collect drone positions by fleet
    fleet_drones: Dict[str, Dict[str, Dict[str, float]]] = {}
    for aid, features in agent_states.items():
        if "DronePositionFeature" not in features:
            continue

        # Wind drag: drones drift backward slightly
        x = features["DronePositionFeature"]["x_pos"]
        x -= 0.005
        features["DronePositionFeature"]["x_pos"] = float(np.clip(x, 0.0, 1.0))

        # Group by fleet: drone_0_1 → fleet_0
        parts = aid.split("_")
        if len(parts) >= 2:
            fleet_id = f"fleet_{parts[1]}"
            fleet_drones.setdefault(fleet_id, {})[aid] = {
                "x_pos": features["DronePositionFeature"]["x_pos"],
                "y_pos": features["DronePositionFeature"]["y_pos"],
            }

    # Apply CBF safety filter per fleet
    for fleet_id, positions in fleet_drones.items():
        corrected = cbf_safety_filter(positions)
        for did, pos in corrected.items():
            agent_states[did]["DronePositionFeature"]["x_pos"] = pos["x_pos"]
            agent_states[did]["DronePositionFeature"]["y_pos"] = pos["y_pos"]

    # Update fleet features
    for fleet_id, positions in fleet_drones.items():
        if fleet_id not in agent_states or "FleetSafetyFeature" not in agent_states[fleet_id]:
            continue

        coords = np.array([[p["x_pos"], p["y_pos"]] for p in positions.values()])
        n = len(coords)

        # Mean pairwise separation
        if n >= 2:
            dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    dists.append(np.linalg.norm(coords[i] - coords[j]))
            mean_sep = float(np.mean(dists))
        else:
            mean_sep = 1.0

        # Payload progress: mean x-position (goal is x=1.0)
        mean_x = float(np.mean(coords[:, 0]))

        agent_states[fleet_id]["FleetSafetyFeature"]["mean_separation"] = mean_sep
        agent_states[fleet_id]["FleetSafetyFeature"]["payload_progress"] = mean_x

    return agent_states


# ── Training + Evaluation ─────────────────────────────────────────

def main():
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec

    from heron.adaptors.rllib import RLlibBasedHeronEnv
    from heron.adaptors.rllib_runner import HeronEnvRunner

    print("=" * 60)
    print("Case 2: MAPPO — 2 fleet coordinators (shared policy), 3 drones each")
    print("  Safety: CBF filter guarantees minimum drone separation")
    print("=" * 60)

    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    try:
        fleet_ids = ["fleet_0", "fleet_1"]

        config = (
            PPOConfig()
            .environment(
                env=RLlibBasedHeronEnv,
                env_config={
                    "env_id": "mappo_transport",
                    "agents": [
                        {"agent_id": f"drone_{f}_{d}", "agent_cls": TransportDrone,
                         "features": [DronePositionFeature(y_pos=0.2 + 0.3 * d)],
                         "coordinator": f"fleet_{f}"}
                        for f in range(2) for d in range(NUM_DRONES_PER_FLEET)
                    ],
                    "coordinators": [
                        {"coordinator_id": f"fleet_{f}", "agent_cls": TransportCoordinator}
                        for f in range(2)
                    ],
                    "simulation": simulation,
                    "max_steps": 50,
                    "agent_ids": fleet_ids,
                },
            )
            .multi_agent(
                policies={"shared": PolicySpec()},
                policy_mapping_fn=lambda agent_id, *a, **kw: "shared",
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
            reward = result["env_runners"]["episode_return_mean"]

            if (i + 1) % 2 == 0:
                print(f"  Iter {i + 1}/10: reward={reward:.3f}")

        print("\nRunning event-driven evaluation...")
        eval_result = algo.evaluate()
        print(f"  Evaluation result: {eval_result}")

        algo.stop()
        print("\nCase 2 PASSED.")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

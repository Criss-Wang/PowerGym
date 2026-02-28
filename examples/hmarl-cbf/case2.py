"""Case 2: MAPPO with centralized critic and CBF safety-constrained actors.

Hierarchy:
    SystemAgent
    ├── fleet_0 (TransportCoordinator) -- shared policy, action_space = Box(-1,1,(3,))
    │   ├── drone_0_0 (TransportDrone) -- receives 1D velocity command
    │   ├── drone_0_1 (TransportDrone)
    │   └── drone_0_2 (TransportDrone)
    └── fleet_1 (TransportCoordinator) -- shared policy
        ├── drone_1_0 (TransportDrone)
        ├── drone_1_1 (TransportDrone)
        └── drone_1_2 (TransportDrone)

Domain: Multi-drone heavy payload transport with Control Barrier Function (CBF)
    safety guarantees. Drones transport payloads across a warehouse floor while
    a CBF filter ensures minimum separation distances between drones, making
    collision avoidance mathematically guaranteed rather than learned.

Training: MAPPO -- both coordinators share a single policy.
         Centralized critic: coordinator obs includes fleet-aggregate state.
         Safety filter: CBF enforces minimum drone separation in simulation.

Usage:
    cd examples/hmarl-cbf
    python case2_mappo_field_agents.py
"""

from agents import TransportCoordinator, TransportDrone, NUM_DRONES_PER_FLEET
from env_physics import case2_simulation
from features import DronePositionFeature


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
                    "simulation": case2_simulation,
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

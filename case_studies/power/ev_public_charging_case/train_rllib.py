"""
Multi-station EV Charging Environment - RLlib Training Script

Ray 2.40+ / 3.0 Compatible Training Loop

This script demonstrates:
1. Multi-agent PPO training with centralized training, decentralized execution (CTDE)
2. PettingZoo ParallelEnv adapter compatibility with Ray RLlib 2.40+
3. Proper environment registration and agent policy mapping
4. Checkpoint management and training monitoring
"""

import logging
from pathlib import Path
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune import CLIReporter
from ray.tune.stopper import Stopper

# Import the multi-agent charging environment
from case_studies.power.ev_public_charging_case.env.charging_env import SimpleChargingEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
    _metrics_export_port=0,
)

# ============================================================================
# Environment Creator (Ray 2.40+ / 3.0 compatible)
# ============================================================================
def env_creator(config: Dict[str, Any]) -> ParallelPettingZooEnv:
    """
    Create a multi-agent charging environment wrapped for RLlib.

    Ray 2.40+ / 3.0 expects this to return a ParallelPettingZooEnv-compatible
    environment (our PettingZooParallelEnv adapter).

    Args:
        config: Environment configuration dict

    Returns:
        Wrapped environment ready for RLlib training
    """
    env = SimpleChargingEnv(
        arrival_rate=config.get("arrival_rate", 10.0),
        dt=config.get("dt", 300.0),
        num_stations=config.get("num_stations", 2),
    )

    # Ray 2.40+ / 3.0: Wrap in ParallelPettingZooEnv for RLlib
    return ParallelPettingZooEnv(env)


# ============================================================================
# Custom Training Stopper (optional)
# ============================================================================
class EpisodeRewardStopper(Stopper):
    """Stop training if mean reward exceeds threshold."""

    def __init__(self, reward_threshold: float = 100.0, episodes: int = 1000):
        self.reward_threshold = reward_threshold
        self.episodes = episodes

    def __call__(self, trial_id: str, result: Dict) -> bool:
        """Stop if reward threshold reached or episode limit exceeded."""
        episode_count = result.get("num_episodes_done", 0)
        reward_mean = result.get("env_runners", {}).get("episode_reward_mean", -float("inf"))

        if episode_count >= self.episodes:
            logger.info(f"Reached episode limit: {episode_count}")
            return True

        if reward_mean >= self.reward_threshold:
            logger.info(f"Reached reward threshold: {reward_mean:.2f}")
            return True

        return False

    def stop_all(self) -> bool:
        return False


# ============================================================================
# Ray 2.40+ / 3.0 Compatible Configuration
# ============================================================================
def create_ppo_config(num_env_runners: int = 2) -> PPOConfig:
    """
    Create a PPO configuration for multi-agent training.

    Key Ray 2.40+ / 3.0 changes:
    - Uses env_runners instead of deprecated rollout_workers
    - Explicit policy mapping for CTDE pattern
    - Ray 2.40+ compatible resource allocation
    - Uses old API stack for better PettingZoo compatibility

    Args:
        num_env_runners: Number of parallel environment workers

    Returns:
        Configured PPOConfig instance
    """
    config = (
        PPOConfig()
        .env_runners(
            num_env_runners=10,
            num_cpus_per_env_runner=1,
        )
        # --------- API Stack (Ray 2.40+ compatibility) ---------
        # Use old API stack for better PettingZoo environment compatibility
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        # --------- Environment ---------
        .environment(
            env="multi_station_charging_v2",
            env_config={
                "arrival_rate": 10.0,
                "dt": 300.0,
                "num_stations": 2,
            }
        )
        # --------- Framework ---------
        .framework("torch")
        # --------- Training (Ray 2.40+ style) ---------
        .training(
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=10.0,
            train_batch_size=4000
        )
        # --------- Environment Runners (Ray 2.40+) ---------
        # Note: Replaces deprecated rollout_workers config
        .env_runners(
            num_env_runners=num_env_runners,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
        )
        # --------- Resources (Ray 2.40+ compatible) ---------
        .resources(
            num_gpus=0,  # Set to 1 if GPU available
            # Use num_cpus_for_main_process instead of num_cpus_for_local_worker
            num_cpus_for_main_process=2,
        )
        # --------- Multi-Agent Configuration ---------
        .multi_agent(
            policies={
                "shared_policy": (
                    None,  # Use default policy class
                    None,  # Use default observation space
                    None,  # Use default action space
                    {},    # Config for policy
                )
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
    )

    return config


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    """
    Main training loop using Ray Tune (Ray 2.40+ / 3.0 compatible).

    Demonstrates:
    1. Ray cluster initialization
    2. Environment registration
    3. Algorithm configuration
    4. Training with checkpointing and monitoring
    5. Graceful shutdown
    """

    # Initialize Ray cluster
    logger.info("Initializing Ray cluster...")
    if ray.is_initialized():
        ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        num_gpus=0,  # Change to 1 if GPU available
    )
    logger.info(f"Ray cluster initialized: {ray.cluster_resources()}")

    # Register environment with RLlib
    env_name = "multi_station_charging_v2"
    logger.info(f"Registering environment: {env_name}")
    tune.register_env(env_name, env_creator)

    # Create training configuration
    logger.info("Creating PPO configuration...")
    config = create_ppo_config()

    # Setup checkpoint directory
    checkpoint_dir = Path("./checkpoints") / "ev_charging_training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Create stopper
    stopper = EpisodeRewardStopper(reward_threshold=100.0, episodes=500)

    # Setup progress reporter
    progress_reporter = CLIReporter(
        metric_columns=[
            "episode_reward_mean",
            "episode_reward_max",
            "episode_reward_min",
            "episode_len_mean",
            "num_episodes_done",
        ]
    )

    logger.info("Starting training with Ray Tune...")

    # Run training
    try:
        results = tune.run(
            "PPO",
            config=config,
            stop=stopper,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_freq=10,  # Save every 10 iterations
            checkpoint_at_end=True,
            verbose=1,
            progress_reporter=progress_reporter,
            # Ray 2.40+ feature: Reuse actors for efficiency
            reuse_actors=True,
        )

        logger.info("Training completed successfully!")
        logger.info(f"Best reward: {results.best_result['episode_reward_mean']:.2f}")
        logger.info(f"Best checkpoint: {results.best_checkpoint}")

        return results

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        logger.info("Shutting down Ray cluster...")
        ray.shutdown()


# ============================================================================
# Alternative: Simple Training Loop (no Tune)
# ============================================================================
def train_simple(num_iterations: int = 50):
    """
    Simpler training loop without Ray Tune (for debugging).

    Useful for quick testing without the overhead of Tune.

    Args:
        num_iterations: Number of training iterations
    """
    logger.info("Initializing Ray...")
    if ray.is_initialized():
        ray.shutdown()

    ray.init(ignore_reinit_error=True, num_cpus=4)

    env_name = "multi_station_charging_v2"
    tune.register_env(env_name, env_creator)

    config = create_ppo_config(num_env_runners=2)

    logger.info("Building algorithm...")
    # Use build_algo() instead of deprecated build()
    algo = config.build_algo()

    logger.info(f"Starting {num_iterations} training iterations...")

    try:
        for i in range(num_iterations):
            result = algo.train()

            # Extract metrics
            reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0)
            reward_max = result.get("env_runners", {}).get("episode_reward_max", 0)
            episodes = result.get("num_episodes_done", 0)

            logger.info(
                f"Iter {i:3d} | "
                f"Reward (mean/max): {reward_mean:7.2f}/{reward_max:7.2f} | "
                f"Episodes: {episodes:5d}"
            )

            # Save checkpoint periodically
            if i % 10 == 0 and i > 0:
                checkpoint = algo.save()
                logger.info(f"Checkpoint saved: {checkpoint.checkpoint.path}")

        logger.info("Training completed!")

    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        algo.stop()
        ray.shutdown()


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    import sys

    # Use simple training loop for faster feedback
    if len(sys.argv) > 1 and sys.argv[1] == "--tune":
        main()
    else:
        # Default: simple training loop
        train_simple(num_iterations=50)

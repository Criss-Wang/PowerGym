"""
Example 5: Training with RLlib (Ray)
=====================================

This example demonstrates how to train multi-agent policies using Ray RLlib's
MAPPO (Multi-Agent PPO) algorithm on PowerGrid environments.

What you'll learn:
- Wrapping PowerGrid environments for RLlib
- Configuring MAPPO for shared or independent policies
- Training loop with checkpointing
- Evaluating trained policies
- Integration with Weights & Biases (optional)

Architecture:
    RLlib (PPO Algorithm)
    └── ParallelPettingZooEnv (Wrapper)
        └── NetworkedGridEnv (PowerGrid)
            └── Multiple GridAgents

Training Process:
    1. Register environment with RLlib
    2. Configure PPO algorithm
    3. Train for N iterations
    4. Save checkpoints
    5. Evaluate policy

Requirements:
    pip install "ray[rllib]==2.9.0"
    pip install wandb  # Optional, for logging

Runtime: ~10 minutes for 50 iterations (adjust as needed)
"""

import argparse
from datetime import datetime

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Import the example environment from Example 1
# Note: To use this example, you need to run example 01 first or copy the
# SingleMicrogridEnv class. For now, we'll import it using importlib.
import importlib.util
import sys
import os

# Load example 1 dynamically
example_01_path = os.path.join(os.path.dirname(__file__), '01_single_microgrid_basic.py')
spec = importlib.util.spec_from_file_location("example_01", example_01_path)
example_01 = importlib.util.module_from_spec(spec)
sys.modules['example_01'] = example_01
spec.loader.exec_module(example_01)
SingleMicrogridEnv = example_01.SingleMicrogridEnv


def env_creator(env_config):
    """Create environment with RLlib compatibility.

    Args:
        env_config: Configuration dict from RLlib

    Returns:
        ParallelPettingZooEnv wrapper around PowerGrid environment
    """
    # Create base PowerGrid environment
    base_env = SingleMicrogridEnv(env_config)

    # Wrap for RLlib
    return ParallelPettingZooEnv(base_env)


def get_policy_configs(env, shared_policy=True):
    """Get policy configuration.

    Args:
        env: Environment instance
        shared_policy: If True, all agents share one policy (MAPPO)
                      If False, each agent has own policy (IPPO)

    Returns:
        policies: Dict of policy configurations
        policy_mapping_fn: Function mapping agent_id → policy_id
    """
    # Get possible agents from wrapped env
    if hasattr(env, 'possible_agents'):
        possible_agents = env.possible_agents
    elif hasattr(env, 'env') and hasattr(env.env, 'possible_agents'):
        possible_agents = env.env.possible_agents
    else:
        possible_agents = list(env.get_agent_ids())

    if shared_policy:
        # MAPPO: All agents share one policy
        first_agent = possible_agents[0]
        policies = {
            'shared_policy': (
                None,  # Use default policy class
                env.observation_space[first_agent],
                env.action_space[first_agent],
                {},  # Policy config
            )
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: 'shared_policy'
    else:
        # IPPO: Each agent has own policy
        policies = {
            agent_id: (
                None,
                env.observation_space[agent_id],
                env.action_space[agent_id],
                {},
            )
            for agent_id in possible_agents
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id

    return policies, policy_mapping_fn


def train(args):
    """Main training function.

    Args:
        args: Command line arguments
    """
    print("=" * 70)
    print("Example 5: Training with RLlib (MAPPO)")
    print("=" * 70)

    # Initialize Ray
    print("\n[1] Initializing Ray...")
    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus)

    # Register environment
    print(f"\n[2] Registering environment: {args.env_name}")
    register_env(args.env_name, env_creator)

    # Create temporary environment to get spaces
    print("\n[3] Creating environment to inspect spaces...")
    env_config = {
        "max_episode_steps": 24,
        "train": True,
    }
    temp_env = env_creator(env_config)

    # Get policy configuration
    policies, policy_mapping_fn = get_policy_configs(
        temp_env,
        shared_policy=not args.independent_policies
    )

    print(f"\n[4] Policy Configuration:")
    print(f"    Mode: {'IPPO (Independent)' if args.independent_policies else 'MAPPO (Shared)'}")
    print(f"    Policies: {list(policies.keys())}")
    print(f"    Agents: {temp_env.possible_agents if hasattr(temp_env, 'possible_agents') else list(temp_env.get_agent_ids())}")

    # Configure PPO algorithm
    print("\n[5] Configuring PPO algorithm...")
    config = (
        PPOConfig()
        .environment(
            env=args.env_name,
            env_config=env_config,
        )
        .framework("torch")
        .training(
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=args.clip_param,
            vf_clip_param=args.vf_clip_param,
            entropy_coeff=args.entropy_coeff,
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .resources(
            num_gpus=0,  # CPU-only for this example
        )
    )

    print(f"    Train batch size: {args.train_batch_size}")
    print(f"    SGD minibatch size: {args.sgd_minibatch_size}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Num workers: {args.num_workers}")

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_type = "ippo" if args.independent_policies else "mappo"
    exp_name = f"{policy_type}_{args.env_name}_{timestamp}"
    checkpoint_dir = f"checkpoints/{exp_name}"

    print(f"\n[6] Training for {args.iterations} iterations...")
    print(f"    Experiment: {exp_name}")
    print(f"    Checkpoints: {checkpoint_dir}")

    # Train using Ray Tune
    results = tune.run(
        "PPO",
        name=exp_name,
        config=config.to_dict(),
        stop={"training_iteration": args.iterations},
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir="./checkpoints",
        verbose=1 if args.verbose else 0,
    )

    # Print results
    print("\n[7] Training Complete!")
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"    Best checkpoint: {best_result.checkpoint}")
    print(f"    Best reward: {best_result.metrics['episode_reward_mean']:.2f}")
    print(f"    Final iteration: {best_result.metrics['training_iteration']}")

    # Shutdown Ray
    ray.shutdown()

    print("\n[8] Training Summary:")
    print(f"    Total iterations: {args.iterations}")
    print(f"    Policy type: {policy_type.upper()}")
    print(f"    Checkpoint directory: {checkpoint_dir}")
    print(f"    Use this checkpoint to evaluate or deploy the policy")

    print("\n[9] Next Steps:")
    print("    - Load checkpoint and evaluate on test episodes")
    print("    - Visualize training curves (use TensorBoard or W&B)")
    print("    - Deploy trained policy in production environment")
    print("    - Fine-tune hyperparameters for better performance")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MAPPO/IPPO on PowerGrid environment'
    )

    # Environment
    parser.add_argument(
        '--env-name',
        type=str,
        default='single_microgrid',
        help='Environment name for registration (default: single_microgrid)'
    )

    # Training parameters
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Number of training iterations (default: 50)'
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=4000,
        help='Training batch size (default: 4000)'
    )
    parser.add_argument(
        '--sgd-minibatch-size',
        type=int,
        default=128,
        help='SGD minibatch size (default: 128)'
    )
    parser.add_argument(
        '--num-sgd-iter',
        type=int,
        default=10,
        help='Number of SGD iterations per training batch (default: 10)'
    )

    # PPO hyperparameters
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate (default: 5e-5)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor (default: 0.99)'
    )
    parser.add_argument(
        '--lambda',
        type=float,
        default=0.95,
        dest='lambda_',
        help='GAE lambda (default: 0.95)'
    )
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='PPO clip parameter (default: 0.2)'
    )
    parser.add_argument(
        '--vf-clip-param',
        type=float,
        default=10.0,
        help='Value function clip parameter (default: 10.0)'
    )
    parser.add_argument(
        '--entropy-coeff',
        type=float,
        default=0.01,
        help='Entropy coefficient (default: 0.01)'
    )

    # Parallelization
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='Number of rollout workers (default: 2)'
    )
    parser.add_argument(
        '--num-cpus',
        type=int,
        default=4,
        help='Number of CPUs for Ray (default: 4)'
    )

    # Policy type
    parser.add_argument(
        '--independent-policies',
        action='store_true',
        help='Use independent policies (IPPO) instead of shared (MAPPO)'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='Checkpoint frequency (default: 10 iterations)'
    )

    # Logging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""
Example 5: MAPPO Training for Cooperative Multi-Agent Systems
==============================================================

This example demonstrates production-ready training of Multi-Agent PPO (MAPPO)
or Independent PPO (IPPO) on cooperative multi-agent microgrids with full
experiment management capabilities.

What you'll learn:
- Command-line training script with comprehensive arguments
- MAPPO vs IPPO for cooperative tasks
- Shared rewards to encourage cooperation
- Experiment tracking with Weights & Biases
- Checkpointing and resuming training
- Performance monitoring and logging

Architecture:
    RLlib (PPO Algorithm)
    └── ParallelPettingZooEnv (Wrapper)
        └── MultiAgentMicrogrids (PowerGrid)
            ├── GridAgent MG1
            ├── GridAgent MG2
            └── GridAgent MG3

Cooperative Task:
    Agents learn to balance loads, maintain voltage stability, and share
    resources optimally through shared rewards and communication protocols.

Usage:
    # Train with shared policy (MAPPO) for cooperative tasks
    python examples/05_mappo_training.py --iterations 100

    # Train with independent policies (IPPO)
    python examples/05_mappo_training.py --iterations 100 --independent-policies

    # Resume from checkpoint
    python examples/05_mappo_training.py --resume /path/to/checkpoint

    # With W&B logging
    python examples/05_mappo_training.py --wandb --wandb-project powergrid-coop

    # Custom experiment with shared rewards (encourages cooperation)
    python examples/05_mappo_training.py --iterations 200 --share-reward \\
        --lr 5e-5 --hidden-dim 256 --num-workers 8

    # Quick test mode (3 iterations for verification)
    python examples/05_mappo_training.py --test --no-cuda

Requirements:
    pip install "ray[rllib]==2.9.0"
    pip install wandb  # Optional, for experiment tracking

Runtime: ~30 minutes for 100 iterations (depends on workers and hardware)
"""
import os
import warnings

# Suppress warnings before any other imports (applies to subprocesses too)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")  # Suppress all warnings for cleaner output

import argparse
import json
from datetime import datetime

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Import environment
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MAPPO/IPPO on cooperative multi-agent microgrids',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of training iterations')
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode (3 iterations, 1 worker, small batch)')
    parser.add_argument('--train-batch-size', type=int, default=4000,
                        help='Training batch size')
    parser.add_argument('--sgd-minibatch-size', type=int, default=128,
                        help='SGD minibatch size')
    parser.add_argument('--num-sgd-iter', type=int, default=10,
                        help='Number of SGD iterations per training step')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lambda', type=float, default=0.95, dest='lambda_',
                        help='GAE lambda (advantage estimation)')

    # Environment parameters
    parser.add_argument('--penalty', type=float, default=10.0,
                        help='Safety violation penalty coefficient')
    parser.add_argument('--share-reward', action='store_true',
                        help='Use shared rewards across agents (encourages cooperation)')
    parser.add_argument('--no-share-reward', dest='share_reward', action='store_false')
    parser.set_defaults(share_reward=True)

    # Policy parameters
    parser.add_argument('--independent-policies', action='store_true',
                        help='Use independent policies for each agent (IPPO vs MAPPO)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer dimension for policy network')

    # Parallelization
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel rollout workers')
    parser.add_argument('--num-envs-per-worker', type=int, default=1,
                        help='Number of environments per worker')

    # Checkpointing
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                        help='Checkpoint frequency (iterations)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases experiment tracking')
    parser.add_argument('--wandb-project', type=str, default='powergrid-cooperative',
                        help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity (username or team name)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Custom experiment name (auto-generated if not provided)')

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')

    return parser.parse_args()


def env_creator(env_config):
    """Create environment with RLlib compatibility.

    Args:
        env_config: Configuration dict from RLlib

    Returns:
        ParallelPettingZooEnv wrapper around PowerGrid environment
    """
    # Load default config if not provided
    if 'dataset_path' not in env_config:
        from powergrid.envs.configs.config_loader import load_config
        default_config = load_config('ieee34_ieee13')
        # Merge with provided config (provided config takes precedence)
        for key, value in env_config.items():
            default_config[key] = value
        env_config = default_config

    env = MultiAgentMicrogrids(env_config)
    # Wrap with PettingZoo wrapper for RLlib
    return ParallelPettingZooEnv(env)


def get_policy_configs(env, args):
    """Get policy configuration for MAPPO or IPPO.

    Args:
        env: Environment instance
        args: Command line arguments

    Returns:
        Tuple of (policies dict, policy_mapping_fn)
    """
    # Get possible agents - handle wrapped env
    if hasattr(env, 'par_env') and hasattr(env.par_env, 'possible_agents'):
        # ParallelPettingZooEnv wrapper - get from inner env
        possible_agents = env.par_env.possible_agents
    elif hasattr(env, 'possible_agents') and len(env.possible_agents) > 0:
        possible_agents = env.possible_agents
    elif hasattr(env, 'env') and hasattr(env.env, 'possible_agents'):
        possible_agents = env.env.possible_agents
    else:
        # Fallback: get agents from observation/action space keys
        possible_agents = list(env.get_agent_ids())

    if args.independent_policies:
        # IPPO: Each agent has its own independent policy
        policies = {
            agent_id: (None, env.observation_space[agent_id], env.action_space[agent_id], {})
            for agent_id in possible_agents
        }
        policy_mapping_fn = lambda agent_id, *args_, **kwargs: agent_id
        policy_type = "IPPO (Independent Policies)"
    else:
        # MAPPO: All agents share one policy (better for cooperation)
        # Use the first agent's spaces since all agents have identical spaces
        first_agent = possible_agents[0]
        policies = {
            'shared_policy': (
                None,
                env.observation_space[first_agent],
                env.action_space[first_agent],
                {}
            )
        }
        policy_mapping_fn = lambda agent_id, *args_, **kwargs: 'shared_policy'
        policy_type = "MAPPO (Shared Policy)"

    return policies, policy_mapping_fn, policy_type


def main():
    """Main training function."""
    args = parse_args()

    # Apply test mode overrides
    if args.test:
        print("⚠ TEST MODE: Using minimal configuration for quick verification")
        args.iterations = 3
        args.num_workers = 1
        args.train_batch_size = 1000
        args.checkpoint_freq = 1
        args.wandb = False
        args.no_cuda = True

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register environment
    register_env("multi_agent_microgrids", env_creator)

    # Create environment to get spaces
    from powergrid.envs.configs.config_loader import load_config
    env_config = load_config('ieee34_ieee13')

    # Override with command line args
    env_config['train'] = True
    env_config['penalty'] = args.penalty
    env_config['share_reward'] = args.share_reward
    env_config['max_episode_steps'] = 96  # 4 days at 1-hour timesteps

    temp_env = env_creator(env_config)

    # Get policy configuration
    policies, policy_mapping_fn, policy_type = get_policy_configs(temp_env, args)

    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_tag = "ippo" if args.independent_policies else "mappo"
        reward_tag = "shared" if args.share_reward else "indep"
        args.experiment_name = f"{policy_tag}_{reward_tag}_mg3_{timestamp}"

    # Print training configuration
    print("=" * 80)
    print("Cooperative Multi-Agent Microgrid Training with RLlib")
    print("=" * 80)
    print(f"Experiment:        {args.experiment_name}")
    print(f"Policy type:       {policy_type}")
    print(f"Shared reward:     {args.share_reward} (encourages cooperation)")
    print(f"Iterations:        {args.iterations}")
    print(f"Safety penalty:    {args.penalty}")
    print(f"Learning rate:     {args.lr}")
    print(f"Hidden dimension:  {args.hidden_dim}")
    print(f"Rollout workers:   {args.num_workers}")
    print(f"Batch size:        {args.train_batch_size}")
    print(f"Random seed:       {args.seed}")
    print("=" * 80)

    # Configure PPO algorithm
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="multi_agent_microgrids",
            env_config=env_config,
            disable_env_checking=True,
        )
        .framework("torch")
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            clip_param=0.3,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .resources(
            num_gpus=0,
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            num_gpus_per_env_runner=0,
        )
        .debugging(
            seed=args.seed,
        )
    )

    # Set these parameters directly on config object (not supported in method chaining)
    config.train_batch_size = args.train_batch_size
    config.sgd_minibatch_size = args.sgd_minibatch_size
    config.num_sgd_iter = args.num_sgd_iter
    config.model = {
        "fcnet_hiddens": [args.hidden_dim, args.hidden_dim],
        "fcnet_activation": "relu",
        "max_seq_len": 20,  # Required by PPO torch policy
    }
    # Disable observation preprocessing to avoid validation issues with PettingZoo wrapper
    config.preprocessor_pref = None
    config.enable_connectors = False

    # Setup W&B logging if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.experiment_name,
                config=vars(args),
                tags=["cooperative", policy_tag, reward_tag],
            )
            print(f"✓ W&B logging enabled: {args.wandb_project}/{args.experiment_name}")
        except ImportError:
            print("⚠ WARNING: wandb not installed. Install with: pip install wandb")
            args.wandb = False

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Configuration saved to: {config_path}")
    
    # Build algorithm
    if args.resume:
        print(f"⟳ Resuming from checkpoint: {args.resume}")
        algo = config.build()
        algo.restore(args.resume)
    else:
        algo = config.build()

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("-" * 80)
    print(f"{'Iter':>5} | {'Reward':>10} | {'Cost':>10} | {'Episodes':>8} | "
          f"{'Steps':>10} | {'Time':>8}")
    print("-" * 80)

    best_reward = float('-inf')

    for i in range(args.iterations):
        result = algo.train()

        # Extract metrics from nested structure
        env_runners = result.get('env_runners', {})
        reward_mean = env_runners.get('episode_reward_mean', 0)
        episodes = env_runners.get('episodes_this_iter', 0)

        timesteps = result.get('timesteps_total', 0)
        time_total = result.get('time_total_s', 0)

        # Cost is negative reward (since reward = -cost)
        cost_mean = -reward_mean if reward_mean != 0 else 0

        # Print progress
        print(f"{i+1:5d} | {reward_mean:10.2f} | {cost_mean:10.2f} | "
              f"{episodes:8d} | {timesteps:10d} | {time_total:8.1f}s")

        # Log to W&B
        if args.wandb:
            try:
                wandb.log({
                    'iteration': i + 1,
                    'reward_mean': reward_mean,
                    'cost_mean': cost_mean,
                    'episodes': episodes,
                    'timesteps': timesteps,
                    'time_total': time_total,
                })
            except:
                pass

        # Checkpoint
        if (i + 1) % args.checkpoint_freq == 0:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

            # Save best model
            if reward_mean > best_reward:
                best_reward = reward_mean
                best_path = os.path.join(checkpoint_dir, 'best_checkpoint')
                os.makedirs(best_path, exist_ok=True)
                algo.save(best_path)
                print(f"  ★ Best model saved: {best_path} (reward: {best_reward:.2f})")

    print("-" * 80)
    print(f"✓ Training complete!")
    print(f"  Best reward achieved: {best_reward:.2f}")

    # Final checkpoint
    final_path = algo.save(checkpoint_dir)
    print(f"  Final checkpoint: {final_path}")

    # Cleanup
    algo.stop()
    ray.shutdown()

    if args.wandb:
        try:
            wandb.finish()
        except:
            pass

    print("=" * 80)


if __name__ == '__main__':
    main()

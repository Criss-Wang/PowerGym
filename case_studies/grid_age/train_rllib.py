"""RLlib MAPPO training for GridAges microgrid environment.

This script integrates the Heron-based MicrogridEnv with RLlib for
multi-agent PPO (MAPPO) training, matching the original GridAges setup.
"""

import argparse
from typing import Dict, Any
import gymnasium as gym
import numpy as np

# RLlib imports
try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.tune.registry import register_env
    from ray import tune, air
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False
    print("Warning: RLlib not available. Install with: pip install ray[rllib]")

from case_studies.grid_age.envs import MicrogridEnv
from heron.core.observation import Observation


class HeronToGymWrapper(gym.Env):
    """Wrapper to convert Heron MultiAgentEnv to Gymnasium Env for single agent.

    This wrapper adapts a Heron environment for a specific agent to work
    with RLlib's single-agent interface within multi-agent training.
    """

    def __init__(self, heron_env: MicrogridEnv, agent_id: str):
        """Initialize wrapper.

        Args:
            heron_env: Heron MultiAgentEnv instance
            agent_id: Agent ID to wrap
        """
        self.heron_env = heron_env
        self.agent_id = agent_id

        # Get agent
        self.agent = heron_env.registered_agents.get(agent_id)
        if not self.agent:
            raise ValueError(f"Agent {agent_id} not found in environment")

        # Set spaces from agent
        self.action_space = self.agent.get_action_space()

        # Observation space is based on Observation.vector() size
        # We'll get this after first reset
        self._obs_space = None

    @property
    def observation_space(self):
        if self._obs_space is None:
            raise ValueError("Call reset() first to determine observation space")
        return self._obs_space

    def reset(self, *, seed=None, options=None):
        """Reset environment."""
        # Only reset once per environment (shared across agents)
        if not hasattr(self.heron_env, '_reset_done'):
            self.heron_env._reset_done = True
            obs_dict, info_dict = self.heron_env.reset(seed=seed)

            # Set observation space from first observation
            if self._obs_space is None and self.agent_id in obs_dict:
                obs = obs_dict[self.agent_id]
                if isinstance(obs, Observation):
                    obs_vec = obs.vector()
                else:
                    obs_vec = obs
                self._obs_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=obs_vec.shape,
                    dtype=np.float32
                )

            self.heron_env._last_obs = obs_dict
            self.heron_env._last_info = info_dict

        obs = self.heron_env._last_obs[self.agent_id]
        if isinstance(obs, Observation):
            obs = obs.vector()

        info = self.heron_env._last_info.get(self.agent_id, {})
        return obs, info

    def step(self, action):
        """Execute one step."""
        # Convert action to Heron Action format
        from heron.core.action import Action as HeronAction
        heron_action = HeronAction()
        heron_action.set_specs(
            dim_c=4,
            range=(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]))
        )
        heron_action.set_values(c=action)

        # Collect actions from all agents (only this agent acts, others use stored)
        if not hasattr(self.heron_env, '_pending_actions'):
            self.heron_env._pending_actions = {}

        self.heron_env._pending_actions[self.agent_id] = heron_action

        # Check if all agents have submitted actions
        agent_ids = [aid for aid, a in self.heron_env.registered_agents.items()
                     if hasattr(a, 'action_space') and a.action_space is not None]

        if len(self.heron_env._pending_actions) == len(agent_ids):
            # All agents ready - step environment
            obs_dict, rewards_dict, term_dict, trunc_dict, info_dict = \
                self.heron_env.step(self.heron_env._pending_actions)

            # Store results
            self.heron_env._last_obs = obs_dict
            self.heron_env._last_rewards = rewards_dict
            self.heron_env._last_terminated = term_dict
            self.heron_env._last_truncated = trunc_dict
            self.heron_env._last_info = info_dict

            # Clear pending actions
            self.heron_env._pending_actions = {}
        else:
            # Wait for other agents - return cached results
            pass

        # Return this agent's results
        obs = self.heron_env._last_obs[self.agent_id]
        if isinstance(obs, Observation):
            obs = obs.vector()

        reward = self.heron_env._last_rewards.get(self.agent_id, 0.0)
        terminated = self.heron_env._last_terminated.get(self.agent_id, False)
        truncated = self.heron_env._last_truncated.get(self.agent_id, False)
        info = self.heron_env._last_info.get(self.agent_id, {})

        return obs, reward, terminated, truncated, info


def train_with_rllib(
    num_microgrids: int = 3,
    num_iterations: int = 100,
    checkpoint_freq: int = 20,
):
    """Train microgrid policies using RLlib MAPPO.

    Args:
        num_microgrids: Number of microgrids to simulate
        num_iterations: Number of training iterations
        checkpoint_freq: Checkpoint frequency

    Returns:
        Trained algorithm
    """
    if not RLLIB_AVAILABLE:
        raise ImportError("RLlib not installed. Install with: pip install ray[rllib]")

    import ray
    from ray.rllib.algorithms.ppo import PPO

    print("=" * 70)
    print("RLlib MAPPO Training for GridAges Microgrids")
    print("=" * 70)

    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_cpus=4)

    # Step 1: Create probe environment to get spaces
    print("\n1. Creating probe environment...")
    probe_env = MicrogridEnv(num_microgrids=num_microgrids, episode_steps=24)
    obs, _ = probe_env.reset(seed=0)

    # Get agent IDs and spaces
    agent_ids = [aid for aid, a in probe_env.registered_agents.items()
                 if hasattr(a, 'action_space') and a.action_space is not None]

    print(f"   Found {len(agent_ids)} agents: {agent_ids}")

    # Get observation/action spaces
    spaces = {}
    for aid in agent_ids:
        agent = probe_env.registered_agents[aid]
        obs_vec = obs[aid].vector() if isinstance(obs[aid], Observation) else obs[aid]
        spaces[aid] = {
            'obs_space': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=obs_vec.shape, dtype=np.float32
            ),
            'action_space': agent.get_action_space(),
        }
        print(f"   - {aid}: obs_dim={obs_vec.shape[0]}, action_dim={agent.action.dim_c}")

    # Step 2: Configure multi-agent setup
    print("\n2. Configuring multi-agent PPO...")

    # Define policy specs for each agent (heterogeneous networks)
    network_configs = {
        "MG1": [128, 128],
        "MG2": [64, 64],
        "MG3": [64, 128],
    }

    policies = {}
    for aid in agent_ids:
        policies[aid] = (
            None,  # Policy class (None = use default)
            spaces[aid]['obs_space'],
            spaces[aid]['action_space'],
            {
                "model": {
                    "fcnet_hiddens": network_configs.get(aid, [64, 64]),
                    "fcnet_activation": "relu",
                }
            }
        )

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        """Map each agent to its own policy."""
        return agent_id

    # Step 3: Build RLlib configuration
    config = (
        PPOConfig()
        .environment(
            env=MicrogridEnv,
            env_config={
                "num_microgrids": num_microgrids,
                "episode_steps": 24,
            },
        )
        .framework("torch")
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rollouts(
            num_rollout_workers=2,
            num_envs_per_worker=1,
        )
        .resources(
            num_gpus=0,
        )
    )

    print("   ✅ Configuration complete")

    # Step 4: Train
    print(f"\n3. Training for {num_iterations} iterations...")
    print("-" * 70)

    algo = config.build()

    best_reward = -np.inf

    for i in range(num_iterations):
        result = algo.train()

        episode_reward_mean = result.get("episode_reward_mean", 0)
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean

        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1:3d}/{num_iterations}: "
                  f"reward={episode_reward_mean:.2f}, "
                  f"best={best_reward:.2f}")

        # Save checkpoint
        if (i + 1) % checkpoint_freq == 0:
            checkpoint = algo.save()
            print(f"   Checkpoint saved: {checkpoint}")

    print("-" * 70)
    print("✅ Training complete!")

    # Step 5: Evaluate
    print("\n4. Evaluating trained policies...")

    eval_env = MicrogridEnv(num_microgrids=num_microgrids, episode_steps=24)
    eval_obs, _ = eval_env.reset(seed=999)

    eval_rewards = []
    for _ in range(5):  # 5 evaluation episodes
        episode_reward = 0
        obs, _ = eval_env.reset()

        for step in range(24):
            actions = {}
            for aid in agent_ids:
                obs_vec = obs[aid].vector() if isinstance(obs[aid], Observation) else obs[aid]
                action = algo.compute_single_action(
                    observation=obs_vec,
                    policy_id=aid,
                    explore=False,
                )
                from heron.core.action import Action as HeronAction
                heron_action = HeronAction()
                heron_action.set_specs(dim_c=4, range=(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1])))
                heron_action.set_values(c=action)
                actions[aid] = heron_action

            obs, rewards, terminated, truncated, info = eval_env.step(actions)
            episode_reward += sum(r for aid, r in rewards.items() if aid in agent_ids)

            if terminated.get("__all__", False):
                break

        eval_rewards.append(episode_reward)

    print(f"   Eval reward (mean ± std): {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")

    # Cleanup
    algo.stop()
    ray.shutdown()

    print("\n" + "=" * 70)
    print("RLlib MAPPO Training Complete!")
    print("=" * 70)

    return algo


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train GridAges microgrids with RLlib MAPPO")
    parser.add_argument("--num-microgrids", type=int, default=3,
                        help="Number of microgrids")
    parser.add_argument("--num-iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--checkpoint-freq", type=int, default=20,
                        help="Checkpoint frequency")

    args = parser.parse_args()

    if not RLLIB_AVAILABLE:
        print("ERROR: RLlib not installed!")
        print("Install with: pip install ray[rllib]")
        return

    train_with_rllib(
        num_microgrids=args.num_microgrids,
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()

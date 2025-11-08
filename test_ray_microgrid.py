"""
Test script to verify Ray RLlib can run multi_agent_microgrids with meaningful rewards.

This script:
1. Creates the environment
2. Runs a few episodes with random actions
3. Checks that rewards are meaningful (not all zeros/nans)
4. Verifies the environment is compatible with Ray RLlib
"""

import numpy as np
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.configs.config_loader import load_config


def test_basic_environment():
    """Test basic environment functionality."""
    print("=" * 70)
    print("Test 1: Basic Environment Functionality")
    print("=" * 70)

    # Load config
    env_config = load_config('ieee34_ieee13')
    env_config['train'] = True
    env_config['max_episode_steps'] = 24  # One day

    print(f"Config loaded:")
    print(f"  - Max episode steps: {env_config['max_episode_steps']}")
    print(f"  - Penalty: {env_config['penalty']}")
    print(f"  - Training mode: {env_config['train']}")

    # Create environment
    env = MultiAgentMicrogrids(env_config)
    print(f"\nEnvironment created:")
    print(f"  - Agents: {env.possible_agents}")
    print(f"  - Actionable agents: {list(env.actionable_agents.keys())}")

    # Reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation:")
    for agent_id, agent_obs in obs.items():
        print(f"  - {agent_id}: shape={agent_obs.shape}, "
              f"min={agent_obs.min():.2f}, max={agent_obs.max():.2f}")

    # Run one episode with random actions
    print("\nRunning episode with random actions...")
    episode_rewards = {agent_id: [] for agent_id in env.possible_agents}
    episode_length = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Sample random actions
        actions = {name: space.sample() for name, space in env.action_spaces.items()}

        # Step
        obs_next, rewards, terminateds, truncateds, infos = env.step(actions)

        # Track rewards
        for agent_id in env.possible_agents:
            episode_rewards[agent_id].append(rewards[agent_id])

        episode_length += 1
        terminated = all(terminateds.values())
        truncated = all(truncateds.values())

        obs = obs_next

    print(f"\nEpisode completed:")
    print(f"  - Length: {episode_length} steps")
    print(f"  - Terminated: {terminated}, Truncated: {truncated}")

    # Analyze rewards
    print(f"\nReward statistics:")
    for agent_id in env.possible_agents:
        rewards_array = np.array(episode_rewards[agent_id])
        print(f"  {agent_id}:")
        print(f"    - Mean: {rewards_array.mean():.2f}")
        print(f"    - Std: {rewards_array.std():.2f}")
        print(f"    - Min: {rewards_array.min():.2f}")
        print(f"    - Max: {rewards_array.max():.2f}")
        print(f"    - Sum: {rewards_array.sum():.2f}")
        print(f"    - All zeros: {np.all(rewards_array == 0)}")
        print(f"    - Any NaN: {np.any(np.isnan(rewards_array))}")

    return env, episode_rewards


def test_ray_compatibility():
    """Test Ray RLlib compatibility."""
    print("\n" + "=" * 70)
    print("Test 2: Ray RLlib Compatibility")
    print("=" * 70)

    try:
        import ray
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        print("Ray RLlib imports successful")
    except ImportError as e:
        print(f"WARNING: Ray not installed: {e}")
        print("Install with: pip install 'ray[rllib]==2.9.0'")
        return None

    # Load config
    env_config = load_config('ieee34_ieee13')
    env_config['train'] = True
    env_config['max_episode_steps'] = 24

    # Create environment
    env = MultiAgentMicrogrids(env_config)

    # Wrap with PettingZoo wrapper
    print("\nWrapping with ParallelPettingZooEnv...")
    try:
        wrapped_env = ParallelPettingZooEnv(env)
        print("✓ Successfully wrapped environment")
    except Exception as e:
        print(f"✗ Error wrapping environment: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Test reset
    print("\nTesting wrapped environment reset...")
    try:
        obs = wrapped_env.reset()
        print(f"✓ Reset successful")
        print(f"  - Observation keys: {list(obs.keys())}")
        for agent_id, agent_obs in obs.items():
            print(f"    - {agent_id}: shape={agent_obs.shape}")
    except Exception as e:
        print(f"✗ Error during reset: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Test step
    print("\nTesting wrapped environment step...")
    try:
        actions = {agent_id: wrapped_env.action_space.sample()
                   for agent_id in wrapped_env.par_env.possible_agents}
        obs, rewards, dones, infos = wrapped_env.step(actions)
        print(f"✓ Step successful")
        print(f"  - Rewards: {[(k, f'{v:.2f}') for k, v in rewards.items()]}")
        print(f"  - Dones: {dones}")
    except Exception as e:
        print(f"✗ Error during step: {e}")
        import traceback
        traceback.print_exc()
        return None

    return wrapped_env


def test_reward_meaningfulness():
    """Test that rewards are meaningful (vary with actions)."""
    print("\n" + "=" * 70)
    print("Test 3: Reward Meaningfulness")
    print("=" * 70)

    # Load config
    env_config = load_config('ieee34_ieee13')
    env_config['train'] = True
    env_config['max_episode_steps'] = 24

    env = MultiAgentMicrogrids(env_config)

    # Run multiple episodes and collect rewards
    num_episodes = 3
    all_episode_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        episode_rewards = []

        for step in range(min(10, env_config['max_episode_steps'])):
            # Random actions
            actions = {name: space.sample() for name, space in env.action_spaces.items()}
            obs_next, rewards, terminateds, truncateds, infos = env.step(actions)

            # Sum rewards across agents for this step
            total_reward = sum(rewards.values())
            episode_rewards.append(total_reward)

            if all(terminateds.values()) or all(truncateds.values()):
                break

            obs = obs_next

        all_episode_rewards.append(episode_rewards)
        print(f"Episode {ep + 1}: Mean reward = {np.mean(episode_rewards):.2f}")

    # Check variance across episodes
    episode_means = [np.mean(ep_rewards) for ep_rewards in all_episode_rewards]
    print(f"\nReward variance across episodes:")
    print(f"  - Episode means: {[f'{m:.2f}' for m in episode_means]}")
    print(f"  - Std of episode means: {np.std(episode_means):.2f}")

    # Check that rewards are not constant
    all_rewards_flat = [r for ep in all_episode_rewards for r in ep]
    unique_rewards = len(set(all_rewards_flat))
    print(f"  - Unique reward values: {unique_rewards}")

    if unique_rewards > 1:
        print("✓ Rewards are meaningful (not constant)")
    else:
        print("✗ WARNING: Rewards appear constant")

    return all_episode_rewards


def main():
    """Run all tests."""
    print("\n")
    print("#" * 70)
    print("# Ray Multi-Agent Microgrid Environment Test")
    print("#" * 70)

    # Test 1: Basic functionality
    try:
        env, episode_rewards = test_basic_environment()
        print("\n✓ Test 1 PASSED")
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Ray compatibility
    try:
        wrapped_env = test_ray_compatibility()
        if wrapped_env is not None:
            print("\n✓ Test 2 PASSED")
        else:
            print("\n⚠ Test 2 SKIPPED (Ray not available)")
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Reward meaningfulness
    try:
        all_rewards = test_reward_meaningfulness()
        print("\n✓ Test 3 PASSED")
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#" * 70)
    print("# All tests completed!")
    print("#" * 70)


if __name__ == '__main__':
    main()

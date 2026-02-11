"""Event-driven testing for trained microgrid policies.

This script demonstrates testing trained policies in event-driven mode with
realistic timing delays, message passing, and asynchronous execution.
"""

import numpy as np
from typing import Dict

from case_studies.grid_age.envs import MicrogridEnv
from case_studies.grid_age.train import train_microgrid_ctde, NeuralPolicy
from heron.scheduling import TickConfig, JitterType
from heron.scheduling.analysis import EventAnalyzer


def test_event_driven_mode():
    """Test trained policies in event-driven mode with timing delays."""

    print("=" * 70)
    print("GridAges Event-Driven Testing")
    print("=" * 70)

    # Step 1: Train policies using CTDE
    print("\n1. Training policies with CTDE...")
    print("-" * 70)

    env = MicrogridEnv(
        num_microgrids=3,
        episode_steps=24,
        dt=1.0,
    )

    policies = train_microgrid_ctde(
        env,
        num_episodes=50,  # Quick training
        steps_per_episode=24,
        lr=0.02,
        print_every=10,
    )

    print("\n✅ Training complete!")

    # Step 2: Configure event-driven mode with timing delays
    print("\n2. Configuring event-driven mode...")
    print("-" * 70)

    # Attach trained policies to agents
    env.set_agent_policies(policies)

    # Configure tick configs with jitter for realistic timing
    field_tick_config = TickConfig.with_jitter(
        tick_interval=1.0,    # Field agents tick every 1 hour (aligned with episode steps)
        obs_delay=0.01,       # 36 second observation delay
        act_delay=0.02,       # 72 second action delay
        msg_delay=0.01,       # 36 second message delay
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.05,    # 5% timing jitter
        seed=42
    )

    system_tick_config = TickConfig.with_jitter(
        tick_interval=1.0,    # System agent ticks every 1 hour
        obs_delay=0.02,
        act_delay=0.03,
        msg_delay=0.02,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.05,
        seed=43
    )

    # Apply configs to agents
    system_agent = env._system_agent
    system_agent.tick_config = system_tick_config

    for agent_id, agent in env.registered_agents.items():
        if agent_id != "system" and hasattr(agent, 'tick_config'):
            agent.tick_config = field_tick_config

    print(f"✅ Configured {len(env.registered_agents)} agents with timing delays")
    print(f"   - Field agents: tick every {field_tick_config.tick_interval}h")
    print(f"   - Observation delay: {field_tick_config.obs_delay}h")
    print(f"   - Action delay: {field_tick_config.act_delay}h")
    print(f"   - Jitter: {field_tick_config.jitter_ratio*100:.0f}% Gaussian")

    # Step 3: Run event-driven simulation
    print("\n3. Running event-driven simulation...")
    print("-" * 70)

    # Reset environment for event-driven run
    env.reset(seed=999)

    # Create event analyzer for tracking
    event_analyzer = EventAnalyzer(verbose=False, track_data=True)

    # Run event-driven mode for 24 hours (one day)
    result = env.run_event_driven(
        event_analyzer=event_analyzer,
        t_end=24.0,  # 24-hour episode
    )

    print(f"✅ Event-driven simulation complete!")
    print(f"   - Events processed: {len(result.events) if hasattr(result, 'events') else 'N/A'}")
    print(f"   - Observations: {result.observation_count if hasattr(result, 'observation_count') else 0}")
    print(f"   - State updates: {result.state_update_count if hasattr(result, 'state_update_count') else 0}")

    # Step 4: Analyze results
    print("\n4. Event Analysis")
    print("-" * 70)

    # Print analysis summary
    if hasattr(result, 'observation_count'):
        print(f"\nMessage Counts:")
        print(f"   - Observations: {result.observation_count}")
        print(f"   - Global states: {result.global_state_count}")
        print(f"   - Local states: {result.local_state_count}")
        print(f"   - State updates: {result.state_update_count}")
        print(f"   - Action results: {result.action_result_count}")

    # Get final rewards from proxy
    if hasattr(env.proxy_agent, '_step_results') and env.proxy_agent._step_results:
        final_rewards = env.proxy_agent._step_results.get('rewards', {})
        if final_rewards:
            print("\nFinal Step Rewards:")
            for agent_id, reward in sorted(final_rewards.items()):
                if agent_id != 'system':
                    print(f"   - {agent_id}: {reward:.3f}")

    print("\n" + "=" * 70)
    print("Event-Driven Testing Complete!")
    print("=" * 70)

    return result


def compare_training_vs_event_driven():
    """Compare synchronous training mode vs event-driven testing mode."""

    print("\n" + "=" * 70)
    print("Comparison: Training Mode vs Event-Driven Mode")
    print("=" * 70)

    # Create environment
    env = MicrogridEnv(num_microgrids=2, episode_steps=10)  # Shorter for demo

    # Train policies
    print("\n1. Training policies (synchronous CTDE mode)...")
    policies = train_microgrid_ctde(
        env,
        num_episodes=20,
        steps_per_episode=10,
        print_every=10,
    )

    # Test in synchronous mode
    print("\n2. Testing in synchronous mode...")
    env.set_agent_policies(policies)
    obs, _ = env.reset(seed=42)

    agent_ids = [aid for aid, policy in policies.items()]
    sync_total_reward = 0.0

    for step in range(10):
        actions = {aid: policies[aid].forward_deterministic(obs[aid]) for aid in agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        sync_total_reward += sum(r for aid, r in rewards.items() if aid in agent_ids)

        if terminated.get("__all__", False):
            break

    print(f"   Synchronous total reward: {sync_total_reward:.2f}")

    # Test in event-driven mode
    print("\n3. Testing in event-driven mode...")

    # Configure timing
    for agent_id, agent in env.registered_agents.items():
        if hasattr(agent, 'tick_config'):
            agent.tick_config = TickConfig.deterministic(
                tick_interval=1.0,
                obs_delay=0.01,
                act_delay=0.01,
                msg_delay=0.01,
            )

    env.reset(seed=42)
    analyzer = EventAnalyzer(verbose=False, track_data=True)
    result = env.run_event_driven(event_analyzer=analyzer, t_end=10.0)

    print(f"   Event-driven mode:")
    print(f"   - Observations: {result.observation_count}")
    print(f"   - State updates: {result.state_update_count}")

    # Get final rewards
    if hasattr(env.proxy_agent, '_step_results') and env.proxy_agent._step_results:
        final_rewards = env.proxy_agent._step_results.get('rewards', {})
        event_total_reward = sum(r for aid, r in final_rewards.items() if aid in agent_ids)
        print(f"   - Final step rewards sum: {event_total_reward:.2f}")

    print("\n" + "=" * 70)
    print("Both modes executed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    # Run event-driven testing
    result = test_event_driven_mode()

    print("\n")

    # Compare modes
    compare_training_vs_event_driven()

    print("\n✅ All event-driven tests complete!")

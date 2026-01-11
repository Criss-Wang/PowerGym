"""
Example 6: Distributed Mode with ProxyAgent
============================================

This example demonstrates the distributed execution mode of PowerGrid 2.0, showcasing
how the ProxyAgent manages network state distribution between the environment and agents.

What you'll learn:
- How to enable distributed mode in the environment
- The role of ProxyAgent in managing information flow
- Differences between centralized and distributed modes
- How agents receive network state via message passing
- Message broker architecture for distributed systems

Key Differences from Centralized Mode:
- Agents don't access PandaPower network directly
- ProxyAgent acts as intermediary for network state distribution
- Environment publishes aggregated network state to ProxyAgent
- ProxyAgent distributes filtered state to individual agents
- Mimics realistic distributed control systems with limited observability

Architecture:
    Environment (PandaPower Network)
         ↓ (power flow results via messages)
    ProxyAgent (State Distribution Layer)
         ↓ (filtered state to each agent)
    GridAgents (MG1, MG2, MG3)
         ↓ (actions via messages)
    Environment

Runtime: ~30 seconds for 24 timesteps
"""

import numpy as np
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.configs.config_loader import load_config


def run_distributed_mode_example():
    """Run a simple distributed mode example with ProxyAgent."""

    print("=" * 80)
    print("Example 6: Distributed Mode with ProxyAgent")
    print("=" * 80)
    print()

    # ============================================================
    # Step 1: Create Environment with Distributed Mode Enabled
    # ============================================================
    print("Step 1: Creating environment in DISTRIBUTED mode...")
    print("-" * 80)

    # Load configuration
    config = load_config('ieee34_ieee13')
    config['max_episode_steps'] = 24
    config['centralized'] = False  # Enable distributed mode
    config['message_broker'] = 'in_memory'  # Use in-memory message broker

    # Create environment
    env = MultiAgentMicrogrids(config)

    print(f"Environment created with {len(env.possible_agents)} agents:")
    for agent_id in env.possible_agents:
        print(f"  - {agent_id}")
    print()

    # ============================================================
    # Step 2: Inspect ProxyAgent
    # ============================================================
    print("Step 2: Inspecting ProxyAgent...")
    print("-" * 80)

    proxy = env.proxy_agent
    print(f"ProxyAgent ID: {proxy.agent_id}")
    print(f"Number of subordinate agents: {len(proxy.subordinate_agents)}")
    print(f"Subordinate agents: {proxy.subordinate_agents}")
    print(f"Visibility rules: {proxy.visibility_rules if proxy.visibility_rules else 'None (all agents see all state)'}")
    print()

    # ============================================================
    # Step 3: Reset Environment and Observe Message Flow
    # ============================================================
    print("Step 3: Resetting environment and observing message flow...")
    print("-" * 80)

    obs, info = env.reset(seed=42)

    print("After reset:")
    print(f"  - ProxyAgent cache populated: {len(proxy.network_state_cache) > 0}")
    if proxy.network_state_cache:
        print(f"  - Cache keys: {list(proxy.network_state_cache.keys())}")
        if 'agents' in proxy.network_state_cache:
            print(f"  - Agent states available: {list(proxy.network_state_cache['agents'].keys())}")

    print(f"\nObservations received by {len(obs)} agents:")
    for agent_id, observation in obs.items():
        print(f"  - {agent_id}: shape={observation.shape}, mean={np.mean(observation):.3f}")
    print()

    # ============================================================
    # Step 4: Take Steps and Observe ProxyAgent Behavior
    # ============================================================
    print("Step 4: Taking steps and observing ProxyAgent behavior...")
    print("-" * 80)

    episode_rewards = {agent_id: [] for agent_id in env.possible_agents}

    for step in range(5):  # Run 5 steps
        # Sample random actions
        actions = {agent_id: env.action_spaces[agent_id].sample()
                  for agent_id in env.possible_agents}

        # Take step
        next_obs, rewards, dones, truncated, infos = env.step(actions)

        # Record rewards
        for agent_id in env.possible_agents:
            episode_rewards[agent_id].append(rewards[agent_id])

        # Print step summary
        print(f"Step {step + 1}:")
        print(f"  - ProxyAgent cache size: {len(proxy.network_state_cache)} keys")
        print(f"  - Rewards: {', '.join([f'{aid}={rewards[aid]:.2f}' for aid in env.possible_agents])}")

        # Inspect message broker channels
        if step == 0:
            print(f"  - Message broker channels: {len(env.message_broker.channels)}")
            print(f"  - Channel names: {list(env.message_broker.channels.keys())[:3]}...")  # Show first 3

    print()

    # ============================================================
    # Step 5: Summary Statistics
    # ============================================================
    print("Step 5: Summary Statistics")
    print("-" * 80)

    for agent_id, rewards_list in episode_rewards.items():
        total_reward = sum(rewards_list)
        avg_reward = np.mean(rewards_list)
        print(f"{agent_id}:")
        print(f"  - Total reward: {total_reward:.2f}")
        print(f"  - Average reward: {avg_reward:.2f}")
        print(f"  - Min/Max reward: {min(rewards_list):.2f} / {max(rewards_list):.2f}")

    print()


def compare_centralized_vs_distributed():
    """Compare centralized and distributed modes side-by-side."""

    print("=" * 80)
    print("Comparison: Centralized vs Distributed Mode")
    print("=" * 80)
    print()

    # Create both environments
    config_cent = load_config('ieee34_ieee13')
    config_cent['max_episode_steps'] = 24
    config_cent['centralized'] = True

    config_dist = load_config('ieee34_ieee13')
    config_dist['max_episode_steps'] = 24
    config_dist['centralized'] = False
    config_dist['message_broker'] = 'in_memory'

    env_cent = MultiAgentMicrogrids(config_cent)
    env_dist = MultiAgentMicrogrids(config_dist)

    print("Environment Comparison:")
    print("-" * 80)
    print(f"{'Feature':<40} {'Centralized':<20} {'Distributed':<20}")
    print("-" * 80)
    print(f"{'ProxyAgent exists':<40} {str(env_cent.proxy_agent is not None):<20} {str(env_dist.proxy_agent is not None):<20}")
    print(f"{'Message broker exists':<40} {str(env_cent.message_broker is not None):<20} {str(env_dist.message_broker is not None):<20}")
    print(f"{'Number of agents':<40} {len(env_cent.possible_agents):<20} {len(env_dist.possible_agents):<20}")

    # Reset both
    obs_cent, _ = env_cent.reset(seed=42)
    obs_dist, _ = env_dist.reset(seed=42)

    print(f"\nObservation Comparison:")
    print("-" * 80)
    for agent_id in env_cent.possible_agents:
        cent_shape = obs_cent[agent_id].shape
        dist_shape = obs_dist[agent_id].shape
        print(f"{agent_id}:")
        print(f"  - Centralized obs shape: {cent_shape}")
        print(f"  - Distributed obs shape: {dist_shape}")
        print(f"  - Note: Distributed mode has smaller observations (no direct network access)")

    print()


def demonstrate_visibility_rules():
    """Demonstrate how visibility rules control information access."""

    print("=" * 80)
    print("Demonstration: Visibility Rules in ProxyAgent")
    print("=" * 80)
    print()

    # Create environment
    config = load_config('ieee34_ieee13')
    config['max_episode_steps'] = 24
    config['centralized'] = False
    config['message_broker'] = 'in_memory'

    env = MultiAgentMicrogrids(config)

    # Configure visibility rules
    print("Setting visibility rules...")
    print("-" * 80)

    # Example: MG1 can only see voltage, MG2 can see voltage and line loading, MG3 sees nothing
    env.proxy_agent.visibility_rules = {
        'MG1': ['voltage_pu', 'bus_vm_pu'],
        'MG2': ['voltage_pu', 'line_loading', 'bus_vm_pu'],
        'MG3': []  # No network visibility
    }

    print("Visibility rules configured:")
    for agent_id, allowed_keys in env.proxy_agent.visibility_rules.items():
        print(f"  - {agent_id}: {allowed_keys if allowed_keys else 'No network info'}")

    print()

    # Reset and observe
    obs, _ = env.reset(seed=42)

    print("After reset with visibility rules:")
    print("-" * 80)

    # Manually trigger distribution to see filtered state
    env.proxy_agent.distribute_network_state_to_agents()

    # Check what each agent can see by inspecting messages
    broker = env.message_broker
    for agent_id in env.possible_agents:
        from powergrid.messaging.base import ChannelManager
        channel = ChannelManager.info_channel("proxy_agent", agent_id, env._env_id)
        messages = broker.consume(channel, recipient_id=agent_id, env_id=env._env_id, clear=False)

        if messages:
            payload = messages[-1].payload
            print(f"{agent_id} received state with keys: {list(payload.keys()) if payload else 'None'}")
        else:
            print(f"{agent_id} received no messages")

    print()
    print("Note: Visibility rules allow fine-grained control over information sharing")
    print("      in distributed systems, mimicking real-world communication constraints.")
    print()


def main():
    """Run all demonstrations."""

    # Main distributed mode example
    run_distributed_mode_example()

    # Comparison with centralized mode
    compare_centralized_vs_distributed()

    # Visibility rules demonstration
    demonstrate_visibility_rules()

    print("=" * 80)
    print("Example 6 Complete!")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("1. Distributed mode enables realistic multi-agent systems with limited observability")
    print("2. ProxyAgent manages network state distribution via message passing")
    print("3. Visibility rules provide fine-grained control over information access")
    print("4. Message broker architecture supports scalable distributed systems")
    print("5. Both centralized and distributed modes produce valid, functional environments")
    print()


if __name__ == "__main__":
    main()

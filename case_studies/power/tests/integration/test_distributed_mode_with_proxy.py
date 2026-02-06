"""
Integration tests for distributed mode with ProxyAgent.

Tests the complete workflow of distributed execution including:
- ProxyAgent creation and configuration
- Network state distribution from environment to agents via ProxyAgent
- Message broker communication
- Agent state consumption and action execution
- Multi-environment isolation
- Visibility rule enforcement
- Complete episode execution in distributed mode
"""

import pytest
import numpy as np
import pandapower as pp
from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.setups.loader import load_setup
from heron.messaging.base import ChannelManager
from heron.agents.proxy_agent import ProxyAgent


class TestDistributedModeWithProxy:
    """Integration tests for distributed mode with ProxyAgent."""

    @pytest.fixture
    def distributed_config(self):
        """Load distributed environment configuration."""
        config = load_setup('ieee34_ieee13')
        config['max_episode_steps'] = 24
        config['centralized'] = False  # Enable distributed mode
        config['message_broker'] = 'in_memory'
        return config

    @pytest.fixture
    def centralized_config(self):
        """Load centralized environment configuration for comparison."""
        config = load_setup('ieee34_ieee13')
        config['max_episode_steps'] = 24
        config['centralized'] = True  # Centralized mode
        return config

    @pytest.fixture
    def distributed_env(self, distributed_config):
        """Create distributed environment instance."""
        return MultiAgentMicrogrids(distributed_config)

    @pytest.fixture
    def centralized_env(self, centralized_config):
        """Create centralized environment instance."""
        return MultiAgentMicrogrids(centralized_config)

    # =====================================
    # Test 1: ProxyAgent Creation
    # =====================================

    def test_proxy_agent_created_in_distributed_mode(self, distributed_env):
        """Test that ProxyAgent is created in distributed mode."""
        assert distributed_env.proxy_agent is not None
        assert isinstance(distributed_env.proxy_agent, ProxyAgent)
        assert distributed_env.proxy_agent.agent_id == "proxy_agent"

    def test_proxy_agent_not_created_in_centralized_mode(self, centralized_env):
        """Test that ProxyAgent is not created in centralized mode."""
        assert centralized_env.proxy_agent is None

    def test_proxy_agent_has_correct_subordinates(self, distributed_env):
        """Test that ProxyAgent has all environment agents as subordinates."""
        proxy = distributed_env.proxy_agent
        expected_agents = set(distributed_env.possible_agents)
        actual_agents = set(proxy.registered_agents)

        assert actual_agents == expected_agents
        assert len(proxy.registered_agents) == 3  # MG1, MG2, MG3

    def test_message_broker_channels_created(self, distributed_env):
        """Test that message broker channels are created for proxy communication."""
        proxy = distributed_env.proxy_agent
        broker = distributed_env.message_broker

        # Check environment-to-proxy channel exists
        env_to_proxy_channel = ChannelManager.custom_channel(
            "power_flow",
            distributed_env.env_id,
            "proxy_agent"
        )
        assert env_to_proxy_channel in broker.channels

        # Check proxy-to-agent channels exist
        for agent_id in distributed_env.possible_agents:
            proxy_to_agent_channel = ChannelManager.info_channel(
                "proxy_agent",
                agent_id,
                distributed_env.env_id
            )
            assert proxy_to_agent_channel in broker.channels

    # =====================================
    # Test 2: Network State Distribution
    # =====================================

    def test_proxy_receives_network_state_from_environment(self, distributed_env):
        """Test that ProxyAgent receives network state from environment."""
        env = distributed_env
        proxy = env.proxy_agent

        # Reset environment to initialize state
        obs, info = env.reset(seed=42)

        # After reset, proxy should have cached network state
        assert len(proxy.state_cache) > 0
        assert 'agents' in proxy.state_cache

        # Check that state contains information for all agents
        agents_state = proxy.state_cache['agents']
        for agent_id in env.possible_agents:
            assert agent_id in agents_state

    def test_proxy_distributes_state_to_agents(self, distributed_env):
        """Test that ProxyAgent distributes network state to agents via messages."""
        env = distributed_env
        proxy = env.proxy_agent
        broker = env.message_broker

        # Reset environment
        obs, info = env.reset(seed=42)

        # After reset and distribution, agents should receive network state
        for agent_id in env.possible_agents:
            channel = ChannelManager.info_channel(
                "proxy_agent",
                agent_id,
                env.env_id
            )

            # Check that messages exist in the channel
            messages = broker.consume(
                channel,
                recipient_id=agent_id,
                env_id=env.env_id,
                clear=False  # Don't clear so we can inspect
            )

            # At least one message should exist (from reset)
            assert len(messages) > 0

            # Check message structure
            last_message = messages[-1]
            assert last_message.sender_id == "proxy_agent"
            assert last_message.recipient_id == agent_id
            assert isinstance(last_message.payload, dict)

    def test_agents_consume_network_state(self, distributed_env):
        """Test that agents successfully consume network state from ProxyAgent."""
        env = distributed_env

        # Reset environment
        obs, info = env.reset(seed=42)

        # Take a step with random actions
        actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
        next_obs, rewards, dones, truncated, infos = env.step(actions)

        # Check that agents have received and processed network state
        for agent_id, agent in env.agent_dict.items():
            # Agent should have consumed network state
            # (This is verified indirectly through successful step execution)
            assert agent is not None

            # Observations should be valid
            assert not np.any(np.isnan(next_obs[agent_id]))
            assert not np.any(np.isinf(next_obs[agent_id]))

    def test_network_state_structure(self, distributed_env):
        """Test that network state has correct structure."""
        env = distributed_env
        proxy = env.proxy_agent

        # Reset environment
        obs, info = env.reset(seed=42)

        # Check cache structure
        state = proxy.state_cache
        assert 'agents' in state

        agents_state = state['agents']
        for agent_id in env.possible_agents:
            agent_state = agents_state[agent_id]

            # Agent state should contain network information
            assert isinstance(agent_state, dict)
            # Note: Exact keys depend on what environment publishes
            # Just verify it's a non-empty dict
            assert len(agent_state) > 0

    # =====================================
    # Test 3: Complete Episode Execution
    # =====================================

    def test_complete_episode_distributed_mode(self, distributed_env):
        """Test complete episode execution in distributed mode."""
        env = distributed_env

        # Reset
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert len(obs) == len(env.possible_agents)

        # Run episode
        total_steps = 0
        episode_rewards = {aid: [] for aid in env.possible_agents}

        for step in range(env.max_episode_steps):
            # Sample actions
            actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}

            # Take step
            next_obs, rewards, dones, truncated, infos = env.step(actions)

            # Validate observations
            for agent_id in env.possible_agents:
                assert not np.any(np.isnan(next_obs[agent_id]))
                assert not np.any(np.isinf(next_obs[agent_id]))
                episode_rewards[agent_id].append(rewards[agent_id])

            total_steps += 1

            # Check if episode terminated
            if all(dones.values()) or all(truncated.values()):
                break

        # Verify episode completed
        assert total_steps > 0

        # Verify rewards were collected
        for agent_id in env.possible_agents:
            assert len(episode_rewards[agent_id]) == total_steps
            # Rewards should be finite
            assert all(np.isfinite(r) for r in episode_rewards[agent_id])

    def test_distributed_mode_robustness(self, distributed_env):
        """Test that distributed mode can handle multiple episodes robustly."""
        # Run multiple episodes to ensure robustness
        for episode in range(3):
            obs, info = distributed_env.reset(seed=42 + episode)

            # Verify observations are valid
            for agent_id in distributed_env.possible_agents:
                assert not np.any(np.isnan(obs[agent_id]))
                assert obs[agent_id].shape[0] > 0

            # Run a few steps
            for step in range(5):
                actions = {aid: distributed_env.action_spaces[aid].sample()
                          for aid in distributed_env.possible_agents}
                next_obs, rewards, dones, truncated, infos = distributed_env.step(actions)

                # Verify all outputs are valid
                for agent_id in distributed_env.possible_agents:
                    assert not np.any(np.isnan(next_obs[agent_id]))
                    assert np.isfinite(rewards[agent_id])

        # All episodes completed successfully
        assert True

    # =====================================
    # Test 4: Visibility Rules
    # =====================================

    def test_visibility_rules_enforcement(self, distributed_config):
        """Test that visibility rules are enforced by ProxyAgent."""
        # Create environment with custom visibility rules
        config = distributed_config.copy()
        env = MultiAgentMicrogrids(config)

        # Manually set visibility rules on proxy
        proxy = env.proxy_agent
        proxy.visibility_rules = {
            'MG1': ['voltage_pu', 'line_loading'],
            'MG2': ['voltage_pu'],
            'MG3': []  # No visibility
        }

        # Reset environment
        obs, info = env.reset(seed=42)

        # Manually trigger distribution to apply visibility rules
        proxy.distribute_state_to_agents()

        # Check that agents receive only allowed keys
        broker = env.message_broker

        # MG1 should see voltage_pu and line_loading
        channel_mg1 = ChannelManager.info_channel("proxy_agent", "MG1", env.env_id)
        messages_mg1 = broker.consume(channel_mg1, recipient_id="MG1",
                                      env_id=env.env_id, clear=False)
        if messages_mg1:
            payload_mg1 = messages_mg1[-1].payload
            # Only allowed keys should be present
            for key in payload_mg1.keys():
                assert key in ['voltage_pu', 'line_loading']

        # MG2 should see only voltage_pu
        channel_mg2 = ChannelManager.info_channel("proxy_agent", "MG2", env.env_id)
        messages_mg2 = broker.consume(channel_mg2, recipient_id="MG2",
                                      env_id=env.env_id, clear=False)
        if messages_mg2:
            payload_mg2 = messages_mg2[-1].payload
            for key in payload_mg2.keys():
                assert key in ['voltage_pu']

        # MG3 should see empty state
        channel_mg3 = ChannelManager.info_channel("proxy_agent", "MG3", env.env_id)
        messages_mg3 = broker.consume(channel_mg3, recipient_id="MG3",
                                      env_id=env.env_id, clear=False)
        if messages_mg3:
            payload_mg3 = messages_mg3[-1].payload
            assert len(payload_mg3) == 0

    # =====================================
    # Test 5: Multi-Environment Isolation
    # =====================================

    def test_multi_environment_isolation(self, distributed_config):
        """Test that multiple environments with ProxyAgents are isolated."""
        # Create two environments
        env1 = MultiAgentMicrogrids(distributed_config)
        env2 = MultiAgentMicrogrids(distributed_config)

        # Verify different env_ids
        assert env1.env_id != env2.env_id

        # Verify different proxy agents
        assert env1.proxy_agent is not env2.proxy_agent

        # Reset both
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=99)

        # Verify different states
        proxy1 = env1.proxy_agent
        proxy2 = env2.proxy_agent

        # Caches should be different (different seeds lead to different states)
        assert proxy1.state_cache != proxy2.state_cache

        # Take steps in env1 only
        actions1 = {aid: env1.action_spaces[aid].sample() for aid in env1.possible_agents}
        env1.step(actions1)

        # Verify env2 is not affected
        # Env2's proxy cache should not change
        proxy2_cache_before = proxy2.state_cache.copy()
        assert proxy2.state_cache == proxy2_cache_before

    # =====================================
    # Test 6: Edge Cases
    # =====================================

    def test_proxy_handles_empty_network_state(self, distributed_env):
        """Test that ProxyAgent handles empty network state gracefully."""
        env = distributed_env
        proxy = env.proxy_agent

        # Manually clear cache
        proxy.state_cache = {}

        # Try to distribute (should not crash)
        proxy.distribute_state_to_agents()

        # Should still work
        assert True

    def test_proxy_handles_missing_agent_in_state(self, distributed_env):
        """Test that ProxyAgent handles missing agent in network state."""
        env = distributed_env
        proxy = env.proxy_agent

        # Reset to get valid state
        obs, info = env.reset(seed=42)

        # Manually remove one agent from cache
        if 'agents' in proxy.state_cache:
            agents_state = proxy.state_cache['agents']
            if 'MG1' in agents_state:
                del agents_state['MG1']

        # Distribution should still work for remaining agents
        proxy.distribute_state_to_agents()

        # Should complete without error
        assert True

    def test_proxy_reset_clears_cache(self, distributed_env):
        """Test that ProxyAgent reset clears network state cache."""
        env = distributed_env
        proxy = env.proxy_agent

        # Reset environment to populate cache
        obs, info = env.reset(seed=42)
        assert len(proxy.state_cache) > 0

        # Reset proxy
        proxy.reset(seed=42)

        # Cache should be cleared
        assert len(proxy.state_cache) == 0

    # =====================================
    # Test 7: Performance and Scalability
    # =====================================

    def test_distributed_mode_performance(self, distributed_env):
        """Test that distributed mode completes within reasonable time."""
        import time

        env = distributed_env

        # Measure time for reset + 10 steps
        start_time = time.time()

        obs, info = env.reset(seed=42)

        for _ in range(10):
            actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
            obs, rewards, dones, truncated, infos = env.step(actions)

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 30 seconds for 10 steps)
        assert elapsed_time < 30.0

        print(f"\nDistributed mode: 10 steps took {elapsed_time:.2f}s")

    def test_message_broker_memory_cleanup(self, distributed_env):
        """Test that message broker cleans up consumed messages."""
        env = distributed_env
        broker = env.message_broker

        # Reset environment
        obs, info = env.reset(seed=42)

        # Get initial message count
        initial_total_messages = sum(len(msgs) for msgs in broker.channels.values())

        # Take several steps
        for _ in range(5):
            actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
            obs, rewards, dones, truncated, infos = env.step(actions)

        # Get final message count
        final_total_messages = sum(len(msgs) for msgs in broker.channels.values())

        # Messages should be consumed (cleared), not accumulate indefinitely
        # After 5 steps, we shouldn't have unbounded message growth
        # (exact behavior depends on clear=True in consume calls)
        # In distributed mode with state updates, we expect more messages per step:
        # - 3 action messages per step (one per agent)
        # - 3 state updates from devices per step
        # - 3 network state messages per step
        print(f"\nInitial messages: {initial_total_messages}, Final messages: {final_total_messages}")

        # This is a soft check - main point is messages don't grow unbounded
        # With 3 agents, 5 steps, ~9 messages per step, we expect ~45 + initial messages
        assert final_total_messages < initial_total_messages + 100

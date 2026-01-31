"""Comprehensive tests for ProxyAgent.

Tests cover:
1. Proxy agent initialization and setup
2. Network state reception from environment
3. State caching and management
4. State distribution to subordinate agents
5. Visibility rule filtering
6. Multi-environment isolation
7. Edge cases and error handling
"""

import pytest
from typing import Dict, Any

from powergrid.agents.proxy_agent import ProxyAgent, PROXY_LEVEL
from heron.messaging.base import Message, MessageType, ChannelManager
from heron.messaging.in_memory_broker import InMemoryBroker


# =============================================================================
# ProxyAgent Initialization Tests
# =============================================================================

class TestProxyAgentInitialization:
    """Test ProxyAgent initialization."""

    def test_initialization_basic(self):
        """Test basic proxy agent initialization."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        assert proxy.agent_id == "proxy"
        assert proxy.level == PROXY_LEVEL
        assert proxy.env_id == "env_0"
        assert proxy.message_broker == broker
        assert proxy.network_state_cache == {}
        assert proxy.subordinate_agents == []
        assert proxy.visibility_rules == {}

    def test_initialization_with_subordinates(self):
        """Test initialization with subordinate agents."""
        broker = InMemoryBroker()
        subordinates = ["agent_1", "agent_2", "agent_3"]

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=subordinates
        )

        assert proxy.subordinate_agents == subordinates
        assert len(proxy.subordinate_agents) == 3

    def test_initialization_with_visibility_rules(self):
        """Test initialization with visibility rules."""
        broker = InMemoryBroker()
        visibility_rules = {
            "agent_1": ["voltage", "power"],
            "agent_2": ["voltage"],
        }

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            visibility_rules=visibility_rules
        )

        assert proxy.visibility_rules == visibility_rules

    def test_initialization_without_broker_raises_error(self):
        """Test initialization without message broker raises error."""
        with pytest.raises(ValueError, match="requires a message broker"):
            ProxyAgent(agent_id="proxy", message_broker=None)

    def test_channel_setup(self):
        """Test proxy channels are created correctly."""
        broker = InMemoryBroker()
        subordinates = ["agent_1", "agent_2"]

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=subordinates
        )

        # Check env->proxy channel exists
        env_channel = ChannelManager.custom_channel("power_flow", "env_0", "proxy")
        assert env_channel in broker.channels

        # Check proxy->agent channels exist
        for agent_id in subordinates:
            agent_channel = ChannelManager.info_channel("proxy", agent_id, "env_0")
            assert agent_channel in broker.channels

    def test_repr(self):
        """Test string representation."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="my_proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["a1", "a2", "a3"]
        )

        repr_str = repr(proxy)
        assert "ProxyAgent" in repr_str
        assert "my_proxy" in repr_str
        assert "subordinates=3" in repr_str


# =============================================================================
# Network State Reception Tests
# =============================================================================

class TestNetworkStateReception:
    """Test receiving network state from environment."""

    def test_receive_network_state_single_message(self):
        """Test receiving single network state message."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        # Simulate environment sending network state
        channel = ChannelManager.custom_channel("power_flow", "env_0", "proxy")
        state_payload = {
            "converged": True,
            "agents": {
                "agent_1": {"voltage": 1.0, "power": 100.0}
            }
        }

        msg = Message(
            env_id="env_0",
            sender_id="environment",
            recipient_id="proxy",
            timestamp=1.0,
            message_type=MessageType.STATE_UPDATE,
            payload=state_payload
        )
        broker.publish(channel, msg)

        # Receive state
        received_state = proxy.receive_network_state_from_environment()

        assert received_state == state_payload
        assert proxy.network_state_cache == state_payload

    def test_receive_network_state_multiple_messages_uses_latest(self):
        """Test receiving multiple messages uses the latest one."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        channel = ChannelManager.custom_channel("power_flow", "env_0", "proxy")

        # Send multiple states
        for i in range(3):
            state_payload = {
                "converged": True,
                "timestep": i,
                "agents": {}
            }
            msg = Message(
                env_id="env_0",
                sender_id="environment",
                recipient_id="proxy",
                timestamp=float(i),
                message_type=MessageType.STATE_UPDATE,
                payload=state_payload
            )
            broker.publish(channel, msg)

        # Receive state - should get latest
        received_state = proxy.receive_network_state_from_environment()

        assert received_state["timestep"] == 2  # Latest

    def test_receive_network_state_no_message(self):
        """Test receiving when no message is available."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        received_state = proxy.receive_network_state_from_environment()
        assert received_state is None
        assert proxy.network_state_cache == {}

    def test_receive_network_state_clears_messages(self):
        """Test receiving clears consumed messages."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        channel = ChannelManager.custom_channel("power_flow", "env_0", "proxy")
        state_payload = {"converged": True, "agents": {}}

        msg = Message(
            env_id="env_0",
            sender_id="environment",
            recipient_id="proxy",
            timestamp=1.0,
            message_type=MessageType.STATE_UPDATE,
            payload=state_payload
        )
        broker.publish(channel, msg)

        # First receive
        proxy.receive_network_state_from_environment()

        # Second receive should return None (messages cleared)
        received_state = proxy.receive_network_state_from_environment()
        assert received_state is None


# =============================================================================
# State Distribution Tests
# =============================================================================

class TestStateDistribution:
    """Test distributing network state to agents."""

    def test_distribute_state_to_single_agent(self):
        """Test distributing state to single agent."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["agent_1"]
        )

        # Set cached state
        proxy.network_state_cache = {
            "converged": True,
            "agents": {
                "agent_1": {"voltage": 1.0, "power": 100.0}
            }
        }

        # Distribute state
        proxy.distribute_network_state_to_agents()

        # Check message sent
        channel = ChannelManager.info_channel("proxy", "agent_1", "env_0")
        messages = broker.consume(channel, "agent_1", "env_0")

        assert len(messages) == 1
        assert messages[0].payload["voltage"] == 1.0
        assert messages[0].payload["power"] == 100.0

    def test_distribute_state_to_multiple_agents(self):
        """Test distributing state to multiple agents."""
        broker = InMemoryBroker()
        subordinates = ["agent_1", "agent_2", "agent_3"]

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=subordinates
        )

        # Set cached state with different values for each agent
        proxy.network_state_cache = {
            "converged": True,
            "agents": {
                "agent_1": {"voltage": 1.0},
                "agent_2": {"voltage": 0.98},
                "agent_3": {"voltage": 1.02}
            }
        }

        # Distribute state
        proxy.distribute_network_state_to_agents()

        # Verify each agent received their specific state
        for i, agent_id in enumerate(subordinates, 1):
            channel = ChannelManager.info_channel("proxy", agent_id, "env_0")
            messages = broker.consume(channel, agent_id, "env_0")

            assert len(messages) == 1
            assert "voltage" in messages[0].payload

    def test_distribute_state_empty_cache(self):
        """Test distributing when cache is empty does nothing."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["agent_1"]
        )

        # Cache is empty
        proxy.network_state_cache = {}

        # Distribute state - should do nothing
        proxy.distribute_network_state_to_agents()

        # No messages should be sent
        channel = ChannelManager.info_channel("proxy", "agent_1", "env_0")
        messages = broker.consume(channel, "agent_1", "env_0")
        assert len(messages) == 0

    def test_distribute_state_agent_not_in_aggregated_state(self):
        """Test distributing when agent not in aggregated state."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["agent_1", "agent_2"]
        )

        # Only agent_1 in state
        proxy.network_state_cache = {
            "converged": True,
            "agents": {
                "agent_1": {"voltage": 1.0}
            }
        }

        # Distribute state
        proxy.distribute_network_state_to_agents()

        # agent_1 should get state
        channel1 = ChannelManager.info_channel("proxy", "agent_1", "env_0")
        messages1 = broker.consume(channel1, "agent_1", "env_0")
        assert len(messages1) == 1

        # agent_2 should get empty state
        channel2 = ChannelManager.info_channel("proxy", "agent_2", "env_0")
        messages2 = broker.consume(channel2, "agent_2", "env_0")
        assert len(messages2) == 1
        assert messages2[0].payload == {}


# =============================================================================
# Visibility Filtering Tests
# =============================================================================

class TestVisibilityFiltering:
    """Test visibility rule filtering."""

    def test_filter_state_no_rules(self):
        """Test filtering with no visibility rules returns full state."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        agent_state = {"voltage": 1.0, "power": 100.0, "cost": 50.0}
        filtered = proxy._filter_state_for_agent("agent_1", agent_state)

        assert filtered == agent_state

    def test_filter_state_with_rules(self):
        """Test filtering with visibility rules."""
        broker = InMemoryBroker()
        visibility_rules = {
            "agent_1": ["voltage", "power"],
            "agent_2": ["voltage"]
        }

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            visibility_rules=visibility_rules
        )

        agent_state = {"voltage": 1.0, "power": 100.0, "cost": 50.0}

        # agent_1 should see voltage and power
        filtered1 = proxy._filter_state_for_agent("agent_1", agent_state)
        assert filtered1 == {"voltage": 1.0, "power": 100.0}
        assert "cost" not in filtered1

        # agent_2 should see only voltage
        filtered2 = proxy._filter_state_for_agent("agent_2", agent_state)
        assert filtered2 == {"voltage": 1.0}
        assert "power" not in filtered2
        assert "cost" not in filtered2

    def test_filter_state_missing_keys(self):
        """Test filtering when allowed keys don't exist in state."""
        broker = InMemoryBroker()
        visibility_rules = {
            "agent_1": ["voltage", "power", "nonexistent"]
        }

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            visibility_rules=visibility_rules
        )

        agent_state = {"voltage": 1.0}  # power and nonexistent missing

        filtered = proxy._filter_state_for_agent("agent_1", agent_state)

        # Should only include keys that exist
        assert filtered == {"voltage": 1.0}

    def test_distribute_applies_visibility_rules(self):
        """Test state distribution applies visibility filtering."""
        broker = InMemoryBroker()
        visibility_rules = {
            "agent_1": ["voltage"],
            "agent_2": ["power"]
        }

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["agent_1", "agent_2"],
            visibility_rules=visibility_rules
        )

        proxy.network_state_cache = {
            "converged": True,
            "agents": {
                "agent_1": {"voltage": 1.0, "power": 100.0, "cost": 50.0},
                "agent_2": {"voltage": 0.98, "power": 110.0, "cost": 55.0}
            }
        }

        proxy.distribute_network_state_to_agents()

        # Check agent_1 only sees voltage
        channel1 = ChannelManager.info_channel("proxy", "agent_1", "env_0")
        messages1 = broker.consume(channel1, "agent_1", "env_0")
        assert messages1[0].payload == {"voltage": 1.0}

        # Check agent_2 only sees power
        channel2 = ChannelManager.info_channel("proxy", "agent_2", "env_0")
        messages2 = broker.consume(channel2, "agent_2", "env_0")
        assert messages2[0].payload == {"power": 110.0}


# =============================================================================
# On-Demand State Retrieval Tests
# =============================================================================

class TestOnDemandStateRetrieval:
    """Test on-demand state retrieval by agents."""

    def test_get_latest_state_for_agent(self):
        """Test getting latest state for specific agent."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        proxy.network_state_cache = {
            "converged": True,
            "agents": {
                "agent_1": {"voltage": 1.0, "power": 100.0},
                "agent_2": {"voltage": 0.98, "power": 110.0}
            }
        }

        state1 = proxy.get_latest_network_state_for_agent("agent_1")
        assert state1 == {"voltage": 1.0, "power": 100.0}

        state2 = proxy.get_latest_network_state_for_agent("agent_2")
        assert state2 == {"voltage": 0.98, "power": 110.0}

    def test_get_latest_state_with_visibility_rules(self):
        """Test on-demand retrieval applies visibility rules."""
        broker = InMemoryBroker()
        visibility_rules = {"agent_1": ["voltage"]}

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            visibility_rules=visibility_rules
        )

        proxy.network_state_cache = {
            "converged": True,
            "agents": {
                "agent_1": {"voltage": 1.0, "power": 100.0, "cost": 50.0}
            }
        }

        state = proxy.get_latest_network_state_for_agent("agent_1")
        assert state == {"voltage": 1.0}

    def test_get_latest_state_agent_not_in_cache(self):
        """Test retrieving state for agent not in cache."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        proxy.network_state_cache = {
            "converged": True,
            "agents": {}
        }

        state = proxy.get_latest_network_state_for_agent("nonexistent_agent")
        assert state == {}


# =============================================================================
# Multi-Environment Isolation Tests
# =============================================================================

class TestMultiEnvironmentIsolation:
    """Test isolation between multiple environments."""

    def test_multiple_proxies_different_environments(self):
        """Test multiple proxies in different environments are isolated."""
        broker = InMemoryBroker()

        proxy1 = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["agent_1"]
        )

        proxy2 = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_1",
            subordinate_agents=["agent_1"]
        )

        # Send state to both proxies
        channel1 = ChannelManager.custom_channel("power_flow", "env_0", "proxy")
        msg1 = Message(
            env_id="env_0",
            sender_id="environment",
            recipient_id="proxy",
            timestamp=1.0,
            message_type=MessageType.STATE_UPDATE,
            payload={"env": 0, "agents": {"agent_1": {"value": 100}}}
        )
        broker.publish(channel1, msg1)

        channel2 = ChannelManager.custom_channel("power_flow", "env_1", "proxy")
        msg2 = Message(
            env_id="env_1",
            sender_id="environment",
            recipient_id="proxy",
            timestamp=1.0,
            message_type=MessageType.STATE_UPDATE,
            payload={"env": 1, "agents": {"agent_1": {"value": 200}}}
        )
        broker.publish(channel2, msg2)

        # Each proxy should receive only its environment's state
        state1 = proxy1.receive_network_state_from_environment()
        state2 = proxy2.receive_network_state_from_environment()

        assert state1["env"] == 0
        assert state2["env"] == 1

    def test_distribution_environment_isolation(self):
        """Test state distribution respects environment boundaries."""
        broker = InMemoryBroker()

        proxy1 = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["agent_1"]
        )

        proxy2 = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_1",
            subordinate_agents=["agent_1"]
        )

        # Set different states
        proxy1.network_state_cache = {
            "converged": True,
            "agents": {"agent_1": {"voltage": 1.0}}
        }

        proxy2.network_state_cache = {
            "converged": True,
            "agents": {"agent_1": {"voltage": 0.95}}
        }

        # Distribute from both proxies
        proxy1.distribute_network_state_to_agents()
        proxy2.distribute_network_state_to_agents()

        # Verify isolation
        channel1 = ChannelManager.info_channel("proxy", "agent_1", "env_0")
        messages1 = broker.consume(channel1, "agent_1", "env_0")
        assert messages1[0].payload["voltage"] == 1.0

        channel2 = ChannelManager.info_channel("proxy", "agent_1", "env_1")
        messages2 = broker.consume(channel2, "agent_1", "env_1")
        assert messages2[0].payload["voltage"] == 0.95


# =============================================================================
# Reset and Lifecycle Tests
# =============================================================================

class TestProxyAgentLifecycle:
    """Test proxy agent lifecycle operations."""

    def test_reset_clears_cache(self):
        """Test reset clears network state cache."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        # Set some cached state
        proxy.network_state_cache = {
            "converged": True,
            "agents": {"agent_1": {"voltage": 1.0}}
        }

        # Reset
        proxy.reset()

        # Cache should be cleared
        assert proxy.network_state_cache == {}

    def test_observe_returns_empty(self):
        """Test observe returns empty observation."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        obs = proxy.observe()

        assert obs.local == {}
        assert obs.global_info == {}

    def test_act_does_nothing(self):
        """Test act does nothing (proxy doesn't take actions)."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0"
        )

        from heron.core.observation import Observation
        obs = Observation(timestamp=0.0, local={}, global_info={})

        result = proxy.act(obs)
        assert result is None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_subordinates_list(self):
        """Test proxy with no subordinate agents."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=[]
        )

        proxy.network_state_cache = {
            "converged": True,
            "agents": {}
        }

        # Should not raise error
        proxy.distribute_network_state_to_agents()

    def test_visibility_rules_for_nonexistent_agent(self):
        """Test visibility rules for agent not in subordinates."""
        broker = InMemoryBroker()
        visibility_rules = {
            "nonexistent_agent": ["voltage"]
        }

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["agent_1"],
            visibility_rules=visibility_rules
        )

        # Should not raise error
        proxy.network_state_cache = {
            "converged": True,
            "agents": {"agent_1": {"voltage": 1.0}}
        }
        proxy.distribute_network_state_to_agents()

    def test_complex_nested_state(self):
        """Test handling complex nested state structures."""
        broker = InMemoryBroker()
        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            subordinate_agents=["agent_1"]
        )

        complex_state = {
            "converged": True,
            "agents": {
                "agent_1": {
                    "voltage": 1.0,
                    "devices": {
                        "gen1": {"p": 100.0, "q": 20.0},
                        "ess1": {"soc": 0.5}
                    },
                    "buses": [{"id": 1, "v": 1.0}, {"id": 2, "v": 0.98}]
                }
            }
        }

        proxy.network_state_cache = complex_state
        proxy.distribute_network_state_to_agents()

        # Should handle nested structures
        channel = ChannelManager.info_channel("proxy", "agent_1", "env_0")
        messages = broker.consume(channel, "agent_1", "env_0")

        assert len(messages) == 1
        payload = messages[0].payload
        assert payload["voltage"] == 1.0
        assert "devices" in payload
        assert len(payload["buses"]) == 2

    def test_state_immutability(self):
        """Test that filtering doesn't modify original state."""
        broker = InMemoryBroker()
        visibility_rules = {"agent_1": ["voltage"]}

        proxy = ProxyAgent(
            agent_id="proxy",
            message_broker=broker,
            env_id="env_0",
            visibility_rules=visibility_rules
        )

        original_state = {"voltage": 1.0, "power": 100.0, "cost": 50.0}
        filtered_state = proxy._filter_state_for_agent("agent_1", original_state)

        # Original should be unchanged
        assert "power" in original_state
        assert "cost" in original_state

        # Filtered should only have voltage
        assert filtered_state == {"voltage": 1.0}

"""Integration tests for message broker communication.

Tests the HERON message-based inter-agent communication system that enables
asynchronous, decoupled coordination between agents at different hierarchy levels.
"""

import pytest
import numpy as np

from heron.messaging.in_memory_broker import InMemoryBroker
from heron.messaging.base import Message, MessageType
from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.core.feature import FeatureProvider


# =============================================================================
# Test Fixtures
# =============================================================================

class SimpleFeature(FeatureProvider):
    """Simple feature for testing."""
    visibility = ["public"]

    def __init__(self, value: float = 0.0):
        self.value = value

    def vector(self):
        return np.array([self.value], dtype=np.float32)

    def names(self):
        return ["value"]

    def to_dict(self):
        return {"value": self.value}

    @classmethod
    def from_dict(cls, d):
        return cls(value=d.get("value", 0.0))

    def set_values(self, **kwargs):
        if "value" in kwargs:
            self.value = kwargs["value"]


class SimpleFieldAgent(FieldAgent):
    """Simple field agent for communication testing."""

    def set_action(self):
        self.action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))

    def set_state(self):
        self.state.features = [SimpleFeature(value=0.0)]

    def _get_obs(self, proxy=None):
        return self.state.vector()


class SimpleCoordinator(CoordinatorAgent):
    """Simple coordinator for communication testing."""

    def _build_subordinates(self, agent_configs, env_id=None, upstream_id=None):
        agents = {}
        for config in agent_configs:
            agent_id = config.get("id")
            agents[agent_id] = SimpleFieldAgent(
                agent_id=agent_id,
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )
        return agents


# =============================================================================
# Integration Tests
# =============================================================================

class TestMessageBrokerBasics:
    """Test basic message broker functionality."""

    def test_broker_creation_and_channel_setup(self):
        """Test creating broker and setting up channels."""
        broker = InMemoryBroker()

        # Create channels for agents
        broker.create_channel("agent_1")
        broker.create_channel("agent_2")

        assert "agent_1" in broker.channels
        assert "agent_2" in broker.channels

    def test_message_publish_and_consume(self):
        """Test that messages can be published and consumed."""
        broker = InMemoryBroker()
        broker.create_channel("device_1")

        # Publish message
        msg = Message(
            env_id="test_env",
            sender_id="coord_1",
            recipient_id="device_1",
            timestamp=1.0,
            message_type=MessageType.INFO,
            payload={"command": "set_power", "value": 50.0},
        )
        broker.publish("device_1", msg)

        # Consume message
        messages = broker.consume("device_1", "device_1", "test_env")

        assert len(messages) == 1
        assert messages[0].payload["command"] == "set_power"
        assert messages[0].payload["value"] == 50.0


class TestHierarchicalCommunication:
    """Test communication patterns in agent hierarchy."""

    def test_coordinator_to_subordinate_communication(self):
        """Test coordinator sending messages to subordinates."""
        broker = InMemoryBroker()

        config = {
            "agents": [
                {"id": "device_1"},
                {"id": "device_2"},
            ]
        }
        coordinator = SimpleCoordinator(agent_id="coord_1", config=config, env_id="test_env")

        # Set up broker for all agents
        coordinator.set_message_broker(broker)
        for agent in coordinator.subordinates.values():
            agent.set_message_broker(broker)

        # Setup channels
        broker.create_channel("coord_1")
        broker.create_channel("device_1")
        broker.create_channel("device_2")

        # Coordinator sends messages to subordinates
        coordinator.send_message({"action": 0.5}, recipient_id="device_1")
        coordinator.send_message({"action": -0.3}, recipient_id="device_2")

        # Subordinates receive messages
        msg_1 = coordinator.subordinates["device_1"].receive_messages()
        msg_2 = coordinator.subordinates["device_2"].receive_messages()

        assert len(msg_1) == 1
        assert msg_1[0]["action"] == 0.5
        assert len(msg_2) == 1
        assert msg_2[0]["action"] == -0.3

    def test_message_with_env_id_filtering(self):
        """Test that messages are filtered by env_id."""
        broker = InMemoryBroker()
        broker.create_channel("device_1")

        # Publish messages for different environments
        msg1 = Message(
            env_id="env_1",
            sender_id="coord_1",
            recipient_id="device_1",
            timestamp=1.0,
            message_type=MessageType.INFO,
            payload={"env": "1"},
        )
        msg2 = Message(
            env_id="env_2",
            sender_id="coord_1",
            recipient_id="device_1",
            timestamp=1.0,
            message_type=MessageType.INFO,
            payload={"env": "2"},
        )
        broker.publish("device_1", msg1)
        broker.publish("device_1", msg2)

        # Consume only env_1 messages
        messages_env1 = broker.consume("device_1", "device_1", "env_1")
        assert len(messages_env1) == 1
        assert messages_env1[0].payload["env"] == "1"

        # Consume only env_2 messages
        messages_env2 = broker.consume("device_1", "device_1", "env_2")
        assert len(messages_env2) == 1
        assert messages_env2[0].payload["env"] == "2"


class TestActionMessageCommunication:
    """Test action distribution via message broker."""

    def test_action_messages_via_broker(self):
        """Test sending action commands via message broker."""
        broker = InMemoryBroker()
        broker.create_channel("device_1")

        # Simulate coordinator sending action message
        action_message = Message(
            env_id="test_env",
            sender_id="coord_1",
            recipient_id="device_1",
            message_type=MessageType.ACTION,
            payload={"action": [0.75]},
            timestamp=1.0,
        )
        broker.publish("device_1", action_message)

        # Consume action messages
        messages = broker.consume("device_1", "device_1", "test_env")

        assert len(messages) == 1
        assert messages[0].message_type == MessageType.ACTION
        assert messages[0].payload["action"] == [0.75]


class TestBroadcastCommunication:
    """Test broadcast messaging patterns."""

    def test_broadcast_to_channel(self):
        """Test broadcasting messages to a channel."""
        broker = InMemoryBroker()
        broker.create_channel("broadcast_all")

        # Broadcast message
        broadcast_message = Message(
            env_id="test_env",
            sender_id="system",
            recipient_id="broadcast",
            message_type=MessageType.BROADCAST,
            payload={"alert": "grid_instability", "severity": "high"},
            timestamp=1.0,
        )
        broker.publish("broadcast_all", broadcast_message)

        # Consume from broadcast channel
        messages = broker.consume("broadcast_all", "broadcast", "test_env")
        assert len(messages) == 1
        assert messages[0].payload["alert"] == "grid_instability"


class TestMessageOrdering:
    """Test message ordering and timing."""

    def test_messages_received_in_order(self):
        """Test that messages are received in correct order."""
        broker = InMemoryBroker()
        broker.create_channel("device_1")

        # Send multiple messages
        for i in range(5):
            broker.publish(
                "device_1",
                Message(
                    env_id="test_env",
                    sender_id="coord_1",
                    recipient_id="device_1",
                    message_type=MessageType.INFO,
                    payload={"sequence": i},
                    timestamp=float(i),
                )
            )

        # Receive all messages
        messages = broker.consume("device_1", "device_1", "test_env")

        assert len(messages) == 5
        for i, msg in enumerate(messages):
            assert msg.payload["sequence"] == i

    def test_message_queue_cleared_after_consume(self):
        """Test that message queue is cleared after consuming."""
        broker = InMemoryBroker()
        broker.create_channel("device_1")

        # Send message
        broker.publish(
            "device_1",
            Message(
                env_id="test_env",
                sender_id="coord_1",
                recipient_id="device_1",
                message_type=MessageType.INFO,
                payload={"test": True},
                timestamp=1.0,
            )
        )

        # First consume gets the message
        messages_1 = broker.consume("device_1", "device_1", "test_env")
        assert len(messages_1) == 1

        # Second consume is empty (clear=True by default)
        messages_2 = broker.consume("device_1", "device_1", "test_env")
        assert len(messages_2) == 0


class TestEnvironmentIsolation:
    """Test environment isolation in message broker."""

    def test_environments_isolated(self):
        """Test that messages are isolated by environment."""
        broker = InMemoryBroker()
        broker.create_channel("shared_channel")

        # Messages for different environments
        broker.publish(
            "shared_channel",
            Message(
                env_id="env_1",
                sender_id="agent",
                recipient_id="target",
                message_type=MessageType.INFO,
                payload={"from": "env_1"},
                timestamp=1.0,
            )
        )
        broker.publish(
            "shared_channel",
            Message(
                env_id="env_2",
                sender_id="agent",
                recipient_id="target",
                message_type=MessageType.INFO,
                payload={"from": "env_2"},
                timestamp=1.0,
            )
        )

        # Each environment only sees its own messages
        msgs_1 = broker.consume("shared_channel", "target", "env_1")
        msgs_2 = broker.consume("shared_channel", "target", "env_2")

        assert len(msgs_1) == 1
        assert msgs_1[0].payload["from"] == "env_1"
        assert len(msgs_2) == 1
        assert msgs_2[0].payload["from"] == "env_2"


class TestErrorHandling:
    """Test error handling in message communication."""

    def test_send_without_broker_raises(self):
        """Test that sending without broker raises error."""
        agent = SimpleFieldAgent(agent_id="device_1")

        with pytest.raises(RuntimeError, match="no message broker"):
            agent.send_message({"test": True}, recipient_id="device_2")

    def test_receive_without_broker_returns_empty(self):
        """Test that receive without broker returns empty list."""
        agent = SimpleFieldAgent(agent_id="device_1")

        messages = agent.receive_messages()
        assert messages == []


class TestBrokerClose:
    """Test broker cleanup."""

    def test_close_clears_all_data(self):
        """Test that close clears all messages and channels."""
        broker = InMemoryBroker()
        broker.create_channel("test_channel")
        broker.publish(
            "test_channel",
            Message(
                env_id="test",
                sender_id="a",
                recipient_id="b",
                message_type=MessageType.INFO,
                payload={},
                timestamp=1.0,
            )
        )

        broker.close()

        assert len(broker.channels) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

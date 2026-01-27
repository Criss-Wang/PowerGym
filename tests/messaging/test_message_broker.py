"""Comprehensive tests for MessageBroker base class and InMemoryBroker implementation.

Tests cover:
1. Message structure and types
2. Channel management
3. Publish/consume operations
4. Subscriber callbacks
5. Multi-environment isolation
6. Thread safety
7. Edge cases and error handling
"""

import pytest
import threading
import time
from typing import List

from heron.messaging.base import (
    Message,
    MessageType,
    MessageBroker,
    ChannelManager
)
from heron.messaging.memory import InMemoryBroker


# =============================================================================
# Message Tests
# =============================================================================

class TestMessage:
    """Test Message dataclass."""

    def test_message_initialization(self):
        """Test message creation with all fields."""
        msg = Message(
            env_id="env_0",
            sender_id="agent_1",
            recipient_id="agent_2",
            timestamp=1.5,
            message_type=MessageType.ACTION,
            payload={"value": 42}
        )

        assert msg.env_id == "env_0"
        assert msg.sender_id == "agent_1"
        assert msg.recipient_id == "agent_2"
        assert msg.timestamp == 1.5
        assert msg.message_type == MessageType.ACTION
        assert msg.payload == {"value": 42}

    def test_message_types(self):
        """Test all message types."""
        types = [
            MessageType.ACTION,
            MessageType.INFO,
            MessageType.BROADCAST,
            MessageType.STATE_UPDATE,
            MessageType.RESULT,
            MessageType.CUSTOM
        ]

        for msg_type in types:
            msg = Message(
                env_id="env_0",
                sender_id="sender",
                recipient_id="recipient",
                timestamp=0.0,
                message_type=msg_type,
                payload={}
            )
            assert msg.message_type == msg_type

    def test_message_with_complex_payload(self):
        """Test message with nested payload."""
        payload = {
            "action": [1.0, 2.0, 3.0],
            "metadata": {
                "cost": 100.5,
                "safety": 0.0
            },
            "status": "ok"
        }

        msg = Message(
            env_id="env_0",
            sender_id="agent",
            recipient_id="controller",
            timestamp=2.0,
            message_type=MessageType.INFO,
            payload=payload
        )

        assert msg.payload["action"] == [1.0, 2.0, 3.0]
        assert msg.payload["metadata"]["cost"] == 100.5
        assert msg.payload["status"] == "ok"


# =============================================================================
# ChannelManager Tests
# =============================================================================

class TestChannelManager:
    """Test channel naming conventions."""

    def test_action_channel(self):
        """Test action channel naming."""
        channel = ChannelManager.action_channel("parent", "child", "env_0")
        assert channel == "env_env_0__action__parent_to_child"

    def test_info_channel(self):
        """Test info channel naming."""
        channel = ChannelManager.info_channel("child", "parent", "env_0")
        assert channel == "env_env_0__info__child_to_parent"

    def test_broadcast_channel(self):
        """Test broadcast channel naming."""
        channel = ChannelManager.broadcast_channel("agent_1", "env_0")
        assert channel == "env_env_0__broadcast__agent_1"

    def test_state_update_channel(self):
        """Test state update channel naming."""
        channel = ChannelManager.state_update_channel("env_0")
        assert channel == "env_env_0__state_updates"

    def test_custom_channel(self):
        """Test custom channel naming for domain-specific extensions."""
        channel = ChannelManager.custom_channel("power_flow", "env_0", "agent_1")
        assert channel == "env_env_0__power_flow__agent_1"

        # Test with different custom type
        channel2 = ChannelManager.custom_channel("voltage_update", "env_1", "sensor_1")
        assert channel2 == "env_env_1__voltage_update__sensor_1"

    def test_agent_channels_no_upstream(self):
        """Test agent channels with no parent."""
        channels = ChannelManager.agent_channels(
            agent_id="root",
            upstream_id=None,
            subordinate_ids=["child1", "child2"],
            env_id="env_0"
        )

        # Should subscribe to info from children
        assert len(channels['subscribe']) == 2
        assert "env_env_0__info__child1_to_root" in channels['subscribe']
        assert "env_env_0__info__child2_to_root" in channels['subscribe']

        # Should publish actions to children
        assert len(channels['publish']) == 2
        assert "env_env_0__action__root_to_child1" in channels['publish']
        assert "env_env_0__action__root_to_child2" in channels['publish']

    def test_agent_channels_with_upstream(self):
        """Test agent channels with parent."""
        channels = ChannelManager.agent_channels(
            agent_id="middle",
            upstream_id="parent",
            subordinate_ids=["child1"],
            env_id="env_0"
        )

        # Should subscribe to action from parent + info from child
        assert len(channels['subscribe']) == 2
        assert "env_env_0__action__parent_to_middle" in channels['subscribe']
        assert "env_env_0__info__child1_to_middle" in channels['subscribe']

        # Should publish info to parent + action to child
        assert len(channels['publish']) == 2
        assert "env_env_0__info__middle_to_parent" in channels['publish']
        assert "env_env_0__action__middle_to_child1" in channels['publish']

    def test_agent_channels_leaf_node(self):
        """Test channels for leaf agent (no subordinates)."""
        channels = ChannelManager.agent_channels(
            agent_id="leaf",
            upstream_id="parent",
            subordinate_ids=[],
            env_id="env_0"
        )

        # Should only subscribe to parent actions
        assert len(channels['subscribe']) == 1
        assert "env_env_0__action__parent_to_leaf" in channels['subscribe']

        # Should only publish info to parent
        assert len(channels['publish']) == 1
        assert "env_env_0__info__leaf_to_parent" in channels['publish']


# =============================================================================
# InMemoryBroker Core Tests
# =============================================================================

class TestInMemoryBrokerBasics:
    """Test basic InMemoryBroker functionality."""

    def test_broker_initialization(self):
        """Test broker initializes with empty state."""
        broker = InMemoryBroker()
        assert len(broker.channels) == 0
        assert len(broker.subscribers) == 0

    def test_create_channel(self):
        """Test channel creation."""
        broker = InMemoryBroker()
        broker.create_channel("test_channel")

        assert "test_channel" in broker.channels
        assert broker.channels["test_channel"] == []

    def test_publish_single_message(self):
        """Test publishing a single message."""
        broker = InMemoryBroker()
        channel = "test_channel"

        msg = Message(
            env_id="env_0",
            sender_id="sender",
            recipient_id="recipient",
            timestamp=1.0,
            message_type=MessageType.INFO,
            payload={"data": "test"}
        )

        broker.publish(channel, msg)

        assert len(broker.channels[channel]) == 1
        assert broker.channels[channel][0] == msg

    def test_publish_multiple_messages(self):
        """Test publishing multiple messages to same channel."""
        broker = InMemoryBroker()
        channel = "test_channel"

        for i in range(5):
            msg = Message(
                env_id="env_0",
                sender_id=f"sender_{i}",
                recipient_id="recipient",
                timestamp=float(i),
                message_type=MessageType.INFO,
                payload={"index": i}
            )
            broker.publish(channel, msg)

        assert len(broker.channels[channel]) == 5

    def test_consume_messages(self):
        """Test consuming messages with filtering."""
        broker = InMemoryBroker()
        channel = "test_channel"

        # Publish messages for different recipients and environments
        msg1 = Message("env_0", "sender", "agent_1", 1.0, MessageType.INFO, {"id": 1})
        msg2 = Message("env_0", "sender", "agent_2", 2.0, MessageType.INFO, {"id": 2})
        msg3 = Message("env_1", "sender", "agent_1", 3.0, MessageType.INFO, {"id": 3})

        broker.publish(channel, msg1)
        broker.publish(channel, msg2)
        broker.publish(channel, msg3)

        # Consume for agent_1 in env_0
        messages = broker.consume(channel, "agent_1", "env_0", clear=True)

        assert len(messages) == 1
        assert messages[0].payload["id"] == 1
        assert messages[0].recipient_id == "agent_1"
        assert messages[0].env_id == "env_0"

    def test_consume_without_clear(self):
        """Test consuming messages without clearing them."""
        broker = InMemoryBroker()
        channel = "test_channel"

        msg = Message("env_0", "sender", "recipient", 1.0, MessageType.INFO, {"data": "test"})
        broker.publish(channel, msg)

        # Consume without clearing
        messages1 = broker.consume(channel, "recipient", "env_0", clear=False)
        assert len(messages1) == 1

        # Should still be there
        messages2 = broker.consume(channel, "recipient", "env_0", clear=False)
        assert len(messages2) == 1

    def test_consume_with_clear(self):
        """Test consuming messages with clearing."""
        broker = InMemoryBroker()
        channel = "test_channel"

        msg = Message("env_0", "sender", "recipient", 1.0, MessageType.INFO, {"data": "test"})
        broker.publish(channel, msg)

        # Consume with clearing
        messages1 = broker.consume(channel, "recipient", "env_0", clear=True)
        assert len(messages1) == 1

        # Should be gone
        messages2 = broker.consume(channel, "recipient", "env_0", clear=True)
        assert len(messages2) == 0

    def test_consume_empty_channel(self):
        """Test consuming from non-existent channel."""
        broker = InMemoryBroker()
        messages = broker.consume("nonexistent", "agent", "env_0", clear=True)
        assert messages == []


# =============================================================================
# InMemoryBroker Subscribe/Callback Tests
# =============================================================================

class TestInMemoryBrokerSubscribers:
    """Test subscriber callback functionality."""

    def test_subscribe_callback_invoked(self):
        """Test subscriber callback is invoked on publish."""
        broker = InMemoryBroker()
        channel = "test_channel"
        received_messages = []

        def callback(msg: Message):
            received_messages.append(msg)

        broker.subscribe(channel, callback)

        msg = Message("env_0", "sender", "recipient", 1.0, MessageType.INFO, {"data": "test"})
        broker.publish(channel, msg)

        assert len(received_messages) == 1
        assert received_messages[0] == msg

    def test_multiple_subscribers(self):
        """Test multiple subscribers receive messages."""
        broker = InMemoryBroker()
        channel = "test_channel"

        received_1 = []
        received_2 = []

        broker.subscribe(channel, lambda msg: received_1.append(msg))
        broker.subscribe(channel, lambda msg: received_2.append(msg))

        msg = Message("env_0", "sender", "recipient", 1.0, MessageType.INFO, {"data": "test"})
        broker.publish(channel, msg)

        assert len(received_1) == 1
        assert len(received_2) == 1
        assert received_1[0] == msg
        assert received_2[0] == msg

    def test_subscriber_error_handling(self):
        """Test broker handles subscriber errors gracefully."""
        broker = InMemoryBroker()
        channel = "test_channel"
        success_calls = []

        def failing_callback(msg: Message):
            raise ValueError("Subscriber error")

        def success_callback(msg: Message):
            success_calls.append(msg)

        broker.subscribe(channel, failing_callback)
        broker.subscribe(channel, success_callback)

        msg = Message("env_0", "sender", "recipient", 1.0, MessageType.INFO, {"data": "test"})

        # Should not raise exception, but log error
        broker.publish(channel, msg)

        # Success callback should still be invoked
        assert len(success_calls) == 1


# =============================================================================
# Multi-Environment Isolation Tests
# =============================================================================

class TestMultiEnvironmentIsolation:
    """Test environment isolation in broker."""

    def test_clear_environment(self):
        """Test clearing messages for specific environment."""
        broker = InMemoryBroker()

        # Create channels for multiple environments
        channel1 = "env_env_0__test"
        channel2 = "env_env_1__test"

        msg1 = Message("env_0", "s", "r", 1.0, MessageType.INFO, {"env": 0})
        msg2 = Message("env_1", "s", "r", 1.0, MessageType.INFO, {"env": 1})

        broker.publish(channel1, msg1)
        broker.publish(channel2, msg2)

        # Clear env_0
        broker.clear_environment("env_0")

        # env_0 messages should be gone
        assert len(broker.channels[channel1]) == 0

        # env_1 messages should remain
        assert len(broker.channels[channel2]) == 1

    def test_get_environment_channels(self):
        """Test getting all channels for environment."""
        broker = InMemoryBroker()

        # Create channels for different environments
        broker.create_channel("env_env_0__action")
        broker.create_channel("env_env_0__info")
        broker.create_channel("env_env_1__action")

        env_0_channels = broker.get_environment_channels("env_0")

        assert len(env_0_channels) == 2
        assert "env_env_0__action" in env_0_channels
        assert "env_env_0__info" in env_0_channels
        assert "env_env_1__action" not in env_0_channels

    def test_message_filtering_by_environment(self):
        """Test consume filters by environment."""
        broker = InMemoryBroker()
        channel = "test_channel"

        # Publish messages for different environments
        msg1 = Message("env_0", "s", "agent_1", 1.0, MessageType.INFO, {"env": 0})
        msg2 = Message("env_1", "s", "agent_1", 2.0, MessageType.INFO, {"env": 1})
        msg3 = Message("env_0", "s", "agent_1", 3.0, MessageType.INFO, {"env": 0})

        broker.publish(channel, msg1)
        broker.publish(channel, msg2)
        broker.publish(channel, msg3)

        # Consume for env_0 only
        messages = broker.consume(channel, "agent_1", "env_0", clear=True)

        assert len(messages) == 2
        assert all(msg.env_id == "env_0" for msg in messages)

    def test_message_filtering_by_recipient(self):
        """Test consume filters by recipient."""
        broker = InMemoryBroker()
        channel = "test_channel"

        # Publish messages for different recipients
        msg1 = Message("env_0", "s", "agent_1", 1.0, MessageType.INFO, {"to": 1})
        msg2 = Message("env_0", "s", "agent_2", 2.0, MessageType.INFO, {"to": 2})
        msg3 = Message("env_0", "s", "agent_1", 3.0, MessageType.INFO, {"to": 1})

        broker.publish(channel, msg1)
        broker.publish(channel, msg2)
        broker.publish(channel, msg3)

        # Consume for agent_1 only
        messages = broker.consume(channel, "agent_1", "env_0", clear=True)

        assert len(messages) == 2
        assert all(msg.recipient_id == "agent_1" for msg in messages)


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread safety of InMemoryBroker."""

    def test_concurrent_publishes(self):
        """Test concurrent publishing from multiple threads."""
        broker = InMemoryBroker()
        channel = "test_channel"
        num_threads = 10
        messages_per_thread = 100

        def publish_messages(thread_id: int):
            for i in range(messages_per_thread):
                msg = Message(
                    env_id="env_0",
                    sender_id=f"thread_{thread_id}",
                    recipient_id="receiver",
                    timestamp=float(i),
                    message_type=MessageType.INFO,
                    payload={"thread": thread_id, "index": i}
                )
                broker.publish(channel, msg)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=publish_messages, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have all messages
        assert len(broker.channels[channel]) == num_threads * messages_per_thread

    def test_concurrent_consume(self):
        """Test concurrent consuming from multiple threads."""
        broker = InMemoryBroker()
        channel = "test_channel"

        # Publish messages for different recipients
        for i in range(100):
            msg = Message(
                env_id="env_0",
                sender_id="sender",
                recipient_id=f"agent_{i % 10}",
                timestamp=float(i),
                message_type=MessageType.INFO,
                payload={"index": i}
            )
            broker.publish(channel, msg)

        consumed_counts = {}

        def consume_messages(agent_id: str):
            messages = broker.consume(channel, agent_id, "env_0", clear=True)
            consumed_counts[agent_id] = len(messages)

        threads = []
        for i in range(10):
            agent_id = f"agent_{i}"
            t = threading.Thread(target=consume_messages, args=(agent_id,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each agent should get their 10 messages
        assert sum(consumed_counts.values()) == 100
        for count in consumed_counts.values():
            assert count == 10


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_close_broker(self):
        """Test closing broker clears all data."""
        broker = InMemoryBroker()
        channel = "test_channel"

        msg = Message("env_0", "s", "r", 1.0, MessageType.INFO, {"data": "test"})
        broker.publish(channel, msg)
        broker.subscribe(channel, lambda msg: None)

        broker.close()

        assert len(broker.channels) == 0
        assert len(broker.subscribers) == 0

    def test_repr(self):
        """Test string representation."""
        broker = InMemoryBroker()

        # Empty broker
        repr_str = repr(broker)
        assert "InMemoryBroker" in repr_str
        assert "channels=0" in repr_str
        assert "messages=0" in repr_str

        # After publishing
        msg = Message("env_0", "s", "r", 1.0, MessageType.INFO, {})
        broker.publish("ch1", msg)
        broker.publish("ch1", msg)
        broker.publish("ch2", msg)

        repr_str = repr(broker)
        assert "channels=2" in repr_str
        assert "messages=3" in repr_str

    def test_publish_to_nonexistent_channel_auto_creates(self):
        """Test publishing to non-existent channel auto-creates it."""
        broker = InMemoryBroker()
        msg = Message("env_0", "s", "r", 1.0, MessageType.INFO, {})

        broker.publish("new_channel", msg)

        assert "new_channel" in broker.channels
        assert len(broker.channels["new_channel"]) == 1

    def test_consume_partial_clear(self):
        """Test consuming clears only matching messages."""
        broker = InMemoryBroker()
        channel = "test_channel"

        # Publish messages for different recipients
        msg1 = Message("env_0", "s", "agent_1", 1.0, MessageType.INFO, {"id": 1})
        msg2 = Message("env_0", "s", "agent_2", 2.0, MessageType.INFO, {"id": 2})
        msg3 = Message("env_0", "s", "agent_1", 3.0, MessageType.INFO, {"id": 3})

        broker.publish(channel, msg1)
        broker.publish(channel, msg2)
        broker.publish(channel, msg3)

        # Consume for agent_1
        broker.consume(channel, "agent_1", "env_0", clear=True)

        # agent_2 message should still be there
        remaining = broker.channels[channel]
        assert len(remaining) == 1
        assert remaining[0].recipient_id == "agent_2"

    def test_empty_payload(self):
        """Test message with empty payload."""
        broker = InMemoryBroker()
        msg = Message("env_0", "s", "r", 1.0, MessageType.INFO, {})

        broker.publish("channel", msg)
        messages = broker.consume("channel", "r", "env_0")

        assert len(messages) == 1
        assert messages[0].payload == {}

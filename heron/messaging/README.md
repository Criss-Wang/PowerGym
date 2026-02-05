# HERON Messaging

This module provides pub/sub messaging infrastructure for agent communication in distributed multi-agent systems.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Message Broker                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         Channels                                 │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │    │
│  │  │   action     │  │    info      │  │     broadcast        │   │    │
│  │  │  channel     │  │   channel    │  │      channel         │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Publishers ──publish()──►  Channels  ──consume()──►  Subscribers       │
│                                        ──subscribe()──► Callbacks        │
└─────────────────────────────────────────────────────────────────────────┘

Agent A                                                         Agent B
┌──────────┐     publish(action_channel, msg)      ┌──────────┐
│Coordinator├─────────────────────────────────────►│  Field   │
│          │◄─────────────────────────────────────┤  Agent   │
└──────────┘     publish(info_channel, msg)        └──────────┘
```

## File Structure

```
heron/messaging/
├── __init__.py         # Public exports
├── base.py             # Abstract interfaces and utilities
└── in_memory_broker.py # In-memory implementation
```

## Components

### Message

Container for agent-to-agent communication.

```python
from heron.messaging import Message, MessageType

msg = Message(
    env_id="env_0",              # Environment ID (for vectorized envs)
    sender_id="coordinator_1",   # Sender agent ID
    recipient_id="field_1",      # Recipient agent ID
    timestamp=10.5,              # Message timestamp
    message_type=MessageType.ACTION,
    payload={"setpoint": 0.5}    # Arbitrary dict payload
)
```

### MessageType

Enumeration of standard message types.

| Type | Description |
|------|-------------|
| `ACTION` | Action commands from parent to child |
| `INFO` | Information from child to parent |
| `BROADCAST` | Broadcast to all listeners |
| `STATE_UPDATE` | State update notifications |
| `RESULT` | Generic result messages |
| `CUSTOM` | Domain-specific (use with MessageTypeRegistry) |

```python
from heron.messaging import MessageType, MessageTypeRegistry

# Register custom message types for your domain
MessageTypeRegistry.register("market_bid", "Trading bid message")
MessageTypeRegistry.register("sensor_reading", "Sensor data update")

# Check if a type is registered
if MessageTypeRegistry.is_registered("market_bid"):
    msg = Message(
        ...,
        message_type=MessageType.CUSTOM,
        payload={"custom_type": "market_bid", "price": 50.0}
    )
```

### MessageBroker (Abstract)

Abstract interface for message broker implementations.

```python
from heron.messaging import MessageBroker

class MyBroker(MessageBroker):
    def create_channel(self, channel_name: str) -> None:
        """Create a named channel."""
        ...

    def publish(self, channel: str, message: Message) -> None:
        """Publish message to channel."""
        ...

    def consume(
        self,
        channel: str,
        recipient_id: str,
        env_id: str,
        clear: bool = True
    ) -> List[Message]:
        """Consume messages for a specific recipient."""
        ...

    def subscribe(
        self,
        channel: str,
        callback: Callable[[Message], None]
    ) -> None:
        """Subscribe to channel with callback."""
        ...

    def clear_environment(self, env_id: str) -> None:
        """Clear all messages for an environment."""
        ...

    def get_environment_channels(self, env_id: str) -> List[str]:
        """Get all channels for an environment."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
```

### InMemoryBroker

Thread-safe in-memory implementation for single-process environments.

```python
from heron.messaging import InMemoryBroker, Message, MessageType

# Create broker
broker = InMemoryBroker()

# Publish a message
msg = Message(
    env_id="env_0",
    sender_id="coordinator",
    recipient_id="field_1",
    timestamp=0.0,
    message_type=MessageType.ACTION,
    payload={"power_setpoint": 100.0}
)
broker.publish("env_0__action__coordinator_to_field_1", msg)

# Consume messages
messages = broker.consume(
    channel="env_0__action__coordinator_to_field_1",
    recipient_id="field_1",
    env_id="env_0",
    clear=True  # Remove after consuming
)

# Subscribe for reactive handling
def on_message(msg: Message):
    print(f"Received: {msg.payload}")

broker.subscribe("env_0__broadcast__coordinator", on_message)

# Clean up
broker.clear_environment("env_0")  # Clear all env_0 messages
broker.close()  # Release resources
```

### ChannelManager

Utility for generating consistent channel names.

```python
from heron.messaging import ChannelManager

# Action channel: parent -> child
channel = ChannelManager.action_channel(
    upstream_id="coordinator_1",
    node_id="field_1",
    env_id="env_0"
)
# Returns: "env_env_0__action__coordinator_1_to_field_1"

# Info channel: child -> parent
channel = ChannelManager.info_channel(
    node_id="field_1",
    upstream_id="coordinator_1",
    env_id="env_0"
)
# Returns: "env_env_0__info__field_1_to_coordinator_1"

# Broadcast channel
channel = ChannelManager.broadcast_channel(
    agent_id="coordinator_1",
    env_id="env_0"
)
# Returns: "env_env_0__broadcast__coordinator_1"

# State update channel (environment-wide)
channel = ChannelManager.state_update_channel(env_id="env_0")
# Returns: "env_env_0__state_updates"

# Result channel
channel = ChannelManager.result_channel(env_id="env_0", agent_id="field_1")
# Returns: "env_env_0__results__field_1"

# Custom domain-specific channel
channel = ChannelManager.custom_channel(
    channel_type="market",
    env_id="env_0",
    agent_id="trader_1"
)
# Returns: "env_env_0__market__trader_1"
```

### ChannelRegistry

Registry for documenting custom channel types.

```python
from heron.messaging import ChannelRegistry

# Register custom channel types
ChannelRegistry.register("market", "P2P trading marketplace channel")
ChannelRegistry.register("telemetry", "Sensor telemetry data channel")

# Check registration
if ChannelRegistry.is_registered("market"):
    print("Market channels are supported")

# List all registered types
all_types = ChannelRegistry.get_all()
```

## Usage Patterns

### Pattern 1: Hierarchical Communication

```python
from heron.messaging import InMemoryBroker, Message, MessageType, ChannelManager

broker = InMemoryBroker()

# Coordinator sends action to field agent
def send_action(coordinator_id, field_id, action, env_id="default"):
    channel = ChannelManager.action_channel(coordinator_id, field_id, env_id)
    msg = Message(
        env_id=env_id,
        sender_id=coordinator_id,
        recipient_id=field_id,
        timestamp=current_time,
        message_type=MessageType.ACTION,
        payload={"action": action}
    )
    broker.publish(channel, msg)

# Field agent sends info to coordinator
def send_info(field_id, coordinator_id, info, env_id="default"):
    channel = ChannelManager.info_channel(field_id, coordinator_id, env_id)
    msg = Message(
        env_id=env_id,
        sender_id=field_id,
        recipient_id=coordinator_id,
        timestamp=current_time,
        message_type=MessageType.INFO,
        payload=info
    )
    broker.publish(channel, msg)

# Coordinator receives info from all subordinates
def receive_subordinate_info(coordinator_id, subordinate_ids, env_id="default"):
    all_info = {}
    for sub_id in subordinate_ids:
        channel = ChannelManager.info_channel(sub_id, coordinator_id, env_id)
        messages = broker.consume(channel, coordinator_id, env_id)
        if messages:
            all_info[sub_id] = [m.payload for m in messages]
    return all_info
```

### Pattern 2: Agent Integration

```python
from heron.agents import FieldAgent, CoordinatorAgent
from heron.messaging import InMemoryBroker

# Create broker and agents
broker = InMemoryBroker()

coordinator = CoordinatorAgent(agent_id="coord_1", env_id="env_0")
field_agents = [
    FieldAgent(agent_id=f"field_{i}", upstream_id="coord_1", env_id="env_0")
    for i in range(3)
]

# Configure message broker for all agents
coordinator.set_message_broker(broker)
for agent in field_agents:
    agent.set_message_broker(broker)

# Now agents can communicate via broker
# Coordinator sends action
coordinator.send_action_to_subordinate("field_0", action_value)

# Field agent receives action
actions = field_agents[0].receive_action_messages()
```

### Pattern 3: Vectorized Environments

```python
from heron.messaging import InMemoryBroker

# Single broker for multiple environments
broker = InMemoryBroker()

num_envs = 4
for env_idx in range(num_envs):
    env_id = f"env_{env_idx}"
    # Each environment has isolated channels
    # Publish/consume with env_id filtering

# Reset specific environment
broker.clear_environment("env_2")

# Get all channels for an environment
channels = broker.get_environment_channels("env_0")
```

### Pattern 4: Reactive Subscriptions

```python
from heron.messaging import InMemoryBroker, ChannelManager

broker = InMemoryBroker()

# Subscribe to broadcast channel
def handle_broadcast(msg):
    print(f"Broadcast from {msg.sender_id}: {msg.payload}")

broadcast_channel = ChannelManager.broadcast_channel("system", "env_0")
broker.subscribe(broadcast_channel, handle_broadcast)

# Now any publish to this channel triggers the callback
# (callbacks are called outside the lock to prevent deadlocks)
```

## Channel Naming Convention

```
env_{env_id}__{channel_type}__{details}

Examples:
- env_0__action__coordinator_1_to_field_1   # Action: parent -> child
- env_0__info__field_1_to_coordinator_1     # Info: child -> parent
- env_0__broadcast__coordinator_1            # Broadcast from agent
- env_0__state_updates                       # Environment state updates
- env_0__results__field_1                    # Results for specific agent
- env_0__market__trader_1                    # Custom domain channel
```

## Thread Safety

`InMemoryBroker` is thread-safe:
- All operations use a `threading.Lock`
- Subscriber callbacks are invoked outside the lock to prevent deadlocks
- Safe for multi-threaded training/inference

```python
import threading
from heron.messaging import InMemoryBroker

broker = InMemoryBroker()

# Safe to use from multiple threads
def publisher_thread():
    broker.publish(channel, msg)

def consumer_thread():
    messages = broker.consume(channel, recipient, env_id)
```

## API Reference

### Message

| Field | Type | Description |
|-------|------|-------------|
| `env_id` | `str` | Environment identifier |
| `sender_id` | `str` | Sender agent ID |
| `recipient_id` | `str` | Recipient agent ID |
| `timestamp` | `float` | Message timestamp |
| `message_type` | `MessageType` | Type of message |
| `payload` | `Dict[str, Any]` | Message content |

### MessageBroker

| Method | Description |
|--------|-------------|
| `create_channel(name)` | Create a named channel |
| `publish(channel, message)` | Publish message to channel |
| `consume(channel, recipient_id, env_id, clear=True)` | Get messages for recipient |
| `subscribe(channel, callback)` | Register callback for channel |
| `clear_environment(env_id)` | Clear all messages for environment |
| `get_environment_channels(env_id)` | List channels for environment |
| `close()` | Clean up resources |

### ChannelManager

| Method | Description |
|--------|-------------|
| `action_channel(upstream_id, node_id, env_id)` | Parent -> child actions |
| `info_channel(node_id, upstream_id, env_id)` | Child -> parent info |
| `broadcast_channel(agent_id, env_id)` | Agent broadcasts |
| `state_update_channel(env_id)` | Environment state updates |
| `result_channel(env_id, agent_id)` | Results for agent |
| `custom_channel(type, env_id, agent_id)` | Domain-specific channel |
| `agent_channels(agent_id, upstream_id, subordinate_ids, env_id)` | All channels for agent |

### InMemoryBroker

| Attribute | Description |
|-----------|-------------|
| `channels` | Dict mapping channel names to message lists |
| `subscribers` | Dict mapping channel names to callback lists |
| `lock` | Thread lock for synchronization |

## Extending with Custom Brokers

For distributed systems, implement `MessageBroker` with your backend:

```python
from heron.messaging import MessageBroker, Message
from typing import Callable, List

class RedisBroker(MessageBroker):
    """Redis-based message broker for distributed systems."""

    def __init__(self, redis_url: str):
        import redis
        self.client = redis.from_url(redis_url)

    def create_channel(self, channel_name: str) -> None:
        # Redis pub/sub channels are auto-created
        pass

    def publish(self, channel: str, message: Message) -> None:
        # Serialize and publish
        self.client.publish(channel, serialize(message))

    def consume(
        self,
        channel: str,
        recipient_id: str,
        env_id: str,
        clear: bool = True
    ) -> List[Message]:
        # Use Redis lists for persistent queues
        ...

    # Implement remaining methods...
```

# Message Broker

The message broker is the communication backbone in PowerGrid 2.0's distributed mode, enabling realistic distributed control simulation.

---

## Overview

### Purpose

The message broker decouples agents from the environment and from each other, enabling:
- **Realistic communication patterns** (mimicking SCADA systems)
- **Multi-process execution** (agents on different machines)
- **Production deployment** (replace in-memory with Kafka/RabbitMQ)

### Design Philosophy

```python
# Without broker (centralized)
agent.observe(environment.state)
environment.apply_action(agent.action)

# With broker (distributed)
broker.publish('env/state', environment.state)
state = broker.consume('env/state')
broker.publish('agent/action', action)
```

**Key insight**: Agents never directly access the environment or each other.

---

## Architecture

### Interface

```python
from abc import ABC, abstractmethod

class MessageBroker(ABC):
    @abstractmethod
    def publish(self, channel: str, message: Message):
        """Publish message to channel"""
        pass

    @abstractmethod
    def consume(self, channel: str, ...) -> List[Message]:
        """Consume messages from channel"""
        pass

    @abstractmethod
    def create_channel(self, channel: str):
        """Create new channel"""
        pass
```

### Message Structure

```python
@dataclass
class Message:
    env_id: str           # Environment identifier
    sender_id: str        # Sender agent ID
    recipient_id: str     # Recipient (or "broadcast")
    timestamp: float      # Message timestamp
    message_type: MessageType  # Action, info, state_update, etc.
    payload: Dict[str, Any]    # Arbitrary data
```

**Example**:

```python
message = Message(
    env_id='rollout_0',
    sender_id='MG1',
    recipient_id='environment',
    timestamp=0.5,
    message_type=MessageType.STATE_UPDATE,
    payload={'P_MW': 0.5, 'Q_MVAr': 0.1}
)
```

---

## Channel Management

### Channel Naming Convention

```python
class ChannelManager:
    @staticmethod
    def action_channel(env_id: str) -> str:
        return f"{env_id}/actions"

    @staticmethod
    def state_update_channel(env_id: str) -> str:
        return f"{env_id}/state_updates"

    @staticmethod
    def network_state_channel(env_id: str) -> str:
        return f"{env_id}/network_state"

    @staticmethod
    def device_action_channel(env_id: str, agent_id: str) -> str:
        return f"{env_id}/agent/{agent_id}/device_actions"
```

**Benefits**:
- Consistent naming across codebase
- Multi-environment isolation (via `env_id`)
- Easy to debug (human-readable channel names)

### Standard Channels

| Channel | Publisher | Consumer | Content |
|---------|-----------|----------|---------|
| `{env}/actions` | Environment | GridAgents | RL actions |
| `{env}/state_updates` | DeviceAgents | Environment | Device P, Q, SOC |
| `{env}/network_state` | Environment | GridAgents | Voltages, loading |
| `{env}/agent/{id}/device_actions` | GridAgent | DeviceAgents | Device setpoints |
| `{env}/p2p_trades` | Environment | GridAgents | Trading confirmations |

---

## Implementation: InMemoryBroker

### Overview

The `InMemoryBroker` uses Python dictionaries and lists for fast local simulation:

```python
class InMemoryBroker(MessageBroker):
    def __init__(self):
        self._channels: Dict[str, List[Message]] = {}

    def publish(self, channel: str, message: Message):
        if channel not in self._channels:
            self._channels[channel] = []
        self._channels[channel].append(message)

    def consume(self, channel: str, clear=True) -> List[Message]:
        messages = self._channels.get(channel, [])
        if clear:
            self._channels[channel] = []
        return messages
```

### Features

✅ **Fast**: O(1) publish, O(n) consume
✅ **Simple**: No external dependencies
✅ **Deterministic**: Perfect for testing
❌ **Single-process only**: Cannot distribute across machines

### Usage

```python
from powergrid.messaging import InMemoryBroker

broker = InMemoryBroker()

# Publish
broker.publish('test/channel', Message(...))

# Consume (destructive read)
messages = broker.consume('test/channel', clear=True)

# Peek (non-destructive)
messages = broker.consume('test/channel', clear=False)
```

---

## Implementation: KafkaBroker (Future)

### Design Sketch

```python
class KafkaBroker(MessageBroker):
    def __init__(self, bootstrap_servers: str):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda m: json.dumps(m).encode()
        )
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode())
        )

    def publish(self, channel: str, message: Message):
        self.producer.send(channel, value=message.to_dict())

    def consume(self, channel: str, timeout_ms=1000):
        self.consumer.subscribe([channel])
        records = self.consumer.poll(timeout_ms=timeout_ms)
        messages = []
        for topic_partition, records in records.items():
            for record in records:
                messages.append(Message.from_dict(record.value))
        return messages
```

### Benefits

✅ **Distributed**: Agents on different machines
✅ **Scalable**: Kafka handles millions of msgs/sec
✅ **Persistent**: Messages stored on disk
✅ **Production-ready**: Battle-tested in industry

### Deployment Architecture

```
┌─────────────┐         ┌─────────────┐
│  Machine 1  │         │  Machine 2  │
│ ┌─────────┐ │         │ ┌─────────┐ │
│ │GridAgent│ │         │ │GridAgent│ │
│ │  MG1    │ │         │ │  MG2    │ │
│ └────┬────┘ │         │ └────┬────┘ │
└──────┼──────┘         └──────┼──────┘
       │                       │
       └───────┬───────────────┘
               │
        ┌──────▼──────┐
        │    Kafka    │
        │   Cluster   │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Machine 0  │
        │ ┌─────────┐ │
        │ │  Env    │ │
        │ │PandaPower│ │
        │ └─────────┘ │
        └─────────────┘
```

---

## Message Flow Examples

### Example 1: Action Distribution

```python
# Environment publishes actions to all agents
actions = {'MG1': action_mg1, 'MG2': action_mg2}
channel = ChannelManager.action_channel('env_0')
broker.publish(channel, Message(
    env_id='env_0',
    sender_id='environment',
    recipient_id='broadcast',
    timestamp=timestep,
    message_type=MessageType.ACTION,
    payload=actions
))

# GridAgent MG1 consumes its action
messages = broker.consume(channel)
for msg in messages:
    if 'MG1' in msg.payload:
        my_action = msg.payload['MG1']
```

### Example 2: State Update Collection

```python
# Device publishes state update
channel = ChannelManager.state_update_channel('env_0')
broker.publish(channel, Message(
    env_id='env_0',
    sender_id='ESS1',
    recipient_id='environment',
    timestamp=timestep,
    message_type=MessageType.STATE_UPDATE,
    payload={
        'agent_id': 'ESS1',
        'device_type': 'storage',
        'P_MW': 0.5,
        'Q_MVAr': 0.1,
        'SOC': 0.75
    }
))

# Environment collects all state updates
messages = broker.consume(channel)
for msg in messages:
    device_id = msg.payload['agent_id']
    P = msg.payload['P_MW']
    Q = msg.payload['Q_MVAr']
    # Apply to network
```

### Example 3: Network State Broadcast

```python
# Environment publishes network state
channel = ChannelManager.network_state_channel('env_0')
broker.publish(channel, Message(
    env_id='env_0',
    sender_id='environment',
    recipient_id='broadcast',
    timestamp=timestep,
    message_type=MessageType.POWER_FLOW_RESULT,
    payload={
        'voltages': net.res_bus['vm_pu'].tolist(),
        'line_loading': net.res_line['loading_percent'].tolist(),
        'converged': converged
    }
))

# Agents observe network state
messages = broker.consume(channel)
network_state = messages[0].payload if messages else None
```

---

## Performance

### Message Overhead

**InMemoryBroker**:
- Publish: ~0.001 ms
- Consume: ~0.005 ms (for 100 messages)
- Total overhead: ~6% vs centralized mode

**KafkaBroker (estimated)**:
- Publish: ~1-5 ms (network latency)
- Consume: ~5-10 ms (network latency)
- Total overhead: ~20-30% vs in-memory

**Mitigation strategies**:
- Batch messages
- Async publish/consume
- Local caching

### Scalability

**Message volume** (per step):
- N microgrids, M devices each
- Actions: N messages (env → agents)
- State updates: N×M messages (devices → env)
- Network state: N messages (env → agents)
- **Total**: ~N×(M+2) messages/step

**Example**:
- 3 microgrids, 3 devices: ~15 msgs/step
- 100 microgrids, 5 devices: ~700 msgs/step
- Kafka throughput: 100K+ msgs/sec → supports 1000+ envs

---

## Testing and Debugging

### Message Inspection

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

broker = InMemoryBroker()
broker.publish('test', Message(...))

# Inspect all channels
print(broker._channels.keys())

# Inspect message count
print(len(broker._channels['test']))

# Peek messages without consuming
messages = broker.consume('test', clear=False)
for msg in messages:
    print(msg.payload)
```

### Message Replay

```python
# Record messages
recorded_messages = []
original_publish = broker.publish

def recording_publish(channel, message):
    recorded_messages.append((channel, message))
    original_publish(channel, message)

broker.publish = recording_publish

# ... run simulation ...

# Replay messages
for channel, message in recorded_messages:
    print(f"{channel}: {message.payload}")
```

---

## Best Practices

### ✅ Do's

- **Use ChannelManager** for consistent naming
- **Include env_id** in all messages (for multi-env isolation)
- **Set clear=True** when consuming (avoid memory leaks)
- **Validate payloads** before publishing
- **Log failed publishes** for debugging

### ❌ Don'ts

- **Don't hardcode** channel names
- **Don't publish large payloads** (>1MB) - use references instead
- **Don't forget to consume** - unconsumed messages accumulate
- **Don't assume ordering** - messages may arrive out of order
- **Don't block on consume** - use timeouts

---

## Migration Path

### Development → Production

1. **Develop with InMemoryBroker**: Fast iteration
2. **Test with InMemoryBroker + delays**: Add artificial latency
3. **Switch to KafkaBroker**: Same interface, different backend
4. **Deploy to cluster**: Agents on separate machines

**Code changes**: Minimal (just broker initialization)

```python
# Development
broker = InMemoryBroker()

# Production
broker = KafkaBroker(bootstrap_servers='kafka:9092')

# Same code works with both!
env = NetworkedGridEnv(config, message_broker=broker)
```

---

## Next Steps

- **Agents**: How agents use the broker in [Agents](agents.md)
- **Devices**: How devices publish updates in [Devices](devices.md)
- **Implementation**: See source code in `powergrid/messaging/`

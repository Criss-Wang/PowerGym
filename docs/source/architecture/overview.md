# Architecture Overview

PowerGrid 2.0's architecture is designed for **realistic distributed control** while maintaining **research flexibility**.

---

## System Layers

```
┌─────────────────────────────────────────────────────┐
│              RL Training Layer                       │
│         (RLlib, Stable-Baselines3, Custom)          │
└────────────────────┬────────────────────────────────┘
                     │ PettingZoo API
┌────────────────────▼────────────────────────────────┐
│         Environment Layer                            │
│      NetworkedGridEnv (Multi-Agent)                 │
│   - Manages agents and power flow                   │
│   - Computes rewards and observations               │
└────────────────────┬────────────────────────────────┘
                     │
              ┌──────▼──────┐
              │MessageBroker│ (Distributed mode only)
              └──────┬──────┘
         ┌───────────┼───────────┐
         │           │           │
┌────────▼───┐  ┌───▼────┐  ┌───▼────┐
│ GridAgent  │  │GridAgent│  │GridAgent│
│    MG1     │  │   MG2   │  │   MG3   │
└────────┬───┘  └────┬───┘  └────┬───┘
         │           │           │
    ┌────▼───┐  ┌───▼────┐  ┌───▼────┐
    │Devices │  │Devices │  │Devices │
    │ESS, DG │  │ESS, DG │  │ESS, DG │
    └────┬───┘  └────┬───┘  └────┬───┘
         │           │           │
         └───────────┼───────────┘
                     │
          ┌──────────▼──────────┐
          │   Physics Engine    │
          │  PandaPower AC Flow │
          └─────────────────────┘
```

---

## Core Design Principles

### 1. Separation of Concerns

**Environment**: Owns the physics (PandaPower network)
- Runs AC power flow
- Manages network topology
- Computes safety metrics

**Agents**: Own control logic
- Observe local state
- Make decisions
- Send commands to devices

**Devices**: Own device physics
- Execute dynamics (SOC updates, ramp rates)
- Respect physical limits
- Publish state updates

### 2. Message-Based Communication

In distributed mode, **all** communication is message-based:

```python
# Environment publishes actions
broker.publish('env/actions', Message({'MG1': action}))

# Agents consume and respond
action = broker.consume('env/actions')
broker.publish('agent/MG1/device_actions', Message({...}))

# Devices update and publish state
broker.publish('env/state_updates', Message({'P_MW': 0.5}))
```

**Benefit**: Drop-in replacement for real communication systems (Kafka, MQTT, etc.)

### 3. Dual-Mode Architecture

**Same codebase, different execution:**

```python
# Centralized: agents.step() called directly
for agent in agents:
    agent.step_centralized(obs, action)

# Distributed: agents communicate via messages
broker.publish('env/actions', actions)
for agent in agents:
    agent.step_distributed()  # Reads from broker
```

**Key insight**: Distributed mode enforces realistic constraints without changing agent code.

---

## Component Responsibilities

### Environment (`NetworkedGridEnv`)

**Responsibilities**:
- Initialize network and agents
- Publish actions to agents (distributed mode)
- Consume state updates from devices
- Run power flow simulation
- Compute rewards and observations
- Manage episode lifecycle

**Key Methods**:
```python
def reset(self) -> tuple[dict, dict]:
    """Initialize episode"""

def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
    """Execute one timestep"""

def _run_power_flow(self):
    """Run PandaPower AC power flow"""
```

### GridAgent

**Responsibilities**:
- Observe microgrid state
- Execute RL policy
- Coordinate subordinate devices
- Communicate with peer agents

**Key Methods**:
```python
def observe(self, global_state) -> Observation:
    """Create observation for RL policy"""

def step_centralized(self, obs, action):
    """Centralized execution (direct calls)"""

def step_distributed(self):
    """Distributed execution (message-based)"""
```

### DeviceAgent

**Responsibilities**:
- Wrap physical device
- Execute device dynamics
- Respond to coordination signals
- Publish state updates (distributed mode)

**Key Methods**:
```python
def step(self):
    """Update device state"""

def _publish_state_updates(self):
    """Publish P, Q to environment"""
```

### MessageBroker

**Responsibilities**:
- Route messages between components
- Support pub/sub pattern
- Handle channels and topics

**Key Methods**:
```python
def publish(self, channel: str, message: Message):
    """Publish message to channel"""

def consume(self, channel: str, ...) -> list[Message]:
    """Consume messages from channel"""
```

---

## Execution Flow

### Centralized Mode

```
1. Environment.step(actions)
2. For each agent:
   - agent.step_centralized(obs, action)
   - agent decomposes action → device commands
   - devices.step() (update state)
3. Environment._apply_state_to_network()
4. Environment._run_power_flow()
5. Environment._compute_rewards()
6. Return observations, rewards
```

**Characteristics**:
- Direct method calls (fast)
- Tight coupling
- Full observability

### Distributed Mode

```
1. Environment.step(actions)
2. Environment.publish('env/actions', actions)
3. For each agent:
   - agent.step_distributed()
   - agent.consume('env/actions')
   - agent.publish('agent/{id}/device_actions', ...)
4. For each device:
   - device.consume('agent/{id}/device_actions')
   - device.step()
   - device.publish('env/state_updates', {P, Q})
5. Environment.consume('env/state_updates')
6. Environment._apply_state_updates_to_network()
7. Environment._run_power_flow()
8. Environment.publish('env/network_state', {V, loading})
9. Agents.consume('env/network_state')
10. Environment._compute_rewards()
11. Return observations, rewards
```

**Characteristics**:
- Message-based (realistic)
- Loose coupling
- Limited observability

---

## Data Flow

### Observation Flow

```
PandaPower Network
    ↓ (voltages, line loading)
Environment
    ↓ (network state + device states)
GridAgent.observe()
    ↓ (observation dict/vector)
RL Policy
```

### Action Flow

```
RL Policy
    ↓ (action vector)
GridAgent.act()
    ↓ (device commands)
DeviceAgent.execute()
    ↓ (P, Q setpoints)
PandaPower Network
```

### Reward Flow

```
Environment
    ↓ (collect costs)
GridAgent.compute_cost()
    ↓ (sum device costs)
DeviceAgent.compute_cost()
    ↓
Environment._compute_rewards()
    ↓ (cost + penalties)
RL Policy (for training)
```

---

## State Management

### Environment State

```python
{
    'timestep': int,
    'network': pandapower.Network,
    'agents': dict[str, GridAgent],
    'converged': bool,
    'voltage_violations': float,
    'line_loading_violations': float,
}
```

### Agent State

```python
{
    'agent_id': str,
    'devices': list[DeviceAgent],
    'observation': Observation,
    'action': Action,
    'cost': float,
    'safety': float,
}
```

### Device State

```python
{
    'agent_id': str,
    'device_type': str,
    'P_MW': float,
    'Q_MVAr': float,
    'SOC': float,  # for storage
    'cost': float,
    'safety': float,
}
```

---

## Scalability

### Horizontal Scaling (More Microgrids)

- **Centralized**: Linear complexity O(N)
- **Distributed**: Constant per-agent complexity O(1)

Message volume scales linearly:
- 3 microgrids: ~15 messages/step
- 100 microgrids: ~500 messages/step
- Kafka handles 100K+ msgs/sec easily

### Vertical Scaling (More Devices per Microgrid)

- GridAgent complexity: O(M) where M = # devices
- Network simulation: O(B) where B = # buses
- Bottleneck: Power flow solver (typically fast for <1000 buses)

---

## Extension Points

### Add New Device Type

```python
from powergrid.devices import Device

class MyDevice(Device):
    def step(self):
        """Update device dynamics"""
        pass

    def compute_cost(self):
        """Calculate operational cost"""
        self.cost = ...

    def compute_safety(self, converged):
        """Calculate safety violations"""
        self.safety = ...
```

### Add New Protocol

```python
from powergrid.core.protocols import VerticalProtocol

class MyProtocol(VerticalProtocol):
    def coordinate(self, subordinate_obs, parent_action):
        """Custom coordination logic"""
        return {sub_id: signal for sub_id, ... }
```

### Add New Broker

```python
from powergrid.messaging import MessageBroker

class KafkaBroker(MessageBroker):
    def publish(self, channel, message):
        """Publish to Kafka topic"""
        self.producer.send(channel, message)

    def consume(self, channel, ...):
        """Consume from Kafka topic"""
        return self.consumer.poll(...)
```

---

## Performance Considerations

### Bottlenecks

1. **Power Flow**: ~1-5ms per solve (PandaPower)
2. **Message Passing**: ~0.1ms per message (in-memory)
3. **Device Updates**: ~0.1ms per device
4. **RL Policy**: Depends on model size

### Optimization Tips

- **Vectorize** device operations where possible
- **Batch** message publishing/consuming
- **Cache** network computations (topology, admittance matrix)
- **Parallelize** independent agent steps

---

## Next Steps

- **Message Broker**: Detailed design in [Message Broker](message_broker.md)
- **Agents**: Agent implementation in [Agents](agents.md)
- **Devices**: Device models in [Devices](devices.md)

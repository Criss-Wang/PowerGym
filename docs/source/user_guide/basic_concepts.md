# Basic Concepts

This page introduces the core concepts and terminology used in PowerGrid 2.0.

---

## Overview

PowerGrid 2.0 is a **multi-agent reinforcement learning environment** for power grid control. It simulates microgrids with distributed energy resources (DERs) and enables agents to learn coordination strategies.

---

## Key Components

### 1. Agents

**GridAgent**
- Represents a **microgrid controller**
- The primary RL-trainable agent
- Manages multiple devices within a microgrid
- Can trade energy with other microgrids

**DeviceAgent**
- Represents a **physical device** (generator, storage, solar panel)
- Managed by a GridAgent
- Responds to coordination signals (prices, setpoints)
- Not directly RL-trainable (follows local optimization or rules)

```
# Example hierarchy
GridAgent "MG1"
├── DeviceAgent "ESS1" (Battery Storage)
├── DeviceAgent "DG1"  (Diesel Generator)
└── DeviceAgent "PV1"  (Solar Panel)
```

---

### 2. Environment

**NetworkedGridEnv**
- PettingZoo `ParallelEnv` interface
- Manages multiple GridAgents
- Runs AC power flow simulation (PandaPower)
- Handles message-based communication (distributed mode)

```python
from powergrid.envs.multi_agent import MultiAgentMicrogrids

env = MultiAgentMicrogrids(config)
obs, info = env.reset()

# Standard PettingZoo interface
for agent_id in env.agents:
    action = policy(obs[agent_id])

obs, rewards, dones, truncated, infos = env.step(actions)
```

---

### 3. Devices

PowerGrid 2.0 supports various types of distributed energy resources:

| Device Type | Description | Control Variables |
|-------------|-------------|-------------------|
| **ESS** | Energy Storage System | Charge/discharge power, reactive power |
| **DG** | Distributed Generator | Active/reactive power output |
| **RES** | Renewable Energy Source | Reactive power (active is forecast-driven) |
| **Shunt** | Capacitor Bank | Reactive power compensation |
| **Transformer** | Voltage Regulator (OLTC) | Tap position |

---

### 4. Networks

PowerGrid 2.0 uses **PandaPower** for realistic AC power flow simulation:

- **IEEE 13-bus**: Standard distribution test feeder
- **IEEE 34-bus**: Larger distribution system
- **Custom networks**: Build your own with PandaPower

Each microgrid has its own network, and microgrids can be interconnected.

---

## Execution Modes

### Centralized Mode

- **Traditional MARL**: Agents have full observability
- **Direct access**: Agents can read/write network state directly
- **Use case**: Algorithm development and prototyping
- **Performance**: Faster training, simpler implementation

```yaml
centralized: true
```

### Distributed Mode

- **Realistic control**: Agents communicate via messages only
- **Limited observability**: Agents see only what's published
- **Use case**: Realistic validation before deployment
- **Performance**: Slight overhead (~6%), same final results

```yaml
centralized: false
message_broker: 'in_memory'
```

**Key Insight**: Develop algorithms in centralized mode, validate in distributed mode.

---

## Coordination Mechanisms

### Vertical Protocols (Parent → Child)

GridAgents coordinate their DeviceAgents using vertical protocols:

- **Price Signals**: Broadcast electricity price, devices optimize locally
- **Setpoints**: Directly command device power outputs
- **Custom**: Implement your own coordination logic

```python
from powergrid.agents.protocols import PriceSignalProtocol

protocol = PriceSignalProtocol(initial_price=50.0)
grid_agent = GridAgent(vertical_protocol=protocol, ...)
```

### Horizontal Protocols (Peer ↔ Peer)

GridAgents coordinate with each other using horizontal protocols:

- **P2P Trading**: Decentralized energy marketplace
- **Consensus**: Distributed agreement (e.g., frequency regulation)
- **Custom**: Implement your own peer coordination

```python
config = {
    'horizontal_protocol': 'p2p_trading',
    ...
}
```

---

## Message Broker

In distributed mode, all communication flows through a **message broker**:

```
Environment ←→ MessageBroker ←→ GridAgents ←→ DeviceAgents
```

**Benefits**:
- Realistic distributed control simulation
- Decouples agents from environment
- Ready for Kafka/RabbitMQ deployment

**Message Types**:
- **Actions**: Environment → Agents (RL actions)
- **State Updates**: Devices → Environment (P, Q, status)
- **Network State**: Environment → Agents (voltages, loading)
- **Coordination**: Agents ↔ Agents (trades, consensus)

---

## Observation and Action Spaces

### Observation Space

GridAgents observe:
- **Local**: Own devices' states (SOC, power output, limits)
- **Network**: Voltage, line loading, frequency (if available)
- **Coordination**: Messages from other agents

```python
obs = {
    'MG1': {
        'local': {...},      # Device states
        'network': {...},    # Network measurements
        'messages': [...]    # Coordination messages
    },
    'MG2': {...},
    ...
}
```

### Action Space

GridAgents output:
- **Continuous**: Device setpoints or prices
- **Discrete**: Discrete choices (e.g., trading decisions)
- **Mixed**: Combination via `Dict` space

```python
action_space = Dict({
    'continuous': Box(low=-1, high=1, shape=(n_devices,)),
    'discrete': Discrete(n_choices)
})
```

---

## Rewards

GridAgents receive rewards based on:
- **Operational Cost**: Minimize generation cost
- **Safety Violations**: Penalize voltage violations, overloading
- **Coordination**: Incentivize trading, consensus

```python
reward = -(cost + penalty * safety_violations) + coordination_bonus
```

Rewards can be:
- **Individual**: Each agent gets its own reward
- **Shared**: All agents share the same global reward

---

## Episode Structure

A typical training episode:

1. **Reset**: Initialize network, devices, agents
2. **Step Loop**: For each timestep (e.g., 15-minute intervals):
   - Agents observe state
   - Agents output actions
   - Environment applies actions
   - Devices update state
   - Power flow runs
   - Rewards computed
3. **Terminate**: After fixed episode length (e.g., 96 steps = 24 hours)

```python
env.reset()
for t in range(episode_length):
    actions = {aid: policy(obs[aid]) for aid in env.agents}
    obs, rewards, dones, truncated, infos = env.step(actions)
```

---

## Next Steps

- **Centralized vs Distributed**: Learn the differences in [Centralized vs Distributed Mode](centralized_vs_distributed.md)
- **Configuration**: Customize your environment in [Configuration Guide](configuration.md)
- **Hands-on**: Follow the [Getting Started Tutorial](../getting_started.md)

---

## Glossary

- **MARL**: Multi-Agent Reinforcement Learning
- **DER**: Distributed Energy Resource
- **ESS**: Energy Storage System
- **DG**: Distributed Generator
- **RES**: Renewable Energy Source
- **OLTC**: On-Load Tap Changer
- **SOC**: State of Charge (battery)
- **P/Q**: Active / Reactive Power
- **AC**: Alternating Current
- **PandaPower**: Open-source power system analysis tool
- **PettingZoo**: Multi-agent RL environment standard

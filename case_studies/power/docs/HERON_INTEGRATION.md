# HERON Framework Integration Guide

This document describes how the PowerGrid case study integrates with the HERON multi-agent framework.

---

## Table of Contents

- [Overview](#overview)
- [Agent Hierarchy Mapping](#agent-hierarchy-mapping)
- [Timing Parameters](#timing-parameters)
- [Execution Modes](#execution-modes)
- [Message Broker Integration](#message-broker-integration)
- [Feature System](#feature-system)
- [CTDE Workflow](#ctde-workflow)

---

## Overview

The PowerGrid case study demonstrates how to build a domain-specific multi-agent system on top of HERON. The key design principle is **extension over modification**: PowerGrid agents extend HERON's base agents, inheriting all framework capabilities while adding power-system-specific functionality.

```
HERON Framework                    PowerGrid Case Study
================                   ====================
Agent                        -->   (base for all)
  └── FieldAgent             -->   DeviceAgent
        └── (devices)        -->     └── Generator
                                     └── ESS
                                     └── Transformer
  └── CoordinatorAgent       -->   GridAgent
        └── (coordinators)   -->     └── PowerGridAgent
  └── ProxyAgent             -->   ProxyAgent (extended)
  └── SystemAgent            -->   (DSO coordination)
```

---

## Agent Hierarchy Mapping

### Level 1: Field Agents (Devices)

| HERON Class | PowerGrid Class | Purpose |
|-------------|-----------------|---------|
| `FieldAgent` | `DeviceAgent` | Base class for power devices |
| - | `Generator` | Dispatchable generation (solar, wind, DG) |
| - | `ESS` | Energy Storage Systems |
| - | `Transformer` | Transformers with tap changers |

**Key overrides in DeviceAgent:**
- `set_device_action()` - Define device-specific action space
- `set_device_state()` - Define device-specific state features
- `reset_device()` - Reset device to initial state
- `update_state()` - Apply physics/dynamics
- `update_cost_safety()` - Compute cost and safety metrics

### Level 2: Coordinator Agents (Grid Controllers)

| HERON Class | PowerGrid Class | Purpose |
|-------------|-----------------|---------|
| `CoordinatorAgent` | `GridAgent` | Base coordinator for power grids |
| - | `PowerGridAgent` | Full PandaPower integration |

**Key features in PowerGridAgent:**
- PandaPower network management
- Device aggregation and coordination
- Network state extraction
- Power flow integration

### Level 3: System Agents

| HERON Class | PowerGrid Class | Purpose |
|-------------|-----------------|---------|
| `SystemAgent` | (DSO Agent) | System-wide coordination |
| `ProxyAgent` | `ProxyAgent` | Network state distribution |

---

## Timing Parameters

All HERON agents support timing parameters for realistic simulation:

### Parameter Definitions

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `tick_interval` | Time between agent steps | Devices: 1s, Grids: 60s |
| `obs_delay` | Observation latency | 0.1-1.0s |
| `act_delay` | Action effect delay | 0.1-2.0s |
| `msg_delay` | Message delivery delay | 0.1-1.0s |

### Usage Example

```python
from powergrid.agents.generator import Generator
from powergrid.agents.power_grid_agent import PowerGridAgent

# Device with fast response
gen = Generator(
    agent_id="gen_1",
    device_config=gen_config,
    tick_interval=1.0,      # Step every second
    obs_delay=0.1,          # 100ms sensor latency
    act_delay=0.2,          # 200ms actuator delay
    msg_delay=0.05,         # 50ms communication
)

# Grid coordinator with slower control loop
grid = PowerGridAgent(
    net=pandapower_net,
    grid_config=grid_config,
    tick_interval=60.0,     # Step every minute
    obs_delay=1.0,          # 1s aggregation delay
    act_delay=2.0,          # 2s setpoint delay
    msg_delay=0.5,          # 500ms to devices
)
```

### Timing in Event-Driven Mode

In Option B (event-driven), the `EventScheduler` uses these parameters:

1. **tick_interval**: Determines when `AGENT_TICK` events fire
2. **obs_delay**: Agent observes state from `t - obs_delay`
3. **act_delay**: Actions scheduled at `t + act_delay`
4. **msg_delay**: Messages delivered at `t + msg_delay`

---

## Execution Modes

### Option A: Synchronous (Training)

All agents step together in `env.step()`. Used for CTDE training.

```python
# Environment in centralized mode
env_config['centralized'] = True
env = MultiAgentMicrogrids(env_config)

# All agents step synchronously
obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

**Characteristics:**
- Perfect synchronization
- Full observability for critic networks
- Shared rewards enable cooperation
- Fast simulation

### Option B: Event-Driven (Testing)

Agents tick independently via `EventScheduler`. Used for testing policy robustness.

```python
# Environment in distributed mode
env_config['centralized'] = False
env = MultiAgentMicrogrids(env_config)

# Configure message broker
env.configure_agents_for_distributed()

# Agents step with realistic timing
obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

**Characteristics:**
- Heterogeneous tick rates
- Communication delays
- Local observations only
- Realistic distributed system behavior

---

## Message Broker Integration

### InMemoryBroker

The HERON `InMemoryBroker` enables distributed communication:

```python
from heron.messaging.in_memory_broker import InMemoryBroker
from heron.messaging.base import Message, MessageType, ChannelManager

# Create broker
broker = InMemoryBroker()

# Create channel for agent
channel = ChannelManager.agent_channel("gen_1", "env_1")
broker.create_channel(channel)

# Publish action message
message = Message(
    env_id="env_1",
    sender_id="grid_agent",
    recipient_id="gen_1",
    timestamp=0.0,
    message_type=MessageType.ACTION,
    payload={"action": [0.5, 0.0]},
)
broker.publish(channel, message)

# Consume messages
messages = broker.consume(channel, recipient_id="gen_1", env_id="env_1")
```

### Distributed Mode Setup

```python
from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids

env = MultiAgentMicrogrids(env_config)

# This method:
# 1. Creates InMemoryBroker if needed
# 2. Calls configure_for_distributed() on all grid agents
# 3. Grid agents propagate to their devices
# 4. Creates communication channels
env.configure_agents_for_distributed()
```

---

## Feature System

PowerGrid extends HERON's feature system for power-domain state:

### HERON Features

| Feature | Purpose |
|---------|---------|
| `FeatureProvider` | Base class for state features |
| `Observation` | Agent observation container |
| `State` | Collection of features |

### PowerGrid Features

| Feature | Fields | Visibility |
|---------|--------|------------|
| `ElectricalBasePh` | P_MW, Q_MVAr, Vm_pu, Va_rad | owner, coordinator |
| `StorageBlock` | soc, capacity, throughput | owner |
| `PowerLimits` | p_min, p_max, q_min, q_max | owner, coordinator |
| `BusVoltages` | vm_pu[], va_deg[] | coordinator |
| `LineFlows` | p_mw[], q_mvar[], loading% | coordinator |
| `NetworkMetrics` | total_gen, total_load, losses | coordinator |

### Visibility Control

Features support visibility-based filtering:

```python
from powergrid.core.features.electrical import ElectricalBasePh

# Feature only visible to owner and coordinator
electrical = ElectricalBasePh(
    P_MW=1.0,
    Q_MVAr=0.5,
    visibility=["owner", "coordinator"],
)

# Check if agent can observe this feature
can_observe = electrical.is_observable_by(
    requestor_id="grid_agent",
    requestor_level=2,
    owner_id="gen_1",
    owner_level=1,
)
```

---

## CTDE Workflow

### 1. Training Phase (Centralized)

```python
from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.setups.loader import load_setup

# Load config for training
env_config = load_setup('ieee34_ieee13')
env_config['centralized'] = True
env_config['share_reward'] = True  # Collective optimization
env_config['train'] = True

env = MultiAgentMicrogrids(env_config)

# Train with RLlib MAPPO
# - Shared policy enables cooperation
# - Shared rewards align agent objectives
```

### 2. Testing Phase (Decentralized)

```python
# Reconfigure for testing
env_config['centralized'] = False
env_config['share_reward'] = False
env_config['train'] = False

test_env = MultiAgentMicrogrids(env_config)
test_env.configure_agents_for_distributed()

# Load trained policy and test
obs, info = test_env.reset()
while not done:
    actions = trained_policy.compute_actions(obs)
    obs, rewards, terminateds, truncateds, infos = test_env.step(actions)
```

### 3. Evaluation Metrics

```python
# Collective metrics for CTDE evaluation
metrics = env.get_collective_metrics()
print(f"Total Cost: {metrics['total_cost']}")
print(f"Cooperation Score: {metrics['cooperation_score']}")

# Power-grid specific metrics
pg_metrics = env.get_power_grid_metrics()
print(f"Voltage Violations: {pg_metrics['voltage_violations']}")
print(f"Line Overloads: {pg_metrics['line_overloads']}")
```

---

## Key Implementation Patterns

### 1. Extending FieldAgent

```python
from powergrid.agents.device_agent import DeviceAgent

class MyDevice(DeviceAgent):
    def set_device_action(self):
        # Define action space
        self.action.set_specs(dim_c=2, range=([0, 0], [1, 1]))

    def set_device_state(self):
        # Define state features
        self.state.features = [MyFeature(...)]

    def update_state(self, **kwargs):
        # Apply device dynamics
        pass
```

### 2. Extending CoordinatorAgent

```python
from powergrid.agents.power_grid_agent import GridAgent

class MyCoordinator(GridAgent):
    def _build_device_agents(self, configs, env_id, upstream_id):
        # Build subordinate devices
        devices = {}
        for config in configs:
            device = MyDevice(config=config)
            devices[device.agent_id] = device
        return devices
```

### 3. Creating Custom Environment

```python
from powergrid.envs.networked_grid_env import NetworkedGridEnv

class MyEnv(NetworkedGridEnv):
    def _build_agents(self):
        # Create agent hierarchy
        return {"agent_1": my_agent}

    def _build_net(self):
        # Create PandaPower network
        return pandapower_net

    def _reward_and_safety(self):
        # Compute rewards
        return rewards, safety
```

---

## Further Reading

- [HERON Core Documentation](../../../docs/)
- [PandaPower Documentation](https://pandapower.readthedocs.io/)
- [RLlib Multi-Agent Tutorial](https://docs.ray.io/en/latest/rllib/rllib-env.html)

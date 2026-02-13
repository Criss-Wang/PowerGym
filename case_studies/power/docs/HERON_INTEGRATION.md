# HERON Framework Integration Guide

This document describes how the PowerGrid case study integrates with the HERON multi-agent framework.

---

## Table of Contents

- [Overview](#overview)
- [Agent Hierarchy Mapping](#agent-hierarchy-mapping)
- [Core Components](#core-components)
- [Timing Configuration](#timing-configuration)
- [Execution Modes](#execution-modes)
- [Message Broker Integration](#message-broker-integration)
- [Feature System](#feature-system)
- [CTDE Workflow](#ctde-workflow)
- [Key Implementation Patterns](#key-implementation-patterns)

---

## Overview

The PowerGrid case study demonstrates how to build a domain-specific multi-agent system on top of HERON. The key design principle is **extension over modification**: PowerGrid agents extend HERON's base agents, inheriting all framework capabilities while adding power-system-specific functionality.

```
HERON Framework                    PowerGrid Case Study
================                   ====================
Agent (ABC)                  -->   (abstract base)
  │
  ├── FieldAgent (L1)        -->   DeviceAgent
  │     └── (devices)        -->     ├── Generator
  │                                  ├── ESS
  │                                  └── Transformer
  │
  ├── HierarchicalAgent      -->   (shared base for L2/L3)
  │     │
  │     ├── CoordinatorAgent (L2) --> GridAgent
  │     │     └── (coordinators)  -->   └── PowerGridAgent
  │     │
  │     └── SystemAgent (L3) -->   (DSO coordination)
  │
  └── ProxyAgent (L0)        -->   (state distribution)
```

### Key HERON Imports

```python
# Agents
from heron.agents import Agent, FieldAgent, CoordinatorAgent, SystemAgent, ProxyAgent

# Core
from heron.core import Action, Observation, FeatureProvider
from heron.core import FieldAgentState, CoordinatorAgentState, SystemAgentState
from heron.core import Policy, RandomPolicy

# Protocols
from heron.protocols import SetpointProtocol, PriceSignalProtocol
from heron.protocols import PeerToPeerTradingProtocol, ConsensusProtocol

# Messaging
from heron.messaging import Message, MessageType, InMemoryBroker, ChannelManager

# Scheduling
from heron.scheduling import EventScheduler, TickConfig, JitterType, EventType

# Environments
from heron.envs import HeronEnvCore, BaseEnv, MultiAgentEnv
from heron.envs import PettingZooParallelEnv, RLlibMultiAgentEnv
```

---

## Agent Hierarchy Mapping

### Level 0: Proxy Agent (State Distribution)

| HERON Class | Purpose | Key Methods |
|-------------|---------|-------------|
| `ProxyAgent` | State distribution with visibility filtering | `update_state()`, `get_state_for_agent()`, `get_state_at_time()` |

**Key Attributes:**
- `state_cache`: Current environment state
- `state_history`: Historical states for delayed observations
- `visibility_rules`: Dict mapping agent IDs to allowed state keys
- `registered_agents`: List of agents allowed to request state

### Level 1: Field Agents (Devices)

| HERON Class | PowerGrid Class | Purpose |
|-------------|-----------------|---------|
| `FieldAgent` | `DeviceAgent` | Base class for power devices |
| - | `Generator` | Dispatchable generation (solar, wind, DG) |
| - | `ESS` | Energy Storage Systems |
| - | `Transformer` | Transformers with tap changers |

**Key HERON methods to override in DeviceAgent:**
- `set_action()` - Define action space via `self.action.set_specs()`
- `set_state()` - Add features via `self.state.features.append()`
- `reset_agent()` - Custom reset logic
- `update_state()` - Handle environment feedback

**Key HERON attributes:**
- `state`: `FieldAgentState` - agent's local state with features
- `action`: `Action` - continuous/discrete action representation
- `policy`: Optional `Policy` - local decision-making
- `protocol`: Optional `Protocol` - coordination protocol
- `tick_config`: `TickConfig` - timing configuration

### Level 2: Coordinator Agents (Grid Controllers)

| HERON Class | PowerGrid Class | Purpose |
|-------------|-----------------|---------|
| `CoordinatorAgent` | `GridAgent` | Base coordinator for power grids |
| - | `PowerGridAgent` | Full PandaPower integration |

**Key HERON methods:**
- `_build_subordinates()` - Build device agents from config
- `coordinate_subordinates()` - Execute coordination via protocol
- `get_joint_observation_space()` - Construct joint obs space
- `get_joint_action_space()` - Construct joint action space

**Key HERON attributes:**
- `subordinates`: Dict of `FieldAgent` instances
- `state`: `CoordinatorAgentState` - aggregated state
- `protocol`: Optional `Protocol` - coordination strategy

### Level 3: System Agents

| HERON Class | PowerGrid Class | Purpose |
|-------------|-----------------|---------|
| `SystemAgent` | (DSO Agent) | System-wide coordination |

**Key HERON methods:**
- `set_state()` - Add system-level features
- `set_action()` - Configure system action space
- `_build_subordinates()` - Build coordinators from config
- `update_from_environment()` - Handle environment state
- `get_state_for_environment()` - Return actions/state to env

**Convenience:**
- Property alias: `coordinators` ↔ `subordinates` (more descriptive naming)

---

## Core Components

### Action (`heron.core.Action`)

Mixed continuous/discrete action representation:

```python
from heron.core import Action

action = Action()
action.set_specs(
    dim_c=2,                # Continuous dimensions
    dim_d=1,                # Discrete action heads
    ncats=[5],              # Categories per discrete head
    range=(np.array([0.0, -1.0]), np.array([10.0, 1.0]))
)

# Sample, set, normalize
action.sample(seed=42)
action.set_values(c=[0.5, 0.3], d=[2])
scaled = action.scale()     # Map to [-1, 1]
vec = action.vector()       # Flatten to array
```

### Observation (`heron.core.Observation`)

Structured observation container:

```python
from heron.core import Observation

obs = Observation(
    local={"voltage": 1.02, "power": np.array([100.0, 50.0])},
    global_info={"grid_frequency": 60.0},
    timestamp=10.5
)

vec = obs.vector()          # Flatten for RL
dict_obs = obs.to_dict()    # Serialize
obs = Observation.from_dict(dict_obs)  # Deserialize
```

---

## Timing Configuration

All HERON agents support timing via `TickConfig` for realistic simulation:

### TickConfig API

```python
from heron.scheduling import TickConfig, JitterType

# Deterministic timing (fast training)
config = TickConfig.deterministic(
    tick_interval=1.0,      # Time between agent steps
    obs_delay=0.0,          # Observation latency
    act_delay=0.0,          # Action effect delay
    msg_delay=0.0           # Message delivery delay
)

# Realistic timing with jitter (testing)
config = TickConfig.with_jitter(
    tick_interval=1.0,
    obs_delay=0.1,
    act_delay=0.2,
    msg_delay=0.05,
    jitter_type=JitterType.GAUSSIAN,  # or UNIFORM, NONE
    jitter_ratio=0.1,       # ±10% variance
    seed=42
)

# Get jittered values
next_tick = config.next_tick_interval()
obs_delay = config.next_obs_delay()
act_delay = config.next_act_delay()
msg_delay = config.next_msg_delay()
```

### Typical Values

| Parameter | Devices (L1) | Coordinators (L2) | System (L3) |
|-----------|--------------|-------------------|-------------|
| `tick_interval` | 1s | 60s | 300s |
| `obs_delay` | 0.05-0.1s | 0.5-1.0s | 1.0-2.0s |
| `act_delay` | 0.1-0.2s | 1.0-2.0s | 2.0-5.0s |
| `msg_delay` | 0.02-0.05s | 0.1-0.5s | 0.5-1.0s |

### Usage Example

```python
from powergrid.agents.generator import Generator
from powergrid.agents.power_grid_agent import PowerGridAgent
from heron.scheduling import TickConfig, JitterType

# Device with fast response
gen = Generator(
    agent_id="gen_1",
    device_config=gen_config,
    tick_config=TickConfig.with_jitter(
        tick_interval=1.0,
        obs_delay=0.1,
        act_delay=0.2,
        msg_delay=0.05,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1
    ),
)

# Grid coordinator with slower control loop
grid = PowerGridAgent(
    net=pandapower_net,
    grid_config=grid_config,
    tick_config=TickConfig.with_jitter(
        tick_interval=60.0,
        obs_delay=1.0,
        act_delay=2.0,
        msg_delay=0.5,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.2
    ),
)
```

### Timing in Event-Driven Mode

In Option B (event-driven), the `EventScheduler` uses `TickConfig`:

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

### Channel Naming Convention

HERON uses structured channel names:

```
env_{env_id}__{channel_type}__{details}

Examples:
- env_0__action__coordinator_1_to_field_1      # Parent → Child
- env_0__info__field_1_to_coordinator_1        # Child → Parent
- env_0__broadcast__coordinator_1              # Broadcast
- env_0__state_updates                         # Environment updates
- env_0__observation__field_1_to_coordinator_1 # Async observations
```

### ChannelManager API

```python
from heron.messaging import ChannelManager

# Generate channel names
action_channel = ChannelManager.action_channel(
    upstream_id="coordinator_1",
    node_id="field_1",
    env_id="env_0"
)

info_channel = ChannelManager.info_channel(
    node_id="field_1",
    upstream_id="coordinator_1",
    env_id="env_0"
)

broadcast_channel = ChannelManager.broadcast_channel(
    agent_id="coordinator_1",
    env_id="env_0"
)

observation_channel = ChannelManager.observation_channel(
    sender_id="field_1",
    recipient_id="coordinator_1",
    env_id="env_0"
)
```

### InMemoryBroker

The HERON `InMemoryBroker` enables distributed communication:

```python
from heron.messaging import InMemoryBroker, Message, MessageType, ChannelManager

# Create broker
broker = InMemoryBroker()

# Create channel for agent
channel = ChannelManager.action_channel("coordinator_1", "gen_1", "env_0")
broker.create_channel(channel)

# Publish action message
message = Message(
    env_id="env_0",
    sender_id="coordinator_1",
    recipient_id="gen_1",
    timestamp=0.0,
    message_type=MessageType.ACTION,
    payload={"action": [0.5, 0.0]},
)
broker.publish(channel, message)

# Consume messages
messages = broker.consume(channel, recipient_id="gen_1", env_id="env_0")
```

### MessageType Enum

- `ACTION`: Parent to child action commands
- `INFO`: Child to parent information
- `BROADCAST`: Broadcast to all subscribers
- `STATE_UPDATE`: State update notifications
- `RESULT`: Generic results
- `CUSTOM`: Domain-specific messages

### Agent Message Methods

```python
# Agents can send/receive messages directly
agent.set_message_broker(broker)

# Send message
agent.send_message(
    content={"measurements": [...], "status": "ok"},
    recipient_id=agent.upstream_id,
    message_type="INFO"
)

# Receive messages
messages = agent.receive_messages(sender_id="coordinator_1", clear=True)

# Send action to subordinate
agent.send_action_to_subordinate(recipient_id="field_1", action=action_array)

# Receive observations from subordinates (async mode)
obs_dict = agent.receive_observations_from_subordinates()
```

### Distributed Mode Setup

```python
from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids

env = MultiAgentMicrogrids(env_config)

# HeronEnvCore method that:
# 1. Creates InMemoryBroker if needed
# 2. Sets broker on all registered agents
# 3. Creates communication channels
env.configure_agents_for_distributed()
```

---

## Feature System

PowerGrid extends HERON's feature system for power-domain state:

### HERON Feature Classes

| Class | Purpose |
|-------|---------|
| `FeatureProvider` | Abstract base class for state features |
| `FieldAgentState` | L1 agent state container |
| `CoordinatorAgentState` | L2 agent state container |
| `SystemAgentState` | L3 agent state container |

### Visibility Options

Features declare who can observe them:

- `"public"`: Visible to all agents
- `"owner"`: Only the owning agent
- `"upper_level"`: Agents one level above
- `"system"`: System-level (L3) only

### Creating Custom Features

```python
from heron.core import FeatureProvider
import numpy as np

class MyFeature(FeatureProvider):
    visibility = ["owner", "upper_level"]  # Who can observe

    def __init__(self, param1=1.0):
        self.value1 = 0.0
        self.value2 = 0.0
        self.param1 = param1

    def vector(self) -> np.ndarray:
        """Convert to numpy array for observations"""
        return np.array([self.value1, self.value2], dtype=np.float32)

    def names(self) -> list:
        """Field names corresponding to vector"""
        return ["value1", "value2"]

    def to_dict(self) -> dict:
        """Serialize for communication/logging"""
        return {"value1": self.value1, "value2": self.value2, "param1": self.param1}

    @classmethod
    def from_dict(cls, d: dict) -> "MyFeature":
        """Deserialize from dictionary"""
        f = cls(d.get("param1", 1.0))
        f.value1 = d.get("value1", 0.0)
        f.value2 = d.get("value2", 0.0)
        return f

    def set_values(self, **kwargs) -> None:
        """Update values from keyword arguments"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def reset(self, **overrides) -> "MyFeature":
        """Reset to initial state"""
        self.value1 = 0.0
        self.value2 = 0.0
        return self
```

### PowerGrid Features

| Feature | Fields | Visibility |
|---------|--------|------------|
| `ElectricalBasePh` | P_MW, Q_MVAr, Vm_pu, Va_rad | owner, upper_level |
| `StorageBlock` | soc, capacity, throughput | owner, upper_level |
| `PowerLimits` | p_min, p_max, q_min, q_max | owner, upper_level |
| `BusVoltages` | vm_pu[], va_deg[] | system |
| `LineFlows` | p_mw[], q_mvar[], loading% | system |
| `NetworkMetrics` | total_gen, total_load, losses | system |

### Using State with Features

```python
from heron.core import FieldAgentState
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.storage import StorageBlock

# Create state and add features
state = FieldAgentState(owner_id="ess_1", owner_level=1)
state.features.append(ElectricalBasePh())
state.features.append(StorageBlock(soc=0.5, e_capacity_MWh=5.0))

# Update feature values
state.update_feature("ElectricalBasePh", p_MW=1.0, q_MVAr=0.5)

# Get full state vector
vec = state.vector()

# Get visibility-filtered observation
obs = state.observed_by(
    requestor_id="coordinator_1",
    requestor_level=2
)

# Serialize
dict_state = state.to_dict()
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
from heron.agents import FieldAgent
from heron.core import FieldAgentState
import numpy as np

class DeviceAgent(FieldAgent):
    """Base class for power devices."""

    def set_action(self):
        """Define action space - REQUIRED"""
        self.action.set_specs(
            dim_c=2,
            range=(np.array([0.0, -1.0]), np.array([10.0, 1.0]))
        )

    def set_state(self):
        """Add features to state - REQUIRED"""
        self.state = FieldAgentState(
            owner_id=self.agent_id,
            owner_level=self.level
        )
        self.state.features.append(ElectricalBasePh())

    def reset_agent(self, **kwargs):
        """Custom reset logic - OPTIONAL"""
        self.state.reset()

    def update_state(self, **kwargs):
        """Handle environment feedback - OPTIONAL"""
        if "voltage" in kwargs:
            self.state.update_feature("ElectricalBasePh", v_pu=kwargs["voltage"])
```

### 2. Extending CoordinatorAgent

```python
from heron.agents import CoordinatorAgent
from heron.core import CoordinatorAgentState

class GridAgent(CoordinatorAgent):
    """Base coordinator for power grids."""

    def _build_subordinates(self, configs, env_id, upstream_id):
        """Build device agents from config - REQUIRED"""
        devices = {}
        for config in configs:
            device = DeviceAgent(
                agent_id=config["id"],
                env_id=env_id,
                upstream_id=upstream_id
            )
            devices[device.agent_id] = device
        return devices

    def set_state(self):
        """Add coordinator-level features"""
        self.state = CoordinatorAgentState(
            owner_id=self.agent_id,
            owner_level=self.level
        )
        self.state.features.append(BusVoltages())
```

### 3. Extending SystemAgent

```python
from heron.agents import SystemAgent
from heron.core import SystemAgentState

class DSOAgent(SystemAgent):
    """Distribution System Operator."""

    def set_state(self):
        """Add system-level features"""
        self.state = SystemAgentState(
            owner_id=self.agent_id,
            owner_level=self.level
        )
        self.state.features.append(SystemFrequency())

    def set_action(self):
        """Configure system action space"""
        self.action.set_specs(
            dim_c=1,
            range=(np.array([-0.1]), np.array([0.1]))
        )

    def _build_subordinates(self, configs, env_id, upstream_id):
        """Build coordinators"""
        return {
            c["id"]: GridAgent(
                agent_id=c["id"],
                env_id=env_id,
                upstream_id=upstream_id
            )
            for c in configs
        }
```

### 4. Creating Custom Environment

```python
from heron.envs import PettingZooParallelEnv

class NetworkedGridEnv(PettingZooParallelEnv):
    """Multi-agent power grid environment."""

    def __init__(self, env_config):
        super().__init__(env_id=env_config.get("env_id", "grid"))

        # Build agents
        self.agent_dict = self._build_agents()

        # Register with HERON
        for agent in self.agent_dict.values():
            self.register_agent(agent)

        # Setup spaces using HeronEnvCore helpers
        self._init_spaces(
            action_spaces=self.get_agent_action_spaces(),
            observation_spaces=self.get_agent_observation_spaces()
        )

    def reset(self, *, seed=None, options=None):
        self.reset_agents(seed=seed)  # HERON helper
        obs = self.get_observations()  # HERON helper
        return obs, {}

    def step(self, actions):
        self.apply_actions(actions)  # HERON helper
        # ... physics simulation ...
        obs = self.get_observations()
        return obs, rewards, terminateds, truncateds, infos
```

### 5. Event-Driven Mode Setup

```python
# Setup event-driven mode (testing)
env.setup_event_driven()  # Creates EventScheduler

# Configure handlers
env.setup_default_handlers(
    global_state_fn=lambda: env.get_state(),
    on_action_effect=lambda aid, act: env.apply_action(aid, act)
)

# Run simulation
num_events = env.run_event_driven(t_end=3600.0, max_events=None)
```

---

## HeronEnvCore Methods

The `HeronEnvCore` mixin (inherited by `PettingZooParallelEnv`) provides:

| Method | Purpose |
|--------|---------|
| `register_agent(agent)` | Register agent with environment |
| `get_heron_agent(agent_id)` | Get agent by ID |
| `get_observations(global_state)` | Get observations for all agents |
| `apply_actions(actions)` | Apply actions to agents |
| `reset_agents(seed)` | Reset all registered agents |
| `setup_event_driven()` | Initialize event scheduler |
| `run_event_driven(t_end)` | Run event-driven simulation |
| `configure_agents_for_distributed()` | Setup message broker |

---

## Further Reading

- [HERON Core Documentation](../../../docs/)
- [Build from Scratch Guide](./BUILD_FROM_SCRATCH.md)
- [PandaPower Documentation](https://pandapower.readthedocs.io/)
- [RLlib Multi-Agent Tutorial](https://docs.ray.io/en/latest/rllib/rllib-env.html)

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
- Manages multiple DeviceAgents within a microgrid
- Can trade energy with other microgrids
- Maintains `GridState` with network-level features (voltages, line flows)
- Decomposes joint actions into per-device commands

**DeviceAgent**
- Represents a **physical device** (generator, storage, solar panel)
- Managed by a parent GridAgent
- Maintains `DeviceState` composed of `FeatureProvider`s
- Executes device actions via `Action` class (continuous/discrete)
- Can be RL-trainable or follow rule-based policies
- Responds to coordination signals (prices, setpoints) from GridAgent

**Agent Hierarchy**:
```
GridAgent "MG1" (level=2)
├── DeviceAgent "ESS1" (level=1) - Battery Storage
│   └── State: ElectricalBasePh, StorageBlock, PowerLimits, Status
├── DeviceAgent "DG1" (level=1) - Diesel Generator
│   └── State: ElectricalBasePh, PowerLimits, InverterBlock, Status
└── DeviceAgent "PV1" (level=1) - Solar Panel
    └── State: ElectricalBasePh, PowerLimits, Status
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

### 5. State and Feature System

PowerGrid uses a **modular feature-based state representation** where agent states are composed of multiple `FeatureProvider` objects.

**FeatureProvider**
- Base class for all state features
- Encapsulates observable/controllable attributes
- Defines visibility rules for multi-agent access control
- Provides vectorization for ML observations
- Supports serialization for communication

**Common Features**:

| Feature Type | Description | Example Attributes |
|-------------|-------------|-------------------|
| **ElectricalBasePh** | Active/reactive power state | P_MW, Q_MVAr, S_MVA, pf |
| **StorageBlock** | Battery state | soc, e_capacity_MWh, p_ch_max_MW, p_dsc_max_MW |
| **PowerLimits** | Device power constraints | p_min_MW, p_max_MW, q_min_MVAr, q_max_MVAr, s_rated_MVA |
| **InverterBlock** | Inverter parameters | inv_eff, pf_mode, var_mode |
| **StatusBlock** | Device status | in_service, controllable, state |
| **BusVoltages** | Network voltages | vm_pu, va_deg, bus_names |
| **LineFlows** | Line power flows | p_from_mw, q_from_mvar, loading_percent |
| **NetworkMetrics** | Grid-wide metrics | total_gen_mw, total_load_mw, total_loss_mw |

**Visibility Levels**:
- `"public"`: All agents can observe
- `"owner"`: Only owning agent can observe
- `"system"`: System-level agents (level ≥ 3) can observe
- `"upper_level"`: Agents one level above owner can observe

**Example**:
```python
from powergrid.core.state import DeviceState
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.storage import StorageBlock

# Create device state from features
state = DeviceState(
    owner_id="ess1",
    owner_level=1,
    features=[
        ElectricalBasePh(P_MW=0.5, Q_MVAr=0.1),
        StorageBlock(soc=0.8, e_capacity_MWh=2.0, p_ch_max_MW=1.0, p_dsc_max_MW=1.0)
    ]
)

# Vectorize for ML
vector = state.vector()  # Returns flat numpy array

# Update features
state.update({
    "ElectricalBasePh": {"P_MW": 0.6},
    "StorageBlock": {"soc": 0.75}
})

# Observe with access control
observable = state.observed_by(requestor_id="grid1", requestor_level=2)
# Returns: {"ElectricalBasePh": array([...]), "StorageBlock": array([...])}
```

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

PowerGrid uses a structured `Observation` dataclass:

```python
@dataclass
class Observation:
    local: Dict[str, Any]           # Local agent state
    global_info: Dict[str, Any]     # Global information visible to agent
    messages: List[Message]         # Inter-agent messages
    timestamp: float                # Current simulation time
```

**GridAgent observations**:
- **local**: Aggregated device states from subordinate DeviceAgents
- **global_info**: Network state (voltages, line loading) from environment
- **messages**: Coordination messages from peer GridAgents

**DeviceAgent observations**:
- **local**: Own device state features (e.g., SOC, power limits, electrical state)
- **global_info**: Network info at device bus (voltage, frequency)
- **messages**: Coordination signals from parent GridAgent

**Visibility Control**: Features define who can observe them via visibility rules:
```python
class ElectricalBasePh(FeatureProvider):
    visibility = ["owner", "upper_level"]  # Owner and parent can observe

# Only visible to agents with permission
obs_dict = state.observed_by(requestor_id="grid1", requestor_level=2)
```

### Action Space

PowerGrid uses a flexible `Action` dataclass with continuous and discrete components:

```python
@dataclass
class Action:
    c: np.ndarray      # Continuous actions (e.g., MW, MVAr setpoints)
    d: np.ndarray      # Discrete actions (e.g., on/off, tap position)
    dim_c: int         # Number of continuous dimensions
    dim_d: int         # Number of discrete dimensions
    ncats: List[int]   # Categories per discrete dimension
    range: Tuple       # (lower_bounds, upper_bounds) for continuous
```

**Automatic Gymnasium Space Conversion**:
```python
# Pure continuous
action = Action()
action.set_specs(dim_c=4, range=([-1, -0.5, -1, -0.5], [1, 0.5, 1, 0.5]))
space = action.space  # Returns Box(low=[-1, -0.5, -1, -0.5], high=[1, 0.5, 1, 0.5])

# Mixed continuous + discrete
action.set_specs(dim_c=2, dim_d=1, ncats=[3], range=(...))
space = action.space  # Returns Dict({"c": Box(...), "d": Discrete(3)})
```

**Normalization Support**:
```python
# RL agent outputs normalized [-1, 1]
normalized = agent.act(obs)

# Convert to physical units
action.unscale(normalized)  # Now action.c contains physical values

# Or normalize physical values
physical = action.c
normalized = action.scale()  # Returns [-1, 1] normalized version
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

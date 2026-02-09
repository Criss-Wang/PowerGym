# HERON Agents

This module provides the agent hierarchy for hierarchical multi-agent reinforcement learning (MARL) systems.

## Agent Hierarchy

HERON uses a 3-level hierarchical structure:

```
                    ┌─────────────────┐
         L3        │   SystemAgent    │     System-wide coordination
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
 L2 │ Coordinator A │ │ Coordinator B │ │ Coordinator C │  Regional coordination
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
      ┌─────┴─────┐     ┌─────┴─────┐     ┌─────┴─────┐
      │     │     │     │     │     │     │     │     │
      ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
 L1 ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐  Individual units
    │F1 │ │F2 │ │F3 │ │F4 │ │F5 │ │F6 │ │F7 │ │F8 │ │F9 │
    └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
```

## Agent Types

| Agent | Level | Role | Subordinates |
|-------|-------|------|--------------|
| `FieldAgent` | L1 | Individual units, sensors, actuators | None |
| `CoordinatorAgent` | L2 | Regional controllers, aggregators | FieldAgents |
| `SystemAgent` | L3 | Central controller, system operator | CoordinatorAgents |
| `ProxyAgent` | L0 | State distribution, visibility filtering | N/A |

## File Structure

```
heron/agents/
├── __init__.py           # Public exports
├── base.py               # Agent base class with messaging support
├── field_agent.py        # L1 field agents
├── coordinator_agent.py  # L2 coordinator agents
├── system_agent.py       # L3 system agents
├── hierarchical_agent.py # Shared L2/L3 functionality
└── proxy_agent.py        # State distribution proxy
```

## Execution Modes

HERON supports two execution modes that share the same agent implementations:

### Option A: Synchronous (Training)

All agents step together within `env.step()`. Suitable for RL training with CTDE (Centralized Training, Decentralized Execution).

```python
from heron.agents import FieldAgent, CoordinatorAgent

# Create agents
field_agents = [FieldAgent(f"field_{i}") for i in range(3)]
coordinator = CoordinatorAgent(
    agent_id="coordinator_1",
    subordinates={a.agent_id: a for a in field_agents}
)

# Training loop
obs = coordinator.observe(global_state)
action = policy(obs)  # Centralized policy
coordinator.act(obs, upstream_action=action)
```

### Option B: Event-Driven (Testing)

Agents tick independently at their own intervals with configurable delays. Suitable for realistic simulation and testing.

```python
from heron.scheduling import EventScheduler, TickConfig

# Configure timing
scheduler = EventScheduler()
coordinator._tick_config = TickConfig.with_jitter(
    tick_interval=5.0,   # Coordinator ticks every 5s
    msg_delay=0.1,       # 100ms message delay
    jitter_ratio=0.1,    # 10% timing variation
)

# Register with scheduler
scheduler.register_agent(
    coordinator.agent_id,
    tick_config=coordinator._tick_config
)

# Run simulation
scheduler.run_until(t_end=100.0)
```

In event-driven mode, hierarchical agents (L2/L3) always use async observations - collecting whatever observations have arrived from subordinates via the message broker.

## Two-Phase Update Flow

HERON uses a two-phase update pattern for agent-environment interaction:

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1                                  │
│  Agent.act() / Agent.tick()                                     │
│    → Compute action                                              │
│    → _update_action_features() updates action-dependent state   │
│    → Environment calls collect_agent_states()                   │
│    → Environment runs simulation with collected states          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 2                                  │
│  Environment returns simulation results                         │
│    → distribute_environment_results() called                    │
│    → Agent.update_from_environment() updates result features    │
└─────────────────────────────────────────────────────────────────┘
```

This enables features that depend on actions (e.g., power setpoint) to be set before simulation, while features that depend on simulation results (e.g., actual power output) are updated after.

## Quick Start

### Creating a Custom Field Agent

```python
import numpy as np
from heron.agents import FieldAgent
from heron.core.feature import FeatureProvider

# 1. Define state features
class TemperatureFeature(FeatureProvider):
    visibility = ["owner", "upper_level"]

    def __init__(self):
        self.temp = 20.0

    def vector(self):
        return np.array([self.temp], dtype=np.float32)

    def names(self):
        return ["temperature"]

    def to_dict(self):
        return {"temp": self.temp}

    @classmethod
    def from_dict(cls, d):
        f = cls()
        f.temp = d.get("temp", 20.0)
        return f

    def set_values(self, **kwargs):
        if "temp" in kwargs:
            self.temp = kwargs["temp"]

# 2. Create custom agent by overriding extension hooks
class ThermostatAgent(FieldAgent):
    def set_action(self):
        # Continuous action: temperature setpoint [18, 26]
        self.action.set_specs(
            dim_c=1,
            range=(np.array([18.0]), np.array([26.0]))
        )

    def set_state(self):
        # Add temperature feature to state
        self.state.features.append(TemperatureFeature())

    def _update_action_features(self, action, observation):
        # Phase 1: Update features based on action
        if action is not None:
            self.state.features[0].temp = float(action[0])

    def update_state(self, **kwargs):
        # Phase 2: Handle environment feedback
        if "measured_temp" in kwargs:
            self.state.features[0].temp = kwargs["measured_temp"]

# 3. Use the agent
agent = ThermostatAgent(agent_id="thermostat_1")
obs = agent.observe()
agent.act(obs, upstream_action=np.array([22.0]))
```

### Creating a Coordinator

```python
from heron.agents import CoordinatorAgent
from heron.protocols import SetpointProtocol

# Create coordinator with protocol
coordinator = CoordinatorAgent(
    agent_id="zone_controller",
    protocol=SetpointProtocol(),
)
coordinator.subordinates = {a.agent_id: a for a in field_agents}

# Get joint observation and action spaces
obs_space = coordinator.get_joint_observation_space()
act_space = coordinator.get_joint_action_space()

# Coordinate subordinates
obs = coordinator.observe()
coordinator.act(obs, upstream_action=joint_action)
```

### Creating a System Agent

```python
from heron.agents import SystemAgent
from heron.core.feature import FeatureProvider

class GridSystemAgent(SystemAgent):
    def set_state(self):
        # Add system-level features
        self.state.features.append(SystemFrequencyFeature())

    def set_action(self):
        # System-level action (e.g., frequency regulation)
        self.action.set_specs(dim_c=1, range=(np.array([-0.1]), np.array([0.1])))

    def _build_subordinates(self, configs, env_id, upstream_id):
        # Build coordinators from config
        return {c["id"]: MyCoordinator(c["id"]) for c in configs}

system = GridSystemAgent(
    agent_id="grid_system",
    config={"coordinators": [{"id": "coord_1"}, {"id": "coord_2"}]}
)
```

### Using ProxyAgent for Visibility Control

```python
from heron.agents import ProxyAgent

# Create proxy with visibility rules
proxy = ProxyAgent(
    agent_id="proxy",
    registered_agents=["sensor_1", "controller_1"],
    visibility_rules={
        "sensor_1": ["reading"],      # Sensor sees only readings
        "controller_1": ["reading", "status"],  # Controller sees more
    }
)

# Update proxy with environment state
proxy.update_state(env_state)

# Agents request filtered state
state = proxy.get_state_for_agent(
    agent_id="sensor_1",
    requestor_level=1
)
```

## Extension Hooks

Agents provide hooks for customization. Override these in subclasses:

### FieldAgent Hooks

| Hook | Purpose | When Called |
|------|---------|-------------|
| `set_action()` | Define action space | During `__init__` |
| `set_state()` | Define state features | During `__init__` |
| `reset_agent(**kwargs)` | Custom reset logic | During `reset()` |
| `_update_action_features(action, obs)` | Phase 1: Update action-dependent features | After `act()`/`tick()` |
| `update_state(**kwargs)` | Phase 2: Handle environment feedback | During `update_from_environment()` |
| `feasible_action()` | Clamp action to feasible range | Optional, called by subclass |

### SystemAgent Hooks

| Hook | Purpose | When Called |
|------|---------|-------------|
| `set_state()` | Define system-level state features | During `__init__` |
| `set_action()` | Define system-level action space | During `__init__` |
| `reset_system(**kwargs)` | Custom system reset logic | During `reset()` |
| `_build_subordinates(configs, env_id, upstream_id)` | Build coordinator agents | During `__init__` |

## Key Concepts

### Coordination Protocols

Agents use protocols to coordinate with subordinates:

- **SetpointProtocol**: Direct action assignment (centralized)
- **PriceSignalProtocol**: Indirect coordination via price signals (decentralized)
- **SystemProtocol**: System-level directives

```python
from heron.protocols import SetpointProtocol, PriceSignalProtocol

# Centralized control
coordinator.protocol = SetpointProtocol()

# Decentralized coordination
coordinator.protocol = PriceSignalProtocol(initial_price=50.0)
```

### Tick Configuration

Control agent timing in event-driven mode:

```python
from heron.scheduling import TickConfig, JitterType

# Deterministic timing (training)
config = TickConfig.deterministic(
    tick_interval=1.0,
    obs_delay=0.0,
    act_delay=0.0,
)

# With jitter (testing)
config = TickConfig.with_jitter(
    tick_interval=1.0,
    obs_delay=0.1,
    act_delay=0.05,
    msg_delay=0.02,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=42,
)

agent = FieldAgent(agent_id="field_1", tick_config=config)
```

### Message Broker Integration

For distributed execution, agents communicate via message broker:

```python
from heron.messaging import InMemoryBroker

broker = InMemoryBroker()

# Configure agents
coordinator.set_message_broker(broker)
for agent in field_agents:
    agent.set_message_broker(broker)

# Agents automatically use broker for communication
coordinator.send_action_to_subordinate("field_1", action)
field_agent.receive_action_messages()
```

### Visibility Control

Two visibility mechanisms are supported:

1. **Key-based** (`visibility_rules`): Simple dict mapping agent_id → allowed state keys
2. **FeatureProvider-based** (`is_observable_by()`): Level-aware visibility rules

```python
# Key-based (via ProxyAgent)
proxy = ProxyAgent(
    visibility_rules={"agent_1": ["voltage", "power"]}
)

# FeatureProvider-based (in feature definition)
class VoltageFeature(FeatureProvider):
    visibility = ["owner", "upper_level"]  # Only owner and L2+ can see
```

## API Reference

### Agent (Base Class)

```python
class Agent(ABC):
    # Attributes
    agent_id: AgentID
    level: int
    upstream_id: Optional[AgentID]
    env_id: Optional[str]
    subordinates: Dict[AgentID, Agent]

    # Core lifecycle methods
    def reset(self, *, seed=None, **kwargs) -> None
    def observe(self, global_state=None, proxy=None, **kwargs) -> Observation
    def act(self, observation, upstream_action=None) -> Optional[Action]
    def tick(self, scheduler, current_time, global_state=None, proxy=None) -> None

    # Two-phase update
    def _update_action_features(self, action, observation) -> None  # Override in subclass
    def update_from_environment(self, env_state: Dict) -> None
    def get_state_for_environment(self) -> Dict[str, Any]

    # Messaging
    def set_message_broker(self, broker: MessageBroker) -> None
    def send_message(self, content, recipient_id, message_type=MessageType.INFO) -> None
    def receive_action_messages(self, clear=True) -> List[Any]
    def send_action_to_subordinate(self, subordinate_id, action, scheduler=None) -> None
    def send_observation_to_upstream(self, observation, scheduler=None) -> None
```

### FieldAgent

```python
class FieldAgent(Agent):
    # Attributes
    state: FieldAgentState
    action: Action
    policy: Optional[Policy]
    protocol: Optional[Protocol]
    action_space: gym.Space
    observation_space: gym.Space

    # Extension hooks (override in subclasses)
    def set_action(self) -> None
    def set_state(self) -> None
    def reset_agent(self, **kwargs) -> None
    def update_state(self, **kwargs) -> None
    def feasible_action(self) -> None
```

### HierarchicalAgent (L2/L3 Base)

```python
class HierarchicalAgent(Agent):
    # Attributes
    subordinates: Dict[AgentID, Agent]
    state: State
    protocol: Optional[Protocol]
    policy: Optional[Policy]

    # Joint spaces for RL
    def get_joint_observation_space(self) -> gym.Space
    def get_joint_action_space(self) -> gym.Space
    def get_subordinate_action_spaces(self) -> Dict[AgentID, gym.Space]

    # Coordination
    def coordinate_subordinates(self, observation, action) -> None

    # Abstract methods (implement in subclasses)
    def _build_subordinates(self, configs, env_id, upstream_id) -> Dict[AgentID, Agent]
    def _get_subordinate_obs_key(self) -> str
    def _get_state_obs_key(self) -> str
    def _get_subordinate_action_dim(self, subordinate) -> int
```

### CoordinatorAgent

```python
class CoordinatorAgent(HierarchicalAgent):
    # Manages FieldAgents (L1)
    subordinates: Dict[AgentID, FieldAgent]
    state: CoordinatorAgentState
```

### SystemAgent

```python
class SystemAgent(HierarchicalAgent):
    # Manages CoordinatorAgents (L2)
    subordinates: Dict[AgentID, CoordinatorAgent]
    coordinators: Dict[AgentID, CoordinatorAgent]  # Alias for subordinates
    state: SystemAgentState
    action: Action

    # Extension hooks
    def set_state(self) -> None
    def set_action(self) -> None
    def reset_system(self, **kwargs) -> None

    # Visibility and broadcasting
    def filter_state_for_agent(self, state, requestor_id, requestor_level) -> Dict
    def broadcast_to_coordinators(self, message: Dict) -> None
```

### ProxyAgent

```python
class ProxyAgent(Agent):
    # State management
    def update_state(self, state: Dict) -> None
    def get_state_at_time(self, target_time: float) -> Dict
    def get_state_for_agent(self, agent_id, requestor_level, ...) -> Dict

    # Registration
    def register_agent(self, agent_id: AgentID) -> None
    def set_visibility_rules(self, agent_id, allowed_keys: List[str]) -> None

    # Message broker support
    def receive_state_from_environment(self) -> Optional[Dict]
    def distribute_state_to_agents(self) -> None
```

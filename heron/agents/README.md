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

## Execution Modes

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
coordinator.tick_config = TickConfig.with_jitter(
    tick_interval=5.0,   # Coordinator ticks every 5s
    msg_delay=0.1,       # 100ms message delay
    jitter_ratio=0.1,    # 10% timing variation
)

# Register with scheduler
scheduler.register_agent(
    coordinator.agent_id,
    tick_config=coordinator.tick_config
)

# Run simulation
scheduler.run_until(t_end=100.0)
```

## File Structure

```
heron/agents/
├── __init__.py           # Public exports
├── base.py               # Agent base class
├── field_agent.py        # L1 field agents
├── coordinator_agent.py  # L2 coordinator agents
├── system_agent.py       # L3 system agents
├── hierarchical_agent.py # Shared L2/L3 functionality
└── proxy_agent.py        # State distribution proxy
```

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

# 2. Create custom agent
class ThermostatAgent(FieldAgent):
    def set_action(self):
        # Continuous action: temperature setpoint [18, 26]
        self.action.set_specs(
            dim_c=1,
            range=(np.array([18.0]), np.array([26.0]))
        )

    def set_state(self):
        self.state.features.append(TemperatureFeature())

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
    subordinates={a.agent_id: a for a in field_agents}
)

# Get joint observation and action spaces
obs_space = coordinator.get_joint_observation_space()
act_space = coordinator.get_joint_action_space()

# Coordinate subordinates
obs = coordinator.observe()
coordinator.act(obs, upstream_action=joint_action)
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
        "controller_1": ["*"],        # Controller sees everything
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

agent.tick_config = config
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

## API Reference

### Agent Base Class

```python
class Agent(ABC):
    # Core methods
    def reset(self, *, seed=None, **kwargs) -> None
    def observe(self, global_state=None, **kwargs) -> Observation
    def act(self, observation, **kwargs) -> Optional[Action]
    def tick(self, scheduler, current_time, global_state=None, proxy=None) -> None

    # Messaging
    def set_message_broker(self, broker) -> None
    def send_message(self, content, recipient_id, message_type="INFO") -> None
    def receive_messages(self, sender_id=None, clear=True) -> List[Dict]

    # Properties
    agent_id: str
    level: int
    tick_config: TickConfig
    subordinates: Dict[str, Agent]
    upstream_id: Optional[str]
```

### FieldAgent

```python
class FieldAgent(Agent):
    # Override these in subclasses
    def set_action(self) -> None      # Define action space
    def set_state(self) -> None       # Define state features
    def reset_agent(self, **kwargs)   # Custom reset logic
    def update_state(self, **kwargs)  # Handle env feedback

    # Attributes
    state: FieldAgentState
    action: Action
    policy: Optional[Policy]
    protocol: Optional[Protocol]
```

### CoordinatorAgent / SystemAgent

```python
class CoordinatorAgent(HierarchicalAgent):
    # Joint spaces for RL
    def get_joint_observation_space() -> gym.Space
    def get_joint_action_space() -> gym.Space

    # Coordination
    def coordinate_subordinates(action) -> None

    # Attributes
    subordinates: Dict[str, FieldAgent]
    state: CoordinatorAgentState
```

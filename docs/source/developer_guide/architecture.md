# Architecture Overview

HERON provides a layered architecture with clear separation of concerns between the domain-agnostic framework and domain-specific case studies.

## Project Structure

```
heron/                          # Domain-agnostic MARL framework
├── agents/                     # Hierarchical agent abstractions
│   ├── base.py                 # Agent base class with level property
│   ├── field_agent.py          # Leaf-level agents (local sensing/actuation)
│   ├── coordinator_agent.py    # Mid-level agents (manages child agents)
│   ├── system_agent.py         # Top-level agents (global coordination)
│   └── proxy_agent.py          # Proxy agent for distributed execution
│
├── core/                       # Core abstractions
│   ├── action.py               # Action with continuous/discrete support
│   ├── observation.py          # Observation with local/global/messages
│   ├── state.py                # State with FeatureProvider composition
│   ├── feature.py              # FeatureProvider with visibility tags
│   └── policies.py             # Policy abstractions (random, rule-based)
│
├── protocols/                  # Coordination protocols
│   ├── base.py                 # Protocol, CommunicationProtocol interfaces
│   ├── vertical.py             # SetpointProtocol, PriceSignalProtocol
│   └── horizontal.py           # P2PTradingProtocol, ConsensusProtocol
│
├── messaging/                  # Message broker system
│   ├── base.py                 # MessageBroker interface, ChannelManager
│   └── memory.py               # InMemoryBroker implementation
│
├── envs/                       # Base environment interfaces
│   └── base.py                 # Abstract environment classes
│
└── utils/                      # Common utilities
    ├── typing.py               # Type definitions
    ├── array_utils.py          # Array manipulation utilities
    └── registry.py             # Feature registry
```

## Design Principles

### 1. Hierarchical Agents

Multi-level hierarchy with configurable depth:

```
Level 3: SystemAgent (global coordination)
         └── Level 2: CoordinatorAgent (zone management)
                      └── Level 1: FieldAgent (local control)
```

Each level has distinct responsibilities:
- **FieldAgent**: Local sensing, actuation, state management
- **CoordinatorAgent**: Manages subordinate agents, aggregates observations
- **SystemAgent**: Global objectives, price signals, system constraints

### 2. Feature-Based State

Composable `FeatureProvider` with visibility tags:

```python
class FeatureProvider(ABC):
    def __init__(self, visibility: list[str]):
        self.visibility = visibility

    @abstractmethod
    def vector(self) -> np.ndarray:
        """Return feature as numpy array."""
        pass

    @abstractmethod
    def dim(self) -> int:
        """Return feature dimension."""
        pass
```

### 3. Protocol-Driven Coordination

Protocols define coordination patterns:

- **Vertical**: Top-down control (setpoints, prices)
- **Horizontal**: Peer coordination (trading, consensus)

```python
class Protocol(ABC):
    @abstractmethod
    def execute(self, agents: list[Agent]) -> dict:
        """Execute coordination logic."""
        pass
```

### 4. Message Broker Abstraction

Decoupled communication via `MessageBroker`:

```python
class MessageBroker(ABC):
    @abstractmethod
    async def publish(self, channel: str, message: dict) -> None:
        pass

    @abstractmethod
    async def consume(self, channel: str) -> dict:
        pass

    @abstractmethod
    def create_channel(self, channel: str) -> None:
        pass
```

## Data Flow

### Centralized Mode

```
┌──────────────────────────────────────────┐
│              Environment                  │
│  ┌────────────────────────────────────┐  │
│  │         Global State               │  │
│  └────────────────────────────────────┘  │
│         │              │              │   │
│         ▼              ▼              ▼   │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐
│    │ Agent 1 │    │ Agent 2 │    │ Agent 3 │
│    │ observe │    │ observe │    │ observe │
│    │   act   │    │   act   │    │   act   │
│    └─────────┘    └─────────┘    └─────────┘
│         │              │              │   │
│         └──────────────┼──────────────┘   │
│                        ▼                  │
│              ┌─────────────────┐         │
│              │ Apply Actions   │         │
│              └─────────────────┘         │
└──────────────────────────────────────────┘
```

### Distributed Mode

```
┌──────────────────────────────────────────┐
│            Message Broker                 │
│  ┌────────────────────────────────────┐  │
│  │    Channel: control_signals        │  │
│  │    Channel: agent_actions          │  │
│  │    Channel: peer_messages          │  │
│  └────────────────────────────────────┘  │
│         ▲              ▲              ▲   │
│         │              │              │   │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐
│    │ Proxy 1 │    │ Proxy 2 │    │ Proxy 3 │
│    └────┬────┘    └────┬────┘    └────┬────┘
│         │              │              │   │
│    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
│    │ Agent 1 │    │ Agent 2 │    │ Agent 3 │
│    └─────────┘    └─────────┘    └─────────┘
└──────────────────────────────────────────┘
```

## Extension Points

| Component | How to Extend |
|-----------|---------------|
| Agents | Subclass `Agent`, `FieldAgent`, `CoordinatorAgent` |
| Features | Implement `FeatureProvider` |
| Protocols | Implement `Protocol` or `CommunicationProtocol` |
| Brokers | Implement `MessageBroker` |
| Environments | Subclass `ParallelEnv` or case study base classes |

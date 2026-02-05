# HERON Protocols

This module provides coordination protocols for multi-agent systems, enabling both hierarchical (vertical) and peer-to-peer (horizontal) coordination patterns.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Protocol                                    │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │   CommunicationProtocol     │  │      ActionProtocol             │   │
│  │                             │  │                                 │   │
│  │  • WHAT to communicate      │  │  • HOW to coordinate actions    │   │
│  │  • Compute messages/signals │  │  • Centralized vs decentralized │   │
│  └─────────────────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

                    Vertical Protocols                 Horizontal Protocols
                    (Agent-owned)                      (Environment-owned)
                         │                                    │
            ┌────────────┼────────────┐          ┌───────────┼───────────┐
            │            │            │          │           │           │
            ▼            ▼            ▼          ▼           ▼           ▼
       Setpoint      Price       System      Trading    Consensus    No-op
       Protocol     Signal      Protocol    Protocol    Protocol
                   Protocol
```

## File Structure

```
heron/protocols/
├── __init__.py     # Public exports
├── base.py         # Abstract interfaces and no-op implementations
├── vertical.py     # Hierarchical coordination (parent -> children)
└── horizontal.py   # Peer coordination (agent <-> agent)
```

## Protocol Architecture

Every protocol consists of two components:

| Component | Responsibility | Example |
|-----------|----------------|---------|
| `CommunicationProtocol` | **WHAT** to communicate | Price signals, setpoints, bids |
| `ActionProtocol` | **HOW** to coordinate actions | Centralized control, decentralized response |

```python
from heron.protocols import Protocol

class MyProtocol(Protocol):
    def __init__(self):
        super().__init__(
            communication_protocol=MyCommunication(),
            action_protocol=MyActionCoordination()
        )
```

## Vertical Protocols (Hierarchical)

Vertical protocols handle **Parent → Children** coordination. Each agent owns its vertical protocol to coordinate subordinates.

### SetpointProtocol

Centralized control where parent directly assigns actions to children.

```python
from heron.protocols import SetpointProtocol
import numpy as np

# Create protocol
protocol = SetpointProtocol()

# Coordinator uses protocol to distribute joint action
coordinator.protocol = protocol

# Joint action [0.5, 0.3, 0.2] is split among 3 subordinates
coordinator.act(obs, upstream_action=np.array([0.5, 0.3, 0.2]))
```

**Characteristics:**
- Communication: Sends setpoint assignments to subordinates
- Action: Centralized - coordinator directly controls subordinate actions
- Use case: Deterministic control, RL training with joint action space

### PriceSignalProtocol

Decentralized coordination via price signals. Children respond independently.

```python
from heron.protocols import PriceSignalProtocol

# Create with initial price
protocol = PriceSignalProtocol(initial_price=50.0)

# Update price based on supply/demand
protocol.price = 75.0  # High demand -> higher price

# Subordinates receive price and decide independently
# e.g., generators increase output, batteries discharge
```

**Characteristics:**
- Communication: Broadcasts price signal to all subordinates
- Action: Decentralized - subordinates respond to price independently
- Use case: Market-based coordination, demand response

### SystemProtocol

System-level coordination from L3 SystemAgent to L2 CoordinatorAgents.

```python
from heron.protocols import SystemProtocol

# System agent coordinates coordinators
system = SystemAgent(
    agent_id="grid_system",
    protocol=SystemProtocol()
)

# System sends directives (frequency regulation, emergency commands, etc.)
system.act(obs, upstream_action=system_directive)
```

**Characteristics:**
- Communication: Sends system-wide directives to coordinators
- Action: Centralized by default (customizable)
- Use case: System-level control, emergency response

## Horizontal Protocols (Peer-to-Peer)

Horizontal protocols handle **Peer ↔ Peer** coordination. The environment owns and runs horizontal protocols since they require a global view.

### PeerToPeerTradingProtocol

P2P marketplace for resource trading between agents.

```python
from heron.protocols import PeerToPeerTradingProtocol

# Create with configurable parameters
protocol = PeerToPeerTradingProtocol(
    trading_fee=0.01,
    demand_field="net_demand",      # Field name in agent state
    cost_field="marginal_cost",     # Field name in agent state
    default_cost=50.0,
    buy_price_multiplier=1.2,       # Buyers bid at cost * 1.2
    sell_price_multiplier=0.8,      # Sellers offer at cost * 0.8
)

# Environment runs the protocol
messages, actions = protocol.coordinate(
    coordinator_state=None,  # Not used in horizontal
    subordinate_states=agent_states,
)
# Returns trade confirmations for each agent
```

**How it works:**
1. Agents provide `net_demand` (positive=buy, negative=sell) and `marginal_cost` in their state
2. Protocol collects bids/offers based on demand and cost
3. Market clearing matches buyers and sellers
4. Trade confirmations sent to participating agents

**Characteristics:**
- Communication: Market clearing, trade confirmations
- Action: Adjusts agent setpoints based on trades
- Use case: Energy trading, resource sharing

### ConsensusProtocol

Distributed consensus via gossip algorithm.

```python
from heron.protocols import ConsensusProtocol

# Create with configurable parameters
protocol = ConsensusProtocol(
    max_iterations=10,
    tolerance=0.01,
    value_field="control_value",    # Field name in agent state
    default_value=0.0,
)

# Environment runs consensus
messages, actions = protocol.coordinate(
    coordinator_state=None,
    subordinate_states=agent_states,
    context={"topology": {"adjacency": adjacency_dict}}  # Optional
)
# Returns consensus values for each agent
```

**How it works:**
1. Agents provide `control_value` in their state
2. Protocol iteratively averages values with neighbors
3. Continues until convergence or max_iterations
4. Consensus values sent to all agents

**Characteristics:**
- Communication: Iterative averaging until convergence
- Action: None - agents use consensus values in their own policies
- Use case: Distributed control, coordination without central authority

## No-Op Protocols

For agents that don't need coordination:

```python
from heron.protocols import NoProtocol, NoHorizontalProtocol

# No vertical coordination
agent.protocol = NoProtocol()

# No horizontal coordination
env.horizontal_protocol = NoHorizontalProtocol()
```

## Custom Protocols

### Creating a Custom Vertical Protocol

```python
from heron.protocols.base import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
)
from typing import Any, Dict, Optional

class MyCommProtocol(CommunicationProtocol):
    def __init__(self):
        self.neighbors = set()

    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_states: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compute messages to send to each receiver."""
        messages = {}
        for receiver_id, state in receiver_states.items():
            messages[receiver_id] = {
                "type": "my_signal",
                "value": compute_signal(sender_state, state)
            }
        return messages

class MyActionProtocol(ActionProtocol):
    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[str, Any],
        coordination_messages: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Compute actions for each subordinate."""
        actions = {}
        for sub_id, state in subordinate_states.items():
            # Centralized: decompose coordinator action
            # Decentralized: return None (agents decide independently)
            actions[sub_id] = compute_subordinate_action(
                coordinator_action, state, coordination_messages
            )
        return actions

class MyProtocol(Protocol):
    def __init__(self):
        super().__init__(
            communication_protocol=MyCommProtocol(),
            action_protocol=MyActionProtocol()
        )
```

### Creating a Custom Horizontal Protocol

```python
from heron.protocols.horizontal import HorizontalProtocol

class MyHorizontalProtocol(HorizontalProtocol):
    def __init__(self, my_param: float = 1.0):
        super().__init__(
            communication_protocol=MyHorizontalComm(my_param),
            action_protocol=MyHorizontalAction()
        )
```

## Domain-Agnostic Configuration

Protocols are configurable to work across different domains:

```python
# Power grid domain (default)
trading = PeerToPeerTradingProtocol()

# Supply chain domain
trading = PeerToPeerTradingProtocol(
    demand_field="inventory_deficit",
    cost_field="unit_cost",
    default_cost=10.0,
)

# Ride-sharing domain
consensus = ConsensusProtocol(
    value_field="surge_price",
    default_value=1.0,
)

# Temperature control domain
consensus = ConsensusProtocol(
    value_field="temperature_setpoint",
    default_value=20.0,
)
```

## Protocol Comparison

| Protocol | Direction | Coordination | Action Control | Use Case |
|----------|-----------|--------------|----------------|----------|
| `SetpointProtocol` | Vertical | Setpoint assignment | Centralized | Direct control |
| `PriceSignalProtocol` | Vertical | Price broadcast | Decentralized | Market-based |
| `SystemProtocol` | Vertical | System directives | Centralized | System-level |
| `PeerToPeerTradingProtocol` | Horizontal | Market clearing | Adjustment-based | P2P trading |
| `ConsensusProtocol` | Horizontal | Gossip averaging | None (agents decide) | Distributed control |
| `NoProtocol` | Vertical | None | None | Independent agents |
| `NoHorizontalProtocol` | Horizontal | None | None | No peer coord |

## API Reference

### Protocol (Base)

```python
class Protocol:
    communication_protocol: CommunicationProtocol
    action_protocol: ActionProtocol

    def no_op(self) -> bool:
        """Check if this is a no-operation protocol."""

    def coordinate(
        self,
        coordinator_state: Any,
        subordinate_states: Dict[str, Any],
        coordinator_action: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """Execute full coordination cycle.

        Returns:
            (messages, actions) tuple
        """
```

### CommunicationProtocol (Abstract)

```python
class CommunicationProtocol(ABC):
    neighbors: Set[str]

    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_states: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compute messages for each receiver."""

    def add_neighbor(self, agent: str) -> None
    def init_neighbors(self, neighbors: List[str]) -> None
```

### ActionProtocol (Abstract)

```python
class ActionProtocol(ABC):
    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[str, Any],
        coordination_messages: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """Compute actions for each subordinate."""
```

### Vertical Protocols

| Class | Parameters |
|-------|------------|
| `SetpointProtocol` | (none) |
| `PriceSignalProtocol` | `initial_price: float = 50.0` |
| `SystemProtocol` | `communication_protocol`, `action_protocol` (optional) |

### Horizontal Protocols

| Class | Parameters |
|-------|------------|
| `PeerToPeerTradingProtocol` | `trading_fee`, `demand_field`, `cost_field`, `default_cost`, `buy_price_multiplier`, `sell_price_multiplier` |
| `ConsensusProtocol` | `max_iterations`, `tolerance`, `value_field`, `default_value` |
| `NoHorizontalProtocol` | (none) |

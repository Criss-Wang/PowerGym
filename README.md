# HERON: Hierarchical Environments for Realistic Observability in Networks

A **domain-agnostic Multi-Agent Reinforcement Learning (MARL) framework** with a power systems case study. HERON provides modular abstractions for hierarchical agents, coordination protocols, and realistic observability constraints.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Setting Up Your Own Project](#setting-up-your-own-project)
- [Included Case Study](#included-case-study)
- [Documentation](#documentation)
- [Development](#development)

---

## Features

### Core Framework
- **Hierarchical Agent System**: Multi-level hierarchy with configurable depth (e.g., Field → Coordinator → System)
- **Feature-based State**: Composable `FeatureProvider` system with extensible visibility tags (defaults: `owner`, `coordinator`, `system`, `global`)
- **Coordination Protocols**: Vertical (Setpoint, Price Signal) and Horizontal (P2P Trading, Consensus)
- **Message Broker System**: `InMemoryBroker` with extensible interface for Kafka/RabbitMQ
- **Dual-mode Execution**: Support centralized and de-centralized execution modes with easy configurations

### RL Integration
- Works with canonical RL algorithm libraries like **RLlib** and **Stable-Baselines3**
- Support plugin-and-play for canonical MARL environments like **PettingZoo**, **MAgent** and **SMAC**

---

## Architecture Overview

HERON provides a layered architecture with clear separation of concerns:

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
│   ├── feature.py              # FeatureProvider with extensible visibility tags
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

### Key Design Principles

1. **Hierarchical Agents**: Multi-level hierarchy with configurable depth; agents at each level have distinct responsibilities
2. **Feature-based State**: Composable `FeatureProvider` with extensible visibility tags for information sharing control
3. **Protocol-driven Coordination**: Vertical protocols for top-down control, horizontal protocols for peer coordination
4. **Message Broker Abstraction**: Decoupled communication via `MessageBroker` interface

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Criss-Wang/PowerGym.git
cd PowerGym
```

### Step 2: Create Virtual Environment

```bash
# Create a new virtual environment
python3 -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Upgrade pip
pip install -U pip
```

### Step 3: Install the Package

Choose the installation option that fits your needs:

```bash
# Basic installation (core framework only)
pip install -e .

# With power grid domain support
pip install -e ".[powergrid]"

# With multi-agent RL support (RLlib, PettingZoo)
pip install -e ".[multi_agent]"

# Full installation (all features)
pip install -e ".[all]"

# For development (includes testing and linting tools)
pip install -e ".[dev,all]"
```

### Step 4: Verify Installation

```bash
# Test the installation
python -c "import heron; import powergrid; print('Installation successful')"

# Run tests (optional)
pytest tests/ -v
```

---

## Quick Start

### Core Concepts

HERON provides a hierarchical agent framework with configurable levels:

```python
from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.core.action import Action
from heron.core.feature import FeatureProvider

# 1. Define Features with visibility levels
class TemperatureFeature(FeatureProvider):
    """A feature visible only to its owner."""

    def __init__(self, value: float = 20.0):
        self.value = value

    def to_array(self) -> np.ndarray:
        return np.array([self.value], dtype=np.float32)

    @property
    def visibility(self) -> list[str]:
        return ["owner"]  # Default tags: "owner", "coordinator", "system", "global" (extensible)

# 2. Create Agents at different hierarchy levels
class SensorAgent(Agent):
    """Field agent at level 1 - collects local observations."""
    level = 1  # Configurable: can be any positive integer

    def observe(self) -> Observation:
        return Observation(local={"temp": self.temperature.to_array()})

    def act(self, obs: Observation, action: Action) -> Action:
        # Process action from coordinator
        return action

# 3. Use Protocols for coordination
from heron.protocols.vertical import SetpointProtocol
from heron.protocols.horizontal import ConsensusProtocol

# Vertical: coordinator sends setpoints to field agents
vertical_protocol = SetpointProtocol()

# Horizontal: peer agents reach consensus
horizontal_protocol = ConsensusProtocol(max_iterations=10, tolerance=0.01)
```

> 💡 The code above shows **direct execution** (`observe()` → `act()`). For distributed hierarchical control with message passing, agents can use `await agent.step_distributed()` which recursively coordinates through the agent tree via the message broker.

### Agent Hierarchy

Agents are organized in a tree structure where each level has distinct responsibilities:

| Level | Role | Example |
|-------|------|---------|
| Leaf | **Field Agent** - Local sensing and actuation | Sensor, Device |
| Mid | **Coordinator** - Manages child agents in a subtree | Zone Controller |
| Root | **System Agent** - Global coordination | System Operator |

The hierarchy depth is configurable. A simple setup might use 2 levels (Field + Coordinator), while complex systems can use 4+ levels.

### Visibility Levels

Features use string-based visibility tags to control information sharing. HERON provides 4 default tags, but you can define custom tags for your domain:

```python
# Default visibility tags
local_feature.visibility = ["owner"]                    # Only the owning agent
shared_feature.visibility = ["owner", "coordinator"]    # Owner and its coordinator
system_feature.visibility = ["system"]                  # System-level agents only
public_feature.visibility = ["global"]                  # Everyone

# Custom visibility tags (domain-specific)
neighbor_feature.visibility = ["owner", "neighbor"]     # Owner and neighboring agents
region_feature.visibility = ["region_a"]                # Agents in region A only
```

The visibility system is tag-based: an agent can see a feature if it holds any of the feature's visibility tags.

### Message Broker

For hierarchical execution with `step_distributed()`, agents communicate through a `MessageBroker`:

```python
from heron.messaging.memory import InMemoryBroker
from heron.messaging.base import ChannelManager

# Create broker
broker = InMemoryBroker()

# Create agents with broker
coordinator = CoordinatorAgent(
    agent_id="coordinator",
    message_broker=broker,
    subordinates={"field_1": field_agent_1, "field_2": field_agent_2}
)
field_agent = FieldAgent(
    agent_id="field_1",
    message_broker=broker,
    upstream_id="coordinator"
)

# Hierarchical execution (async)
await coordinator.step_distributed()  # Recursively coordinates all subordinates
```

The `MessageBroker` interface can be extended for distributed systems (e.g., Kafka, RabbitMQ) by implementing `publish()`, `consume()`, and `create_channel()`.

---

## Setting Up Your Own Project

This section guides you through creating a new project that uses HERON as a dependency.

### Project Structure

Your new project will have this minimal structure:

```
my_project/
├── .venv/                      # Virtual environment (created by python -m venv)
├── my_domain/                  # Your domain-specific code
│   ├── __init__.py
│   ├── agents/                 # Your custom agents
│   │   ├── __init__.py
│   │   └── my_agent.py
│   ├── envs/                   # Your custom environments
│   │   ├── __init__.py
│   │   └── my_env.py
│   └── utils/                  # Your utilities
│       └── __init__.py
├── examples/                   # Your example scripts
│   └── train_my_agent.py
├── tests/                      # Your tests
│   └── test_my_env.py
├── pyproject.toml              # Project configuration
└── README.md
```

**Optional directories** (add as needed for your domain):
- `my_domain/core/` - Only needed when extending `heron.core` classes (e.g., custom features, state management)
- `my_domain/setups/` or `data/` - For environment configuration and data files (structure is flexible)

### Quick Setup (Recommended)

From the cloned PowerGym repository:

```bash
make new-project NAME=my_project DOMAIN=my_domain
```

This creates the entire project structure with `pyproject.toml` and `README.md`. Then:

```bash
cd my_project
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[heron,dev]"
```

Then create the `pyproject.toml` manually (see Step 3 below).

---

### Manual Step-by-Step Setup

#### 1. Create Project Directory

```bash
mkdir my_project
cd my_project
```

#### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
```

#### 3. Create pyproject.toml

Create `pyproject.toml` with your project configuration:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-domain"
version = "0.1.0"
description = "My custom domain using HERON framework"
requires-python = ">=3.10"
dependencies = [
    "gymnasium>=1.0.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
]

[project.optional-dependencies]
heron = [
    # Install HERON from GitHub
    "heron-marl @ git+https://github.com/Criss-Wang/PowerGym.git",
]
dev = [
    "pytest>=7.0.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["my_domain*"]
```

#### 4. Create Directory Structure

```bash
mkdir -p my_domain/{agents,envs,utils}
mkdir -p examples tests

# Create __init__.py files
touch my_domain/__init__.py
touch my_domain/agents/__init__.py
touch my_domain/envs/__init__.py
touch my_domain/utils/__init__.py
```

#### 5. Install Dependencies

```bash
# Install your project with HERON
pip install -e ".[heron,dev]"

# Or install HERON directly from source (for development)
pip install -e /path/to/PowerGym[all]
```

#### 6. Create a Custom Agent

Create `my_domain/agents/my_agent.py`:

```python
"""Example custom agent using HERON framework."""

import numpy as np
from heron.agents.base import Agent
from heron.core.action import Action
from heron.core.observation import Observation
from heron.core.state import FieldAgentState
from heron.core.feature import FeatureProvider


class MyCustomFeature(FeatureProvider):
    """Custom feature for your domain."""

    def __init__(self, value: float = 0.0):
        super().__init__(visibility=["owner"])
        self.value = value

    def vector(self) -> np.ndarray:
        return np.array([self.value], dtype=np.float32)

    def dim(self) -> int:
        return 1


class MyAgent(Agent):
    """Custom agent for your domain."""

    def __init__(self, agent_id: str, initial_value: float = 0.0):
        super().__init__(agent_id, level=1)

        # Initialize state with custom features
        self.state = FieldAgentState()
        self.my_feature = MyCustomFeature(initial_value)
        self.state.add_feature("my_feature", self.my_feature)

        # Initialize action space
        self.action = Action()
        lb = np.array([-1.0], dtype=np.float32)
        ub = np.array([1.0], dtype=np.float32)
        self.action.set_specs(dim_c=1, range=(lb, ub))

    def observe(self) -> Observation:
        """Generate observation from current state."""
        return Observation(
            local={"state": self.state.vector()},
            timestamp=0.0
        )

    def act(self, observation: Observation, **kwargs):
        """Process observation and return action."""
        # Your control logic here
        self.action.sample()
        return self.action

    def update_state(self):
        """Update internal state based on action."""
        # Update your feature based on action
        self.my_feature.value += self.action.c[0] * 0.1

    def reset(self, seed=None):
        """Reset agent state."""
        self.my_feature.value = 0.0
        self.action.reset()
```

#### 7. Create a Custom Environment

Create `my_domain/envs/my_env.py`:

```python
"""Example custom environment using HERON framework."""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from my_domain.agents.my_agent import MyAgent


class MyEnvironment(ParallelEnv):
    """Custom multi-agent environment."""

    metadata = {"name": "my_environment_v0"}

    def __init__(self, config: dict = None):
        config = config or {}
        self.max_steps = config.get("max_steps", 100)
        self._step = 0

        # Create agents
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()

        self._agents = {
            agent_id: MyAgent(agent_id, initial_value=0.0)
            for agent_id in self.possible_agents
        }

        # Define spaces
        self.observation_spaces = {
            agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            for agent_id in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        """Reset environment."""
        self._step = 0
        self.agents = self.possible_agents.copy()

        for agent in self._agents.values():
            agent.reset(seed=seed)

        observations = {
            agent_id: agent.observe().local["state"]
            for agent_id, agent in self._agents.items()
        }
        infos = {agent_id: {} for agent_id in self.agents}

        return observations, infos

    def step(self, actions):
        """Execute one step."""
        self._step += 1

        # Apply actions
        for agent_id, action in actions.items():
            self._agents[agent_id].action.c[:] = action
            self._agents[agent_id].update_state()

        # Get observations
        observations = {
            agent_id: agent.observe().local["state"]
            for agent_id, agent in self._agents.items()
        }

        # Compute rewards (example: minimize absolute value)
        rewards = {
            agent_id: -abs(agent.my_feature.value)
            for agent_id, agent in self._agents.items()
        }

        # Check termination
        terminated = self._step >= self.max_steps
        terminateds = {agent_id: terminated for agent_id in self.agents}
        terminateds["__all__"] = terminated

        truncateds = {agent_id: False for agent_id in self.agents}
        truncateds["__all__"] = False

        infos = {agent_id: {} for agent_id in self.agents}

        return observations, rewards, terminateds, truncateds, infos

    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id):
        return self.action_spaces[agent_id]
```

#### 8. Create an Example Script

Create `examples/train_my_agent.py`:

```python
"""Example training script."""

from my_domain.envs.my_env import MyEnvironment


def main():
    print("Creating environment...")
    env = MyEnvironment({"max_steps": 100})

    print("Running random policy...")
    obs, info = env.reset(seed=42)

    total_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}

    for step in range(100):
        # Random actions
        actions = {
            agent_id: env.action_space(agent_id).sample()
            for agent_id in env.agents
        }

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        if terminateds["__all__"]:
            break

    print(f"\nTotal rewards: {total_rewards}")
    print("Done!")


if __name__ == "__main__":
    main()
```

#### 9. Run Your Example

```bash
python examples/train_my_agent.py
```

---

## Included Case Study

The repository includes a complete **Power Grid** case study demonstrating HERON applied to multi-agent microgrid control with PandaPower integration.

📖 **See [case_studies/power/README.md](case_studies/power/README.md) for full documentation**, including:
- IEEE 13, 34, 123-bus test networks
- Device models (Generator, ESS, Transformer)
- Multi-agent environments with pre-configured setups
- MAPPO training examples with RLlib

Quick install:
```bash
pip install -e ".[powergrid]"      # Power grid support
pip install -e ".[all]"            # Full installation with RL
```

### Adding a New Case Study to This Repo

To contribute a new domain case study (e.g., robotics, traffic):

1. Copy the `case_studies/power/` structure as a template
2. Update `pyproject.toml` to include your package path:
   ```toml
   [tool.setuptools.packages.find]
   where = [".", "case_studies/power", "case_studies/your_domain"]
   ```
3. Add optional dependencies under `[project.optional-dependencies]` if needed

> 💡 **For standalone projects** (separate repository), use `make new-project` or follow [Setting Up Your Own Project](#setting-up-your-own-project).

---

## Documentation

HERON provides comprehensive documentation for different learning paths:

### Getting Started
- **[Hello World Example](examples/00_hello_world.py)**: Minimal runnable example without domain dependencies
- **[Getting Started Guide](docs/source/getting_started.md)**: Installation and first steps
- **[Key Concepts](docs/source/key_concepts.md)**: Core abstractions with code examples

### User Guides
- **[Basic Concepts](docs/source/user_guide/basic_concepts.md)**: Agent hierarchy, features, protocols
- **[Event-Driven Execution](docs/source/user_guide/event_driven_execution.md)**: Testing with realistic timing
- **[Centralized vs Distributed](docs/source/user_guide/centralized_vs_distributed.md)**: Execution mode comparison

### Developer Guides
- **[Architecture](docs/source/developer_guide/architecture.md)**: System design and components
- **[Extending Agents](docs/source/developer_guide/extending_agents.md)**: Creating custom agents
- **[Custom Protocols](docs/source/developer_guide/custom_protocols.md)**: Building coordination protocols

### Reference
- **[Glossary](docs/source/glossary.md)**: Definitions of HERON terminology
- **[Scheduling Module](heron/scheduling/README.md)**: Event-driven scheduling details

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev,all]"

# Run heron core tests
pytest tests/ -v

# Run power grid case study tests
pytest case_studies/power/tests/ -v

# Run all tests
pytest tests/ case_studies/power/tests/ -v

# Run with coverage
pytest tests/ case_studies/power/tests/ --cov=heron --cov=powergrid --cov-report=html
```

### Code Style

```bash
# Format code
black heron/ case_studies/ tests/

# Lint code
ruff check heron/ case_studies/ tests/

# Type check
mypy heron/ case_studies/power/powergrid/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Citation

If you use HERON in your research, please cite:
TBD

## Contact

- **Issues**: [GitHub Issues](https://github.com/Criss-Wang/PowerGym/issues)
- **Email**: zhenlin.wang.criss@gmail.com

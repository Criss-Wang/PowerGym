# HERON: Hierarchical Environments for Realistic Observability in Networks

A **domain-agnostic Multi-Agent Reinforcement Learning (MARL) framework** with a power systems case study. HERON provides modular abstractions for hierarchical agents, coordination protocols, and realistic observability constraints.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Setting Up Your Own Project](#setting-up-your-own-project)
- [Running Examples](#running-examples)
- [Power Grid Domain (Case Study)](#power-grid-domain-case-study)
- [Development](#development)

---

## Features

### Core Framework (`heron/`)
- **Hierarchical Agent System**: Field (L1), Coordinator (L2), and System (L3) agents
- **Feature-based State Representation**: Composable `FeatureProvider` system with 4 visibility levels
- **Coordination Protocols**: Vertical (Setpoint, Price Signal) and Horizontal (P2P Trading, Consensus)
- **Message Broker System**: InMemoryBroker with extensible interface for Kafka/RabbitMQ
- **Flexible Action System**: `Action` class with scale/unscale for normalized [-1, 1] control
- **Mixed Action Spaces**: Continuous (`Box`) and discrete (`Discrete`/`MultiDiscrete`)

### Power Grid Domain (`powergrid/`)
- **PandaPower Integration**: Seamless network modeling and power flow
- **Device Models**: Generator, ESS, Transformer with realistic dynamics
- **IEEE Test Networks**: IEEE 13, 34, 123-bus systems included
- **Safety Framework**: Penalties for voltage, loading, SOC, power factor violations
- **Cost Helpers**: Quadratic, piecewise linear, ramping costs

### RL Integration
- Works with **RLlib** (MAPPO/IPPO), **Stable-Baselines3**, and custom agents
- **PettingZoo ParallelEnv** for multi-agent environments
- Centralized and distributed execution modes

---

## Architecture Overview

```
heron/                          # Domain-agnostic MARL framework
├── agents/                     # Hierarchical agent abstractions
│   ├── base.py                 # Agent base class
│   ├── field_agent.py          # Level 1 (device-level)
│   ├── coordinator_agent.py    # Level 2 (area coordinator)
│   └── system_agent.py         # Level 3 (system operator)
├── core/                       # Core abstractions
│   ├── action.py               # Action with continuous/discrete support
│   ├── observation.py          # Observation with local/global/messages
│   ├── state.py                # State with FeatureProvider composition
│   └── policies.py             # Policy abstractions
├── protocols/                  # Coordination protocols
│   ├── base.py                 # Protocol interfaces
│   ├── vertical.py             # Setpoint, Price Signal protocols
│   └── horizontal.py           # P2P Trading, Consensus protocols
├── messaging/                  # Message broker system
│   ├── base.py                 # MessageBroker interface
│   └── memory.py               # InMemoryBroker implementation
├── features/                   # Feature system
│   └── base.py                 # FeatureProvider with visibility levels
└── envs/                       # Base environment interfaces

powergrid/                      # Power systems case study
├── agents/                     # Power-specific agents
│   ├── power_grid_agent.py     # Coordinator with PandaPower
│   ├── generator.py            # Dispatchable generator
│   ├── storage.py              # Energy storage system
│   └── proxy_agent.py          # System agent for distributed mode
├── features/                   # Power-specific features
│   ├── electrical.py           # P, Q, voltage features
│   ├── network.py              # Bus voltages, line flows
│   └── storage.py              # SOC, energy features
├── networks/                   # IEEE/CIGRE test networks
│   ├── ieee13.py
│   ├── ieee34.py
│   └── ieee123.py
├── envs/                       # Power environments
│   ├── multi_agent/            # Multi-agent environments
│   └── single_agent/           # Single-agent environments
└── utils/                      # Power-specific utilities
    ├── cost.py
    └── safety.py
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/hepengli/powergrid.git
cd powergrid
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
python -c "import heron; import powergrid; print('Installation successful!')"

# Run tests (optional)
pytest tests/ -v
```

---

## Quick Start

### Single Microgrid Environment

```python
from powergrid.envs.single_agent.ieee13_mg import IEEE13Env

# Create environment
env = IEEE13Env({"episode_length": 24, "train": True})
obs, info = env.reset()

# Run simulation
for _ in range(24):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.2f}, Converged: {info.get('converged')}")
```

### Multi-Agent Environment (Centralized)

```python
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids

# Create multi-agent environment
env_config = {
    "centralized": True,
    "max_episode_steps": 24,
    "train": True,
}
env = MultiAgentMicrogrids(env_config)
obs_dict, info = env.reset()

# Each agent acts independently
for _ in range(24):
    actions = {agent_id: env.action_spaces[agent_id].sample()
               for agent_id in env.agents}
    obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)
```

---

## Setting Up Your Own Project

This section guides you through creating a new project that uses HERON as a dependency.

### Project Structure

Create a new directory with this structure:

```
my_project/
├── .venv/                      # Virtual environment (created by python -m venv)
├── my_domain/                  # Your domain-specific code
│   ├── __init__.py
│   ├── agents/                 # Your custom agents
│   │   ├── __init__.py
│   │   └── my_agent.py
│   ├── features/               # Your custom features
│   │   ├── __init__.py
│   │   └── my_features.py
│   ├── envs/                   # Your custom environments
│   │   ├── __init__.py
│   │   └── my_env.py
│   └── utils/                  # Your utilities
│       └── __init__.py
├── data/                       # Your data files
│   └── my_data.pkl
├── examples/                   # Your example scripts
│   └── train_my_agent.py
├── tests/                      # Your tests
│   └── test_my_env.py
├── pyproject.toml              # Project configuration
└── README.md
```

### Step-by-Step Setup

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
    "heron-marl @ git+https://github.com/hepengli/powergrid.git",
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
mkdir -p my_domain/{agents,features,envs,utils}
mkdir -p data examples tests

# Create __init__.py files
touch my_domain/__init__.py
touch my_domain/agents/__init__.py
touch my_domain/features/__init__.py
touch my_domain/envs/__init__.py
touch my_domain/utils/__init__.py
```

#### 5. Install Dependencies

```bash
# Install your project with HERON
pip install -e ".[heron]"

# Or install HERON directly from source (for development)
pip install -e /path/to/powergrid[all]
```

#### 6. Create a Custom Agent

Create `my_domain/agents/my_agent.py`:

```python
"""Example custom agent using HERON framework."""

from heron.agents.base import Agent
from heron.core.action import Action
from heron.core.observation import Observation
from heron.core.state import FieldAgentState
from heron.features.base import FeatureProvider


class MyCustomFeature(FeatureProvider):
    """Custom feature for your domain."""

    def __init__(self, value: float = 0.0):
        super().__init__(visibility=["owner"])
        self.value = value

    def vector(self):
        import numpy as np
        return np.array([self.value], dtype=np.float32)

    def dim(self):
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
        import numpy as np
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

## Running Examples

The repository includes several ready-to-run examples:

```bash
# Activate virtual environment
source .venv/bin/activate

# Example 1: Single microgrid with centralized control
python examples/01_single_microgrid_basic.py

# Example 2: Multi-microgrid with P2P trading
python examples/02_multi_microgrid_p2p.py

# Example 3: Price coordination protocol
python examples/03_price_coordination.py

# Example 4: Custom device implementation
python examples/04_custom_device.py

# Example 5: MAPPO training (requires ray[rllib])
python examples/05_mappo_training.py --test

# Example 6: Distributed mode with proxy agent
python examples/06_distributed_mode_with_proxy.py
```

### MAPPO Training with RLlib

```bash
# Install RLlib dependencies
pip install -e ".[multi_agent]"

# Run training
python examples/05_mappo_training.py --iterations 100 --num-workers 2

# Quick test run
python examples/05_mappo_training.py --test
```

---

## Power Grid Domain (Case Study)

### Example Networks

The repository includes standard IEEE test systems:

#### IEEE 13-Bus System
<img src="docs/images/ieee13.png" alt="IEEE 13 Bus System" width="500"/>

#### IEEE 34-Bus System
<img src="docs/images/ieee34.png" alt="IEEE 34 Bus System" width="700"/>

### Creating a Power Grid Environment

```python
from powergrid.envs.multi_agent.networked_grid_env import NetworkedGridEnv
from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.networks.ieee13 import IEEE13Bus
from heron.protocols.vertical import SetpointProtocol


class MyPowerGridEnv(NetworkedGridEnv):
    """Custom power grid environment."""

    def _build_net(self):
        # Create IEEE 13-bus network
        net = IEEE13Bus("MG1")

        # Create grid agent with devices
        mg_agent = PowerGridAgent(
            net=net,
            grid_config={
                "name": "MG1",
                "base_power": 1.0,
                "devices": [
                    {
                        "type": "Generator",
                        "name": "gen1",
                        "device_state_config": {
                            "bus": "Bus 633",
                            "p_max_MW": 2.0,
                            "p_min_MW": 0.5,
                        },
                    },
                    {
                        "type": "ESS",
                        "name": "ess1",
                        "device_state_config": {
                            "bus": "Bus 634",
                            "e_capacity_MWh": 5.0,
                            "p_max_MW": 1.0,
                        },
                    },
                ],
            },
            protocol=SetpointProtocol(),
        )

        self.possible_agents = ["MG1"]
        self.agent_dict = {"MG1": mg_agent}
        self.net = net

        return net

    def _reward_and_safety(self):
        rewards = {aid: -agent.cost for aid, agent in self.agent_dict.items()}
        safety = {aid: agent.safety for aid, agent in self.agent_dict.items()}
        return rewards, safety
```

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev,all]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/core/test_actions.py -v

# Run with coverage
pytest tests/ --cov=heron --cov=powergrid --cov-report=html
```

### Code Style

```bash
# Format code
black heron/ powergrid/ tests/

# Lint code
ruff check heron/ powergrid/ tests/

# Type check
mypy heron/ powergrid/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Citation

If you use HERON in your research, please cite:

```bibtex
@article{heron2024,
  title={HERON: Hierarchical Environments for Realistic Observability in Networks},
  author={Li, Hepeng and Wang, Zhenlin},
  year={2024}
}
```

## Contact

- **Issues**: [GitHub Issues](https://github.com/hepengli/powergrid/issues)
- **Email**: hepeng.li@maine.edu

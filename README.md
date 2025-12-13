# PowerGrid Gym Environment

A lightweight, production-style **Gymnasium** environment for **power grid control**, built on [pandapower](https://www.pandapower.org/).  
It provides modular device models (DG, RES, ESS, Shunt, Transformer, Grid) with clean action/observation spaces, centralized safety metrics, and pluggable rewards — designed for Reinforcement Learning (RL) and Multi-Agent RL research.

---

## Highlights

- ⚡ **Hierarchical agent system**: `DeviceAgent` and `GridAgent` with modular state/action abstractions
- 🔌 **Pandapower integration** with idempotent device → network attachment
- 🧩 **Feature-based state representation**: Composable `FeatureProvider` system with visibility rules
- 👥 **Multi-agent environments**: PettingZoo ParallelEnv for networked microgrids
- 🌐 **Distributed execution**: Message-based communication between agents (centralized & distributed modes)
- 🎛️ **Mixed action spaces**: continuous (`Box`) and discrete (`Discrete` / `MultiDiscrete`) combined in a `Dict`
- 🔄 **Flexible action system**: `Action` class with scale/unscale for normalized [-1, 1] control
- 🛡️ **Safety framework**: penalties for over-rating, power factor, SOC, voltage, line loading, etc.
- 💰 **Cost helpers**: quadratic, piecewise linear, ramping, tap wear, energy settlement
- 🎯 **Coordination protocols**: Price signals, setpoint control, P2P trading, consensus
- 📨 **Message broker system**: InMemoryBroker with extensible interface for Kafka/RabbitMQ
- 👁️ **Observability control**: Multi-level visibility system (public, owner, system, upper_level)
- ✅ **Comprehensive tests** for devices, agents, and environments
- 🧪 **RL-ready**: works with Stable-Baselines3, RLlib (MAPPO/PPO), and custom agents

---

## Installation

### Option 1: Install from PyPI (coming soon)

```bash
pip install powergrid
```

### Option 2: Install from source

```bash
# Clone the repository
git clone https://github.com/hepengli/powergrid.git
cd powergrid

# Install in editable mode for development
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Option 3: Python venv (recommended for isolation)

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Upgrade pip
pip install -U pip

# Install from source
pip install -e .
```

# Quick Start

## Single-Agent Environment

```python
from powergrid.envs.single_agent.ieee13_mg import IEEE13Env

# Create environment: agent acts in [-1,1] for the continuous part
env = IEEE13Env({"episode_length": 24, "train": True})
obs, info = env.reset()

# Take a random step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("reward=", reward, "converged=", info.get("converged"))
```

## Multi-Agent Environment (Centralized)

```python
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids

# Create multi-agent environment (3 networked microgrids)
env_config = {
    "centralized": True,  # Traditional multi-agent RL
    "max_episode_steps": 96,
    "train": True,
    "dataset_path": "data/data2023-2024.pkl"
}
env = MultiAgentMicrogrids(env_config)
obs_dict, info = env.reset()

# Each agent acts independently
actions = {agent_id: env.action_spaces[agent_id].sample()
           for agent_id in env.agents}
obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)
```

## Multi-Agent Environment (Distributed)

```python
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids

# Create distributed multi-agent environment
env_config = {
    "centralized": False,  # Distributed execution with message broker
    "message_broker": "in_memory",  # InMemoryBroker (can extend to Kafka)
    "max_episode_steps": 96,
    "train": True,
    "dataset_path": "data/data2023-2024.pkl"
}
env = MultiAgentMicrogrids(env_config)

# Agents communicate via message broker, never access network directly
obs_dict, info = env.reset()
actions = {agent_id: policy(obs_dict[agent_id]) for agent_id in env.agents}
obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)
```

**Key Difference**:
- **Centralized**: Agents directly access PandaPower network (traditional MARL)
- **Distributed**: Agents communicate via messages only (realistic distributed control)

## Action Space

PowerGrid uses a flexible `Action` dataclass that supports:

- **Continuous actions (`c`)**: Device setpoints in physical units (MW, MVAr)
- **Discrete actions (`d`)**: Categorical choices (e.g., transformer taps, on/off status)

Actions are automatically exposed as Gymnasium spaces:
- Pure continuous → `Box`
- Pure discrete → `Discrete` or `MultiDiscrete`
- Mixed → `Dict({"c": Box, "d": Discrete|MultiDiscrete})`

The `Action` class provides built-in normalization:
```python
# Agent outputs normalized action in [-1, 1]
normalized_action = agent.act(obs)

# Action.unscale() converts to physical units
action.unscale(normalized_action)  # Now action.c contains physical values

# Or use action.scale() to normalize existing physical values
physical_action = action.c
normalized = action.scale()  # Returns [-1, 1] normalized version
```

## Example Networks

This repository includes standard IEEE test systems used for demonstration and validation.  
Below are the single-line diagrams of two networks:

### IEEE 13-Bus System

<img src="docs/images/ieee13.png" alt="IEEE 13 Bus System" width="500"/>

### IEEE 34-Bus System
<img src="docs/images/ieee34.png" alt="IEEE 34 Bus System" width="700"/>

## Example algorithm running
Using `Ray` and *MAPPO*
```bash
python examples/train_mappo_microgrids.py --iterations 5 --num-workers 2 --checkpoint-freq 5 --no-cuda
```
# Power Grid Domain - HERON Case Study

This case study demonstrates HERON applied to power systems with multi-agent microgrid control, focusing on CTDE (Centralized Training with Decentralized Execution) for collective optimization.

---

## Table of Contents

- [Overview](#overview)
- [HERON Framework Integration](#heron-framework-integration)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CTDE Workflow](#ctde-workflow)
- [Running Examples](#running-examples)
- [MAPPO Training](#mappo-training-with-rllib)
- [Environment Setups](#environment-setups)
- [Custom Environments](#creating-a-custom-power-grid-environment)

---

## Overview

| Component | Description |
|-----------|-------------|
| **Networks** | IEEE 13, 34, 123-bus test systems via PandaPower |
| **Devices** | Generator, ESS (Energy Storage), Transformer |
| **Agents** | `PowerGridAgent` (coordinator), device agents (field level) |
| **Features** | Electrical (P, Q, V), Storage (SOC), Network metrics |
| **Training** | CTDE with shared rewards for cooperative behavior |
| **Testing** | Event-driven mode with realistic timing constraints |

---

## HERON Framework Integration

This case study fully integrates with the HERON multi-agent framework:

### 3-Level Agent Hierarchy

```
SystemAgent (Level 3) - DSO / System Coordinator
    └── CoordinatorAgent / PowerGridAgent (Level 2) - Microgrid Controllers
            └── FieldAgent / DeviceAgent (Level 1) - Generators, ESS, etc.
```

### Execution Modes

| Mode | Training | Testing |
|------|----------|---------|
| **Option A (Synchronous)** | ✓ Used for CTDE training | - |
| **Option B (Event-Driven)** | - | ✓ Used for robustness testing |

### Timing Parameters

All agents support configurable timing for realistic simulation:

- `tick_interval`: How often an agent steps (e.g., 1s for devices, 60s for grid)
- `obs_delay`: Observation latency (agent sees state from t - obs_delay)
- `act_delay`: Action effect delay (action takes effect at t + act_delay)
- `msg_delay`: Communication latency (messages arrive after msg_delay)

---

## Architecture

```
case_studies/power/
├── powergrid/                  # Python package
│   ├── agents/                 # Power-specific agents (extends HERON)
│   │   ├── power_grid_agent.py # CoordinatorAgent with PandaPower integration
│   │   ├── device_agent.py     # FieldAgent base for power devices
│   │   ├── generator.py        # Dispatchable generator device
│   │   ├── storage.py          # Energy storage system (ESS)
│   │   └── proxy_agent.py      # ProxyAgent for distributed mode
│   │
│   ├── core/                   # Extensions to heron.core
│   │   ├── features/           # Power-specific features
│   │   │   ├── electrical.py   # P, Q, voltage features
│   │   │   ├── network.py      # Bus voltages, line flows
│   │   │   ├── storage.py      # SOC, energy capacity
│   │   │   └── power_limits.py # Power limit features
│   │   │
│   │   └── state/              # Power-specific state
│   │       └── state.py        # Device and grid state classes
│   │
│   ├── networks/               # IEEE/CIGRE test networks
│   │   ├── ieee13.py           # IEEE 13-bus feeder
│   │   ├── ieee34.py           # IEEE 34-bus feeder
│   │   └── ieee123.py          # IEEE 123-bus feeder
│   │
│   ├── envs/                   # Power environments
│   │   ├── networked_grid_env.py      # Base (extends PettingZooParallelEnv)
│   │   └── multi_agent_microgrids.py  # Multi-microgrid environment
│   │
│   └── setups/                 # Environment setups
│       ├── loader.py           # Setup loading utilities
│       └── ieee34_ieee13/      # Example setup
│           ├── config.yml      # Environment configuration
│           └── data.pkl        # Time series data
│
├── examples/                   # Example scripts
│   ├── 01_single_microgrid_basic.py
│   ├── 04_custom_device.py
│   ├── 05_mappo_training.py       # CTDE training with MAPPO
│   ├── 06_distributed_mode_with_proxy.py
│   ├── 07_event_driven_mode.py    # Event-driven testing
│   └── 08_ctde_e2e_validation.py  # E2E validation
│
└── tests/                      # Power grid tests
```

---

## Installation

```bash
# Install with power grid support
pip install -e ".[powergrid]"

# Or full installation with RL support
pip install -e ".[all]"
```

---

## Quick Start

### Multi-Agent Microgrids with CTDE

```python
from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.setups.loader import load_setup

# Load environment configuration
env_config = load_setup("ieee34_ieee13")
env_config.update({
    "centralized": True,       # Training mode (Option A)
    "share_reward": True,      # Shared rewards for cooperation
    "max_episode_steps": 24,
    "train": True,
})
env = MultiAgentMicrogrids(env_config)
obs_dict, info = env.reset()

# Training loop with shared rewards
for _ in range(24):
    actions = {agent_id: env.action_spaces[agent_id].sample()
               for agent_id in env.agents}
    obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)

# Get collective metrics
metrics = env.get_collective_metrics()
print(f"Total Cost: {metrics['total_cost']:.2f}")
print(f"Cooperation Score: {metrics['cooperation_score']:.3f}")
```

---

## CTDE Workflow

The recommended workflow for CTDE (Centralized Training with Decentralized Execution):

### 1. Training Phase (Centralized - Option A)

```python
# Configure for centralized training
env_config['centralized'] = True
env_config['share_reward'] = True  # Encourages cooperation

env = MultiAgentMicrogrids(env_config)
# Train with MAPPO/IPPO using RLlib
```

### 2. Testing Phase (Decentralized - Option B)

```python
# Configure for decentralized testing
env_config['centralized'] = False
env_config['share_reward'] = False  # Individual rewards for evaluation

env = MultiAgentMicrogrids(env_config)
env.configure_agents_for_distributed()  # Setup message broker

# Test trained policy with realistic timing constraints
obs, info = env.reset()
while not done:
    actions = get_actions_from_trained_policy(obs)
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

### 3. Validation

```bash
# Run comprehensive E2E validation
python examples/08_ctde_e2e_validation.py
```

---

## Running Examples

```bash
# From project root
source .venv/bin/activate
cd case_studies/power

# Example 1: Single microgrid basics
python examples/01_single_microgrid_basic.py

# Example 4: Custom device implementation
python examples/04_custom_device.py

# Example 5: MAPPO training (requires ray[rllib])
python examples/05_mappo_training.py --test

# Example 5: MAPPO with event-driven validation
python examples/05_mappo_training.py --iterations 100 --event-driven-test

# Example 6: Distributed mode with proxy agent
python examples/06_distributed_mode_with_proxy.py

# Example 7: Event-driven mode demonstration
python examples/07_event_driven_mode.py

# Example 8: CTDE E2E validation
python examples/08_ctde_e2e_validation.py
```

---

## MAPPO Training with RLlib

```bash
# Install RLlib dependencies
pip install -e ".[multi_agent]"

cd case_studies/power

# Train with shared rewards (MAPPO for cooperation)
python examples/05_mappo_training.py --iterations 100 --share-reward

# Train with independent policies (IPPO)
python examples/05_mappo_training.py --iterations 100 --independent-policies

# Quick test run
python examples/05_mappo_training.py --test

# Training with event-driven validation
python examples/05_mappo_training.py --iterations 100 --event-driven-test
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--iterations` | 3 | Number of training iterations |
| `--share-reward` | True | Enable shared rewards (CTDE) |
| `--independent-policies` | False | Use IPPO instead of MAPPO |
| `--event-driven-test` | False | Run event-driven validation after training |
| `--test` | False | Quick test mode (3 iterations) |

---

## Environment Setups

A **setup** is a complete environment definition containing configuration and data. Each setup is a subdirectory under `powergrid/setups/` with:
- `config.yml`: Environment configuration (agent hierarchy, network topology, device parameters)
- `data.pkl`: Time series data (load profiles, prices, renewable generation)

### Loading a Setup

```python
from powergrid.setups.loader import load_setup, get_available_setups

# List available setups
print(get_available_setups())  # ['ieee34_ieee13']

# Load a setup
config = load_setup("ieee34_ieee13")
```

---

## Creating a Custom Power Grid Environment

```python
from powergrid.envs.networked_grid_env import NetworkedGridEnv
from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.networks.ieee13 import IEEE13Bus
from heron.protocols.vertical import SetpointProtocol


class MyPowerGridEnv(NetworkedGridEnv):
    """Custom power grid environment with HERON integration."""

    def _build_agents(self):
        return {}  # Built in _build_net

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
            # HERON timing parameters
            tick_interval=60.0,
            obs_delay=1.0,
            act_delay=2.0,
            msg_delay=0.5,
        )

        self.agent_dict = {"MG1": mg_agent}
        return net

    def _reward_and_safety(self):
        rewards = {aid: -agent.cost for aid, agent in self.agent_dict.items()}
        safety = {aid: agent.safety for aid, agent in self.agent_dict.items()}
        return rewards, safety
```

---

## Example Networks

The repository includes standard IEEE test systems:

### IEEE 13-Bus System
<img src="../../docs/images/ieee13.png" alt="IEEE 13 Bus System" width="500"/>

### IEEE 34-Bus System
<img src="../../docs/images/ieee34.png" alt="IEEE 34 Bus System" width="700"/>

---

## Further Reading

- [HERON Framework Documentation](../../docs/)
- [HERON Integration Guide](docs/HERON_INTEGRATION.md)
- [PandaPower Documentation](https://pandapower.readthedocs.io/)

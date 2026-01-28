# Power Grid Domain - HERON Case Study

The `powergrid/` package demonstrates HERON applied to power systems with multi-agent microgrid control.

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Example Networks](#example-networks)
- [Running Examples](#running-examples)
- [MAPPO Training](#mappo-training-with-rllib)
- [Custom Environments](#creating-a-custom-power-grid-environment)

---

## Architecture

```
powergrid/                      # Power systems case study
├── agents/                     # Power-specific agents
│   ├── power_grid_agent.py     # Coordinator with PandaPower integration
│   ├── device_agent.py         # Base for power device agents
│   ├── generator.py            # Dispatchable generator device
│   ├── storage.py              # Energy storage system (ESS)
│   ├── transformer.py          # Transformer with tap changer
│   └── proxy_agent.py          # System agent for distributed mode
│
├── features/                   # Power-specific features
│   ├── electrical.py           # P, Q, voltage features
│   ├── network.py              # Bus voltages, line flows, network metrics
│   ├── storage.py              # SOC, energy capacity features
│   ├── power_limits.py         # Power limit features
│   └── thermal.py              # Thermal constraint features
│
├── networks/                   # IEEE/CIGRE test networks
│   ├── ieee13.py               # IEEE 13-bus feeder
│   ├── ieee34.py               # IEEE 34-bus feeder
│   ├── ieee123.py              # IEEE 123-bus feeder
│   └── cigre_lv.py             # CIGRE low-voltage network
│
├── envs/                       # Power environments
│   ├── multi_agent/            # Multi-agent environments
│   │   ├── networked_grid_env.py
│   │   └── multi_agent_microgrids.py
│   └── single_agent/           # Single-agent environments
│       ├── ieee13_mg.py
│       └── ieee34_mg.py
│
├── optimization/               # Power system optimization
│   └── misocp.py               # Mixed-integer SOCP solver
│
└── utils/                      # Power-specific utilities
    ├── cost.py                 # Cost functions (quadratic, piecewise)
    ├── safety.py               # Safety penalties (voltage, SOC, loading)
    └── phase.py                # Phase utilities for 3-phase systems
```

## Overview

| Component | Description |
|-----------|-------------|
| **Networks** | IEEE 13, 34, 123-bus test systems via PandaPower |
| **Devices** | Generator, ESS (Energy Storage), Transformer |
| **Agents** | `PowerGridAgent` (coordinator), device agents (field level) |
| **Features** | Electrical (P, Q, V), Storage (SOC), Network metrics |

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

### Single Microgrid

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

### Multi-Agent Microgrids

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

## Example Networks

The repository includes standard IEEE test systems:

### IEEE 13-Bus System
<img src="../docs/images/ieee13.png" alt="IEEE 13 Bus System" width="500"/>

### IEEE 34-Bus System
<img src="../docs/images/ieee34.png" alt="IEEE 34 Bus System" width="700"/>

---

## Running Examples

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

---

## MAPPO Training with RLlib

```bash
# Install RLlib dependencies
pip install -e ".[multi_agent]"

# Run training
python examples/05_mappo_training.py --iterations 100 --num-workers 2

# Quick test run
python examples/05_mappo_training.py --test
```

---

## Creating a Custom Power Grid Environment

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

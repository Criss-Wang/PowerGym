# GridAges Case Study

Multi-agent microgrid control using Heron framework, migrated from the [GridAges](https://github.com/hepengli/GridAges) codebase.

## Overview

This case study implements a hierarchical multi-agent reinforcement learning system for coordinated dispatch of networked microgrids. The system models:

- **3 Microgrids** (MG1, MG2, MG3) connected to a Distribution System Operator (DSO)
- **Devices per microgrid**:
  - Energy Storage System (ESS): 2 MWh capacity, ±0.5 MW
  - Distributed Generator (DG): 0.5-0.66 MW diesel generator
  - Solar PV: 0.1 MW
  - Wind Turbine: 0.1 MW
  - Grid connection with dynamic pricing

## Architecture

```
MicrogridEnv (MultiAgentEnv)
  └── SystemAgent
        ├── MG1_FieldAgent (controls ESS, DG, PV, Wind)
        ├── MG2_FieldAgent (controls ESS, DG, PV, Wind)
        └── MG3_FieldAgent (controls ESS, DG, PV, Wind)
```

## Structure

- **`features/`**: Device feature providers (ESS, DG, RES, Grid)
- **`agents/`**: Microgrid field agents
- **`envs/`**: Multi-agent microgrid environment
- **`tests/`**: Unit and integration tests

## Usage

### Training
```python
from case_studies.grid_age.envs.microgrid_env import MicrogridEnv
from case_studies.grid_age.train import train_microgrid_ctde

env = MicrogridEnv(num_microgrids=3)
policies = train_microgrid_ctde(env, num_episodes=100)
```

### Testing
```python
# Event-driven testing with trained policies
env.set_agent_policies(policies)
result = env.run_event_driven(t_end=24.0)  # 24-hour episode
```

## References

- Original GridAges: https://github.com/hepengli/GridAges
- Hepeng Li, "Optimal Operation of Networked Microgrids With Distributed Multi-Agent Reinforcement Learning," IEEE PES General Meeting 2024

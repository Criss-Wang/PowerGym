# HERON: Hierarchical Environments for Realistic Observability in Networks

A **domain-agnostic Multi-Agent Reinforcement Learning (MARL) framework** for hierarchical control systems with realistic observability constraints.

---

## Why HERON?

| Challenge | HERON Solution |
|-----------|----------------|
| Flat agent structures don't scale | **3-level hierarchy**: Field → Coordinator → System |
| Full observability is unrealistic | **Visibility rules** control what each agent sees |
| Training ≠ deployment timing | **Dual-mode execution**: sync training, async testing |
| Domain-specific code everywhere | **Pluggable protocols** for any coordination pattern |

---

## Architecture at a Glance

```
                    ┌─────────────────┐
         L3        │   SystemAgent    │        System-wide coordination
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
 L2 │ Coordinator A │ │ Coordinator B │ │ Coordinator C │   Regional coordination
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
      ┌─────┴─────┐     ┌─────┴─────┐     ┌─────┴─────┐
      ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
 L1 ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐   Local units
    │F1 │ │F2 │ │F3 │ │F4 │ │F5 │ │F6 │ │F7 │ │F8 │ │F9 │
    └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
```

```
heron/
├── agents/      # Agent hierarchy (Field, Coordinator, System, Proxy)
├── core/        # Action, Observation, State, FeatureProvider, Policy
├── protocols/   # Vertical (Setpoint, Price) & Horizontal (Trading, Consensus)
├── envs/        # Environment base classes & framework adapters
├── messaging/   # Message broker (InMemoryBroker, extensible to Kafka/RabbitMQ)
└── scheduling/  # Event-driven simulation (TickConfig, EventScheduler)
```

---

## Installation

```bash
# Clone and setup
git clone https://github.com/Criss-Wang/PowerGym.git
cd PowerGym
python3 -m venv .venv && source .venv/bin/activate

# Choose your installation
pip install -e .                    # Core framework only
pip install -e ".[powergrid]"       # + Power grid case study
pip install -e ".[multi_agent]"     # + RLlib, PettingZoo
pip install -e ".[all]"             # Everything
pip install -e ".[dev,all]"         # + dev tools (pytest, ruff)
```

---

## Quick Start (5 minutes)

### 1. Create an Agent with Visibility-Controlled State

```python
from heron.agents import FieldAgent
from heron.core import FeatureProvider
import numpy as np

class TemperatureFeature(FeatureProvider):
    visibility = ["owner", "upper_level"]  # Only owner and coordinator can see

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

class ThermostatAgent(FieldAgent):
    def set_action(self):
        self.action.set_specs(dim_c=1, range=(np.array([18.0]), np.array([26.0])))

    def set_state(self):
        self.state.features.append(TemperatureFeature())
```

### 2. Build a Hierarchy with Protocols

```python
from heron.agents import CoordinatorAgent, SystemAgent
from heron.protocols import SetpointProtocol, SystemProtocol

# Create field agents
thermostats = [ThermostatAgent(f"thermo_{i}") for i in range(3)]

# Coordinator manages field agents
coordinator = CoordinatorAgent(
    agent_id="zone_controller",
    protocol=SetpointProtocol(),
    subordinates={a.agent_id: a for a in thermostats}
)

# System manages coordinators
system = SystemAgent(
    agent_id="building_system",
    protocol=SystemProtocol(),
    coordinators={"zone_controller": coordinator}
)
```

### 3. Run Training (Synchronous) or Testing (Event-Driven)

```python
# Option A: Synchronous (Training)
obs = coordinator.observe(global_state)
coordinator.act(obs, upstream_action=joint_action)

# Option B: Event-Driven (Testing)
from heron.scheduling import TickConfig, JitterType

coordinator.tick_config = TickConfig.with_jitter(
    tick_interval=5.0,
    obs_delay=0.1,
    act_delay=0.2,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
)
env.run_event_driven(t_end=3600.0)  # 1 hour simulation
```

---

## Module Documentation

Each module has detailed documentation with examples and API reference:

| Module | Description | README |
|--------|-------------|--------|
| **agents** | Hierarchical agents (Field, Coordinator, System, Proxy) | [heron/agents/README.md](heron/agents/README.md) |
| **core** | Action, Observation, State, FeatureProvider, Policy | [heron/core/README.md](heron/core/README.md) |
| **protocols** | Vertical & horizontal coordination protocols | [heron/protocols/README.md](heron/protocols/README.md) |
| **envs** | Environment bases & framework adapters (PettingZoo, RLlib) | [heron/envs/README.md](heron/envs/README.md) |
| **messaging** | Message broker for distributed execution | [heron/messaging/README.md](heron/messaging/README.md) |
| **scheduling** | Event-driven simulation with timing/jitter | [heron/scheduling/README.md](heron/scheduling/README.md) |

---

## Execution Modes

HERON supports two execution modes for the **same** agents and environments:

| Mode | Use Case | Timing | API |
|------|----------|--------|-----|
| **Synchronous (Option A)** | RL Training | All agents step together | `agent.observe()` → `agent.act()` |
| **Event-Driven (Option B)** | Realistic Testing | Heterogeneous tick rates + delays | `agent.tick()` via `EventScheduler` |

**Why does this matter?**

RL training assumes all agents observe and act simultaneously with zero latency. Real systems don't work that way:
- A field sensor ticks every 100ms, a coordinator every 5s
- Observations are delayed (sensor latency, network)
- Actions take time to execute (actuator response)

Event-driven mode tests your trained policy under these realistic conditions without retraining. Configure `TickConfig` with timing parameters and optional jitter, then run the same environment with `EventScheduler`.

See [heron/scheduling/README.md](heron/scheduling/README.md) for full details.

---

## Case Study: Power Grid

A complete multi-agent microgrid control case study with PandaPower integration.

```bash
pip install -e ".[powergrid]"
```

**Includes:** IEEE 13/34/123-bus networks, device models (Generator, ESS, Transformer), MAPPO training examples.

**Full documentation:** [case_studies/power/README.md](case_studies/power/README.md)

---

## Setting Up Your Own Project

For new domains, HERON provides a project generator:

```bash
make new-project NAME=my_project DOMAIN=my_domain
cd my_project && pip install -e ".[heron,dev]"
```

Or manually create your domain package that imports from `heron.*`:

```
my_project/
├── my_domain/
│   ├── agents/      # Your custom agents extending FieldAgent, etc.
│   ├── envs/        # Your environments using HeronEnvCore
│   └── features/    # Your FeatureProviders
├── experiments/
└── pyproject.toml   # Depends on heron-marl
```

---

## Development

```bash
# Tests
pytest tests/ -v                              # Core tests
pytest case_studies/power/tests/ -v           # Case study tests

# Code quality
black heron/ && ruff check heron/             # Format & lint
mypy heron/                                   # Type check
```

---

## License

MIT License - see [LICENSE.txt](LICENSE.txt)

## Citation

If you use HERON in your research, please cite: TBD

## Contact

- **Issues**: [GitHub Issues](https://github.com/Criss-Wang/PowerGym/issues)
- **Email**: zhenlin.wang.criss@gmail.com

# HERON

**Hierarchical Environments for Realistic Observability in Networks**

A domain-agnostic multi-agent reinforcement learning (MARL) framework for building hierarchical control systems with realistic distributed execution.

---

## Key Features

### Domain-Agnostic Framework

HERON provides building blocks for any hierarchical multi-agent system:

- **Hierarchical Agents**: System → Coordinator → Field agent hierarchy
- **Feature-Based State**: Composable state representations with visibility control
- **Coordination Protocols**: Vertical (top-down) and horizontal (peer-to-peer)
- **Dual Execution Modes**: Centralized training, distributed deployment

### Dual Execution Modes

- **Centralized Mode**: Full observability for fast algorithm development
- **Distributed Mode**: Message-based coordination for realistic deployment

Switch modes with a single config line: `mode: centralized/distributed`

### Message Broker Architecture

- Abstract `MessageBroker` interface
- `InMemoryBroker` for local simulation
- Extensible to Kafka/Redis for production

### RL Integration

- PettingZoo `ParallelEnv` interface
- Compatible with RLlib (MAPPO, PPO)
- Stable-Baselines3 support via wrappers

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/heron.git
cd heron
pip install -e .
```

### HERON Core Example

```python
from heron.agents import CoordinatorAgent, FieldAgent
from heron.protocols.vertical import SetpointProtocol
from heron.messaging.memory import InMemoryBroker

# Create message broker
broker = InMemoryBroker()

# Create hierarchical agents
field_agent = FieldAgent(agent_id='device_1', level=1, broker=broker)
coordinator = CoordinatorAgent(
    agent_id='coordinator_1',
    level=2,
    subordinates=[field_agent],
    protocol=SetpointProtocol(),
    broker=broker
)

# Execution loop
obs = coordinator.observe(global_state)
action = coordinator.act(obs)
```

---

## Case Studies

### PowerGrid Control

A production-ready implementation for distributed power grid control.

```python
from powergrid.envs import MultiAgentMicrogrids

env = MultiAgentMicrogrids(config={
    'network': 'ieee13',
    'num_microgrids': 2,
    'mode': 'centralized'
})

obs, info = env.reset()
for step in range(96):
    actions = {agent: policy(o) for agent, o in obs.items()}
    obs, rewards, dones, truncs, info = env.step(actions)
```

See [Case Studies](use_cases/index) for full documentation.

---

## Architecture

```
HERON Framework (Domain-Agnostic)
├── heron/
│   ├── agents/          # Hierarchical agent abstractions
│   ├── core/            # State, Action, Observation, Feature
│   ├── protocols/       # Vertical & Horizontal coordination
│   ├── messaging/       # Message broker interface
│   └── envs/            # Base environment interface

└── Case Studies
    └── power/           # PowerGrid case study
        ├── agents/      # GridAgent, DeviceAgent
        ├── features/    # Electrical, Storage features
        ├── networks/    # IEEE test systems
        └── envs/        # MultiAgentMicrogrids
```

---

## Documentation

- [API Reference](api/index) - HERON core API
- [Case Studies](use_cases/index) - PowerGrid and other implementations
- [Protocols Guide](api/heron/protocols) - Coordination protocols

---

## Contributing

We welcome contributions:

- New coordination protocols
- Additional case studies
- Kafka/Redis broker implementations
- Documentation improvements

---

## License

MIT License

---

## Contact

**Authors**: Hepeng Li, Zhenlin Wang

---

## Citation

If you use HERON in your research, please cite:

```bibtex
@software{heron,
  author = {Li, Hepeng and Wang, Zhenlin},
  title = {HERON: Hierarchical Environments for Realistic Observability in Networks},
  year = {2025},
  url = {https://github.com/yourusername/heron}
}
```

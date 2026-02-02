# HERON Tutorials: Build a Case Study from Scratch

Learn HERON by building a complete multi-agent RL system for power grid control.

## What Makes HERON Different?

| Challenge | Traditional Approach | HERON Solution |
|-----------|---------------------|----------------|
| Observation filtering | Manual code per agent | `visibility = ['owner', 'upper_level']` |
| Coordination protocols | Hardcoded logic | Pluggable `SetpointProtocol`, `PriceSignalProtocol` |
| Realistic testing | Not supported | **Dual execution modes** (sync + event-driven) |
| Agent hierarchy | Manual implementation | Built-in `FieldAgent` → `CoordinatorAgent` |

## Prerequisites

```bash
pip install ray[rllib] pettingzoo gymnasium numpy
```

## Tutorial Overview

### Core Tutorials (Build a Complete System)

| Notebook | Topic | Time | What You'll Learn |
|----------|-------|------|-------------------|
| [01_understanding_heron](01_understanding_heron.ipynb) | Concepts | 5 min | HERON's agent-centric approach, key abstractions |
| [02_features_and_state](02_features_and_state.ipynb) | Features | 10 min | Building `FeatureProvider` for observations |
| [03_building_agents](03_building_agents.ipynb) | Agents | 15 min | `FieldAgent` (devices), `CoordinatorAgent` (hierarchy) |
| [04_building_environment](04_building_environment.ipynb) | Environment | 15 min | PettingZoo-compatible multi-agent env |
| [05_training_with_rllib](05_training_with_rllib.ipynb) | Training | 10 min | MAPPO training with RLlib |
| [06_event_driven_testing](06_event_driven_testing.ipynb) | Dual Mode | 15 min | Event-driven testing, `EventScheduler`, `TickConfig` |

### Advanced Tutorials (Customization & Extension)

| Notebook | Topic | Time | What You'll Learn |
|----------|-------|------|-------------------|
| [07_configuration_and_datasets](07_configuration_and_datasets.ipynb) | Configuration | 10 min | YAML configs, pickle datasets, time-series data |
| [08_custom_protocols](08_custom_protocols.ipynb) | Protocols | 10 min | Building coordination protocols (setpoint, price, consensus) |
| [09_adding_custom_devices](09_adding_custom_devices.ipynb) | Devices | 15 min | Creating custom device agents with `DeviceAgent` |

**Total time:** ~105 minutes (reading + coding)

## Quick Start

If you just want to see the end result:

```bash
# Run the simplified training example
jupyter notebook 05_training_with_rllib.ipynb
```

## What You'll Build

```
SimpleMicrogridEnv (Environment)
├── mg_0 (CoordinatorAgent)
│   ├── mg_0_bat (FieldAgent - Battery)
│   └── mg_0_gen (FieldAgent - Generator)
├── mg_1 (CoordinatorAgent)
│   ├── mg_1_bat (FieldAgent - Battery)
│   └── mg_1_gen (FieldAgent - Generator)
└── mg_2 (CoordinatorAgent)
    ├── mg_2_bat (FieldAgent - Battery)
    └── mg_2_gen (FieldAgent - Generator)
```

## Key HERON Contributions (Demonstrated in Tutorials)

### 1. Declarative Visibility (Tutorial 02)
No manual filtering—features declare who can see them:
```python
class BatterySOC(FeatureProvider):
    visibility = ['owner', 'upper_level']  # Automatic filtering
```

### 2. Agent-Centric Architecture (Tutorial 03)
Agents are first-class citizens with state, timing, and hierarchy:
```python
battery = SimpleBattery(agent_id='bat_1', upstream_id='mg_1', tick_interval=1.0)
```

### 3. HERON Adapters (Tutorial 04)
Use `PettingZooParallelEnv` (not raw `ParallelEnv`) for full HERON features:
```python
class MyEnv(PettingZooParallelEnv):
    def __init__(self):
        super().__init__(env_id="my_env")
        self.register_agent(agent)  # Enables event-driven, messaging, etc.
```

### 4. Dual Execution Modes (Tutorial 06 — Key Differentiator)
Train fast, test realistically—**this cannot be achieved by wrapping PettingZoo**:
```python
# Training: synchronous (fast)
env.step(actions)

# Testing: event-driven (realistic timing, delays, jitter)
env.setup_event_driven()
env.run_event_driven(t_end=100.0)
```

### 5. Pluggable Protocols (Tutorial 08)
Swap coordination strategies without changing environment code:
```python
protocol = SetpointProtocol()      # Direct control
protocol = PriceSignalProtocol()   # Market-based
protocol = ConsensusProtocol()     # Peer negotiation
```

### 6. Extensible Device Registry (Tutorial 09)
Add new device types with a standard interface:
```python
class WindTurbine(DeviceAgent):
    def get_action_space(self): ...
    def update_state(self, ext_state): ...
    def update_cost_safety(self): ...

DEVICE_REGISTRY['WindTurbine'] = WindTurbine
```

## Comparison with Production Code

| Tutorial Code | Production Code (`examples/05_mappo_training.py`) |
|---------------|---------------------------------------------------|
| 3 microgrids | 3 microgrids + DSO |
| 2 features | 14 features |
| Simplified physics | PandaPower integration |
| Single protocol (Tutorial 08) | 4 swappable protocols |
| Sync + event-driven (Tutorial 06) | Full dual-mode with CPS timing |
| 2 device types (Tutorial 09) | Generator, ESS, Transformer, Renewables |

## Next Steps

After completing the core tutorials (01-06):

1. **Explore the full case study**: `powergrid/` directory
2. **Run production training**: `python examples/05_mappo_training.py --test`
3. **Test with realistic timing**: Use Tutorial 06 patterns for event-driven validation

After completing the advanced tutorials (07-09):

4. **Configure environments**: Use Tutorial 07 patterns for YAML + datasets
5. **Create custom protocols**: Use Tutorial 08 to add domain-specific coordination
6. **Add custom devices**: Use Tutorial 09 to extend with new device types
7. **Add your own domain**: Use these patterns for traffic, robotics, etc.

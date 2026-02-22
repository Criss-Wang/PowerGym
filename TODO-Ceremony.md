# Heron Usability: Reducing Ceremony for New Developers

## Current State

`run_epymarl_training.py` requires **~125 lines of user code** and **7 custom classes** for a minimal 2-agent training setup:

| Class | Purpose | Lines |
|---|---|---|
| `DevicePowerFeature` | Feature with `vector()` + `set_values()` | 13 |
| `DeviceAgent` | 6 methods: `init_action`, `set_action`, `apply_action`, `set_state`, `compute_local_reward`, properties | 40 |
| `ZoneCoordinator` | Thin wrapper | 3 |
| `GridSystem` | Empty subclass | 2 |
| `EnvState` | Custom simulation state | 3 |
| `ActionPassingEnv` | 3 abstract methods to implement | 40 |
| Adaptor wiring | Factory + registration boilerplate | 25 |

For comparison, a PettingZoo or Gymnasium custom env is typically 30-50 lines.

---

## TODO Items

### 1. `SimpleFieldAgent` with sensible defaults (High impact)

Most `DeviceAgent` methods follow a predictable pattern. A batteries-included base class would eliminate the most common boilerplate:

```python
# What users write today: ~40 lines
# What they could write:
class DeviceAgent(SimpleFieldAgent):
    features = [PowerFeature(capacity=1.0)]
    action_dim = 1
    action_range = (-1.0, 1.0)

    def compute_local_reward(self, state) -> float:
        return -state["power"] ** 2
```

`SimpleFieldAgent` auto-generates `init_action`, `set_action`, `apply_action`, and `set_state` from the declared `action_dim`/`action_range`/`features`.

### 2. `EnvBuilder` / fluent factory (High impact)

Replace manual `FieldAgent -> CoordinatorAgent -> SystemAgent -> MultiAgentEnv` wiring:

```python
env = (
    heron.EnvBuilder()
    .add_agents("device", DeviceAgent, count=2, features=[PowerFeature()])
    .add_coordinator("zone", subordinates=["device_*"])
    .with_protocol(VerticalProtocol())
    .with_simulation(my_simulation_fn)
    .build()
)
```

Hides `SystemAgent`/`ProxyAgent`/`MessageBroker`/`Scheduler` plumbing for standard setups.

### 3. Simplify the simulation bridge (Medium impact)

The `global_state_to_env_state` / `run_simulation` / `env_state_to_global_state` round-trip is hard to understand. For simple cases, let users provide a single function:

```python
def simulate(agent_states: dict) -> dict:
    for aid, s in agent_states.items():
        s["power"] = np.clip(s["power"], -1.0, 1.0)
    return agent_states
```

The framework handles serialization/deserialization internally.

### 4. `env.as_<framework>()` adaptor methods (Low impact)

Instead of requiring separate adaptor wrapping:

```python
# Current
adapter = HeronEPyMARLAdapter(env_creator=factory, n_discrete=11, max_steps=50)

# Proposed
adapter = env.as_epymarl(n_discrete=11, max_steps=50)
adapter = env.as_rllib(discrete_actions=11)
adapter = env.as_gymnasium()
```

### 5. `NumericFeature` shortcut (Low impact)

`DevicePowerFeature` is 13 lines for 2 floats. A factory can derive `vector()` and `set_values()` from field definitions:

```python
# Current: 13 lines with vector() + set_values()
# Proposed: 1 line
PowerFeature = NumericFeature("power", "capacity", visibility="public")
```

### 6. Ship preset algorithm configs (Medium impact)

The 80-line `ALGO_CONFIGS` + `BASE_CONFIG` block is EPyMARL config that users shouldn't carry:

```python
from heron.adaptors.epymarl import presets
run_training(env, algo=presets.MAPPO, timesteps=5000)
```

### 7. "5-minute quickstart" example (Very high impact for adoption)

A single file, zero to training in <50 lines:

```python
import heron

env = heron.quickstart.make_power_grid(n_devices=4)
env.train(algo="mappo", timesteps=10_000)
env.evaluate(mode="event_driven", t_end=300)
```

---

## Priority Summary

| # | Item | Impact | Effort |
|---|---|---|---|
| 7 | Quickstart example | Very high | Low |
| 1 | `SimpleFieldAgent` | High | Medium |
| 2 | `EnvBuilder` | High | Medium |
| 3 | Simplified simulation bridge | Medium | Medium |
| 6 | Algo config presets | Medium | Low |
| 4 | `env.as_<framework>()` | Low | Low |
| 5 | `NumericFeature` shortcut | Low | Low |

The core architecture (hierarchical agents, visibility, dual-mode execution) is solid. The goal is to **raise the floor** (easy to start) without **lowering the ceiling** (power users can still customize everything).

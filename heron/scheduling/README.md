# HERON Event-Driven Scheduling

This module provides discrete-event simulation capabilities for testing trained MARL policies under realistic timing constraints.

## Overview

HERON supports two execution modes:

| Mode | Use Case | Timing | Communication |
|------|----------|--------|---------------|
| **Synchronous (Option A)** | Training | All agents step together | Direct method calls |
| **Event-Driven (Option B)** | Testing | Heterogeneous tick rates | Message-based with delays |

The scheduling module powers Option B, enabling you to test how trained policies perform when:
- Agents have different observation/action frequencies
- Communication has realistic latency
- Timing has natural variability (jitter)

## Components

### EventScheduler

Priority-queue based scheduler that processes events in timestamp order.

```python
from heron.scheduling import EventScheduler, EventType

scheduler = EventScheduler(start_time=0.0)

# Register agents with timing parameters
scheduler.register_agent(
    agent_id="sensor_1",
    tick_interval=1.0,    # Tick every 1 second
    obs_delay=0.1,        # 100ms observation latency
    act_delay=0.2,        # 200ms action delay
)

scheduler.register_agent(
    agent_id="controller_1",
    tick_interval=5.0,    # Tick every 5 seconds (slower)
)

# Set event handlers
scheduler.set_handler(EventType.AGENT_TICK, my_tick_handler)
scheduler.set_handler(EventType.ACTION_EFFECT, my_action_handler)

# Run simulation
events_processed = scheduler.run_until(t_end=100.0)
```

### Event Types

| Event Type | Description | When Triggered |
|------------|-------------|----------------|
| `AGENT_TICK` | Agent's regular observe/act cycle | At each tick_interval |
| `ACTION_EFFECT` | Delayed action takes effect | After act_delay |
| `MESSAGE_DELIVERY` | Message arrives at recipient | After msg_delay |
| `OBSERVATION_READY` | Delayed observation available | After obs_delay |
| `ENV_UPDATE` | Environment state update | Custom scheduling |
| `CUSTOM` | Domain-specific events | Custom scheduling |

### TickConfig

Centralized timing configuration with optional jitter for realistic testing.

```python
from heron.scheduling import TickConfig, JitterType

# Deterministic config (training)
config = TickConfig.deterministic(
    tick_interval=1.0,
    obs_delay=0.1,
    act_delay=0.2,
)

# With jitter (testing) - adds +/- 10% variability
config = TickConfig.with_jitter(
    tick_interval=1.0,
    obs_delay=0.1,
    act_delay=0.2,
    msg_delay=0.05,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=42,
)

# Use with agent registration
scheduler.register_agent("sensor_1", tick_config=config)
```

## Usage with Environments

The `HeronEnvCore` mixin provides convenience methods for event-driven execution:

```python
from heron.envs.base import HeronEnvCore

class MyEnv(HeronEnvCore):
    def __init__(self):
        self._init_heron_core(env_id="my_env")
        # ... register agents ...

    def run_test(self, t_end: float):
        # Setup event-driven mode
        self.setup_event_driven()

        # Configure handlers
        self.setup_default_handlers(
            global_state_fn=lambda: self.get_state(),
            on_action_effect=lambda aid, act: self.apply_action(aid, act),
        )

        # Run simulation
        return self.run_event_driven(t_end=t_end)
```

## Timing Parameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `tick_interval` | Time between agent ticks | 0.1s - 60s |
| `obs_delay` | Time for observation to be available | 0 - 1s |
| `act_delay` | Time for action to take effect | 0 - 1s |
| `msg_delay` | Time for message to be delivered | 0 - 1s |

**Hierarchical patterns:**
- **Field agents** (sensors): Fast ticks (0.1-1s), small delays
- **Coordinators**: Medium ticks (1-60s), coordinate subordinates
- **System agents**: Slow ticks (60s+), high-level decisions

## Example: Event-Driven Testing

See `case_studies/power/examples/07_event_driven_mode.py` for a complete example.

```python
# After training with synchronous mode...
# Test with realistic timing:

env.setup_event_driven()

# Register agents with realistic timing
for agent_id, agent in env.heron_agents.items():
    config = TickConfig.with_jitter(
        tick_interval=agent.tick_interval,
        obs_delay=0.1,
        act_delay=0.2,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
    )
    env.scheduler.register_agent(agent_id, tick_config=config)

# Run and evaluate
env.run_event_driven(t_end=3600.0)  # 1 hour simulation
```

## API Reference

See the API documentation for detailed method signatures:
- `heron.scheduling.EventScheduler`
- `heron.scheduling.Event`
- `heron.scheduling.EventType`
- `heron.scheduling.TickConfig`
- `heron.scheduling.JitterType`

# HERON Environments

This module provides environment base classes and framework adapters for building multi-agent environments with HERON.

## Architecture

```
                          HeronEnvCore (Mixin)
                    Agent management, event scheduling,
                         message broker support
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
       BaseEnv              MultiAgentEnv          Adapters
    (gym.Env +            (Abstract base)              │
    HeronEnvCore)                                      │
         │                                    ┌────────┴────────┐
         │                                    │                 │
         ▼                                    ▼                 ▼
   Single-agent                    PettingZooParallelEnv  RLlibMultiAgentEnv
   environments                      (pettingzoo.ParallelEnv)  (ray.rllib)
```

## File Structure

```
heron/envs/
├── __init__.py    # Public exports
├── base.py        # HeronEnvCore, BaseEnv, MultiAgentEnv
└── adapters.py    # PettingZooParallelEnv, RLlibMultiAgentEnv
```

## Execution Modes

### Option A: Synchronous (Training)

All agents step together within `env.step()`. Suitable for RL training with CTDE pattern.

```python
# Standard Gymnasium loop
obs, info = env.reset()
for _ in range(1000):
    actions = policy(obs)
    obs, rewards, terminated, truncated, info = env.step(actions)
```

### Option B: Event-Driven (Testing)

Agents tick independently with realistic timing. Tests policy robustness.

```python
# Setup event-driven mode
env.setup_event_driven()
env.setup_default_handlers(
    global_state_fn=lambda: env.get_state(),
    on_action_effect=lambda aid, act: env.apply_action(aid, act)
)

# Run simulation
num_events = env.run_event_driven(t_end=3600.0)
```

## Components

### HeronEnvCore (Mixin)

Core functionality mixin that can be combined with any environment interface.

```python
import gymnasium as gym
from heron.envs import HeronEnvCore
from heron.agents import FieldAgent

class MyEnv(gym.Env, HeronEnvCore):
    def __init__(self):
        super().__init__()
        self._init_heron_core(env_id="my_env")

        # Register agents
        agent = FieldAgent(agent_id="agent_1")
        self.register_agent(agent)

    def reset(self, **kwargs):
        self.reset_agents()
        obs = self.get_observations()
        return obs, {}

    def step(self, actions):
        self.apply_actions(actions)
        # ... physics simulation ...
        obs = self.get_observations()
        return obs, rewards, False, False, {}
```

**Key Methods:**

| Method | Mode | Description |
|--------|------|-------------|
| `_init_heron_core()` | Both | Initialize HERON functionality |
| `register_agent(agent)` | Both | Register an agent |
| `get_observations()` | Training | Collect observations from all agents |
| `apply_actions(actions)` | Training | Apply actions to agents |
| `reset_agents()` | Both | Reset all agents |
| `setup_event_driven()` | Testing | Initialize event scheduler |
| `setup_default_handlers()` | Testing | Configure event callbacks |
| `run_event_driven(t_end)` | Testing | Run simulation |

### BaseEnv

Single-agent Gymnasium environment with HERON support.

```python
from heron.envs import BaseEnv

class MySingleAgentEnv(BaseEnv):
    def __init__(self):
        super().__init__(env_id="single_agent_env")
        # Setup agent...

    def reset(self, *, seed=None, options=None):
        self.reset_agents()
        return observation, {}

    def step(self, action):
        # Apply action, run physics, compute reward
        return obs, reward, terminated, truncated, info
```

### MultiAgentEnv

Abstract base for multi-agent environments (not tied to any framework).

```python
from heron.envs import MultiAgentEnv

class MyMultiAgentEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__(env_id="multi_agent_env")

    def reset(self, *, seed=None, options=None):
        # Return (obs_dict, info)
        pass

    def step(self, actions):
        # Return (obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict)
        pass

    def get_joint_observation_space(self):
        pass

    def get_joint_action_space(self):
        pass
```

### PettingZooParallelEnv

Adapter for PettingZoo's ParallelEnv interface.

```python
from heron.envs import PettingZooParallelEnv

class MyPettingZooEnv(PettingZooParallelEnv):
    def __init__(self):
        super().__init__(env_id="pz_env")

        # Build agents
        self._build_agents()

        # Set PettingZoo-required attributes
        self._set_agent_ids(list(self.heron_agents.keys()))
        self._init_spaces(
            action_spaces=self.get_agent_action_spaces(),
            observation_spaces=self.get_agent_observation_spaces()
        )

    def reset(self, seed=None, options=None):
        self.reset_agents()
        obs = self.get_observations()
        return {aid: o.vector() for aid, o in obs.items()}, {}

    def step(self, actions):
        self.apply_actions(actions)
        # ... physics ...
        obs = self.get_observations()
        return obs, rewards, terminateds, truncateds, infos
```

**PettingZoo API:**

```python
env.agents           # List of active agent IDs
env.possible_agents  # List of all possible agent IDs
env.observation_space(agent_id)  # Get observation space
env.action_space(agent_id)       # Get action space
```

### RLlibMultiAgentEnv

Adapter for Ray RLlib's MultiAgentEnv interface.

```python
from heron.envs import RLlibMultiAgentEnv

class MyRLlibEnv(RLlibMultiAgentEnv):
    def __init__(self, config=None):
        super().__init__(env_id="rllib_env")

        # Build agents
        self._build_agents()

        # Set RLlib-required attributes
        self._set_agent_ids(list(self.heron_agents.keys()))
        self._init_spaces(
            action_spaces=self.get_agent_action_spaces(),
            observation_spaces=self.get_agent_observation_spaces()
        )

    def reset(self, *, seed=None, options=None):
        self.reset_agents()
        obs = self.get_observations()
        return {aid: o.vector() for aid, o in obs.items()}, {}

    def step(self, actions):
        # RLlib expects "__all__" key in terminated/truncated dicts
        terminateds = {"agent_1": False, "__all__": False}
        truncateds = {"agent_1": False, "__all__": False}
        return obs, rewards, terminateds, truncateds, infos
```

## SystemAgent Integration

For hierarchical multi-agent systems, use SystemAgent to manage the agent hierarchy.

### Training (Option A)

```python
from heron.agents import SystemAgent, CoordinatorAgent, FieldAgent
from heron.envs import PettingZooParallelEnv

class HierarchicalEnv(PettingZooParallelEnv):
    def __init__(self):
        super().__init__()

        # Build hierarchy
        field_agents = [FieldAgent(f"field_{i}") for i in range(6)]
        coordinators = [
            CoordinatorAgent("coord_1", subordinates={a.agent_id: a for a in field_agents[:3]}),
            CoordinatorAgent("coord_2", subordinates={a.agent_id: a for a in field_agents[3:]}),
        ]
        system = SystemAgent("system", coordinators={c.agent_id: c for c in coordinators})

        # Register with environment
        self.set_system_agent(system)

    def step(self, actions):
        # Use CTDE pattern
        self.step_with_system_agent(actions, global_state=self.get_state())
        # ... physics simulation ...
        return obs, rewards, terminateds, truncateds, infos
```

### Testing (Option B)

```python
# After training, test with realistic timing
env.reset()

num_events = env.run_event_driven_with_system_agent(
    t_end=3600.0,
    get_global_state=lambda: env.get_state(),
    on_action_effect=lambda aid, act: env.apply_action(aid, act)
)
```

## ProxyAgent Integration

Use ProxyAgent for visibility-filtered state distribution.

```python
from heron.agents import ProxyAgent

# Create proxy with visibility rules
proxy = ProxyAgent(
    agent_id="proxy",
    visibility_rules={
        "field_1": ["voltage", "power"],
        "coord_1": ["*"],  # See everything
    }
)

# Register with environment
env.set_proxy_agent(proxy)

# In step() or tick handlers, update proxy state
env.update_proxy_state(current_state)

# Agents request filtered state through proxy
filtered_state = agent.request_state_from_proxy(env.proxy_agent)
```

## Message Broker

All environments have a message broker (defaults to InMemoryBroker).

```python
# Configure agents for distributed communication
env.configure_agents_for_distributed()

# Setup broker channels
env.setup_broker_channels()

# Environment can publish messages
env.publish_action(sender_id, recipient_id, action)
env.publish_info(sender_id, recipient_id, info)
env.publish_state_update(state)
env.broadcast_to_agents(sender_id, payload)

# Environment can consume messages
actions = env.consume_actions_for_agent(agent_id)
info = env.consume_info_for_agent(agent_id)

# Reset broker for environment
env.clear_broker_environment()
```

## API Reference

### HeronEnvCore

| Method | Description |
|--------|-------------|
| `_init_heron_core(env_id, scheduler, message_broker)` | Initialize core |
| `register_agent(agent)` | Register single agent |
| `register_agents(agents)` | Register multiple agents |
| `get_heron_agent(agent_id)` | Get agent by ID |
| `get_observations(global_state)` | Collect all observations |
| `apply_actions(actions, observations)` | Apply actions to agents |
| `get_agent_action_spaces()` | Get all action spaces |
| `get_agent_observation_spaces()` | Get all observation spaces |
| `reset_agents(seed, **kwargs)` | Reset all agents |
| `setup_event_driven(scheduler)` | Setup event scheduler |
| `set_event_handlers(...)` | Set custom event handlers |
| `setup_default_handlers(...)` | Setup standard handlers |
| `run_event_driven(t_end, max_events)` | Run simulation |
| `configure_agents_for_distributed()` | Setup broker for agents |
| `setup_broker_channels()` | Create broker channels |
| `set_system_agent(system_agent)` | Set SystemAgent |
| `set_proxy_agent(proxy_agent)` | Set ProxyAgent |
| `update_proxy_state(state)` | Update proxy cache |
| `step_with_system_agent(actions, global_state)` | CTDE step |
| `run_event_driven_with_system_agent(...)` | Event-driven with hierarchy |
| `close_heron()` | Cleanup resources |

### Properties

| Property | Description |
|----------|-------------|
| `heron_agents` | Dict of registered agents |
| `heron_coordinators` | Dict of coordinator agents |
| `system_agent` | SystemAgent instance |
| `proxy_agent` | ProxyAgent instance |
| `simulation_time` | Current simulation time |

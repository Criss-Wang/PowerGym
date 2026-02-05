# HERON Core

This module provides the fundamental data structures for the HERON multi-agent framework.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Agent                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │    State     │  │ Observation  │  │       Action         │   │
│  │              │  │              │  │                      │   │
│  │ ┌──────────┐ │  │  local: {}   │  │  c: [continuous]     │   │
│  │ │ Feature  │ │  │  global: {}  │  │  d: [discrete]       │   │
│  │ │ Provider │ │  │  timestamp   │  │  range, ncats        │   │
│  │ └──────────┘ │  │              │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        Policy                             │   │
│  │            forward(observation) -> action                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
heron/core/
├── __init__.py      # Public exports
├── action.py        # Action representation (continuous + discrete)
├── observation.py   # Observation container
├── state.py         # State management with features
├── feature.py       # FeatureProvider base class
└── policies.py      # Policy interface
```

## Components

### Action

Mixed continuous/discrete action representation with normalization support.

```python
from heron.core import Action
import numpy as np

# Continuous action with bounds
action = Action()
action.set_specs(
    dim_c=2,
    range=(np.array([0.0, -1.0]), np.array([10.0, 1.0]))
)
action.sample()  # Random action within bounds
print(action.c)  # e.g., [5.2, 0.3]

# Discrete action
action = Action()
action.set_specs(dim_d=1, ncats=[5])  # 5 choices
action.sample()
print(action.d)  # e.g., [2]

# Mixed continuous + discrete
action = Action()
action.set_specs(
    dim_c=2,
    dim_d=1,
    ncats=[3],
    range=(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
)

# Normalization for RL algorithms
scaled = action.scale()      # Map to [-1, 1]
action.unscale(scaled)       # Map back to physical units

# Create from Gymnasium space
from gymnasium.spaces import Box
action = Action.from_gym_space(Box(low=0, high=1, shape=(3,)))
```

### Observation

Structured observation container separating local and global information.

```python
from heron.core import Observation
import numpy as np

# Create observation
obs = Observation(
    local={"voltage": 1.02, "power": np.array([100.0, 50.0])},
    global_info={"grid_frequency": 60.0},
    timestamp=10.5
)

# Convert to flat vector for RL
vec = obs.vector()  # array([100., 50., 1.02, 60.], dtype=float32)

# Nested observations are flattened recursively
obs = Observation(
    local={
        "sensors": {"temp": 25.0, "pressure": 101.3},
        "status": 1
    }
)
vec = obs.vector()  # Flattens all numeric values
```

**Observation Keys (constants for consistency):**

```python
from heron.core.observation import (
    OBS_KEY_STATE,           # "state" - Agent's state vector
    OBS_KEY_OBSERVATION,     # "observation" - Agent's observation vector
    OBS_KEY_PROXY_STATE,     # "proxy_state" - Delayed observations
    OBS_KEY_SUBORDINATE_OBS, # "subordinate_obs" - Coordinator's subordinate obs
    OBS_KEY_COORDINATOR_STATE,  # "coordinator_state"
    OBS_KEY_COORDINATOR_OBS,    # "coordinator_obs" - System's coordinator obs
    OBS_KEY_SYSTEM_STATE,       # "system_state"
)
```

### State

State management using composable FeatureProviders with visibility rules.

```python
from heron.core import State, FieldAgentState, CoordinatorAgentState, SystemAgentState

# Create state for a field agent
state = FieldAgentState(owner_id="battery_1", owner_level=1)
state.features.append(BatteryFeature())

# Get full state vector
vec = state.vector()  # Concatenates all feature vectors

# Update specific feature
state.update_feature("BatteryFeature", soc=0.8)

# Visibility-filtered observations
obs = state.observed_by(
    requestor_id="coordinator_1",
    requestor_level=2
)
# Returns only features visible to the requestor
```

**State Types by Hierarchy Level:**

| Class | Level | Usage |
|-------|-------|-------|
| `FieldAgentState` | L1 | Individual units |
| `CoordinatorAgentState` | L2 | Regional coordinators |
| `SystemAgentState` | L3 | System-wide state |

### FeatureProvider

Abstract base class for observable state attributes with visibility rules.

```python
from heron.core import FeatureProvider
import numpy as np

class BatteryState(FeatureProvider):
    # Visibility: who can observe this feature
    # Options: "public", "owner", "upper_level", "system"
    visibility = ["owner", "upper_level"]

    def __init__(self, capacity: float = 100.0):
        self.soc = 0.5  # State of charge [0, 1]
        self.capacity = capacity
        self.power = 0.0

    def vector(self) -> np.ndarray:
        """Convert to numpy array for observations."""
        return np.array([self.soc, self.power], dtype=np.float32)

    def names(self) -> list:
        """Field names corresponding to vector elements."""
        return ["soc", "power"]

    def to_dict(self) -> dict:
        """Serialize for communication/logging."""
        return {"soc": self.soc, "capacity": self.capacity, "power": self.power}

    @classmethod
    def from_dict(cls, d: dict) -> "BatteryState":
        """Deserialize from dictionary."""
        f = cls(d.get("capacity", 100.0))
        f.soc = d.get("soc", 0.5)
        f.power = d.get("power", 0.0)
        return f

    def set_values(self, **kwargs) -> None:
        """Update values from keyword arguments."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

# Usage
battery = BatteryState(capacity=200.0)
battery.set_values(soc=0.8, power=50.0)
print(battery.vector())  # [0.8, 50.0]
```

**Visibility Rules:**

| Visibility | Who Can Observe |
|------------|-----------------|
| `"public"` | All agents |
| `"owner"` | Only the owning agent |
| `"upper_level"` | Agents one level above in hierarchy |
| `"system"` | System-level (L3) agents only |

Visibility rules are OR-ed together. A feature with `["owner", "upper_level"]` is visible to both the owner AND agents one level above.

### Policy

Abstract interface for decision-making policies.

```python
from heron.core import Policy, RandomPolicy
from heron.core.observation import Observation

# Abstract policy interface
class MyPolicy(Policy):
    def forward(self, observation: Observation) -> Action:
        """Compute action from observation."""
        # Your policy logic here
        return action

    def reset(self) -> None:
        """Reset policy state (e.g., RNN hidden states)."""
        pass

# Built-in random policy for testing
from gymnasium.spaces import Box

action_space = Box(low=-1, high=1, shape=(3,))
policy = RandomPolicy(action_space, seed=42)

obs = Observation()
action = policy.forward(obs)  # Returns random action
```

## Quick Reference

### Creating a Complete Agent State

```python
import numpy as np
from heron.core import FieldAgentState, FeatureProvider

# 1. Define custom features
class PositionFeature(FeatureProvider):
    visibility = ["public"]

    def __init__(self):
        self.x = 0.0
        self.y = 0.0

    def vector(self):
        return np.array([self.x, self.y], dtype=np.float32)

    def names(self):
        return ["x", "y"]

    def to_dict(self):
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d):
        f = cls()
        f.x = d.get("x", 0.0)
        f.y = d.get("y", 0.0)
        return f

    def set_values(self, **kwargs):
        if "x" in kwargs:
            self.x = kwargs["x"]
        if "y" in kwargs:
            self.y = kwargs["y"]

class VelocityFeature(FeatureProvider):
    visibility = ["owner"]  # Only owner can see velocity

    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0

    def vector(self):
        return np.array([self.vx, self.vy], dtype=np.float32)

    def names(self):
        return ["vx", "vy"]

    def to_dict(self):
        return {"vx": self.vx, "vy": self.vy}

    @classmethod
    def from_dict(cls, d):
        f = cls()
        f.vx = d.get("vx", 0.0)
        f.vy = d.get("vy", 0.0)
        return f

    def set_values(self, **kwargs):
        if "vx" in kwargs:
            self.vx = kwargs["vx"]
        if "vy" in kwargs:
            self.vy = kwargs["vy"]

# 2. Create state with features
state = FieldAgentState(owner_id="robot_1", owner_level=1)
state.features.append(PositionFeature())
state.features.append(VelocityFeature())

# 3. Update features
state.update_feature("PositionFeature", x=10.0, y=5.0)
state.update_feature("VelocityFeature", vx=1.0, vy=0.5)

# 4. Get full state vector
vec = state.vector()  # [10.0, 5.0, 1.0, 0.5]

# 5. Get visibility-filtered observation
# Owner sees everything
owner_obs = state.observed_by("robot_1", requestor_level=1)
# {"PositionFeature": [10., 5.], "VelocityFeature": [1., 0.5]}

# Other agents only see public features
other_obs = state.observed_by("robot_2", requestor_level=1)
# {"PositionFeature": [10., 5.]}
```

### Action Space Patterns

```python
from heron.core import Action
import numpy as np

# Pattern 1: Continuous control (e.g., power setpoints)
action = Action()
action.set_specs(
    dim_c=3,
    range=(
        np.array([0.0, 0.0, -100.0]),   # Lower bounds
        np.array([100.0, 50.0, 100.0])  # Upper bounds
    )
)

# Pattern 2: Discrete control (e.g., mode selection)
action = Action()
action.set_specs(
    dim_d=2,
    ncats=[3, 4]  # First head: 3 choices, Second head: 4 choices
)

# Pattern 3: Hybrid control (e.g., setpoint + mode)
action = Action()
action.set_specs(
    dim_c=1,                              # Continuous setpoint
    dim_d=1,                              # Discrete mode
    ncats=[3],                            # 3 modes
    range=(np.array([0.0]), np.array([1.0]))
)

# Setting values
action.set_values({"c": [0.5], "d": [1]})
action.set_values(c=[0.5], d=[1])  # Keyword args also work

# Clipping to valid range
action.clip()

# Export for logging
vec = action.vector()  # [0.5, 1.0] (c and d concatenated as float32)
```

## API Summary

### Action

| Method | Description |
|--------|-------------|
| `set_specs(dim_c, dim_d, ncats, range)` | Configure action space |
| `sample(seed=None)` | Random sample from action space |
| `reset(action=None, c=None, d=None)` | Reset to neutral or specified value |
| `set_values(action, c=None, d=None)` | Set from various formats |
| `clip()` | Clip to valid range |
| `scale()` | Normalize continuous to [-1, 1] |
| `unscale(x)` | Denormalize from [-1, 1] |
| `vector()` | Flatten to [c..., d...] |
| `copy()` | Deep copy |
| `from_gym_space(space)` | Create from Gymnasium space |
| `from_vector(vec, dim_c, dim_d, ...)` | Create from flat vector |

### Observation

| Method | Description |
|--------|-------------|
| `vector()` | Flatten to numpy array |
| `as_vector()` | Alias for vector() |

### State

| Method | Description |
|--------|-------------|
| `vector()` | Concatenate all feature vectors |
| `reset(overrides=None)` | Reset all features |
| `update(updates)` | Batch update features |
| `update_feature(name, **values)` | Update single feature |
| `observed_by(requestor_id, requestor_level)` | Get visibility-filtered observation |
| `to_dict()` | Serialize to dictionary |

### FeatureProvider

| Method | Description |
|--------|-------------|
| `vector()` | Convert to numpy array (abstract) |
| `names()` | Get field names (abstract) |
| `to_dict()` | Serialize (abstract) |
| `from_dict(d)` | Deserialize (abstract, classmethod) |
| `set_values(**kwargs)` | Update values (abstract) |
| `reset(**overrides)` | Reset to initial state |
| `is_observable_by(...)` | Check visibility |

### Policy

| Method | Description |
|--------|-------------|
| `forward(observation)` | Compute action (abstract) |
| `reset()` | Reset policy state |

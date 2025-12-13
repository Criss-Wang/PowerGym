# State, Action, and Observation System

PowerGrid 2.0 uses a modular architecture for representing agent state, actions, and observations. This design enables flexible composition, visibility control, and seamless integration with RL frameworks.

---

## Overview

The core abstractions are:

- **`State`**: Agent state composed of multiple `FeatureProvider`s
- **`Action`**: Flexible action representation with continuous/discrete support
- **`Observation`**: Structured observation with local/global/message components
- **`FeatureProvider`**: Modular state attributes with visibility rules

---

## State System

### State Classes

PowerGrid defines two main state classes:

#### DeviceState

Represents the state of a single device agent (Generator, ESS, etc.).

```python
from powergrid.core.state import DeviceState
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.storage import StorageBlock

# Create device state with features
state = DeviceState(
    owner_id="ess1",
    owner_level=1,  # Device level
    features=[
        ElectricalBasePh(P_MW=0.5, Q_MVAr=0.1),
        StorageBlock(soc=0.8, e_capacity_MWh=2.0, p_ch_max_MW=1.0, p_dsc_max_MW=1.0)
    ]
)

# Vectorize for ML observations
vector = state.vector()  # Returns flat numpy array

# Update features
state.update({
    "ElectricalBasePh": {"P_MW": 0.6, "Q_MVAr": 0.15},
    "StorageBlock": {"soc": 0.75}
})

# Get observable features for another agent
obs_dict = state.observed_by(
    requestor_id="grid1",
    requestor_level=2  # Grid level
)
```

#### GridState

Represents the state of a grid-level coordinator agent.

```python
from powergrid.core.state import GridState
from powergrid.features.network import BusVoltages, LineFlows, NetworkMetrics

state = GridState(
    owner_id="microgrid1",
    owner_level=2,  # Grid level
    features=[
        BusVoltages(vm_pu=[1.02, 0.98, 1.01], va_deg=[0, -2.1, -1.5]),
        LineFlows(p_from_mw=[0.5, 0.3], q_from_mvar=[0.1, 0.05]),
        NetworkMetrics(total_gen_mw=1.5, total_load_mw=1.2, total_loss_mw=0.05)
    ]
)
```

### Key State Methods

```python
# Vectorize all features
vector = state.vector()  # Concatenates all feature vectors

# Reset state to initial values
state.reset(overrides={"ElectricalBasePh": {"P_MW": 0.0}})

# Update specific feature
state.update_feature("StorageBlock", soc=0.9)

# Check observability
obs = state.observed_by(requestor_id="agent2", requestor_level=2)

# Serialize/deserialize
state_dict = state.to_dict()
```

---

## Action System

### Action Class

The `Action` class provides a unified interface for continuous and discrete actions:

```python
from powergrid.core.action import Action
import numpy as np

# Create action with continuous + discrete components
action = Action()
action.set_specs(
    dim_c=4,  # 4 continuous dimensions (e.g., P1, Q1, P2, Q2)
    dim_d=2,  # 2 discrete dimensions
    ncats=[3, 2],  # First head has 3 categories, second has 2
    range=(
        np.array([-1.0, -0.5, -1.0, -0.5]),  # Lower bounds
        np.array([1.0, 0.5, 1.0, 0.5])        # Upper bounds
    )
)
```

### Setting Actions

```python
# Set from dictionary
action.set_values({"c": [0.5, 0.2, 0.3, 0.1], "d": [1, 0]})

# Set from flat vector
action.set_values([0.5, 0.2, 0.3, 0.1, 1, 0])

# Set via keyword arguments
action.set_values(c=[0.5, 0.2, 0.3, 0.1], d=[1, 0])

# Sample random action
action.sample()
```

### Normalization

Actions can be normalized to [-1, 1] for RL algorithms:

```python
# Scale: physical → normalized [-1, 1]
normalized = action.scale()

# Unscale: normalized [-1, 1] → physical
physical_values = action.unscale([0.5, -0.2, 0.8, -0.3])
```

### Gymnasium Space Conversion

Actions automatically create Gymnasium spaces:

```python
# Get Gymnasium space
space = action.space  # Returns Box, Discrete, MultiDiscrete, or Dict

# For continuous only:
# Box(low=[-1.0, -0.5, -1.0, -0.5], high=[1.0, 0.5, 1.0, 0.5])

# For mixed continuous + discrete:
# Dict({
#     "c": Box(low=[-1.0, -0.5, -1.0, -0.5], high=[1.0, 0.5, 1.0, 0.5]),
#     "d": MultiDiscrete([3, 2])
# })
```

### Clipping and Validation

```python
# Clip to valid ranges
action.clip()  # Clips continuous to range, discrete to [0, ncats-1]

# Reset to neutral
action.reset()  # Sets to midpoint of range (continuous) and 0 (discrete)
```

---

## Observation System

### Observation Class

Structured observations for agents:

```python
from powergrid.core.observation import Observation, Message

obs = Observation(
    local={
        'state': device_state.vector(),
        'observation': full_obs_vector
    },
    global_info={
        'bus_voltages': [1.02, 0.98, 1.01],
        'line_loading': [45.2, 67.8]
    },
    messages=[
        Message(
            sender="microgrid2",
            content={"price": 50.0, "quantity": 0.5},
            timestamp=10.0
        )
    ],
    timestamp=10.0
)

# Convert to flat vector for RL
vector = obs.vector()  # Flattens local + global_info
```

### Message Communication

```python
from powergrid.core.observation import Message

# Create message
msg = Message(
    sender="agent1",
    content={"price": 45.0, "power_offer": 0.8},
    recipient="agent2",  # Or None for broadcast
    timestamp=15.0
)

# Messages are included in observations
obs = Observation(
    local={},
    global_info={},
    messages=[msg],
    timestamp=15.0
)
```

---

## Feature Provider System

### FeatureProvider Base Class

All features inherit from `FeatureProvider`:

```python
from powergrid.features.base import FeatureProvider
from typing import List, Dict, Any
import numpy as np

class MyFeature(FeatureProvider):
    visibility = ["owner"]  # Only owner can observe

    def __init__(self, my_value: float):
        self.my_value = my_value

    def vector(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.my_value], dtype=np.float32)

    def names(self) -> List[str]:
        """Field names corresponding to vector"""
        return ["my_value"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {"my_value": self.my_value}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MyFeature":
        """Deserialize from dictionary"""
        return cls(my_value=d["my_value"])

    def set_values(self, **kwargs: Any) -> None:
        """Update values"""
        if "my_value" in kwargs:
            self.my_value = kwargs["my_value"]
```

### Visibility Rules

Features define who can observe them via the `visibility` class attribute:

```python
class PublicFeature(FeatureProvider):
    visibility = ["public"]  # All agents can observe

class OwnerOnlyFeature(FeatureProvider):
    visibility = ["owner"]  # Only owner can observe

class SystemFeature(FeatureProvider):
    visibility = ["system"]  # Only system-level agents (level >= 3)

class UpperLevelFeature(FeatureProvider):
    visibility = ["upper_level"]  # Only agents one level above owner
```

Visibility is checked via `is_observable_by()`:

```python
feature = OwnerOnlyFeature()

# Owner can observe
visible = feature.is_observable_by(
    requestor_id="device1",
    requestor_level=1,
    owner_id="device1",
    owner_level=1
)  # Returns True

# Non-owner cannot observe
visible = feature.is_observable_by(
    requestor_id="device2",
    requestor_level=1,
    owner_id="device1",
    owner_level=1
)  # Returns False
```

---

## Built-in Features

### Electrical Features

```python
from powergrid.features.electrical import ElectricalBasePh

feature = ElectricalBasePh(
    P_MW=0.5,
    Q_MVAr=0.2,
    S_MVA=0.54,
    pf=0.93
)
```

### Storage Features

```python
from powergrid.features.storage import StorageBlock

feature = StorageBlock(
    soc=0.8,
    e_capacity_MWh=2.0,
    p_ch_max_MW=1.0,
    p_dsc_max_MW=1.0
)
```

### Network Features

```python
from powergrid.features.network import BusVoltages, LineFlows, NetworkMetrics

voltages = BusVoltages(
    vm_pu=[1.02, 0.98, 1.01],
    va_deg=[0, -2.1, -1.5],
    bus_names=["bus1", "bus2", "bus3"]
)

flows = LineFlows(
    p_from_mw=[0.5, 0.3],
    q_from_mvar=[0.1, 0.05],
    loading_percent=[45.2, 67.8],
    line_names=["line1", "line2"]
)

metrics = NetworkMetrics(
    total_gen_mw=1.5,
    total_load_mw=1.2,
    total_loss_mw=0.05,
    total_gen_mvar=0.3,
    total_load_mvar=0.25
)
```

### Status Features

```python
from powergrid.features.status import StatusBlock

feature = StatusBlock(
    in_service=True,
    controllable=True,
    state="online"
)
```

---

## Integration with Agents

### DeviceAgent Example

```python
from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.state import DeviceState
from powergrid.core.action import Action

class MyDeviceAgent(DeviceAgent):
    def set_device_state(self):
        """Initialize device state features"""
        self.state = DeviceState(
            owner_id=self.agent_id,
            owner_level=1,
            features=[
                ElectricalBasePh(P_MW=0.0, Q_MVAr=0.0),
                StorageBlock(soc=0.5, e_capacity_MWh=2.0, p_ch_max_MW=1.0, p_dsc_max_MW=1.0),
                StatusBlock(in_service=True)
            ]
        )

    def set_device_action(self):
        """Initialize device action space"""
        self.action.set_specs(
            dim_c=2,  # P, Q
            range=(
                np.array([-1.0, -0.5]),  # P: [-1, 1] MW, Q: [-0.5, 0.5] MVAr
                np.array([1.0, 0.5])
            )
        )
```

### GridAgent Example

```python
from powergrid.agents.grid_agent import GridAgent
from powergrid.core.state import GridState

class MyGridAgent(GridAgent):
    def _update_grid_state(self, net):
        """Update GridState from PandaPower network"""
        # Extract network results
        bus_vm = net.res_bus.vm_pu.values
        bus_va = net.res_bus.va_degree.values

        # Update state features
        self.state.features = [
            BusVoltages(vm_pu=bus_vm, va_deg=bus_va),
            LineFlows(...),
            NetworkMetrics(...)
        ]
```

---

## Best Practices

### 1. Use Appropriate Visibility

```python
# Device-specific internals: owner only
class InternalState(FeatureProvider):
    visibility = ["owner"]

# Network measurements: public
class NetworkMeasurement(FeatureProvider):
    visibility = ["public"]

# Coordination signals: upper level
class CoordinationSignal(FeatureProvider):
    visibility = ["upper_level"]
```

### 2. Keep Features Focused

Each FeatureProvider should represent a cohesive set of related attributes:

```python
# Good: Focused feature
class ElectricalBasePh(FeatureProvider):
    def __init__(self, P_MW, Q_MVAr, S_MVA, pf):
        ...

# Avoid: Too many unrelated fields
class EverythingFeature(FeatureProvider):
    def __init__(self, P_MW, soc, voltage, price, ...):
        ...  # Too broad!
```

### 3. Use Action Normalization

```python
# RL agents output normalized actions
normalized_action = policy(obs)  # In [-1, 1]

# Unscale to physical units
action.unscale(normalized_action)

# Execute physical action
device.execute(action.c)  # Physical units (MW, MVAr)
```

### 4. Leverage State Composition

```python
# Compose state from multiple features
state = DeviceState(
    owner_id="device1",
    owner_level=1,
    features=[
        ElectricalBasePh(...),
        StorageBlock(...),
        PowerLimits(...),
        StatusBlock(...)
    ]
)

# Automatic vectorization
vector = state.vector()  # Concatenates all features
```

---

## Next Steps

- **Feature Reference**: See [Features Documentation](../api/features.md) for all built-in features
- **Agent Development**: Learn how to create custom agents in [Agent Guide](../developer/custom_agents.md)
- **Examples**: Check out [Examples](../examples/04_custom_device.md) for complete implementations

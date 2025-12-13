# Quick Reference Guide

A quick reference for common PowerGrid operations and patterns.

---

## Creating Custom Agents

### Custom DeviceAgent

```python
from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.state import DeviceState
from powergrid.core.action import Action
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.storage import StorageBlock
import numpy as np

class CustomBattery(DeviceAgent):
    def set_device_state(self):
        """Initialize device state with features"""
        self.state = DeviceState(
            owner_id=self.agent_id,
            owner_level=1,
            features=[
                ElectricalBasePh(P_MW=0.0, Q_MVAr=0.0),
                StorageBlock(
                    soc=0.5,
                    e_capacity_MWh=2.0,
                    p_ch_max_MW=1.0,
                    p_dsc_max_MW=1.0
                )
            ]
        )

    def set_device_action(self):
        """Initialize action space"""
        self.action.set_specs(
            dim_c=2,  # P, Q
            range=(
                np.array([-1.0, -0.5]),  # Lower bounds
                np.array([1.0, 0.5])      # Upper bounds
            )
        )

    def reset_device(self, **kwargs):
        """Reset device to initial state"""
        init_soc = kwargs.get('init_soc', 0.5)
        self.state.update({
            "ElectricalBasePh": {"P_MW": 0.0, "Q_MVAr": 0.0},
            "StorageBlock": {"soc": init_soc}
        })

    def update_state(self):
        """Update device state based on current action"""
        # Extract action
        P_MW = self.action.c[0]
        Q_MVAr = self.action.c[1]

        # Update SOC based on power
        dt = 0.25  # 15-minute timestep
        storage_feature = next(f for f in self.state.features if f.feature_name == "StorageBlock")
        delta_e = P_MW * dt
        new_soc = storage_feature.soc - delta_e / storage_feature.e_capacity_MWh

        # Clip SOC to [0, 1]
        new_soc = np.clip(new_soc, 0.0, 1.0)

        # Update state
        self.state.update({
            "ElectricalBasePh": {"P_MW": P_MW, "Q_MVAr": Q_MVAr},
            "StorageBlock": {"soc": new_soc}
        })

    def update_cost_safety(self):
        """Compute cost and safety penalties"""
        # Cost: degradation cost for charge/discharge
        P_MW = self.action.c[0]
        self.cost = abs(P_MW) * 0.05  # $0.05/MWh degradation

        # Safety: SOC violations
        storage = next(f for f in self.state.features if f.feature_name == "StorageBlock")
        soc_low_penalty = max(0.2 - storage.soc, 0) * 10
        soc_high_penalty = max(storage.soc - 0.9, 0) * 10
        self.safety = soc_low_penalty + soc_high_penalty
```

### Custom GridAgent

```python
from powergrid.agents.grid_agent import PowerGridAgent
from powergrid.core.protocols import NoProtocol

class CustomMicrogrid(PowerGridAgent):
    def __init__(self, net, grid_config, **kwargs):
        # Add custom initialization
        self.my_custom_param = grid_config.get('my_param', 1.0)

        super().__init__(
            net=net,
            grid_config=grid_config,
            protocol=NoProtocol(),
            **kwargs
        )

    def _build_local_observation(self, device_obs, **kwargs):
        """Override to add custom observations"""
        local = super()._build_local_observation(device_obs, **kwargs)

        # Add custom features
        local['custom_metric'] = self._compute_custom_metric()

        return local

    def _compute_custom_metric(self):
        """Custom metric computation"""
        return 42.0
```

---

## Creating Custom Features

### Basic Feature

```python
from powergrid.features.base import FeatureProvider
from typing import List, Dict, Any
import numpy as np

class TemperatureFeature(FeatureProvider):
    """Temperature sensor feature"""

    visibility = ["owner", "upper_level"]  # Who can observe

    def __init__(self, temp_celsius: float = 25.0):
        self.temp_celsius = temp_celsius

    def vector(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.temp_celsius], dtype=np.float32)

    def names(self) -> List[str]:
        """Field names"""
        return ["temp_celsius"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize"""
        return {"temp_celsius": self.temp_celsius}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemperatureFeature":
        """Deserialize"""
        return cls(temp_celsius=d["temp_celsius"])

    def set_values(self, **kwargs: Any) -> None:
        """Update values"""
        if "temp_celsius" in kwargs:
            self.temp_celsius = kwargs["temp_celsius"]

    def reset(self) -> None:
        """Reset to initial state"""
        self.temp_celsius = 25.0
```

### Multi-field Feature

```python
class WeatherFeature(FeatureProvider):
    """Weather conditions feature"""

    visibility = ["public"]  # Publicly observable

    def __init__(
        self,
        temperature_c: float = 25.0,
        solar_irradiance_w_m2: float = 800.0,
        wind_speed_m_s: float = 5.0
    ):
        self.temperature_c = temperature_c
        self.solar_irradiance_w_m2 = solar_irradiance_w_m2
        self.wind_speed_m_s = wind_speed_m_s

    def vector(self) -> np.ndarray:
        """Flatten all fields"""
        return np.array([
            self.temperature_c,
            self.solar_irradiance_w_m2,
            self.wind_speed_m_s
        ], dtype=np.float32)

    def names(self) -> List[str]:
        """Field names"""
        return ["temperature_c", "solar_irradiance_w_m2", "wind_speed_m_s"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize"""
        return {
            "temperature_c": self.temperature_c,
            "solar_irradiance_w_m2": self.solar_irradiance_w_m2,
            "wind_speed_m_s": self.wind_speed_m_s
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeatherFeature":
        """Deserialize"""
        return cls(**d)

    def set_values(self, **kwargs: Any) -> None:
        """Update any subset of fields"""
        if "temperature_c" in kwargs:
            self.temperature_c = kwargs["temperature_c"]
        if "solar_irradiance_w_m2" in kwargs:
            self.solar_irradiance_w_m2 = kwargs["solar_irradiance_w_m2"]
        if "wind_speed_m_s" in kwargs:
            self.wind_speed_m_s = kwargs["wind_speed_m_s"]
```

---

## Working with States

### Creating States

```python
from powergrid.core.state import DeviceState, GridState
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.storage import StorageBlock

# Device state
device_state = DeviceState(
    owner_id="battery1",
    owner_level=1,
    features=[
        ElectricalBasePh(P_MW=0.5, Q_MVAr=0.2),
        StorageBlock(soc=0.75, e_capacity_MWh=2.0, p_ch_max_MW=1.0, p_dsc_max_MW=1.0)
    ]
)

# Grid state
from powergrid.features.network import BusVoltages, NetworkMetrics

grid_state = GridState(
    owner_id="microgrid1",
    owner_level=2,
    features=[
        BusVoltages(vm_pu=[1.02, 0.98], va_deg=[0, -2.1]),
        NetworkMetrics(total_gen_mw=1.5, total_load_mw=1.2, total_loss_mw=0.05)
    ]
)
```

### Updating States

```python
# Update multiple features at once
state.update({
    "ElectricalBasePh": {"P_MW": 0.6, "Q_MVAr": 0.25},
    "StorageBlock": {"soc": 0.7}
})

# Update single feature
state.update_feature("StorageBlock", soc=0.8)

# Reset state
state.reset(overrides={"StorageBlock": {"soc": 0.5}})
```

### Observing States

```python
# Vectorize entire state
vector = state.vector()  # Returns flat numpy array

# Get observable features for specific agent
observable = state.observed_by(
    requestor_id="grid_agent",
    requestor_level=2  # Grid level
)
# Returns: {"ElectricalBasePh": array([...]), "StorageBlock": array([...])}

# Serialize to dictionary
state_dict = state.to_dict()
```

---

## Working with Actions

### Creating Actions

```python
from powergrid.core.action import Action
import numpy as np

# Continuous only
action = Action()
action.set_specs(
    dim_c=4,
    range=(
        np.array([-1.0, -0.5, -1.0, -0.5]),
        np.array([1.0, 0.5, 1.0, 0.5])
    )
)

# Continuous + discrete
action = Action()
action.set_specs(
    dim_c=2,
    dim_d=2,
    ncats=[3, 2],  # First discrete head has 3 categories, second has 2
    range=(
        np.array([-1.0, -0.5]),
        np.array([1.0, 0.5])
    )
)

# Discrete only
action = Action()
action.set_specs(dim_d=3, ncats=[5, 3, 2])
```

### Setting Action Values

```python
# From dictionary
action.set_values({"c": [0.5, 0.2], "d": [1, 0]})

# From flat vector
action.set_values([0.5, 0.2, 1, 0])  # [c..., d...]

# Via keyword arguments
action.set_values(c=[0.5, 0.2], d=[1, 0])

# Sample random
sampled = action.sample(seed=42)
```

### Normalization

```python
# Scale to [-1, 1]
normalized = action.scale()

# Unscale from [-1, 1] to physical
action.unscale([0.5, -0.3])  # Sets action.c

# Reset to neutral
action.reset()  # Midpoint for continuous, 0 for discrete

# Clip to valid ranges
action.clip()
```

### Gymnasium Space Conversion

```python
# Get Gymnasium space
space = action.space

# For continuous + discrete:
# Dict({"c": Box(...), "d": MultiDiscrete([3, 2])})

# Use with Gymnasium
assert space.contains(action.vector())
```

---

## Working with Observations

### Creating Observations

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
            sender="agent2",
            content={"price": 50.0},
            timestamp=10.0
        )
    ],
    timestamp=10.0
)

# Flatten to vector
vector = obs.vector()  # Concatenates local + global_info
```

### Creating Messages

```python
from powergrid.core.observation import Message

# Point-to-point message
msg = Message(
    sender="microgrid1",
    content={"price": 45.0, "power_offer": 0.8},
    recipient="microgrid2",
    timestamp=15.0
)

# Broadcast message
broadcast = Message(
    sender="system",
    content={"grid_frequency": 60.0},
    recipient=None,  # None = broadcast
    timestamp=15.0
)
```

---

## Common Patterns

### Pattern 1: Custom Device with Rule-Based Policy

```python
class RuleBasedBattery(DeviceAgent):
    def set_device_action(self):
        self.action.set_specs(dim_c=1, range=([-1.0], [1.0]))

    def _derive_local_action(self, upstream_action):
        """Ignore upstream, use rule-based logic"""
        obs = self.observe()
        soc = obs.local['state'][2]  # Assuming SOC is 3rd element

        # Simple rule: charge when SOC < 0.3, discharge when SOC > 0.7
        if soc < 0.3:
            power = 0.5  # Charge at 0.5 MW
        elif soc > 0.7:
            power = -0.5  # Discharge at 0.5 MW
        else:
            power = 0.0  # Idle

        self.action.set_values(c=[power])
        return self.action
```

### Pattern 2: Hierarchical Control with RL

```python
from powergrid.core.policies import MLPPolicy

class HierarchicalGrid(PowerGridAgent):
    def __init__(self, net, grid_config, **kwargs):
        # Create RL policy for grid-level decisions
        policy = MLPPolicy(
            obs_dim=grid_config['obs_dim'],
            action_dim=grid_config['action_dim']
        )

        super().__init__(
            net=net,
            grid_config=grid_config,
            policy=policy,
            **kwargs
        )

    async def _derive_downstream_actions(self, upstream_action):
        """Use RL policy to coordinate devices"""
        # RL policy outputs joint action
        joint_action = self.policy.forward(self.observe())

        # Decompose into per-device actions
        return await super()._derive_downstream_actions(joint_action)
```

### Pattern 3: Multi-Agent Communication

```python
from powergrid.messaging.base import MessageBroker
from powergrid.core.observation import Message

class CommunicativeAgent(GridAgent):
    def act(self, observation, upstream_action):
        """Act and communicate with peers"""
        # Normal action
        super().act(observation, upstream_action)

        # Send price signal to peers
        if self.message_broker:
            price_msg = Message(
                sender=self.agent_id,
                content={"electricity_price": self._compute_price()},
                recipient=None,  # Broadcast
                timestamp=self._timestep
            )
            self.message_broker.publish(
                f"agent/{self.agent_id}/price",
                price_msg
            )

    def _compute_price(self):
        """Compute local electricity price"""
        # Example: price based on load
        return 50.0 + self.state.features[0].total_load_mw * 5.0
```

### Pattern 4: Custom Reward Shaping

```python
class CustomRewardGrid(PowerGridAgent):
    def update_cost_safety(self, net):
        """Override to add custom reward shaping"""
        # Base cost and safety
        super().update_cost_safety(net)

        # Add custom penalty for large SOC variations
        soc_variance = 0.0
        for device in self.devices.values():
            if hasattr(device, 'storage'):
                storage = next(f for f in device.state.features if f.feature_name == "StorageBlock")
                soc_variance += (storage.soc - 0.5) ** 2

        self.safety += soc_variance * 10.0  # Penalize deviation from 50% SOC
```

---

## Debugging Tips

### Inspecting State

```python
# Print state features
for feature in state.features:
    print(f"{feature.feature_name}: {feature.to_dict()}")

# Check vector dimensions
print(f"State vector shape: {state.vector().shape}")
print(f"State vector: {state.vector()}")

# Verify observability
obs = state.observed_by("agent1", 1)
print(f"Observable features: {list(obs.keys())}")
```

### Inspecting Action

```python
# Print action specs
print(f"Continuous dim: {action.dim_c}")
print(f"Discrete dim: {action.dim_d}")
print(f"Categories: {action.ncats}")
print(f"Range: {action.range}")

# Check space
print(f"Gymnasium space: {action.space}")
print(f"Contains current action: {action.space.contains(action.vector())}")
```

### Tracing Message Flow

```python
# Enable message logging
class DebugBroker(InMemoryBroker):
    def publish(self, channel, message):
        print(f"[PUBLISH] {channel}: {message.payload}")
        super().publish(channel, message)

    def consume(self, channel, **kwargs):
        messages = super().consume(channel, **kwargs)
        print(f"[CONSUME] {channel}: {len(messages)} messages")
        return messages
```

---

## Testing Custom Components

### Unit Test Example

```python
import pytest
import numpy as np
from powergrid.core.state import DeviceState
from powergrid.features.electrical import ElectricalBasePh

def test_device_state_update():
    """Test state update functionality"""
    state = DeviceState(
        owner_id="test_device",
        owner_level=1,
        features=[ElectricalBasePh(P_MW=0.0, Q_MVAr=0.0)]
    )

    # Update state
    state.update({"ElectricalBasePh": {"P_MW": 0.5, "Q_MVAr": 0.2}})

    # Verify update
    vector = state.vector()
    assert vector[0] == 0.5  # P_MW
    assert vector[1] == 0.2  # Q_MVAr

def test_action_normalization():
    """Test action normalization"""
    action = Action()
    action.set_specs(
        dim_c=2,
        range=(np.array([0.0, -1.0]), np.array([1.0, 1.0]))
    )

    # Set physical values
    action.set_values(c=[0.5, 0.0])

    # Scale to normalized
    normalized = action.scale()

    # Should be [0.0, 0.0] (midpoint of ranges)
    assert np.allclose(normalized, [0.0, 0.0])
```

---

## Performance Tips

1. **Reuse Action/State Objects**: Don't recreate on every step
```python
# Good: Reuse
self.action.set_values(new_values)

# Bad: Recreate
self.action = Action(...)
```

2. **Batch State Updates**: Update multiple features at once
```python
# Good: Batch update
state.update({
    "Feature1": {...},
    "Feature2": {...}
})

# Bad: Sequential updates
state.update_feature("Feature1", ...)
state.update_feature("Feature2", ...)
```

3. **Cache Observations**: Don't recompute unnecessarily
```python
class CachingAgent(DeviceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_obs = None
        self._obs_timestep = -1

    def observe(self, *args, **kwargs):
        if self._timestep != self._obs_timestep:
            self._cached_obs = super().observe(*args, **kwargs)
            self._obs_timestep = self._timestep
        return self._cached_obs
```

---

## Next Steps

- **Full Examples**: See [examples/](../examples/) for complete implementations
- **API Reference**: Detailed API docs in [api/](../api/)
- **Architecture**: Deep dive in [architecture/overview.md](../architecture/overview.md)

# Devices

Device models in PowerGrid 2.0 represent physical distributed energy resources (DERs) with realistic dynamics and constraints.

---

## Device Hierarchy

All devices inherit from a common interface but implement specific physics:

```
Device (Abstract)
├── Generator (Diesel/Gas)
├── EnergyStorage (Battery)
├── RenewableGenerator (Solar/Wind)
├── Compensation (Shunt/Capacitor)
├── Transformer (OLTC)
└── Grid (Main Grid Connection)
```

---

## Common Device Interface

### Core Properties

Every device has:

```python
class Device:
    agent_id: str           # Unique identifier
    bus: str                # Connection bus in network
    P_MW: float            # Active power output
    Q_MVAr: float          # Reactive power output
    cost: float            # Operational cost
    safety: float          # Safety violations
```

### Key Methods

```python
def step(self, dt: float = 1.0):
    """Update device state (dynamics)"""

def compute_cost(self) -> float:
    """Calculate operational cost"""

def compute_safety(self, converged: bool) -> float:
    """Calculate safety violations"""

def get_state(self) -> Dict:
    """Return current state"""
```

---

## Generator

### Overview

Represents a **controllable generator** (diesel, natural gas, etc.)

### Parameters

```python
class Generator:
    p_min_MW: float              # Minimum power
    p_max_MW: float              # Maximum power
    s_rated_MVA: float           # Rated capacity
    cost_curve_coefs: List[float]  # [c0, c1, c2]
    ramp_rate_MW_per_min: float    # Ramping limit
```

### Cost Function

Quadratic cost curve:

```
cost = c0 + c1 * P + c2 * P²
```

**Example**:
- `[100, 72.4, 0.5011]` → ~$100/hr base + ~$72/MWh variable

### Constraints

1. **Power limits**: `p_min_MW ≤ P ≤ p_max_MW`
2. **Ramp rate**: `|P(t) - P(t-1)| ≤ ramp_rate * dt`
3. **Apparent power**: `√(P² + Q²) ≤ s_rated_MVA`

### Example Usage

```python
from powergrid.devices.generator import Generator

dg = Generator(
    agent_id='DG1',
    bus='Bus 675',
    p_min_MW=0.0,
    p_max_MW=0.66,
    s_rated_MVA=1.0,
    cost_curve_coefs=[100, 72.4, 0.5011],
    ramp_rate_MW_per_min=0.1
)

# Apply action
dg.P_MW = 0.5
dg.Q_MVAr = 0.1

# Update dynamics
dg.step(dt=0.25)  # 15-min timestep

# Compute cost
cost = dg.compute_cost()  # ~136.4
```

---

## Energy Storage (ESS)

### Overview

Represents a **battery storage system** with SOC dynamics.

### Parameters

```python
class ESS:
    p_min_MW: float          # Min power (negative = charging)
    p_max_MW: float          # Max power (positive = discharging)
    capacity_MWh: float      # Energy capacity
    max_e_MWh: float         # Max energy level
    min_e_MWh: float         # Min energy level
    init_soc: float          # Initial SOC (0-1)
    efficiency: float        # Round-trip efficiency
    degradation_cost: float  # $/MWh wear cost
```

### SOC Dynamics

```python
def step(self, dt: float):
    """Update SOC based on power flow."""
    # Charging: negative P
    if self.P_MW < 0:
        self.energy_MWh += -self.P_MW * dt * self.efficiency
    # Discharging: positive P
    else:
        self.energy_MWh -= self.P_MW * dt / self.efficiency

    # Enforce limits
    self.energy_MWh = np.clip(
        self.energy_MWh,
        self.min_e_MWh,
        self.max_e_MWh
    )
    self.SOC = self.energy_MWh / self.capacity_MWh
```

### Cost Function

Degradation cost based on throughput:

```
cost = degradation_cost * |P| * dt
```

**Example**: `$5/MWh` wear cost

### Constraints

1. **Power limits**: `p_min_MW ≤ P ≤ p_max_MW`
2. **Energy limits**: `min_e_MWh ≤ E ≤ max_e_MWh`
3. **SOC limits**: `min_soc ≤ SOC ≤ max_soc`

### Example Usage

```python
from powergrid.devices.storage import ESS

ess = ESS(
    agent_id='ESS1',
    bus='Bus 645',
    p_min_MW=-0.5,       # 0.5 MW charging
    p_max_MW=0.5,        # 0.5 MW discharging
    capacity_MWh=2.0,
    max_e_MWh=2.0,
    min_e_MWh=0.2,
    init_soc=0.5,
    efficiency=0.95,
    degradation_cost=5.0
)

# Charge for 15 minutes
ess.P_MW = -0.4  # 0.4 MW charging
ess.step(dt=0.25)
print(ess.SOC)  # ~0.547
```

---

## Renewable Generator

### Overview

Represents **solar or wind** generation with forecast-driven active power.

### Parameters

```python
class RenewableGenerator(Generator):
    source: str  # 'solar' or 'wind'
    forecast: np.ndarray  # Hourly generation forecast
```

### Behavior

- **Active power**: Set by forecast (not controllable)
- **Reactive power**: Controllable by agent
- **Cost**: Zero (renewable energy is free)

### Example Usage

```python
from powergrid.devices.generator import Generator

pv = Generator(
    agent_id='PV1',
    bus='Bus 652',
    p_min_MW=0.0,
    p_max_MW=0.1,
    s_rated_MVA=0.1,
    cost_curve_coefs=[0, 0, 0],
    source='solar'
)

# Active power from forecast
pv.P_MW = solar_forecast[timestep]  # e.g., 0.08 MW

# Agent controls reactive power
pv.Q_MVAr = 0.02  # Reactive support
```

---

## Other Devices

### Compensation (Shunt/Capacitor)

Provides reactive power compensation:

```python
from powergrid.devices.compensation import Compensation

shunt = Compensation(
    agent_id='Shunt1',
    bus='Bus 611',
    q_min_MVAr=0.0,
    q_max_MVAr=0.3
)
```

### Transformer (OLTC)

On-load tap changer for voltage regulation:

```python
from powergrid.devices.transformer import Transformer

oltc = Transformer(
    agent_id='OLTC1',
    from_bus='Bus 650',
    to_bus='Bus 632',
    tap_min=-10,
    tap_max=10,
    tap_step_percent=1.5
)
```

### Grid Connection

Represents main grid connection point:

```python
from powergrid.devices.grid import Grid

grid = Grid(
    agent_id='MainGrid',
    bus='Bus 650',
    price: float  # $/MWh
)
```

---

## Device State Management

### State Structure

```python
state = {
    'agent_id': 'ESS1',
    'device_type': 'storage',
    'P_MW': 0.5,
    'Q_MVAr': 0.1,
    'SOC': 0.75,
    'cost': 12.5,
    'safety': 0.0,
    'converged': True
}
```

### State Publishing (Distributed Mode)

Devices publish state updates to the environment:

```python
def _publish_state_updates(self):
    channel = ChannelManager.state_update_channel(self.env_id)
    message = Message(
        payload={
            'agent_id': self.agent_id,
            'P_MW': float(self.P_MW),
            'Q_MVAr': float(self.Q_MVAr),
            'SOC': float(getattr(self, 'SOC', 0.0)),
        }
    )
    self.message_broker.publish(channel, message)
```

---

## Safety Violations

### Types of Violations

1. **Power limit violations**:
   ```python
   if P > p_max_MW:
       safety += penalty * (P - p_max_MW)
   ```

2. **SOC violations** (ESS only):
   ```python
   if SOC < min_soc:
       safety += penalty * (min_soc - SOC)
   ```

3. **Ramp rate violations**:
   ```python
   if |P(t) - P(t-1)| > ramp_limit:
       safety += penalty * excess_ramp
   ```

### Penalty Scaling

```python
penalty_multiplier = 10.0  # From config
total_safety = sum(device.safety) * penalty_multiplier
```

---

## Integration with PandaPower

### Network Attachment

Devices are attached to PandaPower network elements:

```python
# Generator → sgen (static generator)
pp.create_sgen(
    net,
    bus=bus_idx,
    p_mw=device.P_MW,
    q_mvar=device.Q_MVAr,
    name=device.agent_id
)

# ESS → storage
pp.create_storage(
    net,
    bus=bus_idx,
    p_mw=device.P_MW,
    max_e_mwh=device.max_e_MWh,
    soc_percent=device.SOC * 100,
    name=device.agent_id
)
```

### State Synchronization

After each step:

1. **Devices → Network**: Apply P, Q setpoints
2. **Run power flow**: `pp.runpp(net)`
3. **Network → Devices**: Read voltages, loading for safety check

---

## Performance Considerations

### Computational Cost

- **Device step**: ~0.01 ms per device
- **Cost computation**: ~0.001 ms per device
- **Safety check**: ~0.001 ms per device

**Total for 100 devices**: ~1 ms (negligible)

### Memory Usage

- Each device: ~1 KB
- 1000 devices: ~1 MB (negligible)

---

## Adding Custom Devices

### Template

```python
from powergrid.devices.generator import Generator

class MyDevice(Generator):
    def __init__(self, agent_id, bus, custom_param, **kwargs):
        super().__init__(agent_id, bus, **kwargs)
        self.custom_param = custom_param

    def step(self, dt=1.0):
        """Custom dynamics"""
        # Your logic here
        self.P_MW = self._compute_output()
        super().step(dt)

    def compute_cost(self):
        """Custom cost function"""
        self.cost = self.custom_param * self.P_MW

    def compute_safety(self, converged):
        """Custom safety checks"""
        super().compute_safety(converged)
        # Add custom violations
        if self.custom_condition:
            self.safety += penalty
```

### Registration

Add to device factory in environment:

```python
DEVICE_TYPES = {
    'generator': Generator,
    'storage': ESS,
    'my_device': MyDevice,  # Add custom device
}
```

---

## Next Steps

- **API Reference**: Full device API in [API: Devices](../api/devices.rst)
- **Configuration**: Device parameters in [Configuration Guide](../user_guide/configuration.md)
- **Agents**: How agents control devices in [Agents](agents.md)

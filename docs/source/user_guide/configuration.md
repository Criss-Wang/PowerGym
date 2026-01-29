# Configuration Guide

This guide explains all configuration options for PowerGrid 2.0 environments.

---

## Configuration Methods

### Method 1: YAML File (Recommended)

```yaml
# config.yml
train: true
centralized: false
penalty: 10
episode_length: 96

microgrid_configs:
  - name: "MG1"
    network: "IEEE13Bus"
    devices: [...]
```

Load in Python:

```python
from powergrid.envs.multi_agent import MultiAgentMicrogrids
import yaml

with open('config.yml') as f:
    config = yaml.safe_load(f)

env = MultiAgentMicrogrids(config)
```

### Method 2: Python Dictionary

```python
config = {
    'train': True,
    'centralized': False,
    'penalty': 10,
    'episode_length': 96,
    'microgrid_configs': [...]
}

env = MultiAgentMicrogrids(config)
```

### Method 3: Config Loader

```python
from powergrid.envs.configs.config_loader import load_config

config = load_config('powergrid/envs/configs/ieee34_ieee13.yml')
env = MultiAgentMicrogrids(config)
```

---

## General Settings

### Core Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train` | bool | `true` | Training mode (True) or evaluation (False) |
| `centralized` | bool | `false` | Centralized (True) or distributed (False) mode |
| `episode_length` | int | `96` | Number of timesteps per episode |
| `penalty` | float | `10.0` | Multiplier for safety violation penalties |
| `share_reward` | bool | `false` | All agents share same reward (True) or individual (False) |

**Example**:

```yaml
train: true
centralized: false
episode_length: 96
penalty: 10
share_reward: true
```

### Distributed Mode Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message_broker` | str | `'in_memory'` | Broker type: `'in_memory'` or `'kafka'` |
| `convergence_failure_reward` | float | `-200.0` | Reward penalty for power flow non-convergence |
| `convergence_failure_safety` | float | `20.0` | Safety penalty for non-convergence |

**Example**:

```yaml
centralized: false
message_broker: 'in_memory'
convergence_failure_reward: -200.0
convergence_failure_safety: 20.0
```

---

## Microgrid Configuration

### Basic Structure

```yaml
microgrid_configs:
  - name: "MG1"
    network: "IEEE13Bus"
    load_scale: 0.2
    base_power: 3
    load_area: "AVA"
    renew_area: "NP15"
    connection_bus: "DSO Bus 822"
    devices: [...]
```

### Microgrid Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | ✅ | Unique identifier for microgrid |
| `network` | str | ✅ | PandaPower network: `"IEEE13Bus"`, `"IEEE34Bus"` |
| `load_scale` | float | ❌ | Scale factor for loads (default: 1.0) |
| `base_power` | float | ❌ | Base power in MVA (default: 1.0) |
| `load_area` | str | ❌ | Load profile area from dataset |
| `renew_area` | str | ❌ | Renewable generation area from dataset |
| `connection_bus` | str | ❌ | Bus name for connecting to main grid |
| `devices` | list | ✅ | List of device configurations |

---

## Device Configuration

### Generator (DG)

```yaml
- type: "Generator"
  name: "DG1"
  bus: "Bus 675"
  p_min_MW: 0.0
  p_max_MW: 0.66
  s_rated_MVA: 1.0
  cost_curve_coefs: [100, 72.4, 0.5011]
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | str | Must be `"Generator"` |
| `name` | str | Unique device name |
| `bus` | str | Bus name in network |
| `p_min_MW` | float | Minimum active power (MW) |
| `p_max_MW` | float | Maximum active power (MW) |
| `s_rated_MVA` | float | Rated apparent power (MVA) |
| `cost_curve_coefs` | list | Cost coefficients [c0, c1, c2] for cost = c0 + c1*P + c2*P² |

### Renewable Generator (Solar/Wind)

```yaml
- type: "Generator"
  name: "PV1"
  bus: "Bus 652"
  p_min_MW: 0.0
  p_max_MW: 0.1
  s_rated_MVA: 0.1
  cost_curve_coefs: [0, 0, 0]
  source: "solar"  # or "wind"
```

**Additional Parameters**:

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `source` | str | `"solar"`, `"wind"` | Renewable source type |

**Note**: Active power is forecast-driven; agent controls reactive power only.

### Energy Storage System (ESS)

```yaml
- type: "Storage"
  name: "ESS1"
  bus: "Bus 645"
  p_min_MW: -0.5
  p_max_MW: 0.5
  s_rated_MVA: 1.0
  capacity_MWh: 2.0
  max_e_MWh: 2.0
  min_e_MWh: 0.2
  init_soc: 0.5
  efficiency: 0.95
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | str | Must be `"Storage"` |
| `p_min_MW` | float | Min power (negative = charging) |
| `p_max_MW` | float | Max power (positive = discharging) |
| `capacity_MWh` | float | Energy capacity (MWh) |
| `max_e_MWh` | float | Maximum energy level |
| `min_e_MWh` | float | Minimum energy level |
| `init_soc` | float | Initial state of charge (0-1) |
| `efficiency` | float | Round-trip efficiency (0-1) |

---

## DSO (Main Grid) Configuration

```yaml
dso_config:
  name: "DSO"
  network: "IEEE34Bus"
  load_scale: 0.2
  base_power: 3
  load_area: "BANC"
  renew_area: "NP15"
```

**Parameters**: Same as microgrid configuration, but represents the main distribution grid that microgrids connect to.

---

## Data Configuration

```yaml
data:
  dataset_path: "data/data2023-2024.pkl"
  price_area: "0096WD_7_N001"
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_path` | str | Path to pickle file with load/solar/wind/price data |
| `price_area` | str | Price area identifier for electricity pricing |

### Dataset Format

The dataset pickle file should contain:

```python
{
    'load': {
        'AVA': pd.Series(...),     # Load profiles
        'BANCMID': pd.Series(...),
        ...
    },
    'solar': {
        'NP15': pd.Series(...),    # Solar generation
        ...
    },
    'wind': {
        'NP15': pd.Series(...),    # Wind generation
        ...
    },
    'price': {
        'LMP': pd.Series(...),     # Electricity prices
        ...
    }
}
```

---

## Complete Example

```yaml
# Complete configuration for 3-microgrid system
train: true
centralized: false
penalty: 10
share_reward: true
convergence_failure_reward: -200.0
convergence_failure_safety: 20.0
episode_length: 96

# Main grid
dso_config:
  name: "DSO"
  network: "IEEE34Bus"
  load_scale: 0.2
  base_power: 3
  load_area: "BANC"
  renew_area: "NP15"

# Microgrids
microgrid_configs:
  # Microgrid 1
  - name: "MG1"
    network: "IEEE13Bus"
    load_scale: 0.2
    base_power: 3
    load_area: "AVA"
    renew_area: "NP15"
    connection_bus: "DSO Bus 822"
    devices:
      - type: "Generator"
        name: "DG1"
        bus: "Bus 675"
        p_min_MW: 0.0
        p_max_MW: 0.66
        s_rated_MVA: 1.0
        cost_curve_coefs: [100, 72.4, 0.5011]

      - type: "Generator"
        name: "PV1"
        bus: "Bus 652"
        p_min_MW: 0.0
        p_max_MW: 0.1
        s_rated_MVA: 0.1
        cost_curve_coefs: [0, 0, 0]
        source: "solar"

      - type: "Storage"
        name: "ESS1"
        bus: "Bus 645"
        p_min_MW: -0.5
        p_max_MW: 0.5
        s_rated_MVA: 1.0
        capacity_MWh: 2.0
        max_e_MWh: 2.0
        min_e_MWh: 0.2
        init_soc: 0.5
        efficiency: 0.95

  # Add MG2, MG3 similarly...

# Data sources
data:
  dataset_path: "data/data2023-2024.pkl"
  price_area: "0096WD_7_N001"
```

---

## Advanced Options

### Protocol Configuration (Future)

```yaml
# Vertical protocols (per microgrid)
microgrid_configs:
  - name: "MG1"
    vertical_protocol: "price_signal"
    protocol_params:
      initial_price: 50.0

# Horizontal protocols (environment-wide)
horizontal_protocol: "p2p_trading"
protocol_params:
  trading_fee: 0.01
```

### Network Customization

```python
# Use custom PandaPower network
import pandapower as pp

custom_net = pp.create_empty_network()
# ... build network ...

config = {
    'microgrid_configs': [
        {
            'name': 'MG1',
            'network': custom_net,  # Pass network object directly
            'devices': [...]
        }
    ]
}
```

---

## Configuration Validation

The environment validates configuration on initialization:

```python
try:
    env = MultiAgentMicrogrids(config)
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

**Common Validation Errors**:
- Missing required fields (`name`, `network`, `devices`)
- Invalid device power limits (`p_min_MW` > `p_max_MW`)
- Bus not found in network
- Duplicate device/microgrid names

---

## Best Practices

### ✅ Do's

- **Use YAML files** for complex configurations
- **Version control** your config files
- **Validate early** by creating environment in test mode
- **Document** custom parameters in comments
- **Scale appropriately** (load_scale, base_power) for numerical stability

### ❌ Don'ts

- **Don't hardcode** paths (use relative paths)
- **Don't mix units** (MW vs kW, MWh vs kWh)
- **Don't skip validation** (check config before long training runs)
- **Don't reuse** device names across microgrids without prefixes

---

## Next Steps

- **Try examples**: See [Getting Started](../getting_started.md) for runnable examples
- **Understand modes**: Read [Centralized vs Distributed](centralized_vs_distributed.md)
- **Advanced protocols**: Explore [Protocol Guide](../api/heron/protocols)

---

## Configuration Reference

For the most up-to-date configuration options, see:
- Example config: `powergrid/envs/configs/ieee34_ieee13.yml`
- Config loader: `powergrid/envs/configs/config_loader.py`
- Environment code: `powergrid/envs/multi_agent/networked_grid_env.py`

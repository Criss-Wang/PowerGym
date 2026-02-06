# Building a Power Grid Case Study from Scratch

This guide walks you through building the complete power grid multi-agent reinforcement learning case study from scratch. By the end, you'll understand how to create custom agents, environments, and run distributed RL training using the HERON framework.

**Target Audience**: Developers familiar with Python and basic RL who want to:
- Understand how the power grid case study is structured
- Create their own custom multi-agent environments
- Extend the framework with new device types or protocols

---

## What You'll Build

By following this guide, you'll create a complete multi-agent power grid simulation:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Multi-Agent Microgrid System                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐             │
│    │ Microgrid 1 │     │ Microgrid 2 │     │ Microgrid 3 │             │
│    │  (Agent 1)  │     │  (Agent 2)  │     │  (Agent 3)  │             │
│    ├─────────────┤     ├─────────────┤     ├─────────────┤             │
│    │ Generator   │     │ Generator   │     │ ESS         │             │
│    │ ESS         │     │             │     │             │             │
│    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘             │
│           │                   │                   │                     │
│           └───────────────────┼───────────────────┘                     │
│                               │                                         │
│                    ┌──────────┴──────────┐                              │
│                    │  Distribution Grid   │                              │
│                    │   (IEEE 34-bus)      │                              │
│                    └─────────────────────┘                              │
│                                                                         │
│  Training: MAPPO/IPPO via RLlib    |    Simulation: PandaPower         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Components you'll implement**:
1. **Features**: Observable state (voltage, power, SOC)
2. **Device Agents**: Generator, ESS (Battery), optionally WindTurbine
3. **Grid Agent**: Coordinator managing multiple devices
4. **Environment**: PettingZoo-compatible multi-agent RL environment
5. **Training**: MAPPO/IPPO with RLlib

---

## Table of Contents

1. [Quick Start: Run Examples First](#quick-start-run-examples-first)
2. [Prerequisites](#1-prerequisites)
3. [Project Initialization](#2-project-initialization)
4. [Understanding the Architecture](#3-understanding-the-architecture)
   - [Key Concepts](#key-concepts)
   - [Execution Modes](#execution-modes)
   - [Data Flow Diagram](#data-flow-diagram)
   - [Observation Space Construction](#observation-space-construction)
5. [Step 1: Define Domain Features](#step-1-define-domain-features)
6. [Step 2: Create Device Agents](#step-2-create-device-agents)
7. [Step 3: Build the Grid Agent (Coordinator)](#step-3-build-the-grid-agent-coordinator)
8. [Step 4: Create the Environment](#step-4-create-the-environment)
9. [Step 5: Configure Setups and Datasets](#step-5-configure-setups-and-datasets)
10. [Step 6: Run RL Training](#step-6-run-rl-training)
    - [MAPPO vs IPPO](#63-mappo-vs-ippo-when-to-use-each)
    - [Reward Engineering](#64-reward-engineering)
11. [Step 7: Event-Driven Testing](#step-7-event-driven-testing-key-heron-differentiator) ⭐ **Key Differentiator**
12. [Advanced Topics](#advanced-topics)
13. [Troubleshooting](#troubleshooting)
14. [Summary](#summary)

---

## Quick Start: Run Examples First

Before building from scratch, **run the existing examples** to understand how the system works:

```bash
# Navigate to the case study directory
cd case_studies/power

# Activate virtual environment
source ../../.venv/bin/activate

# Run examples in order of complexity:

# 1. Single microgrid (simplest) - ~30 seconds
python examples/01_single_microgrid_basic.py

# 2. Multi-microgrid with P2P trading - ~1 minute
python examples/02_multi_microgrid_p2p.py

# 3. MAPPO training (quick test) - ~2 minutes
python examples/05_mappo_training.py --test
```

Once you understand the examples, proceed to build your own from scratch.

---

## 1. Prerequisites

### System Requirements

```bash
# Python 3.10 or higher required
python --version  # Should show 3.10+

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install HERON framework with all dependencies
pip install -e ".[all]"
```

### Version Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.10 | Base runtime |
| pandapower | >= 2.13 | Power flow simulation |
| gymnasium | >= 1.0.0 | RL environment interface |
| pettingzoo | >= 1.24.0 | Multi-agent environments |
| ray[rllib] | >= 2.9.0 | Distributed RL training |
| numpy | >= 1.21.0 | Numerical computation |

### Required Knowledge

- Python object-oriented programming (classes, inheritance, dataclasses)
- Basic reinforcement learning concepts (observations, actions, rewards)
- Familiarity with PettingZoo multi-agent environments
- (Optional) Power systems basics for domain understanding

---

## 2. Project Initialization

> **Note**: The power grid case study already exists in `case_studies/power/`. This section explains how to create a **new** case study from scratch using the HERON template.

For **new** HERON projects, use the Makefile to scaffold the structure:

```bash
# From project root (for creating NEW projects only)
make new-project NAME=my_case_study DOMAIN=powergrid
```

This creates the following structure:

```
my_case_study/
├── powergrid/
│   ├── agents/           # Custom agent implementations
│   │   └── __init__.py
│   ├── envs/             # Environment definitions
│   │   └── __init__.py
│   └── utils/            # Domain utilities
│       └── __init__.py
├── examples/             # Runnable example scripts
├── tests/                # Test suite
├── pyproject.toml        # Python package configuration
└── README.md
```

**For the power grid case study**, the complete structure looks like:

```
case_studies/power/
├── powergrid/
│   ├── agents/           # GridAgent, DeviceAgent, Generator, ESS
│   ├── core/
│   │   ├── features/     # Electrical, network, storage features
│   │   └── state/        # DeviceState, GridState
│   ├── envs/             # NetworkedGridEnv, MultiAgentMicrogrids
│   ├── networks/         # IEEE test feeders (13, 34, 123-bus)
│   ├── setups/           # Pre-configured environments
│   ├── optimization/     # SOCP solvers
│   └── utils/            # Cost, safety functions
├── examples/
└── tests/
```

---

## 3. Understanding the Architecture

The HERON framework uses a **hierarchical agent architecture**:

```
┌────────────────────────────────────────────────────────────┐
│                    HERON Framework                          │
├────────────────────────────────────────────────────────────┤
│  Agent (ABC) ─────────────────────────────────────────────  │
│       │                                                     │
│       ├── FieldAgent (L1 - devices)                         │
│       │       ↑ extends                                     │
│       │       └─→ DeviceAgent                               │
│       │           ├─→ Generator                             │
│       │           ├─→ ESS (Energy Storage)                  │
│       │           └─→ Transformer                           │
│       │                                                     │
│       ├── HierarchicalAgent (shared base for L2/L3)         │
│       │       │                                             │
│       │       ├── CoordinatorAgent (L2 - coordinators)      │
│       │       │       ↑ extends                             │
│       │       │       └─→ GridAgent                         │
│       │       │           ↑ extends                         │
│       │       │           └─→ PowerGridAgent (PandaPower)   │
│       │       │                                             │
│       │       └── SystemAgent (L3 - system operators)       │
│       │               ↑ extends                             │
│       │               └─→ DSOAgent                          │
│       │                                                     │
│       └── ProxyAgent (L0 - state distribution)              │
└────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Agent** | Abstract base class with core methods: `reset()`, `observe()`, `act()`, `tick()` |
| **FieldAgent** | L1 - Lowest-level agent controlling a single device |
| **HierarchicalAgent** | Shared base for L2/L3 with subordinate management |
| **CoordinatorAgent** | L2 - Manages multiple FieldAgents, coordinates actions |
| **SystemAgent** | L3 - Top-level system operator, manages CoordinatorAgents |
| **ProxyAgent** | L0 - State distribution with visibility filtering |
| **Protocol** | Communication pattern (SetpointProtocol, PriceSignalProtocol) |
| **FeatureProvider** | Observable state component with visibility rules |
| **State** | Collection of FeatureProviders representing agent's condition |
| **TickConfig** | Timing configuration for event-driven mode |

### Execution Modes

**Centralized Mode** (`centralized=True`):
- Agents directly access the power network object
- Synchronous state updates
- Full observability
- Suitable for training

**Distributed Mode** (`centralized=False`):
- Agents communicate via message broker
- Asynchronous state updates
- Realistic partial observability
- Suitable for deployment

### Data Flow Diagram

Understanding how data flows through the system is critical:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENVIRONMENT STEP                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    action    ┌──────────────┐    setpoints   ┌─────────┐ │
│  │   RL Policy  │ ──────────── │  GridAgent   │ ─────────────── │ Devices │ │
│  │   (MAPPO)    │              │ (Coordinator)│                 │Gen, ESS │ │
│  └──────────────┘              └──────────────┘                 └─────────┘ │
│         ▲                            │                              │       │
│         │                            │ update network               │       │
│         │                            ▼                              │       │
│         │                     ┌──────────────┐                      │       │
│         │                     │  PandaPower  │ ◄────────────────────┘       │
│         │                     │  Power Flow  │   P, Q setpoints             │
│         │                     └──────────────┘                              │
│         │                            │                                      │
│         │                            │ voltages, line loading               │
│         │                            ▼                                      │
│         │                     ┌──────────────┐                              │
│         │                     │  Sync State  │                              │
│         │                     │  + Compute   │                              │
│         │                     │  Cost/Safety │                              │
│         │                     └──────────────┘                              │
│         │                            │                                      │
│         │        observation         │                                      │
│         └────────────────────────────┘                                      │
│                                                                             │
│  reward = -cost - penalty * safety_violation                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Observation Space Construction

The RL observation is built by concatenating features from all devices and network state:

```
Observation Vector (per GridAgent):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Device 1 (Generator)  │ Device 2 (ESS)        │ Network State              │
├───────────────────────┼───────────────────────┼────────────────────────────┤
│ p_MW                  │ p_MW                  │ v_min                      │
│ q_MVAr                │ q_MVAr                │ v_max                      │
│ v_pu                  │ v_pu                  │ max_line_loading           │
│ p_max, p_min          │ soc                   │ total_load                 │
│ status                │ e_capacity            │ price (if available)       │
└───────────────────────┴───────────────────────┴────────────────────────────┘
```

Each agent's observation dimension depends on its devices. The action space is similarly constructed by concatenating device action spaces.

---

## Step 1: Define Domain Features

Features represent observable state components with **declarative visibility**. This is a key HERON contribution—features declare who can see them, eliminating manual observation filtering.

### 1.1 Understanding FeatureProvider

The HERON `FeatureProvider` is an abstract base class that defines the interface for state features:

```python
# Key methods to implement
class FeatureProvider(ABC):
    visibility: List[str]  # Who can observe this feature

    @abstractmethod
    def vector(self) -> np.ndarray:
        """Convert to numpy array for RL observations."""

    @abstractmethod
    def names(self) -> List[str]:
        """Field names corresponding to vector elements."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize for communication/logging."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "FeatureProvider":
        """Deserialize from dictionary."""

    def set_values(self, **kwargs) -> None:
        """Update values from keyword arguments."""

    def reset(self, **overrides) -> "FeatureProvider":
        """Reset to initial state."""
```

### Visibility Options

- `"public"`: Visible to all agents
- `"owner"`: Only the owning agent
- `"upper_level"`: Agents one level above
- `"system"`: System-level (L3) only

### 1.2 Electrical Features

```python
# powergrid/core/features/electrical.py
from heron.core import FeatureProvider
from typing import List
import numpy as np

class ElectricalBasePh(FeatureProvider):
    """Base electrical features for single-phase devices.

    Key HERON pattern: visibility declares who can see this feature.
    - 'owner': The device itself
    - 'upper_level': Parent coordinator (microgrid)
    - 'system': System operator (DSO)
    """

    # Visibility: device + coordinator + system can see electrical state
    visibility = ['owner', 'upper_level', 'system']

    def __init__(self, p_MW=0.0, q_MVAr=0.0, v_pu=1.0, va_deg=0.0):
        # Active power (MW)
        self.p_MW = p_MW
        # Reactive power (MVAr)
        self.q_MVAr = q_MVAr
        # Voltage magnitude (per-unit)
        self.v_pu = v_pu
        # Voltage angle (degrees)
        self.va_deg = va_deg

    def vector(self) -> np.ndarray:
        """Convert to numpy array for RL observation."""
        return np.array([self.p_MW, self.q_MVAr, self.v_pu, self.va_deg], dtype=np.float32)

    def names(self) -> List[str]:
        """Feature names for debugging."""
        return ['p_MW', 'q_MVAr', 'v_pu', 'va_deg']

    def to_dict(self) -> dict:
        return {'p_MW': self.p_MW, 'q_MVAr': self.q_MVAr, 'v_pu': self.v_pu, 'va_deg': self.va_deg}

    @classmethod
    def from_dict(cls, d: dict) -> "ElectricalBasePh":
        return cls(**d)

    def set_values(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    def reset(self, **overrides) -> "ElectricalBasePh":
        self.p_MW = overrides.get('p_MW', 0.0)
        self.q_MVAr = overrides.get('q_MVAr', 0.0)
        self.v_pu = overrides.get('v_pu', 1.0)
        self.va_deg = overrides.get('va_deg', 0.0)
        return self
```

### 1.3 Storage Features

```python
# powergrid/core/features/storage.py
from heron.core import FeatureProvider
from typing import List
import numpy as np

class StorageBlock(FeatureProvider):
    """State-of-charge and capacity features for storage devices.

    Visibility: owner + upper_level (coordinator needs SOC for dispatch decisions)
    """

    visibility = ['owner', 'upper_level']

    def __init__(self, soc=0.5, e_capacity_MWh=1.0, ch_eff=0.95, dsc_eff=0.95):
        # Current state of charge (0-1)
        self.soc = soc
        # Energy capacity (MWh)
        self.e_capacity_MWh = e_capacity_MWh
        # Current energy stored (MWh)
        self.e_stored_MWh = soc * e_capacity_MWh
        # Charge efficiency
        self.ch_eff = ch_eff
        # Discharge efficiency
        self.dsc_eff = dsc_eff

    def vector(self) -> np.ndarray:
        return np.array([self.soc, self.e_capacity_MWh, self.e_stored_MWh], dtype=np.float32)

    def names(self) -> List[str]:
        return ['soc', 'e_capacity_MWh', 'e_stored_MWh']

    def to_dict(self) -> dict:
        return {'soc': self.soc, 'e_capacity_MWh': self.e_capacity_MWh,
                'e_stored_MWh': self.e_stored_MWh, 'ch_eff': self.ch_eff, 'dsc_eff': self.dsc_eff}

    @classmethod
    def from_dict(cls, d: dict) -> "StorageBlock":
        f = cls(d.get('soc', 0.5), d.get('e_capacity_MWh', 1.0))
        f.e_stored_MWh = d.get('e_stored_MWh', f.e_stored_MWh)
        f.ch_eff = d.get('ch_eff', 0.95)
        f.dsc_eff = d.get('dsc_eff', 0.95)
        return f

    def set_values(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    def reset(self, **overrides) -> "StorageBlock":
        self.soc = overrides.get('soc', 0.5)
        self.e_stored_MWh = self.soc * self.e_capacity_MWh
        return self
```

### 1.4 Network Features

```python
# powergrid/core/features/network.py
from heron.core import FeatureProvider
from typing import List
import numpy as np

class BusVoltages(FeatureProvider):
    """Aggregated bus voltage information.

    Visibility: 'system' - only DSO/system operator sees full network voltages
    """

    visibility = ['system']

    def __init__(self, vm_pu=None, v_min=1.0, v_max=1.0):
        # Voltage magnitudes for all buses (per-unit)
        self.vm_pu = vm_pu if vm_pu is not None else np.array([1.0])
        # Minimum voltage in network
        self.v_min = v_min
        # Maximum voltage in network
        self.v_max = v_max

    def vector(self) -> np.ndarray:
        return np.array([self.v_min, self.v_max], dtype=np.float32)

    def names(self) -> List[str]:
        return ['v_min', 'v_max']

    def to_dict(self) -> dict:
        return {'vm_pu': self.vm_pu.tolist(), 'v_min': self.v_min, 'v_max': self.v_max}

    @classmethod
    def from_dict(cls, d: dict) -> "BusVoltages":
        return cls(
            vm_pu=np.array(d.get('vm_pu', [1.0])),
            v_min=d.get('v_min', 1.0),
            v_max=d.get('v_max', 1.0)
        )

    def set_values(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def reset(self, **overrides) -> "BusVoltages":
        self.vm_pu = overrides.get('vm_pu', np.array([1.0]))
        self.v_min = overrides.get('v_min', 1.0)
        self.v_max = overrides.get('v_max', 1.0)
        return self


class LineFlows(FeatureProvider):
    """Line power flow information.

    Visibility: 'system' - only DSO sees line loading
    """

    visibility = ['system']

    def __init__(self, p_MW=None, loading_pct=None, max_loading=0.0):
        # Active power flows (MW)
        self.p_MW = p_MW if p_MW is not None else np.array([0.0])
        # Loading percentage (0-100%)
        self.loading_pct = loading_pct if loading_pct is not None else np.array([0.0])
        # Maximum loading across all lines
        self.max_loading = max_loading

    def vector(self) -> np.ndarray:
        return np.array([self.max_loading / 100.0], dtype=np.float32)

    def names(self) -> List[str]:
        return ['max_loading_norm']

    def to_dict(self) -> dict:
        return {'p_MW': self.p_MW.tolist(), 'loading_pct': self.loading_pct.tolist(),
                'max_loading': self.max_loading}

    @classmethod
    def from_dict(cls, d: dict) -> "LineFlows":
        return cls(
            p_MW=np.array(d.get('p_MW', [0.0])),
            loading_pct=np.array(d.get('loading_pct', [0.0])),
            max_loading=d.get('max_loading', 0.0)
        )

    def set_values(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
```

---

## Step 2: Create Device Agents

Device agents control individual power system components. They extend HERON's `FieldAgent`.

### 2.1 Base Device Agent

```python
# powergrid/agents/device_agent.py
from typing import Any, Dict, Optional
from heron.agents import FieldAgent
from heron.core import FieldAgentState, Policy
from heron.protocols import Protocol
from heron.scheduling import TickConfig
import numpy as np

# Level constants (from heron.agents.base)
FIELD_LEVEL = 1

class DeviceAgent(FieldAgent):
    """Base class for power device agents.

    Extends HERON's FieldAgent with power-grid specific functionality.

    Key HERON methods to override:
    - set_action(): Define action space via self.action.set_specs()
    - set_state(): Add features via self.state.features.append()
    - reset_agent(): Custom reset logic (optional)
    - update_state(): Handle environment feedback (optional)
    """

    def __init__(
        self,
        agent_id: str,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
        device_state_config: Dict[str, Any] = None,
        tick_config: Optional[TickConfig] = None,
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            policy=policy,
            protocol=protocol,
            tick_config=tick_config,
            **kwargs
        )

        # Store configuration
        self._config = device_state_config or {}

        # Initialize cost and safety metrics
        self.cost = 0.0
        self.safety = 0.0

    def set_state(self):
        """Initialize device-specific state. Override in subclasses."""
        self.state = FieldAgentState(
            owner_id=self.agent_id,
            owner_level=FIELD_LEVEL
        )

    def set_action(self):
        """Define device action space. Override in subclasses.

        Example:
            self.action.set_specs(
                dim_c=2,
                range=(np.array([0.0, -1.0]), np.array([10.0, 1.0]))
            )
        """
        raise NotImplementedError("Subclasses must implement set_action()")

    def reset_agent(self, **kwargs):
        """Reset device to initial state. Override in subclasses."""
        self.state.reset()
        self.cost = 0.0
        self.safety = 0.0

    def update_state(self, **kwargs) -> None:
        """Update state from environment feedback. Override in subclasses."""
        pass

    def update_cost_safety(self) -> None:
        """Compute cost and safety metrics. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement update_cost_safety()")
```

### 2.2 Generator Agent

```python
# powergrid/agents/generator.py
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.power_limits import PowerLimits
from powergrid.utils.cost import quadratic_cost

@dataclass
class GeneratorConfig:
    """Configuration for dispatchable generator."""
    bus: str
    p_max_MW: float = 10.0
    p_min_MW: float = 0.0
    q_max_MVAr: float = 5.0
    q_min_MVAr: float = -5.0
    s_rated_MVA: float = 12.0
    cost_curve_coefs: tuple = (0.02, 10.0, 0.0)  # a*P^2 + b*P + c
    startup_time_hr: float = 1.0
    shutdown_time_hr: float = 1.0


class Generator(DeviceAgent):
    """Dispatchable generator device agent."""

    def __init__(self, agent_id: str, device_state_config: Dict[str, Any], **kwargs):
        super().__init__(agent_id=agent_id, device_state_config=device_state_config, **kwargs)

        # Parse configuration
        self.config = GeneratorConfig(**device_state_config)

        # PandaPower element index (set by GridAgent)
        self.pp_idx: int = -1

    def set_state(self):
        """Initialize generator state with features."""
        super().set_state()

        # Add features to state
        self.state.features.append(ElectricalBasePh())
        self.state.features.append(PowerLimits(
            p_max_MW=self.config.p_max_MW,
            p_min_MW=self.config.p_min_MW,
            q_max_MVAr=self.config.q_max_MVAr,
            q_min_MVAr=self.config.q_min_MVAr,
        ))

    def set_action(self):
        """Define action space: [P_setpoint, Q_setpoint] normalized to [-1, 1]."""
        self.action.set_specs(
            dim_c=2,
            range=(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        )

    def apply_action(self, action: np.ndarray) -> None:
        """Apply action to set power output.

        Args:
            action: Array [p_norm, q_norm] in range [-1, 1]
        """
        # Denormalize action to actual power values
        p_range = self.config.p_max_MW - self.config.p_min_MW
        q_range = self.config.q_max_MVAr - self.config.q_min_MVAr

        p_setpoint = self.config.p_min_MW + (action[0] + 1) / 2 * p_range
        q_setpoint = self.config.q_min_MVAr + (action[1] + 1) / 2 * q_range

        # Clip to limits
        p_setpoint = np.clip(p_setpoint, self.config.p_min_MW, self.config.p_max_MW)
        q_setpoint = np.clip(q_setpoint, self.config.q_min_MVAr, self.config.q_max_MVAr)

        # Update state via feature
        self.state.update_feature("ElectricalBasePh", p_MW=p_setpoint, q_MVAr=q_setpoint)

    def update_cost_safety(self) -> None:
        """Compute generation cost using quadratic cost function."""
        # Get electrical feature values
        electrical = next(f for f in self.state.features if f.__class__.__name__ == "ElectricalBasePh")
        p = electrical.p_MW

        a, b, c = self.config.cost_curve_coefs
        self.cost = quadratic_cost(p, a, b, c)

        # Safety: check if within limits
        self.safety = 0.0
        if p > self.config.p_max_MW or p < self.config.p_min_MW:
            self.safety += abs(p - np.clip(p, self.config.p_min_MW, self.config.p_max_MW))
```

### 2.3 Energy Storage Agent

```python
# powergrid/agents/storage.py
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.storage import StorageBlock

@dataclass
class StorageConfig:
    """Configuration for energy storage system."""
    bus: str
    e_capacity_MWh: float = 5.0
    p_max_MW: float = 1.0      # Max discharge
    p_min_MW: float = -1.0     # Max charge (negative)
    soc_max: float = 0.9
    soc_min: float = 0.1
    init_soc: float = 0.5
    ch_eff: float = 0.95
    dsc_eff: float = 0.95


class ESS(DeviceAgent):
    """Energy Storage System device agent."""

    def __init__(self, agent_id: str, device_state_config: Dict[str, Any], **kwargs):
        super().__init__(agent_id=agent_id, device_state_config=device_state_config, **kwargs)
        self.config = StorageConfig(**device_state_config)

    def set_state(self):
        """Initialize ESS state with features."""
        super().set_state()

        # Add features
        self.state.features.append(ElectricalBasePh())
        self.state.features.append(StorageBlock(
            soc=self.config.init_soc,
            e_capacity_MWh=self.config.e_capacity_MWh,
            ch_eff=self.config.ch_eff,
            dsc_eff=self.config.dsc_eff,
        ))

    def set_action(self):
        """Define action space: [P_setpoint] normalized to [-1, 1]."""
        self.action.set_specs(
            dim_c=1,
            range=(np.array([-1.0]), np.array([1.0]))
        )

    def apply_action(self, action: np.ndarray) -> None:
        """Set charge/discharge power."""
        # Denormalize
        p_range = self.config.p_max_MW - self.config.p_min_MW
        p_setpoint = self.config.p_min_MW + (action[0] + 1) / 2 * p_range

        # Get current SOC from storage feature
        storage = next(f for f in self.state.features if f.__class__.__name__ == "StorageBlock")
        current_soc = storage.soc
        dt = 1.0  # 1 hour timestep

        # Apply SOC constraints
        if p_setpoint > 0:  # Discharging
            max_discharge = (current_soc - self.config.soc_min) * self.config.e_capacity_MWh / dt
            p_setpoint = min(p_setpoint, max_discharge * self.config.dsc_eff)
        else:  # Charging
            max_charge = (self.config.soc_max - current_soc) * self.config.e_capacity_MWh / dt
            p_setpoint = max(p_setpoint, -max_charge / self.config.ch_eff)

        # Update electrical feature
        self.state.update_feature("ElectricalBasePh", p_MW=p_setpoint)

    def update_state(self, dt: float = 1.0, **kwargs) -> None:
        """Update SOC based on power flow."""
        electrical = next(f for f in self.state.features if f.__class__.__name__ == "ElectricalBasePh")
        storage = next(f for f in self.state.features if f.__class__.__name__ == "StorageBlock")

        p = electrical.p_MW

        if p > 0:  # Discharging
            energy_out = p * dt / self.config.dsc_eff
        else:  # Charging
            energy_out = p * dt * self.config.ch_eff

        new_soc = storage.soc - energy_out / self.config.e_capacity_MWh
        storage.soc = np.clip(new_soc, self.config.soc_min, self.config.soc_max)
        storage.e_stored_MWh = storage.soc * self.config.e_capacity_MWh

    def update_cost_safety(self) -> None:
        """ESS has no fuel cost, only degradation."""
        self.cost = 0.0  # Could add degradation cost here
        self.safety = 0.0
```

---

## Step 3: Build the Grid Agent (Coordinator)

The `GridAgent` coordinates multiple device agents within a microgrid. It extends HERON's `CoordinatorAgent`.

```python
# powergrid/agents/power_grid_agent.py
from typing import Any, Dict, Optional
import numpy as np
import pandapower as pp

from heron.agents import CoordinatorAgent
from heron.core import CoordinatorAgentState
from heron.protocols import Protocol
from heron.scheduling import TickConfig
from powergrid.agents.device_agent import DeviceAgent
from powergrid.agents.generator import Generator
from powergrid.agents.storage import ESS
from powergrid.core.features.network import BusVoltages, LineFlows

# Level constant
COORDINATOR_LEVEL = 2

# Device type registry
DEVICE_REGISTRY = {
    'Generator': Generator,
    'ESS': ESS,
}


class PowerGridAgent(CoordinatorAgent):
    """Grid-level coordinator with PandaPower integration.

    Manages device agents and interfaces with PandaPower for power flow analysis.

    Key HERON methods:
    - _build_subordinates(): Create device agents from config
    - set_state(): Initialize coordinator state with features
    - coordinate_subordinates(): Distribute actions via protocol
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        grid_config: Dict[str, Any],
        protocol: Protocol,
        policy=None,
        tick_config: Optional[TickConfig] = None,
        **kwargs
    ):
        self._net = net
        self._grid_config = grid_config

        # Call HERON's CoordinatorAgent init
        super().__init__(
            agent_id=grid_config.get('name', 'grid'),
            protocol=protocol,
            policy=policy,
            config={'agents': grid_config.get('devices', [])},
            tick_config=tick_config,
            **kwargs
        )

        # Dataset for time-series data
        self._dataset = None
        self._t = 0

    def set_state(self):
        """Initialize grid state with network-level features."""
        self.state = CoordinatorAgentState(
            owner_id=self.agent_id,
            owner_level=COORDINATOR_LEVEL
        )

        # Add network-level features
        self.state.features.append(BusVoltages())
        self.state.features.append(LineFlows())

    def _build_subordinates(self, configs, env_id, upstream_id):
        """Create device agents from configuration.

        This is the HERON pattern for building subordinate hierarchy.
        Called automatically by CoordinatorAgent.__init__().
        """
        devices = {}

        for device_cfg in configs:
            device_type = device_cfg['type']
            device_name = device_cfg['name']
            state_config = device_cfg.get('device_state_config', {})

            # Create device agent using registry
            DeviceClass = DEVICE_REGISTRY[device_type]
            device = DeviceClass(
                agent_id=device_name,
                device_state_config=state_config,
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )

            # Add device to PandaPower network
            self._add_device_to_network(device, state_config)

            devices[device_name] = device

        return devices

    @property
    def devices(self) -> Dict[str, DeviceAgent]:
        """Alias for subordinates (HERON naming)."""
        return self.subordinates

    def _add_device_to_network(self, device: DeviceAgent, config: Dict) -> None:
        """Add device to PandaPower network."""
        bus_name = config['bus']
        bus_idx = self._get_bus_idx(bus_name)

        if isinstance(device, Generator):
            idx = pp.create_sgen(
                self._net,
                bus=bus_idx,
                p_mw=0.0,
                q_mvar=0.0,
                name=device.agent_id,
            )
            device.pp_idx = idx

        elif isinstance(device, ESS):
            idx = pp.create_storage(
                self._net,
                bus=bus_idx,
                p_mw=0.0,
                max_e_mwh=config['e_capacity_MWh'],
                soc_percent=config.get('init_soc', 0.5) * 100,
                name=device.agent_id,
            )
            device.pp_idx = idx

    def _get_bus_idx(self, bus_name: str) -> int:
        """Get PandaPower bus index from bus name."""
        mask = self._net.bus['name'] == bus_name
        if not mask.any():
            raise ValueError(f"Bus '{bus_name}' not found in network")
        return self._net.bus[mask].index[0]

    def add_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        """Add time-series dataset for load/renewable profiles."""
        self._dataset = dataset

    def step(self, action: np.ndarray) -> None:
        """Execute one timestep with given action.

        1. Distribute action to devices via protocol (HERON pattern)
        2. Update PandaPower network
        3. Run power flow
        4. Update state from results
        """
        # 1. Coordinate devices using HERON's protocol system
        self.coordinate_subordinates(observation=None, action=action)

        # 2. Sync device states to PandaPower network
        for device_id, device in self.devices.items():
            electrical = next(f for f in device.state.features
                           if f.__class__.__name__ == "ElectricalBasePh")
            if isinstance(device, Generator):
                self._net.sgen.at[device.pp_idx, 'p_mw'] = electrical.p_MW
                self._net.sgen.at[device.pp_idx, 'q_mvar'] = electrical.q_MVAr
            elif isinstance(device, ESS):
                self._net.storage.at[device.pp_idx, 'p_mw'] = electrical.p_MW

        # 3. Update loads from dataset
        if self._dataset is not None:
            self._update_loads_from_dataset()

        # 4. Run power flow
        pp.runpp(self._net, algorithm='nr', numba=True)

        # 5. Update state from results
        self._sync_state_from_network()

        # 6. Update cost and safety
        self._update_cost_safety()

        self._t += 1

    def _sync_state_from_network(self) -> None:
        """Update state from PandaPower results."""
        # Update bus voltages feature
        bus_voltages = next(f for f in self.state.features
                          if f.__class__.__name__ == "BusVoltages")
        bus_voltages.vm_pu = self._net.res_bus['vm_pu'].values
        bus_voltages.v_min = self._net.res_bus['vm_pu'].min()
        bus_voltages.v_max = self._net.res_bus['vm_pu'].max()

        # Update line flows feature
        if len(self._net.line) > 0:
            line_flows = next(f for f in self.state.features
                             if f.__class__.__name__ == "LineFlows")
            line_flows.loading_pct = self._net.res_line['loading_percent'].values
            line_flows.max_loading = self._net.res_line['loading_percent'].max()

        # Update device states
        for device in self.devices.values():
            if isinstance(device, Generator):
                bus_idx = self._net.sgen.at[device.pp_idx, 'bus']
                v_pu = self._net.res_bus.at[bus_idx, 'vm_pu']
                device.state.update_feature("ElectricalBasePh", v_pu=v_pu)
            elif isinstance(device, ESS):
                device.update_state()

    def _update_cost_safety(self) -> None:
        """Aggregate cost and safety from all devices plus network violations."""
        self.cost = 0.0
        self.safety = 0.0

        # Device-level costs
        for device in self.devices.values():
            device.update_cost_safety()
            self.cost += device.cost
            self.safety += device.safety

        # Network-level safety (voltage violations)
        bus_voltages = next(f for f in self.state.features
                          if f.__class__.__name__ == "BusVoltages")
        v_min_limit, v_max_limit = 0.95, 1.05
        voltage_violation = np.sum(np.maximum(0, v_min_limit - bus_voltages.vm_pu))
        voltage_violation += np.sum(np.maximum(0, bus_voltages.vm_pu - v_max_limit))
        self.safety += voltage_violation

        # Line overloading
        line_flows = next((f for f in self.state.features
                          if f.__class__.__name__ == "LineFlows"), None)
        if line_flows:
            overload = np.sum(np.maximum(0, line_flows.loading_pct - 100))
            self.safety += overload / 100
```

---

## Step 4: Create the Environment

The environment wraps agents using **HERON's `PettingZooParallelEnv` adapter** (not raw `ParallelEnv`). This is critical for enabling:
- Event-driven execution mode
- Agent registration and management
- Message broker integration

### HeronEnvCore Features

The `HeronEnvCore` mixin (inherited by `PettingZooParallelEnv`) provides:

| Method | Purpose |
|--------|---------|
| `register_agent(agent)` | Register agent with environment |
| `get_heron_agent(agent_id)` | Get agent by ID |
| `get_observations(global_state)` | Get observations for all agents |
| `apply_actions(actions)` | Apply actions to agents |
| `reset_agents(seed)` | Reset all registered agents |
| `setup_event_driven()` | Initialize event scheduler |
| `run_event_driven(t_end)` | Run event-driven simulation |
| `configure_agents_for_distributed()` | Setup message broker |

### 4.1 Base Environment

```python
# powergrid/envs/networked_grid_env.py
from abc import abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from gymnasium.spaces import Box, Dict as SpaceDict

# IMPORTANT: Use HERON's adapter, NOT raw PettingZoo ParallelEnv
from heron.envs import PettingZooParallelEnv
from heron.messaging import InMemoryBroker

from powergrid.agents.power_grid_agent import PowerGridAgent


class NetworkedGridEnv(PettingZooParallelEnv):
    """Base multi-agent environment for networked power grids.

    Uses HERON's PettingZooParallelEnv adapter which provides:
    - register_agent() for agent management
    - setup_event_driven() for realistic timing
    - run_event_driven() for event-based simulation
    - configure_agents_for_distributed() for message broker integration

    Supports both centralized and distributed execution modes.
    """

    metadata = {"render_modes": ["human"], "name": "networked_grid_v0"}

    def __init__(self, env_config: Dict[str, Any]):
        # Initialize HERON's adapter FIRST
        super().__init__(env_id=env_config.get('env_id', 'networked_grid'))

        self.env_config = env_config
        self.max_episode_steps = env_config.get('max_episode_steps', 24)
        self.centralized = env_config.get('centralized', True)
        self.train = env_config.get('train', True)
        self.penalty = env_config.get('penalty', 10.0)
        self.share_reward = env_config.get('share_reward', False)

        # Will be set by subclasses
        self.agent_dict: Dict[str, PowerGridAgent] = {}
        self.net = None

        # Build components
        self.net = self._build_net()
        self._build_agents()

        # HERON pattern: Register agents with HeronEnvCore
        for agent_id, agent in self.agent_dict.items():
            self.register_agent(agent)

        # Setup spaces using HERON helpers
        self._setup_spaces()

        # Setup distributed mode if needed
        if not self.centralized:
            self.configure_agents_for_distributed()

        # Episode state
        self._t = 0

    @abstractmethod
    def _build_net(self):
        """Build PandaPower network. Override in subclasses."""
        pass

    @abstractmethod
    def _build_agents(self) -> Dict[str, PowerGridAgent]:
        """Build grid agents. Override in subclasses."""
        pass

    @abstractmethod
    def _reward_and_safety(self) -> tuple:
        """Compute rewards and safety violations."""
        pass

    def _setup_spaces(self) -> None:
        """Initialize observation and action spaces."""
        obs_spaces = {}
        act_spaces = {}

        for agent_id, agent in self.agent_dict.items():
            # Observation space from agent state
            obs_vec = agent.state.vector()
            obs_spaces[agent_id] = Box(
                low=-np.inf, high=np.inf, shape=obs_vec.shape, dtype=np.float32
            )

            # Action space from agent action
            act_spaces[agent_id] = agent.action.space

        # PettingZoo required attributes
        self.observation_spaces = obs_spaces
        self.action_spaces = act_spaces
        self.possible_agents = list(self.agent_dict.keys())
        self.agents = self.possible_agents.copy()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self._t = 0
        self.agents = self.possible_agents.copy()

        # Use HERON's reset_agents helper
        self.reset_agents(seed=seed)

        # Run initial power flow
        import pandapower as pp
        pp.runpp(self.net)

        # Get observations using HERON helper
        observations = self.get_observations()
        # Convert to vectors for PettingZoo
        obs_dict = {
            agent_id: obs.vector() if hasattr(obs, 'vector') else obs
            for agent_id, obs in observations.items()
        }

        infos = {agent_id: {} for agent_id in self.agents}
        return obs_dict, infos

    def step(self, actions: Dict[str, np.ndarray]):
        """Execute one environment step."""
        # Apply actions using HERON helper
        self.apply_actions(actions)

        self._t += 1

        # Compute rewards and safety
        rewards, safety = self._reward_and_safety()

        # Apply safety penalty
        for agent_id in rewards:
            rewards[agent_id] -= self.penalty * safety[agent_id]

        # Share rewards if configured
        if self.share_reward:
            total_reward = sum(rewards.values())
            avg_reward = total_reward / len(rewards)
            rewards = {agent_id: avg_reward for agent_id in rewards}

        # Get observations using HERON helper
        observations = self.get_observations()
        obs_dict = {
            agent_id: obs.vector() if hasattr(obs, 'vector') else obs
            for agent_id, obs in observations.items()
        }

        # Check termination
        truncated = self._t >= self.max_episode_steps
        terminateds = {agent_id: False for agent_id in self.agents}
        terminateds["__all__"] = False
        truncateds = {agent_id: truncated for agent_id in self.agents}
        truncateds["__all__"] = truncated

        # Build infos
        infos = {
            agent_id: {
                'cost': self.agent_dict[agent_id].cost,
                'safety': safety[agent_id],
            }
            for agent_id in self.agents
        }

        return obs_dict, rewards, terminateds, truncateds, infos

    def observation_space(self, agent_id: str) -> Box:
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id: str) -> Box:
        return self.action_spaces[agent_id]
```

### 4.2 Concrete Environment

```python
# powergrid/envs/multi_agent_microgrids.py
from powergrid.envs.networked_grid_env import NetworkedGridEnv
from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.networks.ieee34 import IEEE34Bus
from heron.protocols.vertical import SetpointProtocol

class MultiAgentMicrogrids(NetworkedGridEnv):
    """Multi-agent environment with 3 networked microgrids.

    Architecture:
        DSO (Distribution System Operator) - IEEE 34-bus network
        ├── Microgrid 1 (IEEE 13-bus)
        ├── Microgrid 2 (IEEE 13-bus)
        └── Microgrid 3 (IEEE 13-bus)
    """

    def _build_net(self):
        """Build combined DSO + microgrid network."""
        import pandapower as pp

        # Create DSO network (IEEE 34-bus)
        dso_net = IEEE34Bus("DSO")

        # Create microgrid networks and fuse to DSO
        mg_configs = self.env_config.get('microgrid_configs', [])

        for mg_config in mg_configs:
            mg_net = IEEE13Bus(mg_config['name'])
            connection_bus = mg_config.get('connection_bus', 'DSO Bus 850')

            # Fuse microgrid to DSO at connection point
            dso_bus_idx = self._get_bus_by_name(dso_net, connection_bus)
            pp.fuse_bus(dso_net, dso_bus_idx, mg_net, 0)  # Fuse at slack bus

        return dso_net

    def _build_agents(self):
        """Build PowerGridAgent for each microgrid."""
        self.agent_dict = {}
        mg_configs = self.env_config.get('microgrid_configs', [])

        for mg_config in mg_configs:
            agent = PowerGridAgent(
                net=self.net,
                grid_config=mg_config,
                protocol=SetpointProtocol(),
                message_broker=self.message_broker,
            )

            # Add dataset if available
            if 'dataset' in self.env_config:
                agent.add_dataset(self.env_config['dataset'])

            self.agent_dict[mg_config['name']] = agent

        self.possible_agents = list(self.agent_dict.keys())

    def _reward_and_safety(self):
        """Compute rewards (negative cost) and safety violations."""
        rewards = {}
        safety = {}

        for agent_id, agent in self.agent_dict.items():
            rewards[agent_id] = -agent.cost
            safety[agent_id] = agent.safety

        return rewards, safety
```

---

## Step 5: Configure Setups and Datasets

### 5.1 Setup Structure

```
powergrid/setups/ieee34_ieee13/
├── config.yml      # Environment configuration
└── data.pkl        # Time-series data
```

### 5.2 Configuration File

```yaml
# config.yml
dataset_path: data.pkl
train: true
penalty: 10.0
share_reward: true
max_episode_steps: 96  # 4 days

dso_config:
  name: DSO
  network: ieee34
  load_area: BANC

microgrid_configs:
  - name: MG1
    connection_bus: "DSO Bus 850"
    base_power: 1.0
    load_scale: 0.1
    devices:
      - type: Generator
        name: gen1
        device_state_config:
          bus: "Bus 633"
          p_max_MW: 2.0
          p_min_MW: 0.5
          cost_curve_coefs: [0.02, 10.0, 0.0]

      - type: ESS
        name: ess1
        device_state_config:
          bus: "Bus 634"
          e_capacity_MWh: 5.0
          p_max_MW: 1.0
          p_min_MW: -1.0
          init_soc: 0.5

  - name: MG2
    connection_bus: "DSO Bus 860"
    devices:
      - type: Generator
        name: gen2
        device_state_config:
          bus: "Bus 633"
          p_max_MW: 1.5
          p_min_MW: 0.3

  - name: MG3
    connection_bus: "DSO Bus 890"
    devices:
      - type: ESS
        name: ess3
        device_state_config:
          bus: "Bus 680"
          e_capacity_MWh: 3.0
```

### 5.3 Dataset Format

```python
# Create dataset (data.pkl)
import pickle
import numpy as np

dataset = {
    'train': {
        'load': np.random.uniform(0.5, 1.2, 8760),      # Hourly load profile
        'solar': np.maximum(0, np.sin(np.linspace(0, 2*np.pi, 24))),  # Daily solar
        'wind': np.random.uniform(0.2, 0.8, 8760),      # Wind profile
        'price': 30 + 20 * np.random.random(8760),      # $/MWh price
    },
    'test': {
        # Same structure for test split
    }
}

with open('data.pkl', 'wb') as f:
    pickle.dump(dataset, f)
```

### 5.4 Setup Loader

```python
# powergrid/setups/loader.py
import os
import pickle
import yaml
from pathlib import Path

SETUPS_DIR = Path(__file__).parent

def get_available_setups() -> list:
    """List all available setup directories."""
    return [
        d.name for d in SETUPS_DIR.iterdir()
        if d.is_dir() and (d / 'config.yml').exists()
    ]

def load_setup(setup_name: str) -> dict:
    """Load setup configuration and resolve dataset path."""
    setup_dir = SETUPS_DIR / setup_name

    if not setup_dir.exists():
        raise ValueError(f"Setup '{setup_name}' not found. Available: {get_available_setups()}")

    # Load config
    with open(setup_dir / 'config.yml') as f:
        config = yaml.safe_load(f)

    # Resolve dataset path
    if 'dataset_path' in config:
        config['dataset_path'] = str(setup_dir / config['dataset_path'])
        config['dataset'] = load_dataset(config['dataset_path'])

    return config

def load_dataset(path: str) -> dict:
    """Load pickled dataset."""
    with open(path, 'rb') as f:
        return pickle.load(f)
```

---

## Step 6: Run RL Training

### 6.1 Basic Training Script

```python
# examples/05_mappo_training.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.setups.loader import load_setup

def env_creator(env_config):
    """Create environment for RLlib."""
    env = MultiAgentMicrogrids(env_config)
    return ParallelPettingZooEnv(env)

def main():
    # Initialize Ray
    ray.init()

    # Register environment
    register_env("multi_agent_microgrids", env_creator)

    # Load configuration
    env_config = load_setup('ieee34_ieee13')
    env_config.update({
        'train': True,
        'share_reward': True,
        'max_episode_steps': 96,
    })

    # Create temporary env for spaces
    temp_env = env_creator(env_config)

    # Configure MAPPO (shared policy)
    config = (
        PPOConfig()
        .environment(
            env="multi_agent_microgrids",
            env_config=env_config,
        )
        .multi_agent(
            policies={
                'shared_policy': (
                    None,
                    temp_env.observation_space['MG1'],
                    temp_env.action_space['MG1'],
                    {}
                )
            },
            policy_mapping_fn=lambda agent_id, *args: 'shared_policy',
        )
        .training(
            lr=5e-5,
            gamma=0.99,
            train_batch_size=4000,
        )
        .env_runners(num_env_runners=4)
    )

    # Build and train
    algo = config.build()

    for i in range(100):
        result = algo.train()
        reward = result.get('env_runners', {}).get('episode_reward_mean', 0)
        print(f"Iteration {i+1}: Reward = {reward:.2f}")

    # Save model
    checkpoint = algo.save('checkpoints/mappo')
    print(f"Saved checkpoint: {checkpoint}")

    algo.stop()
    ray.shutdown()

if __name__ == '__main__':
    main()
```

### 6.2 Running Training

```bash
# Quick test (3 iterations)
python examples/05_mappo_training.py --test

# Full training
python examples/05_mappo_training.py --iterations 100 --num-workers 4

# With Weights & Biases logging
python examples/05_mappo_training.py --iterations 100 --wandb --wandb-project powergrid

# Independent policies (IPPO)
python examples/05_mappo_training.py --iterations 100 --independent-policies
```

### 6.3 MAPPO vs IPPO: When to Use Each

| Algorithm | Policy | Best For | Trade-offs |
|-----------|--------|----------|------------|
| **MAPPO** | Shared policy across all agents | Cooperative tasks, homogeneous agents | Faster learning, less memory, but less flexibility |
| **IPPO** | Independent policy per agent | Heterogeneous agents, competitive tasks | More flexible, but slower learning |

**For power grid microgrids**:
- Use **MAPPO with shared rewards** (`--share-reward`) for cooperative voltage regulation
- Use **IPPO** if microgrids have very different device configurations

### 6.4 Reward Engineering

The default reward structure is:

```python
reward = -cost - penalty * safety_violation
```

Where:
- **cost**: Sum of generation costs (fuel, degradation)
- **safety_violation**: Sum of voltage violations + line overloads
- **penalty**: Configurable weight (default: 10.0)

**Tips for reward design**:

1. **Start with high penalty**: Encourages safe operation first
   ```python
   env_config['penalty'] = 50.0  # Strong safety enforcement
   ```

2. **Use shared rewards for cooperation**:
   ```python
   env_config['share_reward'] = True  # All agents get average reward
   ```

3. **Add custom reward terms** by overriding `_reward_and_safety()`:
   ```python
   def _reward_and_safety(self):
       rewards, safety = super()._reward_and_safety()

       # Add bonus for voltage stability
       for agent_id, agent in self.agent_dict.items():
           v_stability = 1.0 - abs(agent.state.bus_voltages.v_max - 1.0)
           rewards[agent_id] += 0.1 * v_stability

       return rewards, safety
   ```

---

## Step 7: Event-Driven Testing (Key HERON Differentiator)

> **This is what differentiates HERON from PettingZoo wrappers.** Event-driven execution cannot be achieved by wrapping—it requires architectural support built into the framework.

### 7.1 Why Event-Driven Testing?

Policies trained in synchronous mode may fail in deployment where:

| Real-World Factor | Training Mode | Reality |
|-------------------|---------------|---------|
| Agent tick rates | All same | Devices: 1s, Coordinators: 60s, DSO: 300s |
| Observation delay | 0ms | 50-2000ms (SCADA latency) |
| Action delay | 0ms | 100-500ms (actuator response) |
| Timing jitter | None | ±10-40% (network/processing variance) |

### 7.2 Enabling Event-Driven Mode

```python
# In your environment's __init__:
from heron.envs import PettingZooParallelEnv
from heron.scheduling import TickConfig, JitterType

class NetworkedGridEnv(PettingZooParallelEnv):
    def __init__(self, env_config):
        super().__init__(env_id="networked_grid")

        self.event_driven = env_config.get('event_driven', False)

        # ... build agents ...

        # Setup event-driven mode if enabled
        if self.event_driven:
            self._setup_event_driven_mode()

    def _setup_event_driven_mode(self):
        """Configure HERON's event-driven execution."""
        # HERON's built-in method creates the scheduler
        self.setup_event_driven()

        # Setup handlers for agent ticks and action effects
        def on_action_effect(agent_id, action):
            """Called when action effects are applied."""
            agent = self.get_heron_agent(agent_id)
            if agent:
                agent.apply_action(action)

        self.setup_default_handlers(
            global_state_fn=lambda: self._get_global_state(),
            on_action_effect=on_action_effect,
        )
```

### 7.3 Configuring Agent Timing with TickConfig

HERON uses `TickConfig` for configurable timing with optional jitter:

```python
from heron.scheduling import TickConfig, JitterType

class Generator(DeviceAgent):
    def __init__(self, agent_id, device_state_config, **kwargs):
        # Pass TickConfig to HERON agent
        super().__init__(
            agent_id=agent_id,
            tick_config=TickConfig.with_jitter(
                tick_interval=1.0,     # Device ticks every 1 second
                obs_delay=0.05,        # 50ms observation delay
                act_delay=0.1,         # 100ms action delay
                msg_delay=0.02,        # 20ms message delay
                jitter_type=JitterType.GAUSSIAN,
                jitter_ratio=0.1,      # 10% jitter
            ),
            **kwargs
        )

class PowerGridAgent(CoordinatorAgent):
    def __init__(self, net, grid_config, **kwargs):
        super().__init__(
            agent_id=grid_config['name'],
            tick_config=TickConfig.with_jitter(
                tick_interval=60.0,    # Coordinator ticks every 60 seconds
                obs_delay=0.2,         # 200ms for aggregated observations
                act_delay=0.5,         # 500ms for coordinated actions
                msg_delay=0.1,         # 100ms message delay
                jitter_type=JitterType.GAUSSIAN,
                jitter_ratio=0.2,
            ),
            **kwargs
        )
```

### 7.4 Timing Presets

```python
from heron.scheduling import TickConfig, JitterType

# Training: deterministic timing (fast, reproducible)
training_timing = TickConfig.deterministic(
    tick_interval=1.0,
    obs_delay=0.0,
    act_delay=0.0,
    msg_delay=0.0,
)

# Testing: realistic timing with jitter
testing_timing = TickConfig.with_jitter(
    tick_interval=1.0,
    obs_delay=0.1,       # 100ms base delay
    act_delay=0.2,       # 200ms base delay
    msg_delay=0.05,      # 50ms message delay
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.2,    # 20% standard deviation
    seed=42,
)

# CPS-calibrated timing (IEEE 2030 SCADA)
scada_timing = TickConfig.with_jitter(
    tick_interval=60.0,   # SCADA poll interval
    obs_delay=2.0,        # ~2s SCADA latency
    act_delay=1.5,        # ~1.5s command latency
    msg_delay=0.5,        # 500ms message delay
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.4,     # High variance for SCADA
)

# Get jittered values at runtime
next_tick = testing_timing.next_tick_interval()
obs_delay = testing_timing.next_obs_delay()
act_delay = testing_timing.next_act_delay()
```

### 7.5 Running Event-Driven Simulation

```python
def step(self, actions):
    self._timestep += 1

    if self.event_driven:
        return self._step_event_driven(actions)
    else:
        return self._step_synchronous(actions)

def _step_event_driven(self, actions):
    """Event-driven step using HERON's scheduler."""
    # Store actions for agents (HERON will schedule effects)
    for agent_id, action in actions.items():
        self._pending_actions[agent_id] = action

    # Run simulation for one "step" duration
    step_duration = 1.0  # 1 second per step
    end_time = self.scheduler.current_time + step_duration

    # HERON's built-in method runs the event loop
    num_events = self.run_event_driven(t_end=end_time)

    # Get observations using HERON helper
    observations = self.get_observations()
    obs_dict = {
        agent_id: obs.vector() if hasattr(obs, 'vector') else obs
        for agent_id, obs in observations.items()
    }

    # ... compute rewards, terminateds, etc.

    return obs_dict, rewards, terminateds, truncateds, infos
```

### 7.6 Testing Workflow

```python
# 1. Train in synchronous mode (fast)
train_config = {
    'event_driven': False,
    'max_episode_steps': 96,
}
train_env = MultiAgentMicrogrids(train_config)
trained_policy = train_mappo(train_env, iterations=100)

# 2. Test in event-driven mode (realistic)
test_config = {
    'event_driven': True,
    'timing_jitter': 0.2,  # 20% jitter
    'max_episode_steps': 96,
}
test_env = MultiAgentMicrogrids(test_config)

# 3. Evaluate performance gap
sync_reward = evaluate(train_env, trained_policy)
event_reward = evaluate(test_env, trained_policy)

print(f"Synchronous reward: {sync_reward:.2f}")
print(f"Event-driven reward: {event_reward:.2f}")
print(f"Performance gap: {(sync_reward - event_reward) / sync_reward * 100:.1f}%")

# 4. If gap is large, consider:
#    - Training with timing awareness
#    - Adding robustness regularization
#    - Using domain randomization on timing
```

### 7.7 Key Takeaways

| Aspect | Synchronous Mode | Event-Driven Mode |
|--------|-----------------|-------------------|
| **Use case** | Training | Testing/Deployment |
| **Speed** | Fast | Slower |
| **Realism** | Low | High |
| **Reproducibility** | Deterministic | Configurable jitter |
| **HERON method** | `step()` | `run_event_driven()` |

**This dual-mode capability is architectural**—it cannot be achieved by wrapping PettingZoo. HERON's `PettingZooParallelEnv` adapter provides this natively through `setup_event_driven()`, `setup_default_handlers()`, and `run_event_driven()`.

---

## Advanced Topics

### A. Custom Protocols

Protocols define how coordinators communicate with subordinate agents:

```python
# heron/protocols/vertical.py
from heron.protocols.base import Protocol

class SetpointProtocol(Protocol):
    """Direct setpoint control - coordinator sends power setpoints to devices."""

    def coordinate(self, coordinator_action, devices):
        """Distribute coordinator action to devices.

        Args:
            coordinator_action: Flattened action array
            devices: Dict of device agents

        Returns:
            Dict mapping device_id to device action
        """
        device_actions = {}
        offset = 0

        for device_id, device in devices.items():
            action_dim = device.get_action_space().shape[0]
            device_actions[device_id] = coordinator_action[offset:offset + action_dim]
            offset += action_dim

        return device_actions

class PriceSignalProtocol(Protocol):
    """Price-based coordination - coordinator sends price signals."""

    def coordinate(self, price_signal, devices):
        # Devices respond to price signal independently
        device_actions = {}
        for device_id, device in devices.items():
            device_actions[device_id] = device.respond_to_price(price_signal)
        return device_actions
```

### B. Distributed Mode

For realistic deployment scenarios with limited communication:

```python
# Enable distributed mode
env_config = {
    'centralized': False,  # Use message broker
    'message_broker': 'in_memory',
}

env = MultiAgentMicrogrids(env_config)
```

In distributed mode:
- Agents receive network state via messages (not direct access)
- Actions are sent via message broker
- Partial observability is naturally enforced

### C. Adding Custom Devices

To add a new device type:

1. Create device class extending `DeviceAgent`
2. Register in `DEVICE_REGISTRY`
3. Implement required methods: `get_action_space()`, `set_device_action()`, `update_cost_safety()`

```python
# powergrid/agents/wind_turbine.py
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from gymnasium.spaces import Box

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.features.electrical import ElectricalBasePh

@dataclass
class WindTurbineConfig:
    """Configuration for wind turbine."""
    bus: str
    p_rated_MW: float = 2.0  # Rated power capacity

class WindTurbine(DeviceAgent):
    """Wind turbine with curtailment control.

    The turbine generates power based on wind availability (from dataset)
    and can be curtailed via RL action.
    """

    def __init__(self, agent_id: str, device_state_config: Dict[str, Any], **kwargs):
        super().__init__(agent_id=agent_id, device_state_config=device_state_config, **kwargs)
        self.config = WindTurbineConfig(**device_state_config)
        self.state.add_feature('electrical', ElectricalBasePh())
        self.available_power = 0.0  # Set by environment based on wind forecast

    def get_action_space(self):
        # Curtailment factor [0, 1]: 0 = full curtailment, 1 = no curtailment
        return Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def set_device_action(self, action):
        # Apply curtailment to available wind power
        curtailment_factor = action[0]
        self.state.electrical.p_MW = self.available_power * curtailment_factor

    def update_cost_safety(self):
        # Wind has zero fuel cost, but curtailment has opportunity cost
        curtailed = self.available_power - self.state.electrical.p_MW
        self.cost = curtailed * 5.0  # $5/MWh opportunity cost for curtailment
        self.safety = 0.0

# Register in device registry
DEVICE_REGISTRY['WindTurbine'] = WindTurbine
```

### D. Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/agents/test_power_grid_agent.py -v

# Run integration tests
pytest tests/integration/ -v

# Training verification
bash tests/test_train_mappo.sh
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Power Flow Convergence Errors

**Symptom**: `pandapower.powerflow.LoadflowNotConverged`

**Causes and Solutions**:
```python
# Issue: Unrealistic power values
# Solution: Check device limits and clip actions properly
p_setpoint = np.clip(p_setpoint, self.config.p_min_MW, self.config.p_max_MW)

# Issue: Isolated buses or disconnected network
# Solution: Verify network topology before running power flow
import pandapower.topology as top
if not top.unsupplied_buses(net).empty:
    print("Warning: Unsupplied buses detected!")
```

#### 2. Observation Space Mismatch

**Symptom**: `ValueError: observation shape mismatch`

**Solution**: Ensure observation dimensions are consistent:
```python
# Debug observation dimensions
for agent_id, agent in env.agent_dict.items():
    obs = agent.get_observation()
    print(f"{agent_id}: obs_dim={len(obs)}, space={env.observation_spaces[agent_id].shape}")
```

#### 3. Action Space Issues with RLlib

**Symptom**: `KeyError` or dimension mismatch during training

**Solution**: Use flattened action spaces for RLlib compatibility:
```python
# Instead of Dict action space, flatten to Box
if isinstance(action_space, SpaceDict):
    total_dim = sum(s.shape[0] for s in action_space.spaces.values())
    flat_space = Box(low=-1.0, high=1.0, shape=(total_dim,), dtype=np.float32)
```

#### 4. Ray/RLlib Initialization Issues

**Symptom**: Ray crashes or hangs during training

**Solutions**:
```bash
# Reduce workers for debugging
python examples/05_mappo_training.py --num-workers 1

# Disable GPU
python examples/05_mappo_training.py --no-cuda

# Clear Ray cache
rm -rf /tmp/ray/*
```

#### 5. Memory Issues During Training

**Symptom**: OOM errors during long training runs

**Solutions**:
```python
# Reduce batch size
config.train_batch_size = 2000  # Instead of 4000

# Use fewer parallel environments
config.num_envs_per_env_runner = 1
```

#### 6. Device Not Found in Network

**Symptom**: `ValueError: Bus 'Bus XXX' not found in network`

**Solution**: Verify bus names match exactly:
```python
# List all bus names in network
print(net.bus['name'].tolist())

# Ensure config uses exact names
device_state_config = {
    "bus": "Bus 633",  # Must match exactly
    ...
}
```

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Inspect network state**:
   ```python
   import pandapower as pp
   pp.runpp(net)
   print(net.res_bus)  # Bus results
   print(net.res_line)  # Line results
   ```

3. **Test single agent before multi-agent**:
   Start with `examples/01_single_microgrid_basic.py` to verify individual components work.

4. **Use test mode first**:
   ```bash
   python examples/05_mappo_training.py --test
   ```

---

## Summary

Building a power grid case study from scratch involves seven main steps:

| Step | Component | Key HERON Classes |
|------|-----------|-------------------|
| 1 | [Define Features](#step-1-define-domain-features) | `FeatureProvider` → `ElectricalBasePh`, `StorageBlock`, `BusVoltages` |
| 2 | [Create Device Agents](#step-2-create-device-agents) | `FieldAgent` → `DeviceAgent` → `Generator`, `ESS` |
| 3 | [Build Grid Agent](#step-3-build-the-grid-agent-coordinator) | `CoordinatorAgent` → `GridAgent` → `PowerGridAgent` |
| 4 | [Create Environment](#step-4-create-the-environment) | `PettingZooParallelEnv` → `NetworkedGridEnv` |
| 5 | [Configure Setups](#step-5-configure-setups-and-datasets) | `config.yml`, `data.pkl`, `loader.py` |
| 6 | [Run Training](#step-6-run-rl-training) | RLlib PPO, MAPPO/IPPO |
| 7 | [Event-Driven Testing](#step-7-event-driven-testing-key-heron-differentiator) | `EventScheduler`, `TickConfig`, `JitterType` |

### Key HERON Patterns

| Pattern | Description |
|---------|-------------|
| `set_state()` | Override to add features to agent state |
| `set_action()` | Override to define action space via `self.action.set_specs()` |
| `_build_subordinates()` | Override in CoordinatorAgent to create child agents |
| `register_agent()` | Call in environment to register agents with HeronEnvCore |
| `setup_event_driven()` | Call to enable event-driven mode |
| `TickConfig.with_jitter()` | Configure realistic timing with variance |

### Key Takeaways

1. **Start with examples**: Always run existing examples before building custom components
2. **Extend, don't modify**: PowerGrid classes extend HERON classes, inheriting all functionality
3. **Use TickConfig**: Configure timing via `TickConfig` instead of individual parameters
4. **Register agents**: Always call `register_agent()` for each agent in your environment
5. **Use centralized mode for training**: Distributed/event-driven mode is for testing
6. **MAPPO for cooperation**: Use shared policy and shared rewards for cooperative microgrids
7. **Monitor safety violations**: High violations indicate unrealistic configurations

### Recommended Learning Path

```
1. Run examples/01_single_microgrid_basic.py     (understand basic flow)
2. Run examples/05_mappo_training.py --test      (understand RL training)
3. Modify device configurations in config.yml   (customize environment)
4. Add a new device type (e.g., WindTurbine)    (extend the framework)
5. Create a custom environment                   (full customization)
```

### Glossary

| Term | Definition |
|------|------------|
| **DSO** | Distribution System Operator - manages the distribution grid |
| **ESS** | Energy Storage System (battery) |
| **IEEE** | Institute of Electrical and Electronics Engineers |
| **IPPO** | Independent Proximal Policy Optimization |
| **MAPPO** | Multi-Agent Proximal Policy Optimization |
| **MW/MWh** | Megawatt (power) / Megawatt-hour (energy) |
| **MVAr** | Megavolt-ampere reactive (reactive power) |
| **P, Q** | Active power and reactive power |
| **p.u.** | Per-unit (normalized value, typically voltage) |
| **SOC** | State of Charge (0-1, battery level) |

### Further Reading

- [HERON Framework Documentation](../../../docs/) - Core agent architecture
- [HERON Integration Guide](./HERON_INTEGRATION.md) - Detailed integration patterns
- [PandaPower Documentation](https://pandapower.readthedocs.io/) - Power flow simulation
- [RLlib Multi-Agent Training](https://docs.ray.io/en/latest/rllib/rllib-env.html) - RL algorithms
- [PettingZoo Documentation](https://pettingzoo.farama.org/) - Multi-agent environments
- [IEEE Test Feeders](https://site.ieee.org/pes-testfeeders/) - Standard test networks

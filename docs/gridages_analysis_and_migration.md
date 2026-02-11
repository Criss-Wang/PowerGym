# GridAges Analysis and Migration to Heron

## Executive Summary

GridAges is a Gymnasium-based multi-agent reinforcement learning framework for microgrid control and optimization. This document analyzes the codebase structure, execution flows, and provides a detailed migration plan to the Heron framework.

---

## 1. GridAges Architecture Overview

### 1.1 Repository Structure

```
GridAges/
├── gridages/
│   ├── core/
│   │   ├── actions.py       # Action dataclass (continuous + discrete)
│   │   └── state.py         # DeviceState dataclass
│   ├── devices/
│   │   ├── base.py          # Device abstract base class
│   │   ├── storage.py       # ESS (Energy Storage System)
│   │   ├── generator.py     # DG (Distributed Generator) & RES (Renewable Energy)
│   │   ├── grid.py          # DSO Grid connection
│   │   ├── transformer.py   # Transformer with OLTC
│   │   └── compensation.py  # Shunt/compensation devices
│   ├── envs/
│   │   ├── single_agent/    # Single-agent environments
│   │   └── multi_agent/
│   │       ├── base_env.py       # NetworkedGridEnv base class
│   │       └── ieee34_ieee13.py  # MultiAgentMicrogrids implementation
│   ├── networks/            # Network topology definitions
│   ├── utils/              # Utility functions
│   └── test/               # Testing modules
└── examples/
    └── multi_agent/
        └── coordinated_dispatch/
            └── rllib_mappo_networked_mgs/
                └── multi_agent_mgs.py  # Entry point for RLlib training
```

### 1.2 Key Design Patterns

1. **Device Abstraction**: All devices inherit from `Device` base class with standard interface
2. **State/Action Separation**: State (observations) and actions managed separately via dataclasses
3. **PettingZoo Integration**: Multi-agent environments follow PettingZoo parallel API
4. **Pandapower Backend**: Uses Pandapower for AC/DC power flow simulation
5. **Hierarchical Agents**: DSO + multiple microgrids (MG1, MG2, MG3)

---

## 2. Full Execution Flow

### 2.1 Training Entry Point: `multi_agent_mgs.py`

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING INITIALIZATION                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Parse CLI Arguments                                           │
│    - stop_iters (default: 200)                                  │
│    - stop_timesteps (default: 1,000,000)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Register Environment with Ray Tune                           │
│    register_env("env", lambda _: ParallelPettingZooEnv(         │
│        MultiAgentMicrogrids(train=True, penalty=10,             │
│                             share_reward=False)                 │
│    ))                                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Create Probe Environment                                     │
│    probe_env = MultiAgentMicrogrids(...)                        │
│    Extract observation/action spaces for each agent:            │
│    - MG1: obs_space, action_space                              │
│    - MG2: obs_space, action_space                              │
│    - MG3: obs_space, action_space                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Configure Multi-Agent RL Modules                             │
│    For each agent (MG1, MG2, MG3):                             │
│      - Create PPOModuleSpec                                     │
│      - Set network architecture:                                │
│        * MG1: [128, 128]                                        │
│        * MG2: [64, 64]                                          │
│        * MG3: [64, 128]                                         │
│      - Assign observation/action spaces                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Build RLlib Configuration                                    │
│    - Enable new API stack (RL modules, learners, runners)      │
│    - Multi-agent: 1:1 agent-to-policy mapping                  │
│    - Checkpoint interval: 1000 iterations                       │
│    - Environment runners: 0 (local training)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Launch Training via run_rllib_example_script_experiment()   │
│    - Iterate until stop conditions met                          │
│    - Save checkpoints periodically                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Environment Step Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-AGENT ENVIRONMENT STEP                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ step(actions: Dict[AgentID, Action])                            │
│   Input: {"MG1": action1, "MG2": action2, "MG3": action3}     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: ACTION APPLICATION                                     │
│                                                                  │
│ For each agent in [MG1, MG2, MG3]:                             │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ agent_env.set_action(actions[agent_id])                  │ │
│   │   ├─> For each device (ESS, DG, PV, Wind):              │ │
│   │   │     ├─> Extract device action from agent action      │ │
│   │   │     ├─> device.feasible_action() # Constraint check │ │
│   │   │     └─> device.state.update(action)                 │ │
│   │   └─> Update device setpoints in Pandapower network     │ │
│   └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: POWER FLOW SIMULATION                                  │
│                                                                  │
│   pp.runpp(self.net)  # Pandapower AC power flow              │
│   ├─> Solves voltage at all buses                             │
│   ├─> Computes line flows                                     │
│   ├─> Updates load demands                                    │
│   └─> Converged = True/False                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: STATE UPDATE                                           │
│                                                                  │
│ For each agent:                                                 │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ agent_env.update_state()                                 │ │
│   │   ├─> For each device:                                   │ │
│   │   │     ├─> device.update_state(dt)                     │ │
│   │   │     │   # ESS: SOC += P * eff * dt / capacity      │ │
│   │   │     │   # DG: Unit commitment state transitions    │ │
│   │   │     └─> device.update_cost_safety()               │ │
│   │   │         # Calculate cost and safety violations    │ │
│   │   └─> Aggregate agent-level cost/safety               │ │
│   └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: OBSERVATION CONSTRUCTION                               │
│                                                                  │
│ For each agent:                                                 │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ obs = agent_env._get_obs()                               │ │
│   │   ├─> Collect device states:                            │ │
│   │   │     ├─> ESS: [P, Q, SOC]                          │ │
│   │   │     ├─> DG: [P, Q, on]                            │ │
│   │   │     ├─> RES: [P, Q]                               │ │
│   │   │     └─> Grid: [P, Q, price]                       │ │
│   │   ├─> Collect bus voltages (local buses)              │ │
│   │   ├─> Collect load demands                            │ │
│   │   └─> Normalize/scale observations                     │ │
│   └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: REWARD COMPUTATION                                     │
│                                                                  │
│   rewards = self._reward_and_safety()                          │
│   ├─> For each agent:                                          │
│   │     reward = -agent.cost - penalty * agent.safety         │
│   │     # Cost: operational cost (fuel, cycling)              │
│   │     # Safety: voltage violations + line overloads         │
│   │                                                             │
│   └─> If share_reward: average rewards across agents          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6: TERMINATION CHECK                                      │
│                                                                  │
│   terminated = (self._timestep >= self.max_episode_steps)     │
│   truncated = False                                            │
│   info = {} # Empty info dict                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Return: (observations, rewards, terminateds, truncateds, infos) │
│   All as dicts indexed by agent_id                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Episode Reset Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ reset(seed, options)                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Reset timestep counter                                       │
│    self._timestep = 0                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Reset each agent environment                                 │
│    For agent in [MG1, MG2, MG3]:                               │
│      agent_env.reset()                                          │
│        ├─> For each device (ESS, DG, PV, Wind):               │
│        │     ├─> device.reset()                               │
│        │     │   # ESS: SOC = init_soc or random             │
│        │     │   # DG: on = True, P = min_p                  │
│        │     │   # RES: P = 0                                │
│        │     └─> Reset cost/safety to 0                      │
│        └─> Reset Pandapower network state                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Run initial power flow                                       │
│    pp.runpp(self.net)                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Collect initial observations                                 │
│    observations = {agent_id: agent_env._get_obs()              │
│                    for agent_id in agents}                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Return: (observations, infos)                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Full Data Flow

### 3.1 Data Structures

#### Device State (DeviceState)
```python
@dataclass
class DeviceState:
    P: float = 0.0          # Active power (MW)
    Q: float = 0.0          # Reactive power (MVAr)
    on: int = 1             # On/off status

    # Optional attributes (device-dependent)
    Pmax: Optional[float] = None
    Pmin: Optional[float] = None
    Qmax: Optional[float] = None
    Qmin: Optional[float] = None
    soc: Optional[float] = None       # State of charge (ESS)
    price: Optional[float] = None     # Grid price
    shutting: Optional[int] = None    # DG shutdown state
    starting: Optional[int] = None    # DG startup state

    def as_vector() -> np.ndarray:
        """Serialize to normalized numpy array"""
```

#### Action (Action)
```python
@dataclass
class Action:
    c: np.ndarray          # Continuous actions (dim_c,)
    d: np.ndarray          # Discrete actions (1,)

    dim_c: int = 0         # Continuous dimension
    dim_d: int = 0         # Discrete dimension
    range: np.ndarray      # (2, dim_c) bounds for continuous
    ncats: int = 0         # Number of discrete categories

    def sample() -> Action:
        """Sample random action from space"""
```

### 3.2 Agent Observation Space

For each microgrid agent (MG1, MG2, MG3):

```
Observation Vector:
┌─────────────────────────────────────────────────────────────────┐
│ ESS State: [P, Q, SOC]                             (3 dims)     │
├─────────────────────────────────────────────────────────────────┤
│ DG State: [P, Q, on]                               (3 dims)     │
├─────────────────────────────────────────────────────────────────┤
│ PV State: [P, Q]                                   (2 dims)     │
├─────────────────────────────────────────────────────────────────┤
│ Wind State: [P, Q]                                 (2 dims)     │
├─────────────────────────────────────────────────────────────────┤
│ Grid State: [P, Q, price]                          (3 dims)     │
├─────────────────────────────────────────────────────────────────┤
│ Local Bus Voltages: [V_bus1, V_bus2, ...]         (n_bus dims) │
├─────────────────────────────────────────────────────────────────┤
│ Local Load Demands: [P_load1, Q_load1, ...]       (2*n_load)   │
└─────────────────────────────────────────────────────────────────┘
Total: ~20-30 dimensions per agent
```

### 3.3 Agent Action Space

For each microgrid agent:

```
Action Vector:
┌─────────────────────────────────────────────────────────────────┐
│ ESS Power Setpoint: [P_ess]                       (1 dim)       │
│   Range: [-0.5, 0.5] MW (normalized to [-1, 1])               │
├─────────────────────────────────────────────────────────────────┤
│ DG Power Setpoint: [P_dg]                         (1 dim)       │
│   Range: [min_p, max_p] MW (normalized to [-1, 1])            │
├─────────────────────────────────────────────────────────────────┤
│ PV Reactive Power: [Q_pv]                         (1 dim)       │
│   Range: [-max_q, max_q] MVAr (normalized to [-1, 1])         │
├─────────────────────────────────────────────────────────────────┤
│ Wind Reactive Power: [Q_wind]                     (1 dim)       │
│   Range: [-max_q, max_q] MVAr (normalized to [-1, 1])         │
└─────────────────────────────────────────────────────────────────┘
Total: 4 continuous actions per agent
```

### 3.4 Reward Composition

```
Agent Reward = -cost - penalty * safety

Where:
┌─────────────────────────────────────────────────────────────────┐
│ Cost Components:                                                │
│   - ESS cycling cost (degradation)                             │
│   - DG fuel cost (polynomial of P)                             │
│   - DG startup/shutdown costs                                  │
│   - Grid energy purchase cost (P * price * dt)                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Safety Components (violations):                                │
│   - Voltage violations: |V - V_nom| > threshold                │
│   - Line overloading: line_flow > rating                       │
│   - ESS SOC bounds: SOC < min_soc or SOC > max_soc            │
│   - Apparent power violations: sqrt(P^2 + Q^2) > S_max        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Data Flow Through System

```
Training Loop (RLlib)
  │
  ├─> Sample Action: π(obs) → action
  │     │
  │     └─> Neural Network Forward Pass
  │           Input: observation vector
  │           Hidden: [128,128] or [64,64] or [64,128]
  │           Output: action distribution → sample
  │
  ├─> Environment Step: env.step(actions)
  │     │
  │     ├─> Action → Device Setpoints
  │     │     ESS: P_setpoint
  │     │     DG: P_setpoint, on/off
  │     │     RES: Q_setpoint
  │     │
  │     ├─> Power Flow Simulation
  │     │     Pandapower: solves AC power flow
  │     │     Updates: bus voltages, line flows
  │     │
  │     ├─> Device State Updates
  │     │     ESS: SOC += P * eff * dt / capacity
  │     │     DG: on/off state transitions
  │     │     Cost/Safety calculation
  │     │
  │     └─> Observation + Reward
  │           obs: device states + network state
  │           reward: -cost - penalty*safety
  │
  └─> Policy Update: PPO gradient step
        Experience: (obs, action, reward, next_obs)
        Loss: policy loss + value loss + entropy
        Update: θ ← θ - α∇L
```

---

## 4. Device Component Details

### 4.1 Energy Storage System (ESS)

**State Variables:**
- `P`: Active power (MW) - charging (positive) or discharging (negative)
- `Q`: Reactive power (MVAr)
- `SOC`: State of charge (fraction 0-1)

**Dynamics:**
```python
if P > 0:  # Charging
    SOC += P * ch_eff * dt / capacity
else:      # Discharging
    SOC += P / dsc_eff * dt / capacity
```

**Constraints:**
- Power: `min_p_mw ≤ P ≤ max_p_mw`
- SOC: `min_soc ≤ SOC ≤ max_soc`
- Apparent power: `√(P² + Q²) ≤ sn_mva`

**Cost:**
- Cycling cost (polynomial function of |P|)

**Safety:**
- SOC bounds violation
- Apparent power violation

### 4.2 Distributed Generator (DG)

**State Variables:**
- `P`: Active power output (MW)
- `Q`: Reactive power output (MVAr)
- `on`: Unit commitment status (0/1)
- `starting`: Startup counter
- `shutting`: Shutdown counter

**Dynamics:**
- Unit commitment with startup/shutdown delays
- Power ramping constraints

**Constraints:**
- Power: `min_p_mw ≤ P ≤ max_p_mw` (when on)
- Reactive: `min_q_mvar ≤ Q ≤ max_q_mvar`
- Apparent power: `√(P² + Q²) ≤ sn_mva`

**Cost:**
- Fuel cost (polynomial function of P)
- Startup cost
- Shutdown cost

### 4.3 Renewable Energy Source (RES)

**State Variables:**
- `P`: Active power output (MW) - **externally determined**
- `Q`: Reactive power output (MVAr) - **controllable**

**Characteristics:**
- `action_callback = True` (non-dispatchable)
- P determined by availability/weather data
- Q can be controlled for voltage support

**Constraints:**
- Power: `0 ≤ P ≤ max_p_mw`
- Reactive: `-max_q_mvar ≤ Q ≤ max_q_mvar`
- Apparent power: `√(P² + Q²) ≤ sn_mva`

**Cost:**
- Minimal or zero (no fuel cost)

### 4.4 Grid Connection

**State Variables:**
- `P`: Power exchange with main grid (MW)
  - P > 0: buying from grid
  - P < 0: selling to grid
- `Q`: Reactive power exchange (MVAr)
- `price`: Electricity price ($/MWh)

**Cost:**
```python
if P > 0:  # Buying
    cost = P * price * dt
else:      # Selling
    cost = P * price * dt * sell_discount
```

**Safety:**
- No explicit constraints (grid is infinite bus)

---

## 5. Network Topology

### 5.1 Overall Structure

```
                    IEEE 34-Bus System (DSO)
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    Bus 822              Bus 848              Bus 856
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │   MG1   │         │   MG2   │         │   MG3   │
   │ IEEE 13 │         │ IEEE 13 │         │ IEEE 13 │
   └─────────┘         └─────────┘         └─────────┘
```

### 5.2 Each Microgrid (IEEE 13-Bus)

```
Microgrid Resources:
├── Bus 645: ESS (2 MWh, ±0.5 MW)
├── Bus 675: DG (0.5-0.66 MW)
├── Bus 652: PV (0.1 MW) + Wind (0.1 MW)
└── Various Load Buses

Connection Point: Connects to DSO at specific bus
```

### 5.3 Agent-Device Mapping

```
Agent MG1:
  - ESS @ Bus 645
  - DG1 @ Bus 675 (max: 0.66 MW)
  - PV @ Bus 652
  - Wind @ Bus 652
  - Grid connection @ DSO Bus 822

Agent MG2:
  - ESS @ Bus 645
  - DG1 @ Bus 675 (max: 0.60 MW)
  - PV @ Bus 652
  - Wind @ Bus 652
  - Grid connection @ DSO Bus 848

Agent MG3:
  - ESS @ Bus 645
  - DG1 @ Bus 675 (max: 0.50 MW)
  - PV @ Bus 652
  - Wind @ Bus 652
  - Grid connection @ DSO Bus 856
```

---

## 6. Migration Plan to Heron

### 6.1 Component Mapping

| GridAges Component | Heron Equivalent | Migration Strategy |
|-------------------|------------------|-------------------|
| `Device` (base class) | `FieldAgent` | Extend FieldAgent for each device type |
| `DeviceState` | `FeatureProvider` + `State` | Convert to FeatureProvider pattern |
| `Action` | `Action` class | Already compatible, adapt to Heron's Action |
| `GridEnv` (single agent) | `FieldAgent` | Each microgrid becomes a FieldAgent |
| `NetworkedGridEnv` | `CoordinatorAgent` | DSO becomes CoordinatorAgent |
| `MultiAgentMicrogrids` | `SystemAgent` + `MultiAgentEnv` | Top-level system coordination |
| Pandapower integration | `EnvCore.run_simulation()` | Power flow in simulation phase |
| RLlib training | RLlib integration | Use existing Heron training patterns |

### 6.2 Architecture Translation

#### GridAges Hierarchy:
```
MultiAgentMicrogrids (PettingZoo Env)
  └── NetworkedGridEnv
        ├── GridEnv (MG1) → controls devices
        ├── GridEnv (MG2) → controls devices
        └── GridEnv (MG3) → controls devices
```

#### Heron Hierarchy:
```
MicrogridEnv (MultiAgentEnv)
  └── SystemAgent
        └── DSO_CoordinatorAgent (optional)
              ├── MicrogridFieldAgent (MG1)
              │     ├── ESSFieldAgent
              │     ├── DGFieldAgent
              │     ├── PVFieldAgent
              │     └── WindFieldAgent
              ├── MicrogridFieldAgent (MG2)
              │     └── [same devices]
              └── MicrogridFieldAgent (MG3)
                    └── [same devices]
```

**OR Flattened (for direct comparison):**
```
MicrogridEnv (MultiAgentEnv)
  └── SystemAgent
        ├── MG1_FieldAgent (composite agent controlling all MG1 devices)
        ├── MG2_FieldAgent (composite agent controlling all MG2 devices)
        └── MG3_FieldAgent (composite agent controlling all MG3 devices)
```

### 6.3 Implementation Steps

#### Step 1: Create Device Field Agents

**File: `case_studies/power/powergrid/agents/device_agents.py`**

```python
from heron.agents.field_agent import FieldAgent
from heron.core.feature import FeatureProvider
from heron.core.state import FieldAgentState
from heron.core.action import Action
import numpy as np

# Feature Providers
class ESSFeature(FeatureProvider):
    """Energy Storage System feature"""
    visibility = ["public"]

    P: float = 0.0          # Active power (MW)
    Q: float = 0.0          # Reactive power (MVAr)
    soc: float = 0.5        # State of charge
    capacity: float = 2.0   # MWh

    def set_values(self, **kwargs):
        if "P" in kwargs:
            self.P = kwargs["P"]
        if "Q" in kwargs:
            self.Q = kwargs["Q"]
        if "soc" in kwargs:
            self.soc = np.clip(kwargs["soc"], 0.0, 1.0)

class DGFeature(FeatureProvider):
    """Distributed Generator feature"""
    visibility = ["public"]

    P: float = 0.0
    Q: float = 0.0
    on: int = 1
    max_p: float = 0.66
    min_p: float = 0.1

    def set_values(self, **kwargs):
        if "P" in kwargs:
            self.P = np.clip(kwargs["P"], self.min_p if self.on else 0, self.max_p if self.on else 0)
        if "Q" in kwargs:
            self.Q = kwargs["Q"]
        if "on" in kwargs:
            self.on = int(kwargs["on"])

class RESFeature(FeatureProvider):
    """Renewable Energy Source feature"""
    visibility = ["public"]

    P: float = 0.0  # Set externally based on availability
    Q: float = 0.0  # Controllable
    max_p: float = 0.1
    max_q: float = 0.05

    def set_values(self, **kwargs):
        if "P" in kwargs:
            self.P = np.clip(kwargs["P"], 0.0, self.max_p)
        if "Q" in kwargs:
            self.Q = np.clip(kwargs["Q"], -self.max_q, self.max_q)

class GridFeature(FeatureProvider):
    """Grid connection feature"""
    visibility = ["public"]

    P: float = 0.0     # Power exchange (+ buy, - sell)
    Q: float = 0.0
    price: float = 50.0  # $/MWh

    def set_values(self, **kwargs):
        if "P" in kwargs:
            self.P = kwargs["P"]
        if "Q" in kwargs:
            self.Q = kwargs["Q"]
        if "price" in kwargs:
            self.price = kwargs["price"]

# Composite Microgrid Agent
class MicrogridFieldAgent(FieldAgent):
    """Single microgrid agent controlling multiple devices"""

    def __init__(
        self,
        agent_id: str,
        ess_capacity: float = 2.0,
        dg_max_p: float = 0.66,
        upstream_id: Optional[str] = None,
        env_id: Optional[str] = None,
        policy: Optional[Policy] = None,
    ):
        # Initialize features for all devices
        features = [
            ESSFeature(capacity=ess_capacity, soc=0.5),
            DGFeature(max_p=dg_max_p),
            RESFeature(),  # PV
            RESFeature(),  # Wind
            GridFeature(),
        ]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            policy=policy,
        )

        # Store feature indices for easy access
        self.ess_idx = 0
        self.dg_idx = 1
        self.pv_idx = 2
        self.wind_idx = 3
        self.grid_idx = 4

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """4D continuous action: [P_ess, P_dg, Q_pv, Q_wind]"""
        action = Action()
        action.set_specs(
            dim_c=4,
            range=(np.array([-1.0, -1.0, -1.0, -1.0]),
                   np.array([1.0, 1.0, 1.0, 1.0]))
        )
        action.set_values(c=np.zeros(4))
        return action

    def set_action(self, action: Any) -> None:
        """Set action from policy output"""
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        else:
            self.action.set_values(c=action)

    def apply_action(self):
        """Apply actions to device features"""
        # Denormalize actions from [-1, 1] to physical ranges
        ess_feature = self.state.features[self.ess_idx]
        dg_feature = self.state.features[self.dg_idx]
        pv_feature = self.state.features[self.pv_idx]
        wind_feature = self.state.features[self.wind_idx]

        # ESS power: [-0.5, 0.5] MW
        P_ess = self.action.c[0] * 0.5

        # DG power: [min_p, max_p] MW
        P_dg_norm = (self.action.c[1] + 1) / 2  # [0, 1]
        P_dg = dg_feature.min_p + P_dg_norm * (dg_feature.max_p - dg_feature.min_p)

        # PV/Wind reactive power
        Q_pv = self.action.c[2] * pv_feature.max_q
        Q_wind = self.action.c[3] * wind_feature.max_q

        # Update features (these will be used in simulation)
        ess_feature.set_values(P=P_ess)
        dg_feature.set_values(P=P_dg)
        pv_feature.set_values(Q=Q_pv)
        wind_feature.set_values(Q=Q_wind)

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward based on cost and safety"""
        # Extract features
        ess_state = local_state.get("ESSFeature", {})
        dg_state = local_state.get("DGFeature", {})
        grid_state = local_state.get("GridFeature", {})

        # Cost components
        ess_cost = abs(ess_state.get("P", 0.0)) * 0.1  # Cycling cost
        dg_cost = dg_state.get("P", 0.0) ** 2 * 10  # Fuel cost (quadratic)
        grid_cost = grid_state.get("P", 0.0) * grid_state.get("price", 50.0) * 0.001  # Grid cost

        total_cost = ess_cost + dg_cost + grid_cost

        # Safety violations (computed in simulation, stored in state)
        # For now, assume stored in state or computed here
        safety_violation = 0.0  # Placeholder

        # Reward = -cost - penalty * safety
        reward = -total_cost - 10.0 * safety_violation

        return reward
```

#### Step 2: Create Microgrid Environment

**File: `case_studies/power/powergrid/envs/microgrid_env.py`**

```python
from heron.envs.base import MultiAgentEnv
from heron.agents.system_agent import SystemAgent
from case_studies.power.powergrid.agents.device_agents import MicrogridFieldAgent
import pandapower as pp
import pandapower.networks as pn

class MicrogridEnv(MultiAgentEnv):
    """Multi-agent microgrid environment using Heron"""

    def __init__(
        self,
        num_microgrids: int = 3,
        episode_steps: int = 24,
        **kwargs
    ):
        # Create microgrid agents
        mg_agents = {}
        for i in range(1, num_microgrids + 1):
            mg_agents[f"MG{i}"] = MicrogridFieldAgent(
                agent_id=f"MG{i}",
                ess_capacity=2.0,
                dg_max_p=0.66 if i == 1 else (0.60 if i == 2 else 0.50),
            )

        # Create system agent
        system_agent = SystemAgent(
            agent_id="system",
            subordinates=mg_agents,
        )

        # Initialize pandapower network
        self.net = self._create_network(num_microgrids)
        self.episode_steps = episode_steps
        self._timestep = 0

        super().__init__(
            system_agent=system_agent,
            **kwargs
        )

    def _create_network(self, num_microgrids: int):
        """Create networked microgrid topology"""
        # Main grid (DSO)
        net_dso = pn.case_ieee34()

        # Add microgrids
        for i in range(1, num_microgrids + 1):
            net_mg = pn.case_ieee13()
            # Merge networks and connect at specific buses
            # ... (network topology setup)

        return net_dso  # Return combined network

    def run_simulation(self, env_state: Any, *args, **kwargs) -> Any:
        """Run Pandapower AC power flow simulation"""
        # Extract device setpoints from env_state
        # Update pandapower network elements
        # Run power flow
        try:
            pp.runpp(self.net)
            converged = True
        except:
            converged = False

        # Extract results and update env_state
        # - Bus voltages
        # - Line flows
        # - Power injections

        return env_state

    def env_state_to_global_state(self, env_state: Any) -> Dict:
        """Convert simulation results to global state dict"""
        agent_states = {}

        for agent_id, agent in self.registered_agents.items():
            if hasattr(agent, 'state') and agent.state:
                state_dict = agent.state.to_dict(include_metadata=True)

                # Update with simulation results
                # - ESS SOC from power flow
                # - Grid power exchange
                # - Voltage measurements

                agent_states[agent_id] = state_dict

        return {"agent_states": agent_states}

    def global_state_to_env_state(self, global_state: Dict) -> Any:
        """Convert global state to simulation input"""
        # Extract device setpoints from agent states
        # Prepare for power flow simulation
        return global_state  # Or custom EnvState object
```

#### Step 3: Training Script

**File: `case_studies/power/powergrid/train_microgrids.py`**

```python
from heron.envs.base import MultiAgentEnv
from case_studies.power.powergrid.envs.microgrid_env import MicrogridEnv
from heron.core.policies import Policy
import numpy as np

# Use same training approach as test_e2e.py
def train_microgrid_ctde(
    env: MultiAgentEnv,
    num_episodes: int = 100,
    steps_per_episode: int = 24,
):
    """Train microgrid policies using CTDE"""

    # Get agent IDs
    agent_ids = [aid for aid, agent in env.registered_agents.items()
                 if agent.action_space is not None]

    # Initialize policies (one per microgrid)
    policies = {
        aid: NeuralPolicy(obs_dim=agent.observation_space.shape[0])
        for aid, agent in env.registered_agents.items()
        if aid in agent_ids
    }

    # Training loop (same as test_e2e.py)
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)

        for step in range(steps_per_episode):
            # Compute actions
            actions = {
                aid: policies[aid].forward(obs[aid])
                for aid in agent_ids
            }

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Update policies
            # ... (same as test_e2e.py)

    return policies

# Create and train
env = MicrogridEnv(num_microgrids=3)
trained_policies = train_microgrid_ctde(env, num_episodes=100)
```

#### Step 4: RLlib Integration (Optional)

**File: `case_studies/power/powergrid/train_rllib.py`**

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from gymnasium.wrappers import FlattenObservation

def env_creator(config):
    """Create Heron environment for RLlib"""
    env = MicrogridEnv(num_microgrids=3)
    # Wrap if needed for RLlib compatibility
    return env

# Register environment
register_env("heron_microgrid", env_creator)

# Configure RLlib
config = (
    PPOConfig()
    .environment("heron_microgrid")
    .multi_agent(
        policies={
            "MG1": ...,
            "MG2": ...,
            "MG3": ...,
        },
        policy_mapping_fn=lambda agent_id, *args: agent_id,
    )
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
    )
)

# Train
algo = config.build()
for i in range(200):
    result = algo.train()
    print(f"Iteration {i}: reward = {result['episode_reward_mean']}")
```

### 6.4 Key Differences and Adaptations

#### Differences:

1. **Hierarchy Level**:
   - GridAges: Flat multi-agent (each MG is independent agent)
   - Heron: Hierarchical (SystemAgent → FieldAgents)

2. **Device Modeling**:
   - GridAges: Separate Device classes with Device.update_state()
   - Heron: FeatureProvider pattern within agent state

3. **Power Flow Integration**:
   - GridAges: Embedded in NetworkedGridEnv.step()
   - Heron: Separate simulation phase in EnvCore.run_simulation()

4. **Observation/Action Spaces**:
   - GridAges: Directly exposed via Gymnasium spaces
   - Heron: Constructed from FeatureProvider and Action specifications

#### Adaptations Needed:

1. **Feature Providers**: Create custom features for each device type (ESS, DG, RES, Grid)
2. **Action Denormalization**: Convert normalized [-1, 1] actions to physical ranges
3. **Reward Function**: Implement as compute_local_reward() in FieldAgent
4. **Power Flow**: Integrate Pandapower in run_simulation()
5. **Network Topology**: Build network structure in environment initialization

### 6.5 Benefits of Heron Migration

1. **Hierarchical Control**: Natural support for DSO → microgrid → device hierarchy
2. **Event-Driven Testing**: Test policies with realistic timing delays
3. **Protocol Integration**: Use SetpointProtocol for centralized control
4. **Message Broker**: Enable distributed training/testing
5. **Feature Visibility**: Fine-grained control over information sharing
6. **Extensibility**: Easy to add new device types or coordination strategies

---

## 7. Verification and Testing

### 7.1 Migration Checklist

- [ ] Create device FeatureProviders (ESS, DG, RES, Grid)
- [ ] Implement MicrogridFieldAgent with 4D action space
- [ ] Build network topology using Pandapower
- [ ] Integrate power flow in run_simulation()
- [ ] Implement reward computation
- [ ] Test single-agent step
- [ ] Test multi-agent coordination
- [ ] Compare rewards with GridAges baseline
- [ ] Train policies using CTDE
- [ ] Evaluate in event-driven mode

### 7.2 Validation Tests

```python
def test_device_features():
    """Test FeatureProvider implementations"""
    ess = ESSFeature(capacity=2.0, soc=0.5)
    ess.set_values(P=0.3, soc=0.6)
    assert 0 <= ess.soc <= 1.0
    assert ess.vector().shape == (3,)  # P, Q, SOC

def test_action_space():
    """Test action space matches GridAges"""
    agent = MicrogridFieldAgent(agent_id="test")
    assert agent.action_space.shape == (4,)
    assert agent.action.dim_c == 4

def test_power_flow():
    """Test Pandapower integration"""
    env = MicrogridEnv(num_microgrids=1)
    obs, _ = env.reset()
    actions = {"MG1": np.array([0.0, 0.5, 0.0, 0.0])}
    obs, rewards, _, _, _ = env.step(actions)
    # Verify power flow converged

def test_reward_consistency():
    """Compare rewards with GridAges"""
    # Run same scenario in both frameworks
    # Assert rewards are within 5% tolerance
```

---

## 8. Summary

### 8.1 GridAges Key Takeaways

- **Modular device abstraction** with base Device class
- **Pandapower backend** for AC power flow simulation
- **PettingZoo integration** for multi-agent RL
- **Flat agent hierarchy** (each microgrid is independent)
- **Rich reward signal** combining cost and safety

### 8.2 Heron Advantages

- **Hierarchical control** with SystemAgent → CoordinatorAgent → FieldAgent
- **Event-driven execution** for realistic timing
- **Protocol-based coordination** for flexible control strategies
- **Feature visibility** for information control
- **Extensible architecture** for new device types

### 8.3 Migration Effort

**Estimated Lines of Code:**
- Device FeatureProviders: ~200 LOC
- MicrogridFieldAgent: ~150 LOC
- MicrogridEnv: ~200 LOC
- Training script: ~100 LOC
- Tests: ~150 LOC

**Total: ~800 LOC**

**Estimated Time: 3-5 days** (implementation + testing + validation)

---

## Sources

- [GridAges GitHub Repository](https://github.com/hepengli/GridAges)
- [Hepeng Li's Publications](https://www.hepengli.me/publications)
- Hepeng Li, "Optimal Operation of Networked Microgrids With Distributed Multi-Agent Reinforcement Learning," IEEE PES General Meeting 2024 (Best Paper Award)

# PowerGrid 2.0: Architecture Diagrams (Implemented)

**Purpose**: Visual guide for the implemented architecture
**Date**: 2025-11-10
**Version**: 2.0 (Reflects actual implementation)
**Status**: ✅ Up-to-date

---

## Diagram Index

1. [Implemented System Architecture](#1-implemented-system-architecture)
2. [Class Hierarchy (Actual)](#2-class-hierarchy-actual)
3. [Agent Lifecycle Sequence](#3-agent-lifecycle-sequence)
4. [Environment Step Flow (Implemented)](#4-environment-step-flow-implemented)
5. [Protocol System Design](#5-protocol-system-design)
6. [Feature-Based State System](#6-feature-based-state-system)
7. [Module Dependencies (Actual)](#7-module-dependencies-actual)

---

## 1. Implemented System Architecture

```mermaid
graph TB
    subgraph "User Interface"
        User[Researcher/User]
        RLlib[RLlib/Ray]
    end

    subgraph "PowerGrid Core - Implemented"
        subgraph "Environment Layer - envs/"
            MAEnv[NetworkedGridEnv<br/>PettingZoo ParallelEnv]
            HProto[Horizontal Protocol<br/>Environment-Owned]
        end

        subgraph "Agent Layer - agents/"
            GridAgent[GridAgent<br/>Level 2]
            PowerGridAgent[PowerGridAgentV2<br/>PandaPower Integration]
            DevAgent[DeviceAgent<br/>Level 1]
            VProto[Vertical Protocol<br/>Agent-Owned]
        end

        subgraph "Device Layer - devices/"
            Gen[Generator<br/>DG with UC]
            ESS[ESS<br/>Storage]
            Grid[Grid<br/>Buy/Sell]
        end

        subgraph "Core Primitives - core/"
            State[DeviceState<br/>Feature Aggregator]
            Action[Action<br/>Continuous+Discrete]
            Protocols[Protocols<br/>Vertical+Horizontal]
        end

        subgraph "Features - features/"
            Electrical[ElectricalBasePh]
            Storage[StorageBlock]
            Status[StatusBlock]
            Limits[GeneratorLimits<br/>StorageLimits]
        end
    end

    subgraph "Physics Backend"
        PP[PandaPower<br/>AC Power Flow]
    end

    %% User connections
    User -->|trains| RLlib
    RLlib -->|wraps| MAEnv

    %% Environment structure
    MAEnv -->|manages| PowerGridAgent
    MAEnv -->|owns| HProto
    PowerGridAgent -->|is-a| GridAgent
    GridAgent -->|coordinates| DevAgent
    GridAgent -->|owns| VProto

    %% Device wrapping
    DevAgent -->|wraps| Gen
    DevAgent -->|wraps| ESS
    DevAgent -->|wraps| Grid

    %% Core primitives
    GridAgent -->|uses| Protocols
    DevAgent -->|has| State
    DevAgent -->|has| Action
    State -->|aggregates| Electrical
    State -->|aggregates| Storage
    State -->|aggregates| Status
    State -->|aggregates| Limits

    %% Physics
    MAEnv -->|power flow| PP
    PowerGridAgent -->|syncs with| PP

    style MAEnv fill:#fff4e1
    style GridAgent fill:#e8f5e9
    style DevAgent fill:#e8f5e9
    style PowerGridAgent fill:#e8f5e9
    style Gen fill:#f3e5f5
    style ESS fill:#f3e5f5
    style Grid fill:#f3e5f5
    style PP fill:#e3f2fd
    style HProto fill:#ffe0b2
    style VProto fill:#c8e6c9
```

---

## 2. Class Hierarchy (Actual)

```mermaid
classDiagram
    class Agent {
        <<abstract>>
        +agent_id: str
        +level: int
        +action_space: Space
        +observation_space: Space
        +mailbox: List[Message]
        +observe(global_state) Observation
        +act(observation) Action
        +receive_message(msg)
        +send_message(msg, recipients)
        +reset(seed)
    }

    class DeviceAgent {
        +state: DeviceState
        +action: Action
        +policy: Optional[Policy]
        +protocol: Protocol
        +cost: float
        +safety: float
        +set_action_space()
        +set_device_state()
        +_get_action_space() Space
        +_get_observation_space() Space
        +update_state()
        +update_cost_safety()
        +reset_device()
    }

    class GridAgent {
        +devices: Dict[AgentID, DeviceAgent]
        +protocol: Protocol
        +policy: Optional[Policy]
        +centralized: bool
        +observe(global_state) Observation
        +act(observation) Action
        +coordinate_device(obs, action)
        +build_local_observation()
    }

    class PowerGridAgentV2 {
        +net: pandapowerNet
        +sgen: Dict[str, Generator]
        +storage: Dict[str, ESS]
        +base_power: float
        +add_sgen(generators)
        +add_storage(storages)
        +update_state(net, t)
        +sync_global_state(net, t)
        +update_cost_safety(net)
        +_get_obs(net)
    }

    class Generator {
        +_generator_config: GeneratorConfig
        +electrical: ElectricalBasePh
        +status: StatusBlock
        +limits: GeneratorLimits
        +update_state()
        +update_cost_safety()
    }

    class ESS {
        +_ess_config: ESSConfig
        +electrical: ElectricalBasePh
        +storage: StorageBlock
        +limits: StorageLimits
        +update_state()
        +update_cost_safety()
    }

    class Grid {
        +type: str
        +sn_mva: float
        +sell_discount: float
        +update_state()
        +update_cost_safety()
    }

    class DeviceState {
        <<dataclass>>
        +phase_model: PhaseModel
        +phase_spec: PhaseSpec
        +features: List[FeatureProvider]
        +prefix_names: bool
        +vector() Array
        +names() List[str]
        +clamp_()
        +to_dict()
        +from_dict()
    }

    class Action {
        <<dataclass>>
        +c: FloatArray
        +d: IntArray
        +dim_c: int
        +dim_d: int
        +ncats: int|Sequence[int]
        +range: Tuple[Array, Array]
        +masks: List[ndarray]
        +sample()
        +scale() / unscale()
        +clip_()
    }

    class Protocol {
        <<abstract>>
        +no_op() bool
        +sync_global_state(agents, net, t)
        +coordinate_messages(agents, obs, net, t)
        +coordinate_actions(agents, actions, net, t)
    }

    class VerticalProtocol {
        <<abstract>>
        +coordinate(sub_obs, parent_action) Dict
        +coordinate_action(devices, obs, action)
        +coordinate_message(devices, obs, action)
    }

    class HorizontalProtocol {
        <<abstract>>
        +coordinate(agents, obs, topology) Dict
        +coordinate_actions(agents, obs, actions, net)
    }

    class PriceSignalProtocol {
        +price: float
        +coordinate(sub_obs, parent_action) Dict
        +coordinate_message(devices, obs, action)
    }

    class PeerToPeerTradingProtocol {
        +trading_fee: float
        +coordinate(agents, obs, topology) Dict
        -_clear_market(bids, offers) List
    }

    class NetworkedGridEnv {
        <<PettingZoo ParallelEnv>>
        +net: pandapowerNet
        +agent_dict: Dict[str, PowerGridAgentV2]
        +protocol: Protocol
        +possible_agents: List[str]
        +action_spaces: Dict
        +observation_spaces: Dict
        +step(actions) Tuple
        +reset() Tuple
        -_build_net()
        -_reward_and_safety() Tuple
    }

    Agent <|-- DeviceAgent
    Agent <|-- GridAgent
    GridAgent <|-- PowerGridAgentV2

    DeviceAgent <|-- Generator
    DeviceAgent <|-- ESS
    DeviceAgent <|-- Grid

    DeviceAgent o-- DeviceState
    DeviceAgent o-- Action
    DeviceAgent o-- Protocol
    GridAgent o-- DeviceAgent
    GridAgent o-- Protocol

    Protocol <|-- VerticalProtocol
    Protocol <|-- HorizontalProtocol
    VerticalProtocol <|-- PriceSignalProtocol
    HorizontalProtocol <|-- PeerToPeerTradingProtocol

    NetworkedGridEnv o-- PowerGridAgentV2
    NetworkedGridEnv o-- Protocol
```

---

## 3. Agent Lifecycle Sequence

```mermaid
sequenceDiagram
    participant User
    participant Env as NetworkedGridEnv
    participant GridAgent as PowerGridAgentV2
    participant DevAgent as DeviceAgent (ESS)
    participant PP as PandaPower

    Note over User,PP: Initialization

    User->>Env: __init__(env_config)
    Env->>Env: _build_net()

    loop For each microgrid
        Env->>GridAgent: __init__(net, grid_config, devices)

        loop For each device
            GridAgent->>DevAgent: __init__(device_config)
            DevAgent->>DevAgent: set_action_space()
            DevAgent->>DevAgent: set_device_state()
            DevAgent-->>GridAgent: device instance
        end

        GridAgent->>PP: add_sgen/add_storage (sync devices to net)
        GridAgent-->>Env: grid agent instance
    end

    Env->>Env: _init_space() (build action/obs spaces)

    Note over User,PP: Training Loop

    User->>Env: reset(seed=42)

    loop For each GridAgent
        Env->>GridAgent: reset()
        loop For each DeviceAgent
            GridAgent->>DevAgent: reset_device()
        end
    end

    Env->>PP: runpp(net)
    Env->>Env: _get_obs()
    Env-->>User: observations, info

    loop Training steps
        User->>User: policy.forward(obs) → actions
        User->>Env: step(actions)

        Note over Env,DevAgent: 1. Set Actions & Update States

        loop For each GridAgent
            Env->>GridAgent: act(obs, upstream_action=action)
            GridAgent->>GridAgent: protocol.coordinate_action()

            loop For each DeviceAgent
                GridAgent->>DevAgent: _set_device_action(action_slice)
                DevAgent->>DevAgent: update_state()
            end

            GridAgent->>PP: sync device states to net
        end

        Note over Env,PP: 2. Run Power Flow

        Env->>PP: runpp(net)
        PP-->>Env: converged, results

        Note over Env,DevAgent: 3. Sync Results & Compute Metrics

        loop For each GridAgent
            Env->>GridAgent: sync_global_state(net, t)

            loop For each DeviceAgent
                GridAgent->>DevAgent: update P/Q from results
            end

            Env->>GridAgent: update_cost_safety(net)

            loop For each DeviceAgent
                GridAgent->>DevAgent: update_cost_safety()
            end
        end

        Env->>Env: _reward_and_safety()
        Env->>Env: _get_obs()
        Env-->>User: obs, rewards, dones, truncs, infos

        User->>User: policy.update(experience)
    end
```

---

## 4. Environment Step Flow (Implemented)

```mermaid
flowchart TD
    Start([step: actions])

    subgraph "1. Action Distribution & State Update"
        A1[Parse action_dict by agent_id]
        A2{Protocol.no_op?}
        A3[For each GridAgent:<br/>agent.act upstream_action]
        A4[Protocol.coordinate_actions<br/>agents, actions, net, t]
        A5[GridAgent.protocol.coordinate_action<br/>Distribute to DeviceAgents]
        A6[For each DeviceAgent:<br/>update_state]
    end

    subgraph "2. Sync to PandaPower"
        B1[For each GridAgent:<br/>update_state net, t]
        B2[net.sgen/storage ← device.state]
    end

    subgraph "3. Power Flow Solution"
        C1[pp.runpp net]
        C2{Converged?}
        C3[Get results:<br/>vm_pu, va_degree, loading]
        C4[net converged = False]
    end

    subgraph "4. Sync Results to Devices"
        D1{Protocol.no_op?}
        D2[For each GridAgent:<br/>sync_global_state net, t]
        D3[Protocol.sync_global_state<br/>agents, net, t]
        D4[device.state ← net.res_*]
    end

    subgraph "5. Cost & Safety Computation"
        E1[For each GridAgent:<br/>update_cost_safety net]
        E2[For each DeviceAgent:<br/>update_cost_safety]
        E3[Grid cost/safety ← sum devices]
    end

    subgraph "6. Reward & Observation"
        F1[_reward_and_safety]
        F2[Share rewards if configured]
        F3[_get_obs]
        F4[Increment timestep counters]
    end

    End([Return: obs, rewards, dones, truncs, infos])

    Start --> A1
    A1 --> A2
    A2 -->|Yes| A3
    A2 -->|No| A4
    A3 --> A5
    A4 --> A6
    A5 --> A6
    A6 --> B1

    B1 --> B2
    B2 --> C1

    C1 --> C2
    C2 -->|Yes| C3
    C2 -->|No| C4
    C3 --> D1
    C4 --> D1

    D1 -->|Yes| D2
    D1 -->|No| D3
    D2 --> D4
    D3 --> D4
    D4 --> E1

    E1 --> E2
    E2 --> E3
    E3 --> F1

    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> End

    style Start fill:#e8f5e9
    style End fill:#e8f5e9
    style C1 fill:#e3f2fd
    style C2 fill:#fff3e0
    style A2 fill:#fff3e0
    style D1 fill:#fff3e0
```

---

## 5. Protocol System Design

```mermaid
graph TB
    subgraph "Vertical Protocols - Agent-Owned"
        VP[VerticalProtocol<br/>ABC]
        NoP[NoProtocol<br/>No coordination]
        Price[PriceSignalProtocol<br/>Broadcast price]
        Setp[SetpointProtocol<br/>Assign setpoints]
        CentSetp[CentralizedSetpointProtocol<br/>Direct distribution]
    end

    subgraph "Horizontal Protocols - Environment-Owned"
        HP[HorizontalProtocol<br/>ABC]
        NoHP[NoHorizontalProtocol<br/>No peer coordination]
        P2P[PeerToPeerTradingProtocol<br/>Energy marketplace]
        Cons[ConsensusProtocol<br/>Gossip averaging]
    end

    subgraph "Protocol Base"
        Base[Protocol<br/>ABC]
    end

    subgraph "Use Cases"
        subgraph "Vertical - Parent to Child"
            UC1[GridAgent → DeviceAgents<br/>Price signal coordination]
            UC2[GridAgent → DeviceAgents<br/>Centralized setpoints]
        end

        subgraph "Horizontal - Peer to Peer"
            UC3[GridAgent ↔ GridAgent<br/>P2P energy trading]
            UC4[GridAgent ↔ GridAgent<br/>Frequency consensus]
        end
    end

    Base --> VP
    Base --> HP

    VP --> NoP
    VP --> Price
    VP --> Setp
    VP --> CentSetp

    HP --> NoHP
    HP --> P2P
    HP --> Cons

    Price -.->|used in| UC1
    CentSetp -.->|used in| UC2
    P2P -.->|used in| UC3
    Cons -.->|used in| UC4

    style VP fill:#c8e6c9
    style HP fill:#ffe0b2
    style UC1 fill:#e8f5e9
    style UC2 fill:#e8f5e9
    style UC3 fill:#fff4e1
    style UC4 fill:#fff4e1
```

**Key Design Principle**:
- **Vertical**: Decentralized (each agent manages children)
- **Horizontal**: Centralized (environment provides infrastructure)

---

## 6. Feature-Based State System

```mermaid
graph LR
    subgraph "DeviceState Container"
        DS[DeviceState<br/>phase_model<br/>phase_spec<br/>prefix_names]
    end

    subgraph "Feature Providers"
        Elec[ElectricalBasePh<br/>P_MW, Q_MVAr]
        Stor[StorageBlock<br/>soc, e_stored_MWh]
        Stat[StatusBlock<br/>state, t_in_state]
        Conn[PhaseConnection<br/>connection: ABC/ON]
        GenLim[GeneratorLimits<br/>p_max, q_max, s_rated]
        StorLim[StorageLimits<br/>e_max, p_max, q_max]
    end

    subgraph "Outputs"
        Vec[vector<br/>Concatenated float32]
        Names[names<br/>Feature labels]
    end

    DS -->|aggregates| Elec
    DS -->|aggregates| Stor
    DS -->|aggregates| Stat
    DS -->|aggregates| Conn
    DS -->|aggregates| GenLim
    DS -->|aggregates| StorLim

    Elec -->|contributes to| Vec
    Stor -->|contributes to| Vec
    Stat -->|contributes to| Vec
    Conn -->|contributes to| Vec
    GenLim -->|contributes to| Vec
    StorLim -->|contributes to| Vec

    Elec -->|provides| Names
    Stor -->|provides| Names
    Stat -->|provides| Names
    Conn -->|provides| Names
    GenLim -->|provides| Names
    StorLim -->|provides| Names

    style DS fill:#fff4e1
    style Elec fill:#e3f2fd
    style Stor fill:#e3f2fd
    style Stat fill:#e3f2fd
    style Vec fill:#c8e6c9
    style Names fill:#c8e6c9
```

**Phase Context Enforcement**:
1. DeviceState validates phase_model + phase_spec
2. Override all child features' phase context
3. Trigger revalidation on each feature
4. Concatenate vectors in feature order

---

## 7. Module Dependencies (Actual)

```mermaid
graph TD
    subgraph "Layer 1: Primitives"
        Core[core/<br/>state, action, protocols]
        Utils[utils/<br/>cost, safety, phase]
    end

    subgraph "Layer 2: Features"
        Feat[features/<br/>electrical, storage,<br/>status, connection]
    end

    subgraph "Layer 3: Agents"
        Agent[agents/<br/>base, device_agent,<br/>grid_agent]
    end

    subgraph "Layer 4: Devices"
        Dev[devices/<br/>generator, storage,<br/>grid, compensation]
    end

    subgraph "Layer 5: Environments"
        Env[envs/<br/>multi_agent/<br/>networked_grid_env]
    end

    subgraph "Layer 6: Networks"
        Net[networks/<br/>ieee13, ieee34,<br/>ieee123, cigre_lv]
    end

    subgraph "External"
        PP[PandaPower]
        Gym[Gymnasium]
        PZ[PettingZoo]
    end

    Core --> Feat
    Utils --> Feat
    Core --> Agent
    Feat --> Agent

    Agent --> Dev
    Core --> Dev
    Feat --> Dev

    Agent --> Env
    Core --> Env

    Net --> Env

    PP --> Dev
    PP --> Env
    Gym --> Agent
    PZ --> Env

    style Core fill:#e8f5e9
    style Feat fill:#c8e6c9
    style Agent fill:#e8f5e9
    style Dev fill:#f3e5f5
    style Env fill:#fff4e1
    style PP fill:#e3f2fd
```

**Implementation Order**:
1. Core primitives (state, action, protocols)
2. Features (electrical, storage, status)
3. Agents (base, device, grid)
4. Devices (generator, ess, grid)
5. Environments (networked_grid_env)
6. Networks (ieee13, ieee34, etc.)

---

## Quick Reference: Implemented Files

### Core Modules
```
powergrid/
├── core/
│   ├── state.py              # DeviceState, PhaseModel, PhaseSpec
│   ├── action.py             # Action (continuous + discrete)
│   ├── protocols.py          # Vertical + Horizontal protocols
│   └── policies.py           # Policy interface
```

### Agent Layer
```
powergrid/
├── agents/
│   ├── base.py               # Agent ABC, Observation, Message
│   ├── device_agent.py       # DeviceAgent wrapper
│   └── grid_agent.py         # GridAgent, PowerGridAgentV2
```

### Device Layer
```
powergrid/
├── devices/
│   ├── generator.py          # Generator (DG with UC)
│   ├── storage.py            # ESS
│   ├── grid.py               # Grid connection
│   └── compensation.py       # Shunt, SVC (WIP)
```

### Features
```
powergrid/
├── features/
│   ├── base.py               # FeatureProvider protocol
│   ├── electrical.py         # ElectricalBasePh
│   ├── storage.py            # StorageBlock
│   ├── status.py             # StatusBlock
│   ├── connection.py         # PhaseConnection
│   ├── generator_limits.py   # GeneratorLimits
│   └── ...                   # More features
```

### Environment
```
powergrid/
├── envs/
│   └── multi_agent/
│       ├── networked_grid_env.py     # NetworkedGridEnv (PettingZoo)
│       └── multi_agent_microgrids.py # Example environments
```

---

## Usage Patterns

### Pattern 1: Create Device with Features
```python
from powergrid.agents.generator import Generator

gen = Generator(
    agent_id="dg1",
    device_config={
        "device_state_config": {
            "phase_model": "balanced_1ph",
            "p_max_MW": 5.0,
            "q_max_MVAr": 3.0,
            "cost_curve_coefs": [0.01, 1.0, 0.0]
        }
    }
)

# gen.state contains:
# - ElectricalBasePh (P, Q)
# - StatusBlock (UC state)
# - PhaseConnection (bus connection)
# - GeneratorLimits (capability)
```

### Pattern 2: Coordinate with Vertical Protocol
```python
from powergrid.agents.grid_agent import GridAgent
from powergrid.core.protocols import PriceSignalProtocol

grid = GridAgent(
    agent_id="mg1",
    devices=[ess, gen, solar],
    protocol=PriceSignalProtocol(initial_price=50.0)
)

# Protocol broadcasts price to all devices
obs = grid.observe(global_state)
action = grid.act(obs, upstream_action=parent_action)
# → protocol.coordinate_message() sends price to devices
```

### Pattern 3: Multi-Agent Environment
```python
from powergrid.envs.multi_agent.networked_grid_env import NetworkedGridEnv
from powergrid.core.protocols import PeerToPeerTradingProtocol

env = NetworkedGridEnv(env_config={
    "max_episode_steps": 24,
    "protocol": PeerToPeerTradingProtocol(trading_fee=0.01)
})

# Environment coordinates GridAgents via P2P protocol
obs, info = env.reset()
actions = {agent_id: policy(obs[agent_id]) for agent_id in env.agents}
obs, rewards, dones, truncs, infos = env.step(actions)
```

---

## Conclusion

PowerGrid 2.0 implements a **clean, modular architecture** with:
- ✅ Hierarchical agent system (2 levels)
- ✅ Dual protocol system (vertical + horizontal)
- ✅ Feature-based device state
- ✅ PettingZoo-compatible environment
- ✅ PandaPower integration

**Design Philosophy**: Separation of concerns between agent logic, device physics, coordination protocols, and environment simulation.

---

**Document Maintainer**: PowerGrid Development Team
**Last Updated**: 2025-11-10
**Next Update**: After SystemAgent implementation

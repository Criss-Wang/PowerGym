# HERON Flow Diagrams & Novelty Analysis

This document provides visual Mermaid diagrams for the HERON framework's execution and data flows, alongside justifications for the novelty of each design choice.

---

## 1. High-Level Architecture Overview

```mermaid
graph TB
    subgraph "HERON Framework"
        direction TB

        subgraph "Agent Hierarchy"
            SA[SystemAgent L3<br/>Global Coordination<br/>tick: 300s]
            CA[CoordinatorAgent L2<br/>Zone Management<br/>tick: 60s]
            FA1[FieldAgent L1<br/>Local Control<br/>tick: 1s]
            FA2[FieldAgent L1<br/>Local Control<br/>tick: 1s]

            SA --> CA
            CA --> FA1
            CA --> FA2
        end

        subgraph "State Mediation"
            PA[ProxyAgent<br/>State Cache + Visibility Filter]
        end

        subgraph "Communication"
            MB[MessageBroker<br/>Channel-Based Routing]
        end

        subgraph "Execution Engine"
            ES[EventScheduler<br/>Heap-Based Priority Queue]
        end

        subgraph "Coordination"
            PR[Protocol System<br/>Communication + Action]
        end

        SA <-.->|state/obs| PA
        CA <-.->|state/obs| PA
        FA1 <-.->|state/obs| PA
        FA2 <-.->|state/obs| PA

        SA ---|action msgs| MB
        CA ---|action msgs| MB

        ES -.->|triggers| SA
        ES -.->|triggers| CA
        ES -.->|triggers| FA1
        ES -.->|triggers| FA2

        SA ---|coordinate| PR
        CA ---|coordinate| PR
    end

    style SA fill:#e74c3c,color:#fff
    style CA fill:#f39c12,color:#fff
    style FA1 fill:#27ae60,color:#fff
    style FA2 fill:#27ae60,color:#fff
    style PA fill:#3498db,color:#fff
    style MB fill:#9b59b6,color:#fff
    style ES fill:#1abc9c,color:#fff
    style PR fill:#e67e22,color:#fff
```

**Novelty:** Unlike PettingZoo (which standardizes the env-algorithm _interface_), HERON standardizes _what happens inside the environment_: how agents access state, communicate, and coordinate. This is a fundamentally different abstraction level.

---

## 2. Dual Execution Modes

### 2.1 CTDE Training Mode (Synchronous)

```mermaid
sequenceDiagram
    participant RL as RL Algorithm
    participant Env as Environment
    participant SYS as SystemAgent
    participant COORD as CoordinatorAgent
    participant FA as FieldAgent
    participant PX as ProxyAgent

    RL->>Env: step(actions)
    Env->>SYS: execute(actions, proxy)

    Note over SYS: Phase 0: Pre-step hook
    SYS->>PX: get_local_state(sys_id)
    PX-->>SYS: filtered feature vectors
    Note over SYS: sync_state_from_observed()

    Note over SYS,FA: Phase 1: Hierarchical Action Application
    SYS->>SYS: layer_actions(actions)
    SYS->>SYS: handle_self_action → apply_action()
    SYS->>PX: set_local_state(sys_state)

    SYS->>COORD: execute(coord_actions, proxy)
    COORD->>PX: get_local_state(coord_id)
    PX-->>COORD: filtered vectors
    COORD->>COORD: handle_self_action → apply_action()
    COORD->>PX: set_local_state(coord_state)

    COORD->>FA: execute(field_actions, proxy)
    FA->>PX: get_local_state(field_id)
    PX-->>FA: filtered vectors
    FA->>FA: set_action → apply_action()
    FA->>PX: set_local_state(field_state)

    Note over SYS,PX: Phase 2: Physics Simulation
    SYS->>PX: get_global_states()
    PX-->>SYS: all filtered states
    SYS->>SYS: simulate(global_state)
    SYS->>PX: set_global_state(updated)

    Note over SYS,PX: Phase 3: Observation Collection
    loop For each agent
        SYS->>PX: get_observation(agent_id)
        PX-->>SYS: Observation (visibility-filtered)
    end

    Note over SYS,PX: Phase 4: Reward Computation
    loop For each agent
        SYS->>PX: get_local_state(agent_id)
        PX-->>SYS: filtered vectors + subordinate_rewards
        SYS->>SYS: compute_local_reward()
    end

    SYS->>PX: set_step_result(obs, rewards, ...)
    PX-->>Env: get_step_results() → vectorized
    Env-->>RL: (obs, rewards, terminated, truncated, info)
```

### 2.2 Event-Driven Testing Mode (Asynchronous)

```mermaid
sequenceDiagram
    participant ES as EventScheduler
    participant SYS as SystemAgent
    participant BK as MessageBroker
    participant COORD as CoordinatorAgent
    participant FA as FieldAgent
    participant PX as ProxyAgent

    Note over ES: t=0.0
    ES->>SYS: AGENT_TICK
    SYS->>ES: schedule_agent_tick(coordinator)
    SYS->>BK: send_subordinate_action(coord, action)
    SYS->>ES: schedule_simulation(wait_interval)
    SYS->>PX: request obs (msg_delay)

    Note over ES: t=msg_delay
    ES->>PX: MESSAGE_DELIVERY (obs request)
    PX->>PX: get_observation() + get_local_state()
    PX->>ES: schedule response (msg_delay)

    Note over ES: t=2×msg_delay
    ES->>SYS: MESSAGE_DELIVERY (obs response)
    SYS->>SYS: sync_state + compute_action
    SYS->>ES: schedule_action_effect(act_delay)

    Note over ES: t=coord_tick
    ES->>COORD: AGENT_TICK
    COORD->>BK: consume(action_channel) → upstream_action
    COORD->>COORD: compute_action(upstream_action)
    COORD->>BK: send_subordinate_action(field, action)

    Note over ES: t=field_tick
    ES->>FA: AGENT_TICK
    FA->>BK: consume(action_channel) → upstream_action
    FA->>FA: set_action(upstream) [direct!]
    FA->>ES: schedule_action_effect(act_delay)

    Note over ES: t=field_tick + act_delay
    ES->>FA: ACTION_EFFECT
    FA->>FA: apply_action() → update state
    FA->>PX: send state update (msg_delay)

    Note over ES: t=wait_interval
    ES->>SYS: SIMULATION
    SYS->>PX: request global_state
    PX-->>SYS: global_state (after msg_delay)
    SYS->>SYS: run_simulation()
    SYS->>PX: set_global_state(updated)

    Note over ES: State completion triggers reward cascade
    PX->>COORD: set_state_completion
    COORD->>PX: request local_states for subordinates
    PX-->>COORD: filtered local_states
    COORD->>COORD: compute_local_reward()
```

**Novelty:** HERON is the first MARL framework to offer **native dual-mode execution** — synchronous CTDE for training and event-driven for deployment testing — using the same agent hierarchy and protocol system. Existing frameworks (PettingZoo, EPyMARL, MARLlib) only support synchronous `step()`. Event-driven execution cannot be achieved by wrapping; it requires changing the fundamental execution loop.

---

## 3. ProxyAgent: State Mediation Hub

```mermaid
graph TB
    subgraph "ProxyAgent State Cache"
        direction TB
        SC["state_cache['agents']"]
        GC["state_cache['global']"]

        SC --> S1["battery_1: FieldAgentState<br/>features: Dict[str, FeatureProvider]"]
        SC --> S2["battery_2: FieldAgentState"]
        SC --> S3["coordinator_1: CoordAgentState"]
        SC --> S4["system_agent: SystemAgentState"]

        GC --> ENV["env_context: {price, solar, wind}"]
    end

    subgraph "Inbound Operations"
        SET_L["set_local_state(aid, State obj)"]
        SET_G["set_global_state(Dict)"]
    end

    subgraph "Outbound Operations (Visibility-Filtered)"
        GET_L["get_local_state(aid, protocol)<br/>→ Dict[str, np.ndarray]<br/>+ optional subordinate_rewards"]
        GET_G["get_global_states(aid, protocol)<br/>→ Dict[aid, Dict[str, np.ndarray]]<br/>+ env_context"]
        GET_O["get_observation(aid, protocol)<br/>→ Observation(local, global_info)"]
    end

    subgraph "Visibility Filter"
        VF["state.observed_by(requestor_id, level)<br/>public | owner | upper_level | system"]
    end

    SET_L --> SC
    SET_G --> GC
    SC --> VF
    VF --> GET_L
    VF --> GET_G
    VF --> GET_O

    style SC fill:#3498db,color:#fff
    style GC fill:#2980b9,color:#fff
    style VF fill:#e74c3c,color:#fff
```

**Novelty:** The ProxyAgent acts as a **single gatekeeper** for all state access, enforcing visibility rules at the feature level. This prevents the common "global state leak" problem in MARL benchmarks where agents inadvertently access information they shouldn't see. No existing framework provides this mediated access pattern.

---

## 4. Feature Visibility System

```mermaid
graph LR
    subgraph "Feature Definition"
        F1["BatterySOC<br/>visibility: public"]
        F2["InternalTemp<br/>visibility: owner"]
        F3["ZoneLoad<br/>visibility: upper_level"]
        F4["GridFrequency<br/>visibility: system"]
    end

    subgraph "Requestors"
        FA["FieldAgent (L1)<br/>battery_1"]
        CA["CoordinatorAgent (L2)<br/>zone_1"]
        SA["SystemAgent (L3)<br/>grid_operator"]
    end

    F1 -->|"✅"| FA
    F1 -->|"✅"| CA
    F1 -->|"✅"| SA

    F2 -->|"✅ (is owner)"| FA
    F2 -->|"❌"| CA
    F2 -->|"❌"| SA

    F3 -->|"❌"| FA
    F3 -->|"✅ (L2 = L1+1)"| CA
    F3 -->|"❌"| SA

    F4 -->|"❌"| FA
    F4 -->|"❌"| CA
    F4 -->|"✅ (L3 ≥ 3)"| SA

    style F1 fill:#27ae60,color:#fff
    style F2 fill:#e74c3c,color:#fff
    style F3 fill:#f39c12,color:#fff
    style F4 fill:#8e44ad,color:#fff
```

**Novelty:** HERON provides **4-level granular visibility** (public, owner, upper_level, system) as a first-class experimental variable, not a binary on/off switch. Researchers can ablate over visibility configurations to study the impact of information structure on multi-agent learning — something no existing framework supports natively.

---

## 5. Action Passing Flow (Hierarchical Coordination)

```mermaid
flowchart TD
    subgraph SystemAgent["SystemAgent (L3)"]
        S_OBS["Observe via ProxyAgent"]
        S_POL["Policy.forward(obs)"]
        S_COORD["Protocol.coordinate()"]
        S_SEND["send_subordinate_action()"]

        S_OBS --> S_POL --> S_COORD --> S_SEND
    end

    subgraph Broker["MessageBroker"]
        ACH1["action_channel<br/>sys → coord_1"]
        ACH2["action_channel<br/>coord_1 → field_1"]
        ACH3["action_channel<br/>coord_1 → field_2"]
    end

    subgraph CoordinatorAgent["CoordinatorAgent (L2)"]
        C_CHECK["_check_for_upstream_action()"]
        C_DECIDE{Has upstream<br/>action?}
        C_USE["Use upstream action<br/>(priority!)"]
        C_OWN["Own policy.forward()"]
        C_COORD["Protocol.coordinate()"]
        C_SEND["send_subordinate_action()"]

        C_CHECK --> C_DECIDE
        C_DECIDE -->|Yes| C_USE --> C_COORD --> C_SEND
        C_DECIDE -->|No| C_OWN --> C_COORD
    end

    subgraph FieldAgent["FieldAgent (L1)"]
        F_CHECK["_check_for_upstream_action()"]
        F_DECIDE{Has upstream<br/>action?}
        F_DIRECT["set_action(upstream)<br/>DIRECT assignment"]
        F_OWN["Own policy.forward()"]
        F_EFFECT["schedule_action_effect(delay)"]
        F_APPLY["apply_action() → update state"]

        F_CHECK --> F_DECIDE
        F_DECIDE -->|Yes| F_DIRECT --> F_EFFECT --> F_APPLY
        F_DECIDE -->|No| F_OWN --> F_EFFECT
    end

    S_SEND --> ACH1
    ACH1 --> C_CHECK
    C_SEND --> ACH2
    C_SEND --> ACH3
    ACH2 --> F_CHECK

    style SystemAgent fill:#ffeaea
    style CoordinatorAgent fill:#fff3e0
    style FieldAgent fill:#e8f5e9
    style Broker fill:#f3e5f5
```

**Key design decisions:**
- **FieldAgent uses direct `set_action()`** — no compute pipeline, just applies the upstream action immediately. This reflects real-world actuators that execute commands without local deliberation.
- **CoordinatorAgent routes through `compute_action()`** — enabling protocol-based decomposition before passing to subordinates.
- **Upstream actions always have priority** over locally computed policies.

---

## 6. Data Type Transformations Across Boundaries

```mermaid
graph TB
    subgraph "Agent Layer (Rich Objects)"
        STATE["State Object<br/>features: Dict[str, FeatureProvider]<br/>e.g. {'BatterySOC': FeatureProvider(soc=0.5)}"]
        ACTION["Action Object<br/>c: np.array, d: np.array"]
        OBS["Observation Object<br/>local: Dict, global_info: Dict"]
    end

    subgraph "Proxy Layer (Object Storage)"
        CACHE["State Cache<br/>Stores State OBJECTS directly<br/>(not serialized!)"]
        VIS["observed_by() filter<br/>→ Dict[str, np.ndarray]"]
    end

    subgraph "Message Layer (Serialized)"
        MSG_S["state.to_dict(include_metadata=True)<br/>→ {'_owner_id': ..., 'features': {...}}"]
        MSG_O["obs.to_dict()<br/>→ {'local': {...}, 'global_info': {...}}"]
        MSG_A["Message(payload={'action': Action})"]
    end

    subgraph "RL Layer (Vectors)"
        VEC_O["obs.vector() / __array__()<br/>→ np.array([0.5, 100.0, ...])"]
        VEC_R["reward: float"]
    end

    STATE -->|"CTDE: direct ref"| CACHE
    STATE -->|"Event: .to_dict()"| MSG_S
    MSG_S -->|"State.from_dict()"| CACHE

    CACHE --> VIS
    VIS -->|"filtered vectors"| OBS

    OBS -->|"Event: .to_dict()"| MSG_O
    MSG_O -->|"Observation.from_dict()"| OBS

    OBS --> VEC_O
    VIS -->|"compute_local_reward()"| VEC_R

    ACTION -->|"Broker: in payload"| MSG_A
    MSG_A -->|"payload['action']"| ACTION

    style STATE fill:#3498db,color:#fff
    style ACTION fill:#e74c3c,color:#fff
    style OBS fill:#27ae60,color:#fff
    style CACHE fill:#2c3e50,color:#fff
    style VIS fill:#e67e22,color:#fff
```

**Key principle:** Serialization only happens at message boundaries. In CTDE mode, State objects pass by reference to the ProxyAgent — zero serialization overhead during training.

---

## 7. Protocol Composition System

```mermaid
graph TB
    subgraph "Protocol (ABC)"
        P["Protocol.coordinate()<br/>→ (messages, actions)"]
    end

    subgraph "Communication Layer"
        CP["CommunicationProtocol"]
        NC["NoCommunication"]
        SS["StateShareProtocol"]
        CUSTOM_C["Custom: ScoutShare, ..."]

        CP --> NC
        CP --> SS
        CP --> CUSTOM_C
    end

    subgraph "Action Layer"
        AP["ActionProtocol"]
        NA["NoActionCoordination"]
        VD["VectorDecomposition<br/>Split joint action by<br/>subordinate dimensions"]
        CUSTOM_A["Custom: PriceSignal, ..."]

        AP --> NA
        AP --> VD
        AP --> CUSTOM_A
    end

    subgraph "Concrete Protocols"
        NP["NoProtocol<br/>= NoCommunication<br/>+ NoAction"]
        VP["VerticalProtocol<br/>= NoCommunication<br/>+ VectorDecomposition"]
        HP["HorizontalProtocol<br/>= StateShare<br/>+ NoAction"]
    end

    P -->|"has"| CP
    P -->|"has"| AP

    NP -.-> NC
    NP -.-> NA
    VP -.-> NC
    VP -.-> VD
    HP -.-> SS
    HP -.-> NA

    style P fill:#8e44ad,color:#fff
    style NP fill:#bdc3c7
    style VP fill:#3498db,color:#fff
    style HP fill:#e67e22,color:#fff
```

**Novelty:** The two-layer protocol design (Communication + Action) enables composable coordination. Researchers can swap coordination mechanisms without modifying agent code — e.g., compare VerticalProtocol (centralized dispatch) vs. HorizontalProtocol (peer consensus) on the same environment. No existing MARL framework provides this level of protocol composability.

---

## 8. Event Scheduler Architecture

```mermaid
graph TB
    subgraph "Event Priority Queue (Min-Heap)"
        direction LR
        E1["t=0.0<br/>AGENT_TICK<br/>system_agent<br/>prio: default"]
        E2["t=0.5<br/>ACTION_EFFECT<br/>field_1<br/>prio: 0 (highest)"]
        E3["t=0.5<br/>SIMULATION<br/>system<br/>prio: 1"]
        E4["t=0.5<br/>MSG_DELIVERY<br/>proxy<br/>prio: 2"]
        E5["t=1.0<br/>AGENT_TICK<br/>field_1<br/>prio: default"]
    end

    subgraph "Event Types & Priorities"
        ET1["ACTION_EFFECT → priority 0<br/>State changes first"]
        ET2["SIMULATION → priority 1<br/>Physics after actions"]
        ET3["MESSAGE_DELIVERY → priority 2<br/>Communication last"]
        ET4["AGENT_TICK → default<br/>Periodic triggers"]
    end

    subgraph "TickConfig (Per-Agent)"
        TC["tick_interval: 1.0s<br/>obs_delay: 0.1s<br/>act_delay: 0.2s<br/>msg_delay: 0.05s<br/>jitter: GAUSSIAN, 10%"]
    end

    subgraph "Ordering: (timestamp, priority, sequence)"
        ORD["Same timestamp?<br/>→ ACTION_EFFECT before SIMULATION<br/>→ SIMULATION before MESSAGE<br/><br/>Same timestamp & priority?<br/>→ FIFO (sequence number)"]
    end

    style E1 fill:#1abc9c,color:#fff
    style E2 fill:#e74c3c,color:#fff
    style E3 fill:#f39c12,color:#fff
    style E4 fill:#9b59b6,color:#fff
    style E5 fill:#1abc9c,color:#fff
```

**Novelty:** The EventScheduler enables **agent-paced execution** with configurable per-agent timing, jitter, and realistic communication delays. Timing configurations can be calibrated to real CPS standards (IEEE 2030 for SCADA, NTCIP 1202 for traffic), making HERON the first MARL framework to treat execution timing as an experimental variable.

---

## 9. Complete CTDE Step: End-to-End Data Flow

```mermaid
flowchart TB
    START["env.step(actions)"] --> PRE["Pre-step hook<br/>(e.g., update load profiles)"]
    PRE --> SYNC["State Sync: All agents<br/>sync_state_from_observed()"]

    SYNC --> LAYER["layer_actions(actions)<br/>Hierarchical structure:<br/>{self: ..., subordinates: {...}}"]

    LAYER --> ACT["Phase 1: Action Application<br/>(top-down through hierarchy)"]

    ACT --> ACT_SYS["SystemAgent:<br/>set_action → apply_action<br/>proxy.set_local_state()"]
    ACT --> ACT_COORD["CoordinatorAgent:<br/>sync → set_action → apply_action<br/>proxy.set_local_state()"]
    ACT --> ACT_FA["FieldAgent:<br/>sync → set_action → apply_action<br/>proxy.set_local_state()"]

    ACT_SYS --> SIM
    ACT_COORD --> SIM
    ACT_FA --> SIM

    SIM["Phase 2: Simulation<br/>proxy.get_global_states()<br/>run_simulation()<br/>proxy.set_global_state()"]

    SIM --> OBS["Phase 3: Observation<br/>For each agent:<br/>proxy.get_observation(aid)<br/>→ Observation (visibility-filtered)"]

    OBS --> REW["Phase 4: Rewards<br/>For each agent:<br/>proxy.get_local_state(aid)<br/>compute_local_reward()"]

    REW --> CACHE["Phase 5: Cache<br/>proxy.set_step_result()"]

    CACHE --> VEC["Vectorize:<br/>obs.vector() for each agent<br/>→ Dict[aid, np.ndarray]"]

    VEC --> RETURN["Return to RL:<br/>(obs, rewards, terminated,<br/>truncated, info)"]

    style START fill:#2c3e50,color:#fff
    style ACT fill:#e74c3c,color:#fff
    style SIM fill:#f39c12,color:#fff
    style OBS fill:#27ae60,color:#fff
    style REW fill:#3498db,color:#fff
    style RETURN fill:#2c3e50,color:#fff
```

---

## 10. Event-Driven Reward Cascade

```mermaid
sequenceDiagram
    participant FA as FieldAgent
    participant PX as ProxyAgent
    participant COORD as CoordinatorAgent

    Note over FA: ACTION_EFFECT fires
    FA->>FA: apply_action() → update self.state
    FA->>PX: {set_state: "local", body: state.to_dict()}

    Note over PX: Receives state update
    PX->>PX: State.from_dict() → set_local_state()
    PX->>COORD: {set_state_completion: "success"}

    Note over COORD: Parent initiates reward cascade
    COORD->>PX: get_local_state(field_1, protocol)
    COORD->>PX: get_local_state(field_2, protocol)
    COORD->>PX: get_local_state(coord_1, protocol)

    PX-->>COORD: field_1 local_state (filtered)
    PX-->>COORD: field_2 local_state (filtered)
    PX-->>COORD: coord_1 local_state (filtered + sub_rewards)

    COORD->>COORD: For each subordinate:<br/>sync_state + compute_local_reward()
    COORD->>PX: {set_tick_result: {reward: 0.5, terminated: false, ...}}

    Note over COORD: If not terminated → schedule next tick
```

**Novelty:** The parent-initiated reward cascade reflects real CPS patterns where supervisory systems (SCADA, traffic management centers) aggregate status from subordinate devices before making decisions. This naturally produces hierarchical credit assignment.

---

## 11. HERON vs. Existing Frameworks

```mermaid
graph LR
    subgraph "PettingZoo / EPyMARL / MARLlib"
        direction TB
        ENV1["Environment"]
        ALG1["Algorithm"]
        ENV1 <-->|"step(actions)<br/>→ (obs, rewards)"| ALG1
        NOTE1["Standardizes the<br/>ENV ↔ ALGORITHM<br/>interface"]
    end

    subgraph "HERON"
        direction TB
        ENV2["Environment"]
        PROXY["ProxyAgent<br/>(state mediation)"]
        AGENTS["Agent Hierarchy<br/>(3 levels)"]
        SCHED["EventScheduler<br/>(dual modes)"]
        PROTO["Protocols<br/>(composable)"]
        BROKER["MessageBroker<br/>(channels)"]

        ENV2 --- PROXY
        PROXY --- AGENTS
        AGENTS --- SCHED
        AGENTS --- PROTO
        AGENTS --- BROKER

        NOTE2["Standardizes<br/>WHAT HAPPENS<br/>INSIDE the env"]
    end

    ENV1 -.->|"complementary<br/>different level"| ENV2

    style NOTE1 fill:#f8f9fa,stroke:#dee2e6
    style NOTE2 fill:#f8f9fa,stroke:#dee2e6
    style PROXY fill:#3498db,color:#fff
    style SCHED fill:#1abc9c,color:#fff
    style PROTO fill:#e67e22,color:#fff
    style BROKER fill:#9b59b6,color:#fff
```

---

## 12. Message Broker Channel Architecture

```mermaid
graph TB
    subgraph "Channel Naming"
        AC["Action Channels<br/>env_{id}__action__{sender}_to_{recipient}"]
        IC["Info Channels<br/>env_{id}__info__{sender}_to_{recipient}"]
    end

    subgraph "Example Topology"
        direction LR

        SYS["SystemAgent"] -->|"env_0__action__sys_to_coord1"| COORD1["CoordinatorAgent"]
        COORD1 -->|"env_0__action__coord1_to_field1"| F1["FieldAgent 1"]
        COORD1 -->|"env_0__action__coord1_to_field2"| F2["FieldAgent 2"]

        F1 -.->|"env_0__info__field1_to_proxy"| PX["ProxyAgent"]
        F2 -.->|"env_0__info__field2_to_proxy"| PX
        PX -.->|"env_0__info__proxy_to_field1"| F1
        PX -.->|"env_0__info__proxy_to_field2"| F2
    end

    subgraph "Message Types"
        MT1["ACTION: Parent → Child commands"]
        MT2["INFO: Observation requests/responses"]
        MT3["BROADCAST: Multi-recipient"]
        MT4["STATE_UPDATE: State sync"]
    end

    style SYS fill:#e74c3c,color:#fff
    style COORD1 fill:#f39c12,color:#fff
    style F1 fill:#27ae60,color:#fff
    style F2 fill:#27ae60,color:#fff
    style PX fill:#3498db,color:#fff
```

**Novelty:** Environment-isolated channels (`env_{id}__...`) enable parallel training with multiple environment instances without message crosstalk. The typed channel system (action vs. info) enforces communication directionality.

---

## 13. Observation Bundle Design (Event-Driven)

```mermaid
flowchart LR
    subgraph "Agent Request"
        REQ["Agent → Proxy:<br/>{get_info: 'obs',<br/>protocol: protocol}"]
    end

    subgraph "Proxy Construction"
        direction TB
        OBS_BUILD["get_observation(aid, protocol)"]
        LS_BUILD["get_local_state(aid, protocol)"]
        BUNDLE["Bundle BOTH:<br/>{obs: obs.to_dict(),<br/>local_state: local_state}"]

        OBS_BUILD --> BUNDLE
        LS_BUILD --> BUNDLE
    end

    subgraph "Agent Receipt"
        direction TB
        PARSE["Parse response"]
        OBS_R["obs = Observation.from_dict()"]
        SYNC["sync_state_from_observed(local_state)"]
        ACT["compute_action(obs, scheduler)"]

        PARSE --> OBS_R
        PARSE --> SYNC
        OBS_R --> ACT
        SYNC --> ACT
    end

    REQ --> OBS_BUILD
    REQ --> LS_BUILD
    BUNDLE --> PARSE

    style BUNDLE fill:#e74c3c,color:#fff
    style SYNC fill:#3498db,color:#fff
```

**Design principle:** "When an agent asks for an observation, the proxy gives BOTH observation AND local state." This eliminates a separate state-sync round trip, reducing communication overhead by 50%.

---

## Summary of Novel Contributions

| # | Contribution | What It Enables | Why Existing Frameworks Can't Do It |
|---|-------------|----------------|-------------------------------------|
| 1 | **Dual execution modes** (sync + event-driven) | Train with CTDE, test with realistic CPS timing | Requires fundamentally different execution loop, not achievable by wrapping |
| 2 | **ProxyAgent state mediation** | Prevents global state leak, enforces visibility | No existing framework has a centralized state gatekeeper |
| 3 | **4-level feature visibility** | Ablation over information structures | Existing frameworks offer binary (full/partial) observability |
| 4 | **Composable protocol system** | Swap coordination without changing agents | No framework separates communication from action coordination |
| 5 | **Agent-paced EventScheduler** | CPS-calibrated timing as experimental variable | All existing frameworks assume synchronous stepping |
| 6 | **Hierarchical action passing** | Upstream priority + protocol decomposition | Flat agent models can't express hierarchical control |
| 7 | **Channel-isolated MessageBroker** | Parallel training + typed communication | No framework provides environment-isolated message channels |

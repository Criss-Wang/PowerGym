# Paper Improvements for NeurIPS 2026 D&B Workshop

## Overview
This document outlines the improvements to strengthen the PowerGrid paper based on codebase analysis and the shift toward hierarchical multi-agent systems with realistic information constraints.

---

## 1. STRONGER MOTIVATION: Why Existing MARL Simulations Fail

### Problem with Current MARL Platforms:

**Most MARL platforms assume unrealistic execution models:**

1. **Synchronous, Centralized Execution**:
   - Agents act simultaneously with full network observability
   - No modeling of communication delays or information propagation
   - Example: MPE, SMAC, PettingZoo environments assume global state broadcasting

2. **Binary Observability** (All or Nothing):
   - Either full observability (centralized training) or local-only (no coordination info)
   - No systematic framework for fine-grained partial observability
   - PowerGridworld: agents see either full network or just local device state

3. **Homogeneous Agents**:
   - All agents have identical capabilities and information access
   - Doesn't model hierarchical organizational structures (field devices → substations → control center)
   - No role-based information privileges

4. **No Communication Infrastructure Modeling**:
   - Message passing is instantaneous and perfect
   - No bandwidth constraints, packet loss, or latency
   - Communication patterns not grounded in real systems (e.g., SCADA hierarchies)

### Why This Matters for Sim-to-Real Transfer:

**Key Failure Modes When Deploying MARL Policies:**
- **Observability Mismatch**: Policies trained with full observability fail when deployed with realistic SCADA constraints (Section 5.3 shows 23% degradation)
- **Synchronization Assumptions**: Centralized training assumes simultaneous agent actions, but real systems have asynchronous communication
- **Information Dependencies**: Policies learn to depend on unavailable information (e.g., internal states of competing microgrids)

### PowerGrid's Solution:

1. **Hierarchical Agent Architecture**: 3-level system mirroring real SCADA/EMS structures
2. **Composable Partial Observability**: 16 FeatureProviders with 4-level visibility hierarchy
3. **Dual-Mode Execution**: Centralized (development) + Distributed (validation) to expose sim-to-real gaps
4. **Async/Decentralized Observe-Plan-Act**: Message-based coordination with recursive subordinate execution

---

## 2. HIERARCHICAL DESIGN - EXPANDED

### Current Paper Coverage:
§4.2 describes the 3-level hierarchy but lacks detail on **WHY** this design and **HOW** it enables scalability.

### Proposed Enhancements:

#### A. Add Architectural Diagram
```
┌─────────────────────────────────────────────────┐
│  Level 3: ProxyAgent (System Overseer)          │
│  - Full system-level observability              │
│  - Distributes network state via MessageBroker   │
│  - Enforces visibility rules                     │
└──────────────┬──────────────────────────────────┘
               │ (info channel)
      ┌────────┴────────┬──────────────────┐
      ▼                 ▼                   ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Level 2:    │  │ Level 2:    │  │ Level 2:    │
│ GridAgent   │  │ GridAgent   │  │ GridAgent   │
│ (MG1)       │  │ (MG2)       │  │ (MG3)       │
│ - upper_level│  │ - upper_level│  │ - upper_level│
│ - owner     │  │ - owner     │  │ - owner     │
│ - public    │  │ - public    │  │ - public    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                 │                 │
   ┌───┴───┬───┐     ┌──┴──┬───┐      ┌──┴──┬───┐
   ▼       ▼   ▼     ▼     ▼   ▼      ▼     ▼   ▼
  Dev1   Dev2 Dev3  Dev4  Dev5 Dev6  Dev7  Dev8 Dev9
  (ESS)  (Gen)(Grid) (ESS) (Gen)(Grid)(ESS)(Gen)(Grid)
  Level 1: DeviceAgent - owner + public only
```

#### B. Role-Based Observability Mapping

| Agent Level | Observability | Real-World Analog | Information Access |
|-------------|---------------|-------------------|-------------------|
| **DeviceAgent (L1)** | `owner` + `public` | Field device (RTU) | Own P/Q/SOC + system frequency |
| **GridAgent (L2)** | `upper_level` + `owner` + `public` | Substation controller | Subordinate devices + boundary buses + system-wide alerts |
| **ProxyAgent (L3)** | `system` (full) | EMS control center | Full network state for aggregation/filtering |

#### C. Scalability Through Hierarchy

**Problem**: Flat MARL with 60 devices → joint action space of dimension 120+ (infeasible for CTDE)

**Solution**: Hierarchical decomposition
- 60 DeviceAgents → 10 GridAgents (6 devices each)
- Joint action space: 10 GridAgents × 12 dims = 120 dims (tractable)
- Subordinate actions decomposed locally by each GridAgent

**Experimental Evidence** (Table 5):
- Flat MARL (60 device agents): Fails to converge
- Hierarchical (10 grid agents): 4.2× training speedup, convergence in 10K episodes

---

## 3. ASYNC/DECENTRALIZED OBSERVE-PLAN-ACT PATTERN - NEW SECTION

### Proposed New Section: §4.3 Asynchronous Distributed Execution

#### Motivation:
Real distributed control systems operate **asynchronously**:
- SCADA messages have non-zero latency (10-100ms)
- Agents act at different rates (generators: seconds, markets: hours)
- Power flow computation occurs centrally, results propagate hierarchically

#### The `step_distributed()` Method:

PowerGrid implements a **recursive, message-based execution pattern** that captures these realities:

**Algorithm 1: Asynchronous Hierarchical Step**
```
Input: Agent A with upstream U, subordinates S = {S1, ..., Sn}

1. RECEIVE from upstream:
   - action_msg ← MessageBroker.consume(channel: U → A)
   - info_msg ← MessageBroker.consume(channel: U → A)
   - A.update_state(info_msg)

2. PLAN for subordinates:
   - downstream_actions ← A.derive_downstream_actions(action_msg)

3. SEND to subordinates:
   - For each Si ∈ S:
       MessageBroker.publish(channel: A → Si, downstream_actions[Si])

4. EXECUTE subordinates (RECURSIVE, ASYNC):
   - await asyncio.gather(*[Si.step_distributed() for Si in S])
      ↳ Each subordinate runs steps 1-9 concurrently

5. COLLECT from subordinates:
   - For each Si ∈ S:
       info_i ← MessageBroker.consume(channel: Si → A)
       A.subordinates_info[Si] ← info_i

6. ACT locally:
   - local_action ← A.derive_local_action(action_msg)
   - A.execute_local_action(local_action)
   - A.update_state_post_step()

7. PUBLISH to environment:
   - MessageBroker.publish(channel: A → ENV, state_updates)

8. RECEIVE network state (only in distributed mode):
   - network_state ← MessageBroker.consume(channel: ProxyAgent → A)
   - A.update_from_network_state(network_state)

9. SEND info to upstream:
   - compiled_info ← A.compile_info(own_info, subordinates_info)
   - MessageBroker.publish(channel: A → U, compiled_info)
```

#### Key Features:

1. **Asynchronous Subordinate Execution**:
   - Line 4: `await asyncio.gather()` runs all subordinates concurrently
   - Subordinates execute in parallel, not sequentially
   - Mimics real distributed systems where agents act independently

2. **Recursive Decomposition**:
   - Each agent plans for itself and its subordinates
   - Action decomposition happens at each level (not centrally)
   - GridAgent splits joint action → per-device actions

3. **Message-Based Communication**:
   - No direct network access in distributed mode
   - All information flows through MessageBroker
   - ProxyAgent enforces visibility rules (filtering)

4. **Information Propagation**:
   - Network state flows: Environment → ProxyAgent → GridAgents → DeviceAgents
   - Actions flow: RL Policy → GridAgents → DeviceAgents
   - Info flows back: DeviceAgents → GridAgents → ProxyAgent → Environment

#### Comparison with Centralized Mode:

| Aspect | Centralized Mode | Distributed Mode |
|--------|------------------|------------------|
| **Network Access** | Direct (PandaPower net) | Via ProxyAgent messages |
| **Execution** | Synchronous observe() → act() | Async step_distributed() |
| **Observability** | Full network state | Filtered by visibility rules |
| **Communication** | Implicit (shared memory) | Explicit (MessageBroker) |
| **Realism** | Development/debugging | Validation/deployment |

---

## 4. COMMUNICATION PATTERNS - NEW SECTION

### Proposed New Section: §4.4 Message-Based Communication Infrastructure

#### A. MessageBroker Architecture

**Design Philosophy**: Decouple agents from environment and each other

**Key Components**:
1. **MessageBroker**: Pub-sub system managing all agent communication
2. **Channels**: Named pathways for message routing (e.g., `grid1_to_device3_action`)
3. **ChannelManager**: Utility for consistent channel naming

**Message Types**:
- `ACTION`: Coordination actions (parent → subordinate)
- `INFO`: State updates (subordinate → parent, ProxyAgent → agent)
- `POWER_FLOW_RESULT`: Network state (Environment → ProxyAgent)

#### B. Protocol System: Separating Communication from Action

**Key Insight**: Coordination has two orthogonal dimensions:
1. **CommunicationProtocol**: WHAT to communicate (price signals, setpoints, consensus values)
2. **ActionProtocol**: HOW to coordinate actions (centralized control vs. decentralized response)

**Example: SetpointProtocol**
```python
class SetpointProtocol(VerticalProtocol):
    communication_protocol = SetpointCommunicationProtocol()  # Send setpoint assignments
    action_protocol = CentralizedActionProtocol()              # Direct action control
```

**Example: PriceSignalProtocol**
```python
class PriceSignalProtocol(VerticalProtocol):
    communication_protocol = PriceCommunicationProtocol()    # Broadcast prices
    action_protocol = DecentralizedActionProtocol()          # Devices respond independently
```

#### C. Vertical vs. Horizontal Protocols

**Vertical Protocols** (Agent-owned, Parent → Subordinate):
- **SetpointProtocol**: Direct control via power setpoints
- **PriceSignalProtocol**: Indirect control via marginal prices
- Each GridAgent owns its protocol to coordinate subordinates

**Horizontal Protocols** (Environment-owned, Peer ↔ Peer):
- **P2PTradingProtocol**: Market-based energy trading between microgrids
- **ConsensusProtocol**: Distributed voltage/frequency regulation via gossip
- Environment runs protocol as it requires global view

#### D. Communication Flow Example

**Scenario**: GridAgent MG1 coordinates 3 devices using PriceSignalProtocol

1. **Coordinator computes price**:
   ```python
   price = MG1.policy.forward(observation)  # Policy outputs price signal
   ```

2. **Communication protocol broadcasts**:
   ```python
   messages = PriceCommunicationProtocol.compute_coordination_messages(
       sender_state=MG1.state,
       receiver_states={dev1.id: dev1.state, dev2.id: dev2.state, dev3.id: dev3.state},
       context={"coordinator_action": price}
   )
   # Returns: {dev1.id: {"type": "price_signal", "price": 52.3}, ...}
   ```

3. **MessageBroker delivers**:
   ```python
   for device_id, msg in messages.items():
       channel = ChannelManager.info_channel(MG1.id, device_id, env_id)
       MessageBroker.publish(channel, msg)
   ```

4. **Devices respond independently**:
   ```python
   # Each device receives price and acts autonomously
   action = device.policy.forward(observation, price=msg["price"])
   ```

---

## 5. DOMAIN-SPECIFIC ENVIRONMENTS CONTRIBUTION - EVALUATION

### Strength Assessment: **STRONG** ✓

#### Why This is a Valuable Contribution:

**Problem**: MARL research is dominated by **toy/game environments** (MPE, SMAC, Atari) that don't reflect real-world deployment challenges:
- Simplified dynamics (discrete state/action, deterministic transitions)
- Unrealistic information structures (global observability or uniform restrictions)
- No physical constraints (power flow equations, safety limits)

**PowerGrid's Solution**: **Domain-Specific Environment Framework** that bridges this gap

#### Key Features Enabling Cross-Domain Transfer:

1. **Composable FeatureProviders**:
   - Domain-agnostic abstraction for state representation
   - Visibility rules generalize to any information hierarchy
   - Example: `BusVoltages` (power) → `IntersectionOccupancy` (traffic) → `RobotPose` (robotics)

2. **Hierarchical Agent Architecture**:
   - Generalizes beyond power systems
   - Any domain with organizational hierarchies can use this pattern
   - Example: Traffic lights → Intersection controllers → City-wide coordinator

3. **Protocol System**:
   - Communication + Action separation is domain-independent
   - Vertical protocols: Parent coordinates subordinates (universal pattern)
   - Horizontal protocols: Peer coordination (markets, consensus, trading)

4. **Dual-Mode Execution**:
   - Centralized development + Distributed validation
   - Exposes sim-to-real gaps in ANY domain with information constraints
   - Not power-specific!

#### Justification for Including This Contribution:

**YES** - This is a strong, generalizable contribution:
- Provides **benchmark-ready infrastructure** for domain-specific MARL research
- Enables **systematic study** of information requirements across domains
- Addresses **sim-to-real gap** which is a fundamental MARL challenge
- **Under-explored** in existing MARL benchmarks (most are game/toy environments)

### Proposed Addition to Contributions (§1.3):

**New Contribution #5: Domain-Specific Environment Framework**
- Composable abstractions (FeatureProviders, Protocols, Hierarchies) that generalize beyond power systems
- Enables MARL researchers to create domain-specific benchmarks grounded in real-world constraints (physics, safety, information hierarchies)
- Addresses the "toy environment problem" in MARL research where algorithms work in games but fail in practical deployments

---

## 6. CASE STUDIES - NEW SECTION §6

### Proposed New Section: §6 Cross-Domain Case Studies

**Purpose**: Demonstrate that PowerGrid's architectural innovations (hierarchical agents, composable observability, protocol system) generalize beyond power systems.

#### Case Study 1: **Power System** (Current Domain)

**Application**: Multi-microgrid voltage regulation and economic dispatch

**Hierarchy**:
- L1: DeviceAgents (generators, ESS, transformers)
- L2: GridAgents (microgrid controllers)
- L3: ProxyAgent (DSO control center)

**Observability Challenge**: Competing microgrids must coordinate (voltage, frequency) without revealing commercial information (SOC, internal costs)

**Visibility Rules**:
- `owner`: SOC, generation costs, internal constraints
- `upper_level`: Aggregate power output, boundary bus voltages
- `system`: Full network state (DSO only)

**Protocol**: PriceSignalProtocol (GridAgent broadcasts marginal prices → DeviceAgents respond independently)

**Key Findings**:
- Upper-level visibility achieves 95% of full observability performance (Table 3)
- Privacy-preserving coordination possible with only 3.8% performance penalty (Table 4)

---

#### Case Study 2: **Transportation** (Traffic Signal Coordination)

**Application**: Adaptive traffic signal control for urban networks

**Hierarchy**:
- L1: DeviceAgents = Individual traffic lights (phase control)
- L2: GridAgents = Intersection controllers (coordinate 4-8 lights)
- L3: ProxyAgent = City-wide traffic management center

**Observability Challenge**: Intersection controllers need coordination signals (green waves) but shouldn't see fine-grained vehicle positions across the city (privacy)

**Visibility Rules**:
- `owner`: Lane occupancy, queue lengths, pedestrian crossings (local sensors)
- `upper_level`: Aggregate flow rates at boundary roads, upstream intersection states
- `system`: Full city-wide traffic state (TMC only)
- `public`: Emergency vehicle alerts, road closures

**Protocol**: ConsensusProtocol (intersections use gossip algorithm to coordinate green wave timing)

**Adaptation from PowerGrid**:
- `BusVoltages` → `IntersectionOccupancy` FeatureProvider
- `LineFlows` → `RoadFlowRates` FeatureProvider
- AC power flow → Traffic flow model (CTM/LWR)

**Expected Benefits**:
- Hierarchical coordination: Reduce action space from 200 lights → 25 intersection controllers
- Partial observability: Test if green wave coordination requires full city visibility or just neighboring intersections
- Sim-to-real: Centralized training (full visibility) vs. distributed deployment (neighbor-only visibility)

---

#### Case Study 3: **Robotics** (Multi-Robot Warehouse Coordination)

**Application**: Coordinating 50+ mobile robots in Amazon fulfillment center

**Hierarchy**:
- L1: DeviceAgents = Individual robots (path planning, collision avoidance)
- L2: GridAgents = Zone coordinators (manage 5-10 robots in warehouse zone)
- L3: ProxyAgent = Central warehouse management system

**Observability Challenge**: Robots need coordination (avoid collisions, load balance) but communication is bandwidth-limited (can't broadcast full robot poses to all 50 robots)

**Visibility Rules**:
- `owner`: Own pose, velocity, battery, current task
- `upper_level`: Coarse positions of robots in same zone (zone coordinator has this)
- `system`: Full warehouse state (central WMS only)
- `public`: Emergency stops, zone congestion levels

**Protocol**: SetpointProtocol (zone coordinator assigns waypoints → robots execute local path planning)

**Adaptation from PowerGrid**:
- `ElectricalBasePh` (P, Q) → `RobotKinematics` (x, y, θ, v) FeatureProvider
- `StorageBlock` (SOC) → `BatteryState` FeatureProvider
- AC power flow → Multi-robot motion planning

**Expected Benefits**:
- Hierarchical scalability: 50 flat agents → 10 zone coordinators
- Communication efficiency: Robots only receive zone-level coordination signals, not full warehouse state
- Partial observability: Test if collision avoidance requires full robot visibility or just local neighborhood

---

#### Case Study 4: **Game** (StarCraft II Micromanagement)

**Application**: Control 20-30 combat units in StarCraft II battles

**Hierarchy**:
- L1: DeviceAgents = Individual units (Marines, Medivacs, Siege Tanks)
- L2: GridAgents = Squad commanders (manage 5-10 units per squad)
- L3: ProxyAgent = Overall commander (macro strategy)

**Observability Challenge**: Units have fog of war (can't see enemy units outside vision range), must coordinate based on partial information

**Visibility Rules**:
- `owner`: Own HP, position, attack cooldown, energy
- `upper_level`: Aggregate squad HP, squad centroid position, squad formation
- `system`: Full game state (only for evaluation, not available to agents)
- `public`: Enemy units in shared vision, team-wide alerts

**Protocol**: PriceSignalProtocol (squad commander broadcasts "threat level" → units independently decide to engage or retreat)

**Adaptation from PowerGrid**:
- `ElectricalBasePh` → `UnitAttributes` (HP, DPS, position) FeatureProvider
- `CostBlock` → `CombatCost` (expected damage taken) FeatureProvider
- Safety violations → Unit deaths penalty

**Expected Benefits**:
- Hierarchical decomposition: 30 flat agents → 5 squad commanders
- Partial observability: Test if combat micro requires full vision or just squad-level aggregates
- Compare with existing SMAC benchmarks (flat MARL with homogeneous agents)

---

## 7. SUMMARY OF CHANGES TO PAPER STRUCTURE

### Additions:

**§1.2 Why Existing MARL Simulations Fail** (NEW, ~400 words)
- Synchronous execution assumptions
- Binary observability (all or nothing)
- Homogeneous agents
- No communication infrastructure modeling
- Sim-to-real gap examples

**§1.3 Contributions** (UPDATED)
- Add #5: Domain-Specific Environment Framework

**§4.3 Asynchronous Distributed Execution** (NEW, ~800 words)
- Motivation (real systems are async)
- Algorithm 1: step_distributed() pseudocode
- Key features (async subordinates, recursive decomposition, message-based)
- Comparison table: Centralized vs. Distributed

**§4.4 Message-Based Communication Infrastructure** (NEW, ~600 words)
- MessageBroker architecture
- Protocol system (Communication + Action separation)
- Vertical vs. Horizontal protocols
- Communication flow example

**§6 Cross-Domain Case Studies** (NEW, ~1200 words)
- Case Study 1: Power System (existing)
- Case Study 2: Transportation (traffic signals)
- Case Study 3: Robotics (warehouse coordination)
- Case Study 4: Game (StarCraft II micro)

### Enhancements:

**§4.2 Hierarchical Agent Framework** (EXPANDED)
- Add architectural diagram
- Add role-based observability mapping table
- Expand scalability section with experimental evidence

**§5.3 Centralized→Distributed Gap** (ENHANCED)
- Reframe as "sim-to-real gap from observability mismatch"
- Add failure mode analysis (what breaks when deploying centralized policies)

---

## 8. ESTIMATED WORD COUNT CHANGES

| Section | Current | Proposed | Change |
|---------|---------|----------|--------|
| §1.2 Motivation (NEW) | 0 | ~400 | +400 |
| §1.3 Contributions | ~150 | ~200 | +50 |
| §4.2 Hierarchy | ~400 | ~700 | +300 |
| §4.3 Async Pattern (NEW) | 0 | ~800 | +800 |
| §4.4 Communication (NEW) | 0 | ~600 | +600 |
| §6 Case Studies (NEW) | 0 | ~1200 | +1200 |
| **Total** | ~8000 | ~11150 | **+3150** |

**Page Limit Consideration**:
- NeurIPS D&B workshop typically allows 8-10 pages + unlimited appendix
- Main paper can stay ~9 pages, move some details to appendix
- Case studies 2-4 can be condensed (200 words each instead of 400)

---

## 9. NEXT STEPS

1. ✅ Complete this draft document
2. ⬜ Implement changes in main_v4.tex
3. ⬜ Create architectural diagrams (TikZ)
4. ⬜ Add Algorithm 1 (step_distributed pseudocode)
5. ⬜ Add comparison tables
6. ⬜ Proofread and compile LaTeX
7. ⬜ Run experiments for case study validation (if time permits)

---

## 10. KEY MESSAGES TO EMPHASIZE

1. **PowerGrid addresses the sim-to-real gap** in MARL by exposing observability mismatches before deployment

2. **Hierarchical agents + async execution + message-based communication** = realistic distributed control modeling

3. **Domain-agnostic framework** enables moving beyond toy/game environments to practical applications

4. **Systematic partial observability control** (16 FeatureProviders, 4 visibility levels) is novel in MARL benchmarks

5. **Protocol system** (Communication + Action separation) is a clean abstraction for coordination strategies


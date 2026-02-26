# HMARL-CBF Event-Driven Execution Diagrams

Scenario: Two drones (Worker A, Worker B) cooperatively lift a beam in a warehouse, coordinated by a Manager. Under the HMARL-CBF setup, five distinct types of information mismatch can arise. Each is illustrated below.

---

## 1. Collision-Induced Mismatch

An accident disrupts Worker A's state, but the Manager's next observation still reflects the pre-collision world, leading to instructions that assume a reality that no longer exists.

### Flow Diagram

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: Manager Issues Task"]
        A([Payload Order]) --> B[Manager observes global state S₀]
        B --> C["Compute waypoints W_A, W_B"]
        C --> D[Send waypoints to Workers]
        D --> E([Manager sleeps])
    end

    subgraph Phase2["Phase 2: Workers Execute — Collision Occurs"]
        F[Workers act at 100 Hz] --> G[Worker A collides with obstacle]
        G --> H["Worker A damaged / displaced<br/>actual state diverges from plan"]
        H --> I{Manager aware<br/>of collision?}
    end

    subgraph Phase3["Phase 3: Information Mismatch"]
        I -- No --> J["Manager wakes & observes<br/><b>PRE-COLLISION state</b><br/>(sensor lag / stale snapshot)"]
        J --> K["Issues W_A' assuming<br/>Worker A is intact & on-course"]
        K --> L["Worker A cannot execute W_A'<br/>(damaged / wrong position)"]
        L --> M["Worker B follows W_B'<br/>expecting Worker A at W_A'"]
        M --> N{Coordination<br/>breaks down?}
        N -- Yes --> O["Beam pickup fails /<br/>secondary collision risk"]
    end

    subgraph Phase4["Phase 4: Resolution"]
        I -- Yes: immediate detection --> P[Manager replans with true state]
        O --> Q["Worker A reports collision<br/>& actual state"]
        Q --> R[Manager receives ground truth]
        R --> P
        P --> S([Corrected plan issued])
    end

    E --> F
    S -.->|"Next cycle"| B

    style Phase1 fill:#e8f4fd,stroke:#2196F3
    style Phase2 fill:#fce4ec,stroke:#f44336
    style Phase3 fill:#fff3e0,stroke:#FF9800
    style Phase4 fill:#f3e5f5,stroke:#9C27B0
```

### Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Mgr as Manager<br/>(Low-Freq)
    participant WA as Worker A<br/>(100 Hz)
    participant WB as Worker B<br/>(100 Hz)

    note over Mgr,WB: Phase 1 — Manager observes & tasks

    activate Mgr
    Mgr->>Mgr: Observe global state S₀ (both drones healthy)
    Mgr->>WA: Waypoint W_A (approach beam from left)
    Mgr->>WB: Waypoint W_B (approach beam from right)
    deactivate Mgr
    note over Mgr: Sleeps until next macro-step

    note over Mgr,WB: Phase 2 — Workers execute / collision strikes Worker A

    activate WA
    activate WB
    WA->>WA: Act toward W_A
    WB->>WB: Act toward W_B
    note over WA: Collision with unseen obstacle!<br/>Worker A displaced to position P_A'<br/>and partially damaged

    note over Mgr,WB: Phase 3 — Manager wakes, unaware of collision

    activate Mgr
    Mgr->>Mgr: Observe global state S₁
    note over Mgr: S₁ reflects PRE-COLLISION state<br/>(sensor lag / stale global snapshot)<br/>Manager thinks Worker A is on-course
    Mgr->>Mgr: Plan next step assuming Worker A near W_A
    Mgr->>WA: New waypoint W_A' (grip left side of beam)
    Mgr->>WB: New waypoint W_B' (grip right side of beam)
    deactivate Mgr

    note over WA: Cannot reach W_A':<br/>wrong position + damaged actuator
    WA->>WA: Attempt W_A' — fails
    note over WB: Arrives at W_B', waits for<br/>Worker A at W_A'...<br/>Worker A never shows up
    WB->>WB: Arrives at W_B', begins grip sequence

    note over Mgr,WB: Coordination failure: Worker B acts alone on beam

    note over Mgr,WB: (Potentially) Phase 4 — Resolution via bottom-up feedback

    WA->>Mgr: Report: collision at P_A' + damage status
    deactivate WA
    WB->>Mgr: Report: timeout waiting for Worker A
    deactivate WB

    activate Mgr
    note over Mgr: Mismatch resolved —<br/>now sees true post-collision state
    Mgr->>Mgr: Replan: reroute Worker A or reassign task
    Mgr->>WA: Recovery waypoint W_A'' (or stand-down)
    Mgr->>WB: Updated waypoint W_B'' (hold position)
    deactivate Mgr

    activate WA
    activate WB
    WA->>WA: Execute recovery
    WB->>WB: Hold and wait
    deactivate WA
    deactivate WB

    note over Mgr,WB: Cycle repeats with corrected world model
```

---

## 2. Timing-Induced Mismatch

The Manager observes and decides at a slower cadence than the Workers act. Worker B's action is still in-flight when the Manager reads state, so it plans on a snapshot that is already outdated.

### Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Mgr as Manager<br/>(Low-Freq)
    participant WA as Worker A<br/>(100 Hz)
    participant WB as Worker B<br/>(100 Hz)

    note over Mgr,WB: Phase 1 — Manager observes & tasks

    activate Mgr
    Mgr->>Mgr: Observe global state S₀
    Mgr->>WA: Waypoint W_A
    Mgr->>WB: Waypoint W_B
    deactivate Mgr
    note over Mgr: Sleeps until next macro-step

    note over Mgr,WB: Phase 2 — Workers execute (Manager asleep)

    activate WA
    activate WB
    WA->>WA: Act toward W_A (action a_A)
    WB->>WB: Act toward W_B (action a_B)
    note over WB: a_B in progress...<br/>effect not yet visible<br/>in global state

    note over Mgr,WB: Phase 3 — Timing mismatch: Manager wakes too early

    activate Mgr
    Mgr->>Mgr: Observe global state S₁
    note over Mgr: S₁ is STALE — does not<br/>reflect Worker B's in-flight a_B
    Mgr->>Mgr: Plan based on stale S₁
    Mgr->>WA: New waypoint W_A'
    Mgr->>WB: New waypoint W_B' (conflicts with a_B!)
    deactivate Mgr

    note over WB: Conflict: still executing a_B<br/>but received contradicting W_B'

    note over Mgr,WB: (Potentially) Phase 4 — Resolution via bottom-up feedback

    WB->>WB: a_B completes — state actually moved to S₂
    note over WB: S₂ ≠ what Manager assumed
    WB->>Mgr: Report actual state S₂
    deactivate WB
    deactivate WA

    activate Mgr
    Mgr->>Mgr: Observe true state S₂ (mismatch resolved)
    Mgr->>WA: Corrected waypoint W_A''
    Mgr->>WB: Corrected waypoint W_B''
    deactivate Mgr

    activate WA
    activate WB
    WA->>WA: Execute toward W_A''
    WB->>WB: Execute toward W_B''
    deactivate WA
    deactivate WB

    note over Mgr,WB: Cycle repeats
```

---

## 3. Partial Observability Mismatch

Each Worker only sees a local slice of the world via LiDAR (5m radius). Worker A detects an obstacle that Worker B cannot see. They make conflicting local decisions based on different views of the same environment.

### Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Mgr as Manager<br/>(Low-Freq)
    participant WA as Worker A<br/>(100 Hz)
    participant WB as Worker B<br/>(100 Hz)

    note over Mgr,WB: Phase 1 — Manager observes & tasks

    activate Mgr
    Mgr->>Mgr: Observe global state S₀
    Mgr->>WA: Waypoint W_A (approach from aisle 3)
    Mgr->>WB: Waypoint W_B (approach from aisle 4)
    deactivate Mgr
    note over Mgr: Sleeps until next macro-step

    note over Mgr,WB: Phase 2 — Workers execute with different local views

    activate WA
    activate WB
    note over WA: LiDAR sees obstacle<br/>blocking shared corridor
    note over WB: LiDAR clear —<br/>obstacle outside 5m range

    WA->>WA: Reroute: detour through aisle 5
    WB->>WB: Proceed straight toward W_B

    note over WA,WB: Both converge on aisle 5<br/>without knowing it

    note over Mgr,WB: Phase 3 — Peer conflict from asymmetric information

    note over WA: Worker A enters aisle 5 from north
    note over WB: Worker B enters aisle 5 from south
    note over WA,WB: Near-miss / deadlock in aisle 5!<br/>Neither expected the other here

    note over Mgr,WB: (Potentially) Phase 4 — Resolution via peer + bottom-up feedback

    WA->>WB: Peer-to-peer: share local obstacle map
    WB->>WA: Peer-to-peer: share local obstacle map
    note over WA,WB: Merged local views reveal<br/>full picture neither had alone

    WA->>Mgr: Report: obstacle in aisle 3 + deadlock in aisle 5
    deactivate WA
    deactivate WB

    activate Mgr
    Mgr->>Mgr: Update global map with merged local data
    Mgr->>WA: New waypoint W_A' (use aisle 6)
    Mgr->>WB: New waypoint W_B' (stay in aisle 5)
    deactivate Mgr

    activate WA
    activate WB
    WA->>WA: Execute toward W_A'
    WB->>WB: Execute toward W_B'
    deactivate WA
    deactivate WB

    note over Mgr,WB: Cycle repeats with enriched global map
```

---

## 4. Communication Failure Mismatch

A waypoint message from the Manager to Worker B is dropped. Worker B continues executing stale instructions while Worker A and the Manager have moved on to a new plan.

### Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Mgr as Manager<br/>(Low-Freq)
    participant WA as Worker A<br/>(100 Hz)
    participant WB as Worker B<br/>(100 Hz)

    note over Mgr,WB: Phase 1 — Manager observes & tasks

    activate Mgr
    Mgr->>Mgr: Observe global state S₀
    Mgr->>WA: Waypoint W_A
    Mgr->>WB: Waypoint W_B
    deactivate Mgr
    note over Mgr: Sleeps until next macro-step

    note over Mgr,WB: Phase 2 — Normal execution

    activate WA
    activate WB
    WA->>WA: Act toward W_A
    WB->>WB: Act toward W_B
    deactivate WA
    deactivate WB

    note over Mgr,WB: Phase 3 — Manager replans but message to Worker B is dropped

    activate Mgr
    Mgr->>Mgr: Observe state S₁ (new info requires replan)
    Mgr->>WA: New waypoint W_A' (received)
    Mgr--xWB: New waypoint W_B' (DROPPED!)
    deactivate Mgr

    activate WA
    activate WB
    WA->>WA: Execute toward W_A' (new plan)
    note over WB: Never received W_B'<br/>Still following old W_B
    WB->>WB: Continue toward W_B (stale plan)

    note over WA,WB: Workers now on INCOMPATIBLE plans<br/>Worker A expects B at W_B'<br/>Worker B heading to old W_B

    note over Mgr,WB: Phase 4 — Detection and resolution

    activate Mgr
    Mgr->>Mgr: Observe state S₂
    note over Mgr: Worker B not at expected W_B'<br/>Detects plan divergence
    Mgr->>WB: Resend waypoint W_B''
    deactivate Mgr

    WB->>WB: Receives corrected waypoint
    note over WB: Aborts old W_B<br/>Redirects to W_B''

    WA->>WA: Continue toward W_A'
    WB->>WB: Execute toward W_B''
    deactivate WA
    deactivate WB

    note over Mgr,WB: Plans re-aligned / cycle repeats
```

---

## 5. Goal Conflict / Reward Mismatch

The Manager optimizes for task speed (deliver beam ASAP). The Workers optimize for flight safety (minimize energy and collision risk). Their objectives diverge, producing actions that are locally rational but globally incompatible.

### Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Mgr as Manager<br/>(Low-Freq)
    participant WA as Worker A<br/>(100 Hz)
    participant WB as Worker B<br/>(100 Hz)

    note over Mgr,WB: Phase 1 — Manager optimizes for SPEED

    activate Mgr
    Mgr->>Mgr: Observe global state S₀
    Mgr->>Mgr: Objective: minimize delivery time
    Mgr->>WA: Waypoint W_A (aggressive short-cut through narrow gap)
    Mgr->>WB: Waypoint W_B (aggressive short-cut through narrow gap)
    deactivate Mgr

    note over Mgr,WB: Phase 2 — Workers optimize for SAFETY

    activate WA
    activate WB
    note over WA: Objective: minimize collision risk<br/>Narrow gap too risky!
    WA->>WA: Ignore short-cut / take safe wide route
    note over WB: Objective: minimize energy use<br/>Short-cut requires high thrust
    WB->>WB: Take slower energy-efficient path

    note over WA,WB: Neither Worker follows<br/>the Manager's aggressive plan

    note over Mgr,WB: Phase 3 — Manager expects fast arrival but Workers are slow

    activate Mgr
    Mgr->>Mgr: Observe state S₁
    note over Mgr: Both Workers behind schedule<br/>Manager assumes hardware issue
    Mgr->>WA: Even more aggressive waypoint W_A'
    Mgr->>WB: Even more aggressive waypoint W_B'
    deactivate Mgr

    note over WA,WB: Escalation loop:<br/>Manager pushes harder<br/>Workers resist harder

    note over Mgr,WB: (Potentially) Phase 4 — Resolution via reward alignment

    WA->>Mgr: Report: safe route chosen (gap too narrow)
    WB->>Mgr: Report: energy-efficient route chosen
    deactivate WA
    deactivate WB

    activate Mgr
    note over Mgr: Mismatch identified —<br/>speed objective conflicts<br/>with worker safety/energy objectives
    Mgr->>Mgr: Replan with safety-aware objective
    Mgr->>WA: Waypoint W_A'' (wide route, moderate speed)
    Mgr->>WB: Waypoint W_B'' (wide route, moderate speed)
    deactivate Mgr

    activate WA
    activate WB
    WA->>WA: Execute toward W_A'' (acceptable risk)
    WB->>WB: Execute toward W_B'' (acceptable energy)
    deactivate WA
    deactivate WB

    note over Mgr,WB: Objectives aligned / cycle repeats
```

# Hierarchical Agent Design with Kafka Communication

## 1. Agent Step Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AGENT STEP FLOW                              │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │   Kafka Broker       │
                    │   (Action Topic)     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  1. Receive Action   │
                    │     from Parent      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  2. Compile Messages │
                    │     from Kafka       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  3. Identify Parent  │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
     ┌──────────▼──────────┐       ┌─────────▼────────┐
     │  Has Subordinates?  │       │  No Subordinates │
     └──────────┬──────────┘       └─────────┬────────┘
                │ YES                         │
                │                             │
     ┌──────────▼──────────────┐              │
     │  4. Derive Sub-Actions  │              │
     │    Option 1: Model      │              │
     │    Option 2: Manual     │              │
     └──────────┬──────────────┘              │
                │                             │
     ┌──────────▼──────────────┐              │
     │  5. Send A₁...Aₙ to     │              │
     │     Subordinates        │              │
     │     (via Kafka)         │              │
     └──────────┬──────────────┘              │
                │                             │
     ┌──────────▼──────────────┐              │
     │  6. Run sub₁...subₙ     │              │
     │     step() [RECURSIVE]  │              │
     │     (async/sync)        │              │
     └──────────┬──────────────┘              │
                │                             │
     ┌──────────▼──────────────┐              │
     │  7. Collect from Kafka: │              │
     │     • Observations      │              │
     │     • Rewards           │              │
     │     • Status            │              │
     │     • Compiled Info     │              │
     └──────────┬──────────────┘              │
                │                             │
                └──────────────┬──────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  8. Derive Own       │
                    │     Action (aᵢ)      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  9. Execute Own      │
                    │     Action           │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ 10. Collate Partial  │
                    │     Observations &   │
                    │     Update State     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ 11. Build Compiled   │
                    │     Info (own +      │
                    │     subordinates)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ 12. Send to Parent   │
                    │     (via Kafka)      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Kafka Broker       │
                    │   (Info Topic)       │
                    └──────────────────────┘

     [Note: At any point, agent can send info to any other agent via Kafka]
```

## 2. Hierarchical Agent Communication Diagram

```
                        ┌─────────────────┐
                        │  Parent Agent   │
                        │      (P)        │
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    │     Kafka Broker        │
                    │   ┌──────────────────┐  │
                    │   │ Action Topics    │  │
                    │   │ • P→A₁, P→A₂     │  │
                    │   │ Info Topics      │  │
                    │   │ • A₁→P, A₂→P     │  │
                    │   └──────────────────┘  │
                    └────────────┬────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
        ┌───────▼────────┐              ┌────────▼────────┐
        │   Agent A₁     │              │   Agent A₂      │
        │                │              │                 │
        └───────┬────────┘              └────────┬────────┘
                │                                │
    ┌───────────┼──────────┐        ┌───────────┼──────────┐
    │     Kafka Broker     │        │     Kafka Broker     │
    │   ┌──────────────┐   │        │   ┌──────────────┐   │
    │   │ A₁→A₁₁, A₁₂ │   │        │   │ A₂→A₂₁, A₂₂ │   │
    │   │ A₁₁→A₁, A₁₂ │   │        │   │ A₂₁→A₂, A₂₂ │   │
    │   └──────────────┘   │        │   └──────────────┘   │
    └───────────┬──────────┘        └───────────┬──────────┘
                │                                │
        ┌───────┴───────┐              ┌────────┴────────┐
        │               │              │                 │
    ┌───▼────┐   ┌─────▼──┐      ┌────▼───┐   ┌───────▼──┐
    │ A₁₁    │   │  A₁₂   │      │  A₂₁   │   │   A₂₂    │
    │ (Leaf) │   │ (Leaf) │      │ (Leaf) │   │  (Leaf)  │
    └────────┘   └────────┘      └────────┘   └──────────┘
```

## 3. Message Flow Sequence Diagram

```
Parent     Kafka      Agent₁     Kafka      Sub₁₁      Sub₁₂
  │          │          │          │          │          │
  │─Action──>│          │          │          │          │
  │          │──Action─>│          │          │          │
  │          │          │          │          │          │
  │          │          │─Compile──│          │          │
  │          │          │  Msgs    │          │          │
  │          │          │          │          │          │
  │          │          │─Derive──>│          │          │
  │          │          │ Actions  │          │          │
  │          │          │          │          │          │
  │          │          │─A₁₁─────>│──A₁₁────>│          │
  │          │          │─A₁₂─────>│──────────┼─A₁₂─────>│
  │          │          │          │          │          │
  │          │          │          │   ╔══════╧════════╗ │
  │          │          │          │   ║  RECURSIVE    ║ │
  │          │          │          │   ║  STEP FLOW    ║ │
  │          │          │          │   ║  (same as     ║ │
  │          │          │          │   ║  Agent₁)      ║ │
  │          │          │          │   ╚══════╤════════╝ │
  │          │          │          │          │          │
  │          │          │          │<─Info₁₁──│          │
  │          │          │          │<─────────┼─Info₁₂───│
  │          │          │<─Collect─│          │          │
  │          │          │  Info    │          │          │
  │          │          │          │          │          │
  │          │          │─Execute─>│          │          │
  │          │          │ Own Act  │          │          │
  │          │          │          │          │          │
  │          │          │─Collate─>│          │          │
  │          │          │  State   │          │          │
  │          │          │          │          │          │
  │          │          │─Build───>│          │          │
  │          │          │Compiled  │          │          │
  │          │          │  Info    │          │          │
  │          │          │          │          │          │
  │          │<─────────│ Send to  │          │          │
  │          │  Info₁   │ Parent   │          │          │
  │<─Info₁───│          │          │          │          │
  │          │          │          │          │          │
```

## 4. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         AGENT DATA FLOW                          │
└──────────────────────────────────────────────────────────────────┘

INPUT (from Parent via Kafka):
┌─────────────────────┐
│   Action from       │
│   parent_agent      │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                            │
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │ Action       │────────>│ Derive       │                     │
│  │ Decompose    │         │ Sub-Actions  │                     │
│  │ (Model/Manual)│         │ A₁...Aₙ      │                     │
│  └──────────────┘         └──────┬───────┘                     │
│                                   │                             │
│                    ┌──────────────┴──────────────┐              │
│                    │                             │              │
│          ┌─────────▼─────────┐         ┌────────▼────────┐     │
│          │  Subordinate      │   ...   │  Subordinate    │     │
│          │  Processing       │         │  Processing     │     │
│          │  [RECURSIVE]      │         │  [RECURSIVE]    │     │
│          └─────────┬─────────┘         └────────┬────────┘     │
│                    │                            │              │
│          ┌─────────▼────────────────────────────▼────────┐     │
│          │  Collect from Kafka:                         │     │
│          │  • Observations: {obs₁₁, obs₁₂, ...obsₙ}     │     │
│          │  • Rewards: {r₁₁, r₁₂, ...rₙ}                │     │
│          │  • Status: {status₁₁, status₁₂, ...statusₙ}  │     │
│          │  • Compiled: {info₁₁, info₁₂, ...infoₙ}      │     │
│          └─────────┬────────────────────────────────────┘     │
│                    │                                           │
│          ┌─────────▼─────────┐                                │
│          │  Execute Own      │                                │
│          │  Action (aᵢ)      │                                │
│          └─────────┬─────────┘                                │
│                    │                                           │
│          ┌─────────▼─────────┐                                │
│          │  Generate Own     │                                │
│          │  Observation      │                                │
│          │  (obsᵢ)           │                                │
│          └─────────┬─────────┘                                │
│                    │                                           │
└────────────────────┼───────────────────────────────────────────┘
                     │
                     ▼
OUTPUT (to Parent via Kafka):
┌──────────────────────────────────────────────────────────────────┐
│  Compiled Information:                                           │
│  {                                                               │
│    own_observation: obsᵢ,                                        │
│    own_reward: rᵢ,                                               │
│    own_status: statusᵢ,                                          │
│    subordinate_info: {                                           │
│      sub₁: {obs: obs₁, reward: r₁, status: s₁, compiled: ...},  │
│      sub₂: {obs: obs₂, reward: r₂, status: s₂, compiled: ...},  │
│      ...                                                         │
│      subₙ: {obs: obsₙ, reward: rₙ, status: sₙ, compiled: ...}   │
│    }                                                             │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

## 5. Kafka Topic Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KAFKA BROKER TOPICS                          │
└─────────────────────────────────────────────────────────────────┘

ACTION TOPICS (Downstream: Parent → Child):
─────────────────────────────────────────
  parent_id → agent_id_action
  ├─ P → A₁_action
  ├─ P → A₂_action
  ├─ A₁ → A₁₁_action
  ├─ A₁ → A₁₂_action
  ├─ A₂ → A₂₁_action
  └─ A₂ → A₂₂_action

INFO TOPICS (Upstream: Child → Parent):
───────────────────────────────────────
  agent_id → parent_id_info
  ├─ A₁ → P_info
  ├─ A₂ → P_info
  ├─ A₁₁ → A₁_info
  ├─ A₁₂ → A₁_info
  ├─ A₂₁ → A₂_info
  └─ A₂₂ → A₂_info

BROADCAST TOPICS (Any → Any):
─────────────────────────────
  agent_id_broadcast
  └─ Any agent can publish messages to any other agent

MESSAGE STRUCTURE:
────────────────
{
  "sender_id": "agent_id",
  "recipient_id": "target_agent_id",
  "timestamp": "ISO-8601",
  "message_type": "action|info|broadcast",
  "payload": {
    "action": {...},           // for action messages
    "observation": {...},      // for info messages
    "reward": float,          // for info messages
    "status": "string",       // for info messages
    "compiled_info": {...}    // for info messages
  }
}
```

## 6. Execution Flow Options

```
┌─────────────────────────────────────────────────────────────────┐
│              SUBORDINATE EXECUTION PATTERNS                     │
└─────────────────────────────────────────────────────────────────┘

OPTION A: SYNCHRONOUS EXECUTION
─────────────────────────────────
  Agent₁
    ├─> sub₁.step()  [wait for completion]
    │     └─> Result₁
    ├─> sub₂.step()  [wait for completion]
    │     └─> Result₂
    └─> sub₃.step()  [wait for completion]
          └─> Result₃

  Total Time = T₁ + T₂ + T₃


OPTION B: ASYNCHRONOUS EXECUTION
──────────────────────────────────
  Agent₁
    ├─> async sub₁.step()  ┐
    ├─> async sub₂.step()  ├─> [concurrent execution]
    └─> async sub₃.step()  ┘
            │
            └─> await all results
                  └─> {Result₁, Result₂, Result₃}

  Total Time = max(T₁, T₂, T₃)


OPTION C: HYBRID (Protocol-based)
──────────────────────────────────
  Agent₁
    ├─> run_subordinates(execution_mode="async|sync")
    │     │
    │     └─> Protocol determines execution strategy
    │           based on agent configuration
    │
    └─> collect_results_from_kafka()
```

## 7. State Update Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT STATE LIFECYCLE                        │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ Initial State│
    │   (Sᵢ)       │
    └──────┬───────┘
           │
    ┌──────▼────────────────────────────────────────┐
    │  Receive Action from Parent                   │
    └──────┬────────────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────────┐
    │  Process Subordinates (if any)                │
    │  • Send actions                               │
    │  • Execute steps                              │
    │  • Collect info: {obs_sub, r_sub, s_sub}     │
    └──────┬────────────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────────┐
    │  Execute Own Action                           │
    │  • Generate own observation: obsᵢ             │
    │  • Calculate own reward: rᵢ                   │
    └──────┬────────────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────────┐
    │  Collate Observations                         │
    │  local_state = {obsᵢ} ∪ {obs_sub₁...obs_subₙ}│
    └──────┬────────────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────────┐
    │  Update Agent State                           │
    │  Sᵢ₊₁ = f(Sᵢ, aᵢ, local_state)                │
    └──────┬────────────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────────┐
    │  Build Compiled Info                          │
    │  compiled = {                                 │
    │    own: {obs, reward, status, state},         │
    │    subordinates: {...}                        │
    │  }                                            │
    └──────┬────────────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────────┐
    │  Send to Parent via Kafka                     │
    └──────┬────────────────────────────────────────┘
           │
    ┌──────▼───────┐
    │  New State   │
    │   (Sᵢ₊₁)     │
    └──────────────┘
```

## Key Design Principles

1. **Recursive Structure**: Each agent follows the same step flow regardless of hierarchy level
2. **Kafka-Centric Communication**: All inter-agent communication flows through Kafka topics
3. **Flexible Execution**: Supports both synchronous and asynchronous subordinate execution
4. **Information Aggregation**: Each agent builds compiled information from its own observations and subordinate reports
5. **Decoupled Architecture**: Agents communicate only via Kafka, enabling distributed execution
6. **Action Derivation Options**: Model-based or manual action decomposition for subordinates
7. **Broadcast Capability**: Agents can send messages to any other agent at any time

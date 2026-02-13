# HERON Complete Data Flow Documentation

This document provides a comprehensive analysis of how data flows through the HERON framework in both CTDE Training and Event-Driven Testing modes.

---

## **Table of Contents**

1. [Core Data Types](#core-data-types)
2. [State Data Flow](#state-data-flow)
3. [Action Data Flow](#action-data-flow)
4. [Action Passing (Hierarchical Coordination)](#action-passing-hierarchical-coordination)
5. [Observation Data Flow](#observation-data-flow)
6. [Feature Data Flow](#feature-data-flow)
7. [Message Passing Architecture](#message-passing-architecture)
8. [CTDE Training Mode - Complete Flow](#ctde-training-mode---complete-flow)
9. [Event-Driven Testing Mode - Complete Flow](#event-driven-testing-mode---complete-flow)
10. [ProxyAgent State Cache Architecture](#proxyagent-state-cache-architecture)
11. [Protocol-Based Coordination](#protocol-based-coordination)

---

## **Core Data Types**

### **Type Transformation Rules**

| Component | Internal Representation | Storage in Proxy | Message Passing |
|-----------|------------------------|------------------|-----------------|
| **State** | State object (FieldAgentState, etc.) | **State object** | Dict with metadata via `to_dict()` |
| **Action** | Action object | Not stored in proxy | Action object (direct) or Dict (messages) |
| **Observation** | Observation object | Not stored (computed on-demand) | Dict via `obs.to_dict()` |
| **Feature** | FeatureProvider object | Part of State object | Dict via `feature.to_dict()` |
| **Message** | Message object | Broker queues | Serialized payload |

### **Key Principle**
**ProxyAgent stores State objects directly** - This enables feature-level visibility filtering:
- State objects maintained with full type information
- Direct access to `state.observed_by()` for visibility filtering
- Features maintain class identity (FeatureProvider instances)
- Serialization needed only at message boundaries (State <-> Dict)

---

## **State Data Flow**

### **State Object Structure**

```python
# Agent creates State with features
state = FieldAgentState(owner_id="battery_1", owner_level=1)
state.features = [
    BatteryChargeFeature(soc=0.5, capacity=100.0)
]

# State object attributes:
state.owner_id = "battery_1"
state.owner_level = 1
state.features = [FeatureProvider objects]
```

### **State Operations**

#### **1. Init State (Initialization)**

```
Agent.init_state()
├─> Create State object (e.g., FieldAgentState)
├─> Set owner_id = self.agent_id
├─> Set owner_level = self.level
├─> Create and append FeatureProvider objects
└─> Store in self.state

Example:
    self.state = FieldAgentState(owner_id="battery_1", owner_level=1)
    self.state.features.append(BatteryChargeFeature(soc=0.5, capacity=100.0))
```

**Data Outflow:**
- `self.state` -> State object with features

---

#### **2. Set State (Storage in Proxy)**

**CTDE Mode:**
```
Agent updates state -> Send to proxy

agent.apply_action()
├─> Updates self.state.features[0].soc = new_value
└─> State object modified in-place

proxy.set_local_state(agent_id, state)
├─> Input: agent_id="battery_1", state=FieldAgentState(...)
├─> Storage: state_cache["agents"]["battery_1"] = state  # State object!
└─> Stored as STATE OBJECT with full type information!
```

**Event-Driven Mode:**
```
Agent updates state -> Serialize -> Send message -> Proxy stores

action_effect_handler()
├─> apply_action() -> Updates self.state
├─> Serialize: self.state.to_dict(include_metadata=True)
│   └─> Returns: {
│           "_owner_id": "battery_1",
│           "_owner_level": 1,
│           "_state_type": "FieldAgentState",
│           "features": {"BatteryChargeFeature": {"soc": 0.53, "capacity": 100.0}}
│       }
├─> Send message: {"set_state": "local", "body": serialized_dict}
└─> Proxy receives message

proxy.message_delivery_handler()
├─> Extract: agent_id = body["_owner_id"]
├─> Reconstruct: state = State.from_dict(body)
└─> Call: set_local_state(agent_id, state)
    └─> Storage: state_cache["agents"]["battery_1"] = state  # State object!
```

**Data Transformations:**
```
State object -> .to_dict() -> Message -> State.from_dict() -> State object
                          ↓
            proxy.state_cache["agents"][agent_id]
```

---

#### **3. Get State (Retrieval from Proxy)**

**CTDE Mode:**
```
Agent requests own state from proxy

proxy.get_local_state(agent_id, requestor_level)
├─> Input: agent_id="battery_1", requestor_level=1
├─> Retrieval: state_obj = state_cache["agents"]["battery_1"]  # State object!
├─> Apply visibility filtering: state_obj.observed_by(agent_id, requestor_level)
└─> Return: {"BatteryChargeFeature": np.array([0.53, 100.0])}  # Feature vectors!
```

**Event-Driven Mode:**
```
Agent sends message requesting state -> Proxy responds

Agent:
├─> schedule_message_delivery(message={"get_info": "local_state"})
└─> Sent to proxy

Proxy.message_delivery_handler():
├─> Receives request
├─> local_state = self.get_local_state(sender_id, requestor_level)
│   └─> Returns filtered feature vectors
├─> Send response: {"get_local_state_response": {"body": local_state}}
└─> Agent receives dict in message
```

**Data Outflow:**
```
state_cache["agents"][agent_id] -> observed_by() -> Filtered vectors -> Agent
```

**Usage Example (With Visibility Filtering):**
```python
# Agent receives feature vectors from proxy (after visibility filtering)
local_state = proxy.get_local_state(self.agent_id, self.level)
# local_state = {"BatteryChargeFeature": np.array([0.53, 100.0])}  # Numpy arrays!

# Extract values from feature vector
feature_vec = local_state["BatteryChargeFeature"]  # np.array([0.53, 100.0])
soc = feature_vec[0]  # First element is SOC
reward = soc  # Use filtered state, not self.state!
```

---

### **State Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Agent Layer (State Objects)                                     │
├─────────────────────────────────────────────────────────────────┤
│ init_state() -> self.state = FieldAgentState(...)               │
│                self.state.features = [BatteryChargeFeature(...)]│
│                                                                 │
│ apply_action() -> self.state.features[0].soc = new_value        │
│                                                                 │
│ Send to proxy: proxy.set_local_state(aid, state)  # State obj!  │
└─────────────────────────────────────────────────────────────────┘
                            ↓ (No conversion needed!)
┌─────────────────────────────────────────────────────────────────┐
│ Proxy Layer (State Object Storage)                              │
├─────────────────────────────────────────────────────────────────┤
│ state_cache["agents"]["battery_1"] = FieldAgentState(...)       │
│     └─> Contains: [BatteryChargeFeature(soc=0.53, capacity=100)]│
│                                                                 │
│ Stores FULL State objects with FeatureProvider instances!       │
└─────────────────────────────────────────────────────────────────┘
                            ↓ get_local_state() + visibility filtering
┌─────────────────────────────────────────────────────────────────┐
│ Visibility Filtering Layer (state.observed_by())                │
├─────────────────────────────────────────────────────────────────┤
│ state_obj.observed_by(requestor_id, requestor_level)            │
│     ↓ Feature-level filtering                                   │
│ Returns: {"BatteryChargeFeature": np.array([0.53, 100.0])}      │
│          └─> Only visible features, as numpy arrays!            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Agent Layer (Filtered Vector Usage)                             │
├─────────────────────────────────────────────────────────────────┤
│ local_state = proxy.get_local_state(aid) -> Filtered vectors    │
│                                                                 │
│ compute_local_reward(local_state):                              │
│     feature_vec = local_state["BatteryChargeFeature"]  # array  │
│     soc = feature_vec[0]  # Extract from vector                 │
│     return soc                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Action Data Flow**

### **Action Object Structure**

```python
action = Action()
action.set_specs(
    dim_c=1,  # 1 continuous action
    range=(np.array([-1.0]), np.array([1.0]))
)
action.set_values(np.array([0.3]))

# Action object attributes:
action.c = np.array([0.3])      # Continuous actions
action.d = np.array([])         # Discrete actions (empty)
action.dim_c = 1
action.dim_d = 0
action.range = ([-1.0], [1.0])
```

### **Action Operations**

#### **1. Init Action (Initialization)**

```
Agent.init_action()
├─> Create Action object
├─> Set action specs (dim_c, dim_d, range, ncats)
├─> Reset to neutral values
└─> Store in self.action

Example:
    self.action = Action()
    self.action.set_specs(dim_c=1, range=([-1.0], [1.0]))
    self.action.set_values(np.array([0.0]))  # Neutral
```

**Data Outflow:**
- `self.action` -> Action object with specs

---

#### **2. Set Action (Policy Decision or Upstream)**

**CTDE Mode - From External Actions Dict:**
```
env.step(actions)
├─> actions = {"battery_1": action_object, ...}
├─> SystemAgent.execute(actions, proxy)
    ├─> layered = layer_actions(actions)
    │   └─> {"self": actions["system_agent"],
    │        "subordinates": {"battery_1": actions["battery_1"], ...}}
    │
    └─> handle_self_action(layered["self"], proxy)
        ├─> set_action(action)
        │   └─> self.action.set_values(action)
        │       ├─> If action is Action object: Copy c and d values
        │       └─> If action is dict: Extract "c" and "d" keys
        └─> Action object updated!
```

**CTDE Mode - From Policy:**
```
handle_self_action(None, proxy)  # No external action
├─> obs = proxy.get_observation(agent_id)
│   └─> Returns Observation object
├─> action = policy.forward(observation=obs)
│   ├─> obs.__array__() converts to np.ndarray automatically
│   └─> Returns Action object
└─> set_action(action)
    └─> self.action.set_values(action)
```

**Event-Driven Mode:**
```
SystemAgent.tick()
├─> Request observation via message
└─> [Later] Receive observation response

message_delivery_handler() - "get_obs_response"
├─> obs_dict = message["body"]
├─> obs = Observation.from_dict(obs_dict)
└─> compute_action(obs, scheduler)
    ├─> action = policy.forward(observation=obs)
    ├─> set_action(action)
    └─> schedule_action_effect(delay=act_delay)
```

**Data Transformations:**
```
Policy -> Action object -> set_values() -> self.action updated
                                        ↓
                        No storage in proxy!
                        Actions exist only in agents
```

---

#### **3. Apply Action (State Update)**

```
Agent.apply_action()
└─> set_state()
    ├─> Read: current_soc = self.soc (from self.state.features[0].soc)
    ├─> Read: action_value = self.action.c[0]
    ├─> Compute: new_soc = current_soc + action_value * 0.01
    └─> Update: self.state.features[0].set_values(soc=new_soc)
        └─> State object modified in-place!
```

**Data Flow:**
```
self.action (Action object) -> Extract values -> Update self.state (State object)
```

---

### **Action Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Policy Layer                                                    │
├─────────────────────────────────────────────────────────────────┤
│ policy.forward(observation) -> Returns Action object            │
│     action = Action()                                           │
│     action.set_specs(dim_c=1, range=([-1,1]))                   │
│     action.set_values(np.array([0.3]))                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Training Loop / Environment Layer                               │
├─────────────────────────────────────────────────────────────────┤
│ actions[agent_id] = action  # Action object                     │
│ env.step(actions)                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Agent Layer (Execution)                                         │
├─────────────────────────────────────────────────────────────────┤
│ set_action(action) -> self.action.set_values(action)            │
│                                                                 │
│ apply_action() -> Read self.action.c[0]                         │
│                -> Update self.state.features[0].soc             │
└─────────────────────────────────────────────────────────────────┘
```

**Important: Actions are NEVER stored in proxy!** They exist only within agents during the act() phase.

---

## **Action Passing (Hierarchical Coordination)**

### **Overview**

Action passing enables hierarchical coordination where parent agents can send actions to their subordinates. This is essential for centralized control strategies where a coordinator or system agent makes decisions for lower-level agents.

### **Action Passing Modes**

#### **1. Self-Action (Policy-Driven)**
```
Agent computes its own action via policy:

agent.tick() or agent.act()
├─> obs = proxy.get_observation(self.agent_id)
├─> action = self.policy.forward(obs)
└─> self.set_action(action)
```

#### **2. Upstream Action (Parent-Driven)**
```
Parent sends action to subordinate:

Parent (Coordinator/System):
├─> Compute subordinate actions via protocol
├─> self.send_subordinate_action(sub_id, action)
│   └─> broker.publish(action_channel, Message(action))
└─> Actions queued for subordinates

Subordinate (Field/Coordinator):
├─> actions = self.receive_upstream_actions()
│   └─> broker.consume(action_channel)
├─> If upstream action exists:
│   └─> self.set_action(upstream_action)  # Priority over policy!
└─> Else: Use policy
```

### **Action Passing Flow Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│ SystemAgent (Level 3)                                           │
├─────────────────────────────────────────────────────────────────┤
│ 1. Receive global observation                                   │
│ 2. Compute system-level action via policy                       │
│ 3. Protocol.coordinate() decomposes into subordinate actions    │
│ 4. send_subordinate_action() to each coordinator                │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ ACTION Message via broker
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CoordinatorAgent (Level 2)                                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. receive_upstream_actions() from system                       │
│ 2. If upstream action exists: use it (priority!)                │
│ 3. Else: compute own action via policy                          │
│ 4. Protocol.coordinate() decomposes into field actions          │
│ 5. send_subordinate_action() to each field agent                │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ ACTION Message via broker
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ FieldAgent (Level 1)                                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. receive_upstream_actions() from coordinator                  │
│ 2. If upstream action exists: use it (priority!)                │
│ 3. Else: compute own action via policy                          │
│ 4. apply_action() -> Update state                               │
└─────────────────────────────────────────────────────────────────┘
```

### **Action Priority Rules**

```python
def get_action_for_tick(self) -> Action:
    """Determine which action to use."""
    # Priority 1: Upstream action from parent
    upstream_actions = self.receive_upstream_actions()
    if upstream_actions:
        return upstream_actions[0]  # Use first upstream action

    # Priority 2: Self-computed action via policy
    if self.policy is not None:
        obs = self.proxy.get_observation(self.agent_id)
        return self.policy.forward(observation=obs)

    # Priority 3: No action (neutral)
    return None
```

### **Message Broker Integration**

```python
# Channel naming convention
action_channel = ChannelManager.action_channel(
    sender_id="coordinator_1",
    recipient_id="field_1",
    env_id="default"
)
# Result: "env:default/action/coordinator_1->field_1"

# Sending action (Parent)
def send_subordinate_action(self, subordinate_id: str, action: Action):
    channel = ChannelManager.action_channel(
        sender_id=self.agent_id,
        recipient_id=subordinate_id,
        env_id=self.env_id
    )
    message = Message(
        message_type=MessageType.ACTION,
        sender_id=self.agent_id,
        recipient_id=subordinate_id,
        payload={"action": action}
    )
    self._message_broker.publish(channel, message)

# Receiving action (Subordinate)
def receive_upstream_actions(self) -> List[Action]:
    if self.upstream_id is None:
        return []

    channel = ChannelManager.action_channel(
        sender_id=self.upstream_id,
        recipient_id=self.agent_id,
        env_id=self.env_id
    )
    messages = self._message_broker.consume(
        channel=channel,
        recipient_id=self.agent_id,
        env_id=self.env_id
    )
    return [msg.payload["action"] for msg in messages]
```

### **CTDE Mode Action Passing**

```
env.step(actions)
│
└─> SystemAgent.execute(actions, proxy)
    │
    ├─> self.act(actions, proxy)
    │   │
    │   ├─> layered = layer_actions(actions)
    │   │   # Hierarchical structure:
    │   │   # {
    │   │   #   "self": actions["system_agent"],
    │   │   #   "subordinates": {
    │   │   #     "coord_1": {
    │   │   #       "self": actions["coord_1"],
    │   │   #       "subordinates": {"field_1": ..., "field_2": ...}
    │   │   #     }
    │   │   #   }
    │   │   # }
    │   │
    │   ├─> handle_self_action(layered["self"], proxy)
    │   │   ├─> set_action(action)
    │   │   └─> apply_action()
    │   │
    │   └─> handle_subordinate_actions(layered["subordinates"], proxy)
    │       │
    │       └─> For each coordinator:
    │           │
    │           ├─> # Option A: Pass action directly (if in actions dict)
    │           │   coordinator.execute(coord_actions, proxy)
    │           │
    │           └─> # Option B: Send via broker (for distributed)
    │               send_subordinate_action(coord_id, coord_action)
    │
    └─> [Coordinators recursively handle their field agents]
```

### **Event-Driven Mode Action Passing**

```
t=0.0: AGENT_TICK(system_agent)
│
├─> SystemAgent.tick()
│   ├─> Compute system action
│   ├─> Protocol.coordinate() -> subordinate actions
│   └─> For each coordinator:
│       └─> send_subordinate_action(coord_id, action)
│           └─> Queued in broker
│
├─> schedule_agent_tick(coordinator_1) at t=tick_interval
└─> schedule_subordinate_ticks()

t=tick_interval: AGENT_TICK(coordinator_1)
│
├─> CoordinatorAgent.tick()
│   ├─> upstream_actions = receive_upstream_actions()
│   │   └─> Consume from broker (non-blocking)
│   │
│   ├─> If upstream_action:
│   │   └─> set_action(upstream_action)  # Use parent's decision
│   ├─> Else:
│   │   └─> action = policy.forward(obs)  # Own decision
│   │
│   ├─> Protocol.coordinate() -> field actions
│   └─> For each field agent:
│       └─> send_subordinate_action(field_id, action)
│
└─> schedule_agent_tick(field_1, field_2, ...) at t=tick_interval

t=tick_interval: AGENT_TICK(field_1)
│
└─> FieldAgent.tick()
    ├─> upstream_actions = receive_upstream_actions()
    ├─> If upstream_action:
    │   └─> set_action(upstream_action)
    ├─> Else:
    │   └─> action = policy.forward(obs)
    └─> schedule_action_effect(delay=act_delay)
```

---

## **Observation Data Flow**

### **Observation Object Structure**

```python
obs = Observation(
    local={"BatteryChargeFeature": np.array([0.53, 100.0])},
    global_info={"agent_states": {...}},
    timestamp=0.0
)

# Observation attributes:
obs.local = dict        # Agent-specific state (filtered vectors)
obs.global_info = dict  # Global/shared state
obs.timestamp = float   # Current time
```

### **Observation Operations**

#### **1. Build Observation (Construction from Proxy State)**

```
proxy.get_observation(sender_id, requestor_level)
├─> global_state = get_global_states(sender_id, requestor_level)
│   ├─> For each agent in state_cache["agents"]:
│   │   ├─> state_obj = state_cache["agents"][agent_id]
│   │   ├─> filtered = state_obj.observed_by(sender_id, requestor_level)
│   │   └─> Add to global_state if visible
│   │
│   └─> Returns: Dict of filtered feature vectors
│
├─> local_state = get_local_state(sender_id, requestor_level)
│   ├─> state_obj = state_cache["agents"][sender_id]
│   ├─> filtered = state_obj.observed_by(sender_id, requestor_level)
│   └─> Returns: Dict of filtered feature vectors
│
└─> return Observation(
        local=local_state,        # Filtered vectors!
        global_info=global_state, # Filtered vectors!
        timestamp=self._timestep
    )
```

**Data Inflow:**
```
state_cache["agents"] -> observed_by() -> filtered vectors -> Observation
```

**Key Insight:** Observation contains **filtered numpy arrays**, not raw State objects!

---

#### **2. Vectorize Observation (For RL Algorithms)**

```
Observation -> numpy array conversion

obs.vector()
├─> Flatten obs.local dict
│   └─> For each feature_name, feature_vec in local.items():
│       └─> Append feature_vec (already np.array)
│
├─> Flatten obs.global_info dict
│   └─> [Similar recursive flattening]
│
└─> Concatenate all parts
    └─> Returns: np.array([0.53, 100.0, ...], dtype=float32)

# Automatic conversion:
policy.forward(observation=obs)
└─> obs.__array__() called automatically
    └─> Returns obs.vector()
```

**Data Transformations:**
```
Observation(local={...}, global_info={...})
    ↓ .vector() or __array__()
np.ndarray([0.53, 100.0, ...])  # For neural networks
```

---

#### **3. Serialize/Deserialize Observation (Message Passing)**

**Serialization (Sending):**
```
obs = Observation(local={...}, global_info={...}, timestamp=0.0)

obs.to_dict()
└─> Returns: {
        "timestamp": 0.0,
        "local": {"BatteryChargeFeature": [0.53, 100.0]},  # Arrays -> lists
        "global_info": {"agent_states": {...}}
    }

# Used in message passing:
message = {"get_obs_response": {"body": obs.to_dict()}}
```

**Deserialization (Receiving):**
```
obs_dict = message["get_obs_response"]["body"]
# obs_dict = {"timestamp": 0.0, "local": {...}, "global_info": {...}}

obs = Observation.from_dict(obs_dict)
└─> Reconstructs Observation object
    └─> obs.local = {k: np.array(v) for k, v in obs_dict["local"].items()}
    └─> obs.global_info = obs_dict["global_info"]
    └─> obs.timestamp = obs_dict["timestamp"]
```

---

### **Observation Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Proxy State Cache (State Objects)                               │
├─────────────────────────────────────────────────────────────────┤
│ state_cache["agents"]["battery_1"] = FieldAgentState(           │
│     owner_id="battery_1",                                       │
│     features=[BatteryChargeFeature(soc=0.53, capacity=100.0)]   │
│ )                                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓ get_observation() + observed_by()
┌─────────────────────────────────────────────────────────────────┐
│ Observation Construction (Visibility Filtered)                  │
├─────────────────────────────────────────────────────────────────┤
│ Observation(                                                    │
│     local={"BatteryChargeFeature": np.array([0.53, 100.0])},   │
│     global_info={...filtered states...},                        │
│     timestamp=0.0                                               │
│ )                                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓ __array__()
┌─────────────────────────────────────────────────────────────────┐
│ Policy / RL Algorithm                                           │
├─────────────────────────────────────────────────────────────────┤
│ obs_vec = np.array([0.53, 100.0, ...])                          │
│ action = policy.forward(observation=obs)  # Auto-vectorized     │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Feature Data Flow**

### **Feature Object Structure**

```python
class BatteryChargeFeature(FeatureProvider):
    visibility = ["public"]  # Class-level metadata

    soc: float = 0.5
    capacity: float = 100.0

# Feature object attributes (auto-set by metaclass):
feature.feature_name = "BatteryChargeFeature"
feature.visibility = ["public"]
feature.soc = 0.5
feature.capacity = 100.0
```

### **Feature Operations**

#### **1. Feature Creation & Registration**

```
Agent.init_state()
├─> battery_feature = BatteryChargeFeature(soc=0.5, capacity=100.0)
│   └─> FeatureProvider object created
│   └─> Auto-registered in _FEATURE_REGISTRY via FeatureMeta
│
├─> self.state = FieldAgentState(owner_id="battery_1", owner_level=1)
├─> self.state.features.append(battery_feature)
└─> State contains list of FeatureProvider objects
```

**Data Outflow:**
```
FeatureProvider object -> Appended to state.features list
```

---

#### **2. Feature Serialization (State -> Dict)**

```
state.to_dict()
├─> For each feature in self.features:
│   ├─> feature_name = feature.feature_name  # "BatteryChargeFeature"
│   ├─> feature_dict = feature.to_dict()
│   │   └─> Returns: {"soc": 0.5, "capacity": 100.0}
│   └─> result[feature_name] = feature_dict
│
└─> Returns: {
        "_owner_id": "battery_1",
        "_owner_level": 1,
        "_state_type": "FieldAgentState",
        "features": {
            "BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}
        }
    }
```

**Data Transformations:**
```
FeatureProvider object
    ↓ .to_dict()
{"soc": 0.5, "capacity": 100.0}
    ↓ Wrapped with feature_name
{"BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}}
```

---

#### **3. Feature Update**

```
Update feature values in State object

state.update_feature("BatteryChargeFeature", soc=0.8)
├─> Find feature with matching feature_name
├─> Call feature.set_values(soc=0.8)
│   └─> feature.soc = np.clip(0.8, 0.0, 1.0)
└─> Feature object modified in-place!

# Or update via State.update():
state.update({
    "BatteryChargeFeature": {"soc": 0.8}
})
```

---

#### **4. Feature Vectorization**

```
Get numeric representation of feature

feature.vector()
└─> Returns: np.array([feature.soc, feature.capacity], dtype=float32)
    └─> Example: np.array([0.5, 100.0])

state.vector()
├─> For each feature: vectors.append(feature.vector())
├─> Concatenate all feature vectors
└─> Returns: np.array([0.5, 100.0, ...], dtype=float32)
```

**Data Transformations:**
```
FeatureProvider (soc=0.5, capacity=100.0)
    ↓ .vector()
np.array([0.5, 100.0])
```

---

#### **5. Feature Visibility (Observation Filtering)**

```
state.observed_by(requestor_id, requestor_level)
├─> For each feature in state.features:
│   ├─> Check: feature.is_observable_by(requestor_id, requestor_level, owner_id, owner_level)
│   │   ├─> If visibility = ["public"]: Always True
│   │   ├─> If visibility = ["owner"]: True if requestor_id == owner_id
│   │   ├─> If visibility = ["upper_level"]: True if requestor_level == owner_level + 1
│   │   └─> If visibility = ["system"]: True if requestor_level >= 3
│   │
│   └─> If observable: Include feature.vector() in result
│
└─> Returns: {
        "BatteryChargeFeature": np.array([0.5, 100.0])  # Only visible features!
    }
```

**Visibility Enforcement Active** in `proxy.get_local_state()` and `proxy.get_global_states()`.

#### **6. Complete Visibility Filtering Example**

```
Field Agent (battery_1, level=1) requests observation:

proxy.get_observation("battery_1", requestor_level=1)
│
├─> LOCAL STATE (own state):
│   state_obj = state_cache["agents"]["battery_1"]  # FieldAgentState
│   filtered = state_obj.observed_by("battery_1", requestor_level=1)
│   │
│   │   Feature: BatteryChargeFeature (visibility=["public"])
│   │   ├─> is_observable_by("battery_1", 1, "battery_1", 1)
│   │   ├─> Check "public": True
│   │   └─> Include: array([0.503, 100.0])
│   │
│   Returns: {"BatteryChargeFeature": array([0.503, 100.0])}
│
├─> GLOBAL STATE (other agents):
│   For battery_2:
│       state_obj = state_cache["agents"]["battery_2"]
│       filtered = state_obj.observed_by("battery_1", requestor_level=1)
│       │
│       │   Feature: BatteryChargeFeature (visibility=["public"])
│       │   ├─> is_observable_by("battery_1", 1, "battery_2", 1)
│       │   ├─> Check "public": True
│       │   └─> Include: array([0.498, 100.0])
│       │
│       Returns: {"BatteryChargeFeature": array([0.498, 100.0])}
│
│   For coordinator_1:
│       Feature: CoordinatorPrivateFeature (visibility=["owner"])
│       ├─> is_observable_by("battery_1", 1, "coordinator_1", 2)
│       ├─> Check "owner": battery_1 != coordinator_1 -> False
│       └─> NOT included (filtered out!)
│
└─> Return: Observation(
        local={"BatteryChargeFeature": array([0.503, 100.0])},
        global_info={
            "battery_2": {"BatteryChargeFeature": array([0.498, 100.0])}
            # coordinator_1 private features NOT visible!
        },
        timestamp=1.0
    )
```

**Visibility Enforcement:**
- battery_1 sees own state (owner visibility)
- battery_1 sees battery_2 state (public visibility)
- battery_1 does NOT see coordinator private features (filtered out)
- battery_1 does NOT see system-only features (filtered out)

---

### **Feature Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Feature Definition (Class)                                      │
├─────────────────────────────────────────────────────────────────┤
│ class BatteryChargeFeature(FeatureProvider):                    │
│     visibility = ["public"]                                     │
│     soc: float = 0.5                                            │
│     capacity: float = 100.0                                     │
│                                                                 │
│ # Auto-registered via FeatureMeta metaclass!                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓ Instantiation
┌─────────────────────────────────────────────────────────────────┐
│ Feature Object Creation                                         │
├─────────────────────────────────────────────────────────────────┤
│ feature = BatteryChargeFeature(soc=0.5, capacity=100.0)         │
│ feature.feature_name = "BatteryChargeFeature"  # Auto-set       │
└─────────────────────────────────────────────────────────────────┘
                            ↓ Added to State
┌─────────────────────────────────────────────────────────────────┐
│ State Composition                                               │
├─────────────────────────────────────────────────────────────────┤
│ state.features = [feature1, feature2, ...]                      │
│                                                                 │
│ state.vector() -> Concatenates all feature.vector() outputs     │
└─────────────────────────────────────────────────────────────────┘
                            ↓ Proxy retrieval
┌─────────────────────────────────────────────────────────────────┐
│ Visibility Filtering (observed_by)                              │
├─────────────────────────────────────────────────────────────────┤
│ {                                                               │
│     "BatteryChargeFeature": np.array([0.5, 100.0])  # Visible!  │
│     # PrivateFeature filtered out if not authorized             │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Message Passing Architecture**

### **Message System Components**

#### **Message Structure**

```python
@dataclass
class Message:
    message_type: MessageType  # ACTION, INFO, STATE_UPDATE, etc.
    sender_id: str             # Who sent the message
    recipient_id: str          # Who receives the message
    payload: Dict[str, Any]    # Message content
    timestamp: Optional[float] = None  # When message was created
    env_id: str = "default"    # Environment identifier
```

#### **Message Types**

```python
class MessageType(Enum):
    ACTION = "action"          # Parent -> Child action commands
    INFO = "info"              # Information/observation requests
    STATE_UPDATE = "state"     # State synchronization
    BROADCAST = "broadcast"    # Multi-recipient messages
    CUSTOM = "custom"          # Domain-specific types
```

### **Message Broker**

```python
class MessageBroker(ABC):
    """Abstract base for message delivery."""

    @abstractmethod
    def publish(self, channel: str, message: Message) -> None:
        """Send message to channel."""
        pass

    @abstractmethod
    def consume(
        self,
        channel: str,
        recipient_id: str,
        env_id: str
    ) -> List[Message]:
        """Receive messages from channel (non-blocking)."""
        pass

    @abstractmethod
    def create_channel(self, channel_name: str) -> None:
        """Create a new channel."""
        pass
```

#### **In-Memory Broker Implementation**

```python
class InMemoryBroker(MessageBroker):
    """In-process message broker for single-machine execution."""

    def __init__(self):
        self._channels: Dict[str, Queue] = {}

    def publish(self, channel: str, message: Message) -> None:
        if channel not in self._channels:
            self.create_channel(channel)
        self._channels[channel].put(message)

    def consume(self, channel, recipient_id, env_id) -> List[Message]:
        if channel not in self._channels:
            return []

        messages = []
        while not self._channels[channel].empty():
            msg = self._channels[channel].get_nowait()
            if msg.recipient_id == recipient_id:
                messages.append(msg)
        return messages
```

### **Channel Management**

```python
class ChannelManager:
    """Creates standardized channel names."""

    @staticmethod
    def action_channel(sender_id: str, recipient_id: str, env_id: str) -> str:
        return f"env:{env_id}/action/{sender_id}->{recipient_id}"

    @staticmethod
    def info_channel(sender_id: str, recipient_id: str, env_id: str) -> str:
        return f"env:{env_id}/info/{sender_id}->{recipient_id}"

    @staticmethod
    def state_channel(agent_id: str, env_id: str) -> str:
        return f"env:{env_id}/state/{agent_id}"

    @staticmethod
    def broadcast_channel(env_id: str) -> str:
        return f"env:{env_id}/broadcast"
```

### **Message Flow Patterns**

#### **1. Observation Request/Response**

```
Agent -> Proxy (Request):
├─> channel = info_channel(agent_id, PROXY_ID, env_id)
├─> message = Message(
│       message_type=MessageType.INFO,
│       sender_id=agent_id,
│       recipient_id=PROXY_ID,
│       payload={"get_info": "obs"}
│   )
└─> broker.publish(channel, message)

Proxy -> Agent (Response):
├─> obs = get_observation(sender_id, requestor_level)
├─> response = Message(
│       message_type=MessageType.INFO,
│       sender_id=PROXY_ID,
│       recipient_id=agent_id,
│       payload={"get_obs_response": {"body": obs.to_dict()}}
│   )
└─> broker.publish(response_channel, response)
```

#### **2. Action Passing**

```
Coordinator -> FieldAgent:
├─> channel = action_channel(coord_id, field_id, env_id)
├─> message = Message(
│       message_type=MessageType.ACTION,
│       sender_id=coord_id,
│       recipient_id=field_id,
│       payload={"action": action}  # Action object or dict
│   )
└─> broker.publish(channel, message)

FieldAgent receives:
├─> channel = action_channel(coord_id, field_id, env_id)
├─> messages = broker.consume(channel, field_id, env_id)
└─> For msg in messages:
    └─> upstream_action = msg.payload["action"]
```

#### **3. State Update**

```
Agent -> Proxy:
├─> channel = state_channel(agent_id, env_id)
├─> message = Message(
│       message_type=MessageType.STATE_UPDATE,
│       sender_id=agent_id,
│       recipient_id=PROXY_ID,
│       payload={
│           "set_state": "local",
│           "body": state.to_dict(include_metadata=True)
│       }
│   )
└─> broker.publish(channel, message)

Proxy receives and updates:
├─> state_dict = message.payload["body"]
├─> state = State.from_dict(state_dict)
└─> set_local_state(agent_id, state)
```

### **Event-Driven Message Scheduling**

```python
def schedule_message_delivery(
    self,
    scheduler: EventScheduler,
    sender_id: str,
    recipient_id: str,
    message: Dict,
    delay: float = 0.0
):
    """Schedule a message to be delivered after delay."""
    event = Event(
        event_type=EventType.MESSAGE_DELIVERY,
        timestamp=scheduler.current_time + delay,
        agent_id=recipient_id,
        payload={
            "sender": sender_id,
            "recipient": recipient_id,
            "message": message
        }
    )
    scheduler.schedule(event)
```

---

## **CTDE Training Mode - Complete Flow**

### **Phase 0: Initialization**

```
env = EnergyManagementEnv(system_agent=grid_system_agent)
│
├─> EnvCore.__init__()
    │
    ├─> _register_agents()
    │   ├─> Register all agents recursively
    │   ├─> Create ProxyAgent()
    │   └─> Create MessageBroker()
    │
    ├─> _initialize_agents()
    │   └─> For each agent (system -> coord -> field):
    │       │
    │       ├─> agent.init_action()
    │       │   └─> self.action = Action()
    │       │       self.action.set_specs(...)
    │       │       DATA: Action object created
    │       │
    │       ├─> agent.init_state()
    │       │   └─> self.state = FieldAgentState(owner_id, owner_level)
    │       │       self.state.features.append(BatteryChargeFeature(...))
    │       │       DATA: State object with features created
    │       │
    │       └─> proxy.register_agent(agent_id, agent_state=agent.state)
    │           └─> proxy.set_local_state(agent_id, agent.state)
    │               │
    │               STORAGE:
    │               state_cache["agents"]["battery_1"] = agent.state  # State object!
    │
    └─> proxy.init_global_state()
        └─> Compile agent references into global view
```

**Data State After Init:**
```python
proxy.state_cache = {
    "agents": {
        "battery_1": FieldAgentState(...),      # State object!
        "battery_2": FieldAgentState(...),      # State object!
        "coordinator_1": CoordinatorAgentState(...),
        "system_agent": SystemAgentState(...)
    }
}
```

---

### **Phase 1: Reset**

```
obs, info = env.reset(seed=42)
│
├─> scheduler.reset() -> Clear events
├─> broker.clear() -> Clear messages
├─> proxy.reset() -> state_cache = {}  [CLEARED!]
│
├─> system_agent.reset(seed, proxy)
│   ├─> self._timestep = 0.0
│   │
│   ├─> init_action()
│   │   └─> self.action = Action() [Reset]
│   │
│   ├─> init_state()
│   │   └─> self.state = SystemAgentState(...) [Fresh state]
│   │
│   ├─> proxy.set_local_state(self.agent_id, self.state)
│   │   │
│   │   STORAGE:
│   │   state_cache["agents"]["system_agent"] = self.state  # State object!
│   │
│   └─> For each subordinate: subordinate.reset(seed, proxy)
│       └─> [Recursive reset, all update proxy cache]
│
├─> proxy.init_global_state()
│
└─> return self.observe(proxy)
    │
    DATA CONSTRUCTION:
    For each agent:
        obs[agent_id] = proxy.get_observation(agent_id, agent_level)
            ├─> local = state.observed_by(agent_id, agent_level)
            ├─> global_info = {filtered states from other agents}
            └─> Observation(local, global_info, timestamp)

    Returns: {"battery_1": Observation(...), "battery_2": Observation(...)}
```

**Data State After Reset:**
```python
# Agent layer:
agent.state = FieldAgentState(...)  # State objects
agent.action = Action(...)          # Action objects

# Proxy layer:
proxy.state_cache["agents"]["battery_1"] = FieldAgentState(...)  # State objects!

# Returned observations:
obs = {
    "battery_1": Observation(local={filtered vectors}, global_info={...}),
    "battery_2": Observation(...)
}  # Observation objects!
```

---

### **Phase 2: Step Execution**

```
obs, rewards, terminated, truncated, info = env.step(actions)
│
└─> SystemAgent.execute(actions, proxy)
```

#### **Step 2.1: Action Application**

```
PHASE 1: Actions -> State Updates

self.act(actions, proxy)
│
├─> layer_actions(actions)
│   │
│   INPUT: actions = {
│       "battery_1": Action(c=[0.3]),  # Action objects!
│       "battery_2": Action(c=[-0.2]),
│   }
│   │
│   OUTPUT: {
│       "self": None,  # System agent has no action
│       "subordinates": {
│           "coordinator_1": {
│               "self": None,
│               "subordinates": {
│                   "battery_1": actions["battery_1"],
│                   "battery_2": actions["battery_2"],
│               }
│           }
│       }
│   }
│
├─> handle_self_action(layered["self"], proxy)
│   └─> System agent has no action, skipped
│
└─> handle_subordinate_actions(layered["subordinates"], proxy)
    │
    └─> For coordinator_1:
        │
        ├─> coordinator_1.execute(coord_actions, proxy)
        │   │
        │   └─> For battery_1:
        │       │
        │       └─> battery_1.execute(actions["battery_1"], proxy)
        │           ├─> set_action(action)
        │           │   └─> self.action.set_values(action)
        │           │       DATA: self.action.c = [0.3]
        │           │
        │           ├─> apply_action()
        │           │   └─> set_state()
        │           │       ├─> current_soc = self.state.features[0].soc  # 0.5
        │           │       ├─> delta = self.action.c[0] * 0.01  # 0.003
        │           │       ├─> new_soc = 0.5 + 0.003 = 0.503
        │           │       └─> self.state.features[0].set_values(soc=0.503)
        │           │           DATA: State modified in-place!
        │           │
        │           └─> proxy.set_local_state(self.agent_id, self.state)
        │               STORAGE: state_cache["agents"]["battery_1"] = self.state
```

**Data State After Action Application:**
```python
# Agent layer:
agent.action.c = [0.3]  # Action applied
agent.state.features[0].soc = 0.503  # State updated

# Proxy layer:
proxy.state_cache["agents"]["battery_1"] = agent.state  # Same object reference!
```

---

#### **Step 2.2: Simulation**

```
PHASE 2: Physics Simulation

global_state = proxy.get_global_states(system_agent_id, system_level)
│
DATA RETRIEVAL:
├─> For each agent in state_cache["agents"]:
│   └─> filtered = state.observed_by(system_agent_id, system_level)
└─> Returns: Dict of all filtered states (system sees everything)

updated_global_state = simulate(global_state)
│
├─> env_state = global_state_to_env_state(global_state)
│   DATA: Extract physics-relevant values from states
│
├─> updated_env_state = run_simulation(env_state)
│   PHYSICS: Apply environment dynamics
│
└─> return env_state_to_global_state(updated_env_state)
    DATA: Convert back to state format

proxy.set_global_state(updated_global_state)
│
DATA STORAGE:
└─> Update state_cache["agents"] with simulation results
```

---

#### **Step 2.3: Observation Collection**

```
PHASE 3: Collect Observations

obs = self.observe(proxy)
│
└─> For each agent in hierarchy:
    │
    └─> observation = proxy.get_observation(agent_id, agent_level)
        │
        DATA CONSTRUCTION:
        ├─> local_state = get_local_state(agent_id, agent_level)
        │   └─> state.observed_by(agent_id, agent_level)
        │   └─> Returns: {"BatteryChargeFeature": np.array([0.503, 100.0])}
        │
        ├─> global_state = get_global_states(agent_id, agent_level)
        │   └─> {other agents' filtered states}
        │
        └─> return Observation(
                local=local_state,        # Filtered vectors!
                global_info=global_state, # Filtered vectors!
                timestamp=1.0
            )

Returns: {
    "battery_1": Observation(local={...}, global_info={...}),
    "battery_2": Observation(...),
}
```

---

#### **Step 2.4: Reward Computation**

```
PHASE 4: Compute Rewards

rewards = self.compute_rewards(proxy)
│
└─> For each agent:
    │
    ├─> local_state = proxy.get_local_state(agent_id, agent_level)
    │   │
    │   DATA RETRIEVAL:
    │   state.observed_by(agent_id, agent_level)
    │   └─> Returns: {"BatteryChargeFeature": np.array([0.503, 100.0])}
    │
    └─> reward = compute_local_reward(local_state)
        │
        DATA USAGE:
        feature_vec = local_state["BatteryChargeFeature"]
        soc = feature_vec[0]  # Extract from numpy array!
        return soc  # 0.503

Returns: {
    "battery_1": 0.503,
    "battery_2": 0.498,
}
```

---

#### **Step 2.5: Result Caching & Vectorization**

```
PHASE 5: Cache Results

proxy.set_step_result(obs, rewards, terminateds, truncateds, infos)
│
STORAGE:
_step_results = {
    "obs": {"battery_1": Observation(...), ...},  # Observation objects!
    "rewards": {"battery_1": 0.503, ...},
    "terminateds": {...},
    "truncateds": {...},
    "infos": {...}
}

Return to RL algorithm:

proxy.get_step_results()
│
DATA VECTORIZATION:
obs_vectorized = {
    agent_id: observation.vector()
    for agent_id, observation in obs.items()
}
│
├─> observation.vector()
│   ├─> Flatten local: {"BatteryChargeFeature": np.array([0.503, 100.0])}
│   │   └─> [0.503, 100.0]
│   ├─> Flatten global_info
│   └─> Concatenate: np.array([0.503, 100.0, ...], dtype=float32)
│
Returns: (
    obs_vectorized: {"battery_1": np.ndarray([0.503, 100.0, ...]), ...},
    rewards,
    terminateds,
    truncateds,
    infos
)
```

**Final Data Output for RL:**
```python
obs = {
    "battery_1": np.ndarray([0.503, 100.0, ...]),  # Numpy arrays!
    "battery_2": np.ndarray([0.498, 100.0, ...])
}
rewards = {"battery_1": 0.503, "battery_2": 0.498}
```

---

## **Event-Driven Testing Mode - Complete Flow**

### **Timeline Execution with Data Transformations**

#### **t=0.0: AGENT_TICK(system_agent)**

```
SystemAgent.tick(scheduler, current_time=0.0)
│
├─> self._timestep = 0.0  # Update internal time
│
├─> Schedule subordinate ticks:
│   └─> scheduler.schedule_agent_tick("coordinator_1")
│       └─> Event scheduled at t = 0.0 + tick_interval
│
├─> Action passing to subordinates:
│   ├─> protocol.coordinate() -> compute subordinate actions
│   └─> For each coordinator:
│       └─> send_subordinate_action(coord_id, action)
│           └─> broker.publish(action_channel, Message(action))
│
├─> If has policy: Request observation
│   └─> schedule_message_delivery(
│           sender=system_agent,
│           recipient=proxy,
│           message={"get_info": "obs"},
│           delay=msg_delay
│       )
│       │
│       DATA OUTFLOW:
│       Message payload = {"get_info": "obs"}
│       Scheduled at: t = 0.0 + msg_delay
│
└─> Schedule simulation:
    └─> scheduler.schedule_simulation(system_agent, wait_interval)
        └─> Event scheduled at t = 0.0 + wait_interval
```

---

#### **t=msg_delay: MESSAGE_DELIVERY(proxy) - Observation Request**

```
ProxyAgent.message_delivery_handler()
│
├─> Receive: {"get_info": "obs"}
├─> info_type = "obs"
│
├─> obs = get_observation(system_agent, system_level)
│   │
│   DATA CONSTRUCTION:
│   ├─> global_state = {all agents' observed_by() filtered states}
│   ├─> local_state = state.observed_by(system_id, 3)
│   └─> obs = Observation(local, global_info, timestamp)
│
│   DATA TYPE: Observation object
│
├─> Serialize observation for message:
│   obs.to_dict()
│   └─> Returns: {
│           "timestamp": 0.0,
│           "local": {...},
│           "global_info": {...}
│       }
│
└─> schedule_message_delivery(
        sender=proxy,
        recipient=system_agent,
        message={"get_obs_response": {"body": obs_dict}},
        delay=msg_delay
    )
    │
    DATA OUTFLOW:
    Message contains serialized dict, not Observation object!
```

---

#### **t=2*msg_delay: MESSAGE_DELIVERY(system_agent) - Observation Response**

```
SystemAgent.message_delivery_handler()
│
├─> Receive: {"get_obs_response": {"body": obs_dict}}
│   │
│   DATA INFLOW:
│   obs_dict = {"timestamp": 0.0, "local": {...}, "global_info": {...}}
│
├─> Deserialize observation:
│   obs = Observation.from_dict(obs_dict)
│   │
│   DATA TRANSFORMATION:
│   Dict -> Observation object reconstructed
│
└─> compute_action(obs, scheduler)
    │
    ├─> action = policy.forward(observation=obs)
    │   │
    │   DATA AUTO-CONVERSION:
    │   obs.__array__() -> np.ndarray for neural network
    │   │
    │   POLICY OUTPUT:
    │   action = Action(c=[0.15])
    │
    ├─> set_action(action)
    │   DATA UPDATE: self.action.c = [0.15]
    │
    └─> schedule_action_effect(agent_id, delay=act_delay)
        └─> Event scheduled at t = 2*msg_delay + act_delay
```

---

#### **t=2*msg_delay+act_delay: ACTION_EFFECT(system_agent)**

```
action_effect_handler()
│
├─> apply_action()
│   └─> set_state()
│       ├─> Read self.action values
│       ├─> Update self.state features
│       └─> State object modified in-place
│
├─> Serialize state with metadata:
│   state_dict = self.state.to_dict(include_metadata=True)
│   │
│   DATA TRANSFORMATION:
│   State object -> Dict with metadata
│
└─> schedule_message_delivery(
        message={"set_state": "local", "body": state_dict}
    )
```

---

#### **t=coordinator_tick: AGENT_TICK(coordinator_1)**

```
CoordinatorAgent.tick(scheduler, current_time)
│
├─> Receive upstream actions:
│   upstream_actions = receive_upstream_actions()
│   └─> broker.consume(action_channel)
│   └─> Returns: [Action(...)] from system_agent
│
├─> Determine action source:
│   ├─> If upstream_actions not empty:
│   │   └─> action = upstream_actions[0]  # Use parent's decision!
│   └─> Else:
│       └─> action = policy.forward(obs)  # Own decision
│
├─> set_action(action)
│
├─> Protocol.coordinate() -> compute field agent actions
│
├─> Send actions to subordinates:
│   └─> For each field_agent:
│       └─> send_subordinate_action(field_id, field_action)
│           └─> broker.publish(action_channel, Message)
│
└─> Schedule subordinate ticks
```

---

#### **t=field_tick: AGENT_TICK(field_1)**

```
FieldAgent.tick(scheduler, current_time)
│
├─> Receive upstream actions:
│   upstream_actions = receive_upstream_actions()
│   └─> broker.consume(action_channel)
│   └─> Returns: [Action(...)] from coordinator_1
│
├─> Determine action source:
│   ├─> If upstream_actions not empty:
│   │   └─> action = upstream_actions[0]  # Use parent's decision!
│   └─> Else:
│       └─> action = policy.forward(obs)  # Own decision
│
├─> set_action(action)
│
└─> schedule_action_effect(delay=act_delay)
```

---

#### **Reward Computation (After State Updates)**

```
Agent receives local state -> Compute tick results

message_delivery_handler() - "get_local_state_response"
│
├─> Receive: {"get_local_state_response": {"body": local_state}}
│   │
│   DATA INFLOW:
│   local_state = {"BatteryChargeFeature": np.array([0.503, 100.0])}
│
├─> Compute tick result:
│   tick_result = {
│       "reward": compute_local_reward(local_state),
│       "terminated": is_terminated(local_state),
│       "truncated": is_truncated(local_state),
│       "info": get_local_info(local_state)
│   }
│   │
│   DATA USAGE:
│   feature_vec = local_state["BatteryChargeFeature"]
│   soc = feature_vec[0]  # Extract from numpy array
│   return soc  # 0.503
│
└─> schedule_message_delivery(
        message={"set_tick_result": "local", "body": tick_result}
    )
```

---

## **ProxyAgent State Cache Architecture**

### **Cache Structure**

```python
proxy.state_cache = {
    # Per-agent states (State objects!)
    "agents": {
        "battery_1": FieldAgentState(
            owner_id="battery_1",
            owner_level=1,
            features=[BatteryChargeFeature(soc=0.503, capacity=100.0)]
        ),
        "battery_2": FieldAgentState(...),
        "coordinator_1": CoordinatorAgentState(...),
        "system_agent": SystemAgentState(...)
    },

    # Global state (environment-wide data)
    "global": {
        "grid_frequency": 60.0,
        "total_load": 1500.0,
        ...
    }
}

# Agent levels tracked for visibility checks:
proxy._agent_levels = {
    "battery_1": 1,
    "battery_2": 1,
    "coordinator_1": 2,
    "system_agent": 3
}
```

### **State Cache Operations**

#### **SET Operations**

| Method | Input Data | Transformation | Storage Location |
|--------|-----------|----------------|------------------|
| `set_local_state(aid, state)` | `agent_id: str`, `state: State` | None (stores object!) | `state_cache["agents"][aid]` |
| `set_global_state(dict)` | `global_dict: Dict` | None | `state_cache["global"].update(...)` |

#### **GET Operations**

| Method | Retrieval Source | Filtering | Return Type |
|--------|-----------------|-----------|-------------|
| `get_local_state(aid, level)` | `state_cache["agents"][aid]` | `state.observed_by()` | `Dict[str, np.ndarray]` |
| `get_global_states(aid, level)` | `state_cache["agents"]` (all) | `state.observed_by()` | `Dict[str, Dict[str, np.ndarray]]` |
| `get_observation(aid, level)` | Both local + global | Feature visibility | `Observation` |

---

## **Protocol-Based Coordination**

### **Protocol Architecture**

```python
class Protocol(ABC):
    """Combines communication and action coordination."""

    @abstractmethod
    def coordinate(
        self,
        coordinator_state: State,
        coordinator_action: Optional[Action],
        info_for_subordinates: Dict[str, Observation],
        context: Dict
    ) -> Tuple[Dict[str, Dict], Dict[str, Action]]:
        """
        Returns:
            messages: Dict of coordination messages per subordinate
            actions: Dict of actions per subordinate
        """
        pass
```

### **Protocol Implementations**

#### **1. NoProtocol (Default)**
- No coordination messages
- No action decomposition
- Each agent acts independently

#### **2. VerticalProtocol (Hierarchical)**
- Parent coordinates subordinates
- Uses feature-level visibility rules
- Action decomposition from parent to children

```python
class VerticalProtocol(Protocol):
    def coordinate(self, coordinator_state, coordinator_action, info, context):
        # Compute subordinate-specific messages
        messages = {}
        for sub_id, sub_obs in info.items():
            messages[sub_id] = self._compute_message(coordinator_state, sub_obs)

        # Decompose coordinator action into subordinate actions
        actions = {}
        for sub_id in info.keys():
            actions[sub_id] = self._decompose_action(coordinator_action, sub_id)

        return messages, actions
```

#### **3. HorizontalProtocol (Peer Coordination)**
- Peer-to-peer communication
- Lateral action coordination
- Consensus-based decisions

### **Protocol Usage in Execution**

```
CoordinatorAgent.act(actions, proxy)
│
├─> Get observation and local state
│   obs = proxy.get_observation(self.agent_id, self.level)
│   local_state = proxy.get_local_state(self.agent_id, self.level)
│
├─> Get subordinate info
│   info = {sub_id: proxy.get_observation(sub_id, sub_level) for sub_id in subs}
│
├─> Execute protocol
│   messages, sub_actions = self.protocol.coordinate(
│       coordinator_state=self.state,
│       coordinator_action=action,
│       info_for_subordinates=info,
│       context={"timestamp": self._timestep}
│   )
│
├─> Send coordination messages
│   for sub_id, message in messages.items():
│       self.send_info(broker, sub_id, message)
│
└─> Send/pass subordinate actions
    for sub_id, sub_action in sub_actions.items():
        self.send_subordinate_action(sub_id, sub_action)
```

---

## **Complete Data Flow Diagrams**

### **CTDE Training Step - Data Transformations**

```
┌─────────────────────────────────────────────────────────────────┐
│ RL Algorithm Provides Actions                                   │
├─────────────────────────────────────────────────────────────────┤
│ actions = {                                                     │
│     "battery_1": Action(c=[0.3]),  <- Action objects            │
│     "battery_2": Action(c=[-0.2])                               │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓ env.step(actions)
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Action Application                                     │
├─────────────────────────────────────────────────────────────────┤
│ battery_1.set_action(Action(c=[0.3]))                           │
│     ↓ set_values()                                              │
│ self.action.c = [0.3]  <- Action object updated                 │
│                                                                 │
│ battery_1.apply_action()                                        │
│     ↓ set_state()                                               │
│ self.state.features[0].soc = 0.503  <- State object updated     │
│                                                                 │
│ proxy.set_local_state("battery_1", state)                       │
│     ↓ Direct storage                                            │
│ state_cache["agents"]["battery_1"] = state  <- State object!    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Simulation                                             │
├─────────────────────────────────────────────────────────────────┤
│ global_state = proxy.get_global_states("system_agent", 3)       │
│     ↓ observed_by() filtering                                   │
│ {filtered states as feature vectors}                            │
│                                                                 │
│ env_state = global_state_to_env_state(global_state)             │
│ updated_env_state = run_simulation(env_state)                   │
│ updated_global = env_state_to_global_state(updated_env_state)   │
│                                                                 │
│ proxy.set_global_state(updated_global)                          │
│     ↓ Update state objects                                      │
│ state_cache updated with simulation results                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Observation Collection                                 │
├─────────────────────────────────────────────────────────────────┤
│ obs = proxy.get_observation("battery_1", level=1)               │
│     ↓ observed_by() filtering                                   │
│ local = {"BatteryChargeFeature": np.array([0.503, 100.0])}      │
│ global = {filtered states from other agents}                    │
│     ↓ Wrapping                                                  │
│ Observation(local=..., global_info=...)  <- Object              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: Reward Computation                                     │
├─────────────────────────────────────────────────────────────────┤
│ local_state = proxy.get_local_state("battery_1", level=1)       │
│     ↓ observed_by() filtering                                   │
│ {"BatteryChargeFeature": np.array([0.503, 100.0])}              │
│                                                                 │
│ compute_local_reward(local_state)                               │
│     ↓ Array access                                              │
│ soc = local_state["BatteryChargeFeature"][0]                    │
│ return 0.503                                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: Vectorization for RL                                   │
├─────────────────────────────────────────────────────────────────┤
│ obs_dict = {"battery_1": Observation(...)}                      │
│     ↓ .vector()                                                 │
│ obs_vec = {"battery_1": np.array([0.503, 100.0, ...])}          │
│     ↓ Return to RL                                              │
│ RL algorithm receives numpy arrays!                             │
└─────────────────────────────────────────────────────────────────┘
```

---

### **Event-Driven Mode - Action Passing Timeline**

```
t=0.0: AGENT_TICK(system_agent)
├─> Compute system action via policy
├─> Protocol.coordinate() -> subordinate actions
├─> send_subordinate_action(coord_1, action_for_coord)
│   └─> broker.publish(action_channel)
└─> schedule_agent_tick(coord_1)

t=coord_tick: AGENT_TICK(coordinator_1)
├─> receive_upstream_actions() from broker
│   └─> [Action from system_agent]
├─> Use upstream action (priority over policy!)
├─> Protocol.coordinate() -> field actions
├─> send_subordinate_action(field_1, action_for_field)
│   └─> broker.publish(action_channel)
└─> schedule_agent_tick(field_1)

t=field_tick: AGENT_TICK(field_1)
├─> receive_upstream_actions() from broker
│   └─> [Action from coordinator_1]
├─> Use upstream action (priority over policy!)
├─> set_action(upstream_action)
└─> schedule_action_effect(delay=act_delay)

t=field_tick+act_delay: ACTION_EFFECT(field_1)
├─> apply_action() -> Update self.state
└─> schedule_message_delivery to proxy
```

---

## **Key Design Patterns**

### **1. Object-Dict Duality**

**Agent Layer:**
- Works with rich objects (State, Action, Observation)
- Has methods: `.vector()`, `.set_values()`, `.update()`
- Type-safe with IDE support

**Proxy Layer:**
- Stores State objects directly
- Returns filtered vectors via `observed_by()`
- Serialization only at message boundaries

**Boundary:**
- **Agent -> Proxy (CTDE)**: Pass State object directly
- **Agent -> Proxy (Event-Driven)**: Call `.to_dict()` -> Message -> `from_dict()`
- **Proxy -> Agent**: Return filtered feature vectors

---

### **2. Serialization Points**

| Crossing Point | Data Type | Transformation |
|----------------|-----------|----------------|
| Agent -> Proxy (CTDE) | State object | None (direct reference) |
| Agent -> Message | State object | `.to_dict(include_metadata=True)` |
| Agent -> Message | Observation object | `.to_dict()` |
| Message -> Agent | Observation dict | `Observation.from_dict()` |
| Proxy -> Agent | State object | `state.observed_by()` -> filtered vectors |
| Observation -> Policy | Observation object | `.__array__()` auto-converts |
| Agent -> Message (Action) | Action object | `action.to_dict()` or direct |
| Message -> Agent (Action) | Action dict | `Action.from_dict()` or direct |

---

### **3. Data Consistency Rules**

**DO:**
- Use Action objects throughout (no dict conversion in CTDE)
- Pass State objects to proxy in CTDE mode
- Convert State -> Dict only when sending messages (with `include_metadata=True`)
- Use `local_state` parameter (numpy array dict) in reward/info methods
- Access features as vectors: `local_state["FeatureName"][0]` for first element
- Use upstream actions when available (priority over policy)

**DON'T:**
- Send State objects in messages (must serialize to dict first)
- Access dict fields like `local_state["Feature"]["soc"]` (it's now a numpy array!)
- Access `self.state` in `compute_local_reward()` (use filtered parameter)
- Ignore upstream actions (they have priority!)

---

### **4. Action Priority**

```python
def determine_action(self):
    # Priority 1: Upstream action from parent
    upstream_actions = self.receive_upstream_actions()
    if upstream_actions:
        return upstream_actions[0]

    # Priority 2: Policy-computed action
    if self.policy:
        obs = self.proxy.get_observation(self.agent_id, self.level)
        return self.policy.forward(observation=obs)

    # Priority 3: No action
    return None
```

---

### **5. Type Consistency Summary**

- **State**: Object (agent) -> **Object (proxy)** -> **Filtered vectors (via observed_by())**
- **Action**: Object (throughout, passed via broker in event-driven mode)
- **Observation**: Object -> Dict (messages) -> Object (reconstruction)
- **Feature**: Object (in State) -> Stays in State object -> Vector (in observations)
- **Messages**: Always serialized (dicts)

---

## **Complete Example Trace**

### **Single Agent, Single Step - All Data Transformations**

```
=== INITIALIZATION ===
agent.init_state():
    CREATE: FieldAgentState(owner_id="battery_1", owner_level=1)
    CREATE: BatteryChargeFeature(soc=0.5, capacity=100.0)
    STORE: self.state.features = [feature]
    TYPE: State object

proxy.set_local_state("battery_1", state):
    STORE: state_cache["agents"]["battery_1"] = state
    TYPE: State object (direct reference!)

=== STEP EXECUTION ===
env.step({"battery_1": Action(c=[0.3])}):

1. Action Application:
    INPUT: Action(c=[0.3])
    TYPE: Action object

    set_action(action):
        PROCESS: self.action.set_values(action)
        UPDATE: self.action.c = [0.3]
        TYPE: Action object (internal)

    apply_action():
        READ: self.action.c[0] = 0.3
        READ: self.state.features[0].soc = 0.5
        COMPUTE: new_soc = 0.5 + 0.3*0.01 = 0.503
        UPDATE: self.state.features[0].soc = 0.503
        TYPE: State object (modified)

    proxy.set_local_state("battery_1", state):
        STORE: state_cache["agents"]["battery_1"] = state
        TYPE: State object (same reference, updated!)

2. Observation Collection:
    proxy.get_observation("battery_1", level=1):
        FILTER: state.observed_by("battery_1", 1)
        CONSTRUCT: Observation(
            local={"BatteryChargeFeature": np.array([0.503, 100.0])},
            global_info={...filtered states...}
        )
        TYPE: Observation object

3. Reward Computation:
    proxy.get_local_state("battery_1", level=1):
        FILTER: state.observed_by("battery_1", 1)
        RETURN: {"BatteryChargeFeature": np.array([0.503, 100.0])}
        TYPE: Dict[str, np.ndarray]

    compute_local_reward(local_state):
        ACCESS: local_state["BatteryChargeFeature"][0]
        RETURN: 0.503
        TYPE: float

4. Vectorization for RL:
    proxy.get_step_results():
        INPUT: obs = {"battery_1": Observation(...)}
        TRANSFORM: observation.vector() for each
        OUTPUT: {"battery_1": np.array([0.503, 100.0, ...])}
        TYPE: Dict[str, np.ndarray]

=== RETURN TO RL ===
    obs: Dict[str, np.ndarray]     <- Numpy arrays
    rewards: Dict[str, float]      <- Scalars
    terminated: Dict[str, bool]
    truncated: Dict[str, bool]
    info: Dict[str, Dict]
```

---

## **Critical Takeaways**

### **1. Proxy Storage is State Objects**
- State objects stored directly in `state_cache["agents"]`
- Enables feature-level visibility filtering via `state.observed_by()`
- Maintains full type information (FeatureProvider instances)
- Serialization needed only for message passing

### **2. Action Passing is Hierarchical**
- Parent agents can send actions to subordinates via broker
- Upstream actions have priority over policy-computed actions
- Enables centralized control strategies
- Works in both CTDE and event-driven modes

### **3. Visibility Filtering is Active**
- `state.observed_by()` filters features based on visibility rules
- Agents only see features they're authorized to see
- Returns numpy arrays (vectors), not dicts
- Enforced at both local and global state retrieval

### **4. Message Passing Requires Serialization**
- Observation: `.to_dict()` -> send -> `from_dict()` -> reconstruct
- State: `.to_dict(include_metadata=True)` -> send -> `from_dict()` -> reconstruct
- Action: Can be passed directly in CTDE, serialized in event-driven

### **5. Type Consistency**
- **State**: Object (agent) -> Object (proxy) -> Filtered vectors (retrieval)
- **Action**: Object (throughout, no proxy storage)
- **Observation**: Object -> Dict (messages) -> Object (reconstruction)
- **Feature**: Object (in State) -> Vector (in observations)

---

This documentation reflects the **complete version** of the HERON framework with:
- Feature visibility enforcement
- State object storage in proxy
- Hierarchical action passing
- Protocol-based coordination
- Message broker integration
- Both CTDE and event-driven execution modes

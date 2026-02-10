# HERON Complete Data Flow Documentation

This document provides a comprehensive analysis of how data flows through the HERON framework in both CTDE Training and Event-Driven Testing modes.

---

## **Table of Contents**

1. [Core Data Types](#core-data-types)
2. [State Data Flow](#state-data-flow)
3. [Action Data Flow](#action-data-flow)
4. [Observation Data Flow](#observation-data-flow)
5. [Feature Data Flow](#feature-data-flow)
6. [CTDE Training Mode - Complete Flow](#ctde-training-mode---complete-flow)
7. [Event-Driven Testing Mode - Complete Flow](#event-driven-testing-mode---complete-flow)
8. [ProxyAgent State Cache Architecture](#proxyagent-state-cache-architecture)

---

## **Core Data Types**

### **Type Transformation Rules**

| Component | Internal Representation | Storage in Proxy | Message Passing |
|-----------|------------------------|------------------|-----------------|
| **State** | State object (FieldAgentState, etc.) | **State object** (NEW!) | Dict with metadata |
| **Action** | Action object | Not stored in proxy | Action object (no serialization) |
| **Observation** | Observation object | Not stored (computed on-demand) | Dict via `obs.to_dict()` |
| **Feature** | FeatureProvider object | Part of State object | Dict via `feature.to_dict()` |

### **Key Principle (Updated!)**
**ProxyAgent now stores State objects directly** - This enables feature-level visibility filtering:
- ✅ State objects maintained with full type information
- ✅ Direct access to `state.observed_by()` for visibility filtering
- ✅ Features maintain class identity (FeatureProvider instances)
- ⚠️ Serialization still needed at message boundaries (State ↔ Dict)

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
- `self.state` → State object with features

---

#### **2. Set State (Storage in Proxy)**

**CTDE Mode:**
```
Agent updates state → Send to proxy

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
Agent updates state → Serialize → Send message → Proxy stores

action_effect_handler()
├─> apply_action() → Updates self.state
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
├─> Extract: state_dict = body["features"]
└─> Call: set_local_state(agent_id, state_dict)
    └─> Storage: state_cache["agents"]["battery_1"] = state_dict
```

**Data Transformations:**
```
State object → .to_dict() → {"FeatureName": {field: value}}
                          ↓
            proxy.state_cache["agents"][agent_id]
```

---

#### **3. Get State (Retrieval from Proxy)**

**CTDE Mode:**
```
Agent requests own state from proxy

proxy.get_local_state(agent_id, protocol)
├─> Input: agent_id="battery_1"
├─> Retrieval: state_obj = state_cache["agents"]["battery_1"]  # State object!
├─> Apply visibility filtering: state_obj.observed_by(agent_id, requestor_level)
└─> Return: {"BatteryChargeFeature": np.array([0.53, 100.0])}  # Feature vectors!
```

**Event-Driven Mode:**
```
Agent sends message requesting state → Proxy responds

Agent:
├─> schedule_message_delivery(message={"get_info": "local_state"})
└─> Sent to proxy

Proxy.message_delivery_handler():
├─> Receives request
├─> local_state = self.get_local_state(sender_id, protocol)
│   └─> Returns dict from state_cache["agents"][sender_id]
├─> Send response: {"get_local_state_response": {"body": local_state}}
└─> Agent receives dict in message
```

**Data Outflow:**
```
state_cache["agents"][agent_id] → Dict → Returned to agent
                                        ↓
                        Agent uses dict to compute rewards/info
```

**Usage Example (NEW - With Visibility Filtering):**
```python
# Agent receives feature vectors from proxy (after visibility filtering)
local_state = proxy.get_local_state(self.agent_id)
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
│ init_state() → self.state = FieldAgentState(...)                │
│                self.state.features = [BatteryChargeFeature(...)] │
│                                                                  │
│ apply_action() → self.state.features[0].soc = new_value         │
│                                                                  │
│ Send to proxy: proxy.set_local_state(aid, state)  # State object! │
└─────────────────────────────────────────────────────────────────┘
                            ↓ (No conversion!)
┌─────────────────────────────────────────────────────────────────┐
│ Proxy Layer (State Object Storage - NEW!)                      │
├─────────────────────────────────────────────────────────────────┤
│ state_cache["agents"]["battery_1"] = FieldAgentState(...)       │
│     └─> Contains: [BatteryChargeFeature(soc=0.53, capacity=100)]│
│                                                                  │
│ Stores FULL State objects with FeatureProvider instances!       │
└─────────────────────────────────────────────────────────────────┘
                            ↓ get_local_state() + visibility filtering
┌─────────────────────────────────────────────────────────────────┐
│ Visibility Filtering Layer (state.observed_by())               │
├─────────────────────────────────────────────────────────────────┤
│ state_obj.observed_by(requestor_id, requestor_level)            │
│     ↓ Feature-level filtering                                   │
│ Returns: {"BatteryChargeFeature": np.array([0.53, 100.0])}      │
│          └─> Only visible features, as numpy arrays!            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Agent Layer (Filtered Vector Usage)                            │
├─────────────────────────────────────────────────────────────────┤
│ local_state = proxy.get_local_state(aid) → Filtered vectors     │
│                                                                  │
│ compute_local_reward(local_state):                              │
│     feature_vec = local_state["BatteryChargeFeature"]  # array  │
│     soc = feature_vec[0]  # Extract from vector                 │
│     return soc                                                   │
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
- `self.action` → Action object with specs

---

#### **2. Set Action (Policy Decision or External)**

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
Policy → Action object → set_values() → self.action updated
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
self.action (Action object) → Extract values → Update self.state (State object)
```

---

### **Action Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Policy Layer                                                     │
├─────────────────────────────────────────────────────────────────┤
│ policy.forward(observation) → Returns Action object             │
│     action = Action()                                            │
│     action.set_specs(dim_c=1, range=([-1,1]))                   │
│     action.set_values(np.array([0.3]))                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Training Loop / Environment Layer                               │
├─────────────────────────────────────────────────────────────────┤
│ actions[agent_id] = action  # Action object                     │
│ env.step(actions)                                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Agent Layer (Execution)                                         │
├─────────────────────────────────────────────────────────────────┤
│ set_action(action) → self.action.set_values(action)             │
│                                                                  │
│ apply_action() → Read self.action.c[0]                          │
│                → Update self.state.features[0].soc              │
└─────────────────────────────────────────────────────────────────┘
```

**Important: Actions are NEVER stored in proxy!** They exist only within agents during the act() phase.

---

## **Observation Data Flow**

### **Observation Object Structure**

```python
obs = Observation(
    local={"BatteryChargeFeature": {"soc": 0.53, "capacity": 100.0}},
    global_info={"agent_states": {...}},
    timestamp=0.0
)

# Observation attributes:
obs.local = dict        # Agent-specific state
obs.global_info = dict  # Global/shared state
obs.timestamp = float   # Current time
```

### **Observation Operations**

#### **1. Build Observation (Construction from Proxy State)**

```
proxy.get_observation(sender_id, protocol)
├─> global_state = get_global_states(sender_id, protocol)
│   ├─> Retrieval: state_cache["global"]
│   ├─> Filtering: Apply visibility rules based on sender_id
│   │   └─> SystemAgent sees full state
│   │   └─> Other agents see filtered state
│   └─> Returns: Dict
│
├─> local_state = get_local_state(sender_id, protocol)
│   ├─> Retrieval: state_cache["agents"][sender_id]
│   └─> Returns: Dict
│
└─> return Observation(
        local=local_state,        # Dict!
        global_info=global_state, # Dict!
        timestamp=self._timestep
    )
```

**Data Inflow:**
```
state_cache["global"] → Dict → obs.global_info
state_cache["agents"][aid] → Dict → obs.local
```

**Key Insight:** Observation wraps **dicts**, not State objects!

---

#### **2. Vectorize Observation (For RL Algorithms)**

```
Observation → numpy array conversion

obs.vector()
├─> Flatten obs.local dict
│   └─> _flatten_dict_to_list({"BatteryChargeFeature": {"soc": 0.53, "capacity": 100.0}})
│       ├─> Sort keys alphabetically
│       ├─> Extract values recursively
│       └─> Returns: [np.array([0.53]), np.array([100.0])]
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
        "local": {"BatteryChargeFeature": {"soc": 0.53, ...}},
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
    └─> obs.local = obs_dict["local"]
    └─> obs.global_info = obs_dict["global_info"]
    └─> obs.timestamp = obs_dict["timestamp"]
```

---

### **Observation Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Proxy State Cache (Dicts)                                       │
├─────────────────────────────────────────────────────────────────┤
│ state_cache["agents"]["battery_1"] = {                          │
│     "BatteryChargeFeature": {"soc": 0.53, "capacity": 100.0}    │
│ }                                                                │
│ state_cache["global"] = {...}                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓ get_observation()
┌─────────────────────────────────────────────────────────────────┐
│ Observation Construction                                        │
├─────────────────────────────────────────────────────────────────┤
│ Observation(                                                     │
│     local={"BatteryChargeFeature": {...}},  # Dict from cache   │
│     global_info={...},                       # Dict from cache   │
│     timestamp=0.0                                                │
│ )                                                                │
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

# Feature object attributes:
feature.feature_name = "BatteryChargeFeature"  # Auto-set from class name
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
│
├─> self.state = FieldAgentState(owner_id="battery_1", owner_level=1)
├─> self.state.features.append(battery_feature)
└─> State contains list of FeatureProvider objects
```

**Data Outflow:**
```
FeatureProvider object → Appended to state.features list
```

---

#### **2. Feature Serialization (State → Dict)**

```
state.to_dict()
├─> For each feature in self.features:
│   ├─> feature_name = feature.feature_name  # "BatteryChargeFeature"
│   ├─> feature_dict = feature.to_dict()
│   │   └─> Returns: {"soc": 0.5, "capacity": 100.0}
│   └─> feature_dict[feature_name] = feature_dict
│
└─> Returns: {
        "BatteryChargeFeature": {
            "soc": 0.5,
            "capacity": 100.0
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
        "BatteryChargeFeature": np.array([0.5, 100.0])  # Only visible features as vectors!
    }
```

**NEW:** Now actively used in `proxy.get_local_state()` and `proxy.get_global_states()` - visibility filtering enforced!

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
│   │   ├─> Check "public": ✓ True
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
│       │   ├─> Check "public": ✓ True
│       │   └─> Include: array([0.498, 100.0])
│       │
│       Returns: {"BatteryChargeFeature": array([0.498, 100.0])}
│
│   For zone_1 (coordinator):
│       [If coordinator had features with visibility=["owner"], battery_1 can't see them]
│       filtered = {}  # Nothing visible to battery_1
│       └─> Not included in global_info
│
└─> Return: Observation(
        local={"BatteryChargeFeature": array([0.503, 100.0])},  # Own state
        global_info={"battery_2": {"BatteryChargeFeature": array([0.498, 100.0])}},  # Public states only
        timestamp=1.0
    )
```

**Visibility Enforcement:**
- ✅ battery_1 sees own state (owner visibility)
- ✅ battery_1 sees battery_2 state (public visibility)
- ✅ battery_1 does NOT see coordinator private features (filtered out)
- ✅ battery_1 does NOT see system-only features (filtered out)

---

### **Feature Lifecycle Summary**

```
┌─────────────────────────────────────────────────────────────────┐
│ Feature Definition (Class)                                      │
├─────────────────────────────────────────────────────────────────┤
│ class BatteryChargeFeature(FeatureProvider):                    │
│     visibility = ["public"]                                      │
│     soc: float = 0.5                                             │
│     capacity: float = 100.0                                      │
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
│                                                                  │
│ state.vector() → Concatenates all feature.vector() outputs      │
└─────────────────────────────────────────────────────────────────┘
                            ↓ .to_dict()
┌─────────────────────────────────────────────────────────────────┐
│ Serialization for Proxy Storage                                │
├─────────────────────────────────────────────────────────────────┤
│ {                                                                │
│     "BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}     │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
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
    │   └─> Create ProxyAgent()
    │
    ├─> _initialize_agents()
    │   └─> For each agent (system → coord → field):
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
    │           └─> proxy.set_local_state(agent_id, agent_state.to_dict())
    │               │
    │               DATA TRANSFORMATION:
    │               State object → .to_dict()
    │               └─> {"BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}}
    │
    │               STORAGE:
    │               state_cache["agents"]["battery_1"] = state_dict
    │
    └─> proxy.init_global_state()
        └─> Compile all agent states into global state
            │
            DATA COMPILATION:
            state_cache["global"]["agent_states"] = {
                "battery_1": state_cache["agents"]["battery_1"],  # Dict reference
                "battery_2": state_cache["agents"]["battery_2"],
                ...
            }
```

**Data State After Init:**
```python
proxy.state_cache = {
    "agents": {
        "battery_1": {"BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}},
        "battery_2": {"BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}},
    },
    "global": {
        "agent_states": {
            "battery_1": <reference to agents["battery_1"]>,
            "battery_2": <reference to agents["battery_2"]>,
        }
    }
}
```

---

### **Phase 1: Reset**

```
obs, info = env.reset(seed=42)
│
├─> scheduler.reset() → Clear events
├─> broker.clear() → Clear messages
├─> proxy.reset() → state_cache = {}  [CLEARED!]
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
│   ├─> proxy.set_local_state(self.agent_id, self.state.to_dict())
│   │   │
│   │   DATA OUTFLOW:
│   │   self.state (State object)
│   │       ↓ .to_dict()
│   │   state_dict = {"FeatureName": {...}}
│   │       ↓
│   │   state_cache["agents"]["system_agent"] = state_dict
│   │
│   └─> For each subordinate: subordinate.reset(seed, proxy)
│       └─> [Recursive reset, all update proxy cache]
│
├─> proxy.init_global_state()
│   │
│   DATA COMPILATION:
│   state_cache["global"]["agent_states"] = {
│       "battery_1": state_cache["agents"]["battery_1"],
│       "battery_2": state_cache["agents"]["battery_2"],
│       ...
│   }
│
└─> return self.observe(proxy)
    │
    DATA CONSTRUCTION:
    For each agent:
        obs[agent_id] = proxy.get_observation(agent_id)
            ├─> local = state_cache["agents"][agent_id]
            ├─> global_info = state_cache["global"]
            └─> Observation(local, global_info, timestamp)

    Returns: {"battery_1": Observation(...), "battery_2": Observation(...)}
```

**Data State After Reset:**
```python
# Agent layer:
agent.state = FieldAgentState(...)  # State objects
agent.action = Action(...)          # Action objects

# Proxy layer:
proxy.state_cache["agents"]["battery_1"] = {
    "BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}
}  # Dicts!

# Returned observations:
obs = {
    "battery_1": Observation(local={...}, global_info={...}),
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
PHASE 1: Actions → State Updates

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
│           "battery_1": actions["battery_1"],  # Action object
│           "battery_2": actions["battery_2"],
│       }
│   }
│
├─> handle_self_action(layered["self"], proxy)
│   └─> System agent has no action, skipped
│
└─> handle_subordinate_actions(layered["subordinates"], proxy)
    └─> For battery_1:
        │
        └─> battery_1.execute(actions["battery_1"], proxy)
            ├─> set_action(action)
            │   └─> self.action.set_values(action)
            │       │
            │       DATA INFLOW:
            │       action = Action(c=[0.3])
            │           ↓ .set_values(action)
            │       self.action.c = [0.3]  # Updated!
            │
            ├─> apply_action()
            │   └─> set_state()
            │       ├─> current_soc = self.state.features[0].soc  # 0.5
            │       ├─> delta = self.action.c[0] * 0.01  # 0.3 * 0.01 = 0.003
            │       ├─> new_soc = 0.5 + 0.003 = 0.503
            │       └─> self.state.features[0].set_values(soc=0.503)
            │           │
            │           DATA UPDATE:
            │           self.state.features[0].soc = 0.503  # Modified in-place!
            │
            └─> proxy.set_local_state(self.agent_id, self.state.to_dict())
                │
                DATA TRANSFORMATION:
                self.state (State object)
                    ↓ .to_dict()
                {"BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}}
                    ↓
                STORAGE:
                state_cache["agents"]["battery_1"] = state_dict
```

**Data State After Action Application:**
```python
# Agent layer:
agent.action.c = [0.3]  # Action applied
agent.state.features[0].soc = 0.503  # State updated

# Proxy layer:
proxy.state_cache["agents"]["battery_1"] = {
    "BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}
}  # Updated dict!
```

---

#### **Step 2.2: Simulation**

```
PHASE 2: Physics Simulation

global_state = proxy.get_global_states(system_agent_id, protocol)
│
DATA RETRIEVAL:
├─> Returns: state_cache["global"]
└─> Contains: {
        "agent_states": {
            "battery_1": {"BatteryChargeFeature": {"soc": 0.503, ...}},
            "battery_2": {"BatteryChargeFeature": {"soc": 0.498, ...}}
        }
    }

updated_global_state = simulate(global_state)
│
├─> env_state = global_state_to_env_state(global_state)
│   │
│   DATA EXTRACTION:
│   agent_states = global_state["agent_states"]  # Access nested dict
│   state_dict = agent_states["battery_1"]
│   soc = state_dict["BatteryChargeFeature"]["soc"]  # 0.503
│   │
│   OUTPUT: EnvState(battery_soc=0.503)
│
├─> updated_env_state = run_simulation(env_state)
│   │
│   PHYSICS:
│   env_state.battery_soc = np.clip(env_state.battery_soc, 0.0, 1.0)
│   └─> Still 0.503 (already in bounds)
│
└─> return env_state_to_global_state(updated_env_state)
    │
    DATA CONSTRUCTION:
    Create state dicts for each agent:
    battery1_state_dict = {
        "BatteryChargeFeature": {
            "soc": 0.503,  # From simulation
            "capacity": 100.0
        }
    }
    │
    OUTPUT: {
        "agent_states": {
            "battery_1": battery1_state_dict,
            "battery_2": battery2_state_dict,
        }
    }

proxy.set_global_state(updated_global_state)
│
DATA STORAGE:
state_cache["global"].update(updated_global_state)
└─> state_cache["global"]["agent_states"]["battery_1"] = {
        "BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}
    }  # Updated!
```

---

#### **Step 2.3: Observation Collection**

```
PHASE 3: Collect Observations

obs = self.observe(proxy)
│
└─> For each agent in hierarchy:
    │
    └─> observation = proxy.get_observation(agent_id, protocol)
        │
        DATA CONSTRUCTION:
        ├─> global_state = state_cache["global"]
        │   └─> {"agent_states": {...}}
        │
        ├─> local_state = state_cache["agents"]["battery_1"]
        │   └─> {"BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}}
        │
        └─> return Observation(
                local=local_state,        # Dict!
                global_info=global_state, # Dict!
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
    ├─> local_state = proxy.get_local_state(agent_id, protocol)
    │   │
    │   DATA RETRIEVAL:
    │   state_cache["agents"]["battery_1"]
    │   └─> Returns: {"BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}}
    │
    └─> reward = compute_local_reward(local_state)
        │
        DATA USAGE:
        local_state = {"BatteryChargeFeature": {"soc": 0.503, ...}}
        soc = local_state["BatteryChargeFeature"]["soc"]  # Extract from dict!
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
│   ├─> Flatten local: {"BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}}
│   │   └─> [0.503, 100.0]
│   ├─> Flatten global_info: {...}
│   │   └─> [additional values]
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
│   └─> scheduler.schedule_agent_tick("battery_1")
│       └─> Event scheduled at t = 0.0 + tick_interval (5.0) = 5.0
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
│       Message payload = {"get_info": "obs"}  # Simple string
│       Scheduled at: t = 0.0 + 0.2 = 0.2
│
└─> Schedule simulation:
    └─> scheduler.schedule_simulation(system_agent, wait_interval)
        └─> Event scheduled at t = 0.0 + 0.01 = 0.01
```

---

#### **t=0.2: MESSAGE_DELIVERY(proxy) - Observation Request**

```
ProxyAgent.message_delivery_handler()
│
├─> Receive: {"get_info": "obs"}
├─> info_type = "obs"
│
├─> info = get_observation(system_agent, protocol)
│   │
│   DATA CONSTRUCTION:
│   ├─> global_state = state_cache["global"]
│   │   └─> {"agent_states": {"battery_1": {...}, ...}}
│   │
│   ├─> local_state = state_cache["agents"]["system_agent"]
│   │   └─> {} (system agent has no local state)
│   │
│   └─> obs = Observation(
│           local={},
│           global_info={"agent_states": {...}},
│           timestamp=0.0
│       )
│
│   DATA TYPE: Observation object
│
├─> Serialize observation for message:
│   obs.to_dict()
│   └─> Returns: {
│           "timestamp": 0.0,
│           "local": {},
│           "global_info": {"agent_states": {...}}
│       }
│
│   DATA TRANSFORMATION:
│   Observation object → Dict
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
    Scheduled at: t = 0.2 + 0.2 = 0.4
```

---

#### **t=0.4: MESSAGE_DELIVERY(system_agent) - Observation Response**

```
SystemAgent.message_delivery_handler()
│
├─> Receive: {"get_obs_response": {"body": obs_dict}}
│   │
│   DATA INFLOW:
│   obs_dict = {
│       "timestamp": 0.0,
│       "local": {},
│       "global_info": {"agent_states": {...}}
│   }
│
├─> Deserialize observation:
│   obs = Observation.from_dict(obs_dict)
│   │
│   DATA TRANSFORMATION:
│   Dict → Observation object reconstructed
│   obs.local = {}
│   obs.global_info = {"agent_states": {...}}
│   obs.timestamp = 0.0
│
└─> compute_action(obs, scheduler)
    │
    ├─> action = policy.forward(observation=obs)
    │   │
    │   DATA AUTO-CONVERSION:
    │   obs.__array__() called by policy
    │       ↓
    │   obs.vector() → Flatten dicts
    │       ↓
    │   np.array([...], dtype=float32)  # For neural network
    │   │
    │   POLICY COMPUTATION:
    │   neural_net(obs_vector) → action_vector
    │       ↓
    │   action = Action()
    │   action.set_values(action_vector)
    │   │
    │   DATA OUTFLOW:
    │   Action object with c=[0.15]
    │
    ├─> set_action(action)
    │   └─> self.action.set_values(action)
    │       DATA UPDATE: self.action.c = [0.15]
    │
    └─> schedule_action_effect(agent_id, delay=act_delay)
        └─> Event scheduled at t = 0.4 + 0.5 = 0.9
```

---

#### **t=0.9: ACTION_EFFECT(system_agent)**

```
action_effect_handler()
│
├─> apply_action()
│   └─> set_state()
│       ├─> current_soc = self.state.features[0].soc  # Read from State object
│       ├─> new_soc = current_soc + self.action.c[0] * 0.01
│       └─> self.state.features[0].set_values(soc=new_soc)
│           │
│           DATA UPDATE:
│           State object modified in-place
│           self.state.features[0].soc = new_value
│
├─> Serialize state with metadata:
│   state_dict = self.state.to_dict(include_metadata=True)
│   │
│   DATA TRANSFORMATION:
│   State object
│       ↓ .to_dict(include_metadata=True)
│   {
│       "_owner_id": "system_agent",
│       "_owner_level": 3,
│       "_state_type": "SystemAgentState",
│       "features": {"FeatureName": {...}}
│   }
│
└─> schedule_message_delivery(
        sender=system_agent,
        recipient=proxy,
        message={"set_state": "local", "body": state_dict},
        delay=msg_delay
    )
    │
    DATA OUTFLOW:
    Serialized dict with metadata sent in message
    Scheduled at: t = 0.9 + 0.2 = 1.1
```

---

#### **t=1.1: MESSAGE_DELIVERY(proxy) - State Update**

```
ProxyAgent.message_delivery_handler()
│
├─> Receive: {"set_state": "local", "body": state_dict}
│   │
│   DATA INFLOW:
│   state_dict = {
│       "_owner_id": "system_agent",
│       "_owner_level": 3,
│       "_state_type": "SystemAgentState",
│       "features": {"FeatureName": {...}}
│   }
│
├─> Extract metadata:
│   agent_id = state_dict["_owner_id"]  # "system_agent"
│   features = state_dict["features"]   # {"FeatureName": {...}}
│
├─> set_local_state(agent_id, features)
│   │
│   DATA STORAGE:
│   state_cache["agents"]["system_agent"] = features
│   │
│   UPDATE:
│   state_cache["agents"]["system_agent"] updated with new values
│
└─> Send completion message
    └─> {"set_state_completion": "success"}
```

---

#### **t=5.0: AGENT_TICK(battery_1) - Field Agent Tick**

```
FieldAgent.tick(scheduler, current_time=5.0)
│
├─> self._timestep = 5.0
│
├─> If has policy: Request observation
│   └─> schedule_message_delivery(
│           message={"get_info": "obs"}
│       )
│
└─> [Cascade continues with action computation, effect, state update...]
```

---

#### **t=wait_interval+6*msg_delay: Reward Computation**

```
Agent receives local state → Compute tick results

message_delivery_handler() - "get_local_state_response"
│
├─> Receive: {"get_local_state_response": {"body": local_state}}
│   │
│   DATA INFLOW:
│   local_state = {"BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}}
│
├─> Compute tick result:
│   tick_result = {
│       "reward": compute_local_reward(local_state),
│       "terminated": is_terminated(local_state),
│       "truncated": is_truncated(local_state),
│       "info": get_local_info(local_state)
│   }
│   │
│   DATA USAGE IN compute_local_reward():
│   local_state = {"BatteryChargeFeature": {"soc": 0.503, ...}}
│   soc = local_state["BatteryChargeFeature"]["soc"]  # Extract from dict
│   return soc  # 0.503
│   │
│   OUTPUT: tick_result = {
│       "reward": 0.503,
│       "terminated": False,
│       "truncated": False,
│       "info": {}
│   }
│
└─> schedule_message_delivery(
        message={"set_tick_result": "local", "body": tick_result}
    )
    │
    DATA OUTFLOW:
    Tick result dict sent to proxy
```

---

## **ProxyAgent State Cache Architecture**

### **Cache Structure**

```python
proxy.state_cache = {
    # Per-agent states (State objects - NEW!)
    "agents": {
        "battery_1": FieldAgentState(
            owner_id="battery_1",
            owner_level=1,
            features=[BatteryChargeFeature(soc=0.503, capacity=100.0)]
        ),  # State object!
        "battery_2": FieldAgentState(...),  # State object!
        "zone_1": CoordinatorAgentState(...),  # State object!
        "system_agent": SystemAgentState(...)  # State object!
    },

    # Global state (environment-wide data, if needed)
    "global": {
        # Can contain additional environment-wide metrics
        "grid_frequency": 60.0,
        "total_load": 1500.0,
        ...
    }
}

# Agent levels tracked separately for visibility checks:
proxy._agent_levels = {
    "battery_1": 1,
    "battery_2": 1,
    "zone_1": 2,
    "system_agent": 3
}
```

### **State Cache Operations**

#### **SET Operations**

| Method | Input Data | Transformation | Storage Location |
|--------|-----------|----------------|------------------|
| `set_local_state(aid, state)` | `agent_id: str`<br>`state: State` | None (stores object!) | `state_cache["agents"][aid]` |
| `set_global_state(dict)` | `global_dict: Dict` | None | `state_cache["global"].update(...)` |
| `init_global_state()` | None | No longer needed | (Agents stored directly) |

#### **GET Operations**

| Method | Retrieval Source | Filtering | Return Type |
|--------|-----------------|-----------|-------------|
| `get_local_state(aid)` | `state_cache["agents"][aid]` | **state.observed_by()** (NEW!) | `Dict[str, np.ndarray]` |
| `get_global_states(aid)` | `state_cache["agents"]` (all) | **state.observed_by()** (NEW!) | `Dict[str, Dict[str, np.ndarray]]` |
| `get_observation(aid)` | Both local + global | Feature visibility + wrapping | `Observation` |

---

## **Complete Data Flow Diagrams**

### **CTDE Training Step - Data Transformations**

```
┌─────────────────────────────────────────────────────────────────┐
│ RL Algorithm Provides Actions                                   │
├─────────────────────────────────────────────────────────────────┤
│ actions = {                                                      │
│     "battery_1": Action(c=[0.3]),  ← Action objects             │
│     "battery_2": Action(c=[-0.2])                               │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓ env.step(actions)
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Action Application                                     │
├─────────────────────────────────────────────────────────────────┤
│ battery_1.set_action(Action(c=[0.3]))                           │
│     ↓ set_values()                                               │
│ self.action.c = [0.3]  ← Action object updated                  │
│                                                                  │
│ battery_1.apply_action()                                        │
│     ↓ set_state()                                                │
│ self.state.features[0].soc = 0.503  ← State object updated      │
│                                                                  │
│ proxy.set_local_state("battery_1", state.to_dict())             │
│     ↓ Transformation: State → Dict                               │
│ state_cache["agents"]["battery_1"] = {                          │
│     "BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}   │
│ }  ← Dict stored                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Simulation                                             │
├─────────────────────────────────────────────────────────────────┤
│ global_state = proxy.get_global_states("system_agent")          │
│     ↓ Retrieval                                                  │
│ {"agent_states": {"battery_1": {...}, ...}}  ← Dict             │
│                                                                  │
│ env_state = global_state_to_env_state(global_state)             │
│     ↓ Extraction from dict                                       │
│ agent_states = global_state["agent_states"]                     │
│ soc = agent_states["battery_1"]["BatteryChargeFeature"]["soc"]  │
│ EnvState(battery_soc=0.503)  ← Custom env state                 │
│                                                                  │
│ updated_env_state = run_simulation(env_state)                   │
│     ↓ Physics                                                    │
│ env_state.battery_soc = 0.503  ← Still valid                    │
│                                                                  │
│ updated_global = env_state_to_global_state(updated_env_state)   │
│     ↓ Construction of dict                                       │
│ {                                                                │
│     "agent_states": {                                            │
│         "battery_1": {"BatteryChargeFeature": {"soc": 0.503}}   │
│     }                                                            │
│ }  ← Dict structure                                              │
│                                                                  │
│ proxy.set_global_state(updated_global)                          │
│     ↓ .update()                                                  │
│ state_cache["global"]["agent_states"]["battery_1"] updated      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Observation Collection                                │
├─────────────────────────────────────────────────────────────────┤
│ obs = proxy.get_observation("battery_1")                        │
│     ↓ Construction                                               │
│ local = state_cache["agents"]["battery_1"]  ← Dict              │
│ global = state_cache["global"]              ← Dict              │
│     ↓ Wrapping                                                   │
│ Observation(local={...}, global_info={...})  ← Object           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: Reward Computation                                    │
├─────────────────────────────────────────────────────────────────┤
│ local_state = proxy.get_local_state("battery_1")                │
│     ↓ Retrieval                                                  │
│ {"BatteryChargeFeature": {"soc": 0.503, ...}}  ← Dict           │
│                                                                  │
│ compute_local_reward(local_state)                               │
│     ↓ Dict access                                                │
│ soc = local_state["BatteryChargeFeature"]["soc"]                │
│ return 0.503  ← Scalar                                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: Vectorization for RL                                  │
├─────────────────────────────────────────────────────────────────┤
│ obs_dict = {"battery_1": Observation(...)}                      │
│     ↓ .vector()                                                  │
│ obs_vec = {"battery_1": np.array([0.503, 100.0, ...])}          │
│     ↓ Return to RL                                               │
│ RL algorithm receives numpy arrays!                             │
└─────────────────────────────────────────────────────────────────┘
```

---

### **Event-Driven Mode - Data Transformations Timeline**

```
t=0.0: Init
├─> Agent: State objects with features
├─> Proxy: Dicts in state_cache["agents"]
└─> Global: Compiled dict in state_cache["global"]["agent_states"]

t=0.2: Observation Request (Proxy)
├─> Inflow: Request message (string)
├─> Process: Construct Observation from cache dicts
├─> Outflow: Observation.to_dict() → Serialized dict in message

t=0.4: Observation Response (Agent)
├─> Inflow: Serialized observation dict
├─> Process: Observation.from_dict() → Reconstruct object
├─> Outflow: Action object from policy

t=0.9: Action Effect (Agent)
├─> Inflow: self.action (Action object)
├─> Process: Update self.state (State object)
├─> Outflow: state.to_dict(include_metadata=True) → Serialized dict with metadata

t=1.1: State Update (Proxy)
├─> Inflow: Serialized state dict with metadata
├─> Process: Extract agent_id and features
├─> Storage: state_cache["agents"][aid] = features dict

[Rewards computed later with dict access to cached states]
```

---

## **Key Design Patterns**

### **1. Object-Dict Duality**

**Agent Layer:**
- Works with rich objects (State, Action, Observation)
- Has methods: `.vector()`, `.set_values()`, `.update()`
- Type-safe with IDE support

**Proxy Layer:**
- Stores everything as dicts
- Enables serialization for messages
- Uniform data format

**Boundary:**
- **Agent → Proxy**: Call `.to_dict()` when storing
- **Proxy → Agent**: Return dicts, agents access with dict keys
- **Messages**: Always dicts (serialized)

---

### **2. Serialization Points**

| Crossing Point | Data Type | Transformation |
|----------------|-----------|----------------|
| Agent → Proxy (CTDE) | State object | `.to_dict()` |
| Agent → Message | State object | `.to_dict(include_metadata=True)` |
| Agent → Message | Observation object | `.to_dict()` |
| Message → Agent | Observation dict | `Observation.from_dict()` |
| Proxy → Agent | State dict | No transformation (return dict) |
| Observation → Policy | Observation object | `.__array__()` auto-converts |

---

### **3. Data Consistency Rules**

**✅ DO:**
- Use Action objects throughout (no dict conversion)
- **Pass State objects to proxy** (NEW! - proxy stores objects now)
- Convert State → Dict **only when sending messages** (with `include_metadata=True`)
- Use `local_state` parameter (numpy array dict) in reward/info methods, not `self.state`
- Access features as vectors: `local_state["FeatureName"][0]` for first element

**❌ DON'T:**
- Send State objects in messages (must serialize to dict first)
- Access dict fields like `local_state["Feature"]["soc"]` (it's now a numpy array!)
- Access `self.state` in `compute_local_reward()` (use filtered parameter)
- Pass action dicts when Action objects are expected

---

## **Data Inflow/Outflow Summary**

### **ProxyAgent**

**Inflows:**
1. `set_local_state(aid, dict)` ← From agent.reset(), agent.act()
2. `set_global_state(dict)` ← From simulate()
3. Messages with serialized states/observations

**Outflows:**
1. `get_local_state(aid)` → Dict to agent
2. `get_global_states(aid)` → Dict to agent
3. `get_observation(aid)` → Observation object (wraps dicts)
4. Messages with serialized data

**Internal Storage:**
- Everything as dicts in `state_cache`

---

### **Agent**

**Inflows:**
1. Actions from env.step() or policy → Action objects
2. Observations from proxy.get_observation() → Observation objects
3. State dicts from proxy.get_local_state() → Dicts

**Outflows:**
1. Updated states to proxy → Serialized dicts
2. Observations to caller → Observation objects
3. Rewards/info from dicts → Scalars/dicts

**Internal Storage:**
- `self.state` → State object
- `self.action` → Action object
- Never stores dicts internally

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

proxy.set_local_state("battery_1", state.to_dict()):
    INPUT: state_dict = {"BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}}
    STORE: state_cache["agents"]["battery_1"] = state_dict
    TYPE: Dict

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

    proxy.set_local_state("battery_1", state.to_dict()):
        TRANSFORM: State → Dict via .to_dict()
        OUTPUT: {"BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}}
        STORE: state_cache["agents"]["battery_1"] = dict
        TYPE: Dict (stored)

2. Observation Collection:
    proxy.get_observation("battery_1"):
        READ: local = state_cache["agents"]["battery_1"]
        READ: global = state_cache["global"]
        CONSTRUCT: Observation(local=dict, global_info=dict, timestamp=1.0)
        OUTPUT: Observation object
        TYPE: Observation object

    observe() returns:
        OUTPUT: {"battery_1": Observation(...)}
        TYPE: Dict[str, Observation]

3. Reward Computation:
    proxy.get_local_state("battery_1"):
        READ: state_cache["agents"]["battery_1"]
        OUTPUT: {"BatteryChargeFeature": {"soc": 0.503, "capacity": 100.0}}
        TYPE: Dict

    compute_local_reward(local_state):
        INPUT: local_state = {"BatteryChargeFeature": {"soc": 0.503, ...}}
        ACCESS: local_state["BatteryChargeFeature"]["soc"]
        OUTPUT: 0.503
        TYPE: float

4. Vectorization for RL:
    proxy.get_step_results():
        INPUT: obs = {"battery_1": Observation(...)}
        TRANSFORM: observation.vector() for each
        OUTPUT: {"battery_1": np.array([0.503, 100.0, ...])}
        TYPE: Dict[str, np.ndarray]

=== RETURN TO RL ===
    obs: Dict[str, np.ndarray]     ← Numpy arrays
    rewards: Dict[str, float]      ← Scalars
    terminated: Dict[str, bool]
    truncated: Dict[str, bool]
    info: Dict[str, Dict]
```

---

## **Message Passing - Serialization Flow**

### **Observation Message Flow**

```
Agent → Message → Proxy → Message → Agent

1. Agent requests observation:
    DATA: {"get_info": "obs"}
    TYPE: Dict (simple request)

2. Proxy constructs observation:
    local = state_cache["agents"]["battery_1"]     # Dict
    global = state_cache["global"]                 # Dict
    obs = Observation(local, global, timestamp)    # Object

3. Proxy serializes for message:
    obs_dict = obs.to_dict()
    DATA: {
        "timestamp": 0.0,
        "local": {"BatteryChargeFeature": {"soc": 0.503, ...}},
        "global_info": {"agent_states": {...}}
    }
    TYPE: Dict (serialized)

4. Proxy sends message:
    MESSAGE: {"get_obs_response": {"body": obs_dict}}

5. Agent receives and deserializes:
    obs_dict = message["get_obs_response"]["body"]
    obs = Observation.from_dict(obs_dict)
    TYPE: Observation object (reconstructed)

6. Agent uses observation:
    action = policy.forward(observation=obs)
    └─> obs.__array__() → np.ndarray automatically
```

### **State Message Flow**

```
Agent → Message → Proxy

1. Agent updates state:
    self.state.features[0].soc = 0.503
    TYPE: State object (modified)

2. Agent serializes with metadata:
    state_dict = self.state.to_dict(include_metadata=True)
    DATA: {
        "_owner_id": "battery_1",
        "_owner_level": 1,
        "_state_type": "FieldAgentState",
        "features": {"BatteryChargeFeature": {"soc": 0.503, ...}}
    }
    TYPE: Dict with metadata

3. Agent sends message:
    MESSAGE: {"set_state": "local", "body": state_dict}

4. Proxy receives and extracts:
    agent_id = state_dict["_owner_id"]        # "battery_1"
    features = state_dict["features"]         # {"BatteryChargeFeature": {...}}

5. Proxy stores:
    set_local_state(agent_id, features)
    └─> state_cache["agents"]["battery_1"] = features
    TYPE: Dict (without metadata)
```

---

## **Critical Takeaways**

### **1. Proxy Storage is Now State Objects (NEW!)**
- State objects stored directly in `state_cache["agents"]`
- ✅ Enables feature-level visibility filtering via `state.observed_by()`
- ✅ Maintains full type information (FeatureProvider instances)
- ⚠️ Serialization still needed for message passing

### **2. Objects Exist Only in Agent Layer**
- Agents work with State/Action/Observation objects
- Rich APIs and type safety
- Objects never cross proxy boundary (except as dicts)

### **3. Message Passing Requires Serialization**
- Observation: `.to_dict()` → send → `from_dict()` → reconstruct
- State: `.to_dict(include_metadata=True)` → send → extract metadata
- Action: Passed as objects (no serialization needed in current design)

### **4. Dict Access Pattern**
When methods receive `local_state: dict` parameter:
```python
# ✅ CORRECT:
soc = local_state["BatteryChargeFeature"]["soc"]

# ❌ WRONG:
soc = self.state.features[0].soc  # Ignores parameter, uses stale state
```

### **5. Type Consistency (UPDATED!)**
- **State**: Object (agent) → **Object (proxy)** → **Filtered vectors (retrieval via observed_by())**
- **Action**: Object (throughout, no proxy storage)
- **Observation**: Object → Dict (messages) → Object (reconstruction)
- **Feature**: Object (in State) → Stays as object in proxy → Vector (in observations)
- **Visibility Filtering**: State object → `observed_by()` → `Dict[feature_name, np.ndarray]`

---

---

## **NEW: Feature Visibility Architecture (Latest Update)**

### **State Object Storage in Proxy**

The proxy now stores **State objects directly** instead of dicts:
- **Before:** `state_cache["agents"]["battery_1"] = {"BatteryChargeFeature": {"soc": 0.5}}`  (dict)
- **After:** `state_cache["agents"]["battery_1"] = FieldAgentState(...)`  (State object!)

### **Visibility Filtering Active**

When agents request observations:
1. Proxy retrieves State objects from cache
2. Calls `state.observed_by(requestor_id, requestor_level)`
3. Returns filtered feature vectors based on visibility rules
4. Agents receive `Dict[feature_name, np.ndarray]` (vectors, not dicts!)

### **Data Format Change**

**Old format (dict-based):**
```python
local_state = {"BatteryChargeFeature": {"soc": 0.5, "capacity": 100.0}}
soc = local_state["BatteryChargeFeature"]["soc"]  # Dict access
```

**New format (vector-based after visibility filtering):**
```python
local_state = {"BatteryChargeFeature": np.array([0.5, 100.0])}
soc = local_state["BatteryChargeFeature"][0]  # Array access!
```

### **Feature Auto-Registration**

FeatureProvider now uses `FeatureMeta` metaclass for automatic registration:
- All FeatureProvider subclasses auto-register in `_FEATURE_REGISTRY`
- Enables `State.from_dict()` to reconstruct features dynamically
- No manual registration required!

---

This documentation reflects the **latest version** of the HERON framework with:
- ✅ Feature visibility enforcement
- ✅ State object storage in proxy
- ✅ Proper type handling and serialization
- ✅ Security through visibility filtering

# HERON Execution Flows - Complete Reference

This document provides a comprehensive description of the two execution modes in the HERON framework:
- **Mode A**: CTDE (Centralized Training with Decentralized Execution) - Synchronous training mode
- **Mode B**: Event-Driven Testing - Asynchronous testing mode with realistic timing

---

## **Table of Contents**

1. [Key Components](#key-components)
2. [CTDE Training Mode](#ctde-training-mode)
3. [Event-Driven Testing Mode](#event-driven-testing-mode)
4. [Observation & Action Data Flow](#observation--action-data-flow)
5. [ProxyAgent State Management](#proxyagent-state-management)

---

## **Key Components**

### **Agent Hierarchy (3 Levels)**
```
SystemAgent (L3)
  └─> CoordinatorAgent (L2)
      └─> FieldAgent (L1)
```

### **Core Classes**

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| **EnvCore** | Environment base class | `step()`, `run_event_driven()` |
| **SystemAgent** | Top-level coordinator | `execute()`, `tick()`, `simulate()` |
| **ProxyAgent** | State distribution hub | `get_observation()`, `set_local_state()` |
| **Action** | Action representation | `vector()`, `set_values()`, `clip()` |
| **Observation** | Observation with array interface | `vector()`, `__array__()`, `shape` |
| **EventScheduler** | Discrete event simulation | `run_until()`, `schedule_agent_tick()` |

---

## **CTDE Training Mode**

### **1. Initialization Flow**

```
env = CustomEnv(system_agent=..., coordinator_agents=[...])
│
├─> EnvCore.__init__()
│   ├─> _register_agents(system_agent, coordinator_agents)
│   │   ├─> system_agent.set_simulation(
│   │   │       run_simulation,
│   │   │       env_state_to_global_state,
│   │   │       global_state_to_env_state,
│   │   │       wait_interval
│   │   │   )
│   │   ├─> register_agent(system_agent) [recursive for all subordinates]
│   │   └─> Creates ProxyAgent()
│   │
│   ├─> _initialize_agents()
│   │   ├─> For each agent in hierarchy:
│   │   │   ├─> agent.init_action() → Creates Action object
│   │   │   ├─> agent.init_state() → Creates State object (FieldAgentState, etc.)
│   │   │   ├─> proxy.register_agent(agent_id, agent_state=agent.state)
│   │   │   │   └─> proxy.set_local_state(agent.state)
│   │   │   │       └─> state_cache["agents"][agent_id] = state.to_dict()
│   │   │   └─> [Recurse for subordinates]
│   │   │
│   │   └─> proxy.init_global_state()
│   │       └─> state_cache["global"]["agent_states"] = {all agent states}
│   │
│   ├─> MessageBroker.init() and attach agents
│   └─> EventScheduler.init() and attach agents
```

### **2. Reset Flow**

```
obs, info = env.reset(seed=42)
│
├─> scheduler.reset(seed) → Clear event queue
├─> clear_broker_environment() → Clear messages
├─> proxy.reset(seed)
│   └─> state_cache = {} (cleared)
│
├─> system_agent.reset(seed, proxy)
│   ├─> self._timestep = 0.0
│   ├─> init_action() → Reset action to neutral
│   ├─> init_state() → Create fresh state
│   ├─> proxy.set_local_state(self.state)
│   │   └─> state_cache["agents"]["system_agent"] = state.to_dict()
│   │
│   ├─> For each coordinator:
│   │   ├─> coordinator.reset(seed, proxy)
│   │   │   ├─> init_state()
│   │   │   ├─> proxy.set_local_state(self.state)
│   │   │   │
│   │   │   └─> For each field_agent:
│   │   │       ├─> field_agent.reset(seed, proxy)
│   │   │       ├─> init_state()
│   │   │       └─> proxy.set_local_state(self.state)
│   │   │
│   │   └─> [Sets action_space, observation_space]
│   │
│   └─> return self.observe(proxy) → Dict[AgentID, Observation]
│
├─> proxy.init_global_state()
│   └─> state_cache["global"] = compiled from all agent states
│
└─> return (observations, {})
```

### **3. Step Execution Flow**

```
obs, rewards, terminated, truncated, info = env.step(actions)
│
└─> SystemAgent.execute(actions, proxy)
    │
    ├─> PHASE 1: Action Application
    │   │
    │   └─> self.act(actions, proxy)
    │       │
    │       ├─> layered = layer_actions(actions)
    │       │   └─> {
    │       │         "self": actions["system_agent"],
    │       │         "subordinates": {
    │       │           "coord_1": {
    │       │             "self": actions["coord_1"],
    │       │             "subordinates": {
    │       │               "field_1": {...},
    │       │               ...
    │       │             }
    │       │           },
    │       │           ...
    │       │         }
    │       │       }
    │       │
    │       ├─> handle_self_action(layered['self'], proxy)
    │       │   ├─> if action provided:
    │       │   │   └─> set_action(action)
    │       │   ├─> elif has policy:
    │       │   │   ├─> obs = proxy.get_observation(agent_id)
    │       │   │   │   └─> Returns Observation(local={...}, global_info={...})
    │       │   │   └─> set_action(policy.forward(observation=obs))
    │       │   │       └─> Observation.__array__() auto-converts to np.ndarray
    │       │   │
    │       │   ├─> apply_action() → Updates self.state based on self.action
    │       │   └─> proxy.set_local_state(self.state)
    │       │       └─> state_cache["agents"][agent_id] = state.to_dict()
    │       │
    │       └─> handle_subordinate_actions(layered['subordinates'], proxy)
    │           └─> For each subordinate:
    │               └─> subordinate.execute(layered_actions[sub_id], proxy)
    │                   └─> [Recursively calls act() on subordinates]
    │
    ├─> PHASE 2: Simulation
    │   │
    │   ├─> global_state = proxy.get_global_states(system_agent_id, protocol)
    │   │   └─> return state_cache["global"] → Contains all agent states
    │   │
    │   ├─> updated_global_state = simulate(global_state)
    │   │   ├─> env_state = global_state_to_env_state(global_state)
    │   │   ├─> updated_env_state = run_simulation(env_state)
    │   │   │   └─> [User-defined physics simulation]
    │   │   └─> return env_state_to_global_state(updated_env_state)
    │   │
    │   └─> proxy.set_global_state(updated_global_state)
    │       └─> state_cache["global"].update(updated_global_state)
    │
    ├─> PHASE 3: Observation Collection
    │   │
    │   └─> obs = self.observe(proxy)
    │       └─> For each agent in hierarchy:
    │           ├─> observation = proxy.get_observation(agent_id, protocol)
    │           │   ├─> global_state = state_cache["global"]
    │           │   ├─> local_state = state_cache["agents"][agent_id]
    │           │   └─> return Observation(local, global_info, timestamp)
    │           │
    │           └─> Returns: Dict[AgentID, Observation]
    │
    ├─> PHASE 4: Reward & Status Computation
    │   │
    │   ├─> rewards = self.compute_rewards(proxy)
    │   │   └─> For each agent:
    │   │       ├─> local_state = proxy.get_local_state(agent_id)
    │   │       └─> reward = compute_local_reward(local_state)
    │   │
    │   ├─> infos = self.get_info(proxy)
    │   ├─> terminateds = self.get_terminateds(proxy)
    │   └─> truncateds = self.get_truncateds(proxy)
    │
    ├─> PHASE 5: Cache Results
    │   │
    │   └─> proxy.set_step_result(obs, rewards, terminateds, truncateds, infos)
    │       └─> _step_results = {obs: Dict[AgentID, Observation], ...}
    │
    └─> RETURN: proxy.get_step_results()
        │
        ├─> obs_vectorized = {aid: observation.vector() for aid, obs in obs.items()}
        │   └─> Converts Observation objects to np.ndarray for RL
        │
        └─> return (obs_vectorized, rewards, terminateds, truncateds, infos)
```

### **Key Features:**

✅ **Synchronous execution**: All agents act, then simulate, then observe
✅ **Centralized training**: Full observability via proxy
✅ **Type safety**: Observations are Observation objects, auto-vectorized for RL
✅ **State management**: Proxy maintains global and per-agent local states

---

## **Event-Driven Testing Mode**

### **1. Event Types**

| Event Type | When Scheduled | Handler | Purpose |
|------------|----------------|---------|---------|
| `AGENT_TICK` | At tick_interval | `agent_tick_handler()` | Trigger agent's tick() method |
| `MESSAGE_DELIVERY` | After msg_delay | `message_delivery_handler()` | Deliver async messages |
| `ACTION_EFFECT` | After act_delay | `action_effect_handler()` | Apply delayed action effects |
| `SIMULATION` | After wait_interval | `simulation_handler()` | Run physics simulation |

### **2. Initialization & Reset**

Same as CTDE mode - agents register with proxy, states initialized.

After `env.reset()`:
- `state_cache["agents"]` populated with all agent states ✓
- `state_cache["global"]` compiled from agent states ✓
- Scheduler has initial AGENT_TICK event for SystemAgent at t=0 ✓

### **3. Event-Driven Execution Flow**

```
result = env.run_event_driven(event_analyzer, t_end=100.0)
│
└─> for event in scheduler.run_until(t_end=100.0):
    │
    ├─> EventScheduler.process_next()
    │   ├─> event = pop() from priority queue
    │   ├─> current_time = event.timestamp
    │   ├─> handler = get_handler(event.event_type, event.agent_id)
    │   └─> handler(event, scheduler)
    │
    └─> result.add_event_analysis(event_analyzer.parser_event(event))
        └─> EventAnalyzer extracts results from events
```

### **4. Detailed Event Sequence**

#### **Timeline Example** (SystemAgent → Coordinator → 2 FieldAgents):

```
t=0.0: AGENT_TICK(system_agent)
  ├─> SystemAgent.tick()
  │   ├─> Schedule subordinate ticks:
  │   │   └─> schedule_agent_tick(coordinator_1)
  │   │       └─> timestamp = 0.0 + coordinator.tick_interval (60.0) = 60.0
  │   │
  │   ├─> If has policy: Request observation
  │   │   └─> schedule_message_delivery(
  │   │           sender=system_agent, recipient=proxy,
  │   │           message={"get_info": "obs"},
  │   │           delay=msg_delay
  │   │       )
  │   │       └─> Scheduled at t = 0.0 + msg_delay
  │   │
  │   └─> Schedule simulation:
  │       └─> schedule_simulation(system_agent, wait_interval)
  │           └─> Scheduled at t = 0.0 + wait_interval

t=msg_delay: MESSAGE_DELIVERY(proxy) - "get_info:obs" request
  └─> ProxyAgent.message_delivery_handler()
      ├─> obs = get_observation(system_agent, protocol)
      │   ├─> global_state = state_cache["global"] → {...}
      │   ├─> local_state = state_cache["agents"]["system_agent"] → {...}
      │   └─> return Observation(local, global_info, timestamp)
      │
      └─> schedule_message_delivery(
              sender=proxy, recipient=system_agent,
              message={"get_obs_response": {"body": Observation(...)}},
              delay=msg_delay
          )
          └─> Scheduled at t = msg_delay + msg_delay

t=2*msg_delay: MESSAGE_DELIVERY(system_agent) - "get_obs_response"
  └─> SystemAgent.message_delivery_handler()
      ├─> obs = message_content['body'] → Observation object
      │
      └─> compute_action(obs, scheduler)
          ├─> action = policy.forward(observation=obs)
          │   └─> Observation.__array__() converts to np.ndarray
          ├─> set_action(action)
          └─> schedule_action_effect(agent_id, delay=act_delay)
              └─> Scheduled at t = 2*msg_delay + act_delay

t=2*msg_delay+act_delay: ACTION_EFFECT(system_agent)
  └─> Agent.action_effect_handler()
      ├─> apply_action() → Updates self.state
      └─> schedule_message_delivery(
              sender=system_agent, recipient=proxy,
              message={"set_state": "local", "body": self.state},
              delay=msg_delay
          )
          └─> Scheduled at t = 2*msg_delay + act_delay + msg_delay

t=3*msg_delay+act_delay: MESSAGE_DELIVERY(proxy) - "set_state:local"
  └─> ProxyAgent.message_delivery_handler()
      ├─> set_local_state(state) → Updates state_cache["agents"][agent_id]
      └─> schedule_message_delivery(
              message={"set_state_completion": "success"}
          )

t=wait_interval: SIMULATION(system_agent)
  └─> SystemAgent.simulation_handler()
      └─> schedule_message_delivery(
              message={"get_info": "global_state"}
          )

t=wait_interval+msg_delay: MESSAGE_DELIVERY(proxy) - "get_info:global_state"
  └─> ProxyAgent returns global_state

t=wait_interval+2*msg_delay: MESSAGE_DELIVERY(system_agent) - "get_global_state_response"
  └─> SystemAgent.message_delivery_handler()
      ├─> global_state = message_content['body']
      ├─> updated_global_state = simulate(global_state)
      │   ├─> env_state = global_state_to_env_state(global_state)
      │   ├─> updated_env_state = run_simulation(env_state)
      │   └─> return env_state_to_global_state(updated_env_state)
      │
      └─> schedule_message_delivery(
              message={"set_state": "global", "body": updated_global_state}
          )

t=wait_interval+3*msg_delay: MESSAGE_DELIVERY(proxy) - "set_state:global"
  └─> ProxyAgent updates global state
      └─> schedule_message_delivery(
              message={"set_state_completion": "success"}
          )

t=wait_interval+4*msg_delay: MESSAGE_DELIVERY(system_agent) - "set_state_completion"
  └─> SystemAgent.message_delivery_handler()
      ├─> For each subordinate:
      │   └─> schedule_message_delivery(
      │           sender=subordinate_id,
      │           recipient=proxy,
      │           message={"get_info": "local_state"}
      │       )
      │
      └─> schedule_message_delivery(
              sender=system_agent,
              message={"get_info": "local_state"}
          )

t=wait_interval+5*msg_delay: MESSAGE_DELIVERY(proxy) - "get_info:local_state" requests
  └─> ProxyAgent returns local_state to each requester

t=wait_interval+6*msg_delay: MESSAGE_DELIVERY(agents) - "get_local_state_response"
  └─> Agents.message_delivery_handler()
      ├─> tick_result = {
      │       "reward": compute_local_reward(local_state),
      │       "terminated": is_terminated(local_state),
      │       "truncated": is_truncated(local_state),
      │       "info": get_local_info(local_state)
      │   }
      │
      └─> schedule_message_delivery(
              message={"set_tick_result": "local", "body": tick_result}
          )

t=wait_interval+7*msg_delay: MESSAGE_DELIVERY(proxy) - "set_tick_result"
  └─> ProxyAgent.message_delivery_handler()
      └─> print(f"Received tick result from {sender_id}: {tick_result}")
          # Results are captured by EventAnalyzer from the event itself!

t=60.0: AGENT_TICK(coordinator_1)
  └─> [Similar cascade as SystemAgent]

t=1.0: AGENT_TICK(field_1)
  └─> [Similar flow for field agents]
```

### **5. Result Collection via EventAnalyzer**

```
env.run_event_driven(event_analyzer, t_end)
│
└─> For each event:
    └─> event_analyzer.parser_event(event)
        │
        ├─> If event.event_type == MESSAGE_DELIVERY:
        │   ├─> message = event.payload["message"]
        │   │
        │   └─> If "set_tick_result" in message:
        │       ├─> sender_id = event.payload["sender"]
        │       ├─> tick_result = message["body"]
        │       │   └─> {reward, terminated, truncated, info}
        │       │
        │       └─> EventAnalyzer accumulates results per agent
        │
        └─> Returns EventAnalysis for this event

EpisodeResult accumulates all EventAnalysis objects
```

**Key Insight**: ProxyAgent acts as a **message relay**, not a result accumulator. EventAnalyzer is responsible for extracting and aggregating results from the event stream.

---

## **Observation & Action Data Flow**

### **Observation Flow (Both Modes)**

```
Agent needs observation
│
├─> CTDE Mode: Direct call
│   └─> obs = proxy.get_observation(agent_id, protocol)
│
├─> Event-Driven Mode: Via message
│   ├─> Request: {"get_info": "obs"}
│   └─> Response: {"get_obs_response": {"body": observation}}
│
└─> ProxyAgent.get_observation(sender_id, protocol)
    │
    ├─> global_state = get_global_states(sender_id, protocol)
    │   ├─> If sender == SYSTEM_AGENT_ID: Full global state
    │   └─> Else: Filtered by visibility_rules
    │
    ├─> local_state = get_local_state(sender_id, protocol)
    │   └─> return state_cache["agents"][sender_id]
    │
    └─> return Observation(
            local=local_state,
            global_info=global_state,
            timestamp=current_time
        )
        │
        ├─> Observation.__array__() → Auto-converts to np.ndarray for policies
        ├─> Observation.vector() → Explicit vectorization
        ├─> Observation.shape → Array-like shape property
        └─> Observation[i] → Array-like indexing
```

### **Action Flow**

```
CTDE Mode:
  agent.act(actions, proxy)
  └─> set_action(action) → self.action.set_values(action)
  └─> apply_action() → Updates self.state immediately
  └─> proxy.set_local_state(self.state)

Event-Driven Mode:
  agent.compute_action(obs, scheduler)
  └─> set_action(policy.forward(obs))
  └─> schedule_action_effect(delay=act_delay) → Delayed!

  [Later, at ACTION_EFFECT event:]
  agent.action_effect_handler()
  └─> apply_action() → Updates self.state after delay
  └─> schedule message to update proxy
```

**Key Difference**:
- **CTDE**: Actions apply immediately, state updated in proxy synchronously
- **Event-Driven**: Actions apply after delay, state updated in proxy asynchronously via messages

---

## **ProxyAgent State Management**

### **State Cache Structure**

```python
proxy.state_cache = {
    "global": {
        "agent_states": {
            "system_agent": {...},
            "coordinator_1": {...},
            "field_1": {...},
            ...
        },
        # Custom global fields added by environment
        "grid_frequency": 60.0,
        "total_load": 1500.0,
        ...
    },
    "agents": {
        "system_agent": {state.to_dict()},
        "coordinator_1": {state.to_dict()},
        "field_1": {state.to_dict()},
        ...
    }
}
```

### **State Update Methods**

| Method | Input | Effect |
|--------|-------|--------|
| `set_local_state(state: State)` | State object from an agent | Updates `state_cache["agents"][state.owner_id]` |
| `set_global_state(global_dict: Dict)` | Global state dict | Updates `state_cache["global"]` |
| `get_local_state(agent_id)` | Agent ID | Returns `state_cache["agents"][agent_id]` |
| `get_global_states(agent_id)` | Agent ID | Returns filtered global state based on visibility |
| `init_global_state()` | None | Compiles all agent states into global state |

### **Visibility Rules**

```python
# System agent sees everything
if sender_id == SYSTEM_AGENT_ID:
    return state_cache["global"]  # Full visibility

# Other agents filtered by visibility_rules
if sender_id in visibility_rules:
    allowed_keys = visibility_rules[sender_id]
    return {k: v for k, v in global_state.items() if k in allowed_keys}

# Default: Full global state (no filtering)
return state_cache["global"]
```

---

## **Action & Observation Class Enhancements**

### **Observation - Array-Like Interface**

The Observation class now behaves like a numpy array:

```python
obs = Observation(local={"voltage": 1.02}, global_info={"freq": 60.0})

# Automatic conversion for policies:
policy.forward(observation=obs)  # __array__() converts to np.ndarray

# Array-like properties:
obs.shape    # → (2,)
obs.dtype    # → float32
len(obs)     # → 2
obs[0]       # → 1.02
obs[:1]      # → array([1.02])

# Explicit vectorization:
vec = obs.vector()  # → array([1.02, 60.0], dtype=float32)
```

**Implementation**:
```python
def __array__(self, dtype=None) -> np.ndarray:
    vec = self.vector()
    return vec.astype(dtype) if dtype else vec

@property
def shape(self) -> tuple:
    return self.vector().shape

@property
def dtype(self):
    return np.float32

def __len__(self) -> int:
    return len(self.vector())

def __getitem__(self, key):
    return self.vector()[key]
```

### **Action - Continuous/Discrete Support**

```python
action = Action()
action.set_specs(
    dim_c=2,  # 2 continuous actions
    dim_d=1,  # 1 discrete action
    ncats=[5],  # 5 discrete choices
    range=(np.array([0.0, -1.0]), np.array([10.0, 1.0]))
)

# Set from policy output (numpy array):
action.set_values(np.array([5.0, 0.5, 2]))  # [c0, c1, d0]

# Access values:
action.c  # → array([5.0, 0.5])
action.d  # → array([2])

# Vectorize for logging:
action.vector()  # → array([5.0, 0.5, 2.0])
```

---

## **Comparison: CTDE vs Event-Driven**

| Aspect | CTDE Training | Event-Driven Testing |
|--------|---------------|----------------------|
| **Execution** | Synchronous (step-based) | Asynchronous (event-based) |
| **Timing** | Instantaneous | Realistic delays (msg_delay, act_delay) |
| **Action Application** | Immediate in `act()` | Delayed via ACTION_EFFECT |
| **State Updates** | Synchronous to proxy | Asynchronous via messages |
| **Observation Collection** | Direct function calls | Message-based requests |
| **Result Retrieval** | `proxy.get_step_results()` | EventAnalyzer from events |
| **Hierarchy Coordination** | Sequential function calls | Cascading tick events |
| **Use Case** | RL training with full observability | Deployment testing with latency |

---

## **Critical Design Patterns**

### **1. Two-Phase State Updates**

**CTDE Mode**:
```python
# Phase 1: Act
agent.act(actions, proxy)
  └─> apply_action() → Updates state
  └─> proxy.set_local_state(state) → Synchronous

# Phase 2: Simulate
simulate(global_state) → Physics update
proxy.set_global_state(updated_global_state)
```

**Event-Driven Mode**:
```python
# Phase 1: Action computed
compute_action(obs) → Set action
schedule_action_effect(delay)

# Phase 2: Action applied (after delay)
action_effect_handler()
  └─> apply_action() → Updates state
  └─> schedule message to update proxy → Asynchronous
```

### **2. Observation Type Consistency**

**Internal**: All observations are `Observation` objects
**External (for RL)**: Auto-vectorized to `np.ndarray` in `proxy.get_step_results()`

```python
# Internal flow:
obs: Dict[AgentID, Observation] = agent.observe(proxy)

# External return:
obs_vectorized: Dict[AgentID, np.ndarray] = proxy.get_step_results()[0]
```

### **3. Hierarchical Message Routing**

Agents request state from proxy:
```python
# Subordinate requests own state:
schedule_message_delivery(
    sender_id=subordinate_id,  # Sender is the requester
    recipient_id=PROXY_AGENT_ID,
    message={"get_info": "local_state"}
)

# Proxy responds:
schedule_message_delivery(
    sender_id=PROXY_AGENT_ID,
    recipient_id=subordinate_id,  # Response goes back to sender
    message={"get_local_state_response": {"body": {...}}}
)
```

---

## **State Lifecycle**

### **Initialization** (Once per program):
```
agent.init_state()
  → agent.state created
  → proxy.register_agent(agent_id, agent_state)
  → proxy.set_local_state(agent_state)
  → state_cache["agents"][agent_id] populated
```

### **Reset** (Once per episode):
```
agent.reset(proxy)
  → agent.init_state() (fresh state)
  → proxy.set_local_state(agent.state)
  → state_cache cleared and repopulated
  → proxy.init_global_state() compiles global state
```

### **Step/Tick** (Every timestep):
```
CTDE:
  agent.act() → apply_action() → proxy.set_local_state(state)

Event-Driven:
  action_effect → apply_action() → message to proxy → set_local_state(state)
```

---

## **Example: Complete CTDE Training Step**

```python
# Step 0: Reset
env.reset()  # All agents initialized, proxy has states

# Step 1: RL algorithm provides actions
actions = {
    "system_agent": np.array([...]),
    "coordinator_1": np.array([...]),
    "field_1": np.array([...]),
    "field_2": np.array([...]),
}

# Step 2: Environment executes
obs, rewards, terminated, truncated, info = env.step(actions)

# What happened internally:
# 1. SystemAgent.act(actions) → All agents apply actions, update states in proxy
# 2. SystemAgent.simulate() → Physics runs with updated states
# 3. SystemAgent.observe() → Collect observations (Observation objects)
# 4. SystemAgent.compute_rewards() → Compute rewards from states
# 5. proxy.set_step_result() → Cache results
# 6. proxy.get_step_results() → Vectorize observations, return to RL

# Step 3: RL algorithm trains
# obs: Dict[AgentID, np.ndarray] ready for neural network input ✓
```

---

## **Example: Complete Event-Driven Episode**

```python
# Setup
env.reset()
event_analyzer = EventAnalyzer()

# Run simulation
episode_result = env.run_event_driven(event_analyzer, t_end=100.0)

# What happened internally:
# - Events scheduled and processed in time order
# - Agents tick asynchronously with their own intervals
# - Actions applied with realistic delays
# - States updated via message passing
# - Tick results sent to proxy
# - EventAnalyzer extracts results from MESSAGE_DELIVERY events

# Retrieve results
episode_result.get_rewards()      # Extracted from events
episode_result.get_statistics()   # Event counts, timing analysis
```

---

## **Summary: Both Modes Are Production-Ready! ✅**

### **CTDE Training Mode**
- ✅ Proper hierarchical action distribution
- ✅ Synchronized state management via proxy
- ✅ Observations auto-vectorized for RL algorithms
- ✅ Rewards computed from populated agent states
- ✅ Full observability for centralized training

### **Event-Driven Testing Mode**
- ✅ Asynchronous execution with realistic timing
- ✅ Hierarchical tick cascade (agents tick at different rates)
- ✅ Message-based state distribution
- ✅ Delayed action effects
- ✅ Results collected via EventAnalyzer from event stream
- ✅ Tests deployment scenario with communication delays

**Both modes share the same agent hierarchy and state management infrastructure, ensuring consistency between training and testing!**

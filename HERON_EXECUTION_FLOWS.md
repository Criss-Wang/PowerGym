# HERON Execution Flows - Complete Reference

This document provides a comprehensive description of the two execution modes in the HERON framework:
- **Mode A**: CTDE (Centralized Training with Decentralized Execution) - Synchronous training mode
- **Mode B**: Event-Driven Testing - Asynchronous testing mode with realistic timing

---

## **Table of Contents**

1. [Key Components](#key-components)
2. [Agent Hierarchy & Action Passing](#agent-hierarchy--action-passing)
3. [Message Broker Architecture](#message-broker-architecture)
4. [CTDE Training Mode](#ctde-training-mode)
5. [Event-Driven Testing Mode](#event-driven-testing-mode)
6. [Protocol-Based Coordination](#protocol-based-coordination)
7. [Observation & Action Data Flow](#observation--action-data-flow)
8. [ProxyAgent State Management](#proxyagent-state-management)
9. [Handler Registration & Event Processing](#handler-registration--event-processing)
10. [Complete Execution Examples](#complete-execution-examples)

---

## **Key Components**

### **Agent Hierarchy (3 Levels)**
```
SystemAgent (L3) - Root orchestrator
  └─> CoordinatorAgent (L2) - Middle coordination layer
      └─> FieldAgent (L1) - Leaf execution agents
```

### **Core Classes**

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| **EnvCore** | Environment base class | `step()`, `run_event_driven()`, `reset()` |
| **SystemAgent** | Top-level orchestrator | `execute()`, `tick()`, `simulate()` |
| **CoordinatorAgent** | Mid-level coordinator | `execute()`, `tick()`, `coordinate()` |
| **FieldAgent** | Leaf execution agent | `execute()`, `tick()`, `apply_action()` |
| **ProxyAgent** | State distribution hub | `get_observation()`, `set_local_state()` |
| **Action** | Action representation | `vector()`, `set_values()`, `clip()` |
| **Observation** | Observation with array interface | `vector()`, `__array__()`, `shape` |
| **State** | Agent state container | `observed_by()`, `to_dict()`, `from_dict()` |
| **FeatureProvider** | Feature definition | `vector()`, `is_observable_by()` |
| **EventScheduler** | Discrete event simulation | `run_until()`, `schedule_agent_tick()` |
| **MessageBroker** | Inter-agent messaging | `publish()`, `consume()` |
| **Protocol** | Coordination logic | `coordinate()` |

---

## **Agent Hierarchy & Action Passing**

### **Hierarchical Structure**

```
┌─────────────────────────────────────────────────────────────────┐
│ SystemAgent (Level 3)                                           │
│ - Root of hierarchy (no upstream)                               │
│ - Manages simulation/physics                                    │
│ - Orchestrates all subordinates                                 │
│ - Owns global state view                                        │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ - execute() -> Orchestrate hierarchical execution               │
│ - simulate() -> Run environment physics                         │
│ - observe() -> Aggregate observations from all levels           │
│ - compute_rewards() -> Aggregate rewards                        │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ subordinates (coordinators)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CoordinatorAgent (Level 2)                                      │
│ - Has upstream_id pointing to system                            │
│ - Has subordinates (field agents)                               │
│ - Coordinates action decomposition                              │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ - receive_upstream_actions() -> Get action from system          │
│ - coordinate() -> Decompose actions for field agents            │
│ - send_subordinate_action() -> Pass actions to fields           │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ subordinates (field agents)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ FieldAgent (Level 1)                                            │
│ - Leaf nodes (no subordinates)                                  │
│ - Has upstream_id pointing to coordinator                       │
│ - Executes atomic actions                                       │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ - receive_upstream_actions() -> Get action from coordinator     │
│ - set_action() -> Apply received or computed action             │
│ - apply_action() -> Update state based on action                │
│ - compute_local_reward() -> Calculate local reward              │
└─────────────────────────────────────────────────────────────────┘
```

### **Action Passing Protocol**

**Priority Rules:**
1. **Upstream action** from parent (highest priority)
2. **Policy-computed action** if agent has policy
3. **No action** (neutral)

```python
def determine_action(self) -> Optional[Action]:
    """Determine which action to use based on priority."""
    # Priority 1: Check for upstream action from parent
    upstream_actions = self.receive_upstream_actions()
    if upstream_actions:
        return upstream_actions[0]

    # Priority 2: Compute via policy if available
    if self.policy is not None:
        obs = self.proxy.get_observation(self.agent_id, self.level)
        return self.policy.forward(observation=obs)

    # Priority 3: No action
    return None
```

### **Action Flow Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│ SystemAgent computes subordinate actions via protocol           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ protocol.coordinate(                                        │ │
│ │     coordinator_state=self.state,                           │ │
│ │     coordinator_action=action,                              │ │
│ │     info_for_subordinates={coord_id: obs for coords}        │ │
│ │ )                                                           │ │
│ │ -> Returns (messages, subordinate_actions)                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ For each coordinator:                                           │
│     send_subordinate_action(coord_id, coord_action)             │
│     └─> broker.publish(action_channel, Message(action))         │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ ACTION Message via broker
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CoordinatorAgent.tick()                                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ upstream_actions = receive_upstream_actions()               │ │
│ │ -> broker.consume(action_channel)                           │ │
│ │ -> Returns [Action from system]                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ If upstream_actions:                                            │
│     action = upstream_actions[0]  # Use parent's decision       │
│ Else:                                                           │
│     action = policy.forward(obs)  # Own decision                │
│                                                                 │
│ protocol.coordinate() -> field_actions                          │
│ For each field:                                                 │
│     send_subordinate_action(field_id, field_action)             │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ ACTION Message via broker
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ FieldAgent.tick()                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ upstream_actions = receive_upstream_actions()               │ │
│ │ -> broker.consume(action_channel)                           │ │
│ │ -> Returns [Action from coordinator]                        │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ If upstream_actions:                                            │
│     action = upstream_actions[0]  # Use parent's decision       │
│ Else:                                                           │
│     action = policy.forward(obs)  # Own decision                │
│                                                                 │
│ set_action(action)                                              │
│ schedule_action_effect(delay=act_delay)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Message Broker Architecture**

### **Message Structure**

```python
@dataclass
class Message:
    message_type: MessageType  # ACTION, INFO, STATE_UPDATE, etc.
    sender_id: str             # Who sent the message
    recipient_id: str          # Who receives the message
    payload: Dict[str, Any]    # Message content
    timestamp: Optional[float] = None
    env_id: str = "default"
```

### **Message Types**

```python
class MessageType(Enum):
    ACTION = "action"          # Parent -> Child action commands
    INFO = "info"              # Information/observation requests
    STATE_UPDATE = "state"     # State synchronization
    BROADCAST = "broadcast"    # Multi-recipient messages
    CUSTOM = "custom"          # Domain-specific types
```

### **Channel Naming Convention**

```python
class ChannelManager:
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

### **Broker Operations**

```python
class MessageBroker(ABC):
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

    @abstractmethod
    def clear(self) -> None:
        """Clear all channels (used on reset)."""
        pass
```

### **In-Memory Broker (Default)**

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

    def clear(self) -> None:
        for channel in self._channels.values():
            while not channel.empty():
                channel.get_nowait()
```

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
│   │   ├─> Creates ProxyAgent()
│   │   └─> Creates MessageBroker()
│   │
│   ├─> _initialize_agents()
│   │   ├─> For each agent in hierarchy:
│   │   │   ├─> agent.init_action() -> Creates Action object
│   │   │   ├─> agent.init_state() -> Creates State object
│   │   │   ├─> proxy.register_agent(agent_id, agent_state=agent.state)
│   │   │   │   └─> proxy.set_local_state(agent.state)  # State object!
│   │   │   └─> [Recurse for subordinates]
│   │   │
│   │   └─> proxy.init_global_state()
│   │
│   ├─> MessageBroker.init() and attach agents
│   └─> EventScheduler.init() and attach agents
```

### **2. Reset Flow**

```
obs, info = env.reset(seed=42)
│
├─> scheduler.reset(seed) -> Clear event queue
├─> broker.clear() -> Clear all message channels
├─> proxy.reset(seed)
│   └─> state_cache = {} (cleared)
│
├─> system_agent.reset(seed, proxy)
│   ├─> self._timestep = 0.0
│   ├─> init_action() -> Reset action to neutral
│   ├─> init_state() -> Create fresh state
│   ├─> proxy.set_local_state(self.agent_id, self.state)  # State object!
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
│   └─> return self.observe(proxy) -> Dict[AgentID, Observation]
│
├─> proxy.init_global_state()
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
    │       │               "field_1": actions["field_1"],
    │       │               "field_2": actions["field_2"]
    │       │             }
    │       │           }
    │       │         }
    │       │       }
    │       │
    │       ├─> handle_self_action(layered['self'], proxy)
    │       │   ├─> if action provided:
    │       │   │   └─> set_action(action)
    │       │   ├─> elif has policy:
    │       │   │   ├─> obs = proxy.get_observation(agent_id, level)
    │       │   │   └─> set_action(policy.forward(observation=obs))
    │       │   │
    │       │   ├─> apply_action() -> Updates self.state
    │       │   └─> proxy.set_local_state(self.agent_id, self.state)
    │       │
    │       └─> handle_subordinate_actions(layered['subordinates'], proxy)
    │           └─> For each subordinate:
    │               └─> subordinate.execute(layered_actions[sub_id], proxy)
    │                   └─> [Recursively calls act() on subordinates]
    │
    ├─> PHASE 2: Simulation
    │   │
    │   ├─> global_state = proxy.get_global_states(system_agent_id, system_level)
    │   │   └─> Returns filtered feature vectors for all agents
    │   │
    │   ├─> updated_global_state = simulate(global_state)
    │   │   ├─> env_state = global_state_to_env_state(global_state)
    │   │   ├─> updated_env_state = run_simulation(env_state)
    │   │   └─> return env_state_to_global_state(updated_env_state)
    │   │
    │   └─> proxy.set_global_state(updated_global_state)
    │
    ├─> PHASE 3: Observation Collection
    │   │
    │   └─> obs = self.observe(proxy)
    │       └─> For each agent in hierarchy:
    │           ├─> observation = proxy.get_observation(agent_id, level)
    │           │   ├─> local = state.observed_by(agent_id, level)
    │           │   ├─> global_info = {filtered states from other agents}
    │           │   └─> return Observation(local, global_info, timestamp)
    │           │
    │           └─> Returns: Dict[AgentID, Observation]
    │
    ├─> PHASE 4: Reward & Status Computation
    │   │
    │   ├─> rewards = self.compute_rewards(proxy)
    │   │   └─> For each agent:
    │   │       ├─> local_state = proxy.get_local_state(agent_id, level)
    │   │       └─> reward = compute_local_reward(local_state)
    │   │
    │   ├─> infos = self.get_info(proxy)
    │   ├─> terminateds = self.get_terminateds(proxy)
    │   └─> truncateds = self.get_truncateds(proxy)
    │
    ├─> PHASE 5: Cache Results
    │   │
    │   └─> proxy.set_step_result(obs, rewards, terminateds, truncateds, infos)
    │
    └─> RETURN: proxy.get_step_results()
        │
        ├─> obs_vectorized = {aid: obs.vector() for aid, obs in obs.items()}
        │
        └─> return (obs_vectorized, rewards, terminateds, truncateds, infos)
```

### **Key Features:**

- **Synchronous execution**: All agents act, then simulate, then observe
- **Centralized training**: Full observability via proxy
- **Type safety**: Observations are Observation objects, auto-vectorized for RL
- **State management**: Proxy maintains State objects with visibility filtering
- **Action passing**: Hierarchical action distribution via `layer_actions()`

---

## **Event-Driven Testing Mode**

### **1. Event Types**

| Event Type | When Scheduled | Handler | Purpose |
|------------|----------------|---------|---------|
| `AGENT_TICK` | At tick_interval | `agent_tick_handler()` | Trigger agent's tick() method |
| `MESSAGE_DELIVERY` | After msg_delay | `message_delivery_handler()` | Deliver async messages |
| `ACTION_EFFECT` | After act_delay | `action_effect_handler()` | Apply delayed action effects |
| `SIMULATION` | After wait_interval | `simulation_handler()` | Run physics simulation |

### **2. Event Priority**

```python
class EventType(Enum):
    AGENT_TICK = "agent_tick"              # Priority: default
    ACTION_EFFECT = "action_effect"        # Priority: 0 (highest)
    SIMULATION = "simulation"              # Priority: 1
    MESSAGE_DELIVERY = "message_delivery"  # Priority: 2
    OBSERVATION_READY = "observation_ready"
    ENV_UPDATE = "env_update"
    CUSTOM = "custom"
```

### **3. TickConfig - Timing Parameters**

```python
@dataclass
class TickConfig:
    tick_interval: float      # How often agent ticks (e.g., 1.0s)
    obs_delay: float          # Observation latency (e.g., 0.1s)
    act_delay: float          # Action execution delay (e.g., 0.5s)
    msg_delay: float          # Message delivery latency (e.g., 0.2s)

    jitter_type: JitterType   # DETERMINISTIC or GAUSSIAN
    jitter_ratio: float       # Jitter magnitude (0.1 = 10%)

    @classmethod
    def deterministic(cls, tick_interval=1.0, obs_delay=0.1, act_delay=0.5, msg_delay=0.2):
        return cls(tick_interval, obs_delay, act_delay, msg_delay, JitterType.DETERMINISTIC, 0.0)

    @classmethod
    def with_jitter(cls, tick_interval=1.0, jitter_ratio=0.1, jitter_type=JitterType.GAUSSIAN):
        return cls(tick_interval, ..., jitter_type, jitter_ratio)
```

### **4. Event-Driven Execution Flow**

```
result = env.run_event_driven(event_analyzer, t_end=100.0)
│
└─> for event in scheduler.run_until(t_end=100.0):
    │
    ├─> EventScheduler.process_next()
    │   ├─> event = pop() from priority queue (by timestamp, priority, sequence)
    │   ├─> current_time = event.timestamp
    │   ├─> handler = get_handler(event.event_type, event.agent_id)
    │   └─> handler(event, scheduler)
    │
    └─> result.add_event_analysis(event_analyzer.parse_event(event))
```

### **5. Detailed Event Sequence**

#### **Timeline Example** (System -> Coordinator -> 2 Field Agents):

```
t=0.0: AGENT_TICK(system_agent)
├─> SystemAgent.tick()
│   │
│   ├─> self._timestep = 0.0
│   │
│   ├─> Schedule subordinate ticks:
│   │   └─> schedule_agent_tick(coordinator_1)
│   │       └─> timestamp = 0.0 + coordinator.tick_interval
│   │
│   ├─> Action passing to subordinates:
│   │   ├─> protocol.coordinate() -> subordinate_actions
│   │   └─> For each coordinator:
│   │       └─> send_subordinate_action(coord_id, action)
│   │           └─> broker.publish(action_channel, Message(action))
│   │
│   ├─> If has policy: Request observation
│   │   └─> schedule_message_delivery(
│   │           sender=system_agent, recipient=proxy,
│   │           message={"get_info": "obs"},
│   │           delay=msg_delay
│   │       )
│   │
│   └─> Schedule simulation:
│       └─> schedule_simulation(system_agent, wait_interval)
```

```
t=msg_delay: MESSAGE_DELIVERY(proxy) - "get_info:obs" request
└─> ProxyAgent.message_delivery_handler()
    ├─> obs = get_observation(system_agent, system_level)
    │   ├─> local = state.observed_by(system_id, 3)
    │   ├─> global_info = {all agents' observed_by() filtered states}
    │   └─> return Observation(local, global_info, timestamp)
    │
    └─> schedule_message_delivery(
            sender=proxy, recipient=system_agent,
            message={"get_obs_response": {"body": obs.to_dict()}},
            delay=msg_delay
        )
```

```
t=2*msg_delay: MESSAGE_DELIVERY(system_agent) - "get_obs_response"
└─> SystemAgent.message_delivery_handler()
    ├─> obs = Observation.from_dict(message['body'])
    │
    └─> compute_action(obs, scheduler)
        ├─> action = policy.forward(observation=obs)
        │   └─> obs.__array__() auto-converts to np.ndarray
        ├─> set_action(action)
        └─> schedule_action_effect(agent_id, delay=act_delay)
```

```
t=2*msg_delay+act_delay: ACTION_EFFECT(system_agent)
└─> Agent.action_effect_handler()
    ├─> apply_action() -> Updates self.state
    └─> schedule_message_delivery(
            sender=system_agent, recipient=proxy,
            message={"set_state": "local", "body": self.state.to_dict(include_metadata=True)},
            delay=msg_delay
        )
```

```
t=3*msg_delay+act_delay: MESSAGE_DELIVERY(proxy) - "set_state:local"
└─> ProxyAgent.message_delivery_handler()
    ├─> state = State.from_dict(message['body'])
    ├─> set_local_state(agent_id, state)
    └─> schedule_message_delivery(
            message={"set_state_completion": "success"}
        )
```

```
t=coordinator_tick_interval: AGENT_TICK(coordinator_1)
└─> CoordinatorAgent.tick()
    │
    ├─> Receive upstream action:
    │   upstream_actions = receive_upstream_actions()
    │   └─> broker.consume(action_channel)
    │   └─> Returns: [Action from system_agent]
    │
    ├─> Action decision:
    │   If upstream_actions:
    │       action = upstream_actions[0]  # Use parent's decision!
    │   Else:
    │       action = policy.forward(obs)  # Own decision
    │
    ├─> set_action(action)
    │
    ├─> Action passing to subordinates:
    │   ├─> protocol.coordinate() -> field_actions
    │   └─> For each field:
    │       └─> send_subordinate_action(field_id, action)
    │
    └─> Schedule subordinate ticks
```

```
t=field_tick_interval: AGENT_TICK(field_1)
└─> FieldAgent.tick()
    │
    ├─> Receive upstream action:
    │   upstream_actions = receive_upstream_actions()
    │   └─> Returns: [Action from coordinator_1]
    │
    ├─> Action decision:
    │   If upstream_actions:
    │       action = upstream_actions[0]  # Use parent's decision!
    │   Else:
    │       action = policy.forward(obs)
    │
    ├─> set_action(action)
    │
    └─> schedule_action_effect(delay=act_delay)
```

```
t=wait_interval: SIMULATION(system_agent)
└─> SystemAgent.simulation_handler()
    └─> schedule_message_delivery(
            message={"get_info": "global_state"}
        )
```

```
t=wait_interval+msg_delay: MESSAGE_DELIVERY(proxy) - "get_info:global_state"
└─> ProxyAgent returns global_state
```

```
t=wait_interval+2*msg_delay: MESSAGE_DELIVERY(system_agent) - "get_global_state_response"
└─> SystemAgent.message_delivery_handler()
    ├─> global_state = message['body']
    ├─> updated_global_state = simulate(global_state)
    │   ├─> env_state = global_state_to_env_state(global_state)
    │   ├─> updated_env_state = run_simulation(env_state)
    │   └─> return env_state_to_global_state(updated_env_state)
    │
    └─> schedule_message_delivery(
            message={"set_state": "global", "body": updated_global_state}
        )
```

```
t=reward_time: MESSAGE_DELIVERY(agents) - "get_local_state_response"
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
```

### **6. Result Collection via EventAnalyzer**

```
env.run_event_driven(event_analyzer, t_end)
│
└─> For each event:
    └─> event_analyzer.parse_event(event)
        │
        ├─> If event.event_type == MESSAGE_DELIVERY:
        │   ├─> message = event.payload["message"]
        │   │
        │   └─> If "set_tick_result" in message:
        │       ├─> sender_id = event.payload["sender"]
        │       ├─> tick_result = message["body"]
        │       └─> EventAnalyzer accumulates results per agent
        │
        └─> Returns EventAnalysis for this event

EpisodeResult accumulates all EventAnalysis objects
```

---

## **Protocol-Based Coordination**

### **Protocol Architecture**

```python
class Protocol(ABC):
    """Base class for coordination protocols."""

    @abstractmethod
    def coordinate(
        self,
        coordinator_state: State,
        coordinator_action: Optional[Action],
        info_for_subordinates: Dict[str, Observation],
        context: Dict
    ) -> Tuple[Dict[str, Dict], Dict[str, Action]]:
        """
        Compute coordination messages and subordinate actions.

        Returns:
            messages: Dict of coordination messages per subordinate
            actions: Dict of actions per subordinate
        """
        pass
```

### **Protocol Implementations**

#### **1. NoProtocol (Default)**
```python
class NoProtocol(Protocol):
    """No coordination - agents act independently."""

    def coordinate(self, coordinator_state, coordinator_action, info, context):
        # No messages, no action decomposition
        return {}, {}
```

#### **2. VerticalProtocol (Hierarchical)**
```python
class VerticalProtocol(Protocol):
    """Hierarchical coordination - parent controls subordinates."""

    def coordinate(self, coordinator_state, coordinator_action, info, context):
        messages = {}
        actions = {}

        for sub_id, sub_obs in info.items():
            # Compute subordinate-specific message
            messages[sub_id] = self._compute_message(coordinator_state, sub_obs)

            # Decompose action for subordinate
            actions[sub_id] = self._decompose_action(coordinator_action, sub_id)

        return messages, actions

    def _decompose_action(self, coordinator_action, subordinate_id):
        """Decompose coordinator action into subordinate action."""
        # Example: Split action components
        return subordinate_action
```

#### **3. HorizontalProtocol (Peer Coordination)**
```python
class HorizontalProtocol(Protocol):
    """Peer-to-peer coordination - agents coordinate laterally."""

    def coordinate(self, coordinator_state, coordinator_action, info, context):
        # Consensus-based action computation
        actions = self._compute_consensus_actions(info)
        messages = self._compute_peer_messages(info)
        return messages, actions
```

### **Protocol Usage**

```python
# In CoordinatorAgent.act() or tick()
def coordinate_subordinates(self, action: Action, proxy: ProxyAgent):
    # Get subordinate observations
    info = {
        sub_id: proxy.get_observation(sub_id, sub.level)
        for sub_id, sub in self.subordinates.items()
    }

    # Execute protocol
    messages, sub_actions = self.protocol.coordinate(
        coordinator_state=self.state,
        coordinator_action=action,
        info_for_subordinates=info,
        context={"timestamp": self._timestep}
    )

    # Send coordination messages
    for sub_id, message in messages.items():
        self.send_info(broker=self._message_broker, recipient_id=sub_id, info=message)

    # Send/pass subordinate actions
    for sub_id, sub_action in sub_actions.items():
        self.send_subordinate_action(sub_id, sub_action)
```

---

## **Observation & Action Data Flow**

### **Observation Flow (Both Modes)**

```
Agent needs observation
│
├─> CTDE Mode: Direct call
│   └─> obs = proxy.get_observation(agent_id, level)
│       ├─> local = state.observed_by(agent_id, level)
│       ├─> global_info = {filtered states from other agents}
│       └─> return Observation(local, global_info, timestamp)
│
├─> Event-Driven Mode: Via message
│   ├─> Request: schedule_message_delivery({"get_info": "obs"})
│   └─> Response: {"get_obs_response": {"body": obs.to_dict()}}
│       └─> obs = Observation.from_dict(response['body'])
│
└─> ProxyAgent.get_observation(sender_id, level)
    │
    ├─> local_state = get_local_state(sender_id, level)
    │   └─> state.observed_by(sender_id, level)
    │   └─> Returns: {"FeatureName": np.array([...])}
    │
    ├─> global_state = get_global_states(sender_id, level)
    │   └─> For each agent: state.observed_by(sender_id, level)
    │   └─> Returns: {"agent_id": {"FeatureName": np.array([...])}}
    │
    └─> return Observation(
            local=local_state,
            global_info=global_state,
            timestamp=current_time
        )
```

### **Observation Array Interface**

```python
obs = Observation(local={"voltage": np.array([1.02])}, global_info={"freq": 60.0})

# Automatic conversion for policies:
policy.forward(observation=obs)  # __array__() converts to np.ndarray

# Array-like properties:
obs.shape    # -> (2,)
obs.dtype    # -> float32
len(obs)     # -> 2
obs[0]       # -> 1.02
obs[:1]      # -> array([1.02])

# Explicit vectorization:
vec = obs.vector()  # -> array([1.02, 60.0], dtype=float32)
```

### **Action Flow**

```
CTDE Mode:
  agent.act(actions, proxy)
  ├─> layer_actions(actions) -> Hierarchical structure
  ├─> handle_self_action(action)
  │   ├─> set_action(action) -> self.action.set_values(action)
  │   └─> apply_action() -> Updates self.state immediately
  ├─> handle_subordinate_actions(subordinate_actions)
  │   └─> For each: subordinate.execute(action, proxy)
  └─> proxy.set_local_state(self.state)

Event-Driven Mode:
  agent.tick(scheduler)
  ├─> upstream_actions = receive_upstream_actions()
  ├─> If upstream_actions: action = upstream_actions[0]
  │   Else: action via policy or None
  ├─> set_action(action)
  └─> schedule_action_effect(delay=act_delay) -> Delayed!

  [Later, at ACTION_EFFECT event:]
  agent.action_effect_handler()
  ├─> apply_action() -> Updates self.state after delay
  └─> schedule message to update proxy
```

**Key Difference:**
- **CTDE**: Actions apply immediately, state updated in proxy synchronously
- **Event-Driven**: Actions apply after delay, state updated via messages

---

## **ProxyAgent State Management**

### **State Cache Structure**

```python
proxy.state_cache = {
    # Per-agent states (State objects!)
    "agents": {
        "system_agent": SystemAgentState(...),
        "coordinator_1": CoordinatorAgentState(...),
        "field_1": FieldAgentState(...),
        "field_2": FieldAgentState(...),
    },

    # Global state (environment-wide data)
    "global": {
        "grid_frequency": 60.0,
        "total_load": 1500.0,
        ...
    }
}

# Agent levels for visibility checks
proxy._agent_levels = {
    "field_1": 1,
    "field_2": 1,
    "coordinator_1": 2,
    "system_agent": 3
}
```

### **State Operations**

| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `set_local_state(aid, state)` | State object | None | Stores object directly |
| `get_local_state(aid, level)` | agent_id, requestor_level | Dict[str, np.ndarray] | Returns filtered vectors |
| `get_global_states(aid, level)` | agent_id, requestor_level | Dict[aid, Dict] | Returns all filtered states |
| `get_observation(aid, level)` | agent_id, level | Observation | Combines local + global |
| `set_global_state(dict)` | global_dict | None | Updates global state |

### **Visibility Filtering**

```python
def get_local_state(self, agent_id: str, requestor_level: int) -> Dict[str, np.ndarray]:
    """Get agent's state filtered by visibility rules."""
    state_obj = self.state_cache["agents"][agent_id]
    return state_obj.observed_by(agent_id, requestor_level)
    # Returns: {"BatteryChargeFeature": np.array([0.5, 100.0])}

def get_global_states(self, sender_id: str, requestor_level: int) -> Dict[str, Dict]:
    """Get all agents' states filtered by visibility."""
    result = {}
    for agent_id, state_obj in self.state_cache["agents"].items():
        if agent_id != sender_id:  # Exclude self
            filtered = state_obj.observed_by(sender_id, requestor_level)
            if filtered:  # Only include if any features visible
                result[agent_id] = filtered
    return result
```

---

## **Handler Registration & Event Processing**

### **Handler Registration**

```python
class Agent:
    _event_handler_funcs: Dict[str, Callable] = {}

    @classmethod
    def handler(cls, event_type: str):
        """Decorator to register event handlers."""
        def decorator(func):
            cls._event_handler_funcs[event_type] = func
            return func
        return decorator

    def get_handlers(self) -> Dict[str, Callable]:
        """Get bound handler methods."""
        return {
            event_type: getattr(self, func.__name__)
            for event_type, func in self._event_handler_funcs.items()
        }
```

### **Handler Implementation Example**

```python
class FieldAgent(Agent):

    @Agent.handler("agent_tick")
    def agent_tick_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Handle agent tick event."""
        self.tick(scheduler, event.timestamp)

    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Handle action effect event."""
        self.apply_action()
        # Schedule state update message
        self._schedule_state_update(scheduler)

    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Handle message delivery event."""
        message = event.payload["message"]
        sender_id = event.payload["sender"]

        if "get_obs_response" in message:
            obs = Observation.from_dict(message["get_obs_response"]["body"])
            self.compute_action(obs, scheduler)

        elif "get_local_state_response" in message:
            local_state = message["get_local_state_response"]["body"]
            self._compute_and_send_tick_result(local_state, scheduler)
```

### **Event Processing Loop**

```python
class EventScheduler:
    def run_until(self, t_end: float, max_events: int = 100000) -> Iterator[Event]:
        """Run simulation until time limit or max events."""
        events_processed = 0

        while events_processed < max_events and self.event_queue:
            event = self.process_next()

            if event.timestamp > t_end:
                break

            yield event
            events_processed += 1

    def process_next(self) -> Event:
        """Pop and process next event."""
        event = heapq.heappop(self.event_queue)
        self.current_time = event.timestamp

        handler = self.get_handler(event.event_type, event.agent_id)
        if handler:
            handler(event, self)

        return event
```

---

## **Complete Execution Examples**

### **CTDE Training Step**

```python
# Step 0: Reset
env.reset()  # All agents initialized, proxy has states

# Step 1: RL algorithm provides actions
actions = {
    "system_agent": Action(c=[...]),
    "coordinator_1": Action(c=[...]),
    "field_1": Action(c=[0.3]),
    "field_2": Action(c=[-0.2]),
}

# Step 2: Environment executes
obs, rewards, terminated, truncated, info = env.step(actions)

# Internal execution:
# 1. SystemAgent.execute(actions, proxy)
#    a. layer_actions() -> hierarchical structure
#    b. handle_self_action() + handle_subordinate_actions()
#       - Each agent: set_action(), apply_action(), proxy.set_local_state()
#    c. simulate() -> Physics update
#    d. observe() -> Collect observations (with visibility filtering)
#    e. compute_rewards() -> Compute from filtered states
# 2. proxy.set_step_result() -> Cache results
# 3. proxy.get_step_results() -> Vectorize and return

# Step 3: RL algorithm trains
# obs: Dict[AgentID, np.ndarray] ready for neural network input
```

### **Event-Driven Episode**

```python
# Setup
env.reset()
event_analyzer = EventAnalyzer()

# Run simulation
episode_result = env.run_event_driven(event_analyzer, t_end=100.0)

# Internal execution:
# - Events processed in timestamp order
# - Agents tick at their configured intervals
# - Actions passed via broker from parents to children
# - Actions apply with realistic delays
# - States updated via messages
# - EventAnalyzer extracts results from event stream

# Retrieve results
episode_result.get_rewards()      # Extracted from tick_result messages
episode_result.get_statistics()   # Event counts, timing analysis
```

---

## **Comparison: CTDE vs Event-Driven**

| Aspect | CTDE Training | Event-Driven Testing |
|--------|---------------|----------------------|
| **Execution** | Synchronous (step-based) | Asynchronous (event-based) |
| **Timing** | Instantaneous | Realistic delays (msg_delay, act_delay) |
| **Action Application** | Immediate in `act()` | Delayed via ACTION_EFFECT |
| **Action Passing** | Via `layer_actions()` dict | Via broker messages |
| **State Updates** | Synchronous to proxy | Asynchronous via messages |
| **Observation Collection** | Direct function calls | Message-based requests |
| **Result Retrieval** | `proxy.get_step_results()` | EventAnalyzer from events |
| **Hierarchy Coordination** | Sequential function calls | Cascading tick events |
| **Use Case** | RL training with full observability | Deployment testing with latency |

---

## **Critical Design Patterns**

### **1. Two-Phase State Updates**

**CTDE Mode:**
```python
# Phase 1: Act
agent.act(actions, proxy)
└─> apply_action() -> Updates state
└─> proxy.set_local_state(state) -> Synchronous

# Phase 2: Simulate
simulate(global_state) -> Physics update
proxy.set_global_state(updated_global_state)
```

**Event-Driven Mode:**
```python
# Phase 1: Action computed
compute_action(obs) -> Set action
schedule_action_effect(delay)

# Phase 2: Action applied (after delay)
action_effect_handler()
└─> apply_action() -> Updates state
└─> schedule message to update proxy -> Asynchronous
```

### **2. Action Priority Cascade**

```
SystemAgent
├─> Computes action via policy
├─> protocol.coordinate() -> coordinator actions
└─> send_subordinate_action() to coordinators

    CoordinatorAgent
    ├─> receive_upstream_actions() -> Gets system's decision
    ├─> Uses upstream action (priority!)
    ├─> protocol.coordinate() -> field actions
    └─> send_subordinate_action() to fields

        FieldAgent
        ├─> receive_upstream_actions() -> Gets coordinator's decision
        ├─> Uses upstream action (priority!)
        └─> apply_action()
```

### **3. Visibility-Based State Access**

```python
# All state access goes through observed_by()
local_state = state.observed_by(requestor_id, requestor_level)

# Features filtered by visibility rules:
# - "public": All agents can see
# - "owner": Only owning agent can see
# - "upper_level": Only one level above can see
# - "system": Only system-level (L3) can see
```

### **4. Message-Action Duality**

```python
# CTDE: Direct action passing
subordinate.execute(action, proxy)

# Event-Driven: Message-based action passing
send_subordinate_action(subordinate_id, action)
└─> broker.publish(action_channel, Message(action))
```

---

## **Summary: Both Modes Are Production-Ready!**

### **CTDE Training Mode**
- Proper hierarchical action distribution via `layer_actions()`
- Synchronized state management via proxy with State objects
- Observations auto-vectorized for RL algorithms
- Visibility filtering via `state.observed_by()`
- Full observability for centralized training

### **Event-Driven Testing Mode**
- Asynchronous execution with realistic timing
- Hierarchical tick cascade (agents tick at different rates)
- Action passing via message broker
- Message-based state distribution
- Delayed action effects
- Results collected via EventAnalyzer from event stream
- Tests deployment scenario with communication delays

**Both modes share the same agent hierarchy, state management infrastructure, and action passing protocol, ensuring consistency between training and testing!**

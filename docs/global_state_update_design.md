# Global State Update Design: Avoiding Direct Network Modification

## Problem Statement

In the current `NetworkedGridEnv.step()`, agents directly modify the global pandapower network:

```python
# Current problematic design
for agent in self.agent_dict.values():
    agent.update_state(self.net, self._t)  # Direct mutation of global state!
```

This violates the Kafka-based design principles:
1. **Breaks encapsulation**: Agents reach into global environment state
2. **Not message-based**: Direct method calls instead of Kafka messages
3. **Concurrency issues**: Can't parallelize if agents mutate shared state
4. **Batched rollouts**: Multiple environments would share/corrupt `self.net`

## Design Principles

1. **Agents produce state updates as messages**: Instead of mutating `net`, agents send state update messages
2. **Environment applies updates**: Environment collects messages and applies them to `net`
3. **Unidirectional flow**: Agent → Kafka → Environment → Global State
4. **Batch-safe**: Each environment has its own `net`, agents send to their env's Kafka namespace

## Solution: State Update Messages

### Approach: Agents Publish State Updates via Kafka

```
┌──────────────────────────────────────────────────────────────────┐
│                     AGENT STEP FLOW                              │
└──────────────────────────────────────────────────────────────────┘

Agent.execute_own_action():
  1. Compute new device state (P, Q, SoC, etc.)
  2. Package state update as message
  3. Publish to Kafka: env_{env_id}__state_updates

Environment.step():
  1. Trigger agent.step() for all agents (recursive)
  2. Collect state update messages from Kafka
  3. Apply all state updates to self.net
  4. Run power flow on self.net
  5. Collect power flow results
  6. Send results back to agents via Kafka
```

### Implementation

#### 1. New Message Type: STATE_UPDATE

```python
class MessageType(Enum):
    ACTION = "action"
    INFO = "info"
    BROADCAST = "broadcast"
    STATE_UPDATE = "state_update"  # NEW
    POWER_FLOW_RESULT = "power_flow_result"  # NEW
```

#### 2. State Update Message Schema

```python
@dataclass
class StateUpdate:
    """State update from agent to environment."""
    agent_id: str
    device_id: str
    device_type: str  # 'sgen', 'storage', 'gen', 'load'
    updates: Dict[str, Any]  # {'p_mw': 1.5, 'q_mvar': 0.3, 'in_service': True}

@dataclass
class PowerFlowResult:
    """Power flow results from environment to agent."""
    agent_id: str
    device_results: Dict[str, Dict[str, float]]  # {device_id: {p_mw: ..., q_mvar: ...}}
    bus_results: Dict[str, Dict[str, float]]  # {bus_id: {vm_pu: ..., va_degree: ...}}
    converged: bool
```

#### 3. Updated Agent: Publish State Updates

```python
class Agent(ABC):
    """Agent publishes state updates instead of mutating global state."""

    def execute_own_action(
        self,
        action: Any,
        global_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute action and publish state updates to environment.

        Instead of modifying global state directly, agents compute their
        new state and send it as a message to the environment.
        """
        # Compute new device state
        state_updates = self._compute_state_updates(action)

        # Publish to environment via Kafka
        for state_update in state_updates:
            self._publish_state_update(state_update)

    def _publish_state_update(self, state_update: StateUpdate) -> None:
        """Publish state update to environment via Kafka."""
        if not self.kafka_broker:
            return

        topic = TopicManager.state_update_topic(self.env_id)
        message = KafkaMessage(
            env_id=self.env_id,
            sender_id=self.agent_id,
            recipient_id="environment",
            timestamp=self._timestep,
            message_type=MessageType.STATE_UPDATE,
            payload={
                'agent_id': state_update.agent_id,
                'device_id': state_update.device_id,
                'device_type': state_update.device_type,
                'updates': state_update.updates
            }
        )
        self.kafka_broker.publish(topic, message)

    @abstractmethod
    def _compute_state_updates(self, action: Any) -> List[StateUpdate]:
        """Compute state updates from action (subclass-specific)."""
        pass
```

#### 4. DeviceAgent: Compute Device State Updates

```python
class DeviceAgent(Agent):
    """DeviceAgent computes its own state update."""

    def _compute_state_updates(self, action: Any) -> List[StateUpdate]:
        """Compute state update for this device."""
        if action is None:
            return []

        # Set device action
        self._set_device_action(action)

        # Update internal device state (without pandapower)
        self.update_state()  # No self.net argument!

        # Package state for pandapower
        state_update = StateUpdate(
            agent_id=self.agent_id,
            device_id=self.agent_id,  # Device ID same as agent ID
            device_type=self._get_device_type(),
            updates=self._get_pandapower_updates()
        )

        return [state_update]

    @abstractmethod
    def _get_device_type(self) -> str:
        """Return pandapower device type ('sgen', 'storage', etc.)."""
        pass

    @abstractmethod
    def _get_pandapower_updates(self) -> Dict[str, Any]:
        """Return dict of pandapower attributes to update."""
        pass

    @abstractmethod
    def update_state(self, *args, **kwargs) -> None:
        """Update device internal state (no pandapower access)."""
        pass
```

#### 5. GridAgent: Aggregate Subordinate Updates

```python
class GridAgent(Agent):
    """GridAgent aggregates state updates from subordinate devices."""

    def _compute_state_updates(self, action: Any) -> List[StateUpdate]:
        """Aggregate state updates from all subordinate devices.

        GridAgent doesn't have its own device state, so it just collects
        state updates published by its subordinate DeviceAgents.
        """
        # Subordinates already published their state updates during their step()
        # GridAgent just returns empty list (or could aggregate)
        return []

    def execute_own_action(
        self,
        action: Any,
        global_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """GridAgent has no own action execution (coordination only)."""
        # Grid agents coordinate but don't directly control devices
        # State updates come from subordinate DeviceAgents
        pass
```

#### 6. Environment: Apply State Updates

```python
class KafkaMultiAgentEnv(gym.Env):
    """Environment collects and applies state updates."""

    def step(self, actions: Dict[AgentID, Any]) -> tuple:
        """Execute step with state update collection."""
        # 1. Send actions to root agents
        for agent_id, action in actions.items():
            self._send_action_to_agent(agent_id, action)

        # 2. Execute agent steps (recursive, publishes state updates)
        all_info = {}
        for agent_id in self.root_agents:
            agent = self.agents[agent_id]
            compiled_info = agent.step(global_state=self.get_global_state())
            all_info[agent_id] = compiled_info

        # 3. Collect state updates from Kafka
        state_updates = self._collect_state_updates()

        # 4. Apply state updates to pandapower network
        self._apply_state_updates(state_updates)

        # 5. Run power flow
        try:
            pp.runpp(self.net)
            self.net['converged'] = True
        except:
            self.net['converged'] = False

        # 6. Extract power flow results and send to agents
        self._publish_power_flow_results()

        # 7. Extract observations, rewards
        observations = self._extract_observations(all_info)
        rewards = self._extract_rewards(all_info)
        dones = {agent_id: False for agent_id in self.root_agents}
        infos = all_info

        self.timestep += 1
        return observations, rewards, dones, infos

    def _collect_state_updates(self) -> List[StateUpdate]:
        """Collect all state update messages from Kafka."""
        topic = TopicManager.state_update_topic(self.env_id)
        messages = self.kafka_broker.consume(
            topic,
            agent_id="environment",
            env_id=self.env_id
        )

        state_updates = []
        for msg in messages:
            state_update = StateUpdate(
                agent_id=msg.payload['agent_id'],
                device_id=msg.payload['device_id'],
                device_type=msg.payload['device_type'],
                updates=msg.payload['updates']
            )
            state_updates.append(state_update)

        return state_updates

    def _apply_state_updates(self, state_updates: List[StateUpdate]) -> None:
        """Apply state updates to pandapower network."""
        for update in state_updates:
            self._apply_single_update(update)

    def _apply_single_update(self, update: StateUpdate) -> None:
        """Apply single state update to pandapower network.

        Args:
            update: StateUpdate with device_type, device_id, and updates dict
        """
        # Get device index in pandapower
        try:
            device_idx = pp.get_element_index(
                self.net,
                update.device_type,
                update.device_id
            )
        except Exception as e:
            print(f"Warning: Could not find {update.device_type} '{update.device_id}': {e}")
            return

        # Apply updates to pandapower table
        for attr, value in update.updates.items():
            if attr in self.net[update.device_type].columns:
                self.net[update.device_type].loc[device_idx, attr] = value
            else:
                print(f"Warning: Unknown attribute '{attr}' for {update.device_type}")

    def _publish_power_flow_results(self) -> None:
        """Publish power flow results to all agents via Kafka."""
        for agent_id, agent in self.agents.items():
            # Extract relevant results for this agent
            device_results = self._extract_device_results(agent)
            bus_results = self._extract_bus_results(agent)

            # Create power flow result message
            topic = TopicManager.power_flow_result_topic(self.env_id, agent_id)
            message = KafkaMessage(
                env_id=self.env_id,
                sender_id="environment",
                recipient_id=agent_id,
                timestamp=self.timestep,
                message_type=MessageType.POWER_FLOW_RESULT,
                payload={
                    'device_results': device_results,
                    'bus_results': bus_results,
                    'converged': self.net['converged']
                }
            )
            self.kafka_broker.publish(topic, message)

    def _extract_device_results(self, agent: Agent) -> Dict[str, Dict[str, float]]:
        """Extract power flow results for agent's devices."""
        results = {}

        # For GridAgent, extract results for all subordinate devices
        if isinstance(agent, GridAgent):
            for device_id, device in agent.subordinates.items():
                device_type = device._get_device_type()
                try:
                    idx = pp.get_element_index(self.net, device_type, device_id)
                    result_table = self.net[f'res_{device_type}']
                    results[device_id] = result_table.loc[idx].to_dict()
                except:
                    results[device_id] = {}

        # For DeviceAgent, extract result for this device
        elif isinstance(agent, DeviceAgent):
            device_type = agent._get_device_type()
            try:
                idx = pp.get_element_index(self.net, device_type, agent.agent_id)
                result_table = self.net[f'res_{device_type}']
                results[agent.agent_id] = result_table.loc[idx].to_dict()
            except:
                results[agent.agent_id] = {}

        return results

    def _extract_bus_results(self, agent: Agent) -> Dict[str, Dict[str, float]]:
        """Extract bus results relevant to this agent."""
        # TODO: Implement based on agent's bus connections
        return {}
```

#### 7. Updated Topic Manager

```python
class TopicManager:
    """Topic management with state update topics."""

    @staticmethod
    def state_update_topic(env_id: str) -> str:
        """Topic for agents to publish state updates to environment."""
        return f"env_{env_id}__state_updates"

    @staticmethod
    def power_flow_result_topic(env_id: str, agent_id: str) -> str:
        """Topic for environment to publish power flow results to agent."""
        return f"env_{env_id}__power_flow_results_{agent_id}"
```

#### 8. Agent Receives Power Flow Results

```python
class Agent(ABC):
    """Agent receives power flow results from environment."""

    def step(self, global_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Step with power flow result handling."""
        # ... existing step logic ...

        # After executing subordinates and own action
        # Wait for power flow results from environment
        power_flow_results = self._receive_power_flow_results()

        # Update internal state with power flow results
        self._sync_power_flow_results(power_flow_results)

        # Continue with observation and info compilation
        # ...

    def _receive_power_flow_results(self) -> Optional[PowerFlowResult]:
        """Receive power flow results from environment via Kafka."""
        if not self.kafka_broker:
            return None

        topic = TopicManager.power_flow_result_topic(self.env_id, self.agent_id)
        messages = self.kafka_broker.consume(topic, self.agent_id, self.env_id)

        if messages:
            latest_msg = messages[-1]
            return PowerFlowResult(
                agent_id=self.agent_id,
                device_results=latest_msg.payload['device_results'],
                bus_results=latest_msg.payload['bus_results'],
                converged=latest_msg.payload['converged']
            )

        return None

    def _sync_power_flow_results(self, results: Optional[PowerFlowResult]) -> None:
        """Sync power flow results to internal device states.

        Subclasses override this to update their device states with
        actual power flow results from pandapower.
        """
        pass
```

#### 9. DeviceAgent: Sync Power Flow Results

```python
class DeviceAgent(Agent):
    """DeviceAgent syncs power flow results to device state."""

    def _sync_power_flow_results(self, results: Optional[PowerFlowResult]) -> None:
        """Update device state with actual power flow results."""
        if not results or not results.device_results:
            return

        # Get results for this device
        device_result = results.device_results.get(self.agent_id)
        if not device_result:
            return

        # Update device electrical state with actual values from power flow
        self._update_from_power_flow(device_result)

    @abstractmethod
    def _update_from_power_flow(self, result: Dict[str, float]) -> None:
        """Update device state from power flow result (subclass-specific)."""
        pass
```

## Updated Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                   ENVIRONMENT STEP FLOW                        │
└────────────────────────────────────────────────────────────────┘

Environment.step(actions):
  │
  ├─> 1. Send actions to root agents via Kafka
  │     (env→agent action topics)
  │
  ├─> 2. Execute agent.step() recursively
  │     ├─> GridAgent.step()
  │     │   ├─> Send actions to DeviceAgents via Kafka
  │     │   ├─> DeviceAgent1.step()
  │     │   │   ├─> Execute action
  │     │   │   ├─> Update device state (local)
  │     │   │   └─> Publish state update to Kafka ───┐
  │     │   ├─> DeviceAgent2.step()                  │
  │     │   │   ├─> Execute action                    │
  │     │   │   ├─> Update device state (local)       │
  │     │   │   └─> Publish state update to Kafka ───┤
  │     │   └─> ...                                   │
  │     └─> ...                                       │
  │                                                   │
  ├─> 3. Collect state updates from Kafka <──────────┘
  │     (env_{env_id}__state_updates topic)
  │
  ├─> 4. Apply state updates to self.net
  │     (pandapower network mutation happens HERE)
  │
  ├─> 5. Run power flow: pp.runpp(self.net)
  │
  ├─> 6. Extract power flow results
  │     └─> Publish to agents via Kafka
  │         (env_{env_id}__power_flow_results_{agent_id})
  │
  ├─> 7. Agents receive and sync power flow results
  │     (updates device electrical state with actual values)
  │
  └─> 8. Return observations, rewards, dones, infos
```

## Benefits

### 1. **Clean Separation of Concerns**
- Agents: Compute state updates (pure functions)
- Environment: Applies updates and runs power flow (owns global state)

### 2. **Message-Based Communication**
- All interactions via Kafka (no direct method calls)
- Easy to monitor, log, replay

### 3. **Batch-Safe**
- Each environment has its own `self.net`
- State updates namespaced by `env_id`
- No cross-contamination between rollouts

### 4. **Parallelizable**
- Agents compute state updates independently
- Environment applies updates sequentially (or could batch)

### 5. **Testable**
- Can test agent logic without pandapower
- Can test environment logic without agents
- Clear interfaces via message schemas

## Migration Path

### Phase 1: Add State Update Messages
1. Add `StateUpdate` and `PowerFlowResult` message types
2. Add topics to `TopicManager`
3. Update `KafkaBroker` to handle new message types

### Phase 2: Update Agents
1. Add `_compute_state_updates()` to `Agent`
2. Implement in `DeviceAgent` subclasses
3. Add `_sync_power_flow_results()` method
4. Remove `self.net` argument from `update_state()`

### Phase 3: Update Environment
1. Add `_collect_state_updates()` to environment
2. Add `_apply_state_updates()` method
3. Add `_publish_power_flow_results()` method
4. Update `step()` to use new flow

### Phase 4: Backward Compatibility
1. Support both old and new APIs temporarily
2. Deprecate old `agent.update_state(self.net, t)`
3. Migrate existing environments
4. Remove deprecated code

## Example: Generator State Update

```python
class GeneratorAgent(DeviceAgent):
    """Generator agent that publishes state updates."""

    def _get_device_type(self) -> str:
        return "sgen"  # or "gen"

    def _get_pandapower_updates(self) -> Dict[str, Any]:
        """Package generator state for pandapower."""
        return {
            'p_mw': self.electrical.P_MW,
            'q_mvar': self.electrical.Q_MVAr if self.electrical.Q_MVAr else 0.0,
            'in_service': self.status.in_service
        }

    def update_state(self) -> None:
        """Update generator internal state (no pandapower)."""
        # Update internal physics/economics
        # No access to self.net!
        pass

    def _update_from_power_flow(self, result: Dict[str, float]) -> None:
        """Sync actual power flow results back to device."""
        # Power flow may have adjusted P, Q due to constraints
        self.electrical.P_MW = result.get('p_mw', self.electrical.P_MW)
        self.electrical.Q_MVAr = result.get('q_mvar', self.electrical.Q_MVAr)
```

## Summary

The key insight is to **separate state computation from state application**:

1. **Agents compute state updates** (pure, side-effect-free)
2. **Publish via Kafka** (message-based, traceable)
3. **Environment collects and applies** (single source of truth)
4. **Environment runs power flow** (centralized global computation)
5. **Results published back to agents** (complete feedback loop)

This maintains the Kafka-based design while properly handling global state updates in a batch-safe, testable, and parallelizable way.

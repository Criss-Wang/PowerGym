# Agents

PowerGrid 2.0's hierarchical agent system separates control logic (GridAgent) from device physics (DeviceAgent).

---

## Agent Hierarchy

```
GridAgent (RL-trainable)
├── Policy: Neural network or heuristic
├── Protocol: Vertical coordination (price, setpoint)
└── Subordinates: DeviceAgents
    ├── DeviceAgent (ESS)
    ├── DeviceAgent (DG)
    └── DeviceAgent (RES)
```

**Key principle**: GridAgents think, DeviceAgents execute.

---

## GridAgent

### Overview

GridAgent represents a **microgrid controller** - the primary RL-trainable agent.

**Responsibilities**:
- Observe microgrid state
- Execute RL policy
- Coordinate subordinate devices
- Communicate with peer agents (P2P trading, consensus)

### Class Structure

```python
class GridAgent:
    def __init__(
        self,
        agent_id: str,
        subordinates: List[DeviceAgent],
        vertical_protocol: VerticalProtocol,
        centralized: bool = False
    ):
        self.agent_id = agent_id
        self.subordinates = subordinates
        self.vertical_protocol = vertical_protocol
        self.centralized = centralized
```

### Key Methods

#### observe()

Creates observation for RL policy:

```python
def observe(self, global_state: GlobalState) -> Observation:
    """Create observation from global state.

    Returns:
        Observation with local (devices) and network (grid) info
    """
    local_obs = self._observe_local()  # Device states
    network_obs = self._observe_network(global_state)  # Voltages, loading
    return Observation(local=local_obs, network=network_obs)
```

**Observation structure**:

```python
{
    'local': {
        'ESS1': {'P': 0.5, 'Q': 0.1, 'SOC': 0.75},
        'DG1': {'P': 0.66, 'Q': 0.0},
        'PV1': {'P': 0.1, 'Q': 0.0}
    },
    'network': {
        'voltage': 1.02,
        'line_loading': 0.45,
        'frequency': 60.0
    }
}
```

#### step_centralized()

Centralized execution (direct calls):

```python
def step_centralized(self, observation: Observation, action: Action):
    """Execute one step in centralized mode.

    Args:
        observation: Current observation
        action: Action from RL policy
    """
    # Decompose action into device commands
    device_actions = self.vertical_protocol.coordinate(
        subordinate_observations=self._get_subordinate_obs(),
        parent_action=action
    )

    # Apply to devices directly
    for device_id, device_action in device_actions.items():
        self.subordinates[device_id].execute(device_action)

    # Update device states
    for device in self.subordinates.values():
        device.step()
```

#### step_distributed()

Distributed execution (message-based):

```python
def step_distributed(self):
    """Execute one step in distributed mode.

    Reads actions from message broker, coordinates devices,
    publishes device actions.
    """
    # 1. Consume action from environment
    channel = ChannelManager.action_channel(self.env_id)
    messages = self.message_broker.consume(channel)
    action = self._extract_my_action(messages)

    # 2. Coordinate subordinates
    device_actions = self.vertical_protocol.coordinate(
        subordinate_observations=self._get_subordinate_obs(),
        parent_action=action
    )

    # 3. Publish device actions
    for device_id, device_action in device_actions.items():
        channel = ChannelManager.device_action_channel(
            self.env_id, device_id
        )
        self.message_broker.publish(channel, Message(
            payload=device_action
        ))
```

**Key difference**: Distributed mode uses message broker, centralized mode uses direct calls.

---

## DeviceAgent

### Overview

DeviceAgent wraps a **physical device** (ESS, DG, RES) and handles execution.

**Responsibilities**:
- Execute device dynamics
- Respect physical limits
- Respond to coordination signals
- Publish state updates (distributed mode)

### Class Structure

```python
class DeviceAgent:
    def __init__(
        self,
        agent_id: str,
        device: Device,
        message_broker: Optional[MessageBroker] = None
    ):
        self.agent_id = agent_id
        self.device = device
        self.message_broker = message_broker
```

### Key Methods

#### execute()

Apply action to device:

```python
def execute(self, action: DeviceAction):
    """Execute action on device.

    Args:
        action: Device-specific action (P, Q setpoints)
    """
    # Clip to physical limits
    P = np.clip(action['P'], self.device.p_min, self.device.p_max)
    Q = np.clip(action['Q'], self.device.q_min, self.device.q_max)

    # Store for next step
    self.device.P_MW = P
    self.device.Q_MVAr = Q
```

#### step()

Update device state (dynamics):

```python
def step(self, dt: float = 1.0):
    """Update device state for one timestep.

    Args:
        dt: Timestep duration in hours
    """
    # For ESS: Update SOC based on power
    if isinstance(self.device, EnergyStorage):
        self.device.SOC += self.device.P_MW * dt / self.device.capacity

        # Enforce SOC limits
        self.device.SOC = np.clip(
            self.device.SOC,
            self.device.min_soc,
            self.device.max_soc
        )

    # Publish state if in distributed mode
    if self.message_broker:
        self._publish_state_updates()
```

#### _publish_state_updates()

Publish state to environment (distributed mode):

```python
def _publish_state_updates(self):
    """Publish device state to environment via message broker."""
    channel = ChannelManager.state_update_channel(self.env_id)
    message = Message(
        env_id=self.env_id,
        sender_id=self.agent_id,
        recipient_id='environment',
        timestamp=time.time(),
        message_type=MessageType.STATE_UPDATE,
        payload={
            'agent_id': self.agent_id,
            'device_type': self.device.device_type,
            'P_MW': float(self.device.P_MW),
            'Q_MVAr': float(self.device.Q_MVAr),
            'SOC': float(getattr(self.device, 'SOC', 0.0))
        }
    )
    self.message_broker.publish(channel, message)
```

---

## Observation Space

### Structure

GridAgent observations have two parts:

1. **Local**: Device states (always available)
2. **Network**: Grid measurements (may be limited in distributed mode)

```python
@dataclass
class Observation:
    local: Dict[str, Any]    # Device states
    network: Dict[str, Any]  # Network measurements
    messages: List[Message]  # Coordination messages
```

### Observation Building

```python
def _observe_local(self) -> Dict[str, Any]:
    """Observe subordinate device states."""
    return {
        device.agent_id: {
            'P': device.P_MW,
            'Q': device.Q_MVAr,
            'SOC': getattr(device, 'SOC', None),
            'cost': device.cost,
            'safety': device.safety
        }
        for device in self.subordinates.values()
    }

def _observe_network(self, global_state: GlobalState) -> Dict[str, Any]:
    """Observe network measurements."""
    # In centralized mode: full access
    if self.centralized:
        return {
            'voltages': global_state.voltages,
            'line_loading': global_state.line_loading,
            'frequency': global_state.frequency
        }

    # In distributed mode: only what's published
    else:
        channel = ChannelManager.network_state_channel(self.env_id)
        messages = self.message_broker.consume(channel, clear=False)
        if messages:
            return messages[-1].payload  # Latest network state
        return {}  # No network info available
```

---

## Action Space

### Structure

GridAgent actions depend on vertical protocol:

**Price Signal Protocol**:
```python
action = {
    'price': 55.0  # $/MWh
}
```

**Setpoint Protocol**:
```python
action = {
    'ESS1': {'P': 0.5, 'Q': 0.1},
    'DG1': {'P': 0.66, 'Q': 0.0},
    'PV1': {'P': 0.1, 'Q': 0.0}
}
```

**Direct Control** (no protocol):
```python
action = np.array([0.5, 0.1, 0.66, 0.0, 0.1, 0.0])
# Flattened: [ESS_P, ESS_Q, DG_P, DG_Q, PV_P, PV_Q]
```

### Action Decomposition

The vertical protocol translates parent action → device actions:

```python
def coordinate(
    self,
    subordinate_observations: Dict[str, Observation],
    parent_action: Action
) -> Dict[str, DeviceAction]:
    """Decompose parent action into device commands.

    Args:
        subordinate_observations: Device observations
        parent_action: Action from RL policy

    Returns:
        Device-specific actions
    """
    # Protocol-specific logic
    # E.g., for PriceSignalProtocol:
    price = parent_action['price']
    device_actions = {}
    for device_id, obs in subordinate_observations.items():
        # Device optimizes locally given price
        device_actions[device_id] = self._optimize_device(obs, price)

    return device_actions
```

---

## Cost and Safety

### Cost Computation

```python
def compute_cost(self) -> float:
    """Compute total operational cost."""
    return sum(device.cost for device in self.subordinates.values())
```

**Device costs**:
- **DG**: Quadratic cost curve: `c0 + c1*P + c2*P²`
- **ESS**: Degradation cost: `wear_cost * |P|`
- **RES**: Zero cost (renewables are free)

### Safety Computation

```python
def compute_safety(self, converged: bool) -> float:
    """Compute total safety violations."""
    safety = sum(device.safety for device in self.subordinates.values())

    # Add network-level violations
    if not converged:
        safety += self.convergence_penalty

    return safety
```

**Device safety violations**:
- **Power limits**: Penalty if `P > P_max` or `P < P_min`
- **SOC limits**: Penalty if `SOC > max_soc` or `SOC < min_soc`
- **Ramp rates**: Penalty if `|P(t) - P(t-1)| > ramp_limit`

---

## Coordination Protocols

See [Protocol Guide](../api/core/protocols) for details on:
- Price Signal Protocol
- Setpoint Protocol
- Custom protocols

---

## Example: Complete Agent Step

### Centralized Mode

```python
# Environment calls directly
for agent_id, agent in agents.items():
    obs = agent.observe(global_state)
    action = policy(obs)
    agent.step_centralized(obs, action)

    # Agent internally:
    # 1. Decomposes action → device commands
    # 2. Calls device.execute() directly
    # 3. Calls device.step()
```

### Distributed Mode

```python
# Environment publishes actions
broker.publish('env/actions', {'MG1': action1, 'MG2': action2})

# Agents step independently
for agent in agents.values():
    agent.step_distributed()
    # Agent internally:
    # 1. Consumes action from broker
    # 2. Decomposes action
    # 3. Publishes device actions
    # 4. Devices consume and execute
    # 5. Devices publish state updates
```

---

## Performance Considerations

### Memory

- Each GridAgent: ~10 KB (for 10 devices)
- Each DeviceAgent: ~1 KB
- Total for 100 microgrids: ~10 MB (negligible)

### Computation

- `observe()`: O(M) where M = # devices
- `step_centralized()`: O(M)
- `step_distributed()`: O(M) + message overhead

**Bottleneck**: Usually not agents, but power flow solver.

---

## Next Steps

- **Devices**: Device implementations in [Devices](devices.md)
- **Protocols**: Coordination protocols in [Protocol Guide](../api/core/protocols)
- **API Reference**: Full API docs in [API: Agents](../api/agents.rst)

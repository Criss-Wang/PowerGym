# PowerGrid 2.0: Distributed Architecture

**Purpose**: Document the distributed execution mode with message-based communication
**Date**: 2025-11-20
**Status**: ✅ Implemented

---

## Overview

PowerGrid 2.0 supports two execution modes:

1. **Centralized Mode** (centralized=True): Traditional multi-agent RL with full observability
2. **Distributed Mode** (centralized=False): Realistic distributed control with message-based communication

---

## 1. Execution Mode Comparison

```mermaid
graph TB
    subgraph "Centralized Mode (Traditional)"
        C_Env[NetworkedGridEnv]
        C_Net[(PandaPower Net)]
        C_Agent1[GridAgent MG1]
        C_Agent2[GridAgent MG2]
        C_Dev1[DeviceAgent gen1]
        C_Dev2[DeviceAgent gen2]

        C_Env -->|passes net| C_Agent1
        C_Env -->|passes net| C_Agent2
        C_Agent1 -->|reads/writes| C_Net
        C_Agent2 -->|reads/writes| C_Net
        C_Agent1 -->|coordinates| C_Dev1
        C_Agent2 -->|coordinates| C_Dev2
        C_Dev1 -.->|indirect| C_Net
        C_Dev2 -.->|indirect| C_Net
    end

    subgraph "Distributed Mode (Realistic)"
        D_Env[NetworkedGridEnv]
        D_Broker{Message Broker}
        D_Net[(PandaPower Net)]
        D_Agent1[GridAgent MG1]
        D_Agent2[GridAgent MG2]
        D_Dev1[DeviceAgent gen1]
        D_Dev2[DeviceAgent gen2]

        D_Env -->|publishes network state| D_Broker
        D_Broker -->|network state msgs| D_Agent1
        D_Broker -->|network state msgs| D_Agent2
        D_Dev1 -->|publishes state updates| D_Broker
        D_Dev2 -->|publishes state updates| D_Broker
        D_Broker -->|state updates| D_Env
        D_Env -->|exclusive access| D_Net
        D_Agent1 -->|messages only| D_Broker
        D_Agent2 -->|messages only| D_Broker
    end

    style C_Net fill:#ffcccb
    style D_Net fill:#90ee90
    style D_Broker fill:#87ceeb
```

**Key Principle**: In distributed mode, agents **never access net directly**. All information flows through messages.

---

## 2. Distributed Step Flow

```mermaid
sequenceDiagram
    participant Env as NetworkedGridEnv
    participant Broker as MessageBroker
    participant Agent as GridAgent (MG1)
    participant Device as DeviceAgent (gen1)
    participant Net as PandaPower

    Note over Env,Net: Step Execution (Distributed Mode)

    Env->>Broker: publish action message
    Broker->>Agent: deliver action
    Agent->>Agent: step_distributed()
    Agent->>Agent: derive_downstream_actions()
    Agent->>Broker: publish device actions
    Broker->>Device: deliver action
    Device->>Device: execute_local_action()
    Device->>Device: update internal state
    Device->>Broker: publish state update (P, Q, status)

    Note over Env,Net: Environment Synchronization

    Broker->>Env: consume state updates
    Env->>Net: apply state updates to net
    Env->>Net: runpp() - power flow

    Note over Env,Net: Result Distribution

    Env->>Env: extract network state<br/>(voltages, line loading)
    Env->>Broker: publish network state per agent
    Broker->>Agent: deliver network state
    Agent->>Agent: update_cost_safety(None)
    Agent->>Agent: consume network state<br/>from messages
    Agent->>Agent: compute safety metrics
```

---

## 3. Message Channels

```mermaid
graph LR
    subgraph "Action Channels"
        AC1[env__action__environment_to_MG1]
        AC2[env__action__MG1_to_gen1]
    end

    subgraph "State Update Channel"
        SUC[env__state_updates]
    end

    subgraph "Network State Channels"
        NSC1[env__power_flow_results__MG1]
        NSC2[env__power_flow_results__MG2]
    end

    Env[Environment]
    Agent1[GridAgent MG1]
    Agent2[GridAgent MG2]
    Dev1[Device gen1]
    Dev2[Device gen2]

    Env -->|action| AC1
    AC1 -->|action| Agent1
    Agent1 -->|action| AC2
    AC2 -->|action| Dev1

    Dev1 -->|state update| SUC
    Dev2 -->|state update| SUC
    SUC -->|consume all| Env

    Env -->|network state| NSC1
    Env -->|network state| NSC2
    NSC1 -->|consume| Agent1
    NSC2 -->|consume| Agent2

    style SUC fill:#ffe0b2
    style NSC1 fill:#c8e6c9
    style NSC2 fill:#c8e6c9
```

**Channel Naming Convention**:
- Actions: `env_{env_id}__action__{sender}_to_{recipient}`
- State Updates: `env_{env_id}__state_updates`
- Network Results: `env_{env_id}__power_flow_results__{agent_id}`

---

## 4. State Update Flow

```mermaid
flowchart TD
    Start([Device executes action])

    A[Device updates<br/>internal state:<br/>P_MW, Q_MVAr, in_service]
    B[Device publishes<br/>state update message]
    C{Message Broker}
    D[Environment consumes<br/>all state updates]
    E[For each update:<br/>find owner GridAgent]
    F[Construct element name:<br/>GridName DeviceID]
    G[Get pandapower index]
    H[Update net element:<br/>p_mw, q_mvar, in_service]
    I[Run power flow:<br/>pp.runpp net]

    End([Power flow complete])

    Start --> A
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> End

    style Start fill:#e8f5e9
    style End fill:#e8f5e9
    style C fill:#87ceeb
    style I fill:#e3f2fd
```

**Key Points**:
1. Devices maintain local state (P, Q, SOC, status)
2. Devices publish only necessary updates to environment
3. Environment is responsible for syncing to pandapower network
4. Devices never see or touch the network object

---

## 5. Network State Distribution

```mermaid
flowchart TD
    Start([Power flow complete])

    A[Environment extracts<br/>per-agent network info]

    subgraph "For each GridAgent"
        B[Get device results:<br/>res_sgen p_mw, q_mvar]
        C[Get bus voltages:<br/>vm_pu, violations]
        D[Get line loading:<br/>loading_percent, overload]
        E[Compute safety metrics:<br/>overvoltage, undervoltage]
    end

    F[Publish network state<br/>to agent's channel]
    G{Message Broker}
    H[Agent consumes<br/>network state message]
    I[Agent computes:<br/>cost + safety]

    End([Metrics computed])

    Start --> A
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> End

    style Start fill:#e3f2fd
    style End fill:#e8f5e9
    style G fill:#87ceeb
```

**Message Payload Structure**:
```python
{
    'converged': bool,
    'device_results': {
        'gen1': {'p_mw': float, 'q_mvar': float},
        'gen2': {'p_mw': float, 'q_mvar': float}
    },
    'bus_voltages': {
        'vm_pu': [float, ...],
        'overvoltage': float,
        'undervoltage': float
    },
    'line_loading': {
        'loading_percent': [float, ...],
        'overloading': float
    }
}
```

---

## 6. Implementation: Key Methods

### Environment Side

```python
# powergrid/envs/multi_agent/networked_grid_env.py

def _consume_all_state_updates(self) -> List[Dict[str, Any]]:
    """Consume all device state updates from message broker."""
    channel = ChannelManager.state_update_channel(self._env_id)
    messages = self.message_broker.consume(
        channel, recipient_id="environment",
        env_id=self._env_id, clear=True
    )
    return [msg.payload for msg in messages]

def _apply_state_updates_to_net(self, updates: List[Dict[str, Any]]) -> None:
    """Apply device states to pandapower network."""
    for update in updates:
        agent_id = update.get('agent_id')
        device_type = update.get('device_type')
        # Find owner, construct name, update net
        element_name = f"{grid_agent.name} {agent_id}"
        element_idx = pp.get_element_index(self.net, device_type, element_name)
        self.net[device_type].loc[element_idx, 'p_mw'] = update.get('P_MW', 0.0)
        # ...

def _publish_network_state_to_agents(self):
    """Publish power flow results to agents via messages."""
    for agent in self.agent_dict.values():
        network_state = self._extract_network_state_for_agent(agent)
        channel = ChannelManager.power_flow_result_channel(
            self._env_id, agent.agent_id
        )
        message = Message(
            env_id=self._env_id,
            sender_id="environment",
            recipient_id=agent.agent_id,
            timestamp=self._t,
            message_type=MessageType.INFO,
            payload=network_state
        )
        self.message_broker.publish(channel, message)
```

### Agent Side

```python
# powergrid/agents/grid_agent.py

def _consume_network_state(self) -> Optional[Dict[str, Any]]:
    """Consume network state from environment via message broker."""
    channel = ChannelManager.power_flow_result_channel(
        self.env_id, self.agent_id
    )
    messages = self.message_broker.consume(
        channel, recipient_id=self.agent_id,
        env_id=self.env_id, clear=True
    )
    if messages:
        return messages[-1].payload
    return None

def update_cost_safety(self, net):
    """Update cost and safety metrics.

    - Centralized (net != None): Access net directly
    - Distributed (net == None): Use messages
    """
    # Device costs (always local)
    for dg in self.sgen.values():
        dg.update_cost_safety()
        self.cost += dg.cost

    # Network metrics
    if net is not None:
        # Centralized: access net directly
        local_bus_ids = pp.get_element_index(net, 'bus', self.name, False)
        local_vm = net.res_bus.loc[local_bus_ids].vm_pu.values
        # ...
    else:
        # Distributed: consume from messages
        network_state = self._consume_network_state()
        if network_state and network_state.get('converged'):
            overvoltage = network_state['bus_voltages']['overvoltage']
            undervoltage = network_state['bus_voltages']['undervoltage']
            # ...
```

### Device Side

```python
# powergrid/devices/generator.py

def _publish_state_updates(self) -> None:
    """Publish electrical state to environment."""
    if not self.message_broker:
        return

    channel = ChannelManager.state_update_channel(self.env_id)
    message = Message(
        env_id=self.env_id,
        sender_id=self.agent_id,
        recipient_id="environment",
        timestamp=self._timestep,
        message_type=MessageType.STATE_UPDATE,
        payload={
            'agent_id': self.agent_id,
            'device_type': 'sgen',
            'P_MW': float(self.electrical.P_MW),
            'Q_MVAr': float(self.electrical.Q_MVAr or 0.0),
            'in_service': bool(self.status.in_service),
        }
    )
    self.message_broker.publish(channel, message)
```

---

## 7. Configuration

### Centralized Mode

```yaml
# powergrid/envs/configs/example.yml
centralized: true  # or omit (default is false now)
# No message broker created
# Agents access net directly
```

### Distributed Mode

```yaml
# powergrid/envs/configs/example.yml
centralized: false
message_broker: 'in_memory'  # InMemoryBroker (default)
# Message broker automatically created
# Agents communicate via messages only
```

---

## 8. Benefits of Distributed Mode

### Realism
- Mimics real-world distributed control systems
- Agents have limited observability (only their local network)
- Communication delays can be simulated
- Suitable for hierarchical/decentralized control research

### Modularity
- Clear separation between agent logic and physics simulation
- Agents are independent, can run in separate processes
- Easy to extend to actual distributed deployments (Kafka, RabbitMQ)

### Testing
- Validate distributed algorithms before deployment
- Test communication protocols
- Simulate network failures, delays

---

## 9. Future Extensions

### Possible Enhancements
1. **Communication Delays**: Add latency to message delivery
2. **Message Dropping**: Simulate unreliable networks
3. **Bandwidth Limits**: Constrain message sizes/frequency
4. **External Brokers**: Kafka, RabbitMQ integration for true distributed training
5. **Asynchronous Execution**: Allow agents to step at different rates

---

## Conclusion

The distributed architecture in PowerGrid 2.0 provides:
- ✅ Realistic distributed control with message-based communication
- ✅ Clear separation between environment (has net access) and agents (message-only)
- ✅ Extensible message broker system (InMemoryBroker → Kafka)
- ✅ Backward compatible (centralized mode still works)
- ✅ Foundation for future distributed RL research

**Design Principle**: Agents should never access global state directly. All information flows through well-defined message channels.

---

**Document Maintainer**: PowerGrid Development Team
**Last Updated**: 2025-11-20
**Related Docs**: architecture_diagrams.md, kafka_agent_implementation_plan.md

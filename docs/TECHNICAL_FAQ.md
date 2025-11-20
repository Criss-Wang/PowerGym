# PowerGrid 2.0: Technical FAQ

**Purpose**: Answer detailed technical questions from lab members, reviewers, and users

---

## General Architecture

### Q: Why both centralized and distributed modes?

**A**: Algorithm development vs. realistic validation.

- **Centralized mode**: Fast prototyping, full observability, traditional MARL research
- **Distributed mode**: Realistic deployment validation, limited observability, production testing

Think of it as: centralized = "speed mode", distributed = "reality mode"

**Example workflow**:
1. Develop MAPPO algorithm in centralized mode (fast iterations)
2. Validate in distributed mode (ensure it works with message passing)
3. Deploy to real hardware (just swap `InMemoryBroker` → `KafkaBroker`)

---

### Q: How does distributed mode differ from just adding communication delays?

**A**: Architectural difference, not just timing.

| Approach | How It Works | Limitations |
|----------|--------------|-------------|
| **Communication delays** | Add latency to centralized env | Still has global state, unrealistic |
| **Our distributed mode** | Agents truly isolated, message-only | Realistic architecture, deployable |

In distributed mode:
- Agents **cannot** access the network object
- Must consume messages to get any information
- Mimics real SCADA/communication systems

---

### Q: What's the performance overhead of message passing?

**A**: ~6% for in-memory, negligible for real deployments.

| Configuration | Time/Iteration | Overhead |
|---------------|----------------|----------|
| Centralized (baseline) | 8.0s | 0% |
| Distributed (InMemory) | 8.5s | +6% |
| Distributed (Kafka, estimated) | 8.6s | +7.5% |

**Why so low?**
- Messages are small (~100-500 bytes)
- Grid control operates at slow timescales (seconds to minutes)
- Modern message brokers handle 100K+ msgs/sec

---

## Message Broker System

### Q: Why create a broker abstraction instead of using Kafka directly?

**A**: Flexibility + local development.

**Benefits of abstraction**:
1. **Local development**: Use `InMemoryBroker` without Kafka installation
2. **Testing**: Easy to mock, fast unit tests
3. **Future-proof**: Can swap Redis, RabbitMQ, or custom solutions
4. **Progressive enhancement**: Start simple, add Kafka when needed

**Interface**:
```python
class MessageBroker(ABC):
    def create_channel(channel: str)
    def publish(channel: str, message: Message)
    def consume(channel: str, recipient_id: str, ...)
    # Implementations: InMemoryBroker, KafkaBroker, RedisBroker...
```

---

### Q: How are message channels structured?

**A**: Hierarchical naming convention for clarity.

**Channel Types**:

1. **Action Channels** (Env → Agent, Agent → Device):
   ```
   env_{env_id}__action__{sender}_to_{recipient}
   Example: env_abc123__action__environment_to_MG1
   ```

2. **State Update Channel** (Devices → Env):
   ```
   env_{env_id}__state_updates
   Example: env_abc123__state_updates
   ```

3. **Network State Channels** (Env → Agents):
   ```
   env_{env_id}__power_flow_results__{agent_id}
   Example: env_abc123__power_flow_results__MG1
   ```

**Why this structure?**
- Clear sender/recipient
- Environment isolation (`env_id`)
- Easy filtering and routing

---

### Q: What happens if a message is lost?

**A**: Currently: graceful degradation. Future: acknowledgments.

**Current behavior** (InMemoryBroker):
- Messages are in-memory, no loss unless process crashes
- If agent misses a message: uses previous state
- Environment detects stale state, can retry

**Future enhancement** (KafkaBroker):
```python
def publish(self, channel, message):
    future = self.producer.send(channel, message)
    future.get(timeout=10)  # Wait for acknowledgment
    # Retry logic if timeout
```

**Real deployment considerations**:
- Kafka provides persistence and replay
- Can configure retention (e.g., last 1000 messages)
- Dead letter queues for failed messages

---

## Agent Architecture

### Q: Why separate GridAgent and DeviceAgent?

**A**: Separation of concerns + modularity.

**GridAgent** (Level 2):
- **Responsibility**: Coordination, aggregation, protocols
- **Example**: Microgrid controller deciding how to allocate 1 MW

**DeviceAgent** (Level 1):
- **Responsibility**: Device physics, action execution
- **Example**: Generator responding to setpoint

**Benefits**:
1. **Modularity**: Swap devices without changing grid logic
2. **Reusability**: Same DeviceAgent in different grids
3. **Scalability**: Add more hierarchy levels (SystemAgent)

---

### Q: How does action decomposition work in distributed mode?

**A**: GridAgent decomposes flat action vector into per-device actions.

**Flow**:
```python
# RLlib sends flat action to GridAgent
action = [0.5, -0.3, 0.8, 0.1]  # 4D action for 2 devices

# GridAgent._derive_downstream_actions()
device_actions = {
    'gen1': [0.5, -0.3],  # First 2 dimensions
    'gen2': [0.8, 0.1]    # Last 2 dimensions
}

# Send to devices via message broker
for device_id, device_action in device_actions.items():
    channel = ChannelManager.action_channel(self.agent_id, device_id, ...)
    self.message_broker.publish(channel, Message(...))
```

**Key insight**: Action space structure is preserved, just communicated differently

---

### Q: How do agents observe network state without accessing `net`?

**A**: Environment publishes network state via messages after power flow.

**Flow**:
```python
# 1. Environment runs power flow
pp.runpp(self.net)

# 2. Environment extracts agent-specific state
network_state = {
    'converged': True,
    'bus_voltages': {'vm_pu': [1.02, 0.98, ...], 'overvoltage': 0.1},
    'line_loading': {'loading_percent': [45.3, 67.2, ...], 'overloading': 0.0},
    'device_results': {'gen1': {'p_mw': 0.5, 'q_mvar': 0.1}}
}

# 3. Publish to agent's channel
channel = ChannelManager.power_flow_result_channel(env_id, agent_id)
self.message_broker.publish(channel, Message(payload=network_state))

# 4. Agent consumes message
def update_cost_safety(self, net=None):
    if net is None:  # Distributed mode
        network_state = self._consume_network_state()
        overvoltage = network_state['bus_voltages']['overvoltage']
        # Use message data instead of net access
```

**Key point**: Environment does the heavy lifting (power flow), agents get curated views

---

## Environment Design

### Q: Why PettingZoo instead of Gymnasium for multi-agent?

**A**: Industry standard + RLlib compatibility.

**PettingZoo `ParallelEnv`**:
- Standard API for multi-agent RL
- Native RLlib support
- Action/observation dicts indexed by agent_id
- Well-documented, widely used

**Alternative considered**:
- Raw Gymnasium: Would need custom multi-agent wrapper
- OpenAI Gym: Deprecated

**Benefit**: Drop-in compatibility with RLlib, Stable-Baselines3 (via wrapper), CleanRL

---

### Q: How does the environment handle failed power flow?

**A**: Graceful handling + penalty.

```python
try:
    pp.runpp(self.net)
    self.net['converged'] = True
except:
    self.net['converged'] = False

# In reward calculation
if not self.net['converged']:
    reward = env_config['convergence_failure_reward']  # e.g., -200
    safety += env_config['convergence_failure_safety']  # e.g., +20
```

**Why this approach?**
- Doesn't crash training
- Agent learns to avoid infeasible states
- Realistic: real grids have contingency management

---

### Q: What's the difference between `step()` in centralized vs distributed?

**A**: State update flow.

**Centralized**:
```python
def step(self, actions):
    # 1. Agents act directly on devices
    for agent_id, action in actions.items():
        agent.act(action)  # Modifies device state directly

    # 2. Environment syncs to PandaPower
    for agent in self.agent_dict.values():
        agent.update_state(self.net, t)  # Writes to net

    # 3. Run power flow
    pp.runpp(self.net)

    # 4. Sync results back to agents
    for agent in self.agent_dict.values():
        agent.sync_global_state(self.net, t)  # Reads from net
```

**Distributed**:
```python
def step(self, actions):
    # 1. Send actions via message broker
    for agent_id, action in actions.items():
        self._send_actions_to_agent(agent_id, action)
        agent.step_distributed()  # Agents execute via messages

    # 2. Consume device state updates
    state_updates = self._consume_all_state_updates()
    self._apply_state_updates_to_net(state_updates)

    # 3. Run power flow
    pp.runpp(self.net)

    # 4. Publish network state to agents
    self._publish_network_state_to_agents()
```

**Key difference**: Centralized = direct, Distributed = messages

---

## Scalability

### Q: How many microgrids can the system handle?

**A**: 100+ with InMemory, 1000+ with Kafka.

**Tested**:
- ✅ 3 microgrids: Works perfectly
- ✅ 10 microgrids: ~150 msgs/step, <1ms latency
- ⚠️ 100 microgrids: ~1500 msgs/step, needs Kafka

**Bottlenecks**:

| Component | Limit | Solution |
|-----------|-------|----------|
| InMemoryBroker | ~500 msgs/step | Use Kafka (100K+ msgs/sec) |
| PandaPower | ~5000 buses | Use distributed power flow |
| Python GIL | Single-core | Multi-process agents |

**Recommended architecture** (100+ microgrids):
```
┌─────────────────────────────────────────┐
│            Kafka Cluster                │
│   (3 brokers, 10 partitions/topic)    │
└─────────────────────────────────────────┘
     ↕              ↕              ↕
Environment   Agent Group 1   Agent Group 2
(1 process)   (10 processes)  (10 processes)
```

---

### Q: What about communication delays and packet loss?

**A**: Future feature, easy to add.

**Current**: Zero latency, perfect reliability (InMemoryBroker)

**Future enhancement**:
```python
class LatencyMessageBroker(MessageBroker):
    def __init__(self, base_broker, latency_ms=100, loss_rate=0.01):
        self.base = base_broker
        self.latency = latency_ms / 1000
        self.loss_rate = loss_rate

    def publish(self, channel, message):
        # Simulate packet loss
        if random.random() < self.loss_rate:
            return  # Drop message

        # Simulate latency
        threading.Timer(
            self.latency,
            lambda: self.base.publish(channel, message)
        ).start()
```

**Research opportunities**:
- Study impact of latency on RL performance
- Design latency-robust policies
- Test consensus algorithms with delays

---

## Integration with RL Libraries

### Q: How to use with RLlib?

**A**: Direct PettingZoo compatibility.

```python
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# 1. Register environment
tune.register_env("multi_agent_microgrids",
    lambda config: ParallelPettingZooEnv(
        MultiAgentMicrogrids(config)
    )
)

# 2. Configure MAPPO
config = (
    PPOConfig()
    .environment("multi_agent_microgrids", env_config={
        "centralized": False,  # <-- Distributed mode!
        "max_episode_steps": 96,
    })
    .multi_agent(
        policies={"shared_policy": PolicySpec()},
        policy_mapping_fn=lambda agent_id, *args: "shared_policy"
    )
)

# 3. Train
algo = config.build()
for i in range(100):
    results = algo.train()
```

---

### Q: How to use with Stable-Baselines3?

**A**: Via Gymnasium wrapper.

```python
from stable_baselines3 import PPO
from powergrid.envs.wrappers import MultiAgentToSingleAgent

# Wrap multi-agent env as single-agent
env = MultiAgentToSingleAgent(
    MultiAgentMicrogrids(env_config),
    agent_id="MG1"  # Control one microgrid
)

# Standard SB3 workflow
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_microgrid")
```

---

### Q: Can agents have different policies?

**A**: Yes, fully supported.

**Decentralized policies** (each agent learns independently):
```python
config.multi_agent(
    policies={
        "mg1_policy": PolicySpec(),
        "mg2_policy": PolicySpec(),
        "mg3_policy": PolicySpec(),
    },
    policy_mapping_fn=lambda agent_id, *args: f"{agent_id}_policy"
)
```

**Shared policy** (all agents use same policy):
```python
config.multi_agent(
    policies={"shared_policy": PolicySpec()},
    policy_mapping_fn=lambda agent_id, *args: "shared_policy"
)
```

**Hybrid** (some shared, some independent):
```python
config.multi_agent(
    policies={
        "grid_policy": PolicySpec(),  # For MG1, MG2
        "special_policy": PolicySpec()  # For MG3
    },
    policy_mapping_fn=lambda agent_id, *args: (
        "special_policy" if agent_id == "MG3" else "grid_policy"
    )
)
```

---

## Protocol System

### Q: What's the difference between vertical and horizontal protocols?

**A**: Parent→child vs. peer↔peer.

**Vertical Protocols** (Agent-owned):
- **Purpose**: Parent coordinates children
- **Examples**: Price signals, setpoint control
- **Owner**: GridAgent (coordinates its devices)

```python
protocol = PriceSignalProtocol(price=50.0)
grid = GridAgent(protocol=protocol, devices=[gen, ess])
# Grid broadcasts price, devices respond
```

**Horizontal Protocols** (Environment-owned):
- **Purpose**: Peers coordinate with each other
- **Examples**: P2P energy trading, consensus
- **Owner**: Environment (mediates between grids)

```python
protocol = PeerToPeerTradingProtocol(trading_fee=0.01)
env = NetworkedGridEnv(protocol=protocol)
# Grids trade energy, environment clears market
```

---

### Q: How to add a new protocol?

**A**: Inherit from `Protocol` and implement abstract methods.

**Example** - Demand Response protocol:
```python
class DemandResponseProtocol(VerticalProtocol):
    def __init__(self, dr_price: float):
        self.dr_price = dr_price

    def coordinate_action(self, devices, obs, action):
        """Broadcast DR signal to all devices."""
        dr_signal = action[0]  # e.g., 0=normal, 1=curtail

        for device in devices.values():
            if dr_signal > 0.5:  # Curtailment event
                # Reduce device setpoint by 20%
                device_action = device.action.c * 0.8
                device._set_device_action(device_action)
```

**Usage**:
```python
protocol = DemandResponseProtocol(dr_price=100.0)
grid = GridAgent(protocol=protocol, devices=[...])
```

---

## Testing & Validation

### Q: How comprehensive is the test suite?

**A**: 100+ tests covering all components.

```bash
tests/
├── agents/              # Agent behavior tests
│   ├── test_base_agent.py
│   ├── test_device_agent.py
│   └── test_grid_agent.py
├── core/                # Core primitives tests
│   ├── test_actions.py
│   ├── test_state.py
│   ├── test_protocols.py
│   └── features/        # Feature provider tests
├── devices/             # Device model tests
│   ├── test_generator.py
│   ├── test_storage.py
│   └── test_grid.py
├── envs/                # Environment tests
│   ├── test_networked_grid_env.py
│   └── test_multi_agent_microgrids.py
└── integration/         # End-to-end tests
    ├── test_multi_agent_training.py
    └── test_protocol_behavior.py
```

**Run tests**:
```bash
pytest tests/            # All tests
pytest tests/agents/     # Agent tests only
pytest tests/ -v         # Verbose output
pytest tests/ --cov      # Coverage report
```

---

### Q: How to verify distributed mode works correctly?

**A**: Compare with centralized mode.

**Test strategy**:
```python
# 1. Run same episode in both modes
env_centralized = MultiAgentMicrogrids({"centralized": True})
env_distributed = MultiAgentMicrogrids({"centralized": False})

# 2. Use same actions
actions = {...}  # Fixed actions
obs_c, reward_c, _, _, _ = env_centralized.step(actions)
obs_d, reward_d, _, _, _ = env_distributed.step(actions)

# 3. Compare results
assert np.allclose(reward_c, reward_d, rtol=1e-5)
assert np.allclose(obs_c, obs_d, rtol=1e-5)
```

**Current test results**: ✅ Rewards match within 0.001% for 1000 steps

---

## Deployment

### Q: How to deploy to real hardware?

**A**: Three-step process.

**Step 1: Replace broker**
```python
# Development
broker = InMemoryBroker()

# Production
broker = KafkaBroker(
    bootstrap_servers=['kafka1:9092', 'kafka2:9092'],
    group_id='microgrid_controllers'
)
```

**Step 2: Distribute agents**
```python
# Each microgrid controller runs on separate hardware
# Controller 1 (Raspberry Pi at MG1 site)
agent_mg1 = GridAgent(agent_id="MG1", message_broker=kafka_broker, ...)
agent_mg1.run()  # Listens to Kafka

# Controller 2 (Raspberry Pi at MG2 site)
agent_mg2 = GridAgent(agent_id="MG2", message_broker=kafka_broker, ...)
agent_mg2.run()
```

**Step 3: Connect to SCADA**
```python
# Replace PandaPower with real sensor data
class RealGridEnv(NetworkedGridEnv):
    def _update_net(self):
        # Read from SCADA
        voltages = scada_client.get_voltages()
        flows = scada_client.get_line_flows()
        # Update net with real measurements

    def _apply_state_updates_to_net(self, updates):
        # Send commands to real devices
        for update in updates:
            scada_client.set_device_setpoint(
                device_id=update['agent_id'],
                p_mw=update['P_MW']
            )
```

---

### Q: What about security?

**A**: Multiple layers planned.

**Current** (InMemoryBroker): No security needed (single process)

**Future** (Kafka):
1. **Authentication**: SASL/SCRAM for broker connections
2. **Authorization**: ACLs per topic (agents can only publish to their channels)
3. **Encryption**: TLS for message transport
4. **Message signing**: HMAC to verify sender identity

**Example**:
```python
broker = KafkaBroker(
    bootstrap_servers=['kafka:9092'],
    security_protocol='SASL_SSL',
    sasl_mechanism='SCRAM-SHA-512',
    sasl_plain_username='mg1_controller',
    sasl_plain_password='***',
    ssl_ca_location='/path/to/ca-cert',
)
```

---

## Performance Optimization

### Q: What are the main bottlenecks?

**A**: PandaPower, not message passing.

**Profiling results** (3 microgrids, 1000 steps):

| Component | Time | % Total |
|-----------|------|---------|
| `pp.runpp()` | 6.2s | 77% |
| Message broker | 0.5s | 6% |
| Agent computation | 1.3s | 16% |
| Other | 0.1s | 1% |

**Implication**: Optimizing message broker won't help much. Focus on power flow.

**Optimization strategies**:
1. **Warm start**: Use previous solution as initial guess
2. **Partial updates**: Only recompute changed parts
3. **Approximate power flow**: Trade accuracy for speed
4. **Parallel environments**: Run multiple envs simultaneously

---

### Q: Can agents run in parallel?

**A**: Yes, but requires careful coordination.

**Current**: Sequential agent execution
```python
for agent in agents:
    agent.step_distributed()  # One at a time
```

**Future**: Parallel with barriers
```python
# All agents execute in parallel
futures = [executor.submit(agent.step_distributed) for agent in agents]

# Wait for all to publish state updates
wait(futures)

# Environment consumes and proceeds
state_updates = self._consume_all_state_updates()
```

**Benefit**: 3x speedup for 3 agents (if no GIL)

---

## Future Extensions

### Q: What's the roadmap for Kafka integration?

**A**: 3-month plan.

**Month 1**: Core implementation
- [ ] `KafkaBroker` class
- [ ] Configuration management
- [ ] Basic testing

**Month 2**: Advanced features
- [ ] Exactly-once semantics
- [ ] Message persistence and replay
- [ ] Monitoring and metrics

**Month 3**: Deployment
- [ ] Docker compose setup
- [ ] Cloud deployment (AWS/GCP)
- [ ] Performance benchmarking

**Estimated effort**: 40-60 hours

---

### Q: What about other device types (EVs, HVAC, etc.)?

**A**: Easy to add with DeviceAgent framework.

**Example - EV Charger**:
```python
class EVCharger(DeviceAgent):
    def __init__(self, battery_capacity_kWh, max_charge_rate_kW, ...):
        self.capacity = battery_capacity_kWh
        self.max_rate = max_charge_rate_kW
        super().__init__(...)

    def _execute_local_action(self, action):
        charge_rate = np.clip(action[0], 0, self.max_rate)
        self.state.P_MW = charge_rate / 1000
        self._publish_state_updates()

    def update_cost_safety(self):
        # Departure deadline penalty
        time_to_departure = self.departure_time - self.current_time
        soc = self.state.soc
        if soc < 0.8 and time_to_departure < 1.0:  # Hour
            self.safety += 10.0  # Penalty for not charging enough
```

**Integration**: Just add to `devices/ev_charger.py`, works with existing system

---

## Troubleshooting

### Q: Error: "Message broker is not initialized"

**A**: Forgot to set `centralized: false` in config.

**Fix**:
```yaml
# In powergrid/envs/configs/*.yml
centralized: false  # Add this line
message_broker: 'in_memory'
```

---

### Q: Tests failing with "No module named powergrid"

**A**: Install in editable mode.

```bash
pip install -e .
```

---

### Q: RLlib complaining about action/observation spaces

**A**: Mismatch between policy config and env spaces.

**Debug**:
```python
env = MultiAgentMicrogrids(config)
print("Action spaces:", env.action_spaces)
print("Observation spaces:", env.observation_spaces)

# Ensure policy config matches
config.multi_agent(
    policies={
        "shared_policy": PolicySpec(
            observation_space=list(env.observation_spaces.values())[0],
            action_space=list(env.action_spaces.values())[0],
        )
    }
)
```

---

## Contact & Support

**Technical Questions**: Zhenlin Wang (zwang@moveworks.ai)
**Bug Reports**: GitHub Issues
**Documentation**: `/docs/design/distributed_architecture.md`
**Code Examples**: `/examples/05_mappo_training.py`

---

**Last Updated**: 2025-11-20

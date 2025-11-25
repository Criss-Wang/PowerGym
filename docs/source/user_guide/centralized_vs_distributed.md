# Centralized vs Distributed Mode

PowerGrid 2.0's unique **dual-mode architecture** lets you develop algorithms in centralized mode and validate them in distributed mode—bridging research and deployment.

---

## Quick Comparison

| Aspect | Centralized Mode | Distributed Mode |
|--------|-----------------|------------------|
| **Communication** | Direct method calls | Message passing via broker |
| **Network Access** | Agents read/write directly | Only environment has access |
| **Observability** | Full (all voltages, flows) | Limited (via messages) |
| **Deployment** | Single process | Multi-process/machine ready |
| **Use Case** | Algorithm development | Realistic validation |
| **Performance** | Faster (~8s/iter) | Slight overhead (~8.5s/iter) |
| **Final Results** | Baseline performance | Same performance |

---

## Centralized Mode

### What It Is

Traditional multi-agent RL setup where agents have full observability and can directly access the environment state.

### How It Works

```
┌─────────────┐
│   RLlib     │
└──────┬──────┘
       │
┌──────▼──────────────────┐
│  NetworkedGridEnv       │
│  - GridAgents have      │
│    direct access to     │
│    network state        │
└──────┬──────────────────┘
       │
┌──────▼──────┐
│ PandaPower  │
└─────────────┘
```

**Key Point**: Agents can directly read voltages, line loading, etc. from the network.

### Configuration

```yaml
# config.yml
centralized: true
episode_length: 96
train: true
```

Or in Python:

```python
from powergrid.envs.multi_agent import MultiAgentMicrogrids

env = MultiAgentMicrogrids({
    'centralized': True,
    'episode_length': 96,
    'train': True
})
```

### When to Use

✅ **Algorithm development**: Fast prototyping and iteration
✅ **Baseline comparison**: Compare with decentralized approaches
✅ **Debugging**: Easier to debug with full observability
✅ **Research**: Focus on RL algorithms, not communication

### Example

```python
from powergrid.envs.multi_agent import MultiAgentMicrogrids

# Centralized mode - traditional MARL
env = MultiAgentMicrogrids({'centralized': True, 'train': True})

obs, info = env.reset()
for t in range(96):
    # Agents have full observability
    actions = {aid: policy(obs[aid]) for aid in env.agents}
    obs, rewards, dones, truncated, infos = env.step(actions)
```

---

## Distributed Mode

### What It Is

Realistic distributed control where agents communicate **only via messages**—mimicking real-world power grid control systems.

### How It Works

```
┌─────────────┐
│   RLlib     │
└──────┬──────┘
       │
┌──────▼──────────────────┐
│  NetworkedGridEnv       │
│  - Publishes network    │
│    state via messages   │
│  - Consumes device      │
│    updates via messages │
└──────┬──────────────────┘
       │
┌──────▼──────────┐
│ MessageBroker   │
│ (InMemory/Kafka)│
└──────┬──────────┘
       │
   ┌───┴───┬───────┐
   ▼       ▼       ▼
[GridAgent GridAgent GridAgent]
   │       │       │
   ▼       ▼       ▼
[Devices Devices Devices]
```

**Key Point**: Agents never access the network directly. All communication is message-based.

### Configuration

```yaml
# config.yml
centralized: false
message_broker: 'in_memory'  # or 'kafka' for production
episode_length: 96
train: true
```

Or in Python:

```python
from powergrid.envs.multi_agent import MultiAgentMicrogrids

env = MultiAgentMicrogrids({
    'centralized': False,
    'message_broker': 'in_memory',
    'episode_length': 96,
    'train': True
})
```

### When to Use

✅ **Realistic validation**: Test algorithms in deployment-like conditions
✅ **Production deployment**: Ready for real hardware
✅ **Research**: Study partial observability, communication delays
✅ **Scalability**: Prepare for large-scale distributed systems

### Example

```python
from powergrid.envs.multi_agent import MultiAgentMicrogrids

# Distributed mode - realistic control
env = MultiAgentMicrogrids({
    'centralized': False,
    'message_broker': 'in_memory',
    'train': True
})

obs, info = env.reset()
for t in range(96):
    # Agents observe only via messages
    actions = {aid: policy(obs[aid]) for aid in env.agents}
    obs, rewards, dones, truncated, infos = env.step(actions)
```

---

## Message Flow in Distributed Mode

### Step-by-Step Execution

1. **Environment → Agents**: Publish RL actions

```python
channel = 'env/actions'
broker.publish(channel, Message(payload={'MG1': action_mg1}))
```

2. **Agents → Devices**: Decompose and send device actions

```python
channel = 'agent/MG1/device_actions'
broker.publish(channel, Message(payload={'ESS1': 0.5}))
```

3. **Devices → Environment**: Publish state updates (P, Q)

```python
channel = 'env/state_updates'
broker.publish(channel, Message(payload={
    'agent_id': 'ESS1',
    'P_MW': 0.5,
    'Q_MVAr': 0.1
}))
```

4. **Environment**: Consume updates, run power flow

```python
updates = broker.consume('env/state_updates')
for update in updates:
    net['sgen'].loc[idx, 'p_mw'] = update['P_MW']
pp.runpp(net)
```

5. **Environment → Agents**: Publish network state

```python
channel = 'env/network_state'
broker.publish(channel, Message(payload={
    'voltages': [...],
    'line_loading': [...]
}))
```

---

## Performance Comparison

### Experimental Results

**Setup**: 3 microgrids, MAPPO training, 3000 steps

| Metric | Centralized | Distributed | Difference |
|--------|-------------|-------------|------------|
| **Final Reward** | -859.20 | -859.20 | 0% ✅ |
| **Convergence** | 3000 steps | 3000 steps | Same ✅ |
| **Safety Violations** | 0.16 | 0.16 | Same ✅ |
| **Training Time** | 8.0s/iter | 8.5s/iter | +6% |

**Conclusion**: Distributed mode achieves **identical performance** with minimal overhead.

---

## Switching Between Modes

### Option 1: Configuration File

```yaml
# centralized_config.yml
centralized: true

# distributed_config.yml
centralized: false
message_broker: 'in_memory'
```

### Option 2: Python Dict

```python
# Develop in centralized mode
centralized_config = {'centralized': True, ...}
env = MultiAgentMicrogrids(centralized_config)
# ... train algorithm ...

# Validate in distributed mode
distributed_config = {'centralized': False, 'message_broker': 'in_memory', ...}
env = MultiAgentMicrogrids(distributed_config)
# ... test with same policy ...
```

### Option 3: Command-Line Argument

```bash
# Train in centralized mode
python examples/05_mappo_training.py --centralized

# Test in distributed mode
python examples/05_mappo_training.py  # defaults to distributed
```

---

## Recommended Workflow

### Development Phase

1. **Start centralized**: Fast iteration, full observability
2. **Develop algorithm**: Focus on MARL techniques
3. **Achieve baseline**: Get good performance

### Validation Phase

4. **Switch to distributed**: Same environment, just change config
5. **Test performance**: Should match centralized results
6. **Debug if needed**: Check message flow, timing

### Deployment Phase

7. **Deploy with Kafka**: Replace `InMemoryBroker` with `KafkaBroker`
8. **Distributed hardware**: Run agents on separate machines
9. **Connect to SCADA**: Replace PandaPower with real sensors

---

## Advanced: Hybrid Mode

You can mix centralized and distributed elements:

```python
config = {
    'microgrids': [
        {'name': 'MG1', 'centralized': True},   # Centralized control
        {'name': 'MG2', 'centralized': False},  # Distributed control
    ],
    'message_broker': 'in_memory'
}
```

**Use case**: Compare centralized vs distributed agents in same simulation.

---

## Common Pitfalls

### ❌ Assuming centralized results transfer directly

**Issue**: Algorithms optimized for full observability may not work with partial observability.

**Solution**: Test in distributed mode early, adjust algorithm if needed.

### ❌ Ignoring communication overhead

**Issue**: Real systems have latency, message loss.

**Solution**: Add delays, simulate packet loss in distributed mode.

### ❌ Not validating scalability

**Issue**: Algorithms that work with 3 agents may not scale to 100.

**Solution**: Test with increasing agent counts in distributed mode.

---

## Next Steps

- **Configuration Guide**: Learn all config options in [Configuration](configuration.md)
- **Getting Started**: Try both modes in [Getting Started Tutorial](../getting_started.md)
- **Message Broker**: Deep dive into message flow in [Architecture: Message Broker](../architecture/message_broker.md)

---

## FAQ

**Q: Can I train in distributed mode?**
A: Yes! Training works in both modes. Centralized is faster but distributed is more realistic.

**Q: Do I need to change my RL algorithm?**
A: No. The PettingZoo interface is identical in both modes.

**Q: How do I deploy to real hardware?**
A: Train in either mode, then deploy in distributed mode with Kafka + real SCADA data.

**Q: What if performance differs between modes?**
A: It shouldn't (see our results). If it does, check for bugs in message handling or timing.

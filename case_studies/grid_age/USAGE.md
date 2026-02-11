# GridAges Case Study - Usage Guide

## Quick Start

### 1. Basic Training (CTDE)

```bash
source .venv/bin/activate
PYTHONPATH=.:$PYTHONPATH python case_studies/grid_age/train.py
```

This will:
- Train 3 microgrid agents (MG1, MG2, MG3)
- Run for 100 episodes (24 hours each)
- Use policy gradient with critic baseline
- Print progress every 10 episodes

Expected output:
```
Training 3 agents: ['MG1', 'MG2', 'MG3']
Observation dimension: 111
Action dimension: 4 (P_ess, P_dg, Q_pv, Q_wind)

Episode  10/100: avg_return=-13479.09, avg_reward=-187.210
...
Training complete!
```

### 2. Event-Driven Testing

```bash
PYTHONPATH=.:$PYTHONPATH python case_studies/grid_age/test_event_driven.py
```

This will:
- Train policies using CTDE
- Test in event-driven mode with timing delays
- Compare synchronous vs event-driven execution

Expected output:
```
✅ Configured 5 agents with timing delays
   - Field agents: tick every 1.0h
   - Observation delay: 0.01h
   - Action delay: 0.02h
   - Jitter: 5% Gaussian

✅ Event-driven simulation complete!
Final Step Rewards:
   - MG1: -202.004
   - MG2: -207.203
   - MG3: -206.050
```

### 3. RLlib MAPPO Training (if Ray installed)

```bash
PYTHONPATH=.:$PYTHONPATH python case_studies/grid_age/train_rllib.py --num-iterations 100
```

Options:
- `--num-microgrids`: Number of microgrids (default: 3)
- `--num-iterations`: Training iterations (default: 100)
- `--checkpoint-freq`: Checkpoint frequency (default: 20)

Note: Requires `pip install ray[rllib]`

### 4. Run Tests

```bash
source .venv/bin/activate
pytest case_studies/grid_age/tests/ -v
```

Expected: 38/38 tests passing

## Python API Usage

### Creating an Environment

```python
from case_studies.grid_age import MicrogridEnv

# Create 3-microgrid environment
env = MicrogridEnv(
    num_microgrids=3,
    episode_steps=24,  # 24-hour episodes
    dt=1.0,           # 1-hour time steps
)

# Reset
obs, info = env.reset(seed=42)
# obs = {
#     'MG1': Observation(...),  # 111-dim vector
#     'MG2': Observation(...),
#     'MG3': Observation(...),
# }
```

### Training Policies

```python
from case_studies.grid_age import train_microgrid_ctde

# Train for 100 episodes
policies = train_microgrid_ctde(
    env,
    num_episodes=100,
    steps_per_episode=24,
    gamma=0.99,
    lr=0.02,
)

# policies = {'MG1': NeuralPolicy(...), 'MG2': ..., 'MG3': ...}
```

### Using Trained Policies

```python
# Attach policies to environment
env.set_agent_policies(policies)

# Run deterministic evaluation
obs, _ = env.reset(seed=999)

for step in range(24):
    actions = {
        aid: policy.forward_deterministic(obs[aid])
        for aid, policy in policies.items()
    }

    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated.get("__all__", False):
        break
```

### Event-Driven Mode

```python
from heron.scheduling import TickConfig, JitterType
from heron.scheduling.analysis import EventAnalyzer

# Configure timing with jitter
for agent in env.registered_agents.values():
    if hasattr(agent, 'tick_config'):
        agent.tick_config = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.01,
            act_delay=0.02,
            msg_delay=0.01,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.05,
            seed=42
        )

# Reset and run
env.reset(seed=42)
analyzer = EventAnalyzer(verbose=True)
result = env.run_event_driven(event_analyzer=analyzer, t_end=24.0)

print(f"Observations: {result.observation_count}")
print(f"State updates: {result.state_update_count}")
```

## Environment Details

### Agents and Devices

Each microgrid (MG1, MG2, MG3) controls:

| Device | Capacity | Control | State |
|--------|----------|---------|-------|
| ESS | 2 MWh, ±0.5 MW | Active power | P, Q, SOC |
| DG | 0.5-0.66 MW | Active power | P, Q, on |
| PV | 0.1 MW | Reactive power | P (external), Q |
| Wind | 0.1 MW | Reactive power | P (external), Q |
| Grid | Unlimited | N/A | P, Q, price |

### Observation Space (111 dimensions per agent)

```
ESSFeature:     [P, Q, SOC, capacity, min_p, max_p, min_soc, max_soc, ch_eff, dsc_eff]  (11 dims)
DGFeature:      [P, Q, on, max_p, min_p, max_q, min_q, startup, shutdown, ...]          (13 dims)
RESFeature(PV): [P, Q, max_p, max_q, availability]                                       (5 dims)
RESFeature(Wind): [P, Q, max_p, max_q, availability]                                     (5 dims)
GridFeature:    [P, Q, price, sell_discount, max_import, max_export]                    (6 dims)
NetworkFeature: [voltage_min, voltage_max, voltage_avg, max_line_loading, ...]          (6 dims)
... + neighbor features from other agents (public visibility)
```

### Action Space (4 dimensions per agent)

```
action = [P_ess, P_dg, Q_pv, Q_wind]  # All in [-1, 1]

Denormalized to:
- P_ess:  [-0.5, 0.5] MW (charge/discharge)
- P_dg:   [0.1, 0.66] MW (depends on MG)
- Q_pv:   [-0.05, 0.05] MVAr
- Q_wind: [-0.05, 0.05] MVAr
```

### Reward Function

```
reward = -(cost + penalty × safety)

Cost:
  - ESS cycling: |P_ess| × 0.1 × dt
  - DG fuel: (a×P² + b×P + c) × dt
  - Grid energy: P_grid × price × dt

Safety:
  - Voltage violations: (|V - 1.0| - 0.05) × 10 per bus
  - Line overloading: (loading - 1.0) × 20 per line
  - SOC violations: |SOC - limits| × 100
```

### Time Profiles (24-hour)

**Electricity Price:**
- Off-peak (12am-6am): ~30-35 $/MWh
- Mid-peak (6am-12pm, 8pm-12am): ~40-50 $/MWh
- Peak (12pm-8pm): ~55-70 $/MWh

**Solar PV:**
- Night (6pm-6am): 0% availability
- Day (6am-6pm): 10-100% availability (peak at noon)

**Wind:**
- Variable with sinusoidal pattern + noise
- Range: 20-80% availability

## Advanced Usage

### Custom Device Parameters

```python
from case_studies.grid_age.agents import MicrogridFieldAgent

# Create custom microgrid agent
mg = MicrogridFieldAgent(
    agent_id="CustomMG",
    ess_capacity=5.0,      # 5 MWh ESS
    ess_min_p=-1.0,        # 1 MW discharge
    ess_max_p=1.0,         # 1 MW charge
    dg_max_p=1.5,          # 1.5 MW DG
    pv_max_p=0.5,          # 0.5 MW PV
    wind_max_p=0.5,        # 0.5 MW Wind
)
```

### Custom Network Topology

Modify `MicrogridEnv._create_network()` to use different IEEE test systems or create custom topologies.

### Custom Reward Function

Override `MicrogridFieldAgent.compute_local_reward()`:

```python
class CustomMicrogridAgent(MicrogridFieldAgent):
    def compute_local_reward(self, local_state: dict) -> float:
        # Extract features
        ess_vec = local_state.get("ESSFeature", np.zeros(11))

        # Custom reward logic
        soc = float(ess_vec[2])
        reward = soc  # Maximize SOC

        return reward
```

### Hierarchical Control with Protocols

```python
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.protocols import SetpointProtocol

# Create coordinator that controls microgrids
coordinator = CoordinatorAgent(
    agent_id="DSO",
    protocol=SetpointProtocol(),
    subordinates={"MG1": mg1, "MG2": mg2, "MG3": mg3},
    policy=CoordinatorPolicy(),  # Outputs joint actions
)

system = SystemAgent(
    agent_id="system",
    subordinates={"DSO": coordinator}
)

env = MicrogridEnv(system_agent=system)
```

## Troubleshooting

### Issue: Power flow doesn't converge

**Solution:** Check device setpoints are reasonable. Try:
```python
# Reduce action magnitudes
action.set_values(c=np.clip(action.c, -0.5, 0.5))
```

### Issue: Observation dimension mismatch

**Solution:** Use `obs.vector()` to get full concatenated observation:
```python
obs_vec = obs[agent_id].vector()  # Full 111-dim vector
```

### Issue: Import errors

**Solution:** Set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/powergrid:$PYTHONPATH
# OR
PYTHONPATH=.:$PYTHONPATH python your_script.py
```

### Issue: RLlib not available

**Solution:** Install Ray:
```bash
pip install ray[rllib]
```

## Performance Tips

1. **Reduce episode length** for faster training:
   ```python
   env = MicrogridEnv(episode_steps=12)  # 12-hour episodes
   ```

2. **Use fewer microgrids** for initial development:
   ```python
   env = MicrogridEnv(num_microgrids=1)
   ```

3. **Smaller networks** for debugging:
   ```python
   policy = NeuralPolicy(obs_dim=111, hidden_dim=32)  # vs 64
   ```

4. **Profile code** if slow:
   ```bash
   python -m cProfile -o profile.stats case_studies/grid_age/train.py
   ```

## Files Reference

| File | Purpose |
|------|---------|
| `features/device_features.py` | Device FeatureProviders |
| `agents/microgrid_agent.py` | MicrogridFieldAgent |
| `envs/microgrid_env.py` | MicrogridEnv with Pandapower |
| `train.py` | CTDE training script |
| `test_event_driven.py` | Event-driven testing |
| `train_rllib.py` | RLlib MAPPO integration |
| `tests/` | Unit and integration tests |

## Next Steps

1. ✅ **Event-driven testing** - Complete and working
2. **Hyperparameter tuning** - Optimize learning rate, network size
3. ✅ **RLlib integration** - MAPPO training script created
4. **Validation** - Compare with original GridAges results
5. **Extensions** - Add more device types, topologies, control strategies

## Support

For issues or questions:
1. Check test files for examples
2. Review [gridages_analysis_and_migration.md](../../docs/gridages_analysis_and_migration.md)
3. See original GridAges: https://github.com/hepengli/GridAges

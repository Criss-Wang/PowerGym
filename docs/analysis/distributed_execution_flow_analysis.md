# Distributed/Decentralized Execution Flow Analysis

## Executive Summary

**Status**: ✅ **CLEAN** - No net info is passed to subordinate agents in distributed mode

The codebase correctly implements the separation between centralized and distributed execution modes. In distributed mode, agents never receive the `net` object directly. Instead, they receive pre-computed network state metrics via message broker.

---

## Complete Execution Flow

### 1. Environment Entry Point (`networked_grid_env.py`)

```python
def step(self, action_n):
    if self.centralized:
        # CENTRALIZED: Direct net access
        for agent in self.agent_dict.values():
            agent.update_state(self.net, self._t)        # ❌ Net passed
            agent.update_cost_safety(self.net)           # ❌ Net passed
    else:
        # DISTRIBUTED: Message-based communication
        async def run_distributed_steps():
            tasks = []
            for agent_id, action in action_n.items():
                if agent_id in self.actionable_agents:
                    self._send_actions_to_agent(agent_id, action)
                    tasks.append(self.actionable_agents[agent_id].step_distributed())
            await asyncio.gather(*tasks)

        asyncio.run(run_distributed_steps())

        # Update states via messages (no net passed)
        self._update_loads_distributed()
        state_updates = self._consume_all_state_updates()
        self._apply_state_updates_to_net(state_updates)

        # Publish network results via messages
        self._publish_network_state_to_agents()

        # Agents compute cost/safety from received messages
        for agent in self.agent_dict.values():
            agent.update_cost_safety(None)               # ✅ No net passed
```

---

### 2. Hierarchical Agent Flow (`base.py::step_distributed`)

```python
async def step_distributed(self) -> None:
    """Execute one step with hierarchical message-based communication."""

    # 1. Receive action from upstream (via message broker)
    upstream_action = self._get_action_from_upstream()
    upstream_info = self._get_info_from_upstream()
    self._update_state_with_upstream_info(upstream_info)

    # 2-4. Handle subordinates if any
    if self.subordinates:
        # Derive actions for subordinates
        downstream_actions = await self._derive_downstream_actions(upstream_action)

        # Send actions to subordinates via message broker
        self._send_actions_to_subordinates(downstream_actions)

        # Execute subordinate steps recursively
        await self._execute_subordinates()

        # Collect info from subordinates via message broker
        self._collect_subordinates_info()
        self._update_state_with_subordinates_info()

    # 5. Derive and execute own action
    local_action = self._derive_local_action(upstream_action)
    self._execute_local_action(local_action)
    self._update_state_post_step()

    # 6. Publish state updates to environment
    self._update_timestep()
    self._publish_state_updates()
```

**Key Points**:
- All communication via message broker
- No `net` object in this entire flow
- State updates published to environment, not passed around

---

### 3. GridAgent Implementation (`grid_agent.py`)

#### 3.1 Update State (Centralized vs Distributed)

```python
def update_state(self, net, t):
    """CENTRALIZED MODE ONLY - called by environment with net parameter."""
    load_scaling = self.dataset['load'][t]

    # Update network directly
    local_ids = pp.get_element_index(net, 'load', self.name, False)
    net.load.loc[local_ids, 'scaling'] = load_scaling

    # Update all generators with their actions
    for name, generator in self.sgen.items():
        generator.update_state()  # ✅ No net passed to device

        # GridAgent syncs device state to net
        local_ids = pp.get_element_index(net, 'sgen', self.name + ' ' + name)
        values = [generator.electrical.P_MW, generator.electrical.Q_MVAr, ...]
        net.sgen.loc[local_ids, states] = values
```

**In distributed mode**: This method is NOT called. Instead:
- Environment calls `_update_loads_distributed()`
- Devices publish state via `_publish_state_updates()`
- Environment consumes state updates and applies to net

#### 3.2 Update Cost/Safety (Centralized vs Distributed)

```python
def update_cost_safety(self, net):
    """Update cost and safety metrics.

    Args:
        net: PandaPower network (None in distributed mode)
    """
    self.cost, self.safety = 0, 0

    # Always update device-level costs (devices have local state)
    for dg in self.sgen.values():
        dg.update_cost_safety()  # ✅ No net passed to device
        self.cost += dg.cost
        self.safety += dg.safety

    # Network-level safety metrics
    if net is not None:
        # CENTRALIZED: Access net directly
        local_bus_ids = pp.get_element_index(net, 'bus', self.name, False)
        local_vm = net.res_bus.loc[local_bus_ids].vm_pu.values
        overvoltage = np.maximum(local_vm - 1.05, 0).sum()
        # ... compute safety from net
    else:
        # DISTRIBUTED: Receive network state via messages
        if self.message_broker and self.env_id:
            network_state = self._consume_network_state()
            if network_state and network_state.get('converged', False):
                # Extract pre-computed safety metrics from message
                bus_voltages = network_state.get('bus_voltages', {})
                line_loading = network_state.get('line_loading', {})

                overvoltage = bus_voltages.get('overvoltage', 0)    # ✅ Pre-computed
                undervoltage = bus_voltages.get('undervoltage', 0)  # ✅ Pre-computed
                overloading = line_loading.get('overloading', 0)    # ✅ Pre-computed

                self.safety += overloading + overvoltage + undervoltage
```

**Key Point**: In distributed mode, agents receive **pre-computed aggregates**, not raw network data.

---

### 4. Device-Level Implementation

#### 4.1 Generator (`generator.py`)

```python
def update_state(self, **kwargs) -> None:
    """Update generator state with optional kwargs."""
    self._update_uc_status()       # Uses internal action state
    self._update_power_outputs()   # Uses internal action state

    if kwargs:
        self._update_by_kwargs(**kwargs)  # Optional feature updates

def update_cost_safety(self) -> None:
    """Economic cost + S/PF penalties + UC start/stop cost."""
    P = self.electrical.P_MW or 0.0      # ✅ Uses own state
    Q = self.electrical.Q_MVAr or 0.0    # ✅ Uses own state
    on = 1.0 if self.status.state == "online" else 0.0
    dt = self._generator_config.dt_h

    # Cost computation from internal state
    fuel_cost = cost_from_curve(P, self._generator_config.cost_curve_coefs)
    uc_cost = self._uc_cost_step
    self.cost = fuel_cost * on * dt + uc_cost

    # Safety violations from internal state
    violations = self.limits.feasible(P, Q)
    self.safety = np.sum(list(violations.values())) * on * dt
```

**Status**: ✅ **CLEAN** - No network info needed or used

#### 4.2 Storage (`storage.py`)

```python
def update_state(self, **kwargs) -> None:
    """Apply P/Q from action, update SOC/degradation."""
    P_eff, Q_eff = self._update_power_outputs()    # ✅ Uses action
    self._update_storage_dynamics(P_eff)           # ✅ Uses action

    if kwargs:
        self._update_by_kwargs(**kwargs)

def update_cost_safety(self) -> None:
    """Update per-step cost and safety penalties."""
    P = self.electrical.P_MW or 0.0    # ✅ Uses own state
    Q = self.electrical.Q_MVAr or 0.0  # ✅ Uses own state
    dt = self._storage_config.dt_h

    # Degradation cost from internal state
    degr_cost_inc = ...
    self.cost = degr_cost_inc

    # Safety from internal state
    safety = self.storage.soc_violation()
    if self.limits is not None:
        violations = self.limits.feasible(P, Q)
        safety += np.sum(list(violations.values())) * dt
    self.safety = safety
```

**Status**: ✅ **CLEAN** - No network info needed or used

#### 4.3 Transformer (`transformer.py`)

```python
def update_state(self, **kwargs) -> None:
    """Update tap position from action."""
    cfg = self._transformer_config
    if cfg.tap_max is not None and cfg.tap_min is not None and self.action.d.size:
        new_tap = int(self.action.d[0]) + int(cfg.tap_min)
        self.tap_changer.set_values(tap_position=new_tap)

    if kwargs:
        self.tap_changer.set_values(**kwargs)

def update_cost_safety(self, **kwargs) -> None:
    """Update cost from tap changes and safety from loading.

    Args:
        **kwargs: Optional keyword arguments:
            loading_percentage: Transformer loading percentage
    """
    loading_percentage = kwargs.get("loading_percentage", 0.0)  # ⚠️ Needs net info

    # Safety: loading-derived penalty
    self.safety = loading_over_pct(loading_percentage)

    # Cost: tap change operations
    delta = abs(self.tap_changer.tap_position - self._last_tap_position)
    self.cost = tap_change_cost(delta, self._transformer_config.tap_change_cost)
```

**Status**: ⚠️ **POTENTIAL ISSUE** - `loading_percentage` is expected from kwargs but:
- Currently defaults to 0.0 if not provided
- In distributed mode, GridAgent calls `device.update_cost_safety()` with no args
- This means transformer safety is always 0 in distributed mode!

---

## 5. Environment's Network State Publishing

```python
def _publish_network_state_to_agents(self):
    """Publish network state to agents via messages."""

    for agent in self.agent_dict.values():
        network_state = {
            'converged': self.net.get('converged', False),
            'device_results': {},
            'bus_voltages': {},
            'line_loading': {}
        }

        # Device results (sgen outputs)
        for device_name in agent.sgen.keys():
            element_name = f"{agent.name} {device_name}"
            idx = pp.get_element_index(self.net, 'sgen', element_name)
            network_state['device_results'][device_name] = {
                'p_mw': float(self.net.res_sgen.loc[idx, 'p_mw']),
                'q_mvar': float(self.net.res_sgen.loc[idx, 'q_mvar'])
            }

        # Bus voltages - PRE-COMPUTED AGGREGATES
        if network_state['converged']:
            local_bus_ids = pp.get_element_index(self.net, 'bus', agent.name, False)
            bus_voltages = self.net.res_bus.loc[local_bus_ids, 'vm_pu'].values
            network_state['bus_voltages'] = {
                'vm_pu': bus_voltages.tolist(),
                'overvoltage': float(np.maximum(bus_voltages - 1.05, 0).sum()),    # ✅
                'undervoltage': float(np.maximum(0.95 - bus_voltages, 0).sum())   # ✅
            }

            # Line loading - PRE-COMPUTED AGGREGATES
            local_line_ids = pp.get_element_index(self.net, 'line', agent.name, False)
            line_loading = self.net.res_line.loc[local_line_ids, 'loading_percent'].values
            network_state['line_loading'] = {
                'loading_percent': line_loading.tolist(),
                'overloading': float(np.maximum(line_loading - 100, 0).sum() * 0.01)  # ✅
            }

        # Send via message broker
        channel = ChannelManager.power_flow_result_channel(self._env_id, agent.agent_id)
        message = Message(payload=network_state, ...)
        self.message_broker.publish(channel, message)
```

**Key Points**:
- Environment computes safety aggregates (overvoltage, undervoltage, overloading)
- Agents receive only pre-computed metrics, not raw network data
- This preserves information hiding in distributed mode

---

## Issues Found

### Issue 1: Transformer Loading Percentage ⚠️

**Location**: `transformer.py::update_cost_safety()`

**Problem**:
```python
loading_percentage = kwargs.get("loading_percentage", 0.0)
```

In distributed mode:
1. GridAgent calls `device.update_cost_safety()` with no arguments
2. Transformer always gets `loading_percentage=0.0`
3. Transformer safety is always 0 (incorrect)

**Root Cause**:
- Transformer needs network-derived info (loading percentage)
- In centralized mode, this would come from `net.res_trafo`
- In distributed mode, no mechanism exists to pass this to the device

**Root Cause Analysis**:

The environment's `_publish_network_state_to_agents()` currently publishes:
- ✅ Device results (sgen P/Q)
- ✅ Bus voltages (with pre-computed safety metrics)
- ✅ Line loading (with pre-computed safety metrics)
- ❌ **Missing: Transformer loading!**

**CRITICAL**: `GridAgent.update_cost_safety()` does NOT iterate over transformers at all! This means **transformers are completely ignored in cost/safety computation in BOTH centralized and distributed modes**.

**The Fix** (3 parts):

#### Part 1: Add transformer results to published network state

```python
# In networked_grid_env.py::_publish_network_state_to_agents()
# After line 247 (after line_loading block), add:

# Transformer loading for this agent's transformers
if hasattr(agent, 'transformers'):
    try:
        trafo_results = {}
        for trafo_name in agent.transformers.keys():
            element_name = f"{agent.name} {trafo_name}"
            idx = pp.get_element_index(self.net, 'trafo', element_name)
            trafo_results[trafo_name] = {
                'loading_percent': float(self.net.res_trafo.loc[idx, 'loading_percent'])
            }
        network_state['trafo_results'] = trafo_results
    except (KeyError, UserWarning):
        network_state['trafo_results'] = {}
```

#### Part 2: GridAgent iterates over transformers in update_cost_safety

```python
# In grid_agent.py::update_cost_safety()
# After line 842 (after the distributed bus/line safety block), add:

# Update transformer costs/safety
if hasattr(self, 'transformers'):
    if net is not None:
        # Centralized: read from net directly
        for trafo_name, trafo in self.transformers.items():
            element_name = f"{self.name} {trafo_name}"
            local_ids = pp.get_element_index(net, 'trafo', element_name)
            loading_pct = float(net.res_trafo.loc[local_ids, 'loading_percent'].values[0])
            trafo.update_cost_safety(loading_percentage=loading_pct)
            self.cost += trafo.cost
            self.safety += trafo.safety
    else:
        # Distributed: read from received network state
        if self.message_broker and self.env_id:
            network_state = self._consume_network_state()
            trafo_results = network_state.get('trafo_results', {})
            for trafo_name, trafo in self.transformers.items():
                loading_pct = trafo_results.get(trafo_name, {}).get('loading_percent', 0.0)
                trafo.update_cost_safety(loading_percentage=loading_pct)
                self.cost += trafo.cost
                self.safety += trafo.safety
```

#### Part 3: Update sync_global_state to sync transformer results

```python
# In grid_agent.py::sync_global_state()
# After the sgen sync loop (around line 828), add:

# Sync transformer results to devices
if hasattr(self, 'transformers'):
    for trafo_name, trafo in self.transformers.items():
        element_name = f"{self.name} {trafo_name}"
        local_ids = pp.get_element_index(net, 'trafo', element_name)
        loading_pct = float(net.res_trafo.loc[local_ids, 'loading_percent'].values[0])
        # Store in transformer state for potential use
        # (Currently not needed since update_cost_safety passes it via kwargs)
```

**Note**: sync_global_state is only used in centralized mode, so Part 3 is optional if we're passing loading_percentage via kwargs in Part 2.

---

## Summary of Net Info Flow

### Centralized Mode
```
Environment (has net)
    ↓ passes net directly
GridAgent.update_state(net, t)
    ↓ writes to net
    ↓ reads from net
GridAgent.update_cost_safety(net)
    ↓ calls
DeviceAgent.update_state()        # ✅ No net
DeviceAgent.update_cost_safety()  # ✅ No net
```

### Distributed Mode
```
Environment (has net)
    ↓ publishes via messages (pre-computed aggregates)
    ↓
MessageBroker
    ↓
GridAgent.update_cost_safety(None)
    ↓ consumes messages
    ↓ extracts pre-computed metrics
    ↓ calls
DeviceAgent.update_state()        # ✅ No net
DeviceAgent.update_cost_safety()  # ✅ No net (except transformer issue)
```

---

## Conclusion

**Overall Status**: ✅ **Architecture is sound**

The distributed execution flow correctly avoids passing network objects to subordinate agents. All information flows through:
1. **Message Broker** for agent-to-agent communication
2. **Pre-computed Aggregates** for network state metrics
3. **Device Internal State** for device-level computations

**Action Items**:
1. ⚠️ Fix transformer loading percentage issue (see Issue 1 above)
2. ✅ Verify all device implementations follow the "compute from own state" pattern
3. ✅ Ensure all network-derived info is published via messages with pre-computed aggregates

**No net info leakage detected** in the message-passing flow.

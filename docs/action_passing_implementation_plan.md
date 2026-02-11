# Implementation Plan: Action-Passing in Event-Driven Testing Mode

## Overview

Enable parent-to-subordinate action-passing in event-driven testing mode to match the hierarchical control capabilities already available in CTDE training. This allows parent agents (SystemAgent, CoordinatorAgent) to compute and send actions to subordinates instead of having all agents compute their own actions independently.

## Current State vs Target State

### Current State
- **CTDE Training**: ✅ Action-passing works via hierarchical action dict
- **Event-Driven Testing**: ❌ Each agent computes own action via policy
- **Infrastructure**: ✅ Message broker and protocols exist but unused

### Target State
- **Event-Driven Testing**: ✅ Parent agents can send coordinated actions to subordinates
- **Backward Compatible**: ✅ Existing code without protocols continues to work
- **Protocol-Driven**: ✅ Automatic toggle based on protocol type

## Architecture Decisions

### 1. Action Routing Strategy
- **Use existing MessageBroker** for action transport (not event payloads)
- **Use existing Protocol.coordinate()** to compute subordinate actions
- **Actions sent during ACTION_EFFECT phase** after parent's action is applied
- **Subordinates check broker at tick start** before computing own action

### 2. Execution Flow

```
Parent Tick:
  1. Check for upstream action → compute_action()
  2. Policy computes own action
  3. Protocol.coordinate() computes subordinate actions
  4. ACTION_EFFECT: apply_action() + send_subordinate_actions()

Subordinate Tick:
  1. Check message broker for upstream action
  2. If found: use upstream action
  3. If not found: compute own action via policy
  4. ACTION_EFFECT: apply_action()
```

### 3. Backward Compatibility
- **Opt-in via protocol**: Agents without protocols remain self-directed
- **Graceful degradation**: Missing actions fall back to policy
- **No breaking changes**: All existing tests continue to pass

## Implementation Details

### Phase 1: Core Infrastructure (`heron/agents/base.py`)

#### A. Add capability check method (after line ~455)
```python
def should_send_subordinate_actions(self) -> bool:
    """Check if agent should coordinate subordinate actions."""
    if not self.subordinates or not self.protocol:
        return False
    from heron.protocols.base import NoActionCoordination
    return not isinstance(self.protocol.action_protocol, NoActionCoordination)
```

#### B. Add state fields in `__init__` (after line ~76)
```python
self._pending_subordinate_actions: Dict[AgentID, Any] = {}
self._upstream_action: Optional[Any] = None
```

#### C. Modify `compute_action()` (lines 410-418)
Replace entire method:
```python
def compute_action(self, obs: Any, scheduler: EventScheduler):
    """Compute action for self and optionally for subordinates."""
    # Priority: upstream action > policy > no action
    if self._upstream_action is not None:
        self.set_action(self._upstream_action)
        self._upstream_action = None  # Clear after use
    elif self.policy:
        self.set_action(self.policy.forward(observation=obs))
    else:
        if self.level == 1 and not self.upstream_id:
            print(f"Warning: {self} has no policy and no upstream")

    # Compute subordinate actions if protocol supports it
    self._pending_subordinate_actions = {}
    if self.should_send_subordinate_actions():
        subordinate_states = {
            sub_id: sub.state for sub_id, sub in self.subordinates.items()
        }
        messages, actions = self.protocol.coordinate(
            coordinator_state=self.state,
            subordinate_states=subordinate_states,
            coordinator_action=self.action,
            context={"subordinates": self.subordinates, "timestamp": self._timestep}
        )
        self._pending_subordinate_actions = actions

    scheduler.schedule_action_effect(
        agent_id=self.agent_id,
        delay=self._tick_config.act_delay,
    )
```

#### D. Add upstream action check method (after line ~533)
```python
def _check_for_upstream_action(self) -> None:
    """Check message broker for action from upstream parent."""
    if not self.upstream_id or not self._message_broker:
        self._upstream_action = None
        return

    actions = self.receive_upstream_action(
        sender_id=self.upstream_id,
        clear=True,
    )
    self._upstream_action = actions[-1] if actions else None
```

#### E. Add subordinate action sending method (after line ~533)
```python
def send_subordinate_actions(
    self,
    subordinate_actions: Dict[AgentID, Any],
    scheduler: Optional[EventScheduler] = None,
) -> None:
    """Send coordinated actions to subordinates via message broker."""
    if self._message_broker is None:
        raise RuntimeError(
            f"Agent {self.agent_id} has no message broker. "
            "Cannot send subordinate actions."
        )

    for sub_id, action in subordinate_actions.items():
        if action is not None and sub_id in self.subordinates:
            self.send_subordinate_action(
                recipient_id=sub_id,
                action=action,
            )
```

#### F. Modify `tick()` to check upstream (before line ~169)
Add at start of tick():
```python
def tick(self, scheduler: EventScheduler, current_time: float) -> None:
    self._timestep = current_time

    # Check for upstream action from parent
    self._check_for_upstream_action()

    # [Rest of existing tick logic...]
```

### Phase 2: SystemAgent (`heron/agents/system_agent.py`)

#### Override `action_effect_handler()` (replace lines 173-181)
```python
@Agent.handler("action_effect")
def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
    """Apply local action and send subordinate actions if coordinating."""
    self.apply_action()

    # Send coordinated actions to coordinators
    if self._pending_subordinate_actions:
        self.send_subordinate_actions(
            subordinate_actions=self._pending_subordinate_actions,
            scheduler=scheduler,
        )
        self._pending_subordinate_actions = {}

    # Update state in proxy
    scheduler.schedule_message_delivery(
        sender_id=self.agent_id,
        recipient_id=PROXY_AGENT_ID,
        message={"set_state": "local", "body": self.state.to_dict(include_metadata=True)},
        delay=self._tick_config.msg_delay,
    )
```

### Phase 3: CoordinatorAgent (`heron/agents/coordinator_agent.py`)

#### A. Update comment (lines 93-94)
Replace:
```python
# Currently, we assume NO upstream action passed down
```
With:
```python
# Upstream actions checked in base.Agent.tick() via _check_for_upstream_action()
```

#### B. Override `action_effect_handler()` (replace lines 121-129)
```python
@Agent.handler("action_effect")
def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
    """Apply local action and send subordinate actions if coordinating."""
    self.apply_action()

    # Send coordinated actions to field agents
    if self._pending_subordinate_actions:
        self.send_subordinate_actions(
            subordinate_actions=self._pending_subordinate_actions,
            scheduler=scheduler,
        )
        self._pending_subordinate_actions = {}

    # Update state in proxy
    scheduler.schedule_message_delivery(
        sender_id=self.agent_id,
        recipient_id=PROXY_AGENT_ID,
        message={"set_state": "local", "body": self.state.to_dict(include_metadata=True)},
        delay=self._tick_config.msg_delay,
    )
```

### Phase 4: FieldAgent (`heron/agents/field_agent.py`)

#### Update comment (lines 179-180)
Replace:
```python
# Currently, we assume NO upstream action passed down
```
With:
```python
# Upstream actions checked in base.Agent.tick() via _check_for_upstream_action()
```

**Note:** No handler override needed - FieldAgents have no subordinates.

## Critical Files to Modify

| File | Lines Modified | Changes |
|------|---------------|---------|
| `heron/agents/base.py` | ~76, ~410-418, ~533+ | Add state fields, modify compute_action(), add 3 new methods, modify tick() |
| `heron/agents/system_agent.py` | ~173-181 | Override action_effect_handler() |
| `heron/agents/coordinator_agent.py` | ~93-94, ~121-129 | Update comment, override action_effect_handler() |
| `heron/agents/field_agent.py` | ~179-180 | Update comment only |

## Testing Strategy

### Test File Structure
```
tests/
├── unit/
│   └── test_action_passing.py (NEW)
│       ├── test_should_send_subordinate_actions()
│       ├── test_upstream_action_receiving()
│       ├── test_compute_action_with_upstream()
│       └── test_compute_action_without_upstream()
└── integration/
    ├── test_hierarchical_action_passing.py (NEW)
    │   ├── test_setpoint_protocol_action_passing()
    │   ├── test_no_protocol_self_directed()
    │   └── test_mixed_mode()
    └── test_e2e.py (MODIFY)
        └── test_e2e_with_setpoint_protocol() (ADD)
```

### Test Scenarios

#### 1. Hierarchical Action-Passing (SetpointProtocol)
- SystemAgent computes action via policy
- SystemAgent.protocol.coordinate() computes coordinator actions
- CoordinatorAgent receives action via broker
- CoordinatorAgent.protocol.coordinate() computes field actions
- FieldAgent receives and executes action

#### 2. Self-Directed Mode (NoProtocol)
- Agents without protocols compute own actions via policy
- No messages sent via broker
- Existing behavior preserved

#### 3. Mixed Mode
- Some agents use protocols (parent-controlled)
- Other agents self-direct (own policy)
- Both modes coexist

### Regression Tests
- All existing tests in `test_e2e.py` must pass unchanged
- No breaking changes to CTDE training flow

## Edge Cases Handled

### 1. Action Timing Mismatch
- **Problem:** Subordinate ticks before parent's action arrives
- **Solution:** Fall back to policy (graceful degradation)

### 2. Protocol Returns None
- **Problem:** Protocol returns `{agent1: None}` (decentralized mode)
- **Solution:** Skip sending None actions, subordinate uses policy

### 3. No Message Broker
- **Problem:** Agent tries to send without broker
- **Solution:** Raise clear RuntimeError with setup instructions

### 4. Multiple Actions in Broker
- **Problem:** Multiple actions queued (timing bug)
- **Solution:** Use most recent action, log warning

## Verification Steps

### 1. Unit Tests
```bash
source .venv/bin/activate
pytest tests/unit/test_action_passing.py -v
```
Expected: All tests pass

### 2. Integration Tests
```bash
pytest tests/integration/test_hierarchical_action_passing.py -v
```
Expected: 3 scenarios pass (hierarchical, self-directed, mixed)

### 3. End-to-End Test
```bash
pytest tests/integration/test_e2e.py -v
```
Expected: All existing tests pass + new protocol test passes

### 4. Manual Verification
Run modified `test_e2e.py` with SetpointProtocol:
- Add debug print in `compute_action()` to show upstream action usage
- Verify subordinates receive actions from parents
- Check event analyzer output for action message deliveries

### 5. Performance Check
- Event count should increase by ~1-2 per parent per tick
- No memory leaks (actions cleared after consumption)
- Latency < 1ms per action send/receive

## Migration Guide for Existing Code

### Before (Self-Directed)
```python
coordinator = CoordinatorAgent(
    agent_id="coord1",
    subordinates={"field1": field_agent1}
)
# Each agent computes own action
```

### After (Parent-Controlled)
```python
coordinator = CoordinatorAgent(
    agent_id="coord1",
    protocol=SetpointProtocol(),  # ← Enable coordination
    policy=JointPolicy(),          # ← Outputs joint action
    subordinates={"field1": field_agent1}
)
# Coordinator computes actions for subordinates
```

### Backward Compatibility
Existing code without protocols continues to work unchanged - no migration required unless you want parent control.

## Implementation Checklist

- [ ] Phase 1: Implement base.py changes (5 new methods, 2 state fields, modify 2 methods)
- [ ] Phase 2: Override SystemAgent.action_effect_handler()
- [ ] Phase 3: Override CoordinatorAgent.action_effect_handler()
- [ ] Phase 4: Update FieldAgent comments
- [ ] Create unit tests (test_action_passing.py)
- [ ] Create integration tests (test_hierarchical_action_passing.py)
- [ ] Modify test_e2e.py (add protocol test case)
- [ ] Run all tests and verify pass
- [ ] Manual verification with debug prints
- [ ] Performance benchmarking

## Success Criteria

1. ✅ Parent agents can send actions to subordinates in event-driven mode
2. ✅ Subordinates receive and use upstream actions
3. ✅ Backward compatible - existing tests pass unchanged
4. ✅ Protocol-driven toggle works automatically
5. ✅ Graceful degradation when actions missing
6. ✅ 90%+ code coverage for new methods
7. ✅ All 3 test scenarios pass (hierarchical, self-directed, mixed)
8. ✅ Performance overhead < 5ms per coordination cycle

## Estimated Effort

- **Implementation**: ~545 lines across 5 files
- **Testing**: ~300 lines of test code
- **Total Time**: 2-3 days (implementation + testing + verification)

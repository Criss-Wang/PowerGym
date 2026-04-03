# Heron Implementation & Documentation Plan

> **Status**: Draft — last updated 2026-04-03
> **Assumption**: No backward-compatibility constraints. Breaking changes are acceptable.

---

## Dependency Graph

```
                  ┌──────────────────┐
                  │  Tier 0: Engine  │
                  │  ┌─────────────┐ │
                  │  │ Termination │ │
                  │  │ Coord.Reward│ │
                  │  │ Custom Evts │ │
                  │  └──────┬──────┘ │
                  └─────────┼────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼                           ▼
   ┌────────────────────┐     ┌────────────────────┐
   │  Tier 1: Ecosystem │     │  Tier 2: Onboarding│
   │  PettingZoo adaptor│     │  Thermostat env    │
   │  heron.make()      │     │  Sensor Network env│
   │                    │     │  Transport Fleet   │
   └────────┬───────────┘     │  Notebooks         │
            │                 └─────────┬──────────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
               ┌────────────────────┐
               │  Tier 3: Polish    │
               │  Full test suite   │
               │  Guides & docs     │
               │  Website           │
               └────────────────────┘
```

**Critical path**: 0A → 0B → 0C → 2C (Transport Fleet needs all three engine features).
Tiers 1A/1B can run in parallel with Tier 2 once Tier 0 lands.

---

## Tier 0 — Core Engine Completeness

These are load-bearing — every downstream feature assumes they work. Order matters within the tier.

### 0A. Episode Termination / Truncation Mechanism

> Why first: Reward, adaptor, and demo envs all need well-defined episode boundaries.

**Files**: `system_agent.py`, `base.py` (env), `episode_analyzer.py`

- [ ] Make `run_event_driven()` check agent `is_terminated`/`is_truncated` after each physics boundary and stop early (not just at `t_end`)
- [ ] Add `__all__` termination/truncation aggregation in event-driven mode (mirror training mode)
- [ ] Support configurable `__all__` semantics: `any` vs `all` (default `all`)
- [ ] Propagate termination state through `EpisodeAnalyzer` so adaptors see it
- [ ] Tests: early-termination scenario, mixed terminated/truncated agents, `__all__` logic

### 0B. Coordinator Reward Computation

> Why second: Depends on termination (coordinator must know when to stop cascading).

**Files**: `coordinator_agent.py`, `system_agent.py`

- [ ] Fix the rapid-physics-cycle race — queue incoming `MSG_PHYSICS_COMPLETED` instead of overwriting `_pending_sub_rewards` (`coordinator_agent.py:213-219`)
- [ ] Add explicit `compute_local_reward()` on `CoordinatorAgent` — first-class reward hook that receives aggregated subordinate rewards
- [ ] Document the reward timing contract: "coordinator reward fires only after all subordinate rewards complete"
- [ ] Tests: 2-level cascade timing, coordinator reward = f(subordinate rewards), rapid-physics stress test

### 0C. Custom Event Triggers

> Why third: Demo envs (especially Transport Fleet) need this, but it doesn't block termination/reward.

**Files**: `event.py`, `event_scheduler.py`, `base.py` (env), `builder.py`

- [ ] Allow `EventType` extensibility — either open string-based system or `EventType.CUSTOM` with a `custom_type: str` field
- [ ] Add `BaseEnv.register_event(name, priority, handler_target)` or similar init-time registration API
- [ ] Allow injection via `EnvBuilder.add_event_trigger(name, schedule_or_condition, handler)`
- [ ] Wire custom events into the scheduler's priority queue
- [ ] Tests: custom event fires at correct time, custom event interacts with physics, multiple custom event types coexist

---

## Tier 1 — Ecosystem Reach

Can be done in parallel with each other (and with Tier 2) once Tier 0 lands.

### 1A. PettingZoo Adaptor

> Opens the door to CleanRL, Tianshou, EPyMARL, and any AEC/parallel-env consumer.

**New file**: `heron/adaptors/pettingzoo.py`

- [ ] Implement `PettingZooParallelEnv` wrapping `BaseEnv` (parallel API is the better fit for CTDE)
- [ ] Handle `is_active_at(step)` → agent masking (inactive agents return `None` action)
- [ ] Map Heron `State`/`Action` ↔ PettingZoo `observation_spaces`/`action_spaces`
- [ ] Support `render()` stub (no-op initially)
- [ ] Compliance: pass PettingZoo's `parallel_api_test()`
- [ ] Stretch: AEC wrapper for turn-based research

### 1B. `heron.make()` Registry

**New file**: `heron/registry.py`

- [ ] Add `register(id, entry_point, kwargs)` and `make(id, **override_kwargs)` — mirror Gymnasium's pattern
- [ ] Auto-register demo envs on import (Tier 2)
- [ ] Add to `heron/__init__.py`: `from heron.registry import make, register`
- [ ] Tests: register, make, override kwargs, duplicate ID error

---

## Tier 2 — Onboarding & Demonstration

Order matters: each env teaches a progressively harder concept.

### 2A. Thermostat Env — "Hello World"

> Teaches: single field agent, continuous action, simple reward, episode truncation.

**New dir**: `heron/demo_envs/thermostat/`

- [ ] 1 field agent (heater), 1 feature (temperature), action = heat delta
- [ ] Simulation: `T_next = T + action + noise - cooling`
- [ ] Reward: `−|T − T_target|`
- [ ] Truncation at `max_steps`
- [ ] Register as `heron.make("Thermostat-v0")`

### 2B. Sensor Network Env — Visibility Showcase

> Teaches: multiple field agents, observation scoping, horizontal protocol.

**New dir**: `heron/demo_envs/sensor_network/`

- [ ] N sensor agents on a graph, each sees neighbors only
- [ ] Simulation: event propagation across graph, sensors detect/report
- [ ] Reward: detection accuracy, penalize false positives
- [ ] Demonstrates `HorizontalProtocol` for sensor gossip
- [ ] Register as `heron.make("SensorNetwork-v0")`

### 2C. Transport Fleet Env — Hierarchy + Event-Driven

> Teaches: coordinators, vertical protocol, custom events, dual-mode execution.
> **Deps**: 0A (termination), 0B (coordinator reward), 0C (custom events).

**New dir**: `heron/demo_envs/transport_fleet/`

- [ ] Depot coordinator → vehicle field agents
- [ ] Custom events: `new_delivery_request`, `vehicle_breakdown`
- [ ] Coordinator decomposes delivery assignments (vertical protocol)
- [ ] Coordinator reward = total deliveries; vehicle reward = fuel efficiency
- [ ] Demonstrates event-driven mode with reactive agents
- [ ] Register as `heron.make("TransportFleet-v0")`

### 2D. Notebooks

**Dir**: `examples/notebooks/`

- [ ] `01_thermostat_quickstart.ipynb` — build, train, evaluate in < 50 cells
- [ ] `02_sensor_network_visibility.ipynb` — observation scoping deep-dive
- [ ] `03_transport_fleet_hierarchy.ipynb` — hierarchy + event-driven walkthrough
- [ ] `04_training_vs_event_driven.ipynb` — same env, both modes, compare

---

## Tier 3 — Polish & Adoption

Only start after Tiers 0–2 are green.

### 3A. Test Suite Expansion

- [ ] Unit tests for every Tier 0–2 feature (target: every public method)
- [ ] Integration tests: each demo env × {training, event-driven} × {RLlib, PettingZoo}
- [ ] Regression tests for the coordinator race condition fixed in 0B

### 3B. Documentation

- [ ] **`local_state` format doc** + `TypedDict` annotations on `State`, `FieldAgentState`, `CoordinatorAgentState`
- [ ] **Decision tree**: "DefaultHeronEnv vs BaseEnv vs EnvBuilder" — flowchart in README
- [ ] **"Why agents own logic"** explanation — tie to dual-mode benefit
- [ ] **"When You Need Hierarchy"** guide — flat → vertical → horizontal progression
- [ ] **State bridge validation** — runtime type/key checks on `env_state_to_global_state()` output
- [ ] **"Coming from PettingZoo?"** — concept mapping table (PettingZoo → Heron)
- [ ] Main repo docstrings — every public class/method in `heron/`
- [ ] Design principles page

### 3C. Developer Guides

- [ ] **Iterative Feature Guide** — progressive disclosure: field agent → coordinator → event-driven
- [ ] **"Build Your First Benchmark" walkthrough** — 30 min from domain idea to training
- [ ] Simple case study walkthrough (use Thermostat or Transport Fleet, not power grid)
- [ ] Updated README with all guides linked

### 3D. Website

- [ ] Set up MkDocs Material with `mkdocs.yml`
- [ ] Auto-generate API reference from docstrings
- [ ] Host guides, tutorials, and decision trees
- [ ] Deploy to GitHub Pages

---

## Scope Summary

| Phase | What | Estimated Scope |
|---|---|---|
| **0A** | Termination/truncation | ~3 files, ~5 tests |
| **0B** | Coordinator reward fix | ~2 files, ~3 tests |
| **0C** | Custom event triggers | ~4 files, ~4 tests |
| **1A** | PettingZoo adaptor | ~1 new file, ~2 tests |
| **1B** | `heron.make()` | ~1 new file, ~1 test |
| **2A–2C** | 3 demo envs | ~3 new dirs, ~6 tests |
| **2D** | 4 notebooks | ~4 `.ipynb` files |
| **3A** | Test expansion | ~10+ test files |
| **3B–3D** | Docs + website | Markdown + MkDocs |

---

## Future Work (Out of Scope)

These are deliberately deferred — no downstream dependency in the plan above.

- **More robust reward mechanisms** — design once users hit real limitations
- **Experiment runner CLI** — `python -m heron.run --env=X --algo=ippo --seed=42`
- **Observation stacking helper** — utility, not framework core
- **Logging / metrics integration** — important eventually, addable without breaking changes
- **Message Broker extension** — current `InMemoryBroker` is sufficient; defer until concrete use case
- **Complex vector observation** — clarify the gap before designing

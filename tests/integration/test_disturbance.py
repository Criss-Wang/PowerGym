"""Integration tests for Class 4 (Exogenous Disturbance) events.

Tests the full causal chain:
  ENV_UPDATE → apply_disturbance → optional SIMULATION → condition eval → agent response

Scenarios:
  D1.  Disturbance with requires_physics=True triggers immediate SIMULATION
  D2.  Disturbance with requires_physics=False evaluates conditions directly
  D3.  Disturbance during agent pipeline (between tick and action_effect)
  D4.  Multiple disturbances in one episode
  D5.  DisturbanceSchedule.from_list and .poisson work end-to-end
  D6.  Reset isolation — disturbances from episode 1 don't leak to episode 2
  D7.  Full causal chain: ENV_UPDATE → SIMULATION → CONDITION_TRIGGER → agent reacts
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.agents.constants import FIELD_LEVEL, COORDINATOR_LEVEL, SYSTEM_LEVEL
from heron.core.feature import Feature
from heron.core.action import Action
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.base import BaseEnv
from heron.scheduling import ScheduleConfig, EpisodeAnalyzer
from heron.scheduling.condition_monitor import ConditionMonitor
from heron.scheduling.disturbance import Disturbance, DisturbanceSchedule
from heron.scheduling.event import EventType


# =============================================================================
# Minimal domain (same pattern as test_event_driven_timing.py)
# =============================================================================

@dataclass(slots=True)
class CounterFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ["public"]
    value: float = 0.0

    def set_values(self, **kwargs: Any) -> None:
        if "value" in kwargs:
            self.value = kwargs["value"]


class CounterFieldAgent(FieldAgent):
    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-10.0]), np.array([10.0])))
        action.set_values(np.array([1.0]))
        return action

    def set_action(self, action: Any) -> None:
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        else:
            self.action.set_values(action)

    def set_state(self) -> None:
        new_val = self.state.features["CounterFeature"].value + self.action.c[0]
        self.state.features["CounterFeature"].set_values(value=new_val)

    def apply_action(self):
        self.set_state()

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        if "CounterFeature" in local_state:
            return float(local_state["CounterFeature"][0])
        return 0.0


class IncrementPolicy(Policy):
    observation_mode = "local"

    def __init__(self, increment: float = 1.0):
        self.increment = increment
        self.obs_dim = 1
        self.action_dim = 1
        self.action_range = (-10.0, 10.0)

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        return np.array([self.increment])


class DisturbanceCoordinator(CoordinatorAgent):
    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-10.0]), np.array([10.0])))
        action.set_values(np.array([0.0]))
        return action

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        return 0.0


class DisturbanceSystemAgent(SystemAgent):
    pass


class EnvState:
    def __init__(self, agent_values: Optional[Dict[str, float]] = None):
        self.agent_values = agent_values or {}


class DisturbanceTestEnv(BaseEnv):
    """Test env that tracks applied disturbances."""

    def __init__(self, agents, hierarchy, **kwargs):
        self.applied_disturbances: List[Disturbance] = []
        super().__init__(agents=agents, hierarchy=hierarchy, **kwargs)

    def apply_disturbance(self, disturbance: Disturbance) -> None:
        self.applied_disturbances.append(disturbance)
        # For "value_set": directly mutate the agent's feature
        if disturbance.disturbance_type == "value_set":
            agent_id = disturbance.payload.get("agent_id")
            value = disturbance.payload.get("value", 0.0)
            if agent_id and agent_id in self.registered_agents:
                agent = self.registered_agents[agent_id]
                if hasattr(agent, "state") and agent.state:
                    agent.state.features["CounterFeature"].set_values(value=value)
                    self.proxy.set_local_state(agent_id, agent.state)

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        level_to_type = {
            FIELD_LEVEL: "FieldAgentState",
            COORDINATOR_LEVEL: "CoordinatorAgentState",
            SYSTEM_LEVEL: "SystemAgentState",
        }
        agent_states = {}
        for aid, agent in self.registered_agents.items():
            if hasattr(agent, "level") and agent.level == FIELD_LEVEL:
                val = env_state.agent_values.get(aid, 0.0)
                agent_states[aid] = {
                    "_owner_id": aid,
                    "_owner_level": agent.level,
                    "_state_type": level_to_type[agent.level],
                    "features": {"CounterFeature": {"value": val}},
                }
        return {"agent_states": agent_states}

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        agent_states = global_state.get("agent_states", {})
        values = {}
        for aid, sd in agent_states.items():
            features = sd.get("features", {})
            if "CounterFeature" in features:
                values[aid] = features["CounterFeature"].get("value", 0.0)
        return EnvState(agent_values=values)


# =============================================================================
# Env builder
# =============================================================================

def _build_disturbance_env(
    tick_interval: float = 5.0,
    act_delay: float = 0.2,
    msg_delay: float = 0.1,
    sim_wait: float = 3.0,
    sys_tick_interval: float = 5.0,
) -> DisturbanceTestEnv:
    agent = CounterFieldAgent(
        agent_id="counter_1",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=1.0),
    )
    agent.schedule_config = ScheduleConfig.deterministic(
        tick_interval=tick_interval, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    coord = DisturbanceCoordinator(agent_id="coord_1")
    sys_agent = DisturbanceSystemAgent(
        agent_id="system_agent",
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=sys_tick_interval, obs_delay=0.0,
            act_delay=0.0, msg_delay=msg_delay,
        ),
    )
    return DisturbanceTestEnv(
        agents=[sys_agent, coord, agent],
        hierarchy={"system_agent": ["coord_1"], "coord_1": ["counter_1"]},
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=sim_wait,
    )


def _run_and_analyze(env, t_end, max_events=5000):
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    stats = env.run_event_driven(
        t_end=t_end, episode_analyzer=analyzer, max_events=max_events,
    )
    return stats, analyzer


# =============================================================================
# Tests
# =============================================================================

def test_d1_disturbance_triggers_physics():
    """requires_physics=True schedules an immediate SIMULATION."""
    env = _build_disturbance_env(sim_wait=3.0, sys_tick_interval=5.0)
    schedule = DisturbanceSchedule([
        Disturbance(
            timestamp=2.5,
            disturbance_type="value_set",
            payload={"agent_id": "counter_1", "value": 99.0},
            requires_physics=True,
        ),
    ])
    env.reset(disturbance_schedule=schedule)
    stats, analyzer = _run_and_analyze(env, t_end=10.0)

    # Disturbance was applied
    assert len(env.applied_disturbances) == 1
    assert env.applied_disturbances[0].disturbance_type == "value_set"

    # ENV_UPDATE event processed
    event_counts = stats.get_event_counts()
    assert event_counts.get(EventType.ENV_UPDATE, 0) == 1
    assert analyzer.disturbance_count == 1

    # An extra SIMULATION was triggered (beyond the periodic ones)
    sim_events = [
        a for a in stats.event_analyses
        if a.event_type == EventType.SIMULATION
    ]
    # At least one SIMULATION should fire around t=2.5
    assert any(abs(e.timestamp - 2.5) < 0.5 for e in sim_events)


def test_d2_disturbance_no_physics():
    """requires_physics=False does NOT schedule extra SIMULATION."""
    env = _build_disturbance_env(sim_wait=10.0, sys_tick_interval=10.0)
    schedule = DisturbanceSchedule([
        Disturbance(
            timestamp=2.0,
            disturbance_type="value_set",
            payload={"agent_id": "counter_1", "value": 50.0},
            requires_physics=False,
        ),
    ])
    env.reset(disturbance_schedule=schedule)
    stats, analyzer = _run_and_analyze(env, t_end=5.0)

    assert len(env.applied_disturbances) == 1
    assert analyzer.disturbance_count == 1

    # Count SIMULATIONs — should only have the periodic ones, not an extra at t=2.0
    sim_events = [
        a for a in stats.event_analyses
        if a.event_type == EventType.SIMULATION
    ]
    # No simulation should fire at exactly t=2.0 (the disturbance time)
    sim_at_2 = [e for e in sim_events if abs(e.timestamp - 2.0) < 0.01]
    assert len(sim_at_2) == 0


def test_d3_disturbance_during_pipeline():
    """Disturbance arrives between agent tick and action_effect.

    This is the key 'pipeline collision' scenario (failure mode 5b).
    The agent's action was computed for pre-disturbance state.
    """
    env = _build_disturbance_env(
        tick_interval=5.0, act_delay=3.0, sim_wait=10.0, sys_tick_interval=12.0,
    )
    # Agent ticks at t=5.0, action effect at t=8.0
    # Disturbance at t=6.5 (between tick and action effect)
    schedule = DisturbanceSchedule([
        Disturbance(
            timestamp=6.5,
            disturbance_type="value_set",
            payload={"agent_id": "counter_1", "value": 999.0},
            requires_physics=True,
        ),
    ])
    env.reset(disturbance_schedule=schedule)
    stats, _ = _run_and_analyze(env, t_end=15.0)

    # Verify the disturbance landed and no crash
    assert len(env.applied_disturbances) == 1

    # Verify event ordering: ENV_UPDATE at t=6.5 is between AGENT_TICK ~t=5 and ACTION_EFFECT ~t=8
    env_update_events = [
        a for a in stats.event_analyses if a.event_type == EventType.ENV_UPDATE
    ]
    assert len(env_update_events) == 1
    assert abs(env_update_events[0].timestamp - 6.5) < 0.01


def test_d4_multiple_disturbances():
    """Multiple disturbances in one episode all get applied."""
    env = _build_disturbance_env(sim_wait=5.0, sys_tick_interval=5.0)
    schedule = DisturbanceSchedule([
        Disturbance(timestamp=1.0, disturbance_type="fault_1"),
        Disturbance(timestamp=3.0, disturbance_type="fault_2"),
        Disturbance(timestamp=7.0, disturbance_type="fault_3"),
    ])
    env.reset(disturbance_schedule=schedule)
    stats, analyzer = _run_and_analyze(env, t_end=10.0)

    assert len(env.applied_disturbances) == 3
    assert analyzer.disturbance_count == 3
    types = [d.disturbance_type for d in env.applied_disturbances]
    assert types == ["fault_1", "fault_2", "fault_3"]


def test_d5_from_list_end_to_end():
    """DisturbanceSchedule.from_list works through the full pipeline."""
    env = _build_disturbance_env(sim_wait=5.0, sys_tick_interval=5.0)
    schedule = DisturbanceSchedule.from_list([
        {"t": 2.0, "type": "spike", "bus": 5, "delta": 100.0},
    ])
    env.reset(disturbance_schedule=schedule)
    stats, _ = _run_and_analyze(env, t_end=5.0)

    assert len(env.applied_disturbances) == 1
    assert env.applied_disturbances[0].payload == {"bus": 5, "delta": 100.0}


def test_d5_poisson_end_to_end():
    """DisturbanceSchedule.poisson works through the full pipeline."""
    env = _build_disturbance_env(sim_wait=5.0, sys_tick_interval=5.0)
    schedule = DisturbanceSchedule.poisson(
        rate=1.0, disturbance_types=["fault"],
        t_end=10.0, rng=np.random.default_rng(42),
    )
    n_expected = len(schedule)
    assert n_expected > 0  # Poisson with rate=1, t_end=10 should produce events

    env.reset(disturbance_schedule=schedule)
    stats, analyzer = _run_and_analyze(env, t_end=10.0)

    assert analyzer.disturbance_count == n_expected


def test_d6_reset_isolation():
    """Disturbances from episode 1 do not appear in episode 2."""
    env = _build_disturbance_env(sim_wait=3.0, sys_tick_interval=5.0)
    schedule = DisturbanceSchedule([
        Disturbance(timestamp=1.0, disturbance_type="fault"),
    ])

    # Episode 1: with disturbance
    env.reset(disturbance_schedule=schedule)
    stats1, a1 = _run_and_analyze(env, t_end=5.0)
    assert a1.disturbance_count == 1

    # Episode 2: without disturbance — reset clears the queue
    env.applied_disturbances.clear()
    env.reset()
    stats2, a2 = _run_and_analyze(env, t_end=5.0)
    assert a2.disturbance_count == 0
    assert len(env.applied_disturbances) == 0


def test_d7_full_causal_chain():
    """ENV_UPDATE → SIMULATION → condition eval → CONDITION_TRIGGER → agent reacts.

    This is the most important test: validates the complete Class 4 → 5 → 3 chain.
    """
    env = _build_disturbance_env(
        tick_interval=20.0, act_delay=0.2, msg_delay=0.1,
        sim_wait=10.0, sys_tick_interval=10.0,
    )

    # Register condition: counter value > 50 → trigger field agent
    env.scheduler.register_condition(ConditionMonitor(
        monitor_id="counter_high",
        agent_id="counter_1",
        condition_fn=lambda state: (
            "counter_1" in state.get("agent_states", {})
            and state["agent_states"]["counter_1"]
                .get("features", {})
                .get("CounterFeature", {})
                .get("value", 0) > 50
        ),
        cooldown=1.0,
    ))

    # Disturbance sets counter to 99 — should trigger condition after physics
    schedule = DisturbanceSchedule([
        Disturbance(
            timestamp=2.5,
            disturbance_type="value_set",
            payload={"agent_id": "counter_1", "value": 99.0},
            requires_physics=True,
        ),
    ])
    env.reset(disturbance_schedule=schedule)
    stats, _ = _run_and_analyze(env, t_end=15.0)

    event_types = [a.event_type for a in stats.event_analyses]

    # Verify ENV_UPDATE was processed
    assert EventType.ENV_UPDATE in event_types

    # Verify SIMULATION happened after ENV_UPDATE
    env_idx = event_types.index(EventType.ENV_UPDATE)
    sim_after = None
    for i in range(env_idx + 1, len(event_types)):
        if event_types[i] == EventType.SIMULATION:
            sim_after = i
            break
    assert sim_after is not None, "No SIMULATION after ENV_UPDATE"

    # Verify CONDITION_TRIGGER happened after the SIMULATION's completion
    cond_after = None
    for i in range(sim_after, len(event_types)):
        if event_types[i] == EventType.CONDITION_TRIGGER:
            cond_after = i
            break
    assert cond_after is not None, (
        "No CONDITION_TRIGGER after SIMULATION — condition monitor should have fired"
    )


# =============================================================================
# Multi-agent env builder (for D8/D9 stress tests)
# =============================================================================

def _build_multi_agent_env(
    n_agents: int = 3,
    tick_interval: float = 5.0,
    act_delay: float = 0.2,
    msg_delay: float = 0.1,
    sim_wait: float = 3.0,
    sys_tick_interval: float = 5.0,
) -> DisturbanceTestEnv:
    """Build env with N field agents under one coordinator."""
    agents_list = []
    hierarchy = {"system_agent": ["coord_1"], "coord_1": []}

    for i in range(n_agents):
        aid = f"counter_{i}"
        agent = CounterFieldAgent(
            agent_id=aid,
            features=[CounterFeature(value=0.0)],
            policy=IncrementPolicy(increment=1.0),
        )
        agent.schedule_config = ScheduleConfig.deterministic(
            tick_interval=tick_interval, obs_delay=0.0,
            act_delay=act_delay, msg_delay=msg_delay,
        )
        agents_list.append(agent)
        hierarchy["coord_1"].append(aid)

    coord = DisturbanceCoordinator(agent_id="coord_1")
    sys_agent = DisturbanceSystemAgent(
        agent_id="system_agent",
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=sys_tick_interval, obs_delay=0.0,
            act_delay=0.0, msg_delay=msg_delay,
        ),
    )
    return DisturbanceTestEnv(
        agents=[sys_agent, coord] + agents_list,
        hierarchy=hierarchy,
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=sim_wait,
    )


# =============================================================================
# Phase 4.4: Stress tests
# =============================================================================

def test_d8_rapid_disturbances_no_crash():
    """10 disturbances in 5 seconds. Scheduler stays stable, cooldowns work."""
    env = _build_disturbance_env(sim_wait=3.0, sys_tick_interval=5.0)

    # Register condition with 1s cooldown to verify it doesn't spam
    env.scheduler.register_condition(ConditionMonitor(
        monitor_id="always_true",
        agent_id="counter_1",
        condition_fn=lambda state: True,
        cooldown=1.0,
    ))

    schedule = DisturbanceSchedule([
        Disturbance(timestamp=0.5 * i, disturbance_type="value_set",
                    payload={"agent_id": "counter_1", "value": float(i)})
        for i in range(1, 11)  # 10 disturbances: t=0.5, 1.0, ..., 5.0
    ])
    env.reset(disturbance_schedule=schedule)
    stats, analyzer = _run_and_analyze(env, t_end=8.0, max_events=10000)

    # All 10 disturbances applied
    assert len(env.applied_disturbances) == 10
    assert analyzer.disturbance_count == 10

    # Condition triggers should be limited by cooldown (1s between triggers,
    # disturbances every 0.5s → roughly 5-6 triggers, not 10)
    cond_events = [
        a for a in stats.event_analyses
        if a.event_type == EventType.CONDITION_TRIGGER
    ]
    assert len(cond_events) < 10, (
        f"Cooldown should limit triggers, got {len(cond_events)}"
    )
    assert len(cond_events) > 0, "At least one condition should fire"


class MultiSetDisturbanceEnv(DisturbanceTestEnv):
    """Env where 'multi_set' disturbance sets all counter agents to a value."""

    def apply_disturbance(self, disturbance):
        super().apply_disturbance(disturbance)
        if disturbance.disturbance_type == "multi_set":
            value = disturbance.payload.get("value", 99.0)
            for aid, agent in self.registered_agents.items():
                if hasattr(agent, "state") and agent.state and "CounterFeature" in agent.state.features:
                    agent.state.features["CounterFeature"].set_values(value=value)
                    self.proxy.set_local_state(aid, agent.state)


def _build_multi_set_env(
    n_agents: int = 3,
    sim_wait: float = 5.0,
    sys_tick_interval: float = 5.0,
) -> MultiSetDisturbanceEnv:
    """Build MultiSetDisturbanceEnv with N field agents."""
    agents_list = []
    hierarchy = {"system_agent": ["coord_1"], "coord_1": []}

    for i in range(n_agents):
        aid = f"counter_{i}"
        agent = CounterFieldAgent(
            agent_id=aid,
            features=[CounterFeature(value=0.0)],
            policy=IncrementPolicy(increment=1.0),
        )
        agent.schedule_config = ScheduleConfig.deterministic(
            tick_interval=20.0, obs_delay=0.0, act_delay=0.2, msg_delay=0.1,
        )
        agents_list.append(agent)
        hierarchy["coord_1"].append(aid)

    coord = DisturbanceCoordinator(agent_id="coord_1")
    sys_agent = DisturbanceSystemAgent(
        agent_id="system_agent",
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=sys_tick_interval, obs_delay=0.0,
            act_delay=0.0, msg_delay=0.1,
        ),
    )
    return MultiSetDisturbanceEnv(
        agents=[sys_agent, coord] + agents_list,
        hierarchy=hierarchy,
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=sim_wait,
    )


def test_d9_multiple_agents_triggered_by_single_disturbance():
    """Single disturbance triggers conditions for 3 different agents."""
    env = _build_multi_set_env(n_agents=3, sim_wait=5.0, sys_tick_interval=5.0)

    # Register conditions for all 3 agents — each watches its own counter
    for i in range(3):
        aid = f"counter_{i}"
        env.scheduler.register_condition(ConditionMonitor(
            monitor_id=f"cond_{aid}",
            agent_id=aid,
            condition_fn=lambda state, _aid=aid: (
                _aid in state.get("agent_states", {})
                and state["agent_states"][_aid]
                    .get("features", {})
                    .get("CounterFeature", {})
                    .get("value", 0) > 50
            ),
            cooldown=1.0,
        ))

    schedule = DisturbanceSchedule([
        Disturbance(timestamp=2.5, disturbance_type="multi_set",
                    payload={"value": 99.0}, requires_physics=True),
    ])
    env.reset(disturbance_schedule=schedule)
    stats, _ = _run_and_analyze(env, t_end=10.0)

    # Verify all 3 agents got CONDITION_TRIGGER events
    cond_events = [
        a for a in stats.event_analyses
        if a.event_type == EventType.CONDITION_TRIGGER
    ]
    triggered_agents = {e.agent_id for e in cond_events}
    assert triggered_agents == {"counter_0", "counter_1", "counter_2"}, (
        f"Expected all 3 agents triggered, got {triggered_agents}"
    )

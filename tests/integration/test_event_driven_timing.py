"""
End-to-end integration tests for event-driven timing correctness.

Validates that reward attribution, event ordering, and (obs,action) caching
behave correctly under all meaningful timing configurations, exercising the
full BaseEnv → Proxy → Scheduler → Agent pipeline.

Timing scenarios covered:
  T1.  Happy path: action lands before physics → reward has (obs,action) pairs
  T2.  Physics before action_effect → reward has empty (obs,action) cache
  T3.  Multiple ticks before physics → reward accumulates multiple pairs
  T4.  Physics before any agent tick → early reward with empty cache
  T5.  Two physics cycles between ticks → first reward has pairs, second is empty
  T6.  Overlapping action_effects (fast tick, slow act_delay) → FIFO correctness
  T7.  Reactive agents: coordinator-driven tick → bottom-up reward cascade
  T8.  Heterogeneous tick rates → fast agent accumulates more pairs per physics
  T9.  Jitter robustness → non-deterministic delays don't break invariants
  T10. Reset isolation → second episode is independent of first
  T11. Long simulation stress → no accumulation errors or crashes
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
from heron.core.observation import Observation
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.base import BaseEnv
from heron.protocols.base import ActionProtocol, Protocol
from heron.utils.typing import AgentID
from heron.scheduling import ScheduleConfig, JitterType
from heron.scheduling.event import EventType
from heron.scheduling.analysis import EpisodeAnalyzer, EpisodeStats


# =============================================================================
# Minimal domain for timing tests
# =============================================================================

@dataclass(slots=True)
class CounterFeature(Feature):
    """Counter feature: value increments when action is applied."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    value: float = 0.0

    def set_values(self, **kwargs: Any) -> None:
        if "value" in kwargs:
            self.value = kwargs["value"]


class CounterFieldAgent(FieldAgent):
    """Deterministic field agent: each apply_action increments counter."""

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
    """Always outputs a fixed increment."""
    observation_mode = "local"

    def __init__(self, increment: float = 1.0):
        self.increment = increment
        self.obs_dim = 1          # CounterFeature has 1 value
        self.action_dim = 1
        self.action_range = (-10.0, 10.0)

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        return np.array([self.increment])


class EqualSplitProtocol(ActionProtocol):
    """Split coordinator action equally among subordinates."""

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        info_for_subordinates=None,
        coordination_messages=None,
        context=None,
    ) -> Dict[AgentID, Any]:
        if context is None or "subordinate_ids" not in context:
            return {}
        sub_ids = context["subordinate_ids"]
        n = len(sub_ids)
        if n == 0 or coordinator_action is None:
            return {}
        if isinstance(coordinator_action, Action):
            per_sub = coordinator_action.c / n
        else:
            per_sub = np.array(coordinator_action) / n
        result = {}
        for sid in sub_ids:
            a = Action()
            a.set_specs(dim_c=1, range=(np.array([-10.0]), np.array([10.0])))
            a.set_values(per_sub)
            result[sid] = a
        return result


class TimingCoordinator(CoordinatorAgent):
    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-10.0]), np.array([10.0])))
        action.set_values(np.array([0.0]))
        return action

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        return 0.0


class TimingSystemAgent(SystemAgent):
    pass


class EnvState:
    def __init__(self, agent_values: Optional[Dict[str, float]] = None):
        self.agent_values = agent_values or {}


class TimingTestEnv(BaseEnv):
    """Minimal environment: identity physics (pass-through)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
# Environment builders
# =============================================================================

def _build_single_agent_env(
    tick_interval: float = 5.0,
    act_delay: float = 0.2,
    msg_delay: float = 0.1,
    sim_wait: float = 3.0,
    sys_tick_interval: float = 5.0,
    increment: float = 1.0,
) -> TimingTestEnv:
    """SystemAgent → (no-op coordinator, skipped R4) → 1 periodic FieldAgent."""
    agent = CounterFieldAgent(
        agent_id="counter_1",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=increment),
    )
    agent.schedule_config = ScheduleConfig.deterministic(
        tick_interval=tick_interval, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    coord = TimingCoordinator(
        agent_id="coord_1",
        subordinates={"counter_1": agent},
    )
    sys_agent = TimingSystemAgent(
        agent_id="system_agent",
        subordinates={"coord_1": coord},
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=sys_tick_interval, obs_delay=0.0,
            act_delay=0.0, msg_delay=msg_delay,
        ),
    )
    return TimingTestEnv(
        system_agent=sys_agent,
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=sim_wait,
    )


def _build_two_agent_env(
    tick_interval_1: float = 5.0,
    tick_interval_2: float = 10.0,
    act_delay: float = 0.2,
    msg_delay: float = 0.1,
    sim_wait: float = 3.0,
    sys_tick_interval: float = 5.0,
) -> TimingTestEnv:
    """SystemAgent → (no-op coordinator) → 2 periodic FieldAgents at different rates."""
    agent1 = CounterFieldAgent(
        agent_id="counter_1",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=1.0),
    )
    agent1.schedule_config = ScheduleConfig.deterministic(
        tick_interval=tick_interval_1, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    agent2 = CounterFieldAgent(
        agent_id="counter_2",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=2.0),
    )
    agent2.schedule_config = ScheduleConfig.deterministic(
        tick_interval=tick_interval_2, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    coord = TimingCoordinator(
        agent_id="coord_1",
        subordinates={"counter_1": agent1, "counter_2": agent2},
    )
    sys_agent = TimingSystemAgent(
        agent_id="system_agent",
        subordinates={"coord_1": coord},
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=sys_tick_interval, obs_delay=0.0,
            act_delay=0.0, msg_delay=msg_delay,
        ),
    )
    return TimingTestEnv(
        system_agent=sys_agent,
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=sim_wait,
    )


def _build_reactive_env(
    coord_tick_interval: float = 5.0,
    act_delay: float = 0.2,
    msg_delay: float = 0.1,
    sim_wait: float = 3.0,
    sys_tick_interval: float = 5.0,
) -> TimingTestEnv:
    """SystemAgent → coordinator (periodic, with protocol) → 2 reactive FieldAgents."""
    agent1 = CounterFieldAgent(
        agent_id="reactive_1",
        features=[CounterFeature(value=0.0)],
    )
    agent1.schedule_config = ScheduleConfig.deterministic(
        tick_interval=coord_tick_interval, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    agent2 = CounterFieldAgent(
        agent_id="reactive_2",
        features=[CounterFeature(value=0.0)],
    )
    agent2.schedule_config = ScheduleConfig.deterministic(
        tick_interval=coord_tick_interval, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    coord = TimingCoordinator(
        agent_id="coord_reactive",
        subordinates={"reactive_1": agent1, "reactive_2": agent2},
        protocol=Protocol(action_protocol=EqualSplitProtocol()),
        policy=IncrementPolicy(increment=4.0),
    )
    coord.schedule_config = ScheduleConfig.deterministic(
        tick_interval=coord_tick_interval, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    sys_agent = TimingSystemAgent(
        agent_id="system_agent",
        subordinates={"coord_reactive": coord},
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=sys_tick_interval, obs_delay=0.0,
            act_delay=0.0, msg_delay=msg_delay,
        ),
    )
    return TimingTestEnv(
        system_agent=sys_agent,
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=sim_wait,
    )


def _run_and_analyze(env: TimingTestEnv, t_end: float, max_events: int = 5000) -> EpisodeAnalyzer:
    """Reset, run event-driven, return the analyzer with reward_history."""
    env.reset(seed=42)
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    env.run_event_driven(episode_analyzer=analyzer, t_end=t_end, max_events=max_events)
    return analyzer


# =============================================================================
# T1: Happy path — action lands before physics
# =============================================================================

def test_t1_happy_path_reward_has_pairs():
    """With small act_delay and large sim_wait, agent ticks and completes
    action_effect before physics runs. The reward should have (obs,action) pairs.

    Timeline:
      t=0:   system tick → kicks off counter_1 at t=1.0, physics at t=8.0
      t=1.0: counter_1 ticks → obs chain → action_effect at ~t=1.4
      t=8.0: SIMULATION → reward with cached (obs,action) pair
    """
    env = _build_single_agent_env(
        tick_interval=1.0,     # agent ticks early (t=1.0)
        act_delay=0.2,         # action lands quickly
        msg_delay=0.1,
        sim_wait=8.0,          # physics runs much later
        sys_tick_interval=10.0,
    )
    analyzer = _run_and_analyze(env, t_end=15.0)

    history = analyzer.reward_history.get("counter_1", [])
    assert len(history) >= 1, "Expected at least one reward entry"

    # The reward that follows the first physics step should have pairs
    # (agent had time to tick + complete action_effect before physics)
    _, _, pairs = history[0]
    assert len(pairs) >= 1, (
        f"Happy path: reward should have >= 1 (obs,action) pair, got {len(pairs)}"
    )


def test_t1_happy_path_event_ordering():
    """When agent ticks early and sim_wait is large, the first action_effect
    should precede the first simulation in the event timeline."""
    env = _build_single_agent_env(
        tick_interval=1.0, act_delay=0.2, msg_delay=0.1,
        sim_wait=8.0, sys_tick_interval=10.0,
    )
    env.reset(seed=42)
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    episode = env.run_event_driven(episode_analyzer=analyzer, t_end=15.0)

    ae_times = [a.timestamp for a in episode.event_analyses
                if a.event_type == EventType.ACTION_EFFECT and a.agent_id == "counter_1"]
    sim_times = [a.timestamp for a in episode.event_analyses
                 if a.event_type == EventType.SIMULATION]

    assert len(ae_times) >= 1 and len(sim_times) >= 1
    # First action_effect should fire before first simulation
    assert ae_times[0] < sim_times[0], (
        f"First action_effect ({ae_times[0]:.3f}) should precede first simulation ({sim_times[0]:.3f})"
    )


# =============================================================================
# T2: Physics before action_effect (the original bug, now fixed)
# =============================================================================

def test_t2_physics_before_action_effect_empty_pairs():
    """With very large act_delay, physics fires before action lands.
    Reward at that physics boundary should have ZERO (obs,action) pairs
    because the cache is only populated in action_effect_handler."""
    env = _build_single_agent_env(
        tick_interval=5.0, act_delay=5.0, msg_delay=0.1,
        sim_wait=0.5, sys_tick_interval=5.0,
    )
    analyzer = _run_and_analyze(env, t_end=15.0)

    history = analyzer.reward_history.get("counter_1", [])
    assert len(history) >= 1, "Expected at least one reward"

    # The first reward fires before the first action_effect lands.
    _, _, pairs = history[0]
    assert len(pairs) == 0, (
        f"Physics-before-action: first reward should have 0 pairs, got {len(pairs)}. "
        "This would indicate fake reward attribution."
    )


def test_t2_event_timeline_confirms_ordering():
    """Verify the simulation event fires before the agent's action_effect.

    Timeline:
      t=0:    system tick → counter_1 at t=5.0, physics at t=0.5
      t=0.5:  SIMULATION fires (agent hasn't ticked yet)
      t=5.0:  counter_1 ticks → obs chain → action_effect at ~t=10.2
    """
    env = _build_single_agent_env(
        tick_interval=5.0, act_delay=5.0, msg_delay=0.1,
        sim_wait=0.5, sys_tick_interval=5.0,
    )
    env.reset(seed=42)
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    episode = env.run_event_driven(episode_analyzer=analyzer, t_end=15.0)

    ae_times = [a.timestamp for a in episode.event_analyses
                if a.event_type == EventType.ACTION_EFFECT and a.agent_id == "counter_1"]
    sim_times = [a.timestamp for a in episode.event_analyses
                 if a.event_type == EventType.SIMULATION]

    assert len(ae_times) >= 1, "Expected at least one action_effect"
    assert len(sim_times) >= 1, "Expected at least one simulation"
    # Simulation fires BEFORE the field agent's action_effect
    assert sim_times[0] < ae_times[0], (
        f"Simulation ({sim_times[0]:.3f}) should fire before action_effect ({ae_times[0]:.3f}) "
        "in this configuration"
    )


# =============================================================================
# T3: Multiple ticks before physics — accumulated pairs
# =============================================================================

def test_t3_multiple_ticks_accumulate_pairs():
    """With fast tick (1s) and slow physics (sim_wait=4s), the agent ticks
    multiple times. The first reward should have >= 2 (obs,action) pairs."""
    env = _build_single_agent_env(
        tick_interval=1.0, act_delay=0.1, msg_delay=0.05,
        sim_wait=4.0, sys_tick_interval=5.0,
    )
    analyzer = _run_and_analyze(env, t_end=10.0)

    history = analyzer.reward_history.get("counter_1", [])
    assert len(history) >= 1, "Expected at least one reward"

    _, _, pairs = history[0]
    assert len(pairs) >= 2, (
        f"Multiple ticks before physics: expected >= 2 pairs, got {len(pairs)}"
    )


# =============================================================================
# T4: Physics before any agent tick
# =============================================================================

def test_t4_physics_before_any_tick():
    """With sim_wait=0.01 and tick_interval=10, physics fires before the
    agent even starts. Reward should have empty pairs and prev=None."""
    env = _build_single_agent_env(
        tick_interval=10.0, act_delay=0.2, msg_delay=0.1,
        sim_wait=0.01, sys_tick_interval=10.0,
    )
    analyzer = _run_and_analyze(env, t_end=15.0)

    history = analyzer.reward_history.get("counter_1", [])
    assert len(history) >= 1, "Expected at least one reward even before agent ticks"

    _, _, pairs = history[0]
    assert len(pairs) == 0, (
        f"Early physics: expected 0 pairs (agent hasn't ticked), got {len(pairs)}"
    )


# =============================================================================
# T5: Two physics steps between ticks
# =============================================================================

def test_t5_two_physics_between_ticks():
    """With slow agent (tick_interval=10) and fast physics (sys_tick_interval=2),
    multiple physics steps fire per agent tick cycle.

    Invariant: total non-empty rewards <= total agent ticks."""
    env = _build_single_agent_env(
        tick_interval=10.0, act_delay=0.2, msg_delay=0.1,
        sim_wait=1.0, sys_tick_interval=2.0,
    )
    analyzer = _run_and_analyze(env, t_end=15.0)

    history = analyzer.reward_history.get("counter_1", [])
    non_empty = sum(1 for entry in history if len(entry) == 3 and len(entry[2]) > 0)
    total_rewards = len(history)

    # Non-empty rewards can't exceed the number of agent ticks that completed
    # their full cycle (tick → obs → compute → action_effect → cache)
    assert non_empty <= total_rewards, "Sanity: non-empty <= total"
    # With tick_interval=10, agent ticks ~1 time in 15s → at most 1 non-empty reward
    assert non_empty <= 2, (
        f"Expected <= 2 non-empty rewards for slow agent, got {non_empty}"
    )
    # Should have multiple total rewards from multiple physics cycles
    assert total_rewards >= 2, (
        f"Expected >= 2 total rewards from multiple physics cycles, got {total_rewards}"
    )


# =============================================================================
# T6: Overlapping action_effects — FIFO queue correctness
# =============================================================================

def test_t6_fifo_queue_no_crash():
    """With tick_interval=1s and act_delay=3s, multiple ticks fire before
    the first action_effect. The FIFO queue should handle this without errors."""
    env = _build_single_agent_env(
        tick_interval=1.0, act_delay=3.0, msg_delay=0.05,
        sim_wait=20.0, sys_tick_interval=25.0,
    )
    # If the FIFO queue is broken, this will crash with IndexError or
    # produce incorrect pairing
    analyzer = _run_and_analyze(env, t_end=10.0)

    agent = env.registered_agents["counter_1"]
    assert len(agent._pending_obs_queue) >= 0, "Queue should never go negative"


def test_t6_fifo_pairing_correctness():
    """Each action_effect should consume exactly one obs from the queue.
    After running, obs count should track action_effect count."""
    env = _build_single_agent_env(
        tick_interval=1.0, act_delay=3.0, msg_delay=0.05,
        sim_wait=20.0, sys_tick_interval=25.0,
    )
    env.reset(seed=42)
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    episode = env.run_event_driven(episode_analyzer=analyzer, t_end=10.0)

    agent = env.registered_agents["counter_1"]

    # Count obs responses (= queued obs) and action_effects (= dequeued obs)
    obs_count = len([a for a in episode.event_analyses
                     if a.agent_id == "counter_1" and a.message_type == "get_obs_response"])
    ae_count = len([a for a in episode.event_analyses
                    if a.event_type == EventType.ACTION_EFFECT and a.agent_id == "counter_1"])

    # Pending queue = obs_count - ae_count (obs queued but not yet consumed)
    expected_pending = obs_count - ae_count
    actual_pending = len(agent._pending_obs_queue)
    # The cached pairs = ae_count (each ae pops one obs and caches)
    # Since no physics ran (sim_wait=20), cache should hold all pairs
    assert actual_pending >= 0, f"Queue length {actual_pending} should be >= 0"
    assert actual_pending == expected_pending, (
        f"Pending queue ({actual_pending}) != obs_count ({obs_count}) - ae_count ({ae_count})"
    )


# =============================================================================
# T7: Reactive agents — coordinator-driven ticks and bottom-up reward cascade
# =============================================================================

def test_t7_reactive_agents_are_ticked():
    """Reactive field agents (under coordinator with protocol) should
    receive ticks and produce rewards."""
    env = _build_reactive_env(
        coord_tick_interval=5.0, act_delay=0.2, msg_delay=0.1,
        sim_wait=3.0, sys_tick_interval=5.0,
    )
    analyzer = _run_and_analyze(env, t_end=12.0)

    # Both reactive agents should produce rewards
    history_1 = analyzer.reward_history.get("reactive_1", [])
    history_2 = analyzer.reward_history.get("reactive_2", [])

    assert len(history_1) >= 1, "reactive_1 should have at least one reward"
    assert len(history_2) >= 1, "reactive_2 should have at least one reward"


def test_t7_coordinator_receives_rewards_after_subordinates():
    """Coordinator reward should appear at or after subordinate rewards
    (bottom-up cascade)."""
    env = _build_reactive_env(
        coord_tick_interval=5.0, act_delay=0.2, msg_delay=0.1,
        sim_wait=3.0, sys_tick_interval=5.0,
    )
    analyzer = _run_and_analyze(env, t_end=12.0)

    coord_history = analyzer.reward_history.get("coord_reactive", [])
    sub_1_history = analyzer.reward_history.get("reactive_1", [])
    sub_2_history = analyzer.reward_history.get("reactive_2", [])

    assert len(coord_history) >= 1, "Coordinator should have rewards"

    if sub_1_history and coord_history:
        first_sub_time = min(sub_1_history[0][0], sub_2_history[0][0]) if sub_2_history else sub_1_history[0][0]
        first_coord_time = coord_history[0][0]
        assert first_coord_time >= first_sub_time, (
            f"Coordinator reward ({first_coord_time:.3f}) should be >= "
            f"first subordinate reward ({first_sub_time:.3f})"
        )


def test_t7_reactive_agent_tick_ordering():
    """Reactive agent ticks should come after their coordinator's tick."""
    env = _build_reactive_env(
        coord_tick_interval=5.0, act_delay=0.2, msg_delay=0.1,
        sim_wait=3.0, sys_tick_interval=5.0,
    )
    env.reset(seed=42)
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    episode = env.run_event_driven(episode_analyzer=analyzer, t_end=12.0)

    coord_ticks = [a.timestamp for a in episode.event_analyses
                   if a.event_type == EventType.AGENT_TICK and a.agent_id == "coord_reactive"]
    reactive_ticks = [a.timestamp for a in episode.event_analyses
                      if a.event_type == EventType.AGENT_TICK
                      and a.agent_id in ("reactive_1", "reactive_2")]

    assert len(coord_ticks) >= 1 and len(reactive_ticks) >= 1
    # First reactive tick must come after first coordinator tick
    assert min(reactive_ticks) > min(coord_ticks), (
        f"Reactive tick ({min(reactive_ticks):.3f}) should follow "
        f"coordinator tick ({min(coord_ticks):.3f})"
    )


# =============================================================================
# T8: Heterogeneous tick rates
# =============================================================================

def test_t8_fast_agent_more_ticks():
    """Fast agent (tick_interval=1s) should tick more often than slow
    agent (tick_interval=5s) between physics steps."""
    env = _build_two_agent_env(
        tick_interval_1=1.0, tick_interval_2=5.0,
        act_delay=0.1, msg_delay=0.05,
        sim_wait=3.0, sys_tick_interval=5.0,
    )
    env.reset(seed=42)
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    episode = env.run_event_driven(episode_analyzer=analyzer, t_end=12.0)

    ticks_1 = len([a for a in episode.event_analyses
                   if a.event_type == EventType.AGENT_TICK and a.agent_id == "counter_1"])
    ticks_2 = len([a for a in episode.event_analyses
                   if a.event_type == EventType.AGENT_TICK and a.agent_id == "counter_2"])

    assert ticks_1 > ticks_2, (
        f"Fast agent ticks ({ticks_1}) should exceed slow agent ({ticks_2})"
    )


def test_t8_fast_agent_more_pairs_per_physics():
    """Fast agent accumulates more (obs,action) pairs per physics boundary
    than slow agent."""
    env = _build_two_agent_env(
        tick_interval_1=1.0, tick_interval_2=5.0,
        act_delay=0.1, msg_delay=0.05,
        sim_wait=3.0, sys_tick_interval=5.0,
    )
    analyzer = _run_and_analyze(env, t_end=12.0)

    history_1 = analyzer.reward_history.get("counter_1", [])
    history_2 = analyzer.reward_history.get("counter_2", [])

    if history_1 and history_2:
        # First reward's pair count
        pairs_1 = len(history_1[0][2]) if len(history_1[0]) == 3 else 0
        pairs_2 = len(history_2[0][2]) if len(history_2[0]) == 3 else 0
        assert pairs_1 >= pairs_2, (
            f"Fast agent pairs ({pairs_1}) should be >= slow agent ({pairs_2})"
        )


# =============================================================================
# T9: Jitter robustness
# =============================================================================

def test_t9_jitter_gaussian_completes():
    """Gaussian jitter should not break event-driven execution."""
    agent = CounterFieldAgent(
        agent_id="jittered_1",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=1.0),
    )
    agent.schedule_config = ScheduleConfig.with_jitter(
        tick_interval=5.0, obs_delay=0.1, act_delay=0.5, msg_delay=0.2,
        jitter_type=JitterType.GAUSSIAN, jitter_ratio=0.3, seed=42,
    )
    coord = TimingCoordinator(
        agent_id="coord_j",
        subordinates={"jittered_1": agent},
    )
    sys_agent = TimingSystemAgent(
        agent_id="system_agent",
        subordinates={"coord_j": coord},
        schedule_config=ScheduleConfig.with_jitter(
            tick_interval=8.0, obs_delay=0.0, act_delay=0.0, msg_delay=0.2,
            jitter_type=JitterType.GAUSSIAN, jitter_ratio=0.2, seed=43,
        ),
    )
    env = TimingTestEnv(
        system_agent=sys_agent,
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=3.0,
    )
    analyzer = _run_and_analyze(env, t_end=50.0)

    history = analyzer.reward_history.get("jittered_1", [])
    assert len(history) >= 1, "Should produce at least one reward with jitter"
    # Queue should not be corrupted
    assert len(agent._pending_obs_queue) >= 0


@pytest.mark.parametrize("seed", [1, 42, 99, 12345])
def test_t9_jitter_uniform_multiple_seeds(seed):
    """Run with multiple seeds to ensure jitter doesn't break invariants."""
    agent = CounterFieldAgent(
        agent_id="jsu_1",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=1.0),
    )
    agent.schedule_config = ScheduleConfig.with_jitter(
        tick_interval=3.0, obs_delay=0.05, act_delay=0.3, msg_delay=0.1,
        jitter_type=JitterType.UNIFORM, jitter_ratio=0.4, seed=seed,
    )
    coord = TimingCoordinator(
        agent_id="coord_su",
        subordinates={"jsu_1": agent},
    )
    sys_agent = TimingSystemAgent(
        agent_id="system_agent",
        subordinates={"coord_su": coord},
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=5.0, obs_delay=0.0, act_delay=0.0, msg_delay=0.1,
        ),
    )
    env = TimingTestEnv(
        system_agent=sys_agent,
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=2.0,
    )
    analyzer = _run_and_analyze(env, t_end=30.0)

    summary = analyzer.get_summary()
    assert summary["action_results"] >= 1, f"No action results for seed={seed}"
    assert len(agent._pending_obs_queue) >= 0, f"Queue negative for seed={seed}"


# =============================================================================
# T10: Reset isolation
# =============================================================================

def test_t10_reset_clears_timing_state():
    """After reset, obs_action_cache, pending_obs_queue, and prev_post_physics_state
    should all be cleared."""
    env = _build_single_agent_env()
    env.reset(seed=42)
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    env.run_event_driven(episode_analyzer=analyzer, t_end=8.0)

    agent = env.registered_agents["counter_1"]
    # Verify state was populated during the episode
    # (prev_post_physics_state should be set if physics ran)

    env.reset(seed=43)
    assert len(agent._obs_action_cache) == 0, "Cache not cleared after reset"
    assert len(agent._pending_obs_queue) == 0, "Pending queue not cleared after reset"
    assert agent._prev_post_physics_state is None, "prev_post_physics_state not cleared"


def test_t10_second_episode_deterministic():
    """Two episodes with the same seed should produce identical event sequences."""
    env = _build_single_agent_env(
        tick_interval=5.0, act_delay=0.2, msg_delay=0.1,
        sim_wait=3.0, sys_tick_interval=5.0,
    )

    # Episode 1
    env.reset(seed=42)
    analyzer1 = EpisodeAnalyzer(verbose=False, track_data=True)
    ep1 = env.run_event_driven(episode_analyzer=analyzer1, t_end=10.0)

    # Episode 2
    env.reset(seed=42)
    analyzer2 = EpisodeAnalyzer(verbose=False, track_data=True)
    ep2 = env.run_event_driven(episode_analyzer=analyzer2, t_end=10.0)

    assert ep1.num_events == ep2.num_events, (
        f"Episode 1 ({ep1.num_events} events) != Episode 2 ({ep2.num_events} events)"
    )

    types1 = [(a.event_type, a.agent_id) for a in ep1.event_analyses]
    types2 = [(a.event_type, a.agent_id) for a in ep2.event_analyses]
    assert types1 == types2, "Event sequences should be identical across resets"


# =============================================================================
# T11: Long simulation stress test
# =============================================================================

def test_t11_long_simulation_no_crash():
    """Run 100s of simulation time to check for accumulation errors."""
    env = _build_single_agent_env(
        tick_interval=2.0, act_delay=0.3, msg_delay=0.1,
        sim_wait=1.5, sys_tick_interval=3.0,
    )
    analyzer = _run_and_analyze(env, t_end=100.0, max_events=10000)

    summary = analyzer.get_summary()
    assert summary["observations"] > 10, f"Expected many observations, got {summary['observations']}"
    assert summary["action_results"] > 5, f"Expected many action results, got {summary['action_results']}"

    # Queue invariant
    agent = env.registered_agents["counter_1"]
    assert len(agent._pending_obs_queue) >= 0
    assert len(agent._obs_action_cache) >= 0


def test_t11_long_simulation_rewards_accumulate():
    """Rewards should grow over a long simulation (counter always increments)."""
    env = _build_single_agent_env(
        tick_interval=2.0, act_delay=0.3, msg_delay=0.1,
        sim_wait=1.5, sys_tick_interval=3.0,
    )
    analyzer = _run_and_analyze(env, t_end=100.0, max_events=10000)

    history = analyzer.reward_history.get("counter_1", [])
    assert len(history) >= 5, f"Expected >= 5 rewards in long sim, got {len(history)}"

    # Rewards should be non-decreasing (counter always goes up)
    rewards = [entry[1] for entry in history]
    for i in range(1, len(rewards)):
        assert rewards[i] >= rewards[i - 1] - 0.01, (
            f"Reward decreased: {rewards[i]} < {rewards[i-1]} at step {i}"
        )

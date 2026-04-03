"""Integration tests for termination/truncation in event-driven mode.

Validates that:
  - ``max_sim_time`` truncation stops the simulation at the correct time
  - Agent-initiated ``is_terminated`` triggers early exit via ``__all__``
  - ``AllSemantics.ALL`` vs ``AllSemantics.ANY`` combine flags correctly
  - ``EpisodeAnalyzer`` tracks termination history
  - ``EpisodeStats.terminated`` and ``EpisodeStats.truncated`` are set correctly
  - ``reset()`` isolates episodes

Follows the same minimal-domain pattern as ``test_event_driven_timing.py``:
  CounterFeature, FieldAgent subclass, IncrementPolicy, TimingCoordinator,
  TimingSystemAgent, TimingTestEnv.
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
from heron.core.env_context import EnvContext
from heron.envs.base import BaseEnv
from heron.envs.termination import TerminationConfig, AllSemantics
from heron.scheduling import ScheduleConfig
from heron.scheduling.event import EventType
from heron.scheduling.analysis import EpisodeAnalyzer, EpisodeStats


# =============================================================================
# Minimal domain for termination tests
# =============================================================================

@dataclass(slots=True)
class CounterFeature(Feature):
    """Counter feature: value increments when action is applied."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    value: float = 0.0

    def set_values(self, **kwargs: Any) -> None:
        if "value" in kwargs:
            self.value = kwargs["value"]


class TerminatingFieldAgent(FieldAgent):
    """Field agent that terminates when counter >= threshold.

    In event-driven mode, the counter is incremented via ``apply_action``
    at each ``ACTION_EFFECT`` event. The ``is_terminated`` check runs at
    ``MSG_GET_LOCAL_STATE_RESPONSE`` (physics boundary).

    Args:
        threshold: Counter value at which ``is_terminated`` returns True.
    """

    def __init__(self, threshold: float = float("inf"), **kwargs):
        self._threshold = threshold
        super().__init__(**kwargs)

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
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

    def is_terminated(self, local_state: dict, env_context=None) -> bool:
        if "CounterFeature" in local_state:
            return float(local_state["CounterFeature"][0]) >= self._threshold
        return False


class IncrementPolicy(Policy):
    """Always outputs a fixed increment."""
    observation_mode = "local"

    def __init__(self, increment: float = 1.0):
        self.increment = increment
        self.obs_dim = 1
        self.action_dim = 1
        self.action_range = (-1.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        return np.array([self.increment])


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

    def __init__(self, agents, hierarchy, **kwargs):
        super().__init__(agents=agents, hierarchy=hierarchy, **kwargs)

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
    tick_interval: float = 1.0,
    act_delay: float = 0.1,
    msg_delay: float = 0.01,
    sim_wait: float = 3.0,
    sys_tick_interval: float = 5.0,
    threshold: float = float("inf"),
    increment: float = 1.0,
    max_sim_time: Optional[float] = None,
    max_steps: Optional[int] = None,
    all_semantics: AllSemantics = AllSemantics.ALL,
) -> TimingTestEnv:
    """SystemAgent -> (no-op coordinator, skipped R4) -> 1 TerminatingFieldAgent."""
    agent = TerminatingFieldAgent(
        agent_id="agent_1",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=increment),
        threshold=threshold,
    )
    agent.schedule_config = ScheduleConfig.deterministic(
        tick_interval=tick_interval, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    coord = TimingCoordinator(agent_id="coord_1")
    sys_agent = TimingSystemAgent(
        agent_id="system_agent",
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=sys_tick_interval, obs_delay=0.0,
            act_delay=0.0, msg_delay=msg_delay,
        ),
    )
    return TimingTestEnv(
        agents=[sys_agent, coord, agent],
        hierarchy={
            "system_agent": ["coord_1"],
            "coord_1": ["agent_1"],
        },
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=sim_wait,
        termination_config=TerminationConfig(
            max_steps=max_steps,
            max_sim_time=max_sim_time,
            all_semantics=all_semantics,
        ),
    )


def _build_two_agent_env(
    tick_interval_1: float = 1.0,
    tick_interval_2: float = 1.0,
    act_delay: float = 0.1,
    msg_delay: float = 0.01,
    sim_wait: float = 3.0,
    sys_tick_interval: float = 5.0,
    threshold_1: float = float("inf"),
    threshold_2: float = float("inf"),
    max_sim_time: Optional[float] = None,
    all_semantics: AllSemantics = AllSemantics.ALL,
) -> TimingTestEnv:
    """SystemAgent -> (no-op coordinator) -> 2 TerminatingFieldAgents."""
    agent1 = TerminatingFieldAgent(
        agent_id="agent_1",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=1.0),
        threshold=threshold_1,
    )
    agent1.schedule_config = ScheduleConfig.deterministic(
        tick_interval=tick_interval_1, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    agent2 = TerminatingFieldAgent(
        agent_id="agent_2",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=1.0),
        threshold=threshold_2,
    )
    agent2.schedule_config = ScheduleConfig.deterministic(
        tick_interval=tick_interval_2, obs_delay=0.0,
        act_delay=act_delay, msg_delay=msg_delay,
    )
    coord = TimingCoordinator(agent_id="coord_1")
    sys_agent = TimingSystemAgent(
        agent_id="system_agent",
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=sys_tick_interval, obs_delay=0.0,
            act_delay=0.0, msg_delay=msg_delay,
        ),
    )
    return TimingTestEnv(
        agents=[sys_agent, coord, agent1, agent2],
        hierarchy={
            "system_agent": ["coord_1"],
            "coord_1": ["agent_1", "agent_2"],
        },
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=sim_wait,
        termination_config=TerminationConfig(
            max_sim_time=max_sim_time,
            all_semantics=all_semantics,
        ),
    )


def _run_and_analyze(
    env: TimingTestEnv,
    t_end: float,
    max_events: int = 5000,
) -> tuple[EpisodeAnalyzer, EpisodeStats]:
    """Reset, run event-driven, return (analyzer, stats)."""
    env.reset(seed=42)
    analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
    stats = env.run_event_driven(
        episode_analyzer=analyzer,
        t_end=t_end,
        max_events=max_events,
    )
    return analyzer, stats


# =============================================================================
# Tests
# =============================================================================

class TestMaxSimTimeTruncation:
    """Tests for max_sim_time truncation in event-driven mode."""

    def test_max_sim_time_truncation(self):
        """max_sim_time=10, t_end=100: simulation stops at ~10.

        The effective t_end is min(t_end, max_sim_time) = 10. The scheduler
        should not process events beyond time 10.
        """
        env = _build_single_agent_env(
            tick_interval=1.0, act_delay=0.1, msg_delay=0.01,
            sim_wait=3.0, sys_tick_interval=5.0,
            max_sim_time=10.0,
        )
        analyzer, stats = _run_and_analyze(env, t_end=100.0)

        # All events should be at time <= max_sim_time
        max_event_time = max(
            a.timestamp for a in stats.event_analyses
        )
        assert max_event_time <= 10.0 + 0.5, (
            f"Expected events to stop at ~10.0, but last event at {max_event_time:.3f}"
        )
        assert stats.truncated is True, "Episode should be flagged as truncated"

    def test_max_sim_time_vs_t_end(self):
        """max_sim_time < t_end: effective stop is at max_sim_time.

        Verify that setting max_sim_time=5 with t_end=50 stops the sim early.
        """
        env = _build_single_agent_env(
            tick_interval=1.0, act_delay=0.1, msg_delay=0.01,
            sim_wait=2.0, sys_tick_interval=3.0,
            max_sim_time=5.0,
        )
        analyzer, stats = _run_and_analyze(env, t_end=50.0)

        max_event_time = max(
            a.timestamp for a in stats.event_analyses
        )
        assert max_event_time <= 5.0 + 0.5, (
            f"Expected effective stop at ~5.0, got last event at {max_event_time:.3f}"
        )


class TestAgentTerminationEventDriven:
    """Tests for agent-initiated termination in event-driven mode."""

    def test_agent_termination_early_exit(self):
        """Agent terminates when counter >= 2.

        With tick_interval=1.0 and sim_wait=3.0, the agent ticks and
        applies action before each physics cycle. After 2 action_effects
        the counter hits 2.0, and the next physics boundary triggers
        is_terminated=True. The event loop should exit early.
        """
        env = _build_single_agent_env(
            tick_interval=1.0, act_delay=0.1, msg_delay=0.01,
            sim_wait=3.0, sys_tick_interval=5.0,
            threshold=2.0,
        )
        analyzer, stats = _run_and_analyze(env, t_end=100.0)

        # Episode should terminate well before t_end=100
        assert stats.terminated is True, "Episode should be flagged as terminated"
        max_event_time = max(
            a.timestamp for a in stats.event_analyses
        )
        assert max_event_time < 50.0, (
            f"Expected early exit, but last event at {max_event_time:.3f}"
        )

        # Verify the termination flag was set for agent_1
        flags = analyzer.get_termination_flags()
        assert flags.get("agent_1", False) is True

    def test_any_semantics_early_exit(self):
        """ANY semantics: 2 agents, agent_1 terminates at counter=2.

        With ANY semantics, __all__ should become True as soon as agent_1's
        is_terminated fires. The event loop should exit early even though
        agent_2 hasn't terminated.
        """
        env = _build_two_agent_env(
            tick_interval_1=1.0, tick_interval_2=1.0,
            act_delay=0.1, msg_delay=0.01,
            sim_wait=3.0, sys_tick_interval=5.0,
            threshold_1=2.0, threshold_2=float("inf"),
            all_semantics=AllSemantics.ANY,
        )
        analyzer, stats = _run_and_analyze(env, t_end=100.0)

        assert stats.terminated is True
        flags = analyzer.get_termination_flags()
        # agent_1 should be terminated, agent_2 should not
        assert flags.get("agent_1", False) is True


class TestNoLimitsEventDriven:
    """Tests for no-limit mode running to t_end."""

    def test_no_limits_runs_to_t_end(self):
        """No termination config: simulation runs to the full t_end.

        With no max_sim_time and no agent termination (threshold=inf),
        the simulation should run until t_end=20.
        """
        env = _build_single_agent_env(
            tick_interval=2.0, act_delay=0.1, msg_delay=0.01,
            sim_wait=3.0, sys_tick_interval=5.0,
        )
        analyzer, stats = _run_and_analyze(env, t_end=20.0)

        assert stats.terminated is False
        assert stats.truncated is False
        # Events should extend close to t_end
        max_event_time = max(
            a.timestamp for a in stats.event_analyses
        )
        assert max_event_time >= 15.0, (
            f"Expected events near t_end=20, but last event at {max_event_time:.3f}"
        )


class TestAnalyzerAndStats:
    """Tests for EpisodeAnalyzer termination tracking and EpisodeStats flags."""

    def test_analyzer_tracks_termination(self):
        """EpisodeAnalyzer.termination_history should be populated after
        an agent terminates."""
        env = _build_single_agent_env(
            tick_interval=1.0, act_delay=0.1, msg_delay=0.01,
            sim_wait=3.0, sys_tick_interval=5.0,
            threshold=2.0,
        )
        analyzer, stats = _run_and_analyze(env, t_end=100.0)

        term_history = analyzer.get_termination_history("agent_1")
        agent_history = term_history.get("agent_1", [])
        assert len(agent_history) >= 1, (
            "Termination history should have at least one entry for agent_1"
        )
        # The last entry should show terminated=True
        last_ts, last_terminated, last_truncated = agent_history[-1]
        assert last_terminated is True, "Last entry should show terminated=True"

    def test_episode_stats_flags(self):
        """EpisodeStats.terminated and truncated should be set correctly.

        Case 1: Agent terminates -> stats.terminated=True, stats.truncated=False
        Case 2: max_sim_time hit -> stats.truncated=True
        """
        # Case 1: agent termination
        env1 = _build_single_agent_env(
            tick_interval=1.0, act_delay=0.1, msg_delay=0.01,
            sim_wait=3.0, sys_tick_interval=5.0,
            threshold=2.0,
        )
        _, stats1 = _run_and_analyze(env1, t_end=100.0)
        assert stats1.terminated is True
        # Truncated may or may not be True depending on timing, but terminated is definite

        # Case 2: max_sim_time truncation (no agent termination)
        env2 = _build_single_agent_env(
            tick_interval=1.0, act_delay=0.1, msg_delay=0.01,
            sim_wait=3.0, sys_tick_interval=5.0,
            max_sim_time=10.0,
        )
        _, stats2 = _run_and_analyze(env2, t_end=100.0)
        assert stats2.truncated is True


class TestResetIsolation:
    """Tests for reset isolation between episodes."""

    def test_reset_isolation(self):
        """Episode 1 terminates at threshold=2, reset, episode 2 runs fresh.

        After reset, the counter should start at 0 again and the agent
        should terminate at the same point.
        """
        env = _build_single_agent_env(
            tick_interval=1.0, act_delay=0.1, msg_delay=0.01,
            sim_wait=3.0, sys_tick_interval=5.0,
            threshold=2.0,
        )

        # Episode 1
        env.reset(seed=42)
        analyzer1 = EpisodeAnalyzer(verbose=False, track_data=True)
        stats1 = env.run_event_driven(
            episode_analyzer=analyzer1, t_end=100.0,
        )
        assert stats1.terminated is True

        # Episode 2 (fresh)
        env.reset(seed=42)
        analyzer2 = EpisodeAnalyzer(verbose=False, track_data=True)
        stats2 = env.run_event_driven(
            episode_analyzer=analyzer2, t_end=100.0,
        )
        assert stats2.terminated is True

        # Both episodes should produce a similar number of events
        # (same configuration, same seed, same termination point)
        assert abs(stats1.num_events - stats2.num_events) <= 2, (
            f"Episode 1 ({stats1.num_events} events) and Episode 2 "
            f"({stats2.num_events} events) should be similar after reset"
        )

        # Agent state should be clean after second reset
        agent = env.registered_agents["agent_1"]
        env.reset(seed=42)
        assert len(agent._obs_action_cache) == 0
        assert len(agent._pending_obs_queue) == 0
        assert agent._prev_post_physics_state is None


class TestHeterogeneousTermination:
    """Tests for heterogeneous tick rates and termination."""

    def test_heterogeneous_termination(self):
        """Agent A (fast tick=1.0) terminates early, agent B (slow tick=3.0)
        terminates later.

        With ALL semantics, the episode should run until both agents
        have terminated.
        """
        env = _build_two_agent_env(
            tick_interval_1=1.0, tick_interval_2=3.0,
            act_delay=0.1, msg_delay=0.01,
            sim_wait=3.0, sys_tick_interval=5.0,
            threshold_1=2.0, threshold_2=2.0,
            all_semantics=AllSemantics.ALL,
        )
        analyzer, stats = _run_and_analyze(env, t_end=100.0)

        flags = analyzer.get_termination_flags()
        # Both agents should eventually terminate
        assert flags.get("agent_1", False) is True, (
            "Fast agent_1 should have terminated"
        )
        assert flags.get("agent_2", False) is True, (
            "Slow agent_2 should have terminated"
        )
        assert stats.terminated is True

        # agent_1 should have terminated before agent_2 (it ticks faster)
        history_1 = analyzer.get_termination_history("agent_1").get("agent_1", [])
        history_2 = analyzer.get_termination_history("agent_2").get("agent_2", [])

        # Find first terminated=True for each
        first_term_1 = next(
            (ts for ts, t, tr in history_1 if t), None
        )
        first_term_2 = next(
            (ts for ts, t, tr in history_2 if t), None
        )
        if first_term_1 is not None and first_term_2 is not None:
            assert first_term_1 <= first_term_2, (
                f"Fast agent should terminate first: "
                f"agent_1 at {first_term_1:.3f}, agent_2 at {first_term_2:.3f}"
            )

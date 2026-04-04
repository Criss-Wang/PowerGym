"""Tests for custom event trigger feature.

Covers three levels:
  1. Unit tests: decorator registration and handler retrieval
  2. Scheduler tests: scheduling, dispatch, priority, delay, broadcast
  3. Integration tests: full env pipeline with custom events and analyzer tracking
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from heron.agents.base import Agent
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
from heron.scheduling.event import Event, EventType
from heron.scheduling.scheduler import EventScheduler
from heron.scheduling.schedule_config import ScheduleConfig
from heron.scheduling.analysis import EpisodeAnalyzer


# =============================================================================
# Minimal agent subclasses for unit/scheduler tests
# =============================================================================

class SimpleAgent(FieldAgent):
    """Minimal FieldAgent with a custom handler for unit tests."""

    @Agent.custom_handler("my_event")
    def on_my_event(self, event, scheduler):
        self._my_event_data = event.payload.get("key")

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        else:
            self.action.set_values(action)

    def set_state(self, *args, **kwargs) -> None:
        pass

    def apply_action(self):
        pass

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        return 0.0


class MultiHandlerAgent(FieldAgent):
    """Agent with two custom handlers."""

    @Agent.custom_handler("event_a")
    def handle_a(self, event, scheduler):
        self._a_fired = True

    @Agent.custom_handler("event_b")
    def handle_b(self, event, scheduler):
        self._b_fired = True

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        pass

    def set_state(self, *args, **kwargs) -> None:
        pass

    def apply_action(self):
        pass

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        return 0.0


class PlainAgent(FieldAgent):
    """Agent with no custom handlers."""

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        pass

    def set_state(self, *args, **kwargs) -> None:
        pass

    def apply_action(self):
        pass

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        return 0.0


# =============================================================================
# Unit Tests: TestCustomHandlerRegistration
# =============================================================================

class TestCustomHandlerRegistration:
    """Verify that the @Agent.custom_handler decorator registers handlers
    at the class level and that get_custom_handlers returns bound callables."""

    def test_custom_handler_decorator(self):
        """Agent subclass with @Agent.custom_handler('my_event') has it in _custom_event_handler_funcs."""
        assert "my_event" in SimpleAgent._custom_event_handler_funcs
        func = SimpleAgent._custom_event_handler_funcs["my_event"]
        assert callable(func)

    def test_get_custom_handlers_returns_bound(self):
        """get_custom_handlers() returns bound callables that update instance state."""
        agent = SimpleAgent(agent_id="test_agent")
        handlers = agent.get_custom_handlers()

        assert "my_event" in handlers
        assert callable(handlers["my_event"])

        # Call the bound handler and verify it mutates the instance
        event = Event(
            timestamp=0.0,
            event_type=EventType.CUSTOM,
            agent_id="test_agent",
            payload={"key": "hello"},
        )
        scheduler = EventScheduler(start_time=0.0)
        handlers["my_event"](event, scheduler)
        assert agent._my_event_data == "hello"

    def test_multiple_custom_handlers(self):
        """Agent with 2 custom handlers: both registered."""
        assert "event_a" in MultiHandlerAgent._custom_event_handler_funcs
        assert "event_b" in MultiHandlerAgent._custom_event_handler_funcs

        agent = MultiHandlerAgent(agent_id="multi")
        handlers = agent.get_custom_handlers()
        assert len(handlers) == 2
        assert "event_a" in handlers
        assert "event_b" in handlers

    def test_no_custom_handlers_returns_empty(self):
        """Agent without custom handlers returns empty dict."""
        agent = PlainAgent(agent_id="plain")
        handlers = agent.get_custom_handlers()
        assert handlers == {}


# =============================================================================
# Scheduler Tests: TestCustomEventScheduling
# =============================================================================

class TestCustomEventScheduling:
    """Verify custom event scheduling, dispatch, priority, delay, and broadcast."""

    def test_schedule_custom_event(self):
        """Event appears in queue with EventType.CUSTOM and correct payload."""
        scheduler = EventScheduler(start_time=0.0)
        scheduler.schedule_custom_event(
            sender_id="sender",
            recipient_id="recipient",
            custom_type="my_type",
            payload={"data": 42},
        )
        assert scheduler.pending_count == 1
        event = scheduler.peek()
        assert event.event_type == EventType.CUSTOM
        assert event.agent_id == "recipient"
        assert event.payload["custom_type"] == "my_type"
        assert event.payload["sender"] == "sender"
        assert event.payload["data"] == 42

    def test_custom_event_dispatch(self):
        """Register handler manually, schedule event, process_next calls it."""
        scheduler = EventScheduler(start_time=0.0)
        scheduler._active_agents = {"agent_a"}

        call_log = []

        def handler(event, sched):
            call_log.append(event.payload)

        scheduler.set_custom_handler("agent_a", "ping", handler)

        scheduler.schedule_custom_event(
            sender_id="external",
            recipient_id="agent_a",
            custom_type="ping",
            payload={"msg": "hello"},
        )

        processed = scheduler.process_next()
        assert processed is not None
        assert processed.event_type == EventType.CUSTOM
        assert len(call_log) == 1
        assert call_log[0]["msg"] == "hello"

    def test_custom_event_priority(self):
        """Custom event with priority 0 fires before MESSAGE_DELIVERY at priority 2."""
        scheduler = EventScheduler(start_time=0.0)
        scheduler._active_agents = {"agent_a"}

        order = []

        scheduler.set_custom_handler("agent_a", "urgent", lambda e, s: order.append("custom"))
        scheduler.handlers = {
            "agent_a": {
                EventType.MESSAGE_DELIVERY: lambda e, s: order.append("msg"),
            },
        }

        # Both events at the same timestamp; custom has priority 0, msg has priority 2
        scheduler.schedule_custom_event(
            sender_id="sender",
            recipient_id="agent_a",
            custom_type="urgent",
            priority=0,
        )
        scheduler.schedule(Event(
            timestamp=0.0,
            event_type=EventType.MESSAGE_DELIVERY,
            agent_id="agent_a",
            priority=2,
            payload={"sender": "x", "message": {}},
        ))

        scheduler.process_next()
        scheduler.process_next()
        assert order == ["custom", "msg"]

    def test_custom_event_delay(self):
        """Event fires at current_time + delay."""
        scheduler = EventScheduler(start_time=5.0)
        scheduler.schedule_custom_event(
            sender_id="s",
            recipient_id="r",
            custom_type="delayed",
            delay=3.0,
        )
        event = scheduler.peek()
        assert event.timestamp == pytest.approx(8.0)

    def test_unknown_custom_type_raises(self):
        """process_next raises ValueError for unregistered custom type."""
        scheduler = EventScheduler(start_time=0.0)
        scheduler._active_agents = {"agent_a"}

        scheduler.schedule_custom_event(
            sender_id="s",
            recipient_id="agent_a",
            custom_type="unknown_type",
        )

        with pytest.raises(ValueError, match="No custom handler for 'unknown_type'"):
            scheduler.process_next()

    def test_custom_broadcast(self):
        """schedule_custom_broadcast creates one event per recipient."""
        scheduler = EventScheduler(start_time=0.0)
        recipients = ["r1", "r2", "r3"]
        event_ids = scheduler.schedule_custom_broadcast(
            sender_id="broadcaster",
            recipient_ids=recipients,
            custom_type="alert",
            payload={"level": "high"},
        )
        assert len(event_ids) == 3
        assert scheduler.pending_count == 3

        # Verify each event targets a different recipient
        seen_agents = set()
        while scheduler.event_queue:
            e = scheduler.pop()
            assert e.event_type == EventType.CUSTOM
            assert e.payload["custom_type"] == "alert"
            assert e.payload["level"] == "high"
            seen_agents.add(e.agent_id)
        assert seen_agents == {"r1", "r2", "r3"}


# =============================================================================
# Integration test domain
# =============================================================================

@dataclass(slots=True)
class CounterFeature(Feature):
    """Simple counter feature for integration tests."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    value: float = 0.0

    def set_values(self, **kwargs: Any) -> None:
        if "value" in kwargs:
            self.value = kwargs["value"]


class IncrementPolicy(Policy):
    """Always outputs a fixed increment action."""
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


class CustomEventFieldAgent(FieldAgent):
    """Field agent that handles a custom 'test_signal' event."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._signal_received = False

    @Agent.custom_handler("test_signal")
    def on_test_signal(self, event, scheduler):
        self._signal_received = True

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        else:
            self.action.set_values(action)

    def set_state(self, *args, **kwargs) -> None:
        pass

    def apply_action(self):
        pass

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        return 0.0


class SignalCoordinator(CoordinatorAgent):
    """Coordinator that sends a custom 'test_signal' event to subordinates
    during message_delivery_handler when it receives an obs response."""

    def init_action(self, features: List[Feature] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-10.0]), np.array([10.0])))
        action.set_values(np.array([0.0]))
        return action

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        return 0.0

    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event, scheduler):
        """Override to inject a custom event after compute_action."""
        from heron.agents.constants import MSG_GET_OBS_RESPONSE, MSG_KEY_BODY
        message_content = event.payload.get("message", {})

        if MSG_GET_OBS_RESPONSE in message_content:
            response_data = message_content[MSG_GET_OBS_RESPONSE]
            body = response_data[MSG_KEY_BODY]
            obs_dict = body["obs"]
            local_state = body["local_state"]
            obs = Observation.from_dict(obs_dict)

            self.sync_state_from_observed(local_state)
            self.compute_action(obs, scheduler)
            self._cache_obs_action(obs, self.action)

            # Send custom event to subordinates
            for sub_id in self.subordinates:
                scheduler.schedule_custom_event(
                    sender_id=self.agent_id,
                    recipient_id=sub_id,
                    custom_type="test_signal",
                    payload={"source": self.agent_id},
                    delay=0.0,
                )

            # R3: schedule reactive subordinate ticks after action coordination
            if self._should_send_subordinate_actions():
                for sub_id, sub in self.subordinates.items():
                    if not sub.is_periodic:
                        scheduler.schedule_agent_tick(sub_id)
        else:
            # Delegate all other messages to the parent implementation
            super().message_delivery_handler(event, scheduler)


class SignalSystemAgent(SystemAgent):
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


def _build_custom_event_env(
    tick_interval: float = 5.0,
    msg_delay: float = 0.1,
    sim_wait: float = 3.0,
    sys_tick_interval: float = 5.0,
) -> TimingTestEnv:
    """Build env: SystemAgent -> SignalCoordinator -> CustomEventFieldAgent."""
    field = CustomEventFieldAgent(
        agent_id="field_1",
        features=[CounterFeature(value=0.0)],
        policy=IncrementPolicy(increment=1.0),
    )
    field.schedule_config = ScheduleConfig.deterministic(
        tick_interval=tick_interval, obs_delay=0.0,
        act_delay=0.2, msg_delay=msg_delay,
    )

    coord = SignalCoordinator(
        agent_id="coord_1",
        protocol=Protocol(action_protocol=EqualSplitProtocol()),
        policy=IncrementPolicy(increment=1.0),
    )
    coord.schedule_config = ScheduleConfig.deterministic(
        tick_interval=tick_interval, obs_delay=0.0,
        act_delay=0.0, msg_delay=msg_delay,
    )

    sys_agent = SignalSystemAgent(
        agent_id="system_agent",
        schedule_config=ScheduleConfig.deterministic(
            tick_interval=sys_tick_interval, obs_delay=0.0,
            act_delay=0.0, msg_delay=msg_delay,
        ),
    )

    return TimingTestEnv(
        agents=[sys_agent, coord, field],
        hierarchy={
            "system_agent": ["coord_1"],
            "coord_1": ["field_1"],
        },
        scheduler_config={"start_time": 0.0},
        message_broker_config={"buffer_size": 1000},
        simulation_wait_interval=sim_wait,
    )


# =============================================================================
# Integration Tests: TestCustomEventIntegration
# =============================================================================

class TestCustomEventIntegration:
    """Full-pipeline integration tests for custom events."""

    def test_custom_event_in_env(self):
        """Coordinator sends custom event to field agent during event-driven run;
        field agent's custom handler fires."""
        env = _build_custom_event_env()
        env.reset()

        analyzer = EpisodeAnalyzer()
        # Run long enough for at least one full cycle:
        # system tick -> coord tick -> coord obs -> coord computes action
        # -> custom event dispatched -> field agent handler fires
        result = env.run_event_driven(t_end=20.0, episode_analyzer=analyzer)

        field_agent = env.registered_agents["field_1"]
        assert field_agent._signal_received is True, (
            "Custom event handler on field agent was never called"
        )

    def test_custom_event_tracked_by_analyzer(self):
        """EpisodeAnalyzer.custom_event_count increments and message_type is 'custom:test_signal'."""
        env = _build_custom_event_env()
        env.reset()

        analyzer = EpisodeAnalyzer()
        result = env.run_event_driven(t_end=20.0, episode_analyzer=analyzer)

        # At least one custom event should have been tracked
        assert analyzer.custom_event_count >= 1, (
            f"Expected custom_event_count >= 1, got {analyzer.custom_event_count}"
        )

        # Verify message_type format in the episode stats
        msg_type_counts = result.get_message_type_counts()
        assert "custom:test_signal" in msg_type_counts, (
            f"Expected 'custom:test_signal' in message type counts, got {msg_type_counts}"
        )
        assert msg_type_counts["custom:test_signal"] >= 1

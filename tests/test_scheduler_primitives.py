"""Unit tests for scheduler primitives: event cancellation, condition monitors.

Covers Phase 4.1 of impl_todo_2026_03_20.md:
  - Cancellation by ID and by (agent, type)
  - Condition evaluation
  - Cooldown enforcement
  - One-shot deregistration
  - Preemption of scheduled ticks
"""

import pytest

from heron.scheduling.condition_monitor import ConditionMonitor
from heron.scheduling.event import Event, EventType
from heron.scheduling.scheduler import EventScheduler


# =============================================================================
# Event Cancellation
# =============================================================================

class TestCancelEvent:
    def test_cancel_by_id(self):
        s = EventScheduler(start_time=0.0)
        eid = s.schedule(Event(timestamp=5.0, event_type=EventType.AGENT_TICK, agent_id="a"))
        assert s.pending_count == 1

        found = s.cancel_event(eid)
        assert found is True

        # Event still in queue (lazy deletion) but will be skipped
        assert s.pending_count == 1
        e = s.pop()
        assert e.cancelled is True

    def test_cancel_nonexistent_returns_false(self):
        s = EventScheduler(start_time=0.0)
        s.schedule(Event(timestamp=5.0, event_type=EventType.AGENT_TICK, agent_id="a"))
        assert s.cancel_event("nonexistent_id") is False

    def test_cancel_events_by_agent_and_type(self):
        s = EventScheduler(start_time=0.0)
        s.schedule(Event(timestamp=1.0, event_type=EventType.AGENT_TICK, agent_id="a"))
        s.schedule(Event(timestamp=2.0, event_type=EventType.AGENT_TICK, agent_id="a"))
        s.schedule(Event(timestamp=3.0, event_type=EventType.AGENT_TICK, agent_id="b"))
        s.schedule(Event(timestamp=4.0, event_type=EventType.ACTION_EFFECT, agent_id="a"))

        count = s.cancel_events("a", EventType.AGENT_TICK)
        assert count == 2

        # Collect remaining non-cancelled events
        remaining = []
        while s.event_queue:
            e = s.pop()
            if not e.cancelled:
                remaining.append(e)

        assert len(remaining) == 2
        assert remaining[0].agent_id == "b"
        assert remaining[1].agent_id == "a"
        assert remaining[1].event_type == EventType.ACTION_EFFECT

    def test_process_next_skips_cancelled(self):
        s = EventScheduler(start_time=0.0)
        s._active_agents = {"a", "b"}
        # Provide a no-op handler
        s.handlers = {
            "a": {EventType.AGENT_TICK: lambda e, sc: None},
            "b": {EventType.AGENT_TICK: lambda e, sc: None},
        }

        eid1 = s.schedule(Event(timestamp=1.0, event_type=EventType.AGENT_TICK, agent_id="a"))
        s.schedule(Event(timestamp=2.0, event_type=EventType.AGENT_TICK, agent_id="b"))

        s.cancel_event(eid1)

        # process_next should skip "a" and process "b"
        processed = s.process_next()
        assert processed.agent_id == "b"
        assert processed.timestamp == 2.0

    def test_cancel_already_cancelled_returns_false(self):
        s = EventScheduler(start_time=0.0)
        eid = s.schedule(Event(timestamp=1.0, event_type=EventType.AGENT_TICK, agent_id="a"))
        s.cancel_event(eid)
        assert s.cancel_event(eid) is False  # Already cancelled


# =============================================================================
# Condition Monitor Evaluation
# =============================================================================

class TestConditionEvaluation:
    def _make_scheduler(self):
        s = EventScheduler(start_time=0.0)
        s._active_agents = {"agent_1", "agent_2"}
        s.handlers = {
            "agent_1": {EventType.CONDITION_TRIGGER: lambda e, sc: None},
            "agent_2": {EventType.CONDITION_TRIGGER: lambda e, sc: None},
        }
        return s

    def test_condition_fires_when_true(self):
        s = self._make_scheduler()
        s.current_time = 5.0
        s.register_condition(ConditionMonitor(
            monitor_id="test_cond",
            agent_id="agent_1",
            condition_fn=lambda state: state.get("value", 0) > 10,
        ))

        triggered = s.evaluate_conditions({"value": 20})
        assert len(triggered) == 1
        assert triggered[0].event_type == EventType.CONDITION_TRIGGER
        assert triggered[0].agent_id == "agent_1"
        assert triggered[0].payload["monitor_id"] == "test_cond"

    def test_condition_does_not_fire_when_false(self):
        s = self._make_scheduler()
        s.current_time = 5.0
        s.register_condition(ConditionMonitor(
            monitor_id="test_cond",
            agent_id="agent_1",
            condition_fn=lambda state: state.get("value", 0) > 100,
        ))

        triggered = s.evaluate_conditions({"value": 5})
        assert len(triggered) == 0


class TestConditionCooldown:
    def _make_scheduler(self):
        s = EventScheduler(start_time=0.0)
        s._active_agents = {"agent_1"}
        s.handlers = {"agent_1": {EventType.CONDITION_TRIGGER: lambda e, sc: None}}
        return s

    def test_cooldown_suppresses_rapid_triggers(self):
        s = self._make_scheduler()
        s.register_condition(ConditionMonitor(
            monitor_id="cool",
            agent_id="agent_1",
            condition_fn=lambda state: True,
            cooldown=10.0,
        ))

        # First evaluation at t=1: fires
        s.current_time = 1.0
        t1 = s.evaluate_conditions({})
        assert len(t1) == 1

        # Second evaluation at t=5 (within cooldown): suppressed
        s.current_time = 5.0
        t2 = s.evaluate_conditions({})
        assert len(t2) == 0

        # Third evaluation at t=12 (cooldown expired): fires
        s.current_time = 12.0
        t3 = s.evaluate_conditions({})
        assert len(t3) == 1

    def test_zero_cooldown_always_fires(self):
        s = self._make_scheduler()
        s.register_condition(ConditionMonitor(
            monitor_id="no_cool",
            agent_id="agent_1",
            condition_fn=lambda state: True,
            cooldown=0.0,
        ))

        s.current_time = 1.0
        assert len(s.evaluate_conditions({})) == 1
        s.current_time = 1.0  # Same time
        assert len(s.evaluate_conditions({})) == 1


class TestConditionOneShot:
    def _make_scheduler(self):
        s = EventScheduler(start_time=0.0)
        s._active_agents = {"agent_1"}
        s.handlers = {"agent_1": {EventType.CONDITION_TRIGGER: lambda e, sc: None}}
        return s

    def test_one_shot_deregisters_after_first_fire(self):
        s = self._make_scheduler()
        s.register_condition(ConditionMonitor(
            monitor_id="once",
            agent_id="agent_1",
            condition_fn=lambda state: True,
            one_shot=True,
        ))

        s.current_time = 1.0
        t1 = s.evaluate_conditions({})
        assert len(t1) == 1
        # Monitor should be removed
        assert len(s._condition_monitors) == 0

        # Second evaluation: no monitors → no triggers
        s.current_time = 2.0
        t2 = s.evaluate_conditions({})
        assert len(t2) == 0

    def test_non_one_shot_persists(self):
        s = self._make_scheduler()
        s.register_condition(ConditionMonitor(
            monitor_id="persist",
            agent_id="agent_1",
            condition_fn=lambda state: True,
            one_shot=False,
        ))

        s.current_time = 1.0
        s.evaluate_conditions({})
        assert len(s._condition_monitors) == 1

        s.current_time = 2.0
        s.evaluate_conditions({})
        assert len(s._condition_monitors) == 1


class TestConditionPreemption:
    def _make_scheduler(self):
        s = EventScheduler(start_time=0.0)
        s._active_agents = {"agent_1"}
        s.handlers = {
            "agent_1": {
                EventType.CONDITION_TRIGGER: lambda e, sc: None,
                EventType.AGENT_TICK: lambda e, sc: None,
            },
        }
        return s

    def test_preempt_cancels_next_tick(self):
        s = self._make_scheduler()
        # Schedule a periodic tick for agent_1
        s.schedule(Event(timestamp=10.0, event_type=EventType.AGENT_TICK, agent_id="agent_1"))
        assert s.pending_count == 1

        s.register_condition(ConditionMonitor(
            monitor_id="preempt",
            agent_id="agent_1",
            condition_fn=lambda state: True,
            preempt_next_tick=True,
        ))

        s.current_time = 5.0
        triggered = s.evaluate_conditions({})
        assert len(triggered) == 1

        # The AGENT_TICK should now be cancelled
        remaining = []
        while s.event_queue:
            e = s.pop()
            if not e.cancelled:
                remaining.append(e)

        # Only the CONDITION_TRIGGER should remain (the AGENT_TICK was cancelled)
        tick_events = [e for e in remaining if e.event_type == EventType.AGENT_TICK]
        assert len(tick_events) == 0

    def test_no_preempt_preserves_tick(self):
        s = self._make_scheduler()
        s.schedule(Event(timestamp=10.0, event_type=EventType.AGENT_TICK, agent_id="agent_1"))

        s.register_condition(ConditionMonitor(
            monitor_id="no_preempt",
            agent_id="agent_1",
            condition_fn=lambda state: True,
            preempt_next_tick=False,
        ))

        s.current_time = 5.0
        s.evaluate_conditions({})

        # AGENT_TICK should still be there (not cancelled)
        ticks = [e for e in s.event_queue if e.event_type == EventType.AGENT_TICK and not e.cancelled]
        assert len(ticks) == 1


class TestDeregisterCondition:
    def test_deregister_removes_monitor(self):
        s = EventScheduler(start_time=0.0)
        s.register_condition(ConditionMonitor(
            monitor_id="to_remove", agent_id="a", condition_fn=lambda _: True,
        ))
        s.register_condition(ConditionMonitor(
            monitor_id="to_keep", agent_id="b", condition_fn=lambda _: True,
        ))
        assert len(s._condition_monitors) == 2

        found = s.deregister_condition("to_remove")
        assert found is True
        assert len(s._condition_monitors) == 1
        assert s._condition_monitors[0].monitor_id == "to_keep"

    def test_deregister_nonexistent_returns_false(self):
        s = EventScheduler(start_time=0.0)
        assert s.deregister_condition("nope") is False


# =============================================================================
# ConditionMonitor helper constructors
# =============================================================================

class TestConditionMonitorThreshold:
    def test_below_fires(self):
        m = ConditionMonitor.threshold(
            monitor_id="under_v",
            agent_id="agent_1",
            key_path=["agent_states", "bus_5", "vm_pu"],
            threshold=0.95,
            direction="below",
        )
        state = {"agent_states": {"bus_5": {"vm_pu": 0.91}}}
        assert m.condition_fn(state) is True

    def test_below_does_not_fire(self):
        m = ConditionMonitor.threshold(
            monitor_id="under_v",
            agent_id="agent_1",
            key_path=["agent_states", "bus_5", "vm_pu"],
            threshold=0.95,
            direction="below",
        )
        state = {"agent_states": {"bus_5": {"vm_pu": 1.02}}}
        assert m.condition_fn(state) is False

    def test_above_fires(self):
        m = ConditionMonitor.threshold(
            monitor_id="over_load",
            agent_id="agent_1",
            key_path=["loading_pct"],
            threshold=100.0,
            direction="above",
        )
        assert m.condition_fn({"loading_pct": 120.0}) is True
        assert m.condition_fn({"loading_pct": 80.0}) is False

    def test_missing_key_returns_false(self):
        m = ConditionMonitor.threshold(
            monitor_id="safe",
            agent_id="agent_1",
            key_path=["agent_states", "bus_99", "vm_pu"],
            threshold=0.95,
            direction="below",
        )
        state = {"agent_states": {"bus_5": {"vm_pu": 0.5}}}
        assert m.condition_fn(state) is False  # bus_99 doesn't exist

    def test_preserves_cooldown_and_flags(self):
        m = ConditionMonitor.threshold(
            monitor_id="test",
            agent_id="a",
            key_path=["v"],
            threshold=1.0,
            direction="below",
            cooldown=5.0,
            one_shot=True,
            preempt_next_tick=True,
        )
        assert m.cooldown == 5.0
        assert m.one_shot is True
        assert m.preempt_next_tick is True


class TestConditionMonitorNoop:
    def test_noop_never_fires(self):
        m = ConditionMonitor.noop()
        assert m.condition_fn({"anything": 42}) is False
        assert m.condition_fn({}) is False

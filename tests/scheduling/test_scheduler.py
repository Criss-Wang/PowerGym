"""Tests for EventScheduler class."""

import pytest

from heron.scheduling.event import Event, EventType
from heron.scheduling.scheduler import EventScheduler


class TestEventSchedulerInitialization:
    """Test EventScheduler initialization."""

    def test_basic_initialization(self):
        """Test basic scheduler creation."""
        scheduler = EventScheduler()

        assert scheduler.current_time == 0.0
        assert scheduler.event_queue == []
        assert scheduler.pending_count == 0
        assert scheduler.processed_count == 0

    def test_initialization_with_start_time(self):
        """Test initialization with custom start time."""
        scheduler = EventScheduler(start_time=10.0)

        assert scheduler.current_time == 10.0


class TestEventSchedulerAgentRegistration:
    """Test agent registration."""

    def test_register_agent_basic(self):
        """Test basic agent registration."""
        scheduler = EventScheduler()

        scheduler.register_agent("agent_1", tick_interval=1.0)

        assert "agent_1" in scheduler.agent_intervals
        assert scheduler.agent_intervals["agent_1"] == 1.0
        assert scheduler.pending_count == 1  # First tick scheduled

    def test_register_agent_with_delays(self):
        """Test agent registration with delays."""
        scheduler = EventScheduler()

        scheduler.register_agent(
            "agent_1",
            tick_interval=2.0,
            obs_delay=0.5,
            act_delay=1.0,
        )

        assert scheduler.agent_obs_delays["agent_1"] == 0.5
        assert scheduler.agent_act_delays["agent_1"] == 1.0

    def test_register_agent_with_first_tick(self):
        """Test agent registration with custom first tick time."""
        scheduler = EventScheduler(start_time=0.0)

        scheduler.register_agent("agent_1", tick_interval=1.0, first_tick=5.0)

        event = scheduler.peek()
        assert event.timestamp == 5.0

    def test_unregister_agent(self):
        """Test agent unregistration."""
        scheduler = EventScheduler()
        scheduler.register_agent("agent_1", tick_interval=1.0)

        scheduler.unregister_agent("agent_1")

        assert "agent_1" not in scheduler.agent_intervals
        assert "agent_1" not in scheduler._active_agents


class TestEventSchedulerSchedule:
    """Test event scheduling."""

    def test_schedule_event(self):
        """Test scheduling an event."""
        scheduler = EventScheduler()

        event = Event(timestamp=5.0, event_type=EventType.CUSTOM)
        scheduler.schedule(event)

        assert scheduler.pending_count == 1
        assert scheduler.peek() is event

    def test_schedule_multiple_events_ordered(self):
        """Test events are ordered by timestamp."""
        scheduler = EventScheduler()

        scheduler.schedule(Event(timestamp=3.0))
        scheduler.schedule(Event(timestamp=1.0))
        scheduler.schedule(Event(timestamp=2.0))

        assert scheduler.peek().timestamp == 1.0

    def test_schedule_assigns_sequence_numbers(self):
        """Test that sequence numbers are assigned."""
        scheduler = EventScheduler()

        e1 = Event(timestamp=1.0)
        e2 = Event(timestamp=1.0)
        scheduler.schedule(e1)
        scheduler.schedule(e2)

        assert e1.sequence == 0
        assert e2.sequence == 1

    def test_schedule_agent_tick(self):
        """Test schedule_agent_tick helper."""
        scheduler = EventScheduler()
        scheduler.agent_intervals["agent_1"] = 2.0
        scheduler.current_time = 5.0

        scheduler.schedule_agent_tick("agent_1")

        event = scheduler.peek()
        assert event.timestamp == 7.0  # 5.0 + 2.0
        assert event.event_type == EventType.AGENT_TICK
        assert event.agent_id == "agent_1"

    def test_schedule_action_effect(self):
        """Test schedule_action_effect helper."""
        scheduler = EventScheduler()
        scheduler.agent_act_delays["agent_1"] = 0.5
        scheduler.current_time = 1.0

        scheduler.schedule_action_effect("agent_1", action=[1.0, 2.0])

        event = scheduler.peek()
        assert event.timestamp == 1.5  # 1.0 + 0.5
        assert event.event_type == EventType.ACTION_EFFECT
        assert event.payload["action"] == [1.0, 2.0]

    def test_schedule_message_delivery(self):
        """Test schedule_message_delivery helper."""
        scheduler = EventScheduler()
        scheduler.current_time = 2.0

        scheduler.schedule_message_delivery(
            sender_id="sender",
            recipient_id="recipient",
            message={"data": 123},
            delay=1.0,
        )

        event = scheduler.peek()
        assert event.timestamp == 3.0  # 2.0 + 1.0
        assert event.event_type == EventType.MESSAGE_DELIVERY
        assert event.agent_id == "recipient"
        assert event.payload["sender"] == "sender"


class TestEventSchedulerProcessing:
    """Test event processing."""

    def test_peek_returns_next_event(self):
        """Test peek returns next event without removing."""
        scheduler = EventScheduler()
        event = Event(timestamp=1.0)
        scheduler.schedule(event)

        peeked = scheduler.peek()

        assert peeked is event
        assert scheduler.pending_count == 1

    def test_peek_empty_returns_none(self):
        """Test peek returns None when empty."""
        scheduler = EventScheduler()

        assert scheduler.peek() is None

    def test_pop_returns_and_removes_event(self):
        """Test pop returns and removes event."""
        scheduler = EventScheduler()
        event = Event(timestamp=1.0)
        scheduler.schedule(event)

        popped = scheduler.pop()

        assert popped is event
        assert scheduler.pending_count == 0

    def test_pop_empty_returns_none(self):
        """Test pop returns None when empty."""
        scheduler = EventScheduler()

        assert scheduler.pop() is None

    def test_process_next_basic(self):
        """Test processing single event."""
        scheduler = EventScheduler()
        scheduler.schedule(Event(timestamp=1.0))

        result = scheduler.process_next()

        assert result is True
        assert scheduler.current_time == 1.0
        assert scheduler.processed_count == 1

    def test_process_next_empty_returns_false(self):
        """Test process_next returns False when empty."""
        scheduler = EventScheduler()

        result = scheduler.process_next()

        assert result is False

    def test_process_next_calls_handler(self):
        """Test that handler is called for event."""
        scheduler = EventScheduler()
        handled_events = []

        def handler(event, sched):
            handled_events.append(event)

        scheduler.set_handler(EventType.CUSTOM, handler)
        event = Event(timestamp=1.0, event_type=EventType.CUSTOM)
        scheduler.schedule(event)

        scheduler.process_next()

        assert len(handled_events) == 1
        assert handled_events[0] is event

    def test_process_next_auto_schedules_next_tick(self):
        """Test that AGENT_TICK auto-schedules next tick."""
        scheduler = EventScheduler()
        scheduler.register_agent("agent_1", tick_interval=1.0)

        # Initial tick at t=0
        scheduler.process_next()

        # Should have scheduled next tick at t=1
        next_event = scheduler.peek()
        assert next_event.timestamp == 1.0
        assert next_event.agent_id == "agent_1"

    def test_process_skips_unregistered_agent(self):
        """Test events for unregistered agents are skipped."""
        scheduler = EventScheduler()
        scheduler.schedule(Event(
            timestamp=1.0,
            event_type=EventType.AGENT_TICK,
            agent_id="unknown_agent",
        ))

        result = scheduler.process_next()

        assert result is True  # Event was processed (skipped)
        assert scheduler.pending_count == 0  # No next tick scheduled


class TestEventSchedulerRunUntil:
    """Test run_until method."""

    def test_run_until_time_limit(self):
        """Test running until time limit."""
        scheduler = EventScheduler()
        scheduler.register_agent("agent_1", tick_interval=1.0)

        count = scheduler.run_until(t_end=5.0)

        assert count == 6  # t=0, 1, 2, 3, 4, 5 (stops when next event > t_end)
        assert scheduler.current_time == 5.0

    def test_run_until_event_limit(self):
        """Test running until event limit."""
        scheduler = EventScheduler()
        scheduler.register_agent("agent_1", tick_interval=1.0)

        count = scheduler.run_until(t_end=100.0, max_events=3)

        assert count == 3
        assert scheduler.current_time == 2.0

    def test_run_until_empty_queue(self):
        """Test run_until with empty queue."""
        scheduler = EventScheduler()

        count = scheduler.run_until(t_end=10.0)

        assert count == 0


class TestEventSchedulerRunSteps:
    """Test run_steps method."""

    def test_run_steps(self):
        """Test running n steps."""
        scheduler = EventScheduler()
        scheduler.register_agent("agent_1", tick_interval=1.0)

        count = scheduler.run_steps(5)

        assert count == 5
        assert scheduler.processed_count == 5

    def test_run_steps_empty_queue(self):
        """Test run_steps with empty queue."""
        scheduler = EventScheduler()

        count = scheduler.run_steps(5)

        assert count == 0


class TestEventSchedulerReset:
    """Test reset functionality."""

    def test_clear(self):
        """Test clearing event queue."""
        scheduler = EventScheduler()
        scheduler.schedule(Event(timestamp=1.0))
        scheduler.schedule(Event(timestamp=2.0))

        scheduler.clear()

        assert scheduler.pending_count == 0

    def test_reset(self):
        """Test full reset."""
        scheduler = EventScheduler()
        scheduler.register_agent("agent_1", tick_interval=1.0)
        scheduler.run_steps(5)

        scheduler.reset(start_time=0.0)

        assert scheduler.current_time == 0.0
        assert scheduler.processed_count == 0
        assert scheduler.pending_count == 1  # Re-scheduled first tick


class TestEventSchedulerRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__."""
        scheduler = EventScheduler(start_time=5.0)
        scheduler.register_agent("agent_1", tick_interval=1.0)

        repr_str = repr(scheduler)

        assert "EventScheduler" in repr_str
        assert "5.0" in repr_str or "5.000" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

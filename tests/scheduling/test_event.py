"""Tests for Event and EventType classes."""

import pytest

from heron.scheduling.event import Event, EventType


class TestEventType:
    """Test EventType enum."""

    def test_event_type_values(self):
        """Test all event type values exist."""
        assert EventType.AGENT_TICK.value == "agent_tick"
        assert EventType.ACTION_EFFECT.value == "action_effect"
        assert EventType.MESSAGE_DELIVERY.value == "message_delivery"
        assert EventType.OBSERVATION_READY.value == "observation_ready"
        assert EventType.ENV_UPDATE.value == "env_update"
        assert EventType.CUSTOM.value == "custom"


class TestEventInitialization:
    """Test Event initialization."""

    def test_basic_initialization(self):
        """Test basic event creation."""
        event = Event(timestamp=5.0)

        assert event.timestamp == 5.0
        assert event.priority == 0
        assert event.sequence == 0
        assert event.event_type == EventType.AGENT_TICK
        assert event.agent_id is None
        assert event.payload == {}

    def test_initialization_with_all_params(self):
        """Test event with all parameters."""
        event = Event(
            timestamp=10.0,
            priority=2,
            sequence=5,
            event_type=EventType.ACTION_EFFECT,
            agent_id="agent_1",
            payload={"action": [0.5, 0.5]},
        )

        assert event.timestamp == 10.0
        assert event.priority == 2
        assert event.sequence == 5
        assert event.event_type == EventType.ACTION_EFFECT
        assert event.agent_id == "agent_1"
        assert event.payload == {"action": [0.5, 0.5]}


class TestEventOrdering:
    """Test Event comparison and ordering."""

    def test_events_ordered_by_timestamp(self):
        """Test events are ordered primarily by timestamp."""
        e1 = Event(timestamp=1.0)
        e2 = Event(timestamp=2.0)
        e3 = Event(timestamp=0.5)

        events = sorted([e1, e2, e3])

        assert events[0].timestamp == 0.5
        assert events[1].timestamp == 1.0
        assert events[2].timestamp == 2.0

    def test_same_timestamp_ordered_by_priority(self):
        """Test same-timestamp events ordered by priority."""
        e1 = Event(timestamp=1.0, priority=2)
        e2 = Event(timestamp=1.0, priority=0)
        e3 = Event(timestamp=1.0, priority=1)

        events = sorted([e1, e2, e3])

        assert events[0].priority == 0
        assert events[1].priority == 1
        assert events[2].priority == 2

    def test_same_timestamp_priority_ordered_by_sequence(self):
        """Test events with same timestamp/priority ordered by sequence."""
        e1 = Event(timestamp=1.0, priority=0, sequence=3)
        e2 = Event(timestamp=1.0, priority=0, sequence=1)
        e3 = Event(timestamp=1.0, priority=0, sequence=2)

        events = sorted([e1, e2, e3])

        assert events[0].sequence == 1
        assert events[1].sequence == 2
        assert events[2].sequence == 3

    def test_event_comparison_less_than(self):
        """Test event less than comparison."""
        e1 = Event(timestamp=1.0)
        e2 = Event(timestamp=2.0)

        assert e1 < e2
        assert not e2 < e1

    def test_event_comparison_equal(self):
        """Test event equality comparison."""
        e1 = Event(timestamp=1.0, priority=0, sequence=0)
        e2 = Event(timestamp=1.0, priority=0, sequence=0)

        assert e1 == e2

    def test_event_comparison_ignores_non_compare_fields(self):
        """Test that event_type, agent_id, payload don't affect ordering."""
        e1 = Event(timestamp=1.0, event_type=EventType.AGENT_TICK, agent_id="a1")
        e2 = Event(timestamp=1.0, event_type=EventType.ACTION_EFFECT, agent_id="a2")

        # Should be equal for comparison purposes
        assert not (e1 < e2)
        assert not (e2 < e1)


class TestEventRepr:
    """Test Event string representation."""

    def test_repr_basic(self):
        """Test basic __repr__."""
        event = Event(timestamp=1.5, event_type=EventType.AGENT_TICK, agent_id="test")

        repr_str = repr(event)

        assert "Event" in repr_str
        assert "1.5" in repr_str or "1.500" in repr_str
        assert "agent_tick" in repr_str
        assert "test" in repr_str

    def test_repr_with_priority(self):
        """Test __repr__ includes priority."""
        event = Event(timestamp=1.0, priority=5)

        repr_str = repr(event)

        assert "prio=5" in repr_str


class TestEventPayload:
    """Test Event payload handling."""

    def test_empty_payload_default(self):
        """Test default empty payload."""
        event = Event(timestamp=1.0)

        assert event.payload == {}

    def test_payload_with_data(self):
        """Test payload with data."""
        payload = {"action": [1, 2, 3], "message": "test"}
        event = Event(timestamp=1.0, payload=payload)

        assert event.payload == payload

    def test_payload_mutable(self):
        """Test payload is mutable."""
        event = Event(timestamp=1.0)
        event.payload["key"] = "value"

        assert event.payload["key"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

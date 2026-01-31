"""Integration tests for event-driven discrete simulation.

Tests the HERON EventScheduler that enables heterogeneous tick rates and
configurable latencies for realistic multi-agent simulation.
"""

import pytest
import numpy as np

from heron.scheduling.event import Event, EventType
from heron.scheduling.scheduler import EventScheduler
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.feature import FeatureProvider


# =============================================================================
# Test Fixtures
# =============================================================================

class MockFeature(FeatureProvider):
    """Mock feature for testing."""
    visibility = ["public"]

    def __init__(self, value: float = 0.0):
        self.value = value

    def vector(self):
        return np.array([self.value], dtype=np.float32)

    def names(self):
        return ["value"]

    def to_dict(self):
        return {"value": self.value}

    @classmethod
    def from_dict(cls, d):
        return cls(value=d.get("value", 0.0))

    def set_values(self, **kwargs):
        if "value" in kwargs:
            self.value = kwargs["value"]


class SensorAgent(FieldAgent):
    """Simulated sensor with fast tick rate."""

    def __init__(self, agent_id, **kwargs):
        kwargs.setdefault("tick_interval", 1.0)  # 1 second sampling
        super().__init__(agent_id=agent_id, **kwargs)
        self.readings = []

    def set_action(self):
        self.action.set_specs(dim_c=0, dim_d=0)  # Sensor has no actions

    def set_state(self):
        self.state.features = [MockFeature(value=0.0)]

    def _get_obs(self):
        return self.state.vector()

    def tick(self, scheduler, current_time, global_state=None, proxy=None):
        """Record reading at each tick."""
        self._timestep = current_time
        self.readings.append({"time": current_time, "value": self.state.features[0].value})


class ControllerAgent(FieldAgent):
    """Simulated controller with slower tick rate."""

    def __init__(self, agent_id, **kwargs):
        kwargs.setdefault("tick_interval", 5.0)  # 5 second control interval
        super().__init__(agent_id=agent_id, **kwargs)
        self.control_actions = []

    def set_action(self):
        self.action.set_specs(dim_c=1, range=(np.array([0.0]), np.array([1.0])))

    def set_state(self):
        self.state.features = [MockFeature(value=0.0)]

    def _get_obs(self):
        return self.state.vector()

    def tick(self, scheduler, current_time, global_state=None, proxy=None):
        """Compute control action at each tick."""
        self._timestep = current_time
        self.action.sample()
        self.control_actions.append({"time": current_time, "action": self.action.c.copy()})


# =============================================================================
# Integration Tests
# =============================================================================

class TestHeterogeneousTickRates:
    """Test agents with different tick rates in same simulation."""

    def test_different_tick_intervals(self):
        """Test that agents tick at their specified intervals."""
        scheduler = EventScheduler(start_time=0.0)

        # Register agents with different tick rates
        scheduler.register_agent("sensor_fast", tick_interval=1.0)
        scheduler.register_agent("sensor_slow", tick_interval=5.0)

        # Track ticks
        fast_ticks = []
        slow_ticks = []

        def tick_handler(event, sched):
            if event.agent_id == "sensor_fast":
                fast_ticks.append(event.timestamp)
            elif event.agent_id == "sensor_slow":
                slow_ticks.append(event.timestamp)

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)

        # Run simulation for 10 seconds
        scheduler.run_until(t_end=10.0)

        # Fast sensor should tick at t=0,1,2,3,4,5,6,7,8,9,10 (11 times)
        assert len(fast_ticks) == 11
        np.testing.assert_array_almost_equal(
            fast_ticks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )

        # Slow sensor should tick at t=0,5,10 (3 times)
        assert len(slow_ticks) == 3
        np.testing.assert_array_almost_equal(slow_ticks, [0, 5, 10])

    def test_hierarchy_level_tick_rates(self):
        """Test realistic hierarchy tick rates: fast sensors, slow coordinators."""
        scheduler = EventScheduler(start_time=0.0)

        # Field level: fast (1s)
        # Coordinator level: medium (10s)
        # System level: slow (60s)
        scheduler.register_agent("sensor_1", tick_interval=1.0)
        scheduler.register_agent("sensor_2", tick_interval=1.0)
        scheduler.register_agent("coordinator", tick_interval=10.0)
        scheduler.register_agent("system", tick_interval=60.0)

        tick_counts = {"sensor_1": 0, "sensor_2": 0, "coordinator": 0, "system": 0}

        def count_handler(event, sched):
            if event.agent_id in tick_counts:
                tick_counts[event.agent_id] += 1

        scheduler.set_handler(EventType.AGENT_TICK, count_handler)

        # Run for 60 seconds
        scheduler.run_until(t_end=60.0)

        # Verify tick counts match expected rates
        assert tick_counts["sensor_1"] == 61  # t=0,1,...,60
        assert tick_counts["sensor_2"] == 61
        assert tick_counts["coordinator"] == 7  # t=0,10,20,30,40,50,60
        assert tick_counts["system"] == 2  # t=0,60


class TestActionDelays:
    """Test configurable action delays."""

    def test_action_effect_delay(self):
        """Test that action effects are delayed correctly."""
        scheduler = EventScheduler(start_time=0.0)

        # Register agent with action delay
        scheduler.register_agent("controller", tick_interval=5.0, act_delay=2.0)

        action_effects = []

        def tick_handler(event, sched):
            if event.event_type == EventType.AGENT_TICK:
                # Schedule action effect with delay
                sched.schedule_action_effect(
                    event.agent_id,
                    action={"power": 100.0},
                )

        def effect_handler(event, sched):
            action_effects.append({
                "time": event.timestamp,
                "action": event.payload["action"],
            })

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)
        scheduler.set_handler(EventType.ACTION_EFFECT, effect_handler)

        # Run simulation
        scheduler.run_until(t_end=10.0)

        # Action effects should occur 2 seconds after ticks
        # Ticks at t=0,5,10 -> Effects at t=2,7,12 (but we stop at 10)
        assert len(action_effects) == 2
        np.testing.assert_almost_equal(action_effects[0]["time"], 2.0)
        np.testing.assert_almost_equal(action_effects[1]["time"], 7.0)


class TestObservationDelays:
    """Test configurable observation delays."""

    def test_observation_delay_affects_state_view(self):
        """Test that observation delay affects which state agents see."""
        scheduler = EventScheduler(start_time=0.0)

        # Agent with observation delay sees state from the past
        scheduler.register_agent("controller", tick_interval=5.0, obs_delay=1.0)

        # Track what observations are made
        observations_made = []

        # Simulated state that changes over time (including pre-start state)
        state_history = {-1.0: 50, 0.0: 100, 5.0: 200, 10.0: 300}

        def tick_handler(event, sched):
            # With obs_delay=1.0, at t=5, agent sees state from t=4
            obs_time = event.timestamp - sched.agent_obs_delays.get(event.agent_id, 0.0)
            # Find most recent state at or before obs_time
            available_times = [t for t in state_history.keys() if t <= obs_time]
            if available_times:
                state_time = max(available_times)
                observations_made.append({
                    "tick_time": event.timestamp,
                    "obs_time": obs_time,
                    "state_value": state_history[state_time],
                })

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)

        scheduler.run_until(t_end=10.0)

        # At t=0, obs_time=-1, sees state from t=-1 (50)
        # At t=5, obs_time=4, sees state from t=0 (100)
        # At t=10, obs_time=9, sees state from t=5 (200)
        assert observations_made[0]["state_value"] == 50   # t=0 sees t=-1 state
        assert observations_made[1]["state_value"] == 100  # t=5 sees t=0 state
        assert observations_made[2]["state_value"] == 200  # t=10 sees t=5 state


class TestMessageDelays:
    """Test message delivery delays."""

    def test_message_delivery_delay(self):
        """Test that messages are delivered with specified delay."""
        scheduler = EventScheduler(start_time=0.0)

        delivered_messages = []

        # Register the recipient agent (required for events to not be skipped)
        scheduler.register_agent("device", tick_interval=10.0)

        # Schedule a message delivery
        scheduler.schedule_message_delivery(
            sender_id="coordinator",
            recipient_id="device",
            message={"action": 0.5},
            delay=3.0,
        )

        def delivery_handler(event, sched):
            delivered_messages.append({
                "time": event.timestamp,
                "recipient": event.agent_id,
                "message": event.payload["message"],
            })

        scheduler.set_handler(EventType.MESSAGE_DELIVERY, delivery_handler)

        scheduler.run_until(t_end=5.0)

        assert len(delivered_messages) == 1
        assert delivered_messages[0]["time"] == 3.0
        assert delivered_messages[0]["message"]["action"] == 0.5


class TestEventPriority:
    """Test event priority ordering."""

    def test_same_time_different_priority(self):
        """Test that events at same time are ordered by priority."""
        scheduler = EventScheduler(start_time=0.0)

        processed_events = []

        # Schedule events at same time with different priorities
        scheduler.schedule(Event(
            timestamp=1.0,
            priority=2,
            event_type=EventType.CUSTOM,
            payload={"name": "low_priority"},
        ))
        scheduler.schedule(Event(
            timestamp=1.0,
            priority=0,
            event_type=EventType.CUSTOM,
            payload={"name": "high_priority"},
        ))
        scheduler.schedule(Event(
            timestamp=1.0,
            priority=1,
            event_type=EventType.CUSTOM,
            payload={"name": "medium_priority"},
        ))

        def handler(event, sched):
            processed_events.append(event.payload["name"])

        scheduler.set_handler(EventType.CUSTOM, handler)

        scheduler.run_until(t_end=2.0)

        # Should be processed in priority order (low number = high priority)
        assert processed_events == ["high_priority", "medium_priority", "low_priority"]


class TestSimulationControl:
    """Test simulation control methods."""

    def test_run_until_time_limit(self):
        """Test running until time limit."""
        scheduler = EventScheduler(start_time=0.0)
        scheduler.register_agent("agent_1", tick_interval=1.0)

        scheduler.run_until(t_end=5.0)

        assert scheduler.current_time == 5.0

    def test_run_until_event_limit(self):
        """Test running until event limit."""
        scheduler = EventScheduler(start_time=0.0)
        scheduler.register_agent("agent_1", tick_interval=1.0)

        count = scheduler.run_until(t_end=100.0, max_events=3)

        assert count == 3
        assert scheduler.current_time == 2.0  # t=0,1,2 processed

    def test_run_steps(self):
        """Test running exact number of steps."""
        scheduler = EventScheduler(start_time=0.0)
        scheduler.register_agent("agent_1", tick_interval=2.0)
        scheduler.register_agent("agent_2", tick_interval=3.0)

        count = scheduler.run_steps(5)

        assert count == 5
        assert scheduler.processed_count == 5

    def test_reset_simulation(self):
        """Test resetting simulation to initial state."""
        scheduler = EventScheduler(start_time=0.0)
        scheduler.register_agent("agent_1", tick_interval=1.0)

        # Run some steps
        scheduler.run_steps(10)
        assert scheduler.current_time == 9.0
        assert scheduler.processed_count == 10

        # Reset
        scheduler.reset(start_time=0.0)

        assert scheduler.current_time == 0.0
        assert scheduler.processed_count == 0
        assert scheduler.pending_count == 1  # First tick re-scheduled


class TestDeterministicReplay:
    """Test deterministic replay capability."""

    def test_same_events_same_order(self):
        """Test that running same scenario produces same event order."""
        def run_scenario():
            scheduler = EventScheduler(start_time=0.0)
            scheduler.register_agent("fast", tick_interval=1.0)
            scheduler.register_agent("slow", tick_interval=3.0)

            events = []

            def handler(event, sched):
                events.append((event.timestamp, event.agent_id))

            scheduler.set_handler(EventType.AGENT_TICK, handler)
            scheduler.run_until(t_end=6.0)

            return events

        events_run1 = run_scenario()
        events_run2 = run_scenario()

        assert events_run1 == events_run2

    def test_sequence_numbers_ensure_determinism(self):
        """Test that sequence numbers ensure deterministic ordering."""
        scheduler = EventScheduler(start_time=0.0)

        # Schedule many events at same time
        for i in range(10):
            scheduler.schedule(Event(
                timestamp=1.0,
                priority=0,
                event_type=EventType.CUSTOM,
                payload={"id": i},
            ))

        # Events should be processed in sequence order (order of scheduling)
        processed_ids = []

        def handler(event, sched):
            processed_ids.append(event.payload["id"])

        scheduler.set_handler(EventType.CUSTOM, handler)
        scheduler.run_until(t_end=2.0)

        assert processed_ids == list(range(10))


class TestAgentUnregistration:
    """Test dynamic agent unregistration."""

    def test_unregister_stops_future_ticks(self):
        """Test that unregistering agent stops its future ticks."""
        scheduler = EventScheduler(start_time=0.0)

        scheduler.register_agent("agent_1", tick_interval=1.0)

        tick_times = []

        def handler(event, sched):
            tick_times.append(event.timestamp)
            # Unregister after t=2
            if event.timestamp >= 2.0:
                sched.unregister_agent("agent_1")

        scheduler.set_handler(EventType.AGENT_TICK, handler)

        scheduler.run_until(t_end=10.0)

        # Should only have ticks at t=0,1,2, then stop
        assert tick_times == [0.0, 1.0, 2.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

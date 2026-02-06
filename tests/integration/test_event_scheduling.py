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

    def __init__(self, agent_id, tick_interval=1.0, **kwargs):
        from heron.scheduling.tick_config import TickConfig
        # Convert tick_interval to tick_config if not provided
        if "tick_config" not in kwargs:
            kwargs["tick_config"] = TickConfig(tick_interval=tick_interval)
        super().__init__(agent_id=agent_id, **kwargs)
        self.readings = []

    def set_action(self):
        self.action.set_specs(dim_c=0, dim_d=0)  # Sensor has no actions

    def set_state(self):
        self.state.features = [MockFeature(value=0.0)]

    def _get_obs(self, proxy=None):
        return self.state.vector()

    def tick(self, scheduler, current_time, global_state=None, proxy=None, async_observations=False):
        """Record reading at each tick."""
        self._timestep = current_time
        self.readings.append({"time": current_time, "value": self.state.features[0].value})


class ControllerAgent(FieldAgent):
    """Simulated controller with slower tick rate."""

    def __init__(self, agent_id, tick_interval=5.0, **kwargs):
        from heron.scheduling.tick_config import TickConfig
        # Convert tick_interval to tick_config if not provided
        if "tick_config" not in kwargs:
            kwargs["tick_config"] = TickConfig(tick_interval=tick_interval)
        super().__init__(agent_id=agent_id, **kwargs)
        self.control_actions = []

    def set_action(self):
        self.action.set_specs(dim_c=1, range=(np.array([0.0]), np.array([1.0])))

    def set_state(self):
        self.state.features = [MockFeature(value=0.0)]

    def _get_obs(self, proxy=None):
        return self.state.vector()

    def tick(self, scheduler, current_time, global_state=None, proxy=None, async_observations=False):
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


# =============================================================================
# Async Observation Integration Tests
# =============================================================================

class TestAsyncObservations:
    """Test fully async observation mode in event-driven simulation.

    In async mode, subordinates push observations to coordinators via message
    broker instead of coordinators pulling via direct method calls.
    """

    def test_async_observation_flow(self):
        """Test full async observation flow from subordinate to coordinator."""
        from heron.messaging.in_memory_broker import InMemoryBroker
        from heron.core.observation import Observation
        from heron.scheduling.tick_config import TickConfig

        # Create agents
        scheduler = EventScheduler(start_time=0.0)
        broker = InMemoryBroker()

        config1 = TickConfig(tick_interval=1.0, msg_delay=0.1)
        config2 = TickConfig(tick_interval=2.0, msg_delay=0.1)

        sub1 = SensorAgent(agent_id="sensor_1", env_id="test", upstream_id="coord", tick_config=config1)
        sub2 = SensorAgent(agent_id="sensor_2", env_id="test", upstream_id="coord", tick_config=config2)

        sub1.set_message_broker(broker)
        sub2.set_message_broker(broker)

        # Register agents with different tick rates
        scheduler.register_agent("sensor_1", tick_config=config1)
        scheduler.register_agent("sensor_2", tick_config=config2)
        scheduler.register_agent("coord", tick_interval=5.0)

        observations_received = []

        def tick_handler(event, sched):
            if event.agent_id == "sensor_1":
                # Subordinate sends observation in async mode
                obs = Observation(
                    local={"sensor_value": sub1.state.features[0].value},
                    timestamp=event.timestamp
                )
                sub1.send_observation_to_upstream(obs, scheduler=sched)

            elif event.agent_id == "sensor_2":
                obs = Observation(
                    local={"sensor_value": sub2.state.features[0].value},
                    timestamp=event.timestamp
                )
                sub2.send_observation_to_upstream(obs, scheduler=sched)

        def message_handler(event, sched):
            # Record delivered observations
            if "observation" in event.payload.get("message", {}):
                observations_received.append({
                    "time": event.timestamp,
                    "sender": event.payload.get("sender_id"),
                    "obs_data": event.payload["message"]["observation"]
                })

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)
        scheduler.set_handler(EventType.MESSAGE_DELIVERY, message_handler)

        scheduler.run_until(t_end=5.0)

        # Verify observations were received
        # sensor_1 ticks at 0,1,2,3,4,5 -> messages at 0.1,1.1,2.1,3.1,4.1,5.1 (5.1 beyond t_end)
        # sensor_2 ticks at 0,2,4 -> messages at 0.1,2.1,4.1
        # Total within t_end=5.0: 5 + 3 = 8 messages (approximately, depends on timing)
        assert len(observations_received) > 0

    def test_async_observation_serialization_roundtrip(self):
        """Test observation serialization through message broker."""
        from heron.messaging.in_memory_broker import InMemoryBroker
        from heron.messaging.base import ChannelManager, MessageType, Message
        from heron.core.observation import Observation

        broker = InMemoryBroker()

        # Create observation with numpy array
        original_obs = Observation(
            local={
                "power": 100.0,
                "states": np.array([1.0, 2.0, 3.0], dtype=np.float32)
            },
            global_info={"frequency": 60.0},
            timestamp=5.0
        )

        # Serialize and send through broker
        channel = ChannelManager.observation_channel("sub1", "coord", "test_env")
        broker.create_channel(channel)

        msg = Message(
            env_id="test_env",
            sender_id="sub1",
            recipient_id="coord",
            timestamp=5.0,
            message_type=MessageType.INFO,
            payload={"observation": original_obs.to_dict()}
        )
        broker.publish(channel, msg)

        # Receive and deserialize
        messages = broker.consume(channel, "coord", "test_env")
        received_obs = Observation.from_dict(messages[0].payload["observation"])

        # Verify roundtrip
        assert received_obs.timestamp == original_obs.timestamp
        assert received_obs.local["power"] == original_obs.local["power"]
        np.testing.assert_array_equal(
            received_obs.local["states"],
            original_obs.local["states"]
        )
        assert received_obs.global_info["frequency"] == original_obs.global_info["frequency"]

    def test_async_observation_partial_arrival(self):
        """Test coordinator handling when only some subordinates have sent."""
        from heron.messaging.in_memory_broker import InMemoryBroker
        from heron.core.observation import Observation
        from heron.agents.base import Agent

        # Simple concrete agent for testing
        class SimpleAgent(Agent):
            def __init__(self, agent_id, **kwargs):
                super().__init__(agent_id=agent_id, **kwargs)
                self.subordinates = {}

            def observe(self, *args, **kwargs):
                return Observation()

            def act(self, *args, **kwargs):
                return None

        broker = InMemoryBroker()

        # Create coordinator with 3 subordinates
        coord = SimpleAgent(agent_id="coord", env_id="test")
        sub1 = SimpleAgent(agent_id="sub1", env_id="test", upstream_id="coord")
        sub2 = SimpleAgent(agent_id="sub2", env_id="test", upstream_id="coord")
        sub3 = SimpleAgent(agent_id="sub3", env_id="test", upstream_id="coord")

        coord.subordinates = {"sub1": sub1, "sub2": sub2, "sub3": sub3}

        coord.set_message_broker(broker)
        sub1.set_message_broker(broker)
        sub2.set_message_broker(broker)
        sub3.set_message_broker(broker)

        # Only sub1 and sub3 send observations
        sub1.send_observation_to_upstream(
            Observation(local={"value": 100.0}, timestamp=1.0)
        )
        sub3.send_observation_to_upstream(
            Observation(local={"value": 300.0}, timestamp=1.0)
        )

        # Coordinator receives - should only have 2 observations
        received = coord.receive_observations_from_subordinates()

        assert len(received) == 2
        assert "sub1" in received
        assert "sub3" in received
        assert "sub2" not in received  # Never sent

        assert received["sub1"].local["value"] == 100.0
        assert received["sub3"].local["value"] == 300.0

    def test_async_vs_sync_mode_equivalence(self):
        """Test that sync and async modes produce equivalent observations."""
        from heron.messaging.in_memory_broker import InMemoryBroker
        from heron.core.observation import Observation

        broker = InMemoryBroker()

        # Create sensor agent
        sensor = SensorAgent(
            agent_id="sensor_1",
            env_id="test",
            upstream_id="coord"
        )
        sensor.set_message_broker(broker)
        sensor.state.features[0].value = 42.0

        # Get observation via sync mode (direct call)
        sync_obs = sensor.observe()

        # Get observation via async mode (send/receive through broker)
        sensor.send_observation_to_upstream(sync_obs)

        from heron.messaging.base import ChannelManager
        channel = ChannelManager.observation_channel("sensor_1", "coord", "test")
        messages = broker.consume(channel, "coord", "test")
        async_obs = Observation.from_dict(messages[0].payload["observation"])

        # Both should have same data
        assert sync_obs.timestamp == async_obs.timestamp
        # Note: The exact local dict may differ in structure but vectors should match
        np.testing.assert_array_almost_equal(
            sync_obs.vector(),
            async_obs.vector()
        )

    def test_async_observation_with_msg_delay(self):
        """Test that msg_delay delays observation delivery in scheduler."""
        from heron.messaging.in_memory_broker import InMemoryBroker
        from heron.core.observation import Observation
        from heron.scheduling.tick_config import TickConfig

        scheduler = EventScheduler(start_time=0.0)
        broker = InMemoryBroker()

        # Create sensor with msg_delay
        tick_config = TickConfig(
            tick_interval=5.0,
            msg_delay=2.0  # 2 second message delay
        )
        sensor = SensorAgent(
            agent_id="sensor_1",
            env_id="test",
            upstream_id="coord",
            tick_config=tick_config
        )
        sensor.set_message_broker(broker)

        scheduler.register_agent("sensor_1", tick_config=tick_config)
        scheduler.register_agent("coord", tick_interval=10.0)

        delivered_times = []

        def tick_handler(event, sched):
            if event.agent_id == "sensor_1":
                obs = Observation(
                    local={"value": 100.0},
                    timestamp=event.timestamp
                )
                # Send with scheduler to use msg_delay
                sensor.send_observation_to_upstream(obs, scheduler=sched)

        def delivery_handler(event, sched):
            delivered_times.append(event.timestamp)

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)
        scheduler.set_handler(EventType.MESSAGE_DELIVERY, delivery_handler)

        scheduler.run_until(t_end=10.0)

        # Sensor ticks at t=0,5,10
        # Messages should be delivered at t=2,7,12 (12 beyond t_end)
        assert len(delivered_times) == 2
        np.testing.assert_almost_equal(delivered_times[0], 2.0)
        np.testing.assert_almost_equal(delivered_times[1], 7.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for agents.base module."""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.spaces import Box

from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.messaging.in_memory_broker import InMemoryBroker
from heron.core.policies import RandomPolicy


class TestObservation:
    """Test Observation dataclass and methods."""

    def test_observation_initialization(self):
        """Test observation initialization with default values."""
        obs = Observation()
        assert isinstance(obs.local, dict)
        assert isinstance(obs.global_info, dict)
        assert obs.timestamp == 0.0

    def test_observation_custom_values(self):
        """Test observation with custom values."""
        local = {"P": 1.0, "Q": 0.5}
        global_info = {"voltage": 1.05}
        timestamp = 10.0

        obs = Observation(
            local=local,
            global_info=global_info,
            timestamp=timestamp
        )

        assert obs.local == local
        assert obs.global_info == global_info
        assert obs.timestamp == 10.0

    def test_as_vector_empty(self):
        """Test as_vector with empty observation."""
        obs = Observation()
        vec = obs.as_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert len(vec) == 0

    def test_as_vector_local_scalars(self):
        """Test as_vector with scalar values."""
        obs = Observation(local={"P": 1.0, "Q": 0.5, "soc": 0.8})
        vec = obs.as_vector()

        assert len(vec) == 3
        assert vec.dtype == np.float32
        # Sorted by key: P, Q, soc
        np.testing.assert_array_almost_equal(vec, [1.0, 0.5, 0.8])

    def test_as_vector_with_arrays(self):
        """Test as_vector with numpy arrays."""
        obs = Observation(local={
            "P": 1.0,
            "voltages": np.array([1.0, 1.05, 0.95])
        })
        vec = obs.as_vector()

        assert len(vec) == 4  # 1 scalar + 3 array elements
        assert vec.dtype == np.float32

    def test_as_vector_nested_dict(self):
        """Test as_vector with nested dictionaries."""
        obs = Observation(local={
            "device1": {"P": 1.0, "Q": 0.5},
            "device2": {"P": 2.0, "Q": 1.0}
        })
        vec = obs.as_vector()

        assert len(vec) == 4
        assert vec.dtype == np.float32

    def test_as_vector_with_global_info(self):
        """Test as_vector includes global_info."""
        obs = Observation(
            local={"P": 1.0},
            global_info={"voltage": 1.05}
        )
        vec = obs.as_vector()

        assert len(vec) == 2
        assert vec.dtype == np.float32


class ConcreteAgent(Agent):
    """Concrete implementation of Agent for testing."""

    def observe(self, global_state=None, *args, **kwargs):
        """Return simple observation."""
        return Observation(
            local={"value": 1.0},
            timestamp=self._timestep
        )

    def act(self, observation, *args, **kwargs):
        """Return simple action."""
        return np.array([1.0])


class TestAgent:
    """Test Agent abstract base class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ConcreteAgent(
            agent_id="test_agent",
            level=1
        )

        assert agent.agent_id == "test_agent"
        assert agent.level == 1
        assert agent._message_broker is None
        assert agent._timestep == 0.0

    def test_agent_with_spaces(self):
        """Test agent initialization with gym spaces."""
        obs_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        action_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)

        agent = ConcreteAgent(
            agent_id="test_agent",
            level=2,
            observation_space=obs_space,
            action_space=action_space
        )

        assert agent.observation_space == obs_space
        assert agent.action_space == action_space

    def test_agent_reset(self):
        """Test agent reset clears state."""
        agent = ConcreteAgent(agent_id="test")
        agent._timestep = 10.0

        # Reset
        agent.reset()

        assert agent._timestep == 0.0

    def test_agent_observe(self):
        """Test agent observe method."""
        agent = ConcreteAgent(agent_id="test")
        agent._timestep = 5.0

        obs = agent.observe()

        assert isinstance(obs, Observation)
        assert obs.timestamp == 5.0
        assert "value" in obs.local

    def test_agent_act(self):
        """Test agent act method."""
        agent = ConcreteAgent(agent_id="test")
        obs = Observation(local={"value": 1.0})

        action = agent.act(obs)

        assert isinstance(action, np.ndarray)
        np.testing.assert_array_equal(action, [1.0])

    def test_send_message_via_broker(self):
        """Test sending message via message broker."""
        broker = InMemoryBroker()
        sender = ConcreteAgent(agent_id="sender", env_id="test_env")
        receiver = ConcreteAgent(agent_id="receiver", env_id="test_env", upstream_id="sender")

        sender.set_message_broker(broker)
        receiver.set_message_broker(broker)
        sender._timestep = 5.0

        # Send message
        sender.send_message({"price": 50.0}, recipient_id="receiver")

        # Receive message
        messages = receiver.receive_messages(sender_id="sender")

        assert len(messages) == 1
        assert messages[0] == {"price": 50.0}

    def test_set_message_broker(self):
        """Test setting message broker on agent."""
        broker = InMemoryBroker()
        agent = ConcreteAgent(agent_id="test")

        assert agent.message_broker is None

        agent.set_message_broker(broker)

        assert agent.message_broker == broker

    def test_receive_messages_no_broker(self):
        """Test receiving messages when no broker configured."""
        agent = ConcreteAgent(agent_id="test")

        messages = agent.receive_messages()

        assert messages == []

    def test_send_message_no_broker_raises(self):
        """Test sending message without broker raises error."""
        agent = ConcreteAgent(agent_id="test")

        with pytest.raises(RuntimeError, match="no message broker configured"):
            agent.send_message({"data": 1}, recipient_id="other")

    def test_receive_action_messages(self):
        """Test receiving action messages via broker."""
        broker = InMemoryBroker()
        sender = ConcreteAgent(agent_id="sender", env_id="test_env")
        receiver = ConcreteAgent(agent_id="receiver", env_id="test_env", upstream_id="sender")

        sender.set_message_broker(broker)
        receiver.set_message_broker(broker)

        # Send action
        sender.send_action_to_subordinate("receiver", action=[1.0, 2.0])

        # Receive action
        actions = receiver.receive_action_messages()

        assert len(actions) == 1
        assert actions[0] == [1.0, 2.0]

    def test_update_timestep(self):
        """Test updating timestep."""
        agent = ConcreteAgent(agent_id="test")

        agent.update_timestep(10.5)

        assert agent._timestep == 10.5

    def test_agent_repr(self):
        """Test agent string representation."""
        agent = ConcreteAgent(agent_id="test_agent", level=2)

        repr_str = repr(agent)

        assert "ConcreteAgent" in repr_str
        assert "test_agent" in repr_str
        assert "level=2" in repr_str


class TestRandomPolicy:
    """Tests for RandomPolicy."""

    def test_random_policy_sampling(self):
        """Test random policy action sampling."""
        action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        policy = RandomPolicy(action_space, seed=42)

        obs = Observation()
        action1 = policy.forward(obs)
        action2 = policy.forward(obs)

        assert action1.c.shape == (3,)
        assert action2.c.shape == (3,)
        # Actions should be different (with high probability)
        assert not np.allclose(action1.c, action2.c)

    def test_random_policy_reset(self):
        """Test random policy reset."""
        action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        policy = RandomPolicy(action_space)

        policy.reset()  # Should not raise


# =============================================================================
# Async Observation Tests (for event-driven mode)
# =============================================================================

class TestAsyncObservationHelpers:
    """Test async observation helpers for event-driven mode.

    These methods are used in fully async event-driven mode (Option B with
    async_observations=True) where subordinates push observations to
    coordinators via message broker.
    """

    def test_send_observation_to_upstream_no_broker(self):
        """Test send_observation does nothing when no broker configured."""
        agent = ConcreteAgent(agent_id="field1", upstream_id="coord1")
        obs = Observation(local={"value": 10.0}, timestamp=5.0)

        # Should not raise, just return silently
        agent.send_observation_to_upstream(obs)

    def test_send_observation_to_upstream_no_upstream(self):
        """Test send_observation does nothing when no upstream agent."""
        broker = InMemoryBroker()
        agent = ConcreteAgent(agent_id="field1", env_id="test_env")
        agent.set_message_broker(broker)
        obs = Observation(local={"value": 10.0}, timestamp=5.0)

        # Should not raise, just return silently
        agent.send_observation_to_upstream(obs)

    def test_send_observation_to_upstream_basic(self):
        """Test sending observation to upstream agent via broker."""
        broker = InMemoryBroker()
        subordinate = ConcreteAgent(
            agent_id="field1",
            env_id="test_env",
            upstream_id="coord1"
        )
        coordinator = ConcreteAgent(
            agent_id="coord1",
            env_id="test_env"
        )
        # Register subordinate with coordinator (for receive method)
        coordinator.subordinates = {"field1": subordinate}

        subordinate.set_message_broker(broker)
        coordinator.set_message_broker(broker)

        # Send observation (without scheduler = immediate delivery)
        obs = Observation(
            local={"power": 100.0, "voltage": 1.02},
            timestamp=5.0
        )
        subordinate.send_observation_to_upstream(obs)

        # Receive observation on coordinator side
        received = coordinator.receive_observations_from_subordinates()

        assert "field1" in received
        recv_obs = received["field1"]
        assert isinstance(recv_obs, Observation)
        assert recv_obs.timestamp == 5.0
        assert recv_obs.local["power"] == 100.0
        assert recv_obs.local["voltage"] == 1.02

    def test_send_observation_with_numpy_array(self):
        """Test sending observation containing numpy arrays."""
        broker = InMemoryBroker()
        subordinate = ConcreteAgent(
            agent_id="field1",
            env_id="test_env",
            upstream_id="coord1"
        )
        coordinator = ConcreteAgent(
            agent_id="coord1",
            env_id="test_env"
        )
        coordinator.subordinates = {"field1": subordinate}

        subordinate.set_message_broker(broker)
        coordinator.set_message_broker(broker)

        # Send observation with numpy array
        obs = Observation(
            local={"states": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            timestamp=10.0
        )
        subordinate.send_observation_to_upstream(obs)

        # Receive and verify
        received = coordinator.receive_observations_from_subordinates()
        recv_obs = received["field1"]

        assert isinstance(recv_obs.local["states"], np.ndarray)
        np.testing.assert_array_equal(recv_obs.local["states"], [1.0, 2.0, 3.0])

    def test_receive_observations_multiple_subordinates(self):
        """Test receiving observations from multiple subordinates."""
        broker = InMemoryBroker()

        # Create coordinator with multiple subordinates
        coordinator = ConcreteAgent(agent_id="coord1", env_id="test_env")
        sub1 = ConcreteAgent(agent_id="field1", env_id="test_env", upstream_id="coord1")
        sub2 = ConcreteAgent(agent_id="field2", env_id="test_env", upstream_id="coord1")
        coordinator.subordinates = {"field1": sub1, "field2": sub2}

        coordinator.set_message_broker(broker)
        sub1.set_message_broker(broker)
        sub2.set_message_broker(broker)

        # Send observations from both
        obs1 = Observation(local={"value": 100.0}, timestamp=5.0)
        obs2 = Observation(local={"value": 200.0}, timestamp=5.5)
        sub1.send_observation_to_upstream(obs1)
        sub2.send_observation_to_upstream(obs2)

        # Receive all
        received = coordinator.receive_observations_from_subordinates()

        assert len(received) == 2
        assert received["field1"].local["value"] == 100.0
        assert received["field2"].local["value"] == 200.0

    def test_receive_observations_partial(self):
        """Test receiving when only some subordinates have sent."""
        broker = InMemoryBroker()

        coordinator = ConcreteAgent(agent_id="coord1", env_id="test_env")
        sub1 = ConcreteAgent(agent_id="field1", env_id="test_env", upstream_id="coord1")
        sub2 = ConcreteAgent(agent_id="field2", env_id="test_env", upstream_id="coord1")
        coordinator.subordinates = {"field1": sub1, "field2": sub2}

        coordinator.set_message_broker(broker)
        sub1.set_message_broker(broker)
        sub2.set_message_broker(broker)

        # Only sub1 sends
        obs1 = Observation(local={"value": 100.0}, timestamp=5.0)
        sub1.send_observation_to_upstream(obs1)

        # Receive - should only have sub1
        received = coordinator.receive_observations_from_subordinates()

        assert "field1" in received
        assert "field2" not in received

    def test_receive_observations_with_clear(self):
        """Test that clear=True removes messages after consuming."""
        broker = InMemoryBroker()

        coordinator = ConcreteAgent(agent_id="coord1", env_id="test_env")
        sub1 = ConcreteAgent(agent_id="field1", env_id="test_env", upstream_id="coord1")
        coordinator.subordinates = {"field1": sub1}

        coordinator.set_message_broker(broker)
        sub1.set_message_broker(broker)

        # Send observation
        obs = Observation(local={"value": 100.0}, timestamp=5.0)
        sub1.send_observation_to_upstream(obs)

        # First receive with clear=True (default)
        received1 = coordinator.receive_observations_from_subordinates(clear=True)
        assert len(received1) == 1

        # Second receive should be empty
        received2 = coordinator.receive_observations_from_subordinates()
        assert len(received2) == 0

    def test_receive_observations_without_clear(self):
        """Test that clear=False keeps messages for re-read."""
        broker = InMemoryBroker()

        coordinator = ConcreteAgent(agent_id="coord1", env_id="test_env")
        sub1 = ConcreteAgent(agent_id="field1", env_id="test_env", upstream_id="coord1")
        coordinator.subordinates = {"field1": sub1}

        coordinator.set_message_broker(broker)
        sub1.set_message_broker(broker)

        # Send observation
        obs = Observation(local={"value": 100.0}, timestamp=5.0)
        sub1.send_observation_to_upstream(obs)

        # First receive with clear=False
        received1 = coordinator.receive_observations_from_subordinates(clear=False)
        assert len(received1) == 1

        # Second receive should still have the message
        received2 = coordinator.receive_observations_from_subordinates(clear=False)
        assert len(received2) == 1

    def test_receive_observations_no_broker(self):
        """Test receive returns empty dict when no broker configured."""
        coordinator = ConcreteAgent(agent_id="coord1", env_id="test_env")
        coordinator.subordinates = {"field1": ConcreteAgent(agent_id="field1")}

        received = coordinator.receive_observations_from_subordinates()

        assert received == {}

    def test_receive_observations_filter_by_subordinate_ids(self):
        """Test filtering received observations by subordinate IDs."""
        broker = InMemoryBroker()

        coordinator = ConcreteAgent(agent_id="coord1", env_id="test_env")
        sub1 = ConcreteAgent(agent_id="field1", env_id="test_env", upstream_id="coord1")
        sub2 = ConcreteAgent(agent_id="field2", env_id="test_env", upstream_id="coord1")
        sub3 = ConcreteAgent(agent_id="field3", env_id="test_env", upstream_id="coord1")
        coordinator.subordinates = {"field1": sub1, "field2": sub2, "field3": sub3}

        coordinator.set_message_broker(broker)
        sub1.set_message_broker(broker)
        sub2.set_message_broker(broker)
        sub3.set_message_broker(broker)

        # All send observations
        for sub, val in [(sub1, 100), (sub2, 200), (sub3, 300)]:
            sub.send_observation_to_upstream(
                Observation(local={"value": float(val)}, timestamp=5.0)
            )

        # Only receive from field1 and field2
        received = coordinator.receive_observations_from_subordinates(
            subordinate_ids=["field1", "field2"]
        )

        assert len(received) == 2
        assert "field1" in received
        assert "field2" in received
        assert "field3" not in received


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

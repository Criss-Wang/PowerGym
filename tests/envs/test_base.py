"""Tests for heron.envs.base module."""

import pytest
import numpy as np
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

from heron.envs.base import EnvCore, MultiAgentEnv
from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.messaging.in_memory_broker import InMemoryBroker


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id=agent_id, level=1, **kwargs)

    def observe(self, global_state=None, *args, **kwargs):
        return Observation(
            local={"value": 1.0},
            timestamp=self._timestep,
        )

    def act(self, observation, *args, **kwargs):
        return None


class MockCoordinator(CoordinatorAgent):
    """Mock coordinator for testing."""

    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id=agent_id, config={}, **kwargs)

    def _build_subordinates(self, agent_configs, env_id=None, upstream_id=None):
        return {}


class ConcreteMultiAgentEnv(MultiAgentEnv):
    """Concrete implementation of MultiAgentEnv for testing."""

    def reset(self, *, seed=None, options=None):
        return {}, {}

    def step(self, actions):
        return {}, {}, {"__all__": False}, {"__all__": False}, {}


class TestEnvCoreInitialization:
    """Test EnvCore initialization."""

    def test_init_generates_env_id(self):
        """Test that env_id is auto-generated."""
        env = ConcreteMultiAgentEnv()

        assert env.env_id is not None
        assert env.env_id.startswith("env_")

    def test_init_with_custom_env_id(self):
        """Test initialization with custom env_id."""
        env = ConcreteMultiAgentEnv(env_id="my_env")

        assert env.env_id == "my_env"

    def test_init_creates_message_broker(self):
        """Test that message broker is created by default."""
        env = ConcreteMultiAgentEnv()

        assert env.message_broker is not None
        assert isinstance(env.message_broker, InMemoryBroker)

    def test_init_with_custom_broker(self):
        """Test initialization with custom message broker."""
        broker = InMemoryBroker()
        env = ConcreteMultiAgentEnv(message_broker=broker)

        assert env.message_broker is broker

    def test_init_empty_agent_dicts(self):
        """Test that agent dictionaries are initialized empty."""
        env = ConcreteMultiAgentEnv()

        assert env.registered_agents == {}
        assert env.registered_coordinators == {}


class TestEnvCoreAgentManagement:
    """Test agent management functionality."""

    def test_register_agent(self):
        """Test registering a single agent."""
        env = ConcreteMultiAgentEnv()
        agent = MockAgent(agent_id="agent_1")

        env.register_agent(agent)

        assert "agent_1" in env.registered_agents
        assert env.registered_agents["agent_1"] is agent

    def test_register_coordinator_agent(self):
        """Test registering a coordinator agent."""
        env = ConcreteMultiAgentEnv()
        coord = MockCoordinator(agent_id="coord_1")

        env.register_agent(coord)

        assert "coord_1" in env.registered_agents
        assert "coord_1" in env.registered_coordinators

    def test_register_agents_multiple(self):
        """Test registering multiple agents."""
        env = ConcreteMultiAgentEnv()
        agents = [
            MockAgent(agent_id="agent_1"),
            MockAgent(agent_id="agent_2"),
            MockAgent(agent_id="agent_3"),
        ]

        env.register_agents(agents)

        assert len(env.registered_agents) == 3

    def test_get_agent(self):
        """Test getting agent by ID."""
        env = ConcreteMultiAgentEnv()
        agent = MockAgent(agent_id="agent_1")
        env.register_agent(agent)

        retrieved = env.get_agent("agent_1")

        assert retrieved is agent

    def test_get_agent_missing(self):
        """Test getting non-existent agent returns None."""
        env = ConcreteMultiAgentEnv()

        result = env.get_agent("nonexistent")

        assert result is None


class TestEnvCoreObservations:
    """Test observation collection."""

    def test_get_observations(self):
        """Test collecting observations from all agents."""
        env = ConcreteMultiAgentEnv()
        env.register_agents([
            MockAgent(agent_id="agent_1"),
            MockAgent(agent_id="agent_2"),
        ])

        obs = env.get_observations()

        assert len(obs) == 2
        assert "agent_1" in obs
        assert "agent_2" in obs
        assert isinstance(obs["agent_1"], Observation)


class TestEnvCoreApplyActions:
    """Test action application."""

    def test_apply_actions(self):
        """Test applying actions to agents."""
        env = ConcreteMultiAgentEnv()
        env.register_agents([
            MockAgent(agent_id="agent_1"),
            MockAgent(agent_id="agent_2"),
        ])

        actions = {
            "agent_1": np.array([0.5]),
            "agent_2": np.array([0.3]),
        }

        # Should not raise
        env.apply_actions(actions)


class TestEnvCoreSpaces:
    """Test action/observation space methods."""

    def test_get_agent_action_spaces(self):
        """Test getting agent action spaces."""
        env = ConcreteMultiAgentEnv()
        agent = MockAgent(agent_id="agent_1")
        agent.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        env.register_agent(agent)

        spaces = env.get_agent_action_spaces()

        assert "agent_1" in spaces
        assert spaces["agent_1"] is agent.action_space

    def test_get_agent_observation_spaces(self):
        """Test getting agent observation spaces."""
        env = ConcreteMultiAgentEnv()
        agent = MockAgent(agent_id="agent_1")
        agent.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        env.register_agent(agent)

        spaces = env.get_agent_observation_spaces()

        assert "agent_1" in spaces


class TestEnvCoreResetAgents:
    """Test agent reset functionality."""

    def test_reset_agents(self):
        """Test resetting all agents."""
        env = ConcreteMultiAgentEnv()
        agents = [
            MockAgent(agent_id="agent_1"),
            MockAgent(agent_id="agent_2"),
        ]
        env.register_agents(agents)

        # Set timesteps
        for agent in agents:
            agent._timestep = 100.0

        env.reset_agents()

        for agent in env.registered_agents.values():
            assert agent._timestep == 0.0


class TestEnvCoreConfigureDistributed:
    """Test configure_agents_for_distributed method."""

    def test_configure_agents_for_distributed(self):
        """Test configuring agents with message broker."""
        env = ConcreteMultiAgentEnv()
        agents = [
            MockAgent(agent_id="agent_1"),
            MockAgent(agent_id="agent_2"),
        ]
        env.register_agents(agents)

        env.configure_agents_for_distributed()

        for agent in env.registered_agents.values():
            assert agent.message_broker is env.message_broker
            assert agent.env_id == env.env_id


class TestEnvCoreSimulationTime:
    """Test simulation time property."""

    def test_simulation_time_no_scheduler(self):
        """Test simulation time without scheduler."""
        env = ConcreteMultiAgentEnv()
        env._timestep = 5

        assert env.simulation_time == 5.0


class TestEnvCoreBrokerMethods:
    """Test message broker methods."""

    def test_setup_broker_channels(self):
        """Test setting up broker channels."""
        env = ConcreteMultiAgentEnv()
        agent = MockAgent(agent_id="agent_1", upstream_id="coord_1")
        env.register_agent(agent)

        # Should not raise
        env.setup_broker_channels()

    def test_publish_action(self):
        """Test publishing action via broker."""
        env = ConcreteMultiAgentEnv()

        # Should not raise
        env.publish_action(
            sender_id="coord_1",
            recipient_id="agent_1",
            action=np.array([0.5]),
        )

    def test_publish_info(self):
        """Test publishing info via broker."""
        env = ConcreteMultiAgentEnv()

        # Should not raise
        env.publish_info(
            sender_id="agent_1",
            recipient_id="coord_1",
            info={"power": 100},
        )

    def test_publish_state_update(self):
        """Test publishing state update."""
        env = ConcreteMultiAgentEnv()

        # Should not raise
        env.publish_state_update({"total_load": 1000})

    def test_clear_broker_environment(self):
        """Test clearing broker environment."""
        env = ConcreteMultiAgentEnv()
        env.publish_state_update({"test": True})

        env.clear_broker_environment()

        # Should not raise


class TestEnvCoreClose:
    """Test close functionality."""

    def test_close_core(self):
        """Test closing core resources."""
        env = ConcreteMultiAgentEnv()

        # Should not raise
        env.close_core()


class TestMultiAgentEnv:
    """Test MultiAgentEnv class."""

    def test_multi_agent_env_initialization(self):
        """Test MultiAgentEnv initialization."""
        env = ConcreteMultiAgentEnv(env_id="multi_env")

        assert env.env_id == "multi_env"

    def test_multi_agent_env_reset(self):
        """Test MultiAgentEnv reset."""
        env = ConcreteMultiAgentEnv()

        obs, info = env.reset()

        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_multi_agent_env_step(self):
        """Test MultiAgentEnv step."""
        env = ConcreteMultiAgentEnv()
        env.reset()

        obs, rewards, terminated, truncated, info = env.step({})

        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)

    def test_multi_agent_env_close(self):
        """Test MultiAgentEnv close."""
        env = ConcreteMultiAgentEnv()

        # Should not raise
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

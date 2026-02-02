"""Tests for CoordinatorAgent module."""

import pytest
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from heron.agents.coordinator_agent import CoordinatorAgent, COORDINATOR_LEVEL
from heron.agents.field_agent import FieldAgent
from heron.core.observation import Observation
from heron.core.policies import Policy, RandomPolicy
from heron.core.action import Action


class MockFieldAgent(FieldAgent):
    """Mock field agent for testing."""

    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        self.action.set_specs(dim_c=2, range=(np.array([-1, -1]), np.array([1, 1])))

    def set_action(self):
        self.action.set_specs(dim_c=2, range=(np.array([-1, -1]), np.array([1, 1])))

    def set_state(self):
        pass

    def _get_obs(self):
        """Return mock observation vector."""
        return np.array([0.5, 0.5], dtype=np.float32)


class MockPolicy(Policy):
    """Mock policy that returns fixed action."""

    def __init__(self, action_value=None):
        self.action_value = action_value if action_value is not None else np.array([0.5, 0.5])
        self.reset_called = False

    def forward(self, observation: Observation):
        action = Action()
        action.set_specs(dim_c=len(self.action_value), range=(np.full(len(self.action_value), -1), np.full(len(self.action_value), 1)))
        action.set_values(c=self.action_value)
        return action

    def reset(self):
        self.reset_called = True


class ConcreteCoordinatorAgent(CoordinatorAgent):
    """Concrete implementation for testing."""

    def _build_subordinate_agents(self, agent_configs, env_id=None, upstream_id=None):
        agents = {}
        for config in agent_configs:
            agent_id = config.get("id", f"agent_{len(agents)}")
            agents[agent_id] = MockFieldAgent(
                agent_id=agent_id,
                env_id=env_id,
                upstream_id=upstream_id,
            )
        return agents


class TestCoordinatorAgentInitialization:
    """Test CoordinatorAgent initialization."""

    def test_basic_initialization(self):
        """Test basic coordinator initialization."""
        coord = ConcreteCoordinatorAgent(agent_id="coord_1")

        assert coord.agent_id == "coord_1"
        assert coord.level == COORDINATOR_LEVEL
        assert coord.subordinate_agents == {}
        assert coord.protocol is None
        assert coord.policy is None

    def test_initialization_with_config(self):
        """Test initialization with agent configs."""
        config = {
            "agents": [
                {"id": "field_1"},
                {"id": "field_2"},
            ]
        }
        coord = ConcreteCoordinatorAgent(
            agent_id="coord_1",
            config=config,
            env_id="test_env",
        )

        assert len(coord.subordinate_agents) == 2
        assert "field_1" in coord.subordinate_agents
        assert "field_2" in coord.subordinate_agents

    def test_initialization_with_policy(self):
        """Test initialization with policy."""
        policy = MockPolicy()
        coord = ConcreteCoordinatorAgent(
            agent_id="coord_1",
            policy=policy,
        )

        assert coord.policy is policy

    def test_initialization_with_timing_params(self):
        """Test initialization with timing parameters."""
        coord = ConcreteCoordinatorAgent(
            agent_id="coord_1",
            tick_interval=120.0,
            obs_delay=1.0,
            act_delay=2.0,
            msg_delay=0.5,
        )

        assert coord.tick_interval == 120.0
        assert coord.obs_delay == 1.0
        assert coord.act_delay == 2.0
        assert coord.msg_delay == 0.5


class TestCoordinatorAgentReset:
    """Test CoordinatorAgent reset functionality."""

    def test_reset_basic(self):
        """Test basic reset."""
        config = {"agents": [{"id": "field_1"}]}
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", config=config)
        coord._timestep = 100.0

        coord.reset()

        assert coord._timestep == 0.0

    def test_reset_resets_subordinates(self):
        """Test that reset resets subordinate agents."""
        config = {"agents": [{"id": "field_1"}, {"id": "field_2"}]}
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", config=config)

        # Set timesteps on subordinates
        for agent in coord.subordinate_agents.values():
            agent._timestep = 50.0

        coord.reset()

        for agent in coord.subordinate_agents.values():
            assert agent._timestep == 0.0

    def test_reset_resets_policy(self):
        """Test that reset resets policy."""
        policy = MockPolicy()
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", policy=policy)

        coord.reset()

        assert policy.reset_called


class TestCoordinatorAgentObserve:
    """Test CoordinatorAgent observe functionality."""

    def test_observe_empty_subordinates(self):
        """Test observation with no subordinates."""
        coord = ConcreteCoordinatorAgent(agent_id="coord_1")
        coord._timestep = 5.0

        obs = coord.observe()

        assert isinstance(obs, Observation)
        assert obs.timestamp == 5.0
        assert "subordinate_obs" in obs.local

    def test_observe_with_subordinates(self):
        """Test observation aggregates subordinate observations."""
        config = {"agents": [{"id": "field_1"}, {"id": "field_2"}]}
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", config=config)

        obs = coord.observe()

        subordinate_obs = obs.local.get("subordinate_obs", {})
        assert "field_1" in subordinate_obs
        assert "field_2" in subordinate_obs

    def test_observe_with_global_state(self):
        """Test observation includes global state."""
        coord = ConcreteCoordinatorAgent(agent_id="coord_1")
        global_state = {"voltage": 1.05, "frequency": 60.0}

        obs = coord.observe(global_state=global_state)

        assert obs.global_info == global_state


class TestCoordinatorAgentAct:
    """Test CoordinatorAgent act functionality."""

    def test_act_with_upstream_action(self):
        """Test act with upstream action."""
        config = {"agents": [{"id": "field_1"}]}
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", config=config)

        obs = coord.observe()
        upstream_action = {"field_1": np.array([0.5, 0.5])}

        # Should not raise
        coord.act(obs, upstream_action=upstream_action)

    def test_act_with_policy(self):
        """Test act using policy."""
        config = {"agents": [{"id": "field_1"}]}
        policy = MockPolicy(action_value=np.array([0.3, 0.3]))
        coord = ConcreteCoordinatorAgent(
            agent_id="coord_1",
            config=config,
            policy=policy,
        )

        obs = coord.observe()
        coord.act(obs)  # Should use policy

    def test_act_without_action_or_policy_raises(self):
        """Test that act raises error without action or policy."""
        coord = ConcreteCoordinatorAgent(agent_id="coord_1")

        obs = coord.observe()

        with pytest.raises(RuntimeError, match="No action or policy"):
            coord.act(obs)


class TestCoordinatorAgentActionDistribution:
    """Test action distribution functionality."""

    def test_simple_action_distribution_dict(self):
        """Test action distribution with dict action."""
        config = {"agents": [{"id": "field_1"}, {"id": "field_2"}]}
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", config=config)

        obs = coord.observe()
        action = {"field_1": np.array([0.5, 0.5]), "field_2": np.array([0.3, 0.3])}

        result = coord._simple_action_distribution(action, obs.local.get("subordinate_obs", {}))

        assert result == action

    def test_simple_action_distribution_array(self):
        """Test action distribution with flat array."""
        config = {"agents": [{"id": "field_1"}, {"id": "field_2"}]}
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", config=config)

        obs = coord.observe()
        action = np.array([0.1, 0.2, 0.3, 0.4])

        result = coord._simple_action_distribution(action, obs.local.get("subordinate_obs", {}))

        assert len(result) == 2


class TestCoordinatorAgentSpaces:
    """Test action/observation space construction."""

    def test_get_subordinate_action_spaces(self):
        """Test getting subordinate action spaces."""
        config = {"agents": [{"id": "field_1"}, {"id": "field_2"}]}
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", config=config)

        spaces = coord.get_subordinate_action_spaces()

        assert len(spaces) == 2
        assert "field_1" in spaces
        assert "field_2" in spaces

    def test_get_joint_action_space_continuous(self):
        """Test joint action space construction."""
        config = {"agents": [{"id": "field_1"}, {"id": "field_2"}]}
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", config=config)

        space = coord.get_joint_action_space()

        assert isinstance(space, Box)
        # Each agent has dim_c=2, so total should be 4
        assert space.shape[0] == 4

    def test_get_joint_action_space_empty(self):
        """Test joint action space with no subordinates."""
        coord = ConcreteCoordinatorAgent(agent_id="coord_1")

        space = coord.get_joint_action_space()

        assert isinstance(space, Discrete)
        assert space.n == 1


class TestCoordinatorAgentRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__."""
        config = {"agents": [{"id": "field_1"}]}
        coord = ConcreteCoordinatorAgent(agent_id="coord_1", config=config)

        repr_str = repr(coord)

        assert "CoordinatorAgent" in repr_str
        assert "coord_1" in repr_str
        assert "subordinates=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

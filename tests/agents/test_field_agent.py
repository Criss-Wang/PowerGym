"""Tests for FieldAgent module."""

import pytest
import numpy as np
from gymnasium.spaces import Box

from heron.agents.field_agent import FieldAgent, FieldConfig, FIELD_LEVEL
from heron.core.observation import Observation
from heron.core.action import Action
from heron.core.policies import Policy
from heron.core.state import FieldAgentState
from heron.core.feature import FeatureProvider
from heron.scheduling.tick_config import TickConfig


class MockFeature(FeatureProvider):
    """Mock feature for testing."""

    visibility = ["public"]

    def __init__(self, value: float = 1.0):
        self.value = value

    def vector(self):
        return np.array([self.value], dtype=np.float32)

    def names(self):
        return ["value"]

    def to_dict(self):
        return {"value": self.value}

    @classmethod
    def from_dict(cls, d):
        return cls(value=d.get("value", 1.0))

    def set_values(self, **kwargs):
        if "value" in kwargs:
            self.value = kwargs["value"]


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, action_value=None):
        self.action_value = action_value if action_value is not None else np.array([0.5])

    def forward(self, observation: Observation):
        action = Action()
        action.set_specs(dim_c=len(self.action_value), range=(np.full(len(self.action_value), -1), np.full(len(self.action_value), 1)))
        action.set_values(c=self.action_value)
        return action


class ConcreteFieldAgent(FieldAgent):
    """Concrete implementation for testing."""

    def set_action(self):
        self.action.set_specs(
            dim_c=2,
            range=(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        )

    def set_state(self):
        self.state.features = [MockFeature(value=1.0)]


class TestFieldConfig:
    """Test FieldConfig dataclass."""

    def test_field_config_creation(self):
        """Test FieldConfig creation."""
        config = FieldConfig(
            name="test_agent",
            state_config={"key": "value"},
            discrete_action=False,
            discrete_action_cats=2,
        )

        assert config.name == "test_agent"
        assert config.state_config == {"key": "value"}
        assert config.discrete_action is False
        assert config.discrete_action_cats == 2


class TestFieldAgentInitialization:
    """Test FieldAgent initialization."""

    def test_basic_initialization(self):
        """Test basic field agent initialization."""
        agent = ConcreteFieldAgent(agent_id="field_1")

        assert agent.agent_id == "field_1"
        assert agent.level == FIELD_LEVEL
        assert isinstance(agent.state, FieldAgentState)
        assert isinstance(agent.action, Action)

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = {
            "name": "battery",
            "state_config": {"capacity": 100},
            "discrete_action": False,
            "discrete_action_cats": 3,
        }
        agent = ConcreteFieldAgent(agent_id="field_1", config=config)

        assert agent.field_config.name == "battery"
        assert agent.field_config.discrete_action is False

    def test_initialization_with_policy(self):
        """Test initialization with policy."""
        policy = MockPolicy()
        agent = ConcreteFieldAgent(agent_id="field_1", policy=policy)

        assert agent.policy is policy

    def test_initialization_with_timing_params(self):
        """Test initialization with timing parameters."""
        tick_config = TickConfig(
            tick_interval=5.0,
            obs_delay=0.5,
            act_delay=1.0,
            msg_delay=0.2,
        )
        agent = ConcreteFieldAgent(
            agent_id="field_1",
            tick_config=tick_config,
        )

        assert agent._tick_config.tick_interval == 5.0
        assert agent._tick_config.obs_delay == 0.5
        assert agent._tick_config.act_delay == 1.0
        assert agent._tick_config.msg_delay == 0.2

    def test_initialization_sets_spaces(self):
        """Test that action and observation spaces are set."""
        agent = ConcreteFieldAgent(agent_id="field_1")

        assert agent.action_space is not None
        assert agent.observation_space is not None


class TestFieldAgentReset:
    """Test FieldAgent reset functionality."""

    def test_reset_basic(self):
        """Test basic reset."""
        agent = ConcreteFieldAgent(agent_id="field_1")
        agent._timestep = 100.0

        agent.reset()

        assert agent._timestep == 0.0


class TestFieldAgentObserve:
    """Test FieldAgent observe functionality."""

    def test_observe_basic(self):
        """Test basic observation."""
        agent = ConcreteFieldAgent(agent_id="field_1")
        agent._timestep = 5.0

        obs = agent.observe()

        assert isinstance(obs, Observation)
        assert obs.timestamp == 5.0
        assert "state" in obs.local
        assert "observation" in obs.local

    def test_observe_returns_state_vector(self):
        """Test observation contains state vector."""
        agent = ConcreteFieldAgent(agent_id="field_1")

        obs = agent.observe()

        state_vec = obs.local.get("state")
        assert isinstance(state_vec, np.ndarray)


class TestFieldAgentAct:
    """Test FieldAgent act functionality."""

    def test_act_with_upstream_action(self):
        """Test act with upstream action."""
        agent = ConcreteFieldAgent(agent_id="field_1")
        obs = agent.observe()

        upstream_action = np.array([0.5, 0.3])
        agent.act(obs, upstream_action=upstream_action)

        np.testing.assert_array_almost_equal(agent.action.c, upstream_action)

    def test_act_with_policy(self):
        """Test act using local policy."""
        policy = MockPolicy(action_value=np.array([0.7, 0.2]))
        agent = ConcreteFieldAgent(agent_id="field_1", policy=policy)
        obs = agent.observe()

        agent.act(obs)

        # Policy returns [0.7, 0.2]
        np.testing.assert_array_almost_equal(agent.action.c, [0.7, 0.2])

    def test_act_without_action_or_policy_raises(self):
        """Test act raises without action or policy."""
        agent = ConcreteFieldAgent(agent_id="field_1")
        obs = agent.observe()

        with pytest.raises(ValueError, match="No policy defined"):
            agent.act(obs)


class TestFieldAgentActionHandling:
    """Test action handling methods."""

    def test_handle_coordinator_action(self):
        """Test handling coordinator action."""
        agent = ConcreteFieldAgent(agent_id="field_1")
        obs = agent.observe()

        upstream_action = np.array([0.5, 0.5])
        result = agent._handle_coordinator_action(upstream_action, obs)

        np.testing.assert_array_equal(result, upstream_action)

    def test_handle_local_action_with_policy(self):
        """Test handling local action with policy."""
        policy = MockPolicy(action_value=np.array([0.3, 0.4]))
        agent = ConcreteFieldAgent(agent_id="field_1", policy=policy)
        obs = agent.observe()

        result = agent._handle_local_action(obs)

        assert result is not None

    def test_handle_local_action_without_policy_raises(self):
        """Test handling local action without policy raises."""
        agent = ConcreteFieldAgent(agent_id="field_1")
        obs = agent.observe()

        with pytest.raises(ValueError, match="No policy defined"):
            agent._handle_local_action(obs)


class TestFieldAgentSpaces:
    """Test action/observation space methods."""

    def test_get_action_space(self):
        """Test getting action space."""
        agent = ConcreteFieldAgent(agent_id="field_1")

        space = agent._get_action_space()

        assert isinstance(space, Box)
        assert space.shape == (2,)

    def test_get_observation_space(self):
        """Test getting observation space."""
        agent = ConcreteFieldAgent(agent_id="field_1")

        space = agent._get_observation_space()

        assert isinstance(space, Box)


class TestFieldAgentGetObs:
    """Test _get_obs method."""

    def test_get_obs_basic(self):
        """Test basic observation vector."""
        agent = ConcreteFieldAgent(agent_id="field_1")

        obs_vec = agent._get_obs()

        assert isinstance(obs_vec, np.ndarray)
        assert obs_vec.dtype == np.float32


class TestFieldAgentRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__."""
        agent = ConcreteFieldAgent(agent_id="field_1")

        repr_str = repr(agent)

        assert "FieldAgent" in repr_str
        assert "field_1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for SystemAgent module."""

import pytest
import numpy as np

from heron.agents.system_agent import SystemAgent, SYSTEM_LEVEL
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.core.state import State, FieldAgentState
from heron.core.feature import FeatureProvider


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


class MockCoordinator(CoordinatorAgent):
    """Mock coordinator for testing."""

    def __init__(self, agent_id, **kwargs):
        # Skip parent's _build_subordinate_agents by not passing config
        super().__init__(agent_id=agent_id, config={}, **kwargs)

    def _build_subordinate_agents(self, agent_configs, env_id=None, upstream_id=None):
        return {}


class ConcreteSystemAgent(SystemAgent):
    """Concrete implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_state = {}

    def _build_coordinators(self, coordinator_configs, env_id=None, upstream_id=None):
        coordinators = {}
        for config in coordinator_configs:
            coord_id = config.get("id", f"coord_{len(coordinators)}")
            coordinators[coord_id] = MockCoordinator(
                agent_id=coord_id,
                env_id=env_id,
                upstream_id=upstream_id,
            )
        return coordinators

    def update_from_environment(self, env_state):
        self._env_state = env_state

    def get_state_for_environment(self):
        return {"actions": {}}


class TestSystemAgentInitialization:
    """Test SystemAgent initialization."""

    def test_basic_initialization(self):
        """Test basic system agent initialization."""
        system = ConcreteSystemAgent(agent_id="system_1")

        assert system.agent_id == "system_1"
        assert system.level == SYSTEM_LEVEL
        assert system.coordinators == {}
        assert system.upstream_id is None

    def test_initialization_with_default_id(self):
        """Test initialization with default agent ID."""
        system = ConcreteSystemAgent()

        assert system.agent_id == "system_agent"

    def test_initialization_with_coordinators(self):
        """Test initialization with coordinator configs."""
        config = {
            "coordinators": [
                {"id": "coord_1"},
                {"id": "coord_2"},
            ]
        }
        system = ConcreteSystemAgent(
            agent_id="system_1",
            config=config,
            env_id="test_env",
        )

        assert len(system.coordinators) == 2
        assert "coord_1" in system.coordinators
        assert "coord_2" in system.coordinators

    def test_initialization_with_timing_params(self):
        """Test initialization with timing parameters."""
        system = ConcreteSystemAgent(
            agent_id="system_1",
            tick_interval=600.0,
            obs_delay=5.0,
            act_delay=10.0,
            msg_delay=2.0,
        )

        assert system.tick_interval == 600.0
        assert system.obs_delay == 5.0
        assert system.act_delay == 10.0
        assert system.msg_delay == 2.0

    def test_coordinators_have_correct_upstream(self):
        """Test that coordinators have system as upstream."""
        config = {"coordinators": [{"id": "coord_1"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        coord = system.coordinators["coord_1"]
        assert coord.upstream_id == "system_1"


class TestSystemAgentReset:
    """Test SystemAgent reset functionality."""

    def test_reset_basic(self):
        """Test basic reset."""
        system = ConcreteSystemAgent(agent_id="system_1")
        system._timestep = 100.0

        system.reset()

        assert system._timestep == 0.0

    def test_reset_resets_coordinators(self):
        """Test that reset resets coordinators."""
        config = {"coordinators": [{"id": "coord_1"}, {"id": "coord_2"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        # Set timesteps on coordinators
        for coord in system.coordinators.values():
            coord._timestep = 50.0

        system.reset()

        for coord in system.coordinators.values():
            assert coord._timestep == 0.0


class TestSystemAgentObserve:
    """Test SystemAgent observe functionality."""

    def test_observe_empty_coordinators(self):
        """Test observation with no coordinators."""
        system = ConcreteSystemAgent(agent_id="system_1")
        system._timestep = 5.0

        obs = system.observe()

        assert isinstance(obs, Observation)
        assert obs.timestamp == 5.0
        assert "coordinator_obs" in obs.local

    def test_observe_with_coordinators(self):
        """Test observation aggregates coordinator observations."""
        config = {"coordinators": [{"id": "coord_1"}, {"id": "coord_2"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        obs = system.observe()

        coordinator_obs = obs.local.get("coordinator_obs", {})
        assert "coord_1" in coordinator_obs
        assert "coord_2" in coordinator_obs

    def test_observe_with_global_state(self):
        """Test observation includes global state."""
        system = ConcreteSystemAgent(agent_id="system_1")
        global_state = {"total_load": 1000, "frequency": 60.0}

        obs = system.observe(global_state=global_state)

        assert obs.global_info == global_state


class TestSystemAgentAct:
    """Test SystemAgent act functionality."""

    def test_act_without_action(self):
        """Test act without upstream action does nothing."""
        system = ConcreteSystemAgent(agent_id="system_1")
        obs = system.observe()

        # Should not raise
        system.act(obs)

    def test_act_with_dict_action(self):
        """Test act with per-coordinator dict action."""
        config = {"coordinators": [{"id": "coord_1"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        obs = system.observe()
        action = {"coord_1": np.array([0.5, 0.5])}

        # Should not raise
        system.act(obs, upstream_action=action)

    def test_act_with_single_action(self):
        """Test act with single action for all coordinators."""
        config = {"coordinators": [{"id": "coord_1"}, {"id": "coord_2"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        obs = system.observe()
        action = np.array([0.5, 0.5])

        # Should not raise (will apply same action to all)
        system.act(obs, upstream_action=action)


class TestSystemAgentEnvironmentInterface:
    """Test environment interface methods."""

    def test_update_from_environment(self):
        """Test update_from_environment."""
        system = ConcreteSystemAgent(agent_id="system_1")
        env_state = {"load": 100, "voltage": 1.0}

        system.update_from_environment(env_state)

        assert system._env_state == env_state

    def test_get_state_for_environment(self):
        """Test get_state_for_environment."""
        system = ConcreteSystemAgent(agent_id="system_1")

        result = system.get_state_for_environment()

        assert isinstance(result, dict)


class TestSystemAgentVisibilityFiltering:
    """Test visibility filtering functionality."""

    def test_filter_state_for_agent(self):
        """Test filtering state based on visibility."""
        system = ConcreteSystemAgent(agent_id="system_1")
        state = FieldAgentState(owner_id="agent_1", owner_level=1)
        state.features = [MockFeature(value=5.0)]

        filtered = system.filter_state_for_agent(
            state=state,
            requestor_id="agent_1",
            requestor_level=1,
        )

        assert isinstance(filtered, dict)


class TestSystemAgentBroadcast:
    """Test broadcast functionality."""

    def test_broadcast_without_broker(self):
        """Test broadcast without message broker does nothing."""
        config = {"coordinators": [{"id": "coord_1"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)
        system._message_broker = None

        # Should not raise
        system.broadcast_to_coordinators({"signal": "test"})


class TestSystemAgentRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__."""
        config = {"coordinators": [{"id": "coord_1"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        repr_str = repr(system)

        assert "SystemAgent" in repr_str
        assert "system_1" in repr_str
        assert "coordinators=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for SystemAgent module."""

import pytest
import numpy as np
from gymnasium.spaces import Box, Discrete

from heron.agents.system_agent import SystemAgent, SYSTEM_LEVEL
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.core.state import State, FieldAgentState, SystemAgentState
from heron.core.feature import FeatureProvider
from heron.core.policies import Policy
from heron.protocols.vertical import SetpointProtocol, SystemProtocol
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

    def __init__(self, action_value=0.5):
        self.action_value = action_value
        self._reset_called = False

    def forward(self, observation):
        return np.array([self.action_value])

    def reset(self):
        self._reset_called = True


class MockCoordinator(CoordinatorAgent):
    """Mock coordinator for testing."""

    def __init__(self, agent_id, **kwargs):
        # Skip parent's _build_subordinates by not passing config
        super().__init__(agent_id=agent_id, config={}, **kwargs)

    def _build_subordinates(self, configs, env_id=None, upstream_id=None):
        return {}

    def get_joint_action_space(self):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


class ConcreteSystemAgent(SystemAgent):
    """Concrete implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_subordinates(self, configs, env_id=None, upstream_id=None):
        coordinators = {}
        for config in configs:
            coord_id = config.get("id", f"coord_{len(coordinators)}")
            coordinators[coord_id] = MockCoordinator(
                agent_id=coord_id,
                env_id=env_id,
                upstream_id=upstream_id,
            )
        return coordinators


class SystemAgentWithState(SystemAgent):
    """System agent with custom state for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_subordinates(self, configs, env_id=None, upstream_id=None):
        return {}

    def set_state(self):
        self.state.features = [MockFeature(value=42.0)]


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
        tick_config = TickConfig(
            tick_interval=600.0,
            obs_delay=5.0,
            act_delay=10.0,
            msg_delay=2.0,
        )
        system = ConcreteSystemAgent(
            agent_id="system_1",
            tick_config=tick_config,
        )

        assert system._tick_config.tick_interval == 600.0
        assert system._tick_config.obs_delay == 5.0
        assert system._tick_config.act_delay == 10.0
        assert system._tick_config.msg_delay == 2.0

    def test_coordinators_have_correct_upstream(self):
        """Test that coordinators have system as upstream."""
        config = {"coordinators": [{"id": "coord_1"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        coord = system.coordinators["coord_1"]
        assert coord.upstream_id == "system_1"

    def test_initialization_with_protocol(self):
        """Test initialization with protocol."""
        protocol = SetpointProtocol()
        system = ConcreteSystemAgent(
            agent_id="system_1",
            protocol=protocol
        )

        assert system.protocol is protocol

    def test_initialization_with_policy(self):
        """Test initialization with policy."""
        policy = MockPolicy()
        system = ConcreteSystemAgent(
            agent_id="system_1",
            policy=policy
        )

        assert system.policy is policy

    def test_initialization_creates_state(self):
        """Test that initialization creates SystemAgentState."""
        system = ConcreteSystemAgent(agent_id="system_1")

        assert system.state is not None
        assert isinstance(system.state, SystemAgentState)
        assert system.state.owner_id == "system_1"
        assert system.state.owner_level == SYSTEM_LEVEL

    def test_initialization_creates_action(self):
        """Test that initialization creates Action object."""
        system = ConcreteSystemAgent(agent_id="system_1")

        assert system.action is not None


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

    def test_reset_resets_policy(self):
        """Test that reset resets policy."""
        policy = MockPolicy()
        system = ConcreteSystemAgent(agent_id="system_1", policy=policy)

        system.reset()

        assert policy._reset_called

    def test_reset_clears_cached_env_state(self):
        """Test that reset clears cached environment state."""
        system = ConcreteSystemAgent(agent_id="system_1")
        system._cached_env_state = {"test": "data"}

        system.reset()

        assert system._cached_env_state == {}


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

    def test_observe_includes_system_state(self):
        """Test observation includes system state vector."""
        system = SystemAgentWithState(agent_id="system_1")

        obs = system.observe()

        assert "system_state" in obs.local
        assert obs.local["system_state"] is not None


class TestSystemAgentAct:
    """Test SystemAgent act functionality."""

    def test_act_without_action_and_policy(self):
        """Test act without upstream action or policy returns None."""
        system = ConcreteSystemAgent(agent_id="system_1")
        obs = system.observe()

        result = system.act(obs)

        assert result is None

    def test_act_with_dict_action(self):
        """Test act with per-coordinator dict action."""
        config = {"coordinators": [{"id": "coord_1"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        obs = system.observe()
        action = {"coord_1": np.array([0.5, 0.5])}

        result = system.act(obs, upstream_action=action)

        assert result is action

    def test_act_with_single_action(self):
        """Test act with single action for all coordinators."""
        config = {"coordinators": [{"id": "coord_1"}, {"id": "coord_2"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        obs = system.observe()
        action = np.array([0.5, 0.5])

        result = system.act(obs, upstream_action=action)

        assert np.array_equal(result, action)

    def test_act_with_policy(self):
        """Test act uses policy when no upstream action."""
        policy = MockPolicy(action_value=0.75)
        system = ConcreteSystemAgent(agent_id="system_1", policy=policy)

        obs = system.observe()
        result = system.act(obs)

        assert result is not None
        assert result[0] == 0.75

    def test_act_upstream_action_overrides_policy(self):
        """Test upstream action takes precedence over policy."""
        policy = MockPolicy(action_value=0.75)
        system = ConcreteSystemAgent(agent_id="system_1", policy=policy)

        obs = system.observe()
        upstream_action = np.array([0.25])
        result = system.act(obs, upstream_action=upstream_action)

        assert np.array_equal(result, upstream_action)


class TestSystemAgentEnvironmentInterface:
    """Test environment interface methods."""

    def test_update_from_environment(self):
        """Test update_from_environment caches state."""
        system = ConcreteSystemAgent(agent_id="system_1")
        env_state = {"load": 100, "voltage": 1.0}

        system.update_from_environment(env_state)

        assert system._cached_env_state == env_state

    def test_update_from_environment_updates_state_features(self):
        """Test update_from_environment updates state features."""
        system = SystemAgentWithState(agent_id="system_1")
        env_state = {
            "system": {"MockFeature": {"value": 99.0}}
        }

        system.update_from_environment(env_state)

        # Feature should be updated
        assert system.state.features[0].value == 99.0

    def test_get_state_for_environment(self):
        """Test get_state_for_environment."""
        system = ConcreteSystemAgent(agent_id="system_1")

        result = system.get_state_for_environment()

        assert isinstance(result, dict)
        assert "system_state" in result
        assert "coordinators" in result

    def test_get_state_for_environment_includes_coordinators(self):
        """Test get_state_for_environment includes coordinator states."""
        config = {"coordinators": [{"id": "coord_1"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        result = system.get_state_for_environment()

        assert "coord_1" in result["coordinators"]


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

    def test_repr_with_protocol(self):
        """Test __repr__ includes protocol name."""
        protocol = SetpointProtocol()
        system = ConcreteSystemAgent(agent_id="system_1", protocol=protocol)

        repr_str = repr(system)

        assert "SetpointProtocol" in repr_str


class TestSystemAgentState:
    """Test SystemAgentState functionality."""

    def test_state_initialization(self):
        """Test SystemAgentState can be created."""
        state = SystemAgentState(owner_id="system_1", owner_level=3)

        assert state.owner_id == "system_1"
        assert state.owner_level == 3
        assert state.features == []

    def test_state_with_features(self):
        """Test state with features."""
        state = SystemAgentState(owner_id="system_1", owner_level=3)
        state.features = [MockFeature(value=10.0)]

        vec = state.vector()

        assert vec.shape == (1,)
        assert vec[0] == 10.0

    def test_state_update(self):
        """Test state update method."""
        state = SystemAgentState(owner_id="system_1", owner_level=3)
        state.features = [MockFeature(value=10.0)]

        state.update({"MockFeature": {"value": 20.0}})

        assert state.features[0].value == 20.0


class TestSystemAgentExtensionHooks:
    """Test extension hook methods."""

    def test_set_state_hook(self):
        """Test set_state is called during init."""
        system = SystemAgentWithState(agent_id="system_1")

        assert len(system.state.features) == 1
        assert system.state.features[0].value == 42.0

    def test_set_action_hook(self):
        """Test set_action is called during init."""

        class CustomSystem(ConcreteSystemAgent):
            def set_action(self):
                self.action.set_specs(dim_c=2)

        system = CustomSystem(agent_id="system_1")

        assert system.action.dim_c == 2

    def test_reset_system_hook(self):
        """Test reset_system is called during reset."""

        class CustomSystem(ConcreteSystemAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.reset_system_called = False

            def reset_system(self, **kwargs):
                self.reset_system_called = True

        system = CustomSystem(agent_id="system_1")
        system.reset_system_called = False

        system.reset()

        assert system.reset_system_called


class TestSystemAgentJointActionSpace:
    """Test joint action space construction."""

    def test_get_joint_action_space_empty(self):
        """Test joint action space with no coordinators."""
        system = ConcreteSystemAgent(agent_id="system_1")

        space = system.get_joint_action_space()

        assert space is not None
        assert isinstance(space, Discrete)

    def test_get_joint_action_space_with_coordinators(self):
        """Test joint action space aggregation."""
        config = {"coordinators": [{"id": "coord_1"}, {"id": "coord_2"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        space = system.get_joint_action_space()

        assert space is not None
        # MockCoordinator returns Box(2,) so joint should be Box(4,)
        assert isinstance(space, Box)
        assert space.shape == (4,)

    def test_get_subordinate_action_spaces(self):
        """Test getting individual coordinator action spaces."""
        config = {"coordinators": [{"id": "coord_1"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        spaces = system.get_subordinate_action_spaces()

        assert "coord_1" in spaces
        assert isinstance(spaces["coord_1"], Box)


class TestSystemAgentActionDistribution:
    """Test action distribution strategies."""

    def test_distribute_dict_action(self):
        """Test distribution with dict action passes through."""
        config = {"coordinators": [{"id": "coord_1"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        action = {"coord_1": np.array([0.5])}
        result = system._simple_action_distribution(action)

        assert result == action

    def test_distribute_array_action(self):
        """Test distribution splits array action."""
        config = {"coordinators": [{"id": "coord_1"}, {"id": "coord_2"}]}
        system = ConcreteSystemAgent(agent_id="system_1", config=config)

        action = np.array([0.1, 0.2, 0.3, 0.4])
        result = system._simple_action_distribution(action)

        assert "coord_1" in result
        assert "coord_2" in result


class TestSystemProtocol:
    """Test SystemProtocol functionality."""

    def test_system_protocol_initialization(self):
        """Test SystemProtocol can be created."""
        protocol = SystemProtocol()

        assert protocol is not None
        assert protocol.communication_protocol is not None
        assert protocol.action_protocol is not None

    def test_system_agent_with_system_protocol(self):
        """Test SystemAgent can use SystemProtocol."""
        protocol = SystemProtocol()
        system = ConcreteSystemAgent(
            agent_id="system_1",
            protocol=protocol
        )

        assert system.protocol is protocol


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

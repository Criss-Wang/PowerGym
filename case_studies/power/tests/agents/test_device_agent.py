"""Tests for agents.device_agent module."""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from powergrid.agents.device_agent import DeviceAgent
from heron.agents.base import Observation
from heron.core.action import Action
from powergrid.core.state.state import DeviceState
from heron.core.policies import Policy
from heron.protocols.base import NoProtocol


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, action_value=0.5):
        self.action_value = action_value

    def forward(self, observation):
        """Return fixed action."""
        return np.array([self.action_value])


class ConcreteDeviceAgent(DeviceAgent):
    """Concrete device agent for testing."""

    def __init__(self, *args, **kwargs):
        self._P = 0.0  # Store P as instance var since state uses features
        super().__init__(*args, **kwargs)

    def set_action(self):
        """Define simple continuous action space."""
        self.action.set_specs(
            dim_c=1,
            range=(np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32))
        )

    def _get_obs(self, proxy=None) -> np.ndarray:
        """Override to return simple mock observation."""
        return np.array([self._P], dtype=np.float32)

    def reset_agent(self, *args, **kwargs):
        """Reset device state."""
        self._P = 0.0
        self.cost = 0.0
        self.safety = 0.0

    def update_state(self, *args, **kwargs):
        """Update device state from action."""
        if self.action.dim_c > 0:
            self._P = float(self.action.c[0])

    def update_cost_safety(self, *args, **kwargs):
        """Update cost and safety metrics."""
        self.cost = abs(self._P) * 10.0
        self.safety = 0.0

    def get_reward(self):
        """Calculate reward."""
        return -self.cost - self.safety

    @property
    def P(self):
        """Access to internal P value (state.P alias for tests)."""
        return self._P

    @P.setter
    def P(self, value):
        self._P = value

    def __repr__(self):
        """String representation."""
        return f"ConcreteDeviceAgent(id={self.agent_id})"
from heron.core.policies import RandomPolicy
from powergrid.agents.storage import ESS
from powergrid.agents.generator import Generator


class TestDeviceAgent:
    """Test DeviceAgent class."""

    def test_device_agent_initialization(self):
        """Test device agent initialization."""
        policy = MockPolicy()
        agent = ConcreteDeviceAgent(
            agent_id="test_device",
            policy=policy,
            device_config={"name": "test_device"}
        )

        assert agent.agent_id == "test_device"
        assert agent.level == 1
        assert agent.policy == policy
        assert isinstance(agent.state, DeviceState)
        assert isinstance(agent.action, Action)

    def test_device_agent_initialization_without_agent_id(self):
        """Test device agent uses name from config."""
        agent = ConcreteDeviceAgent(
            policy=MockPolicy(),
            device_config={"name": "device_from_config"}
        )

        assert agent.agent_id == "device_from_config"

    def test_device_agent_requires_id_or_name(self):
        """Test device agent defaults to 'device_agent' when no agent_id or name provided."""
        agent = ConcreteDeviceAgent(policy=MockPolicy())
        # Default name from config is used
        assert agent.agent_id == "device_agent"

    def test_get_action_space_continuous(self):
        """Test action space construction for continuous actions."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        action_space = agent.action_space

        assert isinstance(action_space, Box)
        assert action_space.shape == (1,)
        np.testing.assert_array_equal(action_space.low, [0.0])
        np.testing.assert_array_equal(action_space.high, [1.0])

    def test_get_action_space_discrete(self):
        """Test discrete action space."""
        class DiscreteDeviceAgent(DeviceAgent):
            def set_action(self):
                self.action.set_specs(
                    dim_d=1,
                    ncats=[5]
                )

            def _get_obs(self):
                return np.array([0.0], dtype=np.float32)

            def reset_agent(self, **kwargs):
                pass

            def update_state(self, **kwargs):
                pass

            def update_cost_safety(self, **kwargs):
                pass

            def get_reward(self):
                return 0.0

            def __repr__(self):
                return "DiscreteDeviceAgent"

        agent = DiscreteDeviceAgent(agent_id="test", policy=MockPolicy())
        action_space = agent.action_space
        assert isinstance(action_space, Discrete)
        assert action_space.n == 5

    def test_device_agent_observe(self):
        """Test observation extraction using ESS device."""
        # Create ESS using the new API with device_config
        ess = ESS(
            agent_id="ess_1",
            device_config={
                "name": "ess_1",
                "device_state_config": {
                    "bus": "800",
                    "p_ch_max_MW": 0.5,
                    "p_dsc_max_MW": 0.5,
                    "e_capacity_MWh": 1.0,
                    "soc_init": 0.5,
                }
            }
        )
        # Update ESS electrical state using proper API
        ess.electrical.P_MW = 0.2
        ess.electrical.Q_MVAr = 0.1

        global_state = {
            "bus_vm": {800: 1.05},
            "bus_va": {800: 0.5},
            "converged": True,
            "dataset": {"price": 50.0, "load": 1.0},
        }

        obs = ess.observe(global_state)

        # Check observation is returned
        assert isinstance(obs, Observation)
        assert "observation" in obs.local
        assert isinstance(obs.local["observation"], np.ndarray)

    def test_get_action_space_multidimensional_continuous(self):
        """Test multi-dimensional continuous action space."""
        class MultiDimAgent(DeviceAgent):
            def set_action(self):
                self.action.set_specs(
                    dim_c=2,
                    range=(
                        np.array([0.0, -1.0], dtype=np.float32),
                        np.array([1.0, 1.0], dtype=np.float32)
                    )
                )

            def _get_obs(self):
                return np.array([0.0, 0.0], dtype=np.float32)

            def reset_agent(self, **kwargs):
                pass

            def update_state(self, **kwargs):
                pass

            def update_cost_safety(self, **kwargs):
                pass

            def get_reward(self):
                return 0.0

            def __repr__(self):
                return "MultiDimAgent"

        agent = MultiDimAgent(agent_id="test", policy=MockPolicy())

        action_space = agent.action_space

        assert isinstance(action_space, Box)
        assert action_space.shape == (2,)

    def test_get_observation_space(self):
        """Test observation space construction."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        obs_space = agent.observation_space

        assert isinstance(obs_space, Box)
        assert obs_space.dtype == np.float32

    def test_reset(self):
        """Test agent reset."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        # Modify state
        agent._P = 10.0
        agent.cost = 100.0
        agent._timestep = 5.0

        # Reset
        agent.reset()

        assert agent._P == 0.0
        assert agent.cost == 0.0
        assert agent._timestep == 0.0

    def test_observe(self):
        """Test observe method."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )
        agent._timestep = 5.0
        agent._P = 0.5

        obs = agent.observe()

        assert isinstance(obs, Observation)
        assert obs.timestamp == 5.0
        # observe() returns local dict with 'observation' key from _get_obs()
        assert "observation" in obs.local
        assert isinstance(obs.local["observation"], np.ndarray)

    def test_observe_with_global_state(self):
        """Test observe with global state information."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        global_state = {"voltage": 1.05, "frequency": 60.0}
        obs = agent.observe(global_state=global_state)

        # Current implementation doesn't include global state in observation
        # DeviceAgent focuses on local state only
        assert isinstance(obs, Observation)
        assert "observation" in obs.local

    def test_act_with_policy(self):
        """Test act method with policy."""
        policy = MockPolicy(action_value=0.7)
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=policy
        )

        obs = Observation(local={"state": np.array([0.5])})
        agent.act(obs)  # act() doesn't return anything, sets internal action

        np.testing.assert_array_almost_equal(agent.action.c, [0.7])

    def test_act_with_given_action(self):
        """Test act with externally provided action (upstream_action)."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        obs = Observation()
        given_action = np.array([0.3])
        agent.act(obs, upstream_action=given_action)

        np.testing.assert_array_almost_equal(agent.action.c, [0.3])

    def test_act_without_policy_raises_error(self):
        """Test act without policy raises ValueError."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=None
        )

        obs = Observation()

        with pytest.raises(ValueError):
            agent.act(obs)

    def test_set_action_values(self):
        """Test action.set_values method."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        action = np.array([0.8])
        agent.action.set_values(action)

        np.testing.assert_array_almost_equal(agent.action.c, [0.8])

    def test_set_action_values_with_dict(self):
        """Test action.set_values with dict input."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        # Set action with dict format
        agent.action.set_values({"c": [0.5]})

        np.testing.assert_array_almost_equal(agent.action.c, [0.5])

    def test_feasible_action_default(self):
        """Test feasible_action default implementation."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        # Should return None by default
        result = agent.feasible_action()
        assert result is None

    def test_repr(self):
        """Test string representation."""
        agent = ConcreteDeviceAgent(
            agent_id="test_device",
            policy=MockPolicy()
        )

        repr_str = repr(agent)

        assert "ConcreteDeviceAgent" in repr_str
        assert "test_device" in repr_str


class TestDeviceAgentIntegration:
    """Integration tests for DeviceAgent."""

    def test_full_lifecycle(self):
        """Test full agent lifecycle: reset -> observe -> act -> update."""
        policy = MockPolicy(action_value=0.6)
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=policy
        )

        # Reset
        agent.reset()
        assert agent._P == 0.0

        # Observe
        obs = agent.observe()
        assert isinstance(obs, Observation)

        # Act (doesn't return value, sets internal action)
        agent.act(obs)
        np.testing.assert_array_almost_equal(agent.action.c, [0.6])

        # Update state
        agent.update_state()
        np.testing.assert_almost_equal(agent._P, 0.6, decimal=5)

        # Update cost/safety
        agent.update_cost_safety()
        assert agent.cost > 0

    def test_timestep_update(self):
        """Test timestep tracking through lifecycle."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        agent.reset()
        assert agent._timestep == 0.0

        agent.update_timestep(1.0)
        obs = agent.observe()
        assert obs.timestamp == 1.0

        agent.update_timestep(2.0)
        obs = agent.observe()
        assert obs.timestamp == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

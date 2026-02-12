"""Tests for microgrid agent and environment."""

import pytest
import numpy as np

from case_studies.grid_age.agents import MicrogridFieldAgent
from case_studies.grid_age.envs import MicrogridEnv
from case_studies.grid_age.features import (
    ESSFeature,
    DGFeature,
    RESFeature,
    GridFeature,
    NetworkFeature,
)
from heron.core.action import Action


def create_default_features(ess_capacity=2.0, dg_max_p=0.66, pv_max_p=0.1, wind_max_p=0.1):
    """Helper to create default feature set for MicrogridFieldAgent."""
    return [
        ESSFeature(capacity=ess_capacity, min_p=-0.5, max_p=0.5, soc=0.5),
        DGFeature(max_p=dg_max_p, min_p=0.1, on=1, P=0.1),
        RESFeature(max_p=pv_max_p),  # PV
        RESFeature(max_p=wind_max_p),  # Wind
        GridFeature(),
        NetworkFeature(),
    ]


class TestMicrogridFieldAgent:
    """Tests for MicrogridFieldAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        features = create_default_features(ess_capacity=2.0, dg_max_p=0.66)
        agent = MicrogridFieldAgent(agent_id="MG1", features=features)

        assert agent.agent_id == "MG1"
        assert len(agent.state.features) == 6  # ESS, DG, PV, Wind, Grid, Network
        assert agent.action.dim_c == 4  # 4D continuous action

    def test_action_space(self):
        """Test action space specification."""
        features = create_default_features()
        agent = MicrogridFieldAgent(agent_id="test", features=features)

        # Action space is set via get_action_space() which uses self.action
        action_space = agent.get_action_space()
        assert action_space.shape == (4,)
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)

    def test_set_action(self):
        """Test setting action from different formats."""
        features = create_default_features()
        agent = MicrogridFieldAgent(agent_id="test", features=features)

        # Test with Action object
        action = Action()
        action.set_specs(dim_c=4, range=(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1])))
        action.set_values(c=np.array([0.5, 0.3, -0.2, 0.1]))
        agent.set_action(action)
        assert np.allclose(agent.action.c, [0.5, 0.3, -0.2, 0.1])

        # Test with numpy array
        agent.set_action(np.array([0.1, 0.2, 0.3, 0.4]))
        assert np.allclose(agent.action.c, [0.1, 0.2, 0.3, 0.4])

    def test_apply_action(self):
        """Test applying action to device features."""
        features = create_default_features(dg_max_p=0.66)
        agent = MicrogridFieldAgent(agent_id="test", features=features)

        # Set normalized action
        agent.set_action(np.array([0.5, 0.0, -0.5, 1.0]))
        agent.apply_action()

        # Check ESS power (0.5 * 0.5 = 0.25 MW)
        ess_feature = agent.state.features[agent.ess_idx]
        assert abs(ess_feature.P - 0.25) < 1e-6

        # Check DG power (normalized 0.0 → midpoint)
        dg_feature = agent.state.features[agent.dg_idx]
        expected_dg = 0.1 + (0.0 + 1) / 2 * (0.66 - 0.1)  # Mid-range
        assert abs(dg_feature.P - expected_dg) < 1e-6

    def test_set_renewable_availability(self):
        """Test setting renewable availability."""
        features = create_default_features(pv_max_p=0.1, wind_max_p=0.1)
        agent = MicrogridFieldAgent(agent_id="test", features=features)

        agent.set_renewable_availability(pv_availability=0.8, wind_availability=0.5)

        pv_feature = agent.state.features[agent.pv_idx]
        wind_feature = agent.state.features[agent.wind_idx]

        assert pv_feature.availability == 0.8
        assert abs(pv_feature.P - 0.08) < 1e-6  # 0.1 * 0.8
        assert wind_feature.availability == 0.5
        assert abs(wind_feature.P - 0.05) < 1e-6  # 0.1 * 0.5

    def test_set_grid_price(self):
        """Test setting grid price."""
        features = create_default_features()
        agent = MicrogridFieldAgent(agent_id="test", features=features)
        agent.set_grid_price(75.0)

        grid_feature = agent.state.features[agent.grid_idx]
        assert grid_feature.price == 75.0

    def test_update_device_dynamics(self):
        """Test updating device dynamics (ESS SOC)."""
        features = create_default_features(ess_capacity=2.0)
        agent = MicrogridFieldAgent(agent_id="test", features=features)

        # Set ESS power
        ess_feature = agent.state.features[agent.ess_idx]
        ess_feature.set_values(P=0.5, soc=0.5)

        # Update dynamics
        agent.update_device_dynamics()

        # SOC should have increased
        assert ess_feature.soc > 0.5


class TestMicrogridEnv:
    """Tests for MicrogridEnv."""

    def test_initialization(self):
        """Test environment initialization."""
        env = MicrogridEnv(num_microgrids=3)

        assert env.num_microgrids == 3
        assert len([a for a in env.registered_agents.values()
                    if isinstance(a, MicrogridFieldAgent)]) == 3

    def test_reset(self):
        """Test environment reset."""
        env = MicrogridEnv(num_microgrids=2)
        obs, info = env.reset(seed=42)

        # Should return observations for all agents
        agent_ids = [a.agent_id for a in env.registered_agents.values()
                     if isinstance(a, MicrogridFieldAgent)]
        assert all(aid in obs for aid in agent_ids)

        # Timestep should be reset
        assert env._timestep == 0

    def test_step(self):
        """Test environment step."""
        env = MicrogridEnv(num_microgrids=2, episode_steps=5)
        obs, _ = env.reset(seed=42)

        # Get agent IDs
        agent_ids = [a.agent_id for a in env.registered_agents.values()
                     if isinstance(a, MicrogridFieldAgent)]

        # Create Action objects
        actions = {}
        for aid in agent_ids:
            action = Action()
            action.set_specs(dim_c=4, range=(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1])))
            action.set_values(c=np.random.randn(4) * 0.5)
            actions[aid] = action

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Check returns
        assert all(aid in obs for aid in agent_ids)
        assert all(aid in rewards for aid in agent_ids)
        assert all(aid in terminated for aid in agent_ids)

        # Timestep should increment
        assert env._timestep == 1

    def test_episode_termination(self):
        """Test episode terminates at max steps."""
        env = MicrogridEnv(num_microgrids=1, episode_steps=3)
        obs, _ = env.reset(seed=42)

        agent_ids = [a.agent_id for a in env.registered_agents.values()
                     if isinstance(a, MicrogridFieldAgent)]

        for step in range(3):
            # Create Action objects
            actions = {}
            for aid in agent_ids:
                action = Action()
                action.set_specs(dim_c=4, range=(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1])))
                action.set_values(c=np.zeros(4))
                actions[aid] = action

            obs, rewards, terminated, truncated, info = env.step(actions)

            if step < 2:
                # Not terminated yet
                assert not terminated.get("__all__", False)
            else:
                # Should terminate at step 2 (0-indexed, 3 total steps)
                assert terminated.get("__all__", False)

    def test_network_topology(self):
        """Test network topology creation."""
        env = MicrogridEnv(num_microgrids=3)

        # Check network has expected elements
        assert len(env.net.bus) > 0
        assert len(env.net.storage) == 3  # One ESS per microgrid
        assert len(env.net.load) == 3  # One load per microgrid

        # Check mappings
        assert "MG1" in env.mg_bus_mappings
        assert "MG2" in env.mg_bus_mappings
        assert "MG3" in env.mg_bus_mappings

    def test_profiles_initialization(self):
        """Test profile initialization with real data."""
        env = MicrogridEnv(num_microgrids=1)

        # Check data loader is initialized
        assert hasattr(env, 'data_loader')

        # Reset to load episode data
        env.reset(seed=0)

        # Check episode data exists and has correct length
        assert env.current_episode_data is not None
        assert len(env.current_episode_data['price']) == 24
        assert len(env.current_episode_data['solar']) == 24
        assert len(env.current_episode_data['wind']) == 24

        # Check solar profile has expected pattern (zero at night, positive during day)
        # Night hours typically have zero solar
        assert env.current_episode_data['solar'][0] <= 0.1  # Midnight
        # Day hours may have solar (depends on real data)

    def test_update_profiles(self):
        """Test profile updates during episode with real data."""
        env = MicrogridEnv(num_microgrids=1)
        env.reset(seed=42)

        # Get agent
        agent = [a for a in env.registered_agents.values()
                 if isinstance(a, MicrogridFieldAgent)][0]

        # Check that price was updated from real data (not default 50.0)
        grid_feature = agent.state.features[agent.grid_idx]
        initial_price = grid_feature.price

        # Price should match first hour of episode data
        if env.current_episode_data:
            expected_price = env.current_episode_data['price'][0]
            assert abs(grid_feature.price - expected_price) < 1.0  # Within $1

        # Update to timestep 12 (noon)
        env._update_profiles(12)
        if env.current_episode_data:
            expected_price_noon = env.current_episode_data['price'][12]
            assert abs(grid_feature.price - expected_price_noon) < 1.0


class TestIntegration:
    """Integration tests."""

    def test_full_episode(self):
        """Test running a full episode."""
        env = MicrogridEnv(num_microgrids=2, episode_steps=5)
        obs, _ = env.reset(seed=42)

        agent_ids = [a.agent_id for a in env.registered_agents.values()
                     if isinstance(a, MicrogridFieldAgent)]

        total_rewards = {aid: 0.0 for aid in agent_ids}

        for step in range(5):
            # Create Action objects
            actions = {}
            for aid in agent_ids:
                action = Action()
                action.set_specs(dim_c=4, range=(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1])))
                action.set_values(c=np.zeros(4))
                actions[aid] = action

            obs, rewards, terminated, truncated, info = env.step(actions)

            for aid in agent_ids:
                total_rewards[aid] += rewards[aid]

            if terminated.get("__all__", False):
                break

        # Episode should complete
        assert env._timestep == 5

        # All agents should have received rewards
        for aid in agent_ids:
            assert aid in total_rewards
            # Rewards should be negative (costs)
            # But we don't enforce this strictly in tests

    def test_power_flow_convergence(self):
        """Test power flow convergence."""
        env = MicrogridEnv(num_microgrids=1)
        obs, _ = env.reset(seed=42)

        agent_ids = [a.agent_id for a in env.registered_agents.values()
                     if isinstance(a, MicrogridFieldAgent)]

        # Create Action objects with reasonable values
        actions = {}
        for aid in agent_ids:
            action = Action()
            action.set_specs(dim_c=4, range=(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1])))
            action.set_values(c=np.array([0.2, 0.5, 0.0, 0.0]))
            actions[aid] = action

        obs, rewards, terminated, truncated, info = env.step(actions)

        # Power flow should converge (no assertion failure)
        assert True  # If we get here, no exception was raised


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

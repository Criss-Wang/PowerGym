"""Tests for the Thermostat-v0 demo environment."""

import numpy as np
import pytest

from heron.demo_envs.thermostat.env import build_thermostat_env
from heron.demo_envs.thermostat.agents import HeaterAgent, TemperatureFeature
from heron.registry import _registry


class TestThermostatRegistration:
    def test_auto_registered(self):
        import heron.demo_envs  # noqa: F401
        assert "Thermostat-v0" in _registry

    def test_make_creates_env(self):
        import heron
        import heron.demo_envs  # noqa: F401
        env = heron.make("Thermostat-v0")
        assert env is not None
        env.close()

    def test_make_override_kwargs(self):
        import heron
        import heron.demo_envs  # noqa: F401
        env = heron.make("Thermostat-v0", max_steps=50, target_temp=25.0)
        assert env.termination_config.max_steps == 50
        env.close()


class TestThermostatEnv:
    @pytest.fixture
    def env(self):
        e = build_thermostat_env(
            target_temp=22.0,
            initial_temp=18.0,
            max_steps=10,
            noise_scale=0.0,  # deterministic for testing
        )
        yield e
        e.close()

    def test_reset_returns_obs(self, env):
        result = env.reset()
        assert result is not None

    def test_step_returns_5_tuple(self, env):
        env.reset()
        result = env.step({"heater": np.array([0.5])})
        assert len(result) == 5
        obs, rewards, terminated, truncated, infos = result
        assert "heater" in rewards
        assert "__all__" in terminated
        assert "__all__" in truncated

    def test_reward_is_negative_distance(self, env):
        env.reset()
        _, rewards, _, _, _ = env.step({"heater": np.array([0.0])})
        # Reward should be negative (distance from target=22)
        assert rewards["heater"] < 0

    def test_truncation_at_max_steps(self, env):
        env.reset()
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step({"heater": np.array([0.0])})
        assert truncated["__all__"] is True

    def test_heating_increases_temperature(self, env):
        env.reset()
        # Apply positive heat for several steps
        for _ in range(5):
            env.step({"heater": np.array([1.0])})
        # Get heater state — temperature should have increased from 18
        heater = env.get_agent("heater")
        temp = heater.state.features["TemperatureFeature"].temperature
        # Each step: T + 1.0 (heat) - 0.1 (cooling) = +0.9 net
        # After 5 steps: 18 + 5*0.9 = 22.5
        assert temp > 20.0


    def test_initial_temp_preserved_across_resets(self, env):
        env.reset()
        heater = env.get_agent("heater")
        temp = heater.state.features["TemperatureFeature"].temperature
        assert temp == pytest.approx(18.0)

        # Step a few times then reset again
        for _ in range(3):
            env.step({"heater": np.array([1.0])})
        env.reset()
        temp_after = heater.state.features["TemperatureFeature"].temperature
        assert temp_after == pytest.approx(18.0)


class TestHeaterAgent:
    def test_action_space(self):
        agent = HeaterAgent(agent_id="h1")
        assert agent.action.dim_c == 1
        lb, ub = agent.action.range
        assert float(lb[0]) == -1.0
        assert float(ub[0]) == 1.0

    def test_compute_local_reward(self):
        agent = HeaterAgent(agent_id="h1", target_temp=22.0)
        local_state = {
            "features": {"TemperatureFeature": {"temperature": 20.0}}
        }
        reward = agent.compute_local_reward(local_state)
        assert reward == pytest.approx(-2.0)

    def test_compute_local_reward_at_target(self):
        agent = HeaterAgent(agent_id="h1", target_temp=22.0)
        local_state = {
            "features": {"TemperatureFeature": {"temperature": 22.0}}
        }
        reward = agent.compute_local_reward(local_state)
        assert reward == pytest.approx(0.0)

"""Tests for the TwoRoomHeating v0–v7 progressive demo environments."""

import numpy as np
import pytest

from heron.demo_envs.two_room_heating.features import (
    ZoneTemperatureFeature,
    VentStatusFeature,
)
from heron.demo_envs.two_room_heating.agents import HeaterAgent, VentAgent
from heron.registry import _registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env_v0():
    from heron.demo_envs.two_room_heating.env import build_v0

    e = build_v0(
        target_temp=22.0,
        initial_temp_a=18.0,
        initial_temp_b=18.0,
        max_steps=10,
        cooling_rate=0.0,
        coupling_rate=0.0,
    )
    yield e
    e.close()


@pytest.fixture
def env_v1():
    from heron.demo_envs.two_room_heating.env import build_v1

    e = build_v1(max_steps=10, cooling_rate=0.0, coupling_rate=0.0)
    yield e
    e.close()


@pytest.fixture
def env_v2():
    from heron.demo_envs.two_room_heating.env import build_v2

    e = build_v2(max_steps=10, cooling_rate=0.0, coupling_rate=0.0)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Registration tests (all versions)
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_all_versions_registered(self):
        import heron.demo_envs  # noqa: F401

        for v in range(8):
            assert f"TwoRoomHeating-v{v}" in _registry

    def test_make_v0(self):
        import heron
        import heron.demo_envs  # noqa: F401

        env = heron.make("TwoRoomHeating-v0")
        assert env is not None
        env.close()

    def test_make_override_kwargs(self):
        import heron
        import heron.demo_envs  # noqa: F401

        env = heron.make("TwoRoomHeating-v0", max_steps=50, target_temp=25.0)
        assert env.termination_config.max_steps == 50
        env.close()

    def test_levels_metadata(self):
        from heron.demo_envs.two_room_heating import LEVELS

        assert len(LEVELS) == 8
        for i, lvl in enumerate(LEVELS):
            assert lvl["version"] == i
            assert "title" in lvl
            assert "concept" in lvl


# ---------------------------------------------------------------------------
# v0: Heterogeneous tick rates
# ---------------------------------------------------------------------------


class TestV0:
    def test_reset_returns_obs(self, env_v0):
        result = env_v0.reset()
        assert result is not None

    def test_step_returns_5_tuple(self, env_v0):
        env_v0.reset()
        result = env_v0.step({
            "heater_a": np.array([0.5]),
            "heater_b": np.array([0.3]),
        })
        assert len(result) == 5
        obs, rewards, terminated, truncated, infos = result
        assert "heater_a" in rewards
        assert "heater_b" in rewards
        assert "__all__" in terminated
        assert "__all__" in truncated

    def test_reward_is_negative(self, env_v0):
        env_v0.reset()
        _, rewards, _, _, _ = env_v0.step({
            "heater_a": np.array([0.0]),
            "heater_b": np.array([0.0]),
        })
        assert rewards["heater_a"] < 0
        assert rewards["heater_b"] < 0

    def test_truncation_at_max_steps(self, env_v0):
        env_v0.reset()
        for _ in range(10):
            _, _, terminated, truncated, _ = env_v0.step({
                "heater_a": np.array([0.0]),
                "heater_b": np.array([0.0]),
            })
        assert truncated["__all__"] is True

    def test_heating_increases_temperature(self, env_v0):
        env_v0.reset()
        for _ in range(5):
            env_v0.step({
                "heater_a": np.array([1.0]),
                "heater_b": np.array([1.0]),
            })
        heater_a = env_v0.get_agent("heater_a")
        temp_a = heater_a.state.features["ZoneTemperatureFeature"].temperature
        assert temp_a > 18.0

    def test_different_heat_gains(self):
        ha = HeaterAgent(agent_id="a", heat_gain=2.0)
        hb = HeaterAgent(agent_id="b", heat_gain=3.0)
        # heat_gain is stored correctly
        assert hb.heat_gain > ha.heat_gain

    def test_initial_temp_preserved_across_resets(self):
        from heron.demo_envs.two_room_heating.env import build_v0

        env = build_v0(
            initial_temp_a=15.0,
            initial_temp_b=25.0,
            max_steps=10,
            cooling_rate=0.0,
            coupling_rate=0.0,
        )
        env.reset()
        ha = env.get_agent("heater_a")
        hb = env.get_agent("heater_b")
        assert ha.state.features["ZoneTemperatureFeature"].temperature == pytest.approx(15.0)
        assert hb.state.features["ZoneTemperatureFeature"].temperature == pytest.approx(25.0)

        # Step then reset — should restore
        env.step({"heater_a": np.array([1.0]), "heater_b": np.array([-1.0])})
        env.reset()
        assert ha.state.features["ZoneTemperatureFeature"].temperature == pytest.approx(15.0)
        assert hb.state.features["ZoneTemperatureFeature"].temperature == pytest.approx(25.0)
        env.close()

    def test_cross_zone_coupling(self):
        from heron.demo_envs.two_room_heating.env import build_v0

        env = build_v0(
            initial_temp_a=10.0,
            initial_temp_b=30.0,
            max_steps=50,
            cooling_rate=0.0,
            coupling_rate=0.1,  # strong coupling
        )
        env.reset()
        for _ in range(20):
            env.step({"heater_a": np.array([0.0]), "heater_b": np.array([0.0])})
        ha = env.get_agent("heater_a")
        hb = env.get_agent("heater_b")
        temp_a = ha.state.features["ZoneTemperatureFeature"].temperature
        temp_b = hb.state.features["ZoneTemperatureFeature"].temperature
        # Temps should converge toward each other
        assert abs(temp_a - temp_b) < 20.0  # started 20 apart
        env.close()


# ---------------------------------------------------------------------------
# v1: Coordinator
# ---------------------------------------------------------------------------


class TestV1:
    def test_has_coordinator(self, env_v1):
        agent = env_v1.get_agent("building_ctrl")
        assert agent is not None

    def test_step_with_coordinator(self, env_v1):
        env_v1.reset()
        result = env_v1.step({
            "heater_a": np.array([0.5]),
            "heater_b": np.array([0.3]),
        })
        assert len(result) == 5


# ---------------------------------------------------------------------------
# v2: Reactive vent
# ---------------------------------------------------------------------------


class TestV2:
    def test_has_vent(self, env_v2):
        agent = env_v2.get_agent("vent")
        assert agent is not None

    def test_vent_action_space(self):
        agent = VentAgent(agent_id="v1")
        assert agent.action.dim_c == 1
        lb, ub = agent.action.range
        assert float(lb[0]) == 0.0
        assert float(ub[0]) == 1.0


# ---------------------------------------------------------------------------
# Agent unit tests
# ---------------------------------------------------------------------------


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
            "features": {"ZoneTemperatureFeature": {"temperature": 20.0}}
        }
        reward = agent.compute_local_reward(local_state)
        assert reward == pytest.approx(-2.0)

    def test_compute_local_reward_at_target(self):
        agent = HeaterAgent(agent_id="h1", target_temp=22.0)
        local_state = {
            "features": {"ZoneTemperatureFeature": {"temperature": 22.0}}
        }
        reward = agent.compute_local_reward(local_state)
        assert reward == pytest.approx(0.0)

    def test_heat_gain_parameter(self):
        agent = HeaterAgent(agent_id="h1", heat_gain=5.0)
        assert agent.heat_gain == 5.0


class TestVentAgent:
    def test_compute_local_reward(self):
        agent = VentAgent(agent_id="v1")
        local_state = {
            "features": {"VentStatusFeature": {"is_open": 1.0, "cooling_power": 2.0}}
        }
        reward = agent.compute_local_reward(local_state)
        assert reward == pytest.approx(-0.5)

    def test_compute_local_reward_closed(self):
        agent = VentAgent(agent_id="v1")
        local_state = {
            "features": {"VentStatusFeature": {"is_open": 0.0, "cooling_power": 0.0}}
        }
        reward = agent.compute_local_reward(local_state)
        assert reward == pytest.approx(0.0)

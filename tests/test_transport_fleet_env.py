"""Tests for the TransportFleet-v0 demo environment."""

import numpy as np
import pytest

from heron.demo_envs.transport_fleet.env import build_transport_fleet_env
from heron.demo_envs.transport_fleet.agents import (
    DepotCoordinator,
    DepotFeature,
    VehicleAgent,
    VehicleFeature,
)
from heron.registry import _registry


def _get_delivery_points(sim_func):
    """Extract delivery_points list from the sim_func closure."""
    for cell in sim_func.__closure__:
        contents = cell.cell_contents
        if isinstance(contents, list) and all(
            isinstance(item, list) and len(item) == 2 for item in contents
        ):
            return contents
    raise RuntimeError("Could not find delivery_points in closure")


class TestTransportFleetRegistration:
    def test_auto_registered(self):
        import heron.demo_envs  # noqa: F401
        assert "TransportFleet-v0" in _registry

    def test_make_creates_env(self):
        import heron
        import heron.demo_envs  # noqa: F401
        env = heron.make("TransportFleet-v0")
        assert env is not None
        env.close()

    def test_make_override_kwargs(self):
        import heron
        import heron.demo_envs  # noqa: F401
        env = heron.make("TransportFleet-v0", max_steps=50, n_vehicles=2)
        assert env.termination_config.max_steps == 50
        env.close()


class TestTransportFleetEnv:
    @pytest.fixture
    def env(self):
        e = build_transport_fleet_env(
            n_vehicles=3,
            max_steps=20,
            fuel_capacity=100.0,
            breakdown_prob=0.0,   # deterministic: no breakdowns
            new_delivery_prob=0.0,  # deterministic: no new deliveries
            initial_deliveries=3,
            seed=42,
        )
        yield e
        e.close()

    def _zero_actions(self):
        return {
            "vehicle_0": np.array([0.0, 0.0]),
            "vehicle_1": np.array([0.0, 0.0]),
            "vehicle_2": np.array([0.0, 0.0]),
        }

    def _move_actions(self):
        return {
            "vehicle_0": np.array([1.0, 0.0]),
            "vehicle_1": np.array([0.0, 1.0]),
            "vehicle_2": np.array([-1.0, -1.0]),
        }

    def test_reset_returns_obs(self, env):
        result = env.reset()
        assert result is not None

    def test_reset_obs_contains_all_vehicles(self, env):
        obs = env.reset()
        # obs is a tuple (obs_dict, info_dict)
        obs_dict = obs[0] if isinstance(obs, tuple) else obs
        for vid in ["vehicle_0", "vehicle_1", "vehicle_2"]:
            assert vid in obs_dict, f"Missing {vid} in observations"

    def test_step_returns_5_tuple(self, env):
        env.reset()
        result = env.step(self._zero_actions())
        assert len(result) == 5
        obs, rewards, terminated, truncated, infos = result
        assert "__all__" in terminated
        assert "__all__" in truncated

    def test_step_rewards_for_vehicles_and_depot(self, env):
        env.reset()
        _, rewards, _, _, _ = env.step(self._zero_actions())
        for vid in ["vehicle_0", "vehicle_1", "vehicle_2"]:
            assert vid in rewards, f"Missing reward for {vid}"
        assert "depot" in rewards, "Missing reward for depot coordinator"

    def test_vehicles_consume_fuel_when_moving(self, env):
        env.reset()
        env.step(self._move_actions())
        v0 = env.get_agent("vehicle_0")
        fuel = v0.state.features["VehicleFeature"].fuel
        assert fuel < 100.0, "Fuel should decrease after movement"

    def test_vehicles_stay_still_with_zero_action(self, env):
        env.reset()
        env.step(self._zero_actions())
        v0 = env.get_agent("vehicle_0")
        fuel = v0.state.features["VehicleFeature"].fuel
        assert fuel == 100.0, "Fuel should not change with zero movement"

    def test_delivery_completion(self):
        """Place a vehicle exactly at a delivery point and verify completion."""
        env = build_transport_fleet_env(
            n_vehicles=1,
            max_steps=50,
            fuel_capacity=200.0,
            breakdown_prob=0.0,
            new_delivery_prob=0.0,
            initial_deliveries=1,
            seed=0,
        )
        env.reset()

        # Extract delivery_points from the simulation closure
        sim_func = env._user_simulation_func
        delivery_points = _get_delivery_points(sim_func)

        assert len(delivery_points) > 0, "Should have initial delivery points"
        target = delivery_points[0]

        v = env.get_agent("vehicle")
        vx = v.state.features["VehicleFeature"].x
        vy = v.state.features["VehicleFeature"].y
        dx = target[0] - vx
        dy = target[1] - vy
        dist = np.sqrt(dx * dx + dy * dy)
        steps_needed = max(1, int(np.ceil(dist)))
        step_dx = np.clip(dx / max(steps_needed, 1), -1.0, 1.0)
        step_dy = np.clip(dy / max(steps_needed, 1), -1.0, 1.0)

        for _ in range(steps_needed + 2):
            env.step({"vehicle": np.array([step_dx, step_dy])})

        vf = env.get_agent("vehicle").state.features["VehicleFeature"]
        assert vf.deliveries >= 1, "Vehicle should have completed at least 1 delivery"

        env.close()

    def test_coordinator_reward_reflects_team_deliveries(self):
        """Coordinator reward should increase when a delivery is completed."""
        env = build_transport_fleet_env(
            n_vehicles=1,
            max_steps=50,
            fuel_capacity=200.0,
            breakdown_prob=0.0,
            new_delivery_prob=0.0,
            initial_deliveries=1,
            seed=0,
        )
        env.reset()

        v = env.get_agent("vehicle")
        sim_func = env._user_simulation_func
        delivery_points = _get_delivery_points(sim_func)

        total_depot_reward = 0.0
        if len(delivery_points) > 0:
            target = delivery_points[0]
            vx = v.state.features["VehicleFeature"].x
            vy = v.state.features["VehicleFeature"].y
            dx = target[0] - vx
            dy = target[1] - vy
            dist = np.sqrt(dx * dx + dy * dy)
            steps_needed = max(1, int(np.ceil(dist)))
            step_dx = np.clip(dx / max(steps_needed, 1), -1.0, 1.0)
            step_dy = np.clip(dy / max(steps_needed, 1), -1.0, 1.0)

            for _ in range(steps_needed + 2):
                _, rewards, _, _, _ = env.step(
                    {"vehicle": np.array([step_dx, step_dy])}
                )
                total_depot_reward += rewards.get("depot", 0.0)

        assert total_depot_reward > 0.0, "Depot reward should be positive after deliveries"
        env.close()

    def test_truncation_at_max_steps(self, env):
        env.reset()
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(self._zero_actions())
        assert truncated["__all__"] is True


class TestVehicleAgent:
    def test_action_space(self):
        agent = VehicleAgent(agent_id="v1")
        assert agent.action.dim_c == 2
        lb, ub = agent.action.range
        assert float(lb[0]) == -1.0
        assert float(ub[1]) == 1.0

    def test_compute_local_reward_broken(self):
        agent = VehicleAgent(agent_id="v1")
        agent.state.features["VehicleFeature"].is_broken = True
        reward = agent.compute_local_reward({})
        assert reward == pytest.approx(-0.1)


class TestDepotCoordinator:
    def test_compute_local_reward_delta(self):
        depot = DepotCoordinator(agent_id="depot")
        depot.state.features["DepotFeature"].total_deliveries = 0
        r1 = depot.compute_local_reward({})
        assert r1 == pytest.approx(0.0)

        depot.state.features["DepotFeature"].total_deliveries = 2
        r2 = depot.compute_local_reward({})
        assert r2 == pytest.approx(2.0)

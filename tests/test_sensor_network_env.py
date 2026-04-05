"""Tests for the SensorNetwork-v0 demo environment."""

import numpy as np
import pytest

from heron.demo_envs.sensor_network.env import (
    build_sensor_network_env,
    _generate_random_graph,
)
from heron.demo_envs.sensor_network.agents import SensorAgent, SensorFeature
from heron.registry import _registry


class TestSensorNetworkRegistration:
    def test_auto_registered(self):
        import heron.demo_envs  # noqa: F401
        assert "SensorNetwork-v0" in _registry

    def test_make_creates_env(self):
        import heron
        import heron.demo_envs  # noqa: F401
        env = heron.make("SensorNetwork-v0")
        assert env is not None
        env.close()

    def test_make_override_kwargs(self):
        import heron
        import heron.demo_envs  # noqa: F401
        env = heron.make("SensorNetwork-v0", max_steps=25, n_sensors=3)
        assert env.termination_config.max_steps == 25
        env.close()


class TestSensorNetworkEnv:
    @pytest.fixture
    def env(self):
        np.random.seed(42)
        e = build_sensor_network_env(
            n_sensors=5,
            connectivity=0.4,
            spread_prob=0.3,
            max_steps=10,
            graph_seed=42,
        )
        yield e
        e.close()

    def test_reset_returns_obs(self, env):
        result = env.reset()
        assert result is not None

    def test_reset_returns_obs_for_all_sensors(self, env):
        obs, infos = env.reset()
        for i in range(5):
            assert f"sensor_{i}" in obs

    def test_step_returns_5_tuple(self, env):
        env.reset()
        actions = {f"sensor_{i}": 1 for i in range(5)}
        result = env.step(actions)
        assert len(result) == 5
        obs, rewards, terminated, truncated, infos = result
        for i in range(5):
            assert f"sensor_{i}" in rewards
        assert "__all__" in terminated
        assert "__all__" in truncated

    def test_step_rewards_are_floats(self, env):
        env.reset()
        actions = {f"sensor_{i}": 0 for i in range(5)}
        _, rewards, _, _, _ = env.step(actions)
        for aid, reward in rewards.items():
            if aid == "__all__":
                continue
            assert isinstance(reward, float)

    def test_truncation_at_max_steps(self, env):
        env.reset()
        for _ in range(10):
            actions = {f"sensor_{i}": 0 for i in range(5)}
            _, _, terminated, truncated, _ = env.step(actions)
        assert truncated["__all__"] is True

    def test_detection_action_sets_feature(self, env):
        env.reset()
        # All sensors detect
        actions = {f"sensor_{i}": 1 for i in range(5)}
        env.step(actions)
        for i in range(5):
            agent = env.get_agent(f"sensor_{i}")
            detection = agent.state.features["SensorFeature"].detection
            assert detection == 1.0

    def test_no_detection_action(self, env):
        env.reset()
        actions = {f"sensor_{i}": 0 for i in range(5)}
        env.step(actions)
        for i in range(5):
            agent = env.get_agent(f"sensor_{i}")
            detection = agent.state.features["SensorFeature"].detection
            assert detection == 0.0


class TestGraphGeneration:
    def test_correct_number_of_nodes(self):
        adj = _generate_random_graph(10, 0.5, seed=1)
        assert len(adj) == 10

    def test_all_nodes_have_neighbors(self):
        adj = _generate_random_graph(8, 0.1, seed=2)
        for nid, neighbors in adj.items():
            assert len(neighbors) > 0, f"Node {nid} has no neighbors"

    def test_symmetry(self):
        adj = _generate_random_graph(6, 0.5, seed=3)
        for nid, neighbors in adj.items():
            for nbr in neighbors:
                assert nid in adj[nbr], f"{nid} -> {nbr} but not {nbr} -> {nid}"

    def test_high_connectivity_many_edges(self):
        adj = _generate_random_graph(5, 1.0, seed=4)
        # Full connectivity: each node connected to all 4 others
        for nid, neighbors in adj.items():
            assert len(neighbors) == 4


class TestVisibilityScoping:
    """Test that agents only see neighbor information via horizontal protocol."""

    def test_neighbor_avg_detection_reflects_neighbors(self):
        """After a step, neighbor_avg_detection should reflect neighbor detections."""
        env = build_sensor_network_env(
            n_sensors=5,
            connectivity=1.0,  # fully connected for predictable test
            spread_prob=0.0,   # no signal spread for clarity
            max_steps=10,
            graph_seed=42,
        )
        env.reset()
        # sensor_0 detects, others do not
        actions = {
            "sensor_0": 1,
            "sensor_1": 0,
            "sensor_2": 0,
            "sensor_3": 0,
            "sensor_4": 0,
        }
        env.step(actions)
        # Action effects are local to agent state; the simulation reads from
        # the proxy's global state which lags by one step.  Run a second step
        # so the sim sees the detection values set by the first step.
        env.step(actions)
        # In a fully connected graph, sensor_1 has 4 neighbors.
        # Only sensor_0 detected -> avg = 1/4 = 0.25
        s1 = env.get_agent("sensor_1")
        feat = s1.state.features.get("SensorFeature")
        assert feat.neighbor_avg_detection == pytest.approx(0.25)
        env.close()

    def test_isolated_pair_sees_only_partner(self):
        """With low connectivity, agents should only see their direct neighbors."""
        # connectivity=0.0 forces isolate-prevention to create minimal edges
        adj = _generate_random_graph(5, connectivity=0.0, seed=10)
        # Verify the topology is sparse (not fully connected)
        total_edges = sum(len(nbrs) for nbrs in adj.values()) // 2
        assert total_edges < 10  # 10 = C(5,2) = fully connected


class TestSensorAgent:
    def test_action_space_is_discrete_binary(self):
        agent = SensorAgent(agent_id="s0")
        assert agent.action.dim_d == 1
        assert agent.action.ncats == [2]

    def test_compute_local_reward_true_positive(self):
        agent = SensorAgent(agent_id="s0")
        local_state = {
            "features": {"SensorFeature": {"signal_strength": 0.8, "detection": 1.0}}
        }
        assert agent.compute_local_reward(local_state) == 1.0

    def test_compute_local_reward_false_positive(self):
        agent = SensorAgent(agent_id="s0")
        local_state = {
            "features": {"SensorFeature": {"signal_strength": 0.3, "detection": 1.0}}
        }
        assert agent.compute_local_reward(local_state) == -1.0

    def test_compute_local_reward_true_negative(self):
        agent = SensorAgent(agent_id="s0")
        local_state = {
            "features": {"SensorFeature": {"signal_strength": 0.2, "detection": 0.0}}
        }
        assert agent.compute_local_reward(local_state) == 0.5

    def test_compute_local_reward_false_negative(self):
        agent = SensorAgent(agent_id="s0")
        local_state = {
            "features": {"SensorFeature": {"signal_strength": 0.9, "detection": 0.0}}
        }
        assert agent.compute_local_reward(local_state) == 0.0

"""
Tests for heron.envs.adapters.pettingzoo module.

"""

import pytest
import numpy as np
import gymnasium as gym

from heron.envs.base import HeronEnvCore
from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.messaging.in_memory_broker import InMemoryBroker


# ----------------------------
# Optional PettingZoo import
# ----------------------------
pz = pytest.importorskip("pettingzoo")
from pettingzoo import ParallelEnv  # noqa: E402

from heron.envs.adapters.pettingzoo import PettingZooParallelEnv  # noqa: E402


class MockAgent(Agent):
    """Mock HERON agent for adapter tests."""

    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id=agent_id, level=1, **kwargs)
        # tests may override these
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def observe(self, global_state=None, *args, **kwargs):
        return Observation(local={"value": 1.0}, timestamp=self._timestep)

    def act(self, observation, *args, **kwargs):
        return np.zeros(self.action_space.shape, dtype=np.float32)


class MockCoordinator(CoordinatorAgent):
    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id=agent_id, config={}, **kwargs)

    def _build_subordinate_agents(self, agent_configs, env_id=None, upstream_id=None):
        return {}


class ConcretePZEnv(PettingZooParallelEnv):
    def __init__(self, env_id=None, message_broker=None):
        super().__init__(env_id=env_id, message_broker=message_broker)

        # Define agents + spaces (like typical usage)
        self.set_agent_ids(["agent_1", "agent_2"])
        self.init_spaces(
            observation_spaces={
                "agent_1": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
                "agent_2": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            },
            action_spaces={
                "agent_1": gym.spaces.Discrete(2),
                "agent_2": gym.spaces.Discrete(2),
            },
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = list(self.possible_agents)

        obs = {a: np.zeros(self.observation_space(a).shape, dtype=np.float32) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        # If no actions or no agents -> terminate
        if not self.agents or not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        obs = {a: np.ones(self.observation_space(a).shape, dtype=np.float32) for a in self.agents}

        rewards = {a: float(actions.get(a, 0) == 1) for a in self.agents}

        terminations = {a: False for a in self.agents}
        truncations = {a: True for a in self.agents}

        infos = {a: {} for a in self.agents}

        self.agents = []
        return obs, rewards, terminations, truncations, infos


class TestPettingZooParallelEnvAdapter:
    def test_init_generates_env_id(self):
        env = ConcretePZEnv()
        assert env.env_id is not None
        assert env.env_id.startswith("env_")

    def test_init_with_custom_env_id(self):
        env = ConcretePZEnv(env_id="my_env")
        assert env.env_id == "my_env"

    def test_init_creates_message_broker(self):
        env = ConcretePZEnv()
        assert env.message_broker is not None
        assert isinstance(env.message_broker, InMemoryBroker)

    def test_init_with_custom_broker(self):
        broker = InMemoryBroker()
        env = ConcretePZEnv(message_broker=broker)
        assert env.message_broker is broker

    def test_set_agent_ids_updates_agents(self):
        env = ConcretePZEnv()

        env.set_agent_ids(["a", "b", "c"])
        assert env.possible_agents == ["a", "b", "c"]

        env.init_spaces(
            observation_spaces={
                "a": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
                "b": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
                "c": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            },
            action_spaces={
                "a": gym.spaces.Discrete(2),
                "b": gym.spaces.Discrete(2),
                "c": gym.spaces.Discrete(2),
            },
        )

        obs, infos = env.reset()
        assert set(obs.keys()) == {"a", "b", "c"}
        assert set(infos.keys()) == {"a", "b", "c"}

    def test_init_spaces_and_space_accessors(self):
        env = ConcretePZEnv()

        assert "agent_1" in env.observation_spaces
        assert "agent_2" in env.action_spaces

        assert env.observation_space("agent_1") == env.observation_spaces["agent_1"]
        assert env.action_space("agent_2") == env.action_spaces["agent_2"]

    def test_core_delegation_register_agent_and_observe(self):
        env = ConcretePZEnv()
        a1 = MockAgent(agent_id="agent_1")
        a2 = MockAgent(agent_id="agent_2")
        env.register_agents([a1, a2])

        obs = env.get_observations()
        assert set(obs.keys()) == {"agent_1", "agent_2"}
        assert isinstance(obs["agent_1"], Observation)

    def test_core_delegation_apply_actions(self):
        env = ConcretePZEnv()
        env.register_agents([MockAgent(agent_id="agent_1"), MockAgent(agent_id="agent_2")])

        env.apply_actions(
            {"agent_1": np.array([0.1], dtype=np.float32), "agent_2": np.array([0.2], dtype=np.float32)}
        )

    def test_reset_and_step_shapes(self):
        env = ConcretePZEnv()
        obs, infos = env.reset(seed=123)

        assert set(obs.keys()) == {"agent_1", "agent_2"}
        assert set(infos.keys()) == {"agent_1", "agent_2"}

        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs2, rewards, terminations, truncations, infos2 = env.step(actions)

        # Parallel API contract: dicts keyed by agent id
        assert set(obs2.keys()) == {"agent_1", "agent_2"}
        assert set(rewards.keys()) == {"agent_1", "agent_2"}
        assert set(terminations.keys()) == {"agent_1", "agent_2"}
        assert set(truncations.keys()) == {"agent_1", "agent_2"}
        assert set(infos2.keys()) == {"agent_1", "agent_2"}

        assert env.agents == []

    def test_close_does_not_raise(self):
        env = ConcretePZEnv()
        env.close()
        env.close_heron()

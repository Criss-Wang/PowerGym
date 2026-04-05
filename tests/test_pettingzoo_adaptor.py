"""Tests for the PettingZoo Parallel API adaptor.

Structured to match the RLlib adaptor's validation rigour:
  - Unit tests against a demo env (TwoRoomHeating-v0)
  - Integration tests with a custom multi-agent env (action round-trip)
  - Coordinator-exclusion validation
  - Activity-aware (is_active) info propagation
  - Official PettingZoo parallel_api_test
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np
import pytest

import heron
import heron.demo_envs  # auto-registers demo envs
from heron.adaptors.pettingzoo import PettingZooParallelEnv, pettingzoo_env
from heron.agents.constants import PROXY_AGENT_ID, SYSTEM_AGENT_ID
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.builder import EnvBuilder
from heron.protocols.vertical import VerticalProtocol


# The demo env used for basic API tests.
_DEMO_ENV_ID = "TwoRoomHeating-v0"
_DEMO_MAX_STEPS = 20


# =========================================================================
# Custom agents for integration tests
# =========================================================================

@dataclass(slots=True)
class ValueFeature(Feature):
    """Minimal feature for integration testing."""
    visibility: ClassVar[Sequence[str]] = ("public",)
    value: float = 0.0


class SetterAgent(FieldAgent):
    """Agent whose state becomes the action value — easy to verify round-trips."""

    def __init__(self, agent_id: str, features: Optional[List[Feature]] = None, **kwargs):
        super().__init__(agent_id=agent_id, features=features or [ValueFeature()], **kwargs)

    def init_action(self, features: Optional[List[Feature]] = None) -> Action:
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-10.0]), np.array([10.0])))
        return action

    def set_state(self, *args, **kwargs) -> None:
        pass

    def set_action(self, action: Any, *args, **kwargs) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        self.state.update_feature("ValueFeature", value=float(self.action.c[0]))

    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        vf = self.state.features.get("ValueFeature")
        return float(vf.value) if vf else 0.0


class DummyCoordinator(CoordinatorAgent):
    """Coordinator with a trivial action space — should NOT appear in PettingZoo agents."""
    def compute_local_reward(self, local_state: dict, prev_post_physics_state=None) -> float:
        return 0.0


def _noop_sim(agent_states: Dict[str, Dict]) -> Dict[str, Dict]:
    return agent_states


def _build_custom_env(n_agents: int = 2, max_steps: int = 10, with_coordinator: bool = False):
    """Build a minimal env for integration testing."""
    builder = EnvBuilder("pz_test")
    for i in range(n_agents):
        builder.add_agent(
            f"agent_{i}",
            SetterAgent,
            features=[ValueFeature()],
            coordinator="coord" if with_coordinator else None,
        )
    if with_coordinator:
        builder.add_coordinator(
            "coord",
            agent_cls=DummyCoordinator,
            subordinates=[f"agent_{i}" for i in range(n_agents)],
            protocol=VerticalProtocol(),
        )
    return builder.simulation(_noop_sim).termination(max_steps=max_steps).build()


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def pz_env() -> PettingZooParallelEnv:
    """Wrapper around TwoRoomHeating-v0 demo env."""
    return pettingzoo_env(_DEMO_ENV_ID, max_steps=_DEMO_MAX_STEPS)


@pytest.fixture()
def custom_env() -> PettingZooParallelEnv:
    """Wrapper around a custom 2-agent env for action round-trip tests."""
    return PettingZooParallelEnv(heron_env=_build_custom_env(n_agents=2, max_steps=10))


@pytest.fixture()
def coord_env() -> PettingZooParallelEnv:
    """Wrapper around an env with a coordinator (coordinator must be excluded)."""
    return PettingZooParallelEnv(
        heron_env=_build_custom_env(n_agents=2, max_steps=10, with_coordinator=True)
    )


# =========================================================================
# Agent list tests
# =========================================================================

class TestAgentLists:
    def test_possible_agents_excludes_system_and_proxy(self, pz_env):
        assert SYSTEM_AGENT_ID not in pz_env.possible_agents
        assert PROXY_AGENT_ID not in pz_env.possible_agents

    def test_possible_agents_contains_field_agents(self, pz_env):
        assert "heater_a" in pz_env.possible_agents
        assert "heater_b" in pz_env.possible_agents

    def test_agents_equals_possible_agents_after_init(self, pz_env):
        assert pz_env.agents == pz_env.possible_agents

    def test_coordinator_excluded_from_agents(self, coord_env):
        """Coordinators must never appear in PettingZoo agent lists."""
        assert "coord" not in coord_env.possible_agents
        assert SYSTEM_AGENT_ID not in coord_env.possible_agents
        assert set(coord_env.possible_agents) == {"agent_0", "agent_1"}


# =========================================================================
# reset() tests
# =========================================================================

class TestReset:
    def test_reset_returns_obs_and_infos(self, pz_env):
        obs, infos = pz_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(infos, dict)

    def test_reset_obs_keys_match_agents(self, pz_env):
        obs, infos = pz_env.reset()
        assert set(obs.keys()) == set(pz_env.agents)
        assert set(infos.keys()) == set(pz_env.agents)

    def test_reset_obs_are_numpy_arrays(self, pz_env):
        obs, _ = pz_env.reset()
        for aid, ob in obs.items():
            assert isinstance(ob, np.ndarray), f"obs[{aid}] is not ndarray"
            assert ob.dtype == np.float32

    def test_reset_with_seed(self, pz_env):
        obs1, _ = pz_env.reset(seed=42)
        obs2, _ = pz_env.reset(seed=42)
        for aid in pz_env.agents:
            np.testing.assert_array_equal(obs1[aid], obs2[aid])

    def test_reset_accepts_options(self, pz_env):
        pz_env.reset(seed=0, options={"options": 1})

    def test_reset_restores_agents(self, pz_env):
        obs, _ = pz_env.reset()
        for _ in range(25):
            actions = {aid: pz_env.action_space(aid).sample() for aid in pz_env.agents}
            if not pz_env.agents:
                break
            obs, *_ = pz_env.step(actions)
        pz_env.reset()
        assert pz_env.agents == pz_env.possible_agents


# =========================================================================
# step() tests
# =========================================================================

class TestStep:
    def test_step_returns_five_dicts(self, pz_env):
        pz_env.reset()
        actions = {aid: pz_env.action_space(aid).sample() for aid in pz_env.agents}
        result = pz_env.step(actions)
        assert len(result) == 5
        obs, rewards, terminated, truncated, infos = result
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)
        assert isinstance(infos, dict)

    def test_step_obs_are_numpy_float32(self, pz_env):
        pz_env.reset()
        actions = {aid: pz_env.action_space(aid).sample() for aid in pz_env.agents}
        obs, *_ = pz_env.step(actions)
        for aid, ob in obs.items():
            assert isinstance(ob, np.ndarray)
            assert ob.dtype == np.float32

    def test_step_rewards_are_floats(self, pz_env):
        pz_env.reset()
        actions = {aid: pz_env.action_space(aid).sample() for aid in pz_env.agents}
        _, rewards, *_ = pz_env.step(actions)
        for aid, r in rewards.items():
            assert isinstance(r, float), f"reward[{aid}] is not float"

    def test_step_terminated_truncated_are_bool(self, pz_env):
        pz_env.reset()
        actions = {aid: pz_env.action_space(aid).sample() for aid in pz_env.agents}
        _, _, terminated, truncated, _ = pz_env.step(actions)
        for aid in terminated:
            assert isinstance(terminated[aid], bool)
        for aid in truncated:
            assert isinstance(truncated[aid], bool)

    def test_episode_truncation(self, pz_env):
        """After max_steps the episode should end via truncation."""
        pz_env.reset()
        for _ in range(25):
            if not pz_env.agents:
                break
            actions = {aid: pz_env.action_space(aid).sample() for aid in pz_env.agents}
            pz_env.step(actions)
        assert not pz_env.agents, "episode should end within max_steps"

    def test_is_active_in_infos(self, pz_env):
        """Each agent's info dict must contain an 'is_active' flag."""
        pz_env.reset()
        actions = {aid: pz_env.action_space(aid).sample() for aid in pz_env.agents}
        _, _, _, _, infos = pz_env.step(actions)
        for aid in infos:
            assert "is_active" in infos[aid], f"infos[{aid}] missing 'is_active'"
            assert isinstance(infos[aid]["is_active"], bool)


# =========================================================================
# Space tests
# =========================================================================

class TestSpaces:
    def test_observation_space_matches_obs_shape(self, pz_env):
        obs, _ = pz_env.reset()
        for aid in pz_env.agents:
            space = pz_env.observation_space(aid)
            assert space.shape == obs[aid].shape

    def test_action_space_matches_agent_space(self, pz_env):
        for aid in pz_env.possible_agents:
            space = pz_env.action_space(aid)
            ag = pz_env._heron_env.registered_agents[aid]
            assert space is ag.action_space

    def test_observation_space_cached(self, pz_env):
        """PettingZoo requires the same object each call."""
        for aid in pz_env.possible_agents:
            assert pz_env.observation_space(aid) is pz_env.observation_space(aid)

    def test_action_space_cached(self, pz_env):
        """PettingZoo requires the same object each call."""
        for aid in pz_env.possible_agents:
            assert pz_env.action_space(aid) is pz_env.action_space(aid)


# =========================================================================
# Factory function tests
# =========================================================================

class TestFactory:
    def test_pettingzoo_env_returns_parallel_env(self):
        env = pettingzoo_env(_DEMO_ENV_ID)
        assert isinstance(env, PettingZooParallelEnv)
        env.close()

    def test_direct_heron_env_construction(self):
        heron_env = heron.make(_DEMO_ENV_ID)
        env = PettingZooParallelEnv(heron_env=heron_env)
        assert isinstance(env, PettingZooParallelEnv)
        env.close()

    def test_env_id_construction(self):
        env = PettingZooParallelEnv(env_id=_DEMO_ENV_ID)
        assert isinstance(env, PettingZooParallelEnv)
        env.close()

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Either"):
            PettingZooParallelEnv()


# =========================================================================
# render / close / state
# =========================================================================

class TestMisc:
    def test_render_is_noop(self, pz_env):
        assert pz_env.render() is None

    def test_state_not_implemented(self, pz_env):
        with pytest.raises(NotImplementedError):
            pz_env.state()

    def test_close_does_not_raise(self, pz_env):
        pz_env.close()


# =========================================================================
# Integration: action round-trip with custom env
# =========================================================================

class TestActionRoundTrip:
    """Verify that actions pass through the adaptor into the HERON env
    and produce the expected state changes — same level of validation
    as the RLlib integration test ``test_rllib_action_passing.py``."""

    def test_action_reaches_agent_state(self, custom_env):
        """Specific action value should be reflected in agent state after step."""
        custom_env.reset()
        actions = {"agent_0": np.array([3.5]), "agent_1": np.array([-2.0])}
        custom_env.step(actions)

        ag0 = custom_env._heron_env.get_agent("agent_0")
        ag1 = custom_env._heron_env.get_agent("agent_1")
        assert ag0.state.features["ValueFeature"].value == pytest.approx(3.5)
        assert ag1.state.features["ValueFeature"].value == pytest.approx(-2.0)

    def test_reward_reflects_state(self, custom_env):
        """Reward should equal the agent's state value (per SetterAgent)."""
        custom_env.reset()
        actions = {"agent_0": np.array([5.0]), "agent_1": np.array([0.0])}
        _, rewards, _, _, _ = custom_env.step(actions)
        assert rewards["agent_0"] == pytest.approx(5.0)
        assert rewards["agent_1"] == pytest.approx(0.0)

    def test_obs_shape_consistent_across_steps(self, custom_env):
        """Observation shape must not change between reset and step."""
        obs_reset, _ = custom_env.reset()
        actions = {"agent_0": np.array([1.0]), "agent_1": np.array([1.0])}
        obs_step, *_ = custom_env.step(actions)
        for aid in custom_env.possible_agents:
            assert obs_reset[aid].shape == obs_step[aid].shape

    def test_multiple_steps_accumulate(self, custom_env):
        """Running multiple steps with deterministic actions gives predictable state."""
        custom_env.reset()
        for val in [1.0, 2.0, 3.0]:
            actions = {"agent_0": np.array([val]), "agent_1": np.array([-val])}
            custom_env.step(actions)
        ag0 = custom_env._heron_env.get_agent("agent_0")
        ag1 = custom_env._heron_env.get_agent("agent_1")
        # Last action value (3.0 / -3.0) becomes the state (SetterAgent.apply_action)
        assert ag0.state.features["ValueFeature"].value == pytest.approx(3.0)
        assert ag1.state.features["ValueFeature"].value == pytest.approx(-3.0)

    def test_coordinator_not_in_step_results(self, coord_env):
        """Rewards/obs from step should only contain field agent keys, not coordinator."""
        coord_env.reset()
        actions = {"agent_0": np.array([1.0]), "agent_1": np.array([1.0])}
        obs, rewards, terminated, truncated, infos = coord_env.step(actions)
        for d in [obs, rewards, terminated, truncated, infos]:
            assert "coord" not in d, "Coordinator should not appear in step results"
            assert SYSTEM_AGENT_ID not in d


# =========================================================================
# PettingZoo official parallel_api_test
# =========================================================================

class TestPettingZooOfficialAPI:
    def test_parallel_api(self):
        """Run the official PettingZoo parallel_api_test."""
        try:
            from pettingzoo.test import parallel_api_test
        except ImportError:
            pytest.skip("pettingzoo.test not available")

        env = pettingzoo_env(_DEMO_ENV_ID, max_steps=_DEMO_MAX_STEPS)
        parallel_api_test(env, num_cycles=50)
        env.close()

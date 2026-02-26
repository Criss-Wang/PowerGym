"""EPyMARL adapter for HERON multi-agent environments.

Provides ``HeronEPyMARLAdapter`` — a wrapper that converts a HERON
``MultiAgentEnv`` into an EPyMARL-compatible ``MultiAgentEnv`` so that
HERON environments can be trained with EPyMARL algorithms (MAPPO, QMIX,
MADDPG, etc.).

EPyMARL expects a specific environment interface with methods like
``get_obs()``, ``get_state()``, ``get_avail_actions()``, and
``get_env_info()``.  All actions are discrete.

Usage::

    from heron.adaptors.epymarl import HeronEPyMARLAdapter

    adapter = HeronEPyMARLAdapter(
        env_creator=my_heron_env_factory,
        n_discrete=11,
        max_steps=50,
    )
    obs, info = adapter.reset()
    obs, reward, terminated, truncated, info = adapter.step([5, 3])
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from heron.envs.base import MultiAgentEnv as HeronEnv
from heron.core.observation import Observation


def _obs_to_vec(obs) -> np.ndarray:
    """Convert a HERON ``Observation`` (or raw array) to a flat float32 vector."""
    if isinstance(obs, Observation):
        return obs.vector()
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    return np.asarray(obs, dtype=np.float32)


class HeronEPyMARLAdapter:
    """EPyMARL-compatible wrapper for HERON ``MultiAgentEnv``.

    Implements the ``MultiAgentEnv`` interface expected by EPyMARL's
    episode and parallel runners.  Continuous HERON action spaces are
    discretised into ``n_discrete`` bins per dimension.

    Parameters
    ----------
    env_creator : Callable
        Factory that builds and returns a HERON ``MultiAgentEnv``.
    n_discrete : int
        Number of discrete bins per continuous action dimension.
    max_steps : int
        Episode truncation length.
    common_reward : bool
        If True, sum per-agent rewards into a single scalar.
        If False, return per-agent reward tuple.
    reward_scalarisation : str
        How to aggregate rewards when ``common_reward=True``
        ("sum" or "mean").
    seed : int
        Random seed passed to env reset.
    """

    def __init__(
        self,
        env_creator: Callable,
        n_discrete: int = 11,
        max_steps: int = 50,
        common_reward: bool = True,
        reward_scalarisation: str = "sum",
        seed: int = 0,
        **kwargs,
    ):
        from gymnasium.spaces import Box

        self._env_creator = env_creator
        self._n_discrete = n_discrete
        self._seed = seed
        self.episode_limit = max_steps
        self._common_reward = common_reward
        self._reward_scalarisation = reward_scalarisation

        # Build the underlying HERON env
        self.heron_env: HeronEnv = env_creator({})

        # Initial reset to discover agent IDs and obs shapes
        init_obs, _ = self.heron_env.reset(seed=seed)

        # Discover controllable agents (those with action_space != None)
        self._agent_ids: List[str] = sorted(
            aid
            for aid, ag in self.heron_env.registered_agents.items()
            if ag.action_space is not None
        )
        self.n_agents = len(self._agent_ids)

        # Per-agent observation and action info
        sample_obs = _obs_to_vec(init_obs[self._agent_ids[0]])
        self._obs_size = sample_obs.shape[0]

        # Store original continuous action spaces for discretisation
        self._orig_act_spaces: Dict[str, Any] = {}
        for aid in self._agent_ids:
            ag = self.heron_env.registered_agents[aid]
            self._orig_act_spaces[aid] = ag.action_space

        # Total discrete actions = n_discrete (1-D per agent for now)
        # EPyMARL expects uniform action spaces across all agents
        self._n_actions = n_discrete

        # Cached observations (updated on reset/step)
        self._obs: List[np.ndarray] = [
            _obs_to_vec(init_obs[aid]) for aid in self._agent_ids
        ]
        self._step_count = 0

    def reset(self, seed=None, options=None):
        """Reset the environment. Returns (obs_list, info)."""
        self._step_count = 0
        raw_obs, _ = self.heron_env.reset(seed=seed or self._seed)
        self._obs = [_obs_to_vec(raw_obs[aid]) for aid in self._agent_ids]
        return self._obs, {}

    def step(self, actions):
        """Execute one step with discrete actions.

        Args:
            actions: List/array of discrete action indices, one per agent.

        Returns:
            (obs_list, reward, terminated, truncated, info)
        """
        from gymnasium.spaces import Box

        self._step_count += 1

        # Convert discrete actions to continuous HERON format
        heron_actions: Dict[str, Any] = {}
        for i, aid in enumerate(self._agent_ids):
            act_idx = int(actions[i])
            orig = self._orig_act_spaces[aid]
            if isinstance(orig, Box):
                heron_actions[aid] = self._disc_to_cont(act_idx, orig)
            else:
                heron_actions[aid] = act_idx

        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = (
            self.heron_env.step(heron_actions)
        )

        # Update cached observations
        self._obs = [_obs_to_vec(raw_obs[aid]) for aid in self._agent_ids]

        # Aggregate rewards
        rewards = [float(raw_rew.get(aid, 0.0)) for aid in self._agent_ids]
        if self._common_reward:
            if self._reward_scalarisation == "sum":
                reward = sum(rewards)
            else:
                reward = sum(rewards) / len(rewards)
        else:
            reward = tuple(rewards)

        # Termination
        terminated = raw_term.get("__all__", False)
        truncated = self._step_count >= self.episode_limit

        return self._obs, reward, terminated, truncated, {}

    def get_obs(self) -> List[np.ndarray]:
        """Returns all agent observations in a list."""
        return self._obs

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        """Returns observation for agent at index agent_id."""
        return self._obs[agent_id]

    def get_obs_size(self) -> int:
        """Returns the observation dimension."""
        return self._obs_size

    def get_state(self) -> np.ndarray:
        """Returns global state (concatenation of all agent observations)."""
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self) -> int:
        """Returns the global state dimension."""
        return self._obs_size * self.n_agents

    def get_avail_actions(self) -> List[List[int]]:
        """Returns available actions for all agents (all valid)."""
        return [[1] * self._n_actions for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        """Returns available actions for agent at index agent_id."""
        return [1] * self._n_actions

    def get_total_actions(self) -> int:
        """Returns the total number of discrete actions per agent."""
        return self._n_actions

    def get_env_info(self) -> dict:
        """Returns environment metadata for EPyMARL."""
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def get_stats(self) -> dict:
        return {}

    def render(self):
        pass

    def close(self):
        self.heron_env.close()

    def seed(self, seed=None):
        self._seed = seed

    def save_replay(self):
        pass

    def _disc_to_cont(self, act_idx: int, orig_space) -> np.ndarray:
        """Map a discrete action index to a continuous midpoint."""
        lo = orig_space.low
        hi = orig_space.high
        frac = (act_idx + 0.5) / self._n_discrete
        return (lo + frac * (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
#  Algorithm presets
# ---------------------------------------------------------------------------

_BASE_EPYMARL_CONFIG: Dict = {
    "runner": "episode",
    "mac": "basic_mac",
    "env": "heron",
    "common_reward": True,
    "reward_scalarisation": "sum",
    "env_args": {"seed": 0},
    "batch_size_run": 1,
    "test_nepisode": 5,
    "test_interval": 500,
    "test_greedy": True,
    "log_interval": 500,
    "runner_log_interval": 500,
    "learner_log_interval": 500,
    "t_max": 2000,
    "use_cuda": False,
    "buffer_cpu_only": True,
    "use_tensorboard": False,
    "use_wandb": False,
    "wandb_team": None,
    "wandb_project": None,
    "wandb_mode": "offline",
    "wandb_save_model": False,
    "save_model": False,
    "save_model_interval": 50000,
    "checkpoint_path": "",
    "evaluate": False,
    "render": False,
    "load_step": 0,
    "save_replay": False,
    "local_results_path": "results",
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 32,
    "lr": 0.0005,
    "optim_alpha": 0.99,
    "optim_eps": 0.00001,
    "grad_norm_clip": 10,
    "add_value_last_step": True,
    "agent": "rnn",
    "hidden_dim": 64,
    "obs_agent_id": True,
    "obs_last_action": True,
    "obs_individual_obs": False,
    "repeat_id": 1,
    "label": "default_label",
    "hypergroup": None,
}

_ALGO_CONFIGS: Dict[str, Dict] = {
    "mappo": {
        "action_selector": "soft_policies",
        "mask_before_softmax": True,
        "runner": "episode",
        "buffer_size": 10,
        "batch_size_run": 1,
        "batch_size": 10,
        "target_update_interval_or_tau": 0.01,
        "lr": 0.0003,
        "hidden_dim": 128,
        "obs_agent_id": True,
        "obs_last_action": False,
        "obs_individual_obs": False,
        "agent_output_type": "pi_logits",
        "learner": "ppo_learner",
        "entropy_coef": 0.001,
        "use_rnn": True,
        "standardise_returns": False,
        "standardise_rewards": True,
        "q_nstep": 5,
        "critic_type": "cv_critic",
        "epochs": 4,
        "eps_clip": 0.2,
        "name": "mappo",
        "mac": "basic_mac",
        "common_reward": True,
        "reward_scalarisation": "sum",
    },
    "qmix": {
        "action_selector": "epsilon_greedy",
        "epsilon_start": 1.0,
        "epsilon_finish": 0.05,
        "epsilon_anneal_time": 5000,
        "evaluation_epsilon": 0.0,
        "runner": "episode",
        "buffer_size": 5000,
        "target_update_interval_or_tau": 200,
        "obs_agent_id": True,
        "obs_last_action": False,
        "obs_individual_obs": False,
        "standardise_returns": False,
        "standardise_rewards": True,
        "agent_output_type": "q",
        "learner": "q_learner",
        "double_q": True,
        "mixer": "qmix",
        "use_rnn": False,
        "mixing_embed_dim": 32,
        "hypernet_layers": 2,
        "hypernet_embed": 64,
        "name": "qmix",
        "mac": "basic_mac",
        "common_reward": True,
        "reward_scalarisation": "sum",
    },
    "maddpg": {
        "runner": "episode",
        "buffer_size": 5000,
        "target_update_interval_or_tau": 200,
        "obs_agent_id": True,
        "obs_last_action": False,
        "obs_individual_obs": False,
        "mac": "maddpg_mac",
        "reg": 0.001,
        "batch_size": 32,
        "lr": 0.0005,
        "use_rnn": True,
        "standardise_returns": False,
        "standardise_rewards": True,
        "learner": "maddpg_learner",
        "agent_output_type": "pi_logits",
        "hidden_dim": 128,
        "critic_type": "maddpg_critic",
        "name": "maddpg",
        "common_reward": True,
        "reward_scalarisation": "sum",
    },
}


class AlgoPreset:
    """Named EPyMARL algorithm configuration.

    Attributes:
        name: Algorithm identifier (e.g. ``"mappo"``).
        config: Algorithm-specific overrides.
        base_config: Shared base configuration.
    """

    def __init__(self, name: str, config: Dict, base_config: Optional[Dict] = None):
        self.name = name
        self.config = dict(config)
        self.base_config = dict(base_config or _BASE_EPYMARL_CONFIG)

    def to_dict(self, t_max: int = 2000, seed: int = 0) -> Dict:
        """Merge base + algo config with runtime overrides.

        Returns a flat dict ready for ``SimpleNamespace(**d)`` in EPyMARL.
        """
        merged = dict(self.base_config)
        merged.update(self.config)
        merged["t_max"] = t_max
        merged["seed"] = seed
        merged.setdefault("env_args", {})["seed"] = seed
        return merged

    def __repr__(self) -> str:
        return f"AlgoPreset({self.name!r})"


class _Presets:
    """Namespace providing ``presets.MAPPO``, ``presets.QMIX``, ``presets.MADDPG``."""

    MAPPO = AlgoPreset("mappo", _ALGO_CONFIGS["mappo"])
    QMIX = AlgoPreset("qmix", _ALGO_CONFIGS["qmix"])
    MADDPG = AlgoPreset("maddpg", _ALGO_CONFIGS["maddpg"])


presets = _Presets()

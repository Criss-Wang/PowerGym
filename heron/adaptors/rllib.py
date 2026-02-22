"""RLlib adapter for HERON multi-agent environments.

Provides ``RLlibAdapter`` — a thin wrapper that converts a HERON
``MultiAgentEnv`` into an RLlib-compatible ``MultiAgentEnv`` so that
HERON environments can be plugged directly into RLlib training
pipelines (PPO / MAPPO / IPPO, QMIX, etc.).

Usage::

    from ray.rllib.algorithms.ppo import PPOConfig
    from heron.adaptors.rllib import RLlibAdapter

    config = (
        PPOConfig()
        .environment(
            env=RLlibAdapter,
            env_config={
                "env_creator": my_heron_env_factory,
                "max_steps": 100,
            },
        )
    )
    algo = config.build()
    algo.train()
"""

from typing import Any, Dict, Optional, Set

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace, Discrete, MultiDiscrete

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as _RLlibMAEnv
except ImportError as exc:
    raise ImportError(
        "ray[rllib] is required for the RLlib adapter. "
        "Install with: pip install 'ray[rllib]>=2.9.0'"
    ) from exc

from heron.envs.base import MultiAgentEnv as HeronEnv
from heron.core.observation import Observation


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _obs_to_vec(obs) -> np.ndarray:
    """Convert a HERON ``Observation`` (or raw array) to a flat float32 vector."""
    if isinstance(obs, Observation):
        return obs.vector()
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    return np.asarray(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
#  Adapter
# ---------------------------------------------------------------------------

class RLlibAdapter(_RLlibMAEnv):
    """Wraps a HERON ``MultiAgentEnv`` for RLlib training.

    The adapter exposes HERON field-level agents (those whose
    ``action_space`` is not ``None``) as RLlib agents.  HERON
    ``Observation`` objects are flattened to float32 numpy vectors and
    RLlib actions are forwarded directly to the underlying env.

    Parameters (passed via *config* dict)
    --------------------------------------
    env_creator : Callable[[dict], HeronEnv]
        Factory that builds and returns a HERON ``MultiAgentEnv``.
        Must be a **module-level** function so that Ray can pickle it
        for remote workers.
    env_config : dict, optional
        Forwarded to *env_creator* (default ``{}``).
    max_steps : int, optional
        Episode truncation length (default 50).
    agent_ids : list[str], optional
        Subset of agent IDs to expose.  Defaults to every registered
        agent whose ``action_space is not None``.
    discrete_actions : int or None, optional
        If set, continuous (``Box``) action spaces are replaced with
        ``Discrete(N)`` (1-D) or ``MultiDiscrete`` (N-D) and actions
        are mapped back to continuous midpoints.  Useful for
        value-decomposition methods (QMIX / VDN).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}

        self.max_steps: int = config.get("max_steps", 50)
        self._step_count: int = 0
        self._disc_n: Optional[int] = config.get("discrete_actions")

        # ---- build the underlying HERON env ----
        creator = config.get("env_creator")
        if creator is None:
            raise ValueError("config must contain 'env_creator'")
        self.heron_env: HeronEnv = creator(config.get("env_config", {}))

        # One reset to materialise observation shapes and agent spaces
        init_obs, _ = self.heron_env.reset(seed=0)

        # ---- determine exposed agents ----
        if "agent_ids" in config:
            self._agent_ids: Set[str] = set(config["agent_ids"])
        else:
            self._agent_ids = {
                aid
                for aid, ag in self.heron_env.registered_agents.items()
                if ag.action_space is not None
            }

        # ---- per-agent spaces ----
        self._obs_spaces: Dict[str, gym.Space] = {}
        self._act_spaces: Dict[str, gym.Space] = {}
        self._orig_act_spaces: Dict[str, gym.Space] = {}

        for aid in sorted(self._agent_ids):
            ag = self.heron_env.registered_agents[aid]
            obs_vec = _obs_to_vec(init_obs[aid])

            self._obs_spaces[aid] = Box(
                -np.inf, np.inf, shape=obs_vec.shape, dtype=np.float32,
            )
            self._orig_act_spaces[aid] = ag.action_space

            # Optional discretisation for value-based methods
            if self._disc_n is not None and isinstance(ag.action_space, Box):
                n_dims = int(np.prod(ag.action_space.shape))
                if n_dims == 1:
                    self._act_spaces[aid] = Discrete(self._disc_n)
                else:
                    self._act_spaces[aid] = MultiDiscrete(
                        [self._disc_n] * n_dims
                    )
            else:
                self._act_spaces[aid] = ag.action_space

        # PettingZoo-style attributes required by RLlib's new API stack.
        self.possible_agents = sorted(self._agent_ids)
        self.agents = list(self.possible_agents)

        # Dict-based spaces: the new API stack introspects .spaces to
        # map agent IDs → per-policy spaces automatically.
        self.observation_space = DictSpace(
            {aid: self._obs_spaces[aid] for aid in self.possible_agents}
        )
        self.action_space = DictSpace(
            {aid: self._act_spaces[aid] for aid in self.possible_agents}
        )

    # ------------------------------------------------------------------ #
    #  RLlib MultiAgentEnv interface                                       #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        raw, _ = self.heron_env.reset(seed=seed)
        obs = {
            aid: _obs_to_vec(raw[aid])
            for aid in self._agent_ids
            if aid in raw
        }
        return obs, {aid: {} for aid in obs}

    def step(self, action_dict: Dict[str, Any]):
        self._step_count += 1

        # Map (possibly discretised) actions back to HERON format
        heron_actions: Dict[str, Any] = {}
        for aid, act in action_dict.items():
            orig = self._orig_act_spaces.get(aid)
            if self._disc_n is not None and isinstance(orig, Box):
                heron_actions[aid] = self._disc_to_cont(act, orig)
            else:
                heron_actions[aid] = act

        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = (
            self.heron_env.step(heron_actions)
        )

        obs = {
            aid: _obs_to_vec(raw_obs[aid])
            for aid in self._agent_ids
            if aid in raw_obs
        }
        rew = {aid: float(raw_rew.get(aid, 0.0)) for aid in self._agent_ids}

        term = {aid: bool(raw_term.get(aid, False)) for aid in self._agent_ids}
        term["__all__"] = raw_term.get("__all__", False)

        hit_limit = self._step_count >= self.max_steps
        trunc = {aid: hit_limit for aid in self._agent_ids}
        trunc["__all__"] = hit_limit

        info = {aid: raw_info.get(aid, {}) for aid in self._agent_ids}

        return obs, rew, term, trunc, info

    # ------------------------------------------------------------------ #
    #  Space accessors                                                     #
    # ------------------------------------------------------------------ #

    def get_agent_ids(self) -> Set[str]:
        return self._agent_ids

    def observation_space_sample(self, agent_ids=None):
        ids = agent_ids or list(self._agent_ids)
        return {aid: self._obs_spaces[aid].sample() for aid in ids}

    def action_space_sample(self, agent_ids=None):
        ids = agent_ids or list(self._agent_ids)
        return {aid: self._act_spaces[aid].sample() for aid in ids}

    def observation_space_contains(self, x):
        return isinstance(x, dict) and all(
            self._obs_spaces[k].contains(v)
            for k, v in x.items()
            if k in self._obs_spaces
        )

    def action_space_contains(self, x):
        return isinstance(x, dict) and all(
            self._act_spaces[k].contains(v)
            for k, v in x.items()
            if k in self._act_spaces
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _disc_to_cont(self, act, orig_space: Box) -> np.ndarray:
        """Map a discrete (or multi-discrete) action to a continuous midpoint."""
        lo = orig_space.low
        hi = orig_space.high
        act_arr = np.atleast_1d(act).astype(np.float32)
        frac = (act_arr + 0.5) / self._disc_n
        return (lo + frac * (hi - lo)).astype(np.float32)

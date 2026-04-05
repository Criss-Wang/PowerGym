"""PettingZoo Parallel API adaptor for HERON multi-agent environments.

Provides ``PettingZooParallelEnv`` -- a wrapper that exposes a HERON
``BaseEnv`` as a PettingZoo ``ParallelEnv`` so that HERON environments
can interoperate with any library that speaks the PettingZoo interface
(e.g. SuperSuit wrappers, Tianshou, CleanRL multi-agent, etc.).

Usage::

    import heron
    import heron.demo_envs  # auto-registers demo envs
    from heron.adaptors.pettingzoo import pettingzoo_env

    env = pettingzoo_env("TwoRoomHeating-v0")
    obs, infos = env.reset()
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminated, truncated, infos = env.step(actions)
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box
from pettingzoo.utils.env import ParallelEnv

from heron.agents.constants import PROXY_AGENT_ID, SYSTEM_AGENT_ID
from heron.agents.field_agent import FieldAgent
from heron.envs.base import BaseEnv

# Agent IDs that should never be exposed to PettingZoo consumers.
_EXCLUDED_AGENT_IDS = frozenset({PROXY_AGENT_ID, SYSTEM_AGENT_ID})


class PettingZooParallelEnv(ParallelEnv):
    """Wraps a HERON ``BaseEnv`` as a PettingZoo Parallel environment.

    Only *field-level* agents (``FieldAgent`` subclasses, excluding
    system/proxy agents) are exposed to the PettingZoo consumer.
    Coordinator agents are excluded even if they carry an action space.

    Activity-aware: agents whose ``is_active_at(step)`` returns ``False``
    receive zero observations and zero reward on that step, matching the
    RLlib adaptor's behaviour for heterogeneous tick rates.

    Parameters
    ----------
    heron_env : BaseEnv
        An already-constructed HERON environment.
    env_id : str, optional
        If provided (and *heron_env* is ``None``), build the env via
        ``heron.make(env_id, **kwargs)``.
    **kwargs
        Forwarded to ``heron.make()`` when *env_id* is used.
    """

    metadata: dict[str, Any] = {"render_modes": [], "name": "heron_pettingzoo_v0"}

    def __init__(
        self,
        heron_env: Optional[BaseEnv] = None,
        env_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if heron_env is None and env_id is None:
            raise ValueError(
                "Either 'heron_env' (a BaseEnv instance) or 'env_id' "
                "(a registered HERON env ID) must be provided."
            )

        if heron_env is not None:
            self._heron_env = heron_env
        else:
            import heron

            self._heron_env = heron.make(env_id, **kwargs)

        # Perform an initial reset to materialise observation shapes and
        # agent spaces (same pattern as the RLlib adaptor).
        init_obs, _ = self._heron_env.reset(seed=0)

        # Determine exposed agents: only FieldAgent subclasses, excluding
        # system/proxy agents. This prevents coordinators (which may carry
        # a trivial Discrete(1) action space) from leaking through.
        self._agent_ids: list[str] = sorted(
            aid
            for aid, ag in self._heron_env.registered_agents.items()
            if isinstance(ag, FieldAgent) and aid not in _EXCLUDED_AGENT_IDS
        )

        # Build per-agent observation and action spaces.
        self._obs_spaces: dict[str, Box] = {}
        self._act_spaces: dict[str, Any] = {}

        for aid in self._agent_ids:
            ag = self._heron_env.registered_agents[aid]
            obs_vec = np.asarray(init_obs[aid], dtype=np.float32).flatten()
            self._obs_spaces[aid] = Box(
                -np.inf, np.inf, shape=obs_vec.shape, dtype=np.float32,
            )
            self._act_spaces[aid] = ag.action_space

        # PettingZoo required attributes.
        self.possible_agents: list[str] = list(self._agent_ids)
        self.agents: list[str] = list(self.possible_agents)

        # Expose dicts for the base class fallback (used if observation_space /
        # action_space methods are not overridden).
        self.observation_spaces = dict(self._obs_spaces)
        self.action_spaces = dict(self._act_spaces)

    # ------------------------------------------------------------------
    # PettingZoo Parallel API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the underlying HERON env and return observations + infos."""
        raw_obs, _ = self._heron_env.reset(seed=seed)
        self.agents = list(self.possible_agents)

        obs: dict[str, np.ndarray] = {}
        infos: dict[str, dict] = {}
        for aid in self.agents:
            if aid in raw_obs:
                obs[aid] = np.asarray(raw_obs[aid], dtype=np.float32).flatten()
            else:
                obs[aid] = np.zeros(
                    self._obs_spaces[aid].shape, dtype=np.float32,
                )
            infos[aid] = {}

        return obs, infos

    def step(
        self, actions: dict[str, Any],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Step the underlying HERON env with the given actions.

        Activity-aware: on each step, agents whose ``is_active_at(step)``
        returns ``False`` receive zero observations and zero reward.
        The ``is_active`` flag is injected into each agent's ``infos``
        dict (mirrors the RLlib adaptor behaviour).
        """
        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = (
            self._heron_env.step(actions)
        )

        # --- activity-aware filtering (heterogeneous tick rates) ---
        agents_map = self._heron_env.registered_agents
        step = self._heron_env.step_count
        active_now: Set[str] = {
            aid for aid in self._agent_ids
            if aid in agents_map and agents_map[aid].is_active_at(step)
        }

        obs: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        for aid in self.agents:
            is_active = aid in active_now

            # Observations — use raw when active and available, else zeros.
            if is_active and aid in raw_obs:
                obs[aid] = np.asarray(raw_obs[aid], dtype=np.float32).flatten()
            else:
                obs[aid] = np.zeros(
                    self._obs_spaces[aid].shape, dtype=np.float32,
                )

            rewards[aid] = float(raw_rew.get(aid, 0.0))
            terminated[aid] = bool(raw_term.get(aid, False))
            truncated[aid] = bool(raw_trunc.get(aid, False))
            infos[aid] = dict(raw_info.get(aid, {}))
            infos[aid]["is_active"] = is_active

        # Remove dead agents from self.agents (PettingZoo contract).
        self.agents = [
            aid for aid in self.agents
            if not terminated[aid] and not truncated[aid]
        ]

        return obs, rewards, terminated, truncated, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        """Return the observation space for *agent* (cached)."""
        return self._obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """Return the action space for *agent* (cached)."""
        return self._act_spaces[agent]

    def render(self) -> None:
        """Render stub -- no-op."""
        pass

    def close(self) -> None:
        """Close the underlying HERON environment."""
        self._heron_env.close()

    def state(self) -> np.ndarray:
        """Global state (not supported)."""
        raise NotImplementedError(
            "state() is not implemented for PettingZooParallelEnv. "
            "Use per-agent observations instead."
        )


def pettingzoo_env(env_id: str, **kwargs: Any) -> PettingZooParallelEnv:
    """Create a PettingZoo Parallel env from a registered HERON env ID.

    This is the recommended entry point. It imports ``heron.demo_envs``
    to ensure built-in demo environments are registered before lookup.

    Args:
        env_id: Registered HERON environment identifier
            (e.g. ``"TwoRoomHeating-v0"``).
        **kwargs: Override kwargs forwarded to ``heron.make()``.

    Returns:
        A ``PettingZooParallelEnv`` wrapping the HERON env.
    """
    # Ensure demo envs are registered.
    import heron.demo_envs  # noqa: F401
    import heron

    heron_env = heron.make(env_id, **kwargs)
    return PettingZooParallelEnv(heron_env=heron_env)

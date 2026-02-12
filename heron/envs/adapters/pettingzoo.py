# heron/envs/adapters/pettingzoo.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import gymnasium as gym
import numpy as np

from heron.envs.base import HeronEnvCore, AgentID
from heron.messaging.base import MessageBroker

# PettingZoo is an optional dependency
try:
    from pettingzoo import ParallelEnv  # type: ignore

    PETTINGZOO_AVAILABLE = True
    _PZ_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover
    ParallelEnv = None  # type: ignore
    PETTINGZOO_AVAILABLE = False
    _PZ_IMPORT_ERROR = e


class PettingZooParallelEnv(ParallelEnv):  # type: ignore[misc]
    """
    HERON -> PettingZoo ParallelEnv adapter (composition-based).

    Key design choice: composition (self.core) instead of multi-inheritance.
    - Avoid MRO problems
    - Keep HERON core lifecycle separate

    This adapter is intentionally minimal and test-driven:
    - Provides `_set_agent_ids()` and `_init_spaces()` aliases (your tests call these)
    - Delegates HERON core methods via __getattr__
    - Implements PettingZoo Parallel API: reset() and step()
    """

    metadata = {
        "name": "heron_pettingzoo_parallel",
        "render_modes": ["human", "ansi", "rgb_array"],
    }

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
    ):
        if ParallelEnv is None:  # pragma: no cover
            raise ImportError(
                "PettingZoo is required for PettingZooParallelEnv. "
                "Install with: pip install pettingzoo"
            ) from _PZ_IMPORT_ERROR

        # Init PettingZoo base
        ParallelEnv.__init__(self)

        # Create & init HERON core (mixin-style core; must call _init_heron_core)
        self.core = HeronEnvCore()
        self.core._init_heron_core(env_id=env_id, message_broker=message_broker)

        # PettingZoo required fields
        self.possible_agents: List[AgentID] = []
        self.agents: List[AgentID] = []

        # PettingZoo expects these dicts (can be filled later)
        self.observation_spaces: Dict[AgentID, gym.Space] = {}
        self.action_spaces: Dict[AgentID, gym.Space] = {}

    # ----------------------------
    # Delegation to HERON core
    # ----------------------------
    def __getattr__(self, name: str):
        """Delegate missing attributes/methods to HERON core."""
        if name == "core":
            raise AttributeError
        return getattr(self.core, name)

    # ----------------------------
    # Compatibility aliases (tests expect these)
    # ----------------------------
    def _set_agent_ids(self, agent_ids: List[AgentID]) -> None:
        self.set_agent_ids(agent_ids)

    def _init_spaces(
        self,
        observation_spaces: Optional[Dict[AgentID, gym.Space]] = None,
        action_spaces: Optional[Dict[AgentID, gym.Space]] = None,
    ) -> None:
        self.init_spaces(observation_spaces=observation_spaces, action_spaces=action_spaces)

    # ----------------------------
    # Helper: HERON Observation.local -> np.ndarray
    # ----------------------------
    def _to_np_obs(self, local: Any) -> np.ndarray:
        """
        Best-effort conversion of HERON Observation.local -> numpy array.

        Your unit tests use Observation(local={"value": 1.0}),
        and they expect output np.array([1.0]) with shape (1,).
        """
        if isinstance(local, np.ndarray):
            return local
        if isinstance(local, (list, tuple)):
            return np.asarray(local)
        if isinstance(local, dict):
            if len(local) == 1 and "value" in local:
                return np.asarray([local["value"]], dtype=np.float32)

            # If dict has multiple scalar values, pack them in insertion order.
            vals = list(local.values())
            try:
                return np.asarray(vals, dtype=np.float32)
            except Exception:
                return np.asarray(vals, dtype=object)

        # Fallback for scalars / unknown types
        try:
            return np.asarray([local], dtype=np.float32)
        except Exception:
            return np.asarray([local], dtype=object)

    # ----------------------------
    # Public adapter helpers
    # ----------------------------
    def set_agent_ids(self, agent_ids: List[AgentID]) -> None:
        """Define the agent population for this PettingZoo env."""
        self.possible_agents = list(agent_ids)
        self.agents = list(agent_ids)

    def init_spaces(
        self,
        observation_spaces: Optional[Dict[AgentID, gym.Space]] = None,
        action_spaces: Optional[Dict[AgentID, gym.Space]] = None,
    ) -> None:
        """
        Initialize/update per-agent observation/action spaces.

        If None, tries to infer from registered HERON agents.
        """
        if observation_spaces is None:
            observation_spaces = self.core.get_agent_observation_spaces()
        if action_spaces is None:
            action_spaces = self.core.get_agent_action_spaces()

        self.observation_spaces = dict(observation_spaces)
        self.action_spaces = dict(action_spaces)

    # PettingZoo preferred space accessors
    def observation_space(self, agent: AgentID) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gym.Space:
        return self.action_spaces[agent]

    # ----------------------------
    # PettingZoo Parallel API
    # ----------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, dict]]:
        """
        Returns:
          observations: dict[agent_id] -> np.ndarray
          infos: dict[agent_id] -> dict
        """
        # PettingZoo convention: reset re-activates current agents
        self.agents = list(self.possible_agents)

        # HERON: reset agent timesteps and wire message broker info
        self.core.reset_agents()
        self.core.configure_agents_for_distributed()

        observations = self.core.get_observations()
        obs_dict = {aid: self._to_np_obs(observations[aid].local) for aid in observations}
        infos = {aid: {} for aid in self.agents}
        return obs_dict, infos

    def step(
        self,
        actions: Dict[AgentID, np.ndarray],
    ) -> Tuple[
        Dict[AgentID, np.ndarray],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, dict],
    ]:
        """
        Returns:
          observations, rewards, terminations, truncations, infos
        """
        # HERON: apply actions to agents
        self.core.apply_actions(actions)

        # HERON: collect new observations
        observations = self.core.get_observations()
        obs_dict = {aid: self._to_np_obs(observations[aid].local) for aid in observations}

        # Minimal default RL signals (tests only check contract; env author can override)
        rewards = {aid: 0.0 for aid in self.agents}
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: False for aid in self.agents}
        infos = {aid: {} for aid in self.agents}

        return obs_dict, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        self.core.close_heron()

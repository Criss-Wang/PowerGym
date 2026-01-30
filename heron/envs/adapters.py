"""Environment interface adapters for the HERON framework.

This module provides adapters that combine HeronEnvCore with different
environment interfaces (PettingZoo, RLlib, etc.).

Usage:
    # For PettingZoo ParallelEnv compatibility:
    class MyEnv(PettingZooParallelEnv):
        def __init__(self, config):
            super().__init__(env_id="my_env", distributed=config.get("distributed", False))
            # ... setup agents, etc.

    # For RLlib MultiAgentEnv compatibility:
    class MyEnv(RLlibMultiAgentEnv):
        def __init__(self, config):
            super().__init__(env_id="my_env", distributed=config.get("distributed", False))
            # ... setup agents, etc.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from heron.envs.base import HeronEnvCore
from heron.messaging.base import MessageBroker
from heron.utils.typing import AgentID

# Try to import optional dependencies
try:
    from pettingzoo import ParallelEnv

    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    ParallelEnv = object  # Fallback for type hints

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as RLlibBaseEnv

    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False
    RLlibBaseEnv = object  # Fallback for type hints


class PettingZooParallelEnv(ParallelEnv, HeronEnvCore):
    """HERON environment with PettingZoo ParallelEnv interface.

    This adapter combines HeronEnvCore functionality with PettingZoo's
    ParallelEnv interface, suitable for multi-agent environments where
    all agents act simultaneously.

    Subclasses must implement:
        - _build_agents(): Create and return agent dictionary
        - _get_obs(): Get observations for all agents
        - _reward_and_safety(): Compute rewards and safety metrics
        - step(): Execute environment step (can call super().step() for common logic)
        - reset(): Reset environment (can call super().reset() for common logic)

    Attributes:
        agents: List of active agent IDs (PettingZoo API)
        possible_agents: List of all possible agent IDs (PettingZoo API)
        action_spaces: Dictionary mapping agent IDs to action spaces
        observation_spaces: Dictionary mapping agent IDs to observation spaces
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
        distributed: bool = False,
    ):
        """Initialize PettingZoo-compatible HERON environment.

        Args:
            env_id: Environment identifier (auto-generated if not provided)
            message_broker: Optional message broker for distributed mode
            distributed: If True, use distributed execution mode
        """
        if not PETTINGZOO_AVAILABLE:
            raise ImportError(
                "PettingZoo is required for PettingZooParallelEnv. "
                "Install with: pip install pettingzoo"
            )

        ParallelEnv.__init__(self)
        self._init_heron_core(
            env_id=env_id, message_broker=message_broker, distributed=distributed
        )

        # PettingZoo required attributes (initialized in _init_spaces)
        self.action_spaces: Dict[AgentID, gym.Space] = {}
        self.observation_spaces: Dict[AgentID, gym.Space] = {}

        # These will be set after agents are built
        self._possible_agents: List[AgentID] = []
        self._agents: List[AgentID] = []

    @property
    def possible_agents(self) -> List[AgentID]:
        """Return list of all possible agent IDs (PettingZoo API)."""
        return self._possible_agents

    @property
    def agents(self) -> List[AgentID]:
        """Return list of currently active agent IDs (PettingZoo API)."""
        return self._agents

    def observation_space(self, agent: AgentID) -> gym.Space:
        """Return observation space for given agent (PettingZoo API).

        Args:
            agent: Agent ID

        Returns:
            Gymnasium space for the agent's observations
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gym.Space:
        """Return action space for given agent (PettingZoo API).

        Args:
            agent: Agent ID

        Returns:
            Gymnasium space for the agent's actions
        """
        return self.action_spaces[agent]

    def _set_agent_ids(self, agent_ids: List[AgentID]) -> None:
        """Set the agent ID lists after building agents.

        Call this after building agents to set up PettingZoo-required attributes.

        Args:
            agent_ids: List of agent IDs
        """
        self._possible_agents = list(agent_ids)
        self._agents = list(agent_ids)

    def _init_spaces(
        self,
        action_spaces: Dict[AgentID, gym.Space],
        observation_spaces: Dict[AgentID, gym.Space],
    ) -> None:
        """Initialize action and observation spaces.

        Call this after building agents to set up spaces.

        Args:
            action_spaces: Dictionary mapping agent IDs to action spaces
            observation_spaces: Dictionary mapping agent IDs to observation spaces
        """
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces

    def close(self) -> None:
        """Clean up environment resources."""
        self.close_heron()


class RLlibMultiAgentEnv(RLlibBaseEnv, HeronEnvCore):
    """HERON environment with RLlib MultiAgentEnv interface.

    This adapter combines HeronEnvCore functionality with RLlib's
    MultiAgentEnv interface, suitable for training with Ray RLlib.

    Subclasses must implement:
        - _build_agents(): Create and return agent dictionary
        - _get_obs(): Get observations for all agents
        - _reward_and_safety(): Compute rewards and safety metrics
        - step(): Execute environment step
        - reset(): Reset environment

    Note:
        RLlib expects specific return formats:
        - step() returns: (obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict)
        - terminated_dict and truncated_dict must include "__all__" key
        - reset() returns: (obs_dict, info_dict)
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
        distributed: bool = False,
    ):
        """Initialize RLlib-compatible HERON environment.

        Args:
            env_id: Environment identifier (auto-generated if not provided)
            message_broker: Optional message broker for distributed mode
            distributed: If True, use distributed execution mode
        """
        if not RLLIB_AVAILABLE:
            raise ImportError(
                "Ray RLlib is required for RLlibMultiAgentEnv. "
                "Install with: pip install 'ray[rllib]'"
            )

        RLlibBaseEnv.__init__(self)
        self._init_heron_core(
            env_id=env_id, message_broker=message_broker, distributed=distributed
        )

        # RLlib spaces (initialized in _init_spaces)
        self._action_spaces: Dict[AgentID, gym.Space] = {}
        self._observation_spaces: Dict[AgentID, gym.Space] = {}
        self._agent_ids: List[AgentID] = []

    def observation_space_sample(self, agent_ids: Optional[List[AgentID]] = None) -> Dict[AgentID, Any]:
        """Sample observations for given agents.

        Args:
            agent_ids: List of agent IDs to sample for (default: all agents)

        Returns:
            Dictionary mapping agent IDs to sampled observations
        """
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {aid: self._observation_spaces[aid].sample() for aid in agent_ids}

    def action_space_sample(self, agent_ids: Optional[List[AgentID]] = None) -> Dict[AgentID, Any]:
        """Sample actions for given agents.

        Args:
            agent_ids: List of agent IDs to sample for (default: all agents)

        Returns:
            Dictionary mapping agent IDs to sampled actions
        """
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {aid: self._action_spaces[aid].sample() for aid in agent_ids}

    def get_agent_ids(self) -> List[AgentID]:
        """Return list of agent IDs (RLlib API).

        Returns:
            List of agent IDs
        """
        return self._agent_ids

    def observation_space(self, agent: AgentID) -> gym.Space:
        """Return observation space for given agent.

        Args:
            agent: Agent ID

        Returns:
            Gymnasium space for the agent's observations
        """
        return self._observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gym.Space:
        """Return action space for given agent.

        Args:
            agent: Agent ID

        Returns:
            Gymnasium space for the agent's actions
        """
        return self._action_spaces[agent]

    def _set_agent_ids(self, agent_ids: List[AgentID]) -> None:
        """Set the agent ID list after building agents.

        Args:
            agent_ids: List of agent IDs
        """
        self._agent_ids = list(agent_ids)

    def _init_spaces(
        self,
        action_spaces: Dict[AgentID, gym.Space],
        observation_spaces: Dict[AgentID, gym.Space],
    ) -> None:
        """Initialize action and observation spaces.

        Args:
            action_spaces: Dictionary mapping agent IDs to action spaces
            observation_spaces: Dictionary mapping agent IDs to observation spaces
        """
        self._action_spaces = action_spaces
        self._observation_spaces = observation_spaces

    def close(self) -> None:
        """Clean up environment resources."""
        self.close_heron()


# Utility function to check available interfaces
def get_available_interfaces() -> Dict[str, bool]:
    """Check which environment interfaces are available.

    Returns:
        Dictionary mapping interface names to availability status
    """
    return {
        "pettingzoo": PETTINGZOO_AVAILABLE,
        "rllib": RLLIB_AVAILABLE,
    }

"""Environment interface adapters for the HERON framework.

This module provides adapters that combine HeronEnvCore with different
environment interfaces (PettingZoo, RLlib, etc.).

Execution Modes:
    - Option A (Training): Synchronous step() with CTDE pattern
      All agents step together, coordinator aggregates observations,
      centralized policy computes joint action, coordinator distributes.

    - Option B (Testing): Event-driven via EventScheduler
      Each agent ticks independently at its own interval with configurable
      observation/action/message delays. Tests policy robustness.

Usage:
    # For PettingZoo ParallelEnv compatibility:
    class MyEnv(PettingZooParallelEnv):
        def __init__(self, config):
            super().__init__(env_id="my_env")
            # Build agents and system agent
            self._build_agents()
            # For Option A: use step() directly
            # For Option B: call setup_event_driven() then run_event_driven()

    # For RLlib MultiAgentEnv compatibility:
    class MyEnv(RLlibMultiAgentEnv):
        def __init__(self, config):
            super().__init__(env_id="my_env")
            # Build agents and system agent
            self._build_agents()
"""

from typing import Any, Dict, List, Optional

from pettingzoo import ParallelEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym

from heron.envs.base import HeronEnvCore
from heron.messaging.base import MessageBroker
from heron.utils.typing import AgentID


class PettingZooParallelEnv(ParallelEnv, HeronEnvCore):
    """HERON environment with PettingZoo ParallelEnv interface.

    This adapter combines HeronEnvCore functionality with PettingZoo's
    ParallelEnv interface, suitable for multi-agent environments where
    all agents act simultaneously.

    Execution Modes:
        Option A (Training - Synchronous):
            1. Call reset() to initialize
            2. Call step(actions) which internally:
               - Collects observations via get_observations()
               - Applies actions via apply_actions()
               - Runs physics/simulation
               - Returns (obs, rewards, terminated, truncated, infos)

        Option B (Testing - Event-Driven):
            1. Call reset() to initialize
            2. Call setup_event_driven() to create scheduler
            3. Call setup_default_handlers() with callbacks
            4. Call run_event_driven(t_end) to run simulation

        With SystemAgent:
            1. Build SystemAgent with coordinators
            2. Call set_system_agent(system_agent)
            3. Use step_with_system_agent() or run_event_driven_with_system_agent()

    Subclasses must implement:
        - _build_agents(): Create and return agent dictionary
        - _get_obs(): Get observations for all agents
        - step(): Execute environment step
        - reset(): Reset environment

    Attributes:
        agents: List of active agent IDs (PettingZoo API)
        possible_agents: List of all possible agent IDs (PettingZoo API)
        action_spaces: Dictionary mapping agent IDs to action spaces
        observation_spaces: Dictionary mapping agent IDs to observation spaces
        system_agent: Optional SystemAgent for hierarchical management
        proxy_agent: Optional ProxyAgent for state distribution
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
    ):
        """Initialize PettingZoo-compatible HERON environment.

        Args:
            env_id: Environment identifier (auto-generated if not provided)
            message_broker: Optional MessageBroker (defaults to InMemoryBroker)
        """
        ParallelEnv.__init__(self)
        self._init_heron_core(
            env_id=env_id,
            message_broker=message_broker,
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


class RLlibMultiAgentEnv(MultiAgentEnv, HeronEnvCore):
    """HERON environment with RLlib MultiAgentEnv interface.

    This adapter combines HeronEnvCore functionality with RLlib's
    MultiAgentEnv interface, suitable for training with Ray RLlib.

    Execution Modes:
        Option A (Training - Synchronous):
            Standard RLlib training loop using step() and reset().
            This is the primary use case for RLlib environments.

        Option B (Testing - Event-Driven):
            After training, test policies with realistic timing:
            1. Load trained policy
            2. Call setup_event_driven() to create scheduler
            3. Call setup_default_handlers() with policy inference callback
            4. Call run_event_driven(t_end) to run simulation

        With SystemAgent:
            1. Build SystemAgent with coordinators
            2. Call set_system_agent(system_agent)
            3. Use step_with_system_agent() for training
            4. Use run_event_driven_with_system_agent() for testing

    Subclasses must implement:
        - _build_agents(): Create and return agent dictionary
        - _get_obs(): Get observations for all agents
        - step(): Execute environment step
        - reset(): Reset environment

    Note:
        RLlib expects specific return formats:
        - step() returns: (obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict)
        - terminated_dict and truncated_dict must include "__all__" key
        - reset() returns: (obs_dict, info_dict)

    Attributes:
        system_agent: Optional SystemAgent for hierarchical management
        proxy_agent: Optional ProxyAgent for state distribution
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
    ):
        """Initialize RLlib-compatible HERON environment.

        Args:
            env_id: Environment identifier (auto-generated if not provided)
            message_broker: Optional MessageBroker (defaults to InMemoryBroker)
        """
        MultiAgentEnv.__init__(self)
        self._init_heron_core(
            env_id=env_id,
            message_broker=message_broker,
        )

        # RLlib spaces (initialized in _init_spaces)
        self._action_spaces: Dict[AgentID, gym.Space] = {}
        self._observation_spaces: Dict[AgentID, gym.Space] = {}
        self._agent_ids: List[AgentID] = []

    def observation_space_sample(
        self, agent_ids: Optional[List[AgentID]] = None
    ) -> Dict[AgentID, Any]:
        """Sample observations for given agents.

        Args:
            agent_ids: List of agent IDs to sample for (default: all agents)

        Returns:
            Dictionary mapping agent IDs to sampled observations
        """
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {aid: self._observation_spaces[aid].sample() for aid in agent_ids}

    def action_space_sample(
        self, agent_ids: Optional[List[AgentID]] = None
    ) -> Dict[AgentID, Any]:
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

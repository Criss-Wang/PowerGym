"""Abstract environment interfaces for the HERON framework.

This module defines the base environment classes that domain-specific
implementations should extend.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.messaging.base import MessageBroker
from heron.utils.typing import AgentID


class BaseEnv(gym.Env, ABC):
    """Abstract base environment for HERON.

    Extends Gymnasium's Env interface with HERON-specific functionality
    for hierarchical multi-agent systems.

    Attributes:
        env_id: Unique environment identifier
        message_broker: Optional message broker for distributed execution
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
    ):
        """Initialize base environment.

        Args:
            env_id: Environment identifier (auto-generated if not provided)
            message_broker: Optional message broker for distributed mode
        """
        super().__init__()
        self.env_id = env_id or f"env_{id(self)}"
        self.message_broker = message_broker
        self._timestep = 0

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (observation, info)
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        if self.message_broker:
            self.message_broker.clear_environment(self.env_id)


class MultiAgentEnv(BaseEnv):
    """Base class for multi-agent environments.

    Provides common functionality for environments with multiple agents,
    including agent registration, observation collection, and action distribution.

    Supports both centralized and distributed execution modes.

    Attributes:
        agents: Dictionary mapping agent IDs to Agent instances
        coordinators: Dictionary mapping coordinator IDs to CoordinatorAgent instances
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        message_broker: Optional[MessageBroker] = None,
        distributed: bool = False,
    ):
        """Initialize multi-agent environment.

        Args:
            env_id: Environment identifier
            message_broker: Message broker for distributed mode
            distributed: If True, use distributed execution mode
        """
        super().__init__(env_id=env_id, message_broker=message_broker)

        self.distributed = distributed
        self.agents: Dict[AgentID, Agent] = {}
        self.coordinators: Dict[AgentID, CoordinatorAgent] = {}

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the environment.

        Args:
            agent: Agent to register
        """
        self.agents[agent.agent_id] = agent
        if isinstance(agent, CoordinatorAgent):
            self.coordinators[agent.agent_id] = agent

    def register_agents(self, agents: List[Agent]) -> None:
        """Register multiple agents.

        Args:
            agents: List of agents to register
        """
        for agent in agents:
            self.register_agent(agent)

    def get_observations(self, global_state: Optional[Dict[str, Any]] = None) -> Dict[AgentID, Observation]:
        """Collect observations from all agents.

        Args:
            global_state: Optional global state to pass to agents

        Returns:
            Dictionary mapping agent IDs to observations
        """
        observations = {}
        for agent_id, agent in self.agents.items():
            observations[agent_id] = agent.observe(global_state)
        return observations

    def apply_actions(
        self,
        actions: Dict[AgentID, Any],
        observations: Optional[Dict[AgentID, Observation]] = None
    ) -> None:
        """Apply actions to agents.

        Args:
            actions: Dictionary mapping agent IDs to actions
            observations: Optional observations to pass to agents
        """
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                obs = observations.get(agent_id) if observations else None
                self.agents[agent_id].act(obs, upstream_action=action)

    @abstractmethod
    def get_joint_observation_space(self) -> gym.Space:
        """Get the joint observation space for all agents.

        Returns:
            Gymnasium space representing joint observations
        """
        pass

    @abstractmethod
    def get_joint_action_space(self) -> gym.Space:
        """Get the joint action space for all agents.

        Returns:
            Gymnasium space representing joint actions
        """
        pass

    def get_agent_action_spaces(self) -> Dict[AgentID, gym.Space]:
        """Get action spaces for all agents.

        Returns:
            Dictionary mapping agent IDs to their action spaces
        """
        return {
            agent_id: agent.action_space
            for agent_id, agent in self.agents.items()
            if agent.action_space is not None
        }

    def get_agent_observation_spaces(self) -> Dict[AgentID, gym.Space]:
        """Get observation spaces for all agents.

        Returns:
            Dictionary mapping agent IDs to their observation spaces
        """
        return {
            agent_id: agent.observation_space
            for agent_id, agent in self.agents.items()
            if agent.observation_space is not None
        }

    def reset_agents(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset all registered agents.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        for agent in self.agents.values():
            agent.reset(seed=seed, **kwargs)

    async def step_distributed(self) -> None:
        """Execute distributed step using message broker.

        This triggers the hierarchical execution flow where agents
        communicate via the message broker.
        """
        if not self.distributed or not self.message_broker:
            raise RuntimeError(
                "Distributed step requires distributed=True and message_broker"
            )

        # Execute all coordinators (they will recursively execute subordinates)
        import asyncio
        await asyncio.gather(*[
            coord.step_distributed()
            for coord in self.coordinators.values()
        ])

    def get_total_reward(self) -> Dict[str, float]:
        """Get total reward aggregated from all agents.

        Returns:
            Dictionary with total cost and safety
        """
        total_cost = sum(
            agent.cost if hasattr(agent, 'cost') else 0
            for agent in self.agents.values()
        )
        total_safety = sum(
            agent.safety if hasattr(agent, 'safety') else 0
            for agent in self.agents.values()
        )
        return {"cost": total_cost, "safety": total_safety}

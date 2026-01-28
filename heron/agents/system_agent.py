"""System-level agents (L3) for the HERON framework.

SystemAgent is the top level in the agent hierarchy, typically handling
system-wide coordination, visibility filtering, and environment interaction.
"""

from abc import abstractmethod
from typing import Any, Dict as DictType, List, Optional

import numpy as np

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.core.state import State
from heron.utils.typing import AgentID


SYSTEM_LEVEL = 3  # Level identifier for system-level agents


class SystemAgent(Agent):
    """System-level agent for top-level coordination and environment interface.

    SystemAgent serves as the interface between the environment and the agent
    hierarchy. It handles:
    - System-wide state management
    - Visibility filtering for distributed mode
    - Communication with the environment

    In distributed mode, SystemAgent acts as a proxy between the environment
    and the coordinator/field agents, ensuring agents never access the
    environment state directly.

    Attributes:
        coordinators: Dictionary mapping coordinator IDs to CoordinatorAgent instances
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,

        # communication params
        message_broker: Optional["MessageBroker"] = None,
        env_id: Optional[str] = None,

        # SystemAgent specific params
        config: Optional[DictType[str, Any]] = None,
    ):
        """Initialize system agent.

        Args:
            agent_id: Unique identifier
            message_broker: Optional message broker for hierarchical execution
            env_id: Optional environment ID for multi-environment isolation
            config: System agent configuration dictionary
        """
        config = config or {}

        # Build coordinator agents
        coordinator_configs = config.get('coordinators', [])
        self.coordinators = self._build_coordinators(
            coordinator_configs,
            message_broker=message_broker,
            env_id=env_id,
            upstream_id=agent_id
        )

        super().__init__(
            agent_id=agent_id or "system_agent",
            level=SYSTEM_LEVEL,
            message_broker=message_broker,
            upstream_id=None,  # System agent has no upstream
            env_id=env_id,
            subordinates=self.coordinators,
        )

    def _build_coordinators(
        self,
        coordinator_configs: List[DictType[str, Any]],
        message_broker: Optional["MessageBroker"] = None,
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, CoordinatorAgent]:
        """Build coordinator agents from configuration.

        Override this in domain-specific subclasses to create appropriate
        coordinator types based on configuration.

        Args:
            coordinator_configs: List of coordinator configuration dictionaries
            message_broker: Optional message broker for communication
            env_id: Environment ID for multi-environment isolation
            upstream_id: Upstream agent ID (this system agent)

        Returns:
            Dictionary mapping coordinator IDs to CoordinatorAgent instances
        """
        # Default implementation - override in subclasses
        return {}

    # ============================================
    # Core Agent Lifecycle Methods
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset system agent and all coordinators.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)

        # Reset coordinators
        for _, coordinator in self.coordinators.items():
            coordinator.reset(seed=seed, **kwargs)

    def observe(self, global_state: Optional[DictType[str, Any]] = None, *args, **kwargs) -> Observation:
        """Collect observations from coordinators. [Only for synchronous direct execution]

        Args:
            global_state: Environment state

        Returns:
            Aggregated observation from all coordinators
        """
        # Collect coordinator observations
        coordinator_obs = {}
        for coord_id, coordinator in self.coordinators.items():
            coordinator_obs[coord_id] = coordinator.observe(global_state)

        return Observation(
            timestamp=self._timestep,
            local={"coordinator_obs": coordinator_obs},
            global_info=global_state or {},
            messages=[],
        )

    def act(self, observation: Observation, upstream_action: Any = None) -> None:
        """Distribute actions to coordinators. [Only for synchronous direct execution]

        Args:
            observation: Aggregated observation
            upstream_action: Action to distribute
        """
        if upstream_action is None:
            return

        # Distribute actions to coordinators
        coordinator_obs = observation.local.get("coordinator_obs", {})

        if isinstance(upstream_action, dict):
            # Per-coordinator actions
            for coord_id, action in upstream_action.items():
                if coord_id in self.coordinators:
                    obs = coordinator_obs.get(coord_id)
                    if obs:
                        self.coordinators[coord_id].act(obs, upstream_action=action)
        else:
            # Single action for all coordinators
            for coord_id, coordinator in self.coordinators.items():
                obs = coordinator_obs.get(coord_id)
                if obs:
                    coordinator.act(obs, upstream_action=upstream_action)

    # ============================================
    # Abstract Methods for Hierarchical Execution
    # ============================================

    def _derive_local_action(self, upstream_action: Optional[Any]) -> Optional[Any]:
        """Derive local action (none for system agent).

        Args:
            upstream_action: Not used (system agent has no upstream)

        Returns:
            None
        """
        return None

    async def _derive_downstream_actions(
        self,
        upstream_action: Optional[Any]
    ) -> DictType[AgentID, Any]:
        """Derive actions for coordinators.

        Override this to implement system-level action decomposition.

        Args:
            upstream_action: System-level action

        Returns:
            Dict mapping coordinator IDs to their actions
        """
        if upstream_action is None:
            return {}

        if isinstance(upstream_action, dict):
            return upstream_action

        # Default: broadcast same action to all coordinators
        return {coord_id: upstream_action for coord_id in self.coordinators}

    def _execute_local_action(self, action: Optional[Any]) -> None:
        """Execute own action (none for system agent).

        Args:
            action: Not used
        """
        pass

    # ============================================
    # Environment Interface Methods
    # ============================================

    @abstractmethod
    def update_from_environment(self, env_state: DictType[str, Any]) -> None:
        """Update internal state from environment.

        This method should be called by the environment to provide
        the system agent with the current environment state.

        Args:
            env_state: Current environment state
        """
        pass

    @abstractmethod
    def get_state_for_environment(self) -> DictType[str, Any]:
        """Get agent actions/state for environment.

        This method should be called by the environment to get
        the agents' actions to apply.

        Returns:
            Dict containing agent actions/state updates
        """
        pass

    # ============================================
    # Visibility Filtering Methods
    # ============================================

    def filter_state_for_agent(
        self,
        state: State,
        requestor_id: AgentID,
        requestor_level: int
    ) -> DictType[str, np.ndarray]:
        """Filter state based on agent's visibility level.

        Uses the FeatureProvider visibility rules to filter which
        parts of the state are visible to the requesting agent.

        Args:
            state: State to filter
            requestor_id: ID of requesting agent
            requestor_level: Hierarchy level of requesting agent

        Returns:
            Dict of observable features
        """
        return state.observed_by(requestor_id, requestor_level)

    def broadcast_to_coordinators(self, message: DictType[str, Any]) -> None:
        """Broadcast a message to all coordinators.

        In distributed mode, this publishes messages via the message broker.

        Args:
            message: Message content to broadcast
        """
        if not self.message_broker:
            return

        from heron.messaging.base import ChannelManager, Message, MessageType

        for coord_id in self.coordinators:
            channel = ChannelManager.info_channel(
                self.agent_id,
                coord_id,
                self.env_id
            )
            msg = Message(
                env_id=self.env_id,
                sender_id=self.agent_id,
                recipient_id=coord_id,
                timestamp=self._timestep,
                message_type=MessageType.INFO,
                payload=message
            )
            self.message_broker.publish(channel, msg)

    # ============================================
    # Cost/Safety Methods
    # ============================================

    @property
    def cost(self) -> float:
        """Get total cost from all coordinators."""
        return sum(coord.cost for coord in self.coordinators.values())

    @property
    def safety(self) -> float:
        """Get total safety penalty from all coordinators."""
        return sum(coord.safety for coord in self.coordinators.values())

    def get_reward(self) -> DictType[str, float]:
        """Get system-wide reward.

        Returns:
            Dict with total cost and safety values
        """
        return {"cost": self.cost, "safety": self.safety}

    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        num_coords = len(self.coordinators)
        return f"SystemAgent(id={self.agent_id}, coordinators={num_coords})"

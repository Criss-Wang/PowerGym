"""System-level agents (L3) for the HERON framework.

SystemAgent is the top level in the agent hierarchy, typically handling
system-wide coordination, visibility filtering, and environment interaction.

In synchronous mode (Option A - Training), the system agent:
1. Collects observations from all coordinators
2. Optionally computes system-level actions
3. Distributes actions to coordinators
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
    - Visibility filtering for constrained information
    - Communication with coordinators

    Attributes:
        coordinators: Dictionary mapping coordinator IDs to CoordinatorAgent instances
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,

        # hierarchy params
        env_id: Optional[str] = None,

        # SystemAgent specific params
        config: Optional[DictType[str, Any]] = None,

        # timing params (for event-driven scheduling)
        tick_interval: float = 300.0,  # System agents tick least frequently
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,
    ):
        """Initialize system agent.

        Args:
            agent_id: Unique identifier
            env_id: Optional environment ID for multi-environment isolation
            config: System agent configuration dictionary
            tick_interval: Time between agent ticks (default 300s for system level)
            obs_delay: Observation delay
            act_delay: Action delay
            msg_delay: Message delay
        """
        config = config or {}

        # Build coordinator agents
        coordinator_configs = config.get('coordinators', [])
        self.coordinators = self._build_coordinators(
            coordinator_configs,
            env_id=env_id,
            upstream_id=agent_id
        )

        super().__init__(
            agent_id=agent_id or "system_agent",
            level=SYSTEM_LEVEL,
            upstream_id=None,  # System agent has no upstream
            env_id=env_id,
            subordinates=self.coordinators,
            tick_interval=tick_interval,
            obs_delay=obs_delay,
            act_delay=act_delay,
            msg_delay=msg_delay,
        )

    def _build_coordinators(
        self,
        coordinator_configs: List[DictType[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, CoordinatorAgent]:
        """Build coordinator agents from configuration.

        Override this in domain-specific subclasses to create appropriate
        coordinator types based on configuration.

        Args:
            coordinator_configs: List of coordinator configuration dictionaries
            env_id: Environment ID for multi-environment isolation
            upstream_id: Upstream agent ID (this system agent)

        Returns:
            Dictionary mapping coordinator IDs to CoordinatorAgent instances
        """
        # Default implementation - override in subclasses
        return {}

    # ============================================
    # Core Agent Lifecycle Methods (Both Modes)
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset system agent and all coordinators. [Both Modes]

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)

        # Reset coordinators
        for _, coordinator in self.coordinators.items():
            coordinator.reset(seed=seed, **kwargs)

    # ============================================
    # Synchronous Execution (Option A - Training)
    # ============================================

    def observe(self, global_state: Optional[DictType[str, Any]] = None, *args, **kwargs) -> Observation:
        """Collect observations from coordinators. [Both Modes]

        - Training (Option A): Called by environment
        - Testing (Option B): Called internally by tick()

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
        """Distribute actions to coordinators. [Training Only - Direct Call]

        Note: In Testing (Option B), tick() handles action distribution via
        MESSAGE_DELIVERY events instead of calling this method directly.

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
    # Event-Driven Execution (Option B - Testing)
    # ============================================

    def tick(
        self,
        scheduler: "EventScheduler",
        current_time: float,
        global_state: Optional[DictType[str, Any]] = None,
        proxy: Optional["Agent"] = None,
    ) -> None:
        """Execute one tick in event-driven mode. [Testing Only]

        In Option B, SystemAgent:
        1. Updates timestep
        2. Gets observation from coordinators
        3. Optionally computes system-level action
        4. Schedules MESSAGE_DELIVERY events to coordinators with msg_delay

        Args:
            scheduler: EventScheduler for scheduling future events
            current_time: Current simulation time
            global_state: Optional global state for observation
            proxy: Optional ProxyAgent for delayed observations
        """
        self._timestep = current_time

        # Get observation from coordinators
        observation = self.observe(global_state)
        self._last_observation = observation

        # SystemAgent typically doesn't have a policy - it distributes
        # external actions or signals. Override in subclasses for custom behavior.
        # Check message broker for any external actions
        upstream_action = None
        actions = self.receive_action_messages()
        if actions:
            upstream_action = actions[-1]  # Use most recent action

        if upstream_action is None:
            # No action to distribute
            return

        # Distribute actions to coordinators via MESSAGE_DELIVERY events
        if isinstance(upstream_action, dict):
            # Per-coordinator actions
            for coord_id, action in upstream_action.items():
                if coord_id in self.coordinators:
                    scheduler.schedule_message_delivery(
                        sender_id=self.agent_id,
                        recipient_id=coord_id,
                        message={"action": action},
                        delay=self.msg_delay,
                    )
        else:
            # Single action for all coordinators
            for coord_id in self.coordinators:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=coord_id,
                    message={"action": upstream_action},
                    delay=self.msg_delay,
                )

    # ============================================
    # Environment Interface Methods (Both Modes)
    # ============================================

    @abstractmethod
    def update_from_environment(self, env_state: DictType[str, Any]) -> None:
        """Update internal state from environment. [Both Modes]

        This method should be called by the environment to provide
        the system agent with the current environment state.

        Args:
            env_state: Current environment state
        """
        pass

    @abstractmethod
    def get_state_for_environment(self) -> DictType[str, Any]:
        """Get agent actions/state for environment. [Both Modes]

        This method should be called by the environment to get
        the agents' actions to apply.

        Returns:
            Dict containing agent actions/state updates
        """
        pass

    # ============================================
    # Visibility Filtering Methods (Both Modes)
    # ============================================

    def filter_state_for_agent(
        self,
        state: State,
        requestor_id: AgentID,
        requestor_level: int
    ) -> DictType[str, np.ndarray]:
        """Filter state based on agent's visibility level. [Both Modes]

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
        """Broadcast a message to all coordinators. [Both Modes]

        Uses message broker if available for distributed communication.

        Args:
            message: Message content to broadcast
        """
        if self._message_broker is not None:
            for coord_id in self.coordinators:
                self.send_message(message, recipient_id=coord_id)

    # ============================================
    # Utility Methods (Both Modes)
    # ============================================

    def __repr__(self) -> str:
        num_coords = len(self.coordinators)
        return f"SystemAgent(id={self.agent_id}, coordinators={num_coords})"

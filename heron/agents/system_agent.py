"""System-level agents (L3) for the HERON framework.

SystemAgent is the top level in the agent hierarchy, typically handling
system-wide coordination, visibility filtering, and environment interaction.

In synchronous mode (Option A - Training), the system agent:
1. Collects observations from all coordinators
2. Optionally computes system-level actions
3. Distributes actions to coordinators

In event-driven mode (Option B - Testing), the system agent:
1. Operates on its own tick schedule (slowest in hierarchy)
2. Receives coordinator observations
3. Distributes actions via MESSAGE_DELIVERY events
"""

from typing import Any, Dict as DictType, List, Optional

import numpy as np

from heron.agents.hierarchical_agent import HierarchicalAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.observation import (
    OBS_KEY_COORDINATOR_OBS,
    OBS_KEY_SYSTEM_STATE,
)
from heron.core.state import State, SystemAgentState
from heron.utils.typing import AgentID
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig


SYSTEM_LEVEL = 3  # Level identifier for system-level agents


class SystemAgent(HierarchicalAgent):
    """System-level agent for top-level coordination and environment interface.

    SystemAgent serves as the interface between the environment and the agent
    hierarchy. It handles:
    - System-wide state management
    - Visibility filtering for constrained information
    - Communication with coordinators
    - Protocol-based coordination of coordinators
    - Policy-based system-level decision making

    Supports two execution modes:

    **Synchronous Mode (Option A - Training)**:
        Call observe() to aggregate coordinator observations, then act() to
        distribute actions. Suitable for centralized RL training where all
        agents step synchronously.

    **Event-Driven Mode (Option B - Testing)**:
        Call tick() with an EventScheduler. Schedules MESSAGE_DELIVERY events
        to coordinators with configurable msg_delay. Operates on the slowest
        tick schedule in the hierarchy.

    Attributes:
        coordinators: Dictionary mapping coordinator IDs to CoordinatorAgent instances
        state: SystemAgentState for managing system-level state
        action: Action object for system-level actions
        protocol: Optional protocol for coordinating coordinators
        policy: Optional policy for system-level decision making

    Example:
        Create a system agent with coordinators::

            from heron.agents import SystemAgent, CoordinatorAgent
            from heron.protocols import SetpointProtocol

            # Extend SystemAgent for your domain
            class GridSystemAgent(SystemAgent):
                def set_state(self):
                    from powergrid.core.features.system import SystemFrequency
                    self.state.features.append(SystemFrequency())

                def _build_subordinates(self, configs, env_id, upstream_id):
                    return {c["id"]: CoordinatorAgent(c["id"]) for c in configs}

            # Create system with coordinators
            system = GridSystemAgent(
                agent_id="grid_system",
                protocol=SetpointProtocol(),
                config={"coordinators": [{"id": "grid_1"}, {"id": "grid_2"}]}
            )
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,

        # coordination params
        protocol: Optional[Protocol] = None,
        policy: Optional[Policy] = None,

        # hierarchy params
        env_id: Optional[str] = None,

        # SystemAgent specific params
        config: Optional[DictType[str, Any]] = None,

        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
    ):
        """Initialize system agent.

        Args:
            agent_id: Unique identifier
            protocol: Optional protocol for coordinating coordinators
            policy: Optional policy for system-level decision making
            env_id: Optional environment ID for multi-environment isolation
            config: System agent configuration dictionary
            tick_config: Timing configuration. Defaults to TickConfig with
                tick_interval=300.0 (system agents tick least frequently).
                Use TickConfig.deterministic() or TickConfig.with_jitter().
        """
        config = config or {}

        # Store protocol and policy
        self.protocol = protocol
        self.policy = policy

        # Initialize state management (before super().__init__ for _build_subordinates)
        self.state = SystemAgentState(
            owner_id=agent_id or "system_agent",
            owner_level=SYSTEM_LEVEL
        )

        # Initialize action
        self.action = Action()

        # Cache for environment state (available to subclasses)
        self._cached_env_state: DictType[str, Any] = {}

        # Build coordinator agents
        coordinator_configs = config.get('coordinators', [])
        coordinators = self._build_subordinates(
            coordinator_configs,
            env_id=env_id,
            upstream_id=agent_id
        )

        super().__init__(
            agent_id=agent_id or "system_agent",
            level=SYSTEM_LEVEL,
            subordinates=coordinators,
            upstream_id=None,  # System agent has no upstream
            env_id=env_id,
            tick_config=tick_config or TickConfig.deterministic(tick_interval=300.0),
        )

        # Initialize state and action via hooks (subclasses override these)
        self.set_state()
        self.set_action()

    # ============================================
    # Abstract Method Implementations
    # ============================================

    def _build_subordinates(
        self,
        configs: List[DictType[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, CoordinatorAgent]:
        """Build coordinator agents from configuration.

        Override this in domain-specific subclasses to create appropriate
        coordinator types based on configuration.

        Args:
            configs: List of coordinator configuration dictionaries
            env_id: Environment ID for multi-environment isolation
            upstream_id: This agent's ID (coordinators' upstream)

        Returns:
            Dictionary mapping coordinator IDs to CoordinatorAgent instances
        """
        # Default implementation - override in subclasses
        return {}

    def _get_subordinate_obs_key(self) -> str:
        """Get the observation key for coordinator observations."""
        return OBS_KEY_COORDINATOR_OBS

    def _get_state_obs_key(self) -> str:
        """Get the observation key for system state."""
        return OBS_KEY_SYSTEM_STATE

    def _get_subordinate_action_dim(self, subordinate: CoordinatorAgent) -> int:
        """Get the action dimension for a coordinator agent.

        Args:
            subordinate: CoordinatorAgent instance

        Returns:
            Total action dimension from the coordinator's joint action space
        """
        space = subordinate.get_joint_action_space()
        return self._get_space_dim(space)

    def _get_env_state_key(self) -> str:
        """Get the key for system state in env_state dict."""
        return 'system'

    def _get_env_subordinate_key(self) -> str:
        """Get the key for coordinator updates in env_state dict."""
        return 'coordinators'

    # ============================================
    # Extension Hooks (Override in subclasses)
    # ============================================

    def set_state(self) -> None:
        """Define/initialize the system-specific state.

        Override in subclasses to add system-level features to self.state.
        Called during __init__ after state object is created.

        Example:
            def set_state(self):
                from powergrid.core.features.system import SystemFrequency
                self.state.features.append(SystemFrequency(frequency_hz=60.0))
        """
        pass

    def set_action(self) -> None:
        """Define/initialize the system-specific action space.

        Override in subclasses to configure system-level actions.
        Called during __init__ after action object is created.

        Example:
            def set_action(self):
                self.action.set_specs(
                    dim_c=1,  # e.g., frequency regulation signal
                    range=(np.array([-0.1]), np.array([0.1]))
                )
        """
        pass

    def reset_system(self, *args, **kwargs) -> None:
        """Reset system to initial state.

        Override in subclasses to implement custom reset logic.
        Called by reset() after resetting base agent and coordinators.

        Args:
            *args: Positional arguments for reset
            **kwargs: Keyword arguments for reset
        """
        pass

    # ============================================
    # Core Agent Lifecycle Methods (Both Modes)
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset system agent and all coordinators. [Both Modes]

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed, **kwargs)

        # Clear cached environment state
        self._cached_env_state = {}

        # Call subclass reset hook
        self.reset_system(**kwargs)

    # ============================================
    # Environment Interface Overrides (Both Modes)
    # ============================================

    def update_from_environment(self, env_state: DictType[str, Any]) -> None:
        """Update internal state from environment. [Both Modes]

        This method should be called by the environment to provide
        the system agent with the current environment state.

        Override in subclasses for domain-specific state updates.

        Args:
            env_state: Current environment state
        """
        # Cache environment state for subclass access
        self._cached_env_state = env_state or {}

        # Call parent implementation
        super().update_from_environment(env_state)

    def get_state_for_environment(self) -> DictType[str, Any]:
        """Get agent actions/state for environment. [Both Modes]

        This method should be called by the environment to get
        the agents' actions to apply.

        Override in subclasses for domain-specific state collection.

        Returns:
            Dict containing agent actions/state updates
        """
        result = {
            'system_state': self.state.to_dict(),
            'coordinators': {}
        }

        for coord_id, coordinator in self.subordinates.items():
            result['coordinators'][coord_id] = coordinator.get_state_for_environment()

        return result

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
            for coord_id in self.subordinates:
                self.send_message(message, recipient_id=coord_id)

    # ============================================
    # Convenience Property
    # ============================================

    @property
    def coordinators(self) -> DictType[AgentID, CoordinatorAgent]:
        """Alias for subordinates - more descriptive for SystemAgent context."""
        return self.subordinates

    @coordinators.setter
    def coordinators(self, value: DictType[AgentID, CoordinatorAgent]) -> None:
        """Set coordinators (subordinates)."""
        self.subordinates = value

    # ============================================
    # Utility Methods (Both Modes)
    # ============================================

    def __repr__(self) -> str:
        num_coords = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"SystemAgent(id={self.agent_id}, coordinators={num_coords}, protocol={protocol_name})"

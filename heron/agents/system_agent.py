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

from dataclasses import dataclass
from typing import Any, Dict as DictType, List, Optional, TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.observation import Observation
from heron.core.state import State, SystemAgentState
from heron.utils.typing import AgentID

if TYPE_CHECKING:
    from heron.core.policies import Policy
    from heron.protocols.base import Protocol


SYSTEM_LEVEL = 3  # Level identifier for system-level agents


@dataclass
class SystemConfig:
    """Configuration for SystemAgent initialization.

    Attributes:
        name: System agent name
        state_config: Configuration for system state features
        coordinator_build_strategy: How to build coordinators ("config", "manual", "none")
    """
    name: str = "system_agent"
    state_config: DictType[str, Any] = None
    coordinator_build_strategy: str = "config"

    def __post_init__(self):
        if self.state_config is None:
            self.state_config = {}


class SystemAgent(Agent):
    """System-level agent for top-level coordination and environment interface.

    SystemAgent serves as the interface between the environment and the agent
    hierarchy. It handles:
    - System-wide state management
    - Visibility filtering for constrained information
    - Communication with coordinators
    - Protocol-based coordination of coordinators
    - Policy-based system-level decision making

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

                def _build_coordinators(self, configs, env_id, upstream_id):
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
        protocol: Optional["Protocol"] = None,
        policy: Optional["Policy"] = None,

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
            protocol: Optional protocol for coordinating coordinators
            policy: Optional policy for system-level decision making
            env_id: Optional environment ID for multi-environment isolation
            config: System agent configuration dictionary
            tick_interval: Time between agent ticks (default 300s for system level)
            obs_delay: Observation delay
            act_delay: Action delay
            msg_delay: Message delay
        """
        config = config or {}

        # Store protocol and policy
        self.protocol = protocol
        self.policy = policy

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

        # Initialize state management
        self.state = SystemAgentState(
            owner_id=self.agent_id,
            owner_level=SYSTEM_LEVEL
        )

        # Initialize action
        self.action = Action()

        # Cache for environment state
        self._cached_env_state: DictType[str, Any] = {}

        # Initialize state and action via hooks (subclasses override these)
        self._init_state()
        self._init_action()

    def _init_state(self) -> None:
        """Initialize system-specific state attributes.

        Calls set_state() hook for subclass customization.
        """
        self.set_state()

    def _init_action(self) -> None:
        """Initialize system-specific action space.

        Calls set_action() hook for subclass customization.
        """
        self.set_action()

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

        # Reset state
        if hasattr(self, 'state') and self.state is not None:
            self.state.reset()

        # Reset policy if present
        if self.policy is not None and hasattr(self.policy, 'reset'):
            self.policy.reset()

        # Clear cached environment state
        self._cached_env_state = {}

        # Reset coordinators
        for _, coordinator in self.coordinators.items():
            coordinator.reset(seed=seed, **kwargs)

        # Call subclass reset hook
        self.reset_system(**kwargs)

    # ============================================
    # Synchronous Execution (Option A - Training)
    # ============================================

    def observe(self, global_state: Optional[DictType[str, Any]] = None, *args, **kwargs) -> Observation:
        """Collect observations from coordinators. [Both Modes]

        - Training (Option A): Called by environment
        - Testing (Option B): Called internally by tick()

        Args:
            global_state: Environment state
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Aggregated observation from all coordinators
        """
        # Collect coordinator observations
        coordinator_obs = {}
        for coord_id, coordinator in self.coordinators.items():
            coordinator_obs[coord_id] = coordinator.observe(global_state)

        # Build local observation using hook (subclasses can customize)
        local_observation = self._build_system_observation(
            coordinator_obs, *args, **kwargs
        )

        return Observation(
            timestamp=self._timestep,
            local=local_observation,
            global_info=global_state or {},
        )

    def _build_system_observation(
        self,
        coordinator_obs: DictType[AgentID, Observation],
        *args,
        **kwargs
    ) -> Any:
        """Build system-level observation from coordinator observations.

        Override in subclasses for custom aggregation logic.

        Args:
            coordinator_obs: Dictionary mapping coordinator IDs to their observations
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Aggregated system observation dictionary
        """
        return {
            "coordinator_obs": coordinator_obs,
            "system_state": self.state.vector() if hasattr(self, 'state') and self.state else None,
        }

    def act(self, observation: Observation, upstream_action: Any = None) -> Optional[Any]:
        """Compute and distribute actions to coordinators. [Training Only - Direct Call]

        Note: In Testing (Option B), tick() handles action distribution via
        MESSAGE_DELIVERY events instead of calling this method directly.

        Args:
            observation: Aggregated observation
            upstream_action: Pre-computed action to distribute (optional)

        Returns:
            The action that was distributed (for policy training)
        """
        # Determine action: from upstream, policy, or None
        if upstream_action is not None:
            action = upstream_action
        elif self.policy is not None:
            action = self.policy.forward(observation)
        else:
            # No action to distribute
            return None

        # Coordinate subordinates using unified method
        self.coordinate_coordinators(observation, action)

        return action

    def coordinate_coordinators(
        self,
        observation: Observation,
        action: Any,
    ) -> None:
        """Unified coordination method using protocol.

        Args:
            observation: Current observation
            action: Computed action from system
        """
        coordinator_obs = observation.local.get("coordinator_obs", {})

        # Use protocol if available
        if self.protocol is not None:
            self._apply_protocol_coordination(observation, action, coordinator_obs)
        else:
            # Simple action distribution
            self._apply_simple_coordination(action, coordinator_obs)

    def _apply_protocol_coordination(
        self,
        observation: Observation,
        action: Any,
        coordinator_obs: DictType[AgentID, Observation],
    ) -> None:
        """Apply coordination using protocol.

        Args:
            observation: Current observation
            action: System action
            coordinator_obs: Coordinator observations
        """
        # Get coordinator states from observations
        coordinator_states = {
            coord_id: obs.local for coord_id, obs in coordinator_obs.items()
        }

        # Build context for protocol
        context = {
            "subordinates": self.coordinators,
            "system_id": self.agent_id,
            "coordinator_action": action,
            "timestamp": self._timestep,
        }

        # Get coordination messages and actions from protocol
        messages, actions = self.protocol.coordinate(
            coordinator_state=observation.local,
            subordinate_states=coordinator_states,
            coordinator_action=action,
            context=context
        )

        # Apply actions to coordinators
        for coord_id, coord_action in actions.items():
            if coord_id in self.coordinators and coord_action is not None:
                obs = coordinator_obs.get(coord_id)
                if obs:
                    self.coordinators[coord_id].act(obs, upstream_action=coord_action)

    def _apply_simple_coordination(
        self,
        action: Any,
        coordinator_obs: DictType[AgentID, Observation],
    ) -> None:
        """Apply simple action distribution without protocol.

        Args:
            action: System action
            coordinator_obs: Coordinator observations
        """
        # Distribute actions to coordinators
        actions = self._distribute_action_to_coordinators(action, coordinator_obs)

        for coord_id, coord_action in actions.items():
            if coord_id in self.coordinators and coord_action is not None:
                obs = coordinator_obs.get(coord_id)
                if obs:
                    self.coordinators[coord_id].act(obs, upstream_action=coord_action)

    def _distribute_action_to_coordinators(
        self,
        action: Any,
        coordinator_obs: DictType[AgentID, Observation]
    ) -> DictType[AgentID, Any]:
        """Distribute system action to coordinators.

        Override for custom distribution strategies.

        Args:
            action: System-level action
            coordinator_obs: Coordinator observations

        Returns:
            Dict mapping coordinator IDs to their actions
        """
        if isinstance(action, dict):
            return action  # Already per-coordinator

        # Simple split by coordinator action dimensions
        return self._simple_action_distribution(action, coordinator_obs)

    def _simple_action_distribution(
        self,
        action: Any,
        coordinator_obs: DictType[AgentID, Observation]
    ) -> DictType[AgentID, Any]:
        """Simple action distribution without protocol.

        Args:
            action: System action (array or scalar)
            coordinator_obs: Coordinator observations

        Returns:
            Dict mapping coordinator IDs to their actions
        """
        actions = {}

        if isinstance(action, dict):
            return action

        action_arr = np.asarray(action)
        offset = 0

        for coord_id, coordinator in self.coordinators.items():
            if hasattr(coordinator, 'get_joint_action_space'):
                space = coordinator.get_joint_action_space()
                if hasattr(space, 'shape') and space.shape:
                    dim = space.shape[0]
                else:
                    dim = 1
                if offset + dim <= len(action_arr):
                    actions[coord_id] = action_arr[offset:offset + dim]
                else:
                    actions[coord_id] = action_arr  # Broadcast
                offset += dim
            else:
                actions[coord_id] = action_arr  # Broadcast same action

        return actions

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
        3. Computes system-level action (via policy or external)
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

        # Determine action: from message broker or policy
        upstream_action = None
        actions = self.receive_action_messages()
        if actions:
            upstream_action = actions[-1]  # Use most recent action

        # Use policy if no external action
        if upstream_action is None and self.policy is not None:
            upstream_action = self.policy.forward(observation)

        if upstream_action is None:
            # No action to distribute
            return

        # Distribute actions to coordinators via MESSAGE_DELIVERY events
        coordinator_obs = observation.local.get("coordinator_obs", {})
        actions_to_distribute = self._distribute_action_to_coordinators(
            upstream_action, coordinator_obs
        )

        for coord_id, action in actions_to_distribute.items():
            if coord_id in self.coordinators and action is not None:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=coord_id,
                    message={"action": action},
                    delay=self.msg_delay,
                )

    # ============================================
    # Environment Interface Methods (Both Modes)
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

        # Update internal state with environment data
        if hasattr(self, 'state') and self.state and env_state:
            # Extract system-level state updates
            system_updates = env_state.get('system', {})
            if system_updates:
                self.state.update(system_updates)

        # Propagate to coordinators if needed
        for coord_id, coordinator in self.coordinators.items():
            coord_state = env_state.get('coordinators', {}).get(coord_id, {}) if env_state else {}
            if hasattr(coordinator, 'update_from_environment'):
                coordinator.update_from_environment(coord_state)

    def get_state_for_environment(self) -> DictType[str, Any]:
        """Get agent actions/state for environment. [Both Modes]

        This method should be called by the environment to get
        the agents' actions to apply.

        Override in subclasses for domain-specific state collection.

        Returns:
            Dict containing agent actions/state updates
        """
        result = {
            'system_state': self.state.to_dict() if hasattr(self, 'state') and self.state else {},
            'coordinators': {}
        }

        for coord_id, coordinator in self.coordinators.items():
            if hasattr(coordinator, 'get_state_for_environment'):
                result['coordinators'][coord_id] = coordinator.get_state_for_environment()
            else:
                # Fallback: collect device states
                coord_result = {}
                if hasattr(coordinator, 'subordinate_agents'):
                    coord_result['devices'] = {
                        dev_id: dev.state.to_dict() if hasattr(dev, 'state') else {}
                        for dev_id, dev in coordinator.subordinate_agents.items()
                    }
                result['coordinators'][coord_id] = coord_result

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
            for coord_id in self.coordinators:
                self.send_message(message, recipient_id=coord_id)

    # ============================================
    # Space Construction Methods (Both Modes)
    # ============================================

    def get_joint_action_space(self) -> gym.Space:
        """Construct combined action space for all coordinators.

        Returns:
            Gymnasium space representing joint action space
        """
        low_parts, high_parts, discrete_n = [], [], []

        for sp in self.get_coordinator_action_spaces().values():
            if isinstance(sp, Box):
                low_parts.append(sp.low)
                high_parts.append(sp.high)
            elif isinstance(sp, Dict):
                if 'continuous' in sp.spaces or 'c' in sp.spaces:
                    cont_space = sp.spaces.get('continuous', sp.spaces.get('c'))
                    low_parts.append(cont_space.low)
                    high_parts.append(cont_space.high)
                if 'discrete' in sp.spaces or 'd' in sp.spaces:
                    disc_space = sp.spaces.get('discrete', sp.spaces.get('d'))
                    if isinstance(disc_space, Discrete):
                        discrete_n.append(disc_space.n)
                    elif isinstance(disc_space, MultiDiscrete):
                        discrete_n.extend(list(disc_space.nvec))
            elif isinstance(sp, Discrete):
                discrete_n.append(sp.n)
            elif isinstance(sp, MultiDiscrete):
                discrete_n.extend(list(sp.nvec))

        low = np.concatenate(low_parts) if low_parts else np.array([])
        high = np.concatenate(high_parts) if high_parts else np.array([])

        if len(low) and len(discrete_n):
            return Dict({
                "continuous": Box(low=low, high=high, dtype=np.float32),
                'discrete': MultiDiscrete(discrete_n)
            })
        elif len(low):
            return Box(low=low, high=high, dtype=np.float32)
        elif len(discrete_n):
            return MultiDiscrete(discrete_n)
        else:
            return Discrete(1)

    def get_coordinator_action_spaces(self) -> DictType[str, gym.Space]:
        """Get action spaces for all coordinators.

        Returns:
            Dictionary mapping coordinator IDs to their action spaces
        """
        spaces = {}
        for coord_id, coord in self.coordinators.items():
            if hasattr(coord, 'get_joint_action_space'):
                spaces[coord_id] = coord.get_joint_action_space()
            elif hasattr(coord, 'action_space'):
                spaces[coord_id] = coord.action_space
        return spaces

    def get_joint_observation_space(self) -> gym.Space:
        """Construct combined observation space for all coordinators.

        Returns:
            Gymnasium space representing joint observation space
        """
        obs_parts = []

        for coord in self.coordinators.values():
            if hasattr(coord, 'get_joint_observation_space'):
                space = coord.get_joint_observation_space()
                if isinstance(space, Box):
                    obs_parts.append(space.shape[0] if space.shape else 1)
            elif hasattr(coord, 'observation_space'):
                space = coord.observation_space
                if isinstance(space, Box):
                    obs_parts.append(space.shape[0] if space.shape else 1)

        # Add system state size
        system_state_size = len(self.state.vector()) if hasattr(self, 'state') and self.state else 0
        total_size = sum(obs_parts) + system_state_size

        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,) if total_size > 0 else (1,),
            dtype=np.float32
        )

    # ============================================
    # Utility Methods (Both Modes)
    # ============================================

    def __repr__(self) -> str:
        num_coords = len(self.coordinators)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"SystemAgent(id={self.agent_id}, coordinators={num_coords}, protocol={protocol_name})"

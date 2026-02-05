"""Hierarchical agent base class for agents that manage subordinates.

HierarchicalAgent provides shared functionality for CoordinatorAgent and
SystemAgent, including:
- Subordinate management and observation aggregation
- Action distribution (via protocol or simple split)
- Event-driven tick() with message scheduling
- Environment interface (update_from_environment, get_state_for_environment)
- Joint observation/action space construction

This class should NOT be instantiated directly. Use CoordinatorAgent or
SystemAgent instead.
"""

from abc import abstractmethod
from typing import Any, Dict as DictType, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.core.policies import Policy
from heron.utils.typing import AgentID
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig
from heron.scheduling.scheduler import EventScheduler


class HierarchicalAgent(Agent):
    """Base class for agents that manage subordinates (L2+ agents).

    HierarchicalAgent provides the common coordination patterns shared by
    CoordinatorAgent and SystemAgent:

    - **Subordinate Management**: Build and manage a collection of subordinate agents
    - **Observation Aggregation**: Collect observations from all subordinates
    - **Action Distribution**: Distribute actions to subordinates via protocol or simple split
    - **Event-Driven Execution**: tick() method for async message scheduling
    - **Environment Interface**: Bidirectional state exchange with environment

    Subclasses must implement:
    - `_build_subordinates()`: Create subordinate agents from config
    - `_get_subordinate_obs_key()`: Return observation key for subordinate observations
    - `_get_state_obs_key()`: Return observation key for own state
    - `_get_subordinate_action_dim()`: Get action dimension for a subordinate

    Attributes:
        subordinates: Dictionary mapping subordinate IDs to Agent instances
        state: Agent state object (type varies by subclass)
        protocol: Optional coordination protocol
        policy: Optional decision-making policy
    """

    # Subclass should set these
    subordinates: DictType[AgentID, Agent]
    state: Any  # FieldAgentState, CoordinatorAgentState, or SystemAgentState
    protocol: Optional[Protocol]
    policy: Optional[Policy]

    def __init__(
        self,
        agent_id: AgentID,
        level: int,
        subordinates: DictType[AgentID, Agent],
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        tick_config: Optional[TickConfig] = None,
    ):
        """Initialize hierarchical agent.

        Args:
            agent_id: Unique identifier
            level: Hierarchy level (2=coordinator, 3=system)
            subordinates: Dictionary of subordinate agents
            upstream_id: Optional parent agent ID
            env_id: Optional environment ID for multi-environment isolation
            tick_config: Timing configuration for event-driven scheduling
        """
        super().__init__(
            agent_id=agent_id,
            level=level,
            upstream_id=upstream_id,
            subordinates=subordinates,
            env_id=env_id,
            tick_config=tick_config or TickConfig.deterministic(),
        )

    # ============================================
    # Abstract Methods (Must Override)
    # ============================================

    @abstractmethod
    def _build_subordinates(
        self,
        configs: List[DictType[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, Agent]:
        """Build subordinate agents from configuration.

        Args:
            configs: List of subordinate configuration dictionaries
            env_id: Environment ID for multi-environment isolation
            upstream_id: This agent's ID (subordinates' upstream)

        Returns:
            Dictionary mapping agent IDs to Agent instances
        """
        pass

    @abstractmethod
    def _get_subordinate_obs_key(self) -> str:
        """Get the observation key for subordinate observations.

        Returns:
            Observation key string (e.g., OBS_KEY_SUBORDINATE_OBS, OBS_KEY_COORDINATOR_OBS)
        """
        pass

    @abstractmethod
    def _get_state_obs_key(self) -> str:
        """Get the observation key for own state.

        Returns:
            Observation key string (e.g., OBS_KEY_COORDINATOR_STATE, OBS_KEY_SYSTEM_STATE)
        """
        pass

    @abstractmethod
    def _get_subordinate_action_dim(self, subordinate: Agent) -> int:
        """Get the action dimension for a subordinate agent.

        Args:
            subordinate: Subordinate agent

        Returns:
            Total action dimension for this subordinate
        """
        pass

    # ============================================
    # Core Agent Lifecycle Methods (Both Modes)
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent and all subordinates. [Both Modes]

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)

        # Reset own state
        if hasattr(self, 'state') and self.state is not None:
            self.state.reset()

        # Reset policy if present
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.reset()

        # Reset subordinates
        for subordinate in self.subordinates.values():
            subordinate.reset(seed=seed, **kwargs)

    # ============================================
    # Synchronous Execution (Option A - Training)
    # ============================================

    def observe(
        self,
        global_state: Optional[DictType[str, Any]] = None,
        proxy: Optional[Agent] = None,
        *args,
        **kwargs
    ) -> Observation:
        """Collect observations from subordinate agents. [Both Modes]

        - Training (Option A): Called by environment or parent agent
        - Testing (Option B): Called internally by tick()

        Args:
            global_state: Environment state
            proxy: Optional ProxyAgent for delayed observations (passed to subordinates)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Aggregated observation from all subordinates
        """
        # Collect subordinate observations (pass proxy for neighbor observations)
        subordinate_obs = {}
        for sub_id, subordinate in self.subordinates.items():
            subordinate_obs[sub_id] = subordinate.observe(global_state, proxy=proxy)

        # Build local observation using hook (subclasses can customize)
        local_observation = self._build_local_observation(subordinate_obs, *args, **kwargs)

        # Global info aggregation (override in subclasses if needed)
        global_info = global_state or {}

        return Observation(
            timestamp=self._timestep,
            local=local_observation,
            global_info=global_info,
        )

    def _build_local_observation(
        self,
        subordinate_obs: DictType[AgentID, Observation],
        *args,
        **kwargs
    ) -> DictType[str, Any]:
        """Build local observation from subordinate observations. [Both Modes]

        Override in subclasses for custom observation building.

        Args:
            subordinate_obs: Dictionary mapping agent IDs to their observations
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Aggregated local observation dictionary
        """
        return {
            self._get_subordinate_obs_key(): subordinate_obs,
            self._get_state_obs_key(): self.state.vector() if self.state else np.array([])
        }

    def act(self, observation: Observation, upstream_action: Any = None) -> Optional[Any]:
        """Compute and distribute actions to subordinates. [Training Only - Direct Call]

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
        self.coordinate_subordinates(observation, action)

        return action

    # ============================================
    # Coordination Methods (Training Only)
    # ============================================

    def coordinate_subordinates(
        self,
        observation: Observation,
        action: Any,
    ) -> None:
        """Unified coordination method using protocol. [Training Only]

        Coordinates subordinate actions using the protocol's communication and
        action components, or simple action distribution if no protocol.

        Args:
            observation: Current observation
            action: Computed action from this agent
        """
        subordinate_obs = observation.local.get(self._get_subordinate_obs_key(), {})

        if self.protocol is not None:
            self._apply_protocol_coordination(observation, action, subordinate_obs)
        else:
            self._apply_simple_coordination(action, subordinate_obs)

    def _apply_protocol_coordination(
        self,
        observation: Observation,
        action: Any,
        subordinate_obs: DictType[AgentID, Observation],
    ) -> None:
        """Apply coordination using protocol.

        Args:
            observation: Current observation
            action: Action from this agent
            subordinate_obs: Subordinate observations
        """
        # Get subordinate states from observations
        subordinate_states = {
            sub_id: obs.local for sub_id, obs in subordinate_obs.items()
        }

        # Build context for protocol
        context = {
            "subordinates": self.subordinates,
            "coordinator_id": self.agent_id,
            "coordinator_action": action,
            "timestamp": self._timestep,
        }

        # Get coordination messages and actions from protocol
        messages, actions = self.protocol.coordinate(
            coordinator_state=observation.local,
            subordinate_states=subordinate_states,
            coordinator_action=action,
            context=context
        )

        # Apply coordination (send messages and actions to subordinates)
        self._apply_coordination(messages, actions, subordinate_obs)

    def _apply_simple_coordination(
        self,
        action: Any,
        subordinate_obs: DictType[AgentID, Observation],
    ) -> None:
        """Apply simple action distribution without protocol.

        Args:
            action: Action from this agent
            subordinate_obs: Subordinate observations
        """
        actions = self._simple_action_distribution(action)

        for sub_id, sub_action in actions.items():
            if sub_id in self.subordinates and sub_action is not None:
                obs = subordinate_obs.get(sub_id)
                if obs:
                    self.subordinates[sub_id].act(obs, upstream_action=sub_action)

    def _apply_coordination(
        self,
        messages: DictType[AgentID, DictType[str, Any]],
        actions: DictType[AgentID, Any],
        subordinate_obs: DictType[AgentID, Observation],
    ) -> None:
        """Apply coordination: send messages and apply actions to subordinates. [Training Only]

        Args:
            messages: Coordination messages to send
            actions: Actions to apply to subordinates
            subordinate_obs: Subordinate observations for context
        """
        # Send messages to subordinates via message broker (informational)
        if self._message_broker is not None:
            for sub_id, message in messages.items():
                if sub_id in self.subordinates:
                    self.send_message(message, recipient_id=sub_id)

        # Apply actions directly to subordinates
        for sub_id, action in actions.items():
            if sub_id in self.subordinates and action is not None:
                obs = subordinate_obs.get(sub_id)
                if obs:
                    self.subordinates[sub_id].act(obs, upstream_action=action)

    def _simple_action_distribution(self, action: Any) -> DictType[AgentID, Any]:
        """Simple action distribution without protocol. [Both Modes]

        Distributes action to subordinates by splitting based on their
        action dimensions.

        Args:
            action: Joint action to distribute (array, dict, or Action object)

        Returns:
            Dict mapping subordinate IDs to individual actions
        """
        if isinstance(action, dict):
            return action  # Already per-agent dict

        # Handle Action objects by converting to vector
        from heron.core.action import Action
        if isinstance(action, Action):
            action = action.vector()

        # Split flat array by subordinate action dimensions
        action_arr = np.asarray(action)
        actions = {}
        offset = 0

        for sub_id, subordinate in self.subordinates.items():
            dim = self._get_subordinate_action_dim(subordinate)
            if dim > 0:
                actions[sub_id] = action_arr[offset:offset + dim]
                offset += dim

        return actions

    # ============================================
    # Event-Driven Execution (Option B - Testing)
    # ============================================

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
        global_state: Optional[DictType[str, Any]] = None,
        proxy: Optional[Agent] = None,
    ) -> None:
        """Execute one tick in event-driven mode. [Testing Only]

        In Option B, HierarchicalAgent:
        1. Updates timestep
        2. Checks message broker for upstream action (if not top-level)
        3. Gets observation from subordinates
        4. Computes action using upstream action or policy
        5. Schedules MESSAGE_DELIVERY events to subordinates with msg_delay

        Args:
            scheduler: EventScheduler for scheduling future events
            current_time: Current simulation time
            global_state: Optional global state for observation
            proxy: Optional ProxyAgent for delayed observations
        """
        self._timestep = current_time

        # Check message broker for upstream action (non-top-level agents)
        upstream_action = self._get_upstream_action()

        # Get observation from subordinates (pass proxy for neighbor observations)
        observation = self.observe(global_state, proxy=proxy)
        self._last_observation = observation

        # Compute action
        if upstream_action is not None:
            action = upstream_action
        elif self.policy is not None:
            action = self.policy.forward(observation)
        else:
            # No action to take
            return

        # Distribute actions to subordinates via MESSAGE_DELIVERY events
        self._distribute_actions_async(scheduler, observation, action, current_time)

    def _get_upstream_action(self) -> Optional[Any]:
        """Get upstream action from message broker. [Testing Only]

        Override in subclasses to customize upstream action retrieval.
        Top-level agents (SystemAgent) should return None.

        Returns:
            Action from upstream agent, or None if no upstream or no action
        """
        if self.upstream_id is None:
            return None  # Top-level agent

        actions = self.receive_action_messages()
        if actions:
            return actions[-1]  # Use most recent action
        return None

    def _distribute_actions_async(
        self,
        scheduler: EventScheduler,
        observation: Observation,
        action: Any,
        current_time: float,
    ) -> None:
        """Distribute actions to subordinates via MESSAGE_DELIVERY events. [Testing Only]

        Args:
            scheduler: EventScheduler for scheduling events
            observation: Current observation
            action: Action to distribute
            current_time: Current simulation time
        """
        subordinate_obs = observation.local.get(self._get_subordinate_obs_key(), {})

        if self.protocol is not None:
            # Use protocol for action distribution
            subordinate_states = {
                sub_id: obs.local for sub_id, obs in subordinate_obs.items()
            }
            context = {
                "subordinates": self.subordinates,
                "coordinator_id": self.agent_id,
                "coordinator_action": action,
                "timestamp": current_time,
            }
            _, actions = self.protocol.coordinate(
                coordinator_state=observation.local,
                subordinate_states=subordinate_states,
                coordinator_action=action,
                context=context
            )
        else:
            # Simple action distribution
            actions = self._simple_action_distribution(action)

        # Schedule message delivery to subordinates (with msg_delay)
        for sub_id, sub_action in actions.items():
            if sub_id in self.subordinates and sub_action is not None:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=sub_id,
                    message={"action": sub_action},
                    delay=self._tick_config.msg_delay,
                )

    # ============================================
    # Environment Interface Methods (Both Modes)
    # ============================================

    def update_from_environment(self, env_state: DictType[str, Any]) -> None:
        """Receive state updates from environment. [Both Modes]

        Called by the environment (or parent agent) after processing
        action effects. Updates own state and propagates to subordinates.

        Args:
            env_state: State updates for this agent and its subordinates.
                Expected structure varies by level.
        """
        if not env_state:
            return

        # Update own state (subclasses define the key)
        own_state_key = self._get_env_state_key()
        if own_updates := env_state.get(own_state_key, {}):
            self.state.update(own_updates)

        # Propagate to subordinates
        subordinate_key = self._get_env_subordinate_key()
        subordinate_updates = env_state.get(subordinate_key, {})
        for sub_id, subordinate in self.subordinates.items():
            sub_state = subordinate_updates.get(sub_id, {})
            if hasattr(subordinate, 'update_from_environment'):
                subordinate.update_from_environment(sub_state)

    def _get_env_state_key(self) -> str:
        """Get the key for own state in env_state dict.

        Override in subclasses. Default: 'coordinator'
        """
        return 'coordinator'

    def _get_env_subordinate_key(self) -> str:
        """Get the key for subordinate updates in env_state dict.

        Override in subclasses. Default: 'subordinates'
        """
        return 'subordinates'

    def get_state_for_environment(self) -> DictType[str, Any]:
        """Provide current state to environment. [Both Modes]

        Called by the environment to collect agent and subordinate states
        including actions to apply.

        Returns:
            Dict containing agent and subordinate states/actions
        """
        own_state_key = self._get_env_state_key() + '_state'
        subordinate_key = self._get_env_subordinate_key()

        result = {
            own_state_key: self.state.to_dict() if self.state else {},
            subordinate_key: {}
        }

        for sub_id, subordinate in self.subordinates.items():
            if hasattr(subordinate, 'get_state_for_environment'):
                result[subordinate_key][sub_id] = subordinate.get_state_for_environment()
            else:
                # Fallback
                result[subordinate_key][sub_id] = {
                    'state': subordinate.state.to_dict() if hasattr(subordinate, 'state') else {}
                }

        return result

    # ============================================
    # Space Construction Methods (Both Modes)
    # ============================================

    def get_subordinate_action_spaces(self) -> DictType[str, gym.Space]:
        """Get action spaces for all subordinate agents. [Both Modes]

        Returns:
            Dictionary mapping agent IDs to their action spaces
        """
        spaces = {}
        for sub_id, subordinate in self.subordinates.items():
            if hasattr(subordinate, 'get_joint_action_space'):
                spaces[sub_id] = subordinate.get_joint_action_space()
            elif hasattr(subordinate, 'action_space'):
                spaces[sub_id] = subordinate.action_space
        return spaces

    def get_subordinate_observation_spaces(self) -> DictType[str, gym.Space]:
        """Get observation spaces for all subordinate agents. [Both Modes]

        Returns:
            Dictionary mapping agent IDs to their observation spaces
        """
        spaces = {}
        for sub_id, subordinate in self.subordinates.items():
            if hasattr(subordinate, 'get_joint_observation_space'):
                spaces[sub_id] = subordinate.get_joint_observation_space()
            elif hasattr(subordinate, 'observation_space'):
                spaces[sub_id] = subordinate.observation_space
        return spaces

    def get_joint_observation_space(self) -> gym.Space:
        """Construct combined observation space for all subordinates. [Both Modes]

        Returns:
            Gymnasium space representing joint observation space
        """
        obs_parts = []

        for space in self.get_subordinate_observation_spaces().values():
            if isinstance(space, Box):
                obs_parts.append(space.shape[0] if space.shape else 1)

        # Add own state size
        own_state_size = len(self.state.vector()) if self.state else 0
        total_size = sum(obs_parts) + own_state_size

        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,) if total_size > 0 else (1,),
            dtype=np.float32
        )

    def get_joint_action_space(self) -> gym.Space:
        """Construct combined action space for all subordinates. [Both Modes]

        Returns:
            Gymnasium space representing joint action space
        """
        low_parts, high_parts, discrete_n = [], [], []

        for space in self.get_subordinate_action_spaces().values():
            if isinstance(space, Box):
                low_parts.append(space.low)
                high_parts.append(space.high)
            elif isinstance(space, Dict):
                # Handle Dict spaces (e.g., continuous + discrete)
                if 'continuous' in space.spaces or 'c' in space.spaces:
                    cont_space = space.spaces.get('continuous', space.spaces.get('c'))
                    low_parts.append(cont_space.low)
                    high_parts.append(cont_space.high)
                if 'discrete' in space.spaces or 'd' in space.spaces:
                    disc_space = space.spaces.get('discrete', space.spaces.get('d'))
                    if isinstance(disc_space, Discrete):
                        discrete_n.append(disc_space.n)
                    elif isinstance(disc_space, MultiDiscrete):
                        discrete_n.extend(list(disc_space.nvec))
            elif isinstance(space, Discrete):
                discrete_n.append(space.n)
            elif isinstance(space, MultiDiscrete):
                discrete_n.extend(list(space.nvec))

        low = np.concatenate(low_parts) if low_parts else np.array([])
        high = np.concatenate(high_parts) if high_parts else np.array([])

        if len(low) and len(discrete_n):
            return Dict({
                "continuous": Box(low=low, high=high, dtype=np.float32),
                'discrete': MultiDiscrete(discrete_n)
            })
        elif len(low):  # Continuous only
            return Box(low=low, high=high, dtype=np.float32)
        elif len(discrete_n):  # Discrete only
            return MultiDiscrete(discrete_n)
        else:  # No actionable agents
            return Discrete(1)

    # ============================================
    # Utility Methods (Both Modes)
    # ============================================

    def _get_space_dim(self, space: gym.Space) -> int:
        """Get the flat dimension of a gymnasium space.

        Args:
            space: Gymnasium space (Box, Discrete, MultiDiscrete, or Dict)

        Returns:
            Total flat dimension of the space
        """
        if isinstance(space, Box):
            return int(np.prod(space.shape))
        elif isinstance(space, Discrete):
            return 1
        elif isinstance(space, MultiDiscrete):
            return len(space.nvec)
        elif isinstance(space, Dict):
            return sum(self._get_space_dim(s) for s in space.spaces.values())
        else:
            return 0

    def __repr__(self) -> str:
        num_subs = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"{self.__class__.__name__}(id={self.agent_id}, subordinates={num_subs}, protocol={protocol_name})"

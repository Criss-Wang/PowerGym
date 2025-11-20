"""Base agent abstraction for hierarchical multi-agent control.

This module provides the core abstractions for agents in the PowerGrid platform,
supporting hierarchical control, agent-to-agent communication, and flexible
observation/action interfaces.

The Agent class supports two execution modes:
1. Direct execution: Simple observe() → act() flow for flat multi-agent systems
2. Hierarchical execution: Message-based recursive step_hierarchical() for complex hierarchies
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import gymnasium as gym

from powergrid.core.observation import AgentID, Observation
from powergrid.messaging.base import MessageBroker, ChannelManager, Message, MessageType

class Agent(ABC):
    """Abstract base class for all agents in the hierarchy.

    Agents can operate at different levels of the control hierarchy:
    - Device level: Individual DERs (DG, ESS, RES, etc.)
    - Grid level: Microgrid controllers, substations
    - System level: ISO, market operator

    Execution Modes:
    1. Direct execution (original API):
       - Use observe() and act() methods
       - Simple, synchronous execution
       - No message broker required

    2. Hierarchical execution (new API):
       - Use step_hierarchical() method
       - Message-based recursive execution
       - Requires message_broker, supports upstream/subordinate relationships

    Attributes:
        agent_id: Unique identifier for this agent
        level: Hierarchy level (1=device, 2=grid, 3=system)
        observation_space: Gymnasium space for observations
        action_space: Gymnasium space for actions
        message_broker: Optional message broker for hierarchical execution
        upstream_id: Optional upstream agent ID for hierarchical execution
        subordinates: Dict of subordinate agents for hierarchical execution
        env_id: Environment ID for multi-environment isolation
    """

    def __init__(
        self,
        agent_id: AgentID,
        level: int = 1,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,

        # communication params
        message_broker: Optional[MessageBroker] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # hierarchical params
        subordinates: Optional[Dict[AgentID, "Agent"]] = None,
    ):
        """Initialize agent.

        Args:
            agent_id: Unique identifier
            level: Hierarchy level (1=device, 2=grid, 3=system)
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            message_broker: Optional message broker for hierarchical execution
            upstream_id: Optional upstream agent ID for hierarchical execution
            subordinates: Optional dict of subordinate agents
            env_id: Optional environment ID for multi-environment isolation
        """
        self.agent_id = agent_id
        self.level = level
        self.observation_space = observation_space
        self.action_space = action_space

        # Direct execution attributes
        self._timestep: int = 0

        # Hierarchical execution attributes (optional)
        self.message_broker = message_broker
        self.upstream_id = upstream_id
        self.subordinates = subordinates or {}
        self.env_id = env_id
        self.subordinates_info: Dict[AgentID, Dict[str, Any]] = {}

        # Setup message channels if broker is available
        if self.message_broker:
            self._setup_channels()

    # ============================================
    # Core Lifecycle Methods
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent to initial state.

        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional reset parameters
        """
        self._timestep = 0

    # ============================================
    # Direct Execution Mode (Original API)
    # ============================================

    @abstractmethod
    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
        """Extract relevant observations from global state.

        Used for direct execution mode where agents simply observe and act
        without hierarchical communication.

        Args:
            global_state: Complete environment state including:
                - bus voltages/angles
                - device states
                - power flow results
                - dataset values (load, price, etc.)

        Returns:
            Structured observation for this agent
        """
        pass

    @abstractmethod
    def act(self, observation: Observation, *args, **kwargs) -> Any:
        """Compute action from observation.

        Used for direct execution mode where agents receive observations
        and return actions without hierarchical communication.

        Args:
            observation: Structured observation from observe()

        Returns:
            Action in the format defined by action_space
        """
        pass

    # ============================================
    # Hierarchical Execution Mode (New API)
    # ============================================

    def step_distributed(self) -> None:
        """Execute one step with hierarchical message-based communication.

        This method implements full recursive hierarchical execution with
        message broker communication. Requires message_broker to be configured.

        Execution Flow:
        1. Receive action from upstream (via message broker)
        2. Derive and send actions to subordinates (via message broker)
        3. Execute subordinate steps recursively
        4. Collect info from subordinates (via message broker)
        5. Derive and execute own action
        6. Publish state updates to environment
        7. Receive power flow results from environment
        8. Build compiled info (own + subordinates)
        9. Send compiled info to upstream (via message broker)

        Raises:
            RuntimeError: If message_broker is not configured

        Example:
            >>> # Setup hierarchical agents
            >>> broker = create_message_broker("memory")
            >>> device = DeviceAgent(
            ...     agent_id="device1",
            ...     message_broker=broker,
            ...     upstream_id="grid1"
            ... )
            >>> grid = GridAgent(
            ...     agent_id="grid1",
            ...     message_broker=broker,
            ...     subordinates={"device1": device}
            ... )
            >>> # Execute hierarchical step
            >>> compiled_info = grid.step_distributed()
        """
        if not self.message_broker:
            raise RuntimeError(
                f"Agent {self.agent_id} requires message_broker for hierarchical execution. "
                "Pass message_broker in constructor to enable this mode."
            )

        # 1. Receive action from upstream via message broker
        upstream_action = self._get_action_from_upstream()
        upstream_info = self._get_info_from_upstream()
        self._update_state_with_upstream_info(upstream_info)

        # 2-4. Handle subordinates if any
        if self.subordinates:
            # Derive actions for subordinates
            downstream_actions = self._derive_downstream_actions(upstream_action)

            # Send actions to subordinates via message broker
            self._send_actions_to_subordinates(downstream_actions)

            # Execute subordinate steps recursively
            self._execute_subordinates()

            # Collect info from subordinates via message broker
            self._collect_subordinates_info()
            self._update_state_with_subordinates_info()

        # 5. Derive and execute own action
        local_action = self._derive_local_action(upstream_action)
        self._execute_local_action(local_action)
        self._update_state_post_step()

        # 6. Publish state updates to environment
        self._update_timestep()
        self._publish_state_updates()

    # ============================================
    # Abstract Methods for Hierarchical Execution
    # ============================================

    def _derive_local_action(self, upstream_action: Optional[Any]) -> Optional[Any]:
        """Derive local action from upstream action.

        Args:
            upstream_action: Action received from upstream agent

        Returns:
            Local action to execute
        """
        raise NotImplementedError

    def _derive_downstream_actions(
        self,
        upstream_action: Optional[Any]
    ) -> Dict[AgentID, Any]:
        """Derive actions for subordinates from upstream action.

        Args:
            upstream_action: Action received from upstream agent

        Returns:
            Dict mapping subordinate IDs to their actions
        """
        if self.subordinates:
            raise NotImplementedError
        return {}

    def _execute_local_action(self, action: Optional[Any]) -> None:
        """Execute own action and update internal state.

        Subclasses should override this to implement their action execution.
        State updates should be published via _publish_state_updates().

        Args:
            action: Action to execute
        """
        raise NotImplementedError

    # ============================================
    # State Update Hooks
    # ============================================

    def _update_state_with_upstream_info(self, upstream_info: Optional[Dict[str, Any]]) -> None:
        """Update internal state based on info received from upstream agent.

        Args:
            upstream_info: Info dict received from upstream agent
        """
        # Default: no update
        pass

    def _update_state_with_subordinates_info(self) -> None:
        """Update internal state based on info received from subordinates."""
        # Default: no update
        pass

    def _update_state_post_step(self) -> None:
        """Update internal state after executing local action.

        This method can be overridden by subclasses to update internal
        state variables after executing the local action.
        """
        # Default: no update
        pass

    def _publish_state_updates(self) -> None:
        """Publish state updates to environment via message broker.

        Subclasses should override this to publish device state changes
        to the environment for power flow computation.
        """
        # Default: no state updates
        pass

    # ============================================
    # Message Broker Communication
    # ============================================
    def _setup_channels(self) -> None:
        """Setup message channels for hierarchical communication.

        Creates all necessary message channels based on upstream/subordinate relationships.
        Called automatically if message_broker is provided in constructor.
        """
        subordinate_ids = list(self.subordinates.keys())
        channels = ChannelManager.agent_channels(
            self.agent_id,
            self.upstream_id,
            subordinate_ids,
            self.env_id
        )

        # Create all required channels
        for channel in channels['subscribe'] + channels['publish']:
            self.message_broker.create_channel(channel)

    def _get_info_from_upstream(self) -> Optional[Dict[str, Any]]:
        """Receive info from upstream via message broker.

        Returns:
            Info from upstream, or None if no upstream or no info available
        """
        if not self.upstream_id or not self.message_broker:
            return None

        info_channel = ChannelManager.info_channel(
            self.upstream_id,
            self.agent_id,
            self.env_id
        )
        info_message = self.message_broker.consume(
            info_channel,
            self.agent_id,
            self.env_id
        )

        if info_message:
            return info_message[-1].payload

        return None

    def _get_action_from_upstream(self) -> Optional[Any]:
        """Receive action from upstream via message broker.

        Returns:
            Action from upstream, or None if no upstream or no action available
        """
        if not self.upstream_id or not self.message_broker:
            return None

        action_channel = ChannelManager.action_channel(
            self.upstream_id,
            self.agent_id,
            self.env_id
        )
        action_message = self.message_broker.consume(
            action_channel,
            self.agent_id,
            self.env_id
        )

        if action_message:
            return action_message[-1].payload.get('action', None)

        return None

    def _send_actions_to_subordinates(self, actions: Dict[AgentID, Any]) -> None:
        """Send actions to subordinates via message broker.

        Args:
            actions: Dict mapping subordinate agent IDs to actions
        """
        if not self.message_broker:
            return

        for sub_id, action in actions.items():
            channel = ChannelManager.action_channel(
                self.agent_id,
                sub_id,
                self.env_id
            )
            message = Message(
                env_id=self.env_id,
                sender_id=self.agent_id,
                recipient_id=sub_id,
                timestamp=self._timestep,
                message_type=MessageType.ACTION,
                payload={'action': action}
            )
            self.message_broker.publish(channel, message)

    def _execute_subordinates(self) -> None:
        """Execute subordinate agent steps recursively."""
        for _, subordinate in self.subordinates.items():
            subordinate.step_distributed()

    def _collect_subordinates_info(self) -> None:
        """Collect info from subordinates via message broker."""
        if not self.message_broker:
            return

        for sub_id in self.subordinates.keys():
            channel = ChannelManager.info_channel(
                sub_id,
                self.agent_id,
                self.env_id
            )
            messages = self.message_broker.consume(
                channel,
                self.agent_id,
                self.env_id
            )

            if messages:
                latest_msg = messages[-1]
                self.subordinates_info[sub_id] = latest_msg.payload

    # ============================================
    # Utility Methods
    # ============================================

    def _update_timestep(self) -> None:
        """Update internal timestep counter."""
        self._timestep += 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, level={self.level})"

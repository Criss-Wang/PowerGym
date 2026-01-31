"""Coordinator-level agents (L2) for the HERON framework.

CoordinatorAgent manages a set of field agents, implementing coordination
protocols like price signals, setpoints, or consensus algorithms.

In synchronous mode (Option A - Training), the coordinator:
1. Collects observations from all subordinates via observe()
2. Computes joint action using centralized policy
3. Distributes actions to subordinates via coordinate_subordinates()
"""

from typing import Any, Dict as DictType, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.core.observation import Observation
from heron.core.state import CoordinatorAgentState
from heron.core.policies import Policy
from heron.utils.typing import AgentID


COORDINATOR_LEVEL = 2  # Level identifier for coordinator-level agents


class CoordinatorAgent(Agent):
    """Coordinator-level agent for managing field agents.

    CoordinatorAgent coordinates multiple field agents using specified protocols
    and optionally a centralized policy for joint decision-making.

    Attributes:
        subordinate_agents: Dictionary mapping agent IDs to FieldAgent instances
        protocol: Coordination protocol for managing subordinate agents
        policy: Optional centralized policy for joint action computation
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        policy: Optional[Policy] = None,
        protocol: Optional["Protocol"] = None,

        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # CoordinatorAgent specific params
        config: Optional[DictType[str, Any]] = None,

        # timing params (for event-driven scheduling)
        tick_interval: float = 60.0,  # Coordinators typically tick less frequently
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,
    ):
        """Initialize coordinator agent.

        Args:
            agent_id: Unique identifier
            policy: Optional centralized policy for joint action computation
            protocol: Protocol for coordinating subordinate agents
            upstream_id: Optional parent agent ID for hierarchy structure
            env_id: Optional environment ID for multi-environment isolation
            config: Coordinator configuration dictionary
            tick_interval: Time between agent ticks (default 60s for coordinators)
            obs_delay: Observation delay
            act_delay: Action delay
            msg_delay: Message delay
        """
        config = config or {}

        self.protocol = protocol
        self.policy = policy
        self.state = CoordinatorAgentState(
            owner_id=agent_id,
            owner_level=COORDINATOR_LEVEL
        )

        # Build subordinate agents
        agent_configs = config.get('agents', [])
        self.subordinate_agents = self._build_subordinate_agents(
            agent_configs,
            env_id=env_id,
            upstream_id=agent_id  # Subordinates' upstream is this coordinator
        )

        super().__init__(
            agent_id=agent_id,
            level=COORDINATOR_LEVEL,
            upstream_id=upstream_id,
            env_id=env_id,
            subordinates=self.subordinate_agents,
            tick_interval=tick_interval,
            obs_delay=obs_delay,
            act_delay=act_delay,
            msg_delay=msg_delay,
        )

    def _build_subordinate_agents(
        self,
        agent_configs: List[DictType[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, FieldAgent]:
        """Build subordinate agents from configuration.

        Override this in domain-specific subclasses to create appropriate
        agent types based on configuration.

        Args:
            agent_configs: List of agent configuration dictionaries
            env_id: Environment ID for multi-environment isolation
            upstream_id: Upstream agent ID (this coordinator)

        Returns:
            Dictionary mapping agent IDs to FieldAgent instances
        """
        # Default implementation - override in subclasses
        return {}

    # ============================================
    # Core Agent Lifecycle Methods (Both Modes)
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset coordinator and all subordinate agents. [Both Modes]

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)

        # Reset subordinate agents
        for _, agent in self.subordinate_agents.items():
            agent.reset(seed=seed, **kwargs)

        # Reset policy
        if self.policy is not None:
            self.policy.reset()

    # ============================================
    # Synchronous Execution (Option A - Training)
    # ============================================

    def observe(self, global_state: Optional[DictType[str, Any]] = None, *args, **kwargs) -> Observation:
        """Collect observations from subordinate agents. [Both Modes]

        - Training (Option A): Called by environment/system agent
        - Testing (Option B): Called internally by tick()

        Args:
            global_state: Environment state

        Returns:
            Aggregated observation from all subordinate agents
        """
        # Collect subordinate observations
        subordinate_obs = {}
        for agent_id, agent in self.subordinate_agents.items():
            subordinate_obs[agent_id] = agent.observe(global_state)
        local_observation = self._build_local_observation(subordinate_obs, *args, **kwargs)

        # Global info aggregation (override in subclasses if needed)
        global_info = global_state or {}

        # Message aggregation (override in subclasses if needed)
        messages = []

        return Observation(
            timestamp=self._timestep,
            local=local_observation,
            global_info=global_info,
            messages=messages,
        )

    def _build_local_observation(
        self,
        subordinate_obs: DictType[AgentID, Observation],
        *args,
        **kwargs
    ) -> Any:
        """Build local observation from subordinate observations. [Both Modes]

        Called by observe() which is used in both modes.

        Args:
            subordinate_obs: Dictionary mapping agent IDs to their observations
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Aggregated local observation dictionary
        """
        return {
            "subordinate_obs": subordinate_obs,
            "coordinator_state": self.state.vector()
        }

    def act(self, observation: Observation, upstream_action: Any = None) -> None:
        """Compute coordination action and distribute to subordinates. [Training Only - Direct Call]

        Note: In Testing (Option B), tick() handles action distribution via
        MESSAGE_DELIVERY events instead of calling this method directly.

        Args:
            observation: Aggregated observation
            upstream_action: Pre-computed action (if any)

        Raises:
            RuntimeError: If no action or policy is provided
        """
        # Get coordinator action from policy if available
        if upstream_action is not None:
            action = upstream_action
        elif self.policy is not None:
            action = self.policy.forward(observation)
        else:
            raise RuntimeError("No action or policy provided for CoordinatorAgent.")

        # Coordinate subordinates using unified method
        self.coordinate_subordinates(observation, action)

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

        In Option B, CoordinatorAgent:
        1. Updates timestep
        2. Checks message broker for upstream action (from SystemAgent if any)
        3. Gets observation from subordinates (potentially stale in async mode)
        4. Computes joint action using policy
        5. Schedules MESSAGE_DELIVERY events to subordinates with msg_delay

        Note: In event-driven mode, the coordinator does NOT call subordinate.tick()
        directly. Instead, it sends messages that subordinates will process on their
        own tick schedule. This models realistic async communication.

        Args:
            scheduler: EventScheduler for scheduling future events
            current_time: Current simulation time
            global_state: Optional global state for observation
            proxy: Optional ProxyAgent for delayed observations
        """
        self._timestep = current_time

        # Check message broker for upstream action (from SystemAgent)
        upstream_action = None
        actions = self.receive_action_messages()
        if actions:
            upstream_action = actions[-1]  # Use most recent action

        # Get observation from subordinates
        # Note: In async mode, subordinates may have stale state
        observation = self.observe(global_state)
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
        subordinate_obs = observation.local.get("subordinate_obs", {})

        if self.protocol is not None:
            # Use protocol for action distribution
            subordinate_states = {
                agent_id: obs.local for agent_id, obs in subordinate_obs.items()
            }
            context = {
                "subordinates": self.subordinate_agents,
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
            actions = self._simple_action_distribution(action, subordinate_obs)

        # Schedule message delivery to subordinates (with msg_delay)
        for agent_id, sub_action in actions.items():
            if agent_id in self.subordinate_agents and sub_action is not None:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=agent_id,
                    message={"action": sub_action},
                    delay=self.msg_delay,
                )

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
        action components. Delegates to centralized or decentralized paths
        based on the protocol's ActionProtocol type.

        Args:
            observation: Current observation
            action: Computed action from coordinator
        """
        # Prepare subordinate states from observations
        subordinate_obs = observation.local.get("subordinate_obs", {})
        subordinate_states = {
            agent_id: obs.local for agent_id, obs in subordinate_obs.items()
        }

        # Prepare context for protocol
        context = {
            "subordinates": self.subordinate_agents,
            "coordinator_id": self.agent_id,
            "coordinator_action": action,
            "timestamp": observation.timestamp,
        }

        # Execute protocol coordination (communication + action)
        if self.protocol is not None:
            messages, actions = self.protocol.coordinate(
                coordinator_state=observation.local,
                subordinate_states=subordinate_states,
                coordinator_action=action,
                context=context
            )
        else:
            # No protocol - simple action passthrough
            messages = {}
            actions = self._simple_action_distribution(action, subordinate_obs)

        # Route to appropriate execution path
        self._apply_coordination(messages, actions, subordinate_obs)

    def _simple_action_distribution(
        self,
        action: Any,
        subordinate_obs: DictType[AgentID, Observation]
    ) -> DictType[AgentID, Any]:
        """Simple action distribution without protocol. [Both Modes]

        Distributes action to subordinates by splitting based on dimensions.

        Args:
            action: Joint action to distribute
            subordinate_obs: Subordinate observations

        Returns:
            Dict mapping subordinate IDs to individual actions
        """
        actions = {}

        if isinstance(action, dict):
            # Already per-agent dict
            return action

        # Assume flat array, split by agent action dimensions
        action_arr = np.asarray(action)
        offset = 0
        for agent_id, agent in self.subordinate_agents.items():
            total_dim = agent.action.dim_c + agent.action.dim_d
            if total_dim > 0:
                actions[agent_id] = action_arr[offset:offset + total_dim]
                offset += total_dim

        return actions

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
            for agent_id, message in messages.items():
                if agent_id in self.subordinate_agents:
                    self.send_message(message, recipient_id=agent_id)

        # Apply actions directly to subordinates
        for agent_id, action in actions.items():
            if agent_id in self.subordinate_agents and action is not None:
                obs = subordinate_obs.get(agent_id)
                if obs:
                    self.subordinate_agents[agent_id].act(obs, upstream_action=action)

    # ============================================
    # Space Construction Methods (Both Modes)
    # ============================================

    def get_subordinate_action_spaces(self) -> DictType[str, gym.Space]:
        """Get action spaces for all subordinate agents. [Both Modes]

        Returns:
            Dictionary mapping agent IDs to their action spaces
        """
        return {
            agent.agent_id: agent.action_space
            for agent in self.subordinate_agents.values()
        }

    def get_joint_action_space(self) -> gym.Space:
        """Construct combined action space for all subordinate agents. [Both Modes]

        Returns:
            Gymnasium space representing joint action space
        """
        low, high, discrete_n = [], [], []
        for sp in self.get_subordinate_action_spaces().values():
            if isinstance(sp, Box):
                low = np.append(low, sp.low)
                high = np.append(high, sp.high)
            elif isinstance(sp, Dict):
                # Handle Dict spaces (e.g., continuous + discrete)
                if 'continuous' in sp.spaces or 'c' in sp.spaces:
                    cont_space = sp.spaces.get('continuous', sp.spaces.get('c'))
                    low = np.append(low, cont_space.low)
                    high = np.append(high, cont_space.high)
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

    def __repr__(self) -> str:
        num_subs = len(self.subordinate_agents)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"CoordinatorAgent(id={self.agent_id}, subordinates={num_subs}, protocol={protocol_name})"

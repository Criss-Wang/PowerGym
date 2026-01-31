"""Field-level agents (L1) for the HERON framework.

FieldAgent represents the lowest level in the agent hierarchy, typically
managing individual units, sensors, or actuators.

In synchronous mode (Option A - Training), field agents:
1. Receive observations via observe()
2. Execute actions via act() - either from coordinator or own policy
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass

import numpy as np
from gymnasium.spaces import Box, Space

from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.core.state import FieldAgentState
from heron.core.action import Action
from heron.core.policies import Policy
from heron.utils.typing import AgentID

FIELD_LEVEL = 1  # Level identifier for field-level agents


@dataclass
class FieldConfig:
    """Configuration for FieldAgent initialization."""
    name: str
    state_config: Dict[str, Any]
    discrete_action: bool
    discrete_action_cats: int  # Number of categories for discrete action if applicable


class FieldAgent(Agent):
    """Base class for field-level (L1) agents.

    FieldAgent manages individual units and provides:
    - Local state management via FieldAgentState
    - Action handling via Action
    - Policy-based decision making
    - Protocol-based coordination

    FieldAgent only observes its local state. Global information should
    be provided by parent CoordinatorAgent through coordination protocols.

    Attributes:
        state: Agent's local state (FieldAgentState)
        action: Agent's action container
        policy: Decision-making policy (learned or rule-based)
        protocol: Communication/action protocol
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        protocol: Optional["Protocol"] = None,
        policy: Optional[Policy] = None,

        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # FieldAgent specific params
        config: Optional[Dict[str, Any]] = None,

        # timing params (for event-driven scheduling)
        tick_interval: float = 1.0,  # Field agents tick most frequently
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,
    ):
        """Initialize field agent.

        Args:
            agent_id: Agent ID (defaults to name from config)
            policy: Decision policy (optional)
            protocol: Communication protocol (optional)
            upstream_id: Optional upstream agent ID for hierarchy structure
            env_id: Optional environment ID for multi-environment isolation
            config: Agent configuration dict
            tick_interval: Time between agent ticks (default 1s for field agents)
            obs_delay: Observation delay
            act_delay: Action delay
            msg_delay: Message delay
        """
        config = config or {}

        self.field_config = FieldConfig(
            name=config.get("name", "field_agent"),
            state_config=config.get("state_config", {}),
            discrete_action=config.get("discrete_action", False),
            discrete_action_cats=config.get("discrete_action_cats", 2),
        )

        self.state: FieldAgentState = FieldAgentState(
            owner_id=agent_id or self.field_config.name,
            owner_level=FIELD_LEVEL
        )
        self.action: Action = Action()
        self.protocol = protocol
        self.policy: Optional[Policy] = policy

        super().__init__(
            agent_id=agent_id or self.field_config.name,
            level=FIELD_LEVEL,
            upstream_id=upstream_id,
            env_id=env_id,
            subordinates={},  # Field agents have no subordinates
            tick_interval=tick_interval,
            obs_delay=obs_delay,
            act_delay=act_delay,
            msg_delay=msg_delay,
        )

        self._init_action()
        self._init_state()

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    # ============================================
    # Initialization Methods (Both Modes)
    # ============================================

    def _init_action(self) -> None:
        """Initialize agent-specific action space. [Both Modes]"""
        self.set_action()

    def _init_state(self) -> None:
        """Initialize agent-specific state attributes. [Both Modes]"""
        self.set_state()

    def set_action(self) -> None:
        """Define/initialize the agent-specific action. [Both Modes]

        To be overridden by subclasses.
        """
        pass

    def set_state(self) -> None:
        """Define/initialize agent-specific state. [Both Modes]

        To be overridden by subclasses.
        """
        pass

    # ============================================
    # Space Getter Methods (Both Modes)
    # ============================================

    def _get_action_space(self) -> Space:
        """Get action space based on agent action. [Both Modes]

        Returns:
            Gymnasium Space object
        """
        return self.action.space

    def _get_observation_space(self) -> Space:
        """Get observation space based on agent state. [Both Modes]

        Returns:
            Gymnasium Space object
        """
        if hasattr(self, "observation_space") and self.observation_space is not None:
            return self.observation_space
        else:
            sample_obs = self._get_obs()
            if len(sample_obs.shape) == 1:  # Vector observation
                return Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=sample_obs.shape,
                    dtype=np.float32,
                )
            else:
                raise NotImplementedError(
                    "Extend _get_observation_space to handle images, discrete, "
                    "or structured observations."
                )

    def _get_obs(self) -> np.ndarray:
        """Build the current observation vector for this agent. [Both Modes]"""
        obs_vec = self.state.vector()

        # Observe other agents' states via a communication protocol
        if (self.protocol is not None and
            hasattr(self.protocol, 'communication_protocol') and
            self.protocol.communication_protocol and
            hasattr(self.protocol.communication_protocol, 'neighbors')):
            for other_agent in self.protocol.communication_protocol.neighbors:
                other_obs_dict = other_agent.state.observed_by(self.agent_id, self.level)
                other_obs = np.concatenate(
                    list(other_obs_dict.values()), dtype=np.float32
                )
                obs_vec = np.concatenate([obs_vec, other_obs], dtype=np.float32)

        if obs_vec.size == 0:
            raise ValueError("No observations available for the agent.")

        return obs_vec

    # ============================================
    # Core Agent Lifecycle Methods (Both Modes)
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent. [Both Modes]

        Args:
            seed: Random seed
            **kwargs: Additional reset params
        """
        super().reset(seed=seed)
        self.reset_agent(**kwargs)

    # ============================================
    # Synchronous Execution (Option A - Training)
    # ============================================

    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
        """Extract observation from global state. [Both Modes]

        - Training (Option A): Called by coordinator to collect observations
        - Testing (Option B): Called internally by tick()

        Args:
            global_state: Complete environment state (optional)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Structured observation for this agent
        """
        return Observation(
            timestamp=self._timestep,
            local={
                'state': self.state.vector(),
                'observation': self._get_obs()
            }
        )

    def act(self, observation: Observation, upstream_action: Any = None) -> None:
        """Compute and apply action. [Training Only - Direct Call]

        Note: In Testing (Option B), tick() computes actions directly via
        _handle_coordinator_action()/_handle_local_action() instead
        of calling this method.

        Routes to coordinator-directed or self-directed action computation based on
        whether upstream_action is provided.

        Args:
            observation: Structured observation
            upstream_action: Optional action from coordinator (coordinator-directed)
        """
        if upstream_action is not None:
            # Coordinator-directed: use the action provided by upstream coordinator
            action = self._handle_coordinator_action(upstream_action, observation)
        else:
            # Self-directed: compute action using local policy
            action = self._handle_local_action(observation)

        self.action.set_values(action)

    # ============================================
    # Action Handling (Both Modes)
    # ============================================

    def _handle_coordinator_action(
        self,
        upstream_action: Any,
        observation: Observation
    ) -> Any:
        """Handle coordinator-directed action. [Both Modes]

        Args:
            upstream_action: Action assigned by coordinator
            observation: Current observation (unused in coordinator-directed mode)

        Returns:
            Action to execute (passthrough of upstream_action)
        """
        return upstream_action

    def _handle_local_action(self, observation: Observation) -> Any:
        """Handle self-directed action computation using local policy. [Both Modes]

        Args:
            observation: Current observation including messages

        Returns:
            Action computed by local policy

        Raises:
            ValueError: If no policy is defined for self-directed mode
        """
        if self.policy is None:
            raise ValueError(
                "No policy defined for FieldAgent. "
                "Agent requires either upstream_action (coordinator-directed) or policy (self-directed)."
            )

        action = self.policy.forward(observation)
        return action

    # ============================================
    # Event-Driven Execution (Option B - Testing)
    # ============================================

    def tick(
        self,
        scheduler: "EventScheduler",
        current_time: float,
        global_state: Optional[Dict[str, Any]] = None,
        proxy: Optional["Agent"] = None,
    ) -> None:
        """Execute one tick in event-driven mode. [Testing Only]

        In Option B, FieldAgent:
        1. Updates timestep
        2. Checks message broker for upstream action from coordinator
        3. Gets observation (potentially delayed via ProxyAgent)
        4. Computes action using upstream action or own policy
        5. Schedules ACTION_EFFECT with act_delay

        Args:
            scheduler: EventScheduler for scheduling future events
            current_time: Current simulation time
            global_state: Optional global state for observation
            proxy: Optional ProxyAgent for delayed observations
        """
        self._timestep = current_time

        # Check message broker for upstream action from coordinator
        upstream_action = None
        actions = self.receive_action_messages()
        if actions:
            upstream_action = actions[-1]  # Use most recent action

        # Get observation (with delay if proxy provided and obs_delay > 0)
        if proxy is not None and self.obs_delay > 0:
            # Use proxy for delayed observations
            delayed_time = current_time - self.obs_delay
            proxy_state = self.request_state_from_proxy(proxy, at_time=delayed_time)
            observation = self._build_observation_from_proxy(proxy_state)
        else:
            # Direct observation (no delay)
            observation = self.observe(global_state)
        self._last_observation = observation

        # Compute action (coordinator-directed if upstream provided, else self-directed)
        if upstream_action is not None:
            action = self._handle_coordinator_action(upstream_action, observation)
        elif self.policy is not None:
            action = self._handle_local_action(observation)
        else:
            # No action to take - agent is passive
            return

        self.action.set_values(action)

        # Schedule delayed action effect
        if self.act_delay > 0:
            scheduler.schedule_action_effect(
                agent_id=self.agent_id,
                action=self.action.vector(),
                delay=self.act_delay,
            )

    def _build_observation_from_proxy(
        self, proxy_state: Dict[str, Any]
    ) -> Observation:
        """Build observation from proxy state. [Testing Only]

        Override in subclasses for custom observation building from proxy state.

        Args:
            proxy_state: Filtered state dict from ProxyAgent

        Returns:
            Observation built from proxy state
        """
        return Observation(
            timestamp=self._timestep,
            local={
                'state': self.state.vector(),
                'proxy_state': proxy_state,
            }
        )

    # ============================================
    # Agent-Specific Methods (Both Modes)
    # ============================================

    def reset_agent(self, *args, **kwargs) -> None:
        """Reset agent to initial state. [Both Modes]

        To be implemented by subclasses.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        pass

    def update_state(self, *args, **kwargs) -> None:
        """Update agent state. [Both Modes]

        To be implemented by subclasses.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        pass

    def feasible_action(self) -> None:
        """Clamp/adjust current action to ensure feasibility. [Both Modes]

        Optional hook that can be overridden by subclasses.
        """
        return None

    # ============================================
    # Utility Methods (Both Modes)
    # ============================================

    def __repr__(self) -> str:
        return f"FieldAgent(id={self.agent_id}, policy={self.policy}, protocol={self.protocol})"

"""Field-level agents (L1) for the HERON framework.

FieldAgent represents the lowest level in the agent hierarchy, typically
managing individual units, sensors, or actuators.
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

        # communication params
        message_broker: Optional["MessageBroker"] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # FieldAgent specific params
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize field agent.

        Args:
            agent_id: Agent ID (defaults to name from config)
            policy: Decision policy (optional)
            protocol: Communication protocol (optional)
            message_broker: Optional message broker for hierarchical execution
            upstream_id: Optional upstream agent ID for hierarchical execution
            env_id: Optional environment ID for multi-environment isolation
            config: Agent configuration dict
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
        self.cost: float = 0.0
        self.safety: float = 0.0
        self.protocol = protocol
        self.policy: Optional[Policy] = policy

        super().__init__(
            agent_id=agent_id or self.field_config.name,
            level=FIELD_LEVEL,
            message_broker=message_broker,
            upstream_id=upstream_id,
            env_id=env_id,
            subordinates={}  # Field agents have no subordinates
        )

        self._init_action()
        self._init_state()

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    # ============================================
    # Initialization Methods
    # ============================================

    def _init_action(self) -> None:
        """Initialize agent-specific action space."""
        self.set_action()

    def _init_state(self) -> None:
        """Initialize agent-specific state attributes."""
        self.set_state()

    def set_action(self) -> None:
        """Define/initialize the agent-specific action.

        To be overridden by subclasses.
        """
        pass

    def set_state(self) -> None:
        """Define/initialize agent-specific state.

        To be overridden by subclasses.
        """
        pass

    # ============================================
    # Space Getter Methods
    # ============================================

    def _get_action_space(self) -> Space:
        """Get action space based on agent action.

        Returns:
            Gymnasium Space object
        """
        return self.action.space

    def _get_observation_space(self) -> Space:
        """Get observation space based on agent state.

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
        """Build the current observation vector for this agent."""
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
    # Core Agent Lifecycle Methods
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent.

        Args:
            seed: Random seed
            **kwargs: Additional reset params
        """
        super().reset(seed=seed)
        self.reset_agent(**kwargs)

    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
        """Extract observation from global state.

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
        """Compute and apply action.

        Routes to centralized or decentralized action computation based on
        whether upstream_action is provided.

        Args:
            observation: Structured observation
            upstream_action: Optional action from coordinator (centralized mode)
        """
        if upstream_action is not None:
            # Centralized mode: Use coordinator's action directly
            action = self._handle_centralized_action(upstream_action, observation)
        else:
            # Decentralized mode: Compute own action using policy
            action = self._handle_decentralized_action(observation)

        self.action.set_values(action)

    # ============================================
    # Centralized Action Handling
    # ============================================

    def _handle_centralized_action(
        self,
        upstream_action: Any,
        observation: Observation
    ) -> Any:
        """Handle centralized action from coordinator.

        Args:
            upstream_action: Action assigned by coordinator
            observation: Current observation (unused in pure centralized mode)

        Returns:
            Action to execute (passthrough of upstream_action)
        """
        return upstream_action

    # ============================================
    # Decentralized Action Handling
    # ============================================

    def _handle_decentralized_action(self, observation: Observation) -> Any:
        """Handle decentralized action computation using local policy.

        Args:
            observation: Current observation including messages

        Returns:
            Action computed by local policy

        Raises:
            ValueError: If no policy is defined for decentralized mode
        """
        if self.policy is None:
            raise ValueError(
                "No policy defined for FieldAgent in decentralized mode. "
                "Agent requires either upstream_action (centralized) or policy (decentralized)."
            )

        action = self.policy.forward(observation)
        return action

    # ============================================
    # Abstract Methods for Hierarchical Execution
    # ============================================

    def _derive_local_action(self, upstream_action: Optional[Any]) -> Optional[Any]:
        """Derive local action from upstream action.

        If a RL policy is defined, use it to compute the local action.
        Otherwise, pass through the upstream action.

        Args:
            upstream_action: Action received from upstream agent

        Returns:
            Local action to execute
        """
        if self.policy is not None:
            observation = self.observe()
            action = self.policy.forward(observation)
        else:
            action = upstream_action

        # Store action internally
        if action is not None:
            self.action.set_values(action)

        return action

    def _execute_local_action(self, action: Optional[Any]) -> None:
        """Execute own action and update internal state.

        Args:
            action: Action to execute
        """
        self.update_state()

    # ============================================
    # Agent-Specific Methods (Abstract)
    # ============================================

    def reset_agent(self, *args, **kwargs) -> None:
        """Reset agent to initial state (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        pass

    def update_state(self, *args, **kwargs) -> None:
        """Update agent state (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        pass

    def update_cost_safety(self, *args, **kwargs) -> None:
        """Update agent cost and safety metrics (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        pass

    def get_reward(self) -> Dict[str, float]:
        """Get reward signal from agent cost/safety.

        Returns:
            Dict with cost and safety values
        """
        return {"cost": self.cost, "safety": self.safety}

    def feasible_action(self) -> None:
        """Clamp/adjust current action to ensure feasibility.

        Optional hook that can be overridden by subclasses.
        """
        return None

    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        return f"FieldAgent(id={self.agent_id}, policy={self.policy}, protocol={self.protocol})"

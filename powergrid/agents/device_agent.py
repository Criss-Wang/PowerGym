"""Device-level agents that wrap Device objects as autonomous agents.

DeviceAgent provides a bridge between the existing Device abstraction and the
new Agent abstraction, enabling devices to participate in multi-agent control.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass

import numpy as np

from gymnasium.spaces import Box
from gymnasium.spaces import Space

from powergrid.agents.base import Agent, Observation
from powergrid.core.state import DeviceState
from powergrid.core.action import Action
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.utils.typing import AgentID
from powergrid.messaging.base import MessageBroker

DEVICE_LEVEL = 1  # Level identifier for device-level agents

@dataclass
class DeviceConfig:
    """Configuration for DeviceAgent initialization."""
    name: str
    device_state_config: Dict[str, Any]
    discrete_action: bool
    discrete_action_cats: int  # Number of categories for discrete action if applicable


class DeviceAgent(Agent):
    """Wraps a Device as an autonomous agent.

    DeviceAgent maintains compatibility with the existing Device interface while
    adding agent capabilities like observation extraction, communication, and
    pluggable policies.

    DeviceAgent only observes its local device state. Global information should
    be provided by parent GridAgent through coordination protocols/messages.

    Attributes:
        device: Underlying Device object (DG, ESS, RES, etc.)
        policy: Decision-making policy (learned or rule-based)
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        protocol: Protocol = NoProtocol(),
        policy: Optional[Policy] = None,

        # communication params
        message_broker: Optional[MessageBroker] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # DeviceAgent specific params
        device_config: Dict[str, Any] = {},
    ):
        """Initialize device agent.

        Args:
            agent_id: Agent ID (defaults to device name from config)
            policy: Decision policy (optional)
            protocol: Communication protocol (defaults to NoProtocol)
            message_broker: Optional message broker for hierarchical execution
            upstream_id: Optional upstream agent ID for hierarchical execution
            env_id: Optional environment ID for multi-environment isolation
            device_config: Device configuration dict
        """
        self.state: DeviceState = DeviceState(agent_id, DEVICE_LEVEL)
        self.action: Action = Action()
        self.cost: float = 0.0
        self.safety: float = 0.0
        self.protocol: Protocol = protocol
        self.policy: Optional[Policy] = policy

        self.config = DeviceConfig(
            name=device_config.get("name", "device_agent"),
            device_state_config=device_config.get("device_state_config", {}),
            discrete_action=device_config.get("discrete_action", False),
            discrete_action_cats=device_config.get("discrete_action_cats", 2),
        )

        super().__init__(
            agent_id=agent_id or self.config.name,
            level=DEVICE_LEVEL,
            message_broker=message_broker,
            upstream_id=upstream_id,
            env_id=env_id,
            subordinates={}  # Device agents have no subordinates
        )

        self._init_device_action()
        self._init_device_state()

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    # ============================================
    # Initialization Methods
    # ============================================

    def _init_device_action(self) -> None:
        """Initialize device-specific action space
        """
        self.set_device_action()

    def _init_device_state(self) -> None:
        """Initialize device-specific state attributes.
        """
        self.set_device_state()

    def set_device_action(self) -> None:
        """Define/initialize the device-specific action.

        To be overridden by subclasses.
        """
        pass

    def set_device_state(self) -> None:
        """Define/initialize device-specific state.

        To be overridden by subclasses.
        """
        pass

    # ============================================
    # Space Getter Methods
    # ============================================

    def _get_action_space(self) -> Space:
        """ Get action space based on device action.
        Returns:
            - action_space: Gymnasium Space object
        """
        return self.action.space

    def _get_observation_space(self) -> Space:
        """ Get observation space based on device state.
        Returns:
            - observation_space: Gymnasium Space object
        """

        if hasattr(self, "observation_space") and self.observation_space is not None:
            return self.observation_space
        else:
            sample_obs = self._get_obs()
            if len(sample_obs.shape) == 1: # Vector observation
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
        if self.protocol.communication_protocol:
            for other_device in self.protocol.communication_protocol.neighbors:
                other_device_obs_dict = other_device.state.observed_by(self.agent_id, self.level)
                other_device_obs = np.concatenate(
                    list(other_device_obs_dict.values()), dtype=np.float32
                )
                obs_vec = np.concatenate([obs_vec, other_device_obs], dtype=np.float32)

        if obs_vec.size == 0:
            raise ValueError("No observations available for the agent.")

        return obs_vec

    # ============================================
    # Core Agent Lifecycle Methods
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent and underlying device.

        Args:
            seed: Random seed
            **kwargs: Additional reset params (e.g., init_soc for ESS)
        """
        super().reset()
        self.reset_device(**kwargs)

    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
        """Extract device observation from global state.

        Args:
            global_state: Complete environment state (optional)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Structured observation for this device
        """
        # TODO: aggregate global info if needed
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
    # Centralized Action Handling (Direct control)
    # ============================================

    def _handle_centralized_action(
        self,
        upstream_action: Any,
        observation: Observation
    ) -> Any:
        """Handle centralized action from coordinator.

        In centralized mode, the coordinator directly assigns actions (setpoints).
        The device follows these commands without using its own policy.

        Args:
            upstream_action: Action assigned by coordinator (e.g., power setpoint)
            observation: Current observation (unused in pure centralized mode)

        Returns:
            Action to execute (passthrough of upstream_action)
        """
        # TODO: Validate upstream_action against device constraints
        return upstream_action

    # ============================================
    # Decentralized Action Handling (Autonomous decision)
    # ============================================

    def _handle_decentralized_action(self, observation: Observation) -> Any:
        """Handle decentralized action computation using local policy.

        In decentralized mode, devices make their own decisions based on:
        - Local observations (device state)
        - Coordination signals from messages (e.g., prices)
        - Local policy (learned or rule-based)

        Args:
            observation: Current observation including messages

        Returns:
            Action computed by local policy

        Raises:
            ValueError: If no policy is defined for decentralized mode
        """
        if self.policy is None:
            raise ValueError(
                "No policy defined for DeviceAgent in decentralized mode. "
                "Device requires either upstream_action (centralized) or policy (decentralized)."
            )

        # TODO: Extract coordination signals from observation.messages
        # and incorporate them into policy input (e.g., add price to observation)

        action = self.policy.forward(observation)
        return action

    # ============================================
    # Abstract Methods for Hierarchical Execution
    # ============================================
    def _derive_local_action(self, upstream_action: Optional[Any]) -> Optional[Any]:
        """Derive local action from upstream action.

        If a RL policy is defined, use it to compute the local action based on observation.
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

        Subclasses should override this to implement their action execution.
        State updates should be published via _publish_state_updates().

        Args:
            action: Action to execute
        """
        self.update_state()

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
    # Device-Specific Methods (Abstract)
    # ============================================

    def reset_device(self, *args, **kwargs) -> None:
        """Reset device to initial state (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError

    def update_state(self, *args, **kwargs) -> None:
        """Update device state (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError

    def update_cost_safety(self, *args, **kwargs) -> None:
        """Update device cost and safety metrics (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError

    def get_reward(self) -> float:
        """Get reward signal from device cost/safety.

        Returns:
            Reward value (negative cost minus safety penalty)
        """
        return {"cost": self.cost, "safety": self.safety}

    def feasible_action(self) -> None:
        """Clamp/adjust current action to ensure feasibility.

        This is an optional hook that can be overridden by subclasses to
        enforce device-specific constraints on actions.
        """
        return None

    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        """Return string representation of the agent.

        Returns:
            String representation
        """
        return f"DeviceAgent(id={self.agent_id}, policy={self.policy}, protocol={self.protocol})"

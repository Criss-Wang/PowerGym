"""Device-level agents that wrap Device objects as autonomous agents.

DeviceAgent provides a bridge between the existing Device abstraction and the
new Agent abstraction, enabling devices to participate in multi-agent control.
"""

from typing import Any, Dict, Optional

from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from powergrid.agents.base import Agent, Observation
from powergrid.core.action import Action
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.core.state import DeviceState
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
        self.state: DeviceState = DeviceState()
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
        
        self._init_action_space()
        self._init_device_state()
        self._init_observation_space()

        super().__init__(
            agent_id=agent_id or self.config.name,
            level=DEVICE_LEVEL,
            action_space=self._get_action_space(),
            observation_space=self._get_observation_space(),
            message_broker=message_broker,
            upstream_id=upstream_id,
            env_id=env_id,
            subordinates={}  # Device agents have no subordinates
        )

    # ============================================
    # Initialization Methods
    # ============================================

    def _init_action_space(self) -> None:
        """Define action space based on underlying device action.

        This method should be overridden by subclasses to define device-specific action spaces.
        """
        pass

    def _init_device_state(self) -> None:
        """Initialize device-specific state attributes.

        This method can be overridden by subclasses to initialize device-specific state.
        """
        pass

    def _init_observation_space(self) -> None:
        """Define observation space based on device state.

        This method can be overridden by subclasses to define device-specific observation spaces.
        """
        pass

    # ============================================
    # Space Construction Methods
    # ============================================

    def _get_action_space(self) -> gym.Space:
        """Construct Gymnasium action space from device action configuration.

        Returns:
            Gymnasium space for device actions

        Raises:
            ValueError: If action configuration is invalid
        """
        action = self.action

        # Continuous actions
        if action.dim_c > 0:
            if action.range is None:
                raise ValueError("Device action.range must be set for continuous actions.")
            low, high = action.range
            if self.config.discrete_action:
                cats = self.config.discrete_action_cats
                if low.size == 1:
                    return Discrete(cats)
                else:
                    return MultiDiscrete([cats] * low.size)
            return Box(
                low=low,
                high=high,
                dtype=np.float32,
            )

        # Discrete actions
        if action.dim_d > 0:
            if not action.ncats:
                raise ValueError("Device action.ncats must be set to a positive integer for discrete actions.")

            return Discrete(action.ncats)

        raise ValueError("Device must have either continuous or discrete actions defined.")

    def _get_observation_space(self) -> gym.Space:
        """Construct Gymnasium observation space from device state.

        Returns:
            Gymnasium space for device observations
        """
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=self.state.vector().shape,
            dtype=np.float32
        )

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
        obs = Observation(
            timestamp=self._timestep,
        )

        # Local device state only
        obs.local['state'] = self.state.vector().astype(np.float32)

        # TODO: aggregate global info if needed
        obs.global_info = global_state

        return obs

    def act(self, observation: Observation, given_action: Any = None) -> None:
        """Compute action using policy.

        Args:
            observation: Structured observation
            given_action: Action provided by the parent grid agent

        Returns:
            Action in format defined by action_space
        """
        if given_action:
            action = given_action
        elif self.policy is not None:
            action = self.policy.forward(observation)
        else:
            raise ValueError("No action provided and no policy defined for DeviceAgent.")

        self._set_device_action(action)
        # TODO: Add communication logic (send/receive message) if needed

    
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
        if self.policy is not None:
            obs = self.observe()
            action = self.policy.forward(obs)
        else:
            action = upstream_action

        # Store action internally
        if action is not None:
            self._set_device_action(action)

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
    def _set_device_action(self, action: Any) -> None:
        """Set action on underlying device.

        Args:
            action: Action from policy (numpy array)
        """
        # TODO: verify action format matches policy forward output
        assert action.size == self.action.dim_c + self.action.dim_d
        self.action.c[:] = action[:self.action.c.size]
        if self.config.discrete_action:
            cats = self.config.discrete_action_cats
            low, high = self.action.range
            acts = np.linspace(low, high, cats).transpose()
            self.action.c[:] = [a[action[i]] for i, a in enumerate(acts)]
        self.action.d[:] = action[self.action.c.size:]
    
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

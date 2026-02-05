"""Coordinator-level agents (L2) for the HERON framework.

CoordinatorAgent manages a set of field agents, implementing coordination
protocols like price signals, setpoints, or consensus algorithms. It aggregates
observations from subordinates and distributes coordinated actions.

Supports two execution modes:

**Synchronous Mode (Option A - Training)**:
    - Collect observations from subordinates via observe()
    - Compute joint action using centralized policy
    - Distribute actions via coordinate_subordinates()
    - Suitable for centralized RL training

**Event-Driven Mode (Option B - Testing)**:
    - Execute via tick() with EventScheduler integration
    - Receive upstream actions from SystemAgent via message broker
    - Schedule MESSAGE_DELIVERY events to subordinates with msg_delay
    - Models realistic async communication patterns
"""

from typing import Any, Dict as DictType, List, Optional

from heron.agents.hierarchical_agent import HierarchicalAgent
from heron.agents.field_agent import FieldAgent
from heron.core.observation import (
    OBS_KEY_SUBORDINATE_OBS,
    OBS_KEY_COORDINATOR_STATE,
)
from heron.core.state import CoordinatorAgentState
from heron.core.policies import Policy
from heron.utils.typing import AgentID
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig


COORDINATOR_LEVEL = 2  # Level identifier for coordinator-level agents


class CoordinatorAgent(HierarchicalAgent):
    """Coordinator-level agent for managing field agents.

    CoordinatorAgent coordinates multiple field agents using specified protocols
    and optionally a centralized policy for joint decision-making.

    Supports two execution modes:

    **Synchronous Mode (Option A - Training)**:
        Call observe() to aggregate subordinate observations, then act() or
        coordinate_subordinates() to distribute actions. Suitable for centralized
        RL training where all agents step synchronously.

    **Event-Driven Mode (Option B - Testing)**:
        Call tick() with an EventScheduler. Schedules MESSAGE_DELIVERY events to
        subordinates with configurable msg_delay. Models realistic async communication
        where subordinates process messages on their own tick schedule.

    Attributes:
        subordinates: Dictionary mapping agent IDs to FieldAgent instances
        state: Coordinator's aggregated state (CoordinatorAgentState)
        protocol: Coordination protocol for managing subordinate agents
        policy: Optional centralized policy for joint action computation

    Example:
        Create a coordinator managing field agents::

            from heron.agents import CoordinatorAgent
            from heron.protocols import SetpointProtocol

            # Create coordinator with setpoint-based control
            coordinator = CoordinatorAgent(
                agent_id="grid_coordinator",
                protocol=SetpointProtocol(),
            )

            # Subordinates can be added via config or directly
            # coordinator.subordinates = {a.agent_id: a for a in agents}

        Collect observations and distribute actions::

            # Coordinator aggregates subordinate observations
            obs = coordinator.observe(global_state={"time": 0})

            # Distribute joint action to subordinates
            import numpy as np
            joint_action = np.array([0.5, 0.3, 0.2])
            coordinator.act(obs, upstream_action=joint_action)

        Get joint action space for RL training::

            joint_space = coordinator.get_joint_action_space()
            # Returns Box, MultiDiscrete, or Dict space
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,

        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # CoordinatorAgent specific params
        config: Optional[DictType[str, Any]] = None,

        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
    ):
        """Initialize coordinator agent.

        Args:
            agent_id: Unique identifier
            policy: Optional centralized policy for joint action computation
            protocol: Protocol for coordinating subordinate agents
            upstream_id: Optional parent agent ID for hierarchy structure
            env_id: Optional environment ID for multi-environment isolation
            config: Coordinator configuration dictionary
            tick_config: Timing configuration. Defaults to TickConfig with
                tick_interval=60.0 (coordinators tick less frequently than field agents).
                Use TickConfig.deterministic() or TickConfig.with_jitter().
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
        subordinates = self._build_subordinates(
            agent_configs,
            env_id=env_id,
            upstream_id=agent_id  # Subordinates' upstream is this coordinator
        )

        super().__init__(
            agent_id=agent_id,
            level=COORDINATOR_LEVEL,
            subordinates=subordinates,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config or TickConfig.deterministic(tick_interval=60.0),
        )

    # ============================================
    # Abstract Method Implementations
    # ============================================

    def _build_subordinates(
        self,
        configs: List[DictType[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, FieldAgent]:
        """Build subordinate agents from configuration.

        Override this in domain-specific subclasses to create appropriate
        agent types based on configuration.

        Args:
            configs: List of agent configuration dictionaries
            env_id: Environment ID for multi-environment isolation
            upstream_id: Upstream agent ID (this coordinator)

        Returns:
            Dictionary mapping agent IDs to FieldAgent instances
        """
        # Default implementation - override in subclasses
        return {}

    def _get_subordinate_obs_key(self) -> str:
        """Get the observation key for subordinate observations."""
        return OBS_KEY_SUBORDINATE_OBS

    def _get_state_obs_key(self) -> str:
        """Get the observation key for coordinator state."""
        return OBS_KEY_COORDINATOR_STATE

    def _get_subordinate_action_dim(self, subordinate: FieldAgent) -> int:
        """Get the action dimension for a subordinate field agent.

        Args:
            subordinate: FieldAgent instance

        Returns:
            Total action dimension (continuous + discrete)
        """
        return subordinate.action.dim_c + subordinate.action.dim_d

    def _get_env_state_key(self) -> str:
        """Get the key for coordinator state in env_state dict."""
        return 'coordinator'

    def _get_env_subordinate_key(self) -> str:
        """Get the key for subordinate updates in env_state dict."""
        return 'subordinates'


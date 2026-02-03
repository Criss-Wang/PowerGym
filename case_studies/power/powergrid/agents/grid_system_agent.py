"""Grid-level system agent for power system coordination.

GridSystemAgent manages multiple grid coordinators (PowerGridAgent instances)
and provides system-wide coordination like frequency regulation, inter-area
power flow control, and market clearing.

This is the top level (L3) in the HERON hierarchy for power grid applications:
- SystemAgent (L3): GridSystemAgent - Manages multiple grids
- CoordinatorAgent (L2): PowerGridAgent - Manages devices within a grid
- FieldAgent (L1): DeviceAgent (Generator, ESS, etc.) - Individual devices
"""

from typing import Any, Dict as DictType, List, Optional

import numpy as np
from gymnasium.spaces import Box

from heron.agents.system_agent import SystemAgent, SYSTEM_LEVEL
from heron.core.observation import Observation
from heron.core.policies import Policy
from heron.protocols.base import Protocol, NoProtocol
from heron.utils.typing import AgentID
from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.core.state.state import GridSystemState


class GridSystemAgent(SystemAgent):
    """System-level agent for multi-area power grid coordination.

    GridSystemAgent coordinates multiple PowerGridAgent instances,
    handling:
    - Inter-area power flow constraints
    - System frequency regulation
    - Aggregate generation/demand balancing
    - Market clearing (if applicable)

    Attributes:
        grids: Dictionary mapping grid IDs to PowerGridAgent instances
               (alias for coordinators)
        total_generation: Aggregate generation across all grids (MW)
        total_load: Aggregate load across all grids (MW)
        frequency: System frequency (Hz)

    Example:
        Create a system agent managing multiple grids::

            from powergrid.agents import GridSystemAgent, PowerGridAgent
            from heron.protocols import SetpointProtocol

            # Create grid agents
            grid1 = PowerGridAgent(agent_id="grid_1", net=net1)
            grid2 = PowerGridAgent(agent_id="grid_2", net=net2)

            # Create system agent
            system = GridSystemAgent(
                agent_id="grid_system",
                protocol=SetpointProtocol(),
                grids=[grid1, grid2]
            )

            # Use in training loop
            obs = system.observe(global_state)
            system.act(obs, upstream_action=joint_action)
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        protocol: Protocol = NoProtocol(),
        policy: Optional[Policy] = None,

        # hierarchy params
        env_id: Optional[str] = None,

        # GridSystemAgent specific params
        grids: Optional[List[PowerGridAgent]] = None,
        system_config: Optional[DictType[str, Any]] = None,

        # timing params
        tick_interval: float = 300.0,
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,
    ):
        """Initialize grid system agent.

        Args:
            agent_id: Unique identifier
            protocol: System-level coordination protocol
            policy: Optional system-level policy
            env_id: Environment ID
            grids: List of PowerGridAgent instances to manage
            system_config: System configuration dictionary
            tick_interval: Time between agent ticks (default 300s)
            obs_delay: Observation delay
            act_delay: Action delay
            msg_delay: Message delay
        """
        system_config = system_config or {}
        self._system_config = system_config

        # Build config for SystemAgent
        if grids is not None:
            self._init_grids = grids
            config = {'coordinators': []}
        else:
            self._init_grids = None
            config = {'coordinators': system_config.get('grids', [])}

        super().__init__(
            agent_id=agent_id or "grid_system",
            protocol=protocol,
            policy=policy,
            env_id=env_id,
            config=config,
            tick_interval=tick_interval,
            obs_delay=obs_delay,
            act_delay=act_delay,
            msg_delay=msg_delay,
        )

        # Use GridSystemState for power-grid domain
        self.state = GridSystemState(
            owner_id=self.agent_id,
            owner_level=SYSTEM_LEVEL
        )

        # If grids were provided directly, set them up now
        if self._init_grids is not None:
            self.coordinators = {
                grid.agent_id: grid for grid in self._init_grids
            }
            for grid in self._init_grids:
                grid.upstream_id = self.agent_id

        # System-wide metrics
        self.total_generation: float = 0.0
        self.total_load: float = 0.0
        self.frequency: float = 60.0  # Hz (nominal)

        # Re-initialize state with domain-specific features
        self._init_state()

    # ============================================
    # Backward Compatibility: grids <-> coordinators
    # ============================================

    @property
    def grids(self) -> DictType[AgentID, PowerGridAgent]:
        """Alias for coordinators for power-grid domain."""
        return self.coordinators

    @grids.setter
    def grids(self, value: DictType[AgentID, PowerGridAgent]) -> None:
        self.coordinators = value

    # ============================================
    # Extension Hooks
    # ============================================

    def set_state(self) -> None:
        """Initialize system-level state features."""
        from powergrid.core.features.system import (
            SystemFrequency,
            AggregateGeneration,
            AggregateLoad,
        )

        self.state.features = [
            SystemFrequency(frequency_hz=60.0, nominal_hz=60.0),
            AggregateGeneration(total_mw=0.0),
            AggregateLoad(total_mw=0.0),
        ]

    def set_action(self) -> None:
        """Initialize system-level action space.

        System actions might include:
        - Frequency regulation signal
        - Inter-area power flow targets
        - Emergency load shedding commands

        Override in subclasses for specific action spaces.
        """
        # Default: no system-level action (coordinators act independently)
        pass

    def reset_system(self, **kwargs) -> None:
        """Reset system to initial state."""
        self.total_generation = 0.0
        self.total_load = 0.0
        self.frequency = 60.0

        # Reset state features
        if hasattr(self, 'state') and self.state:
            self.state.reset()

    # ============================================
    # Observation Methods
    # ============================================

    def _build_system_observation(
        self,
        coordinator_obs: DictType[AgentID, Observation],
        *args,
        **kwargs
    ) -> Any:
        """Build system observation including aggregate metrics.

        Args:
            coordinator_obs: Dictionary mapping grid IDs to their observations

        Returns:
            System observation dictionary
        """
        # Aggregate generation and load from all grids
        total_gen = 0.0
        total_load = 0.0

        for coord_id, coord in self.coordinators.items():
            if hasattr(coord, 'cost'):
                # PowerGridAgent aggregates device costs
                pass

        return {
            "coordinator_obs": coordinator_obs,
            "grid_obs": coordinator_obs,  # Alias for backward compat
            "system_state": self.state.vector() if self.state else None,
            "total_generation": self.total_generation,
            "total_load": self.total_load,
            "frequency": self.frequency,
        }

    # ============================================
    # Environment Interface
    # ============================================

    def update_from_environment(self, env_state: DictType[str, Any]) -> None:
        """Update system state from environment.

        Args:
            env_state: Environment state containing system metrics and grid states
        """
        super().update_from_environment(env_state)

        if not env_state:
            return

        # Extract system-wide metrics
        self.frequency = env_state.get('frequency', self.frequency)
        self.total_generation = env_state.get('total_generation', 0.0)
        self.total_load = env_state.get('total_load', 0.0)

        # Update state features
        if hasattr(self.state, 'update_feature'):
            self.state.update_feature('SystemFrequency', frequency_hz=self.frequency)
            self.state.update_feature('AggregateGeneration', total_mw=self.total_generation)
            self.state.update_feature('AggregateLoad', total_mw=self.total_load)

    def get_state_for_environment(self) -> DictType[str, Any]:
        """Get system and grid states for environment.

        Returns:
            Dictionary with system metrics and grid states
        """
        result = super().get_state_for_environment()

        # Add system metrics
        result['system_metrics'] = {
            'frequency': self.frequency,
            'total_generation': self.total_generation,
            'total_load': self.total_load,
        }

        return result

    # ============================================
    # Cost/Safety Aggregation
    # ============================================

    @property
    def cost(self) -> float:
        """Aggregate cost from all grids."""
        return sum(
            grid.cost for grid in self.grids.values()
            if hasattr(grid, 'cost')
        )

    @property
    def safety(self) -> float:
        """Aggregate safety penalty from all grids."""
        return sum(
            grid.safety for grid in self.grids.values()
            if hasattr(grid, 'safety')
        )

    def get_reward(self) -> DictType[str, float]:
        """Get system-wide reward.

        Returns:
            Dict with cost, safety, and total reward
        """
        return {
            "cost": self.cost,
            "safety": self.safety,
            "total": -(self.cost + self.safety),
        }

    def update_cost_safety(self, net=None) -> None:
        """Update cost and safety metrics from all grids.

        Args:
            net: Optional network object (passed to grids)
        """
        for grid in self.grids.values():
            if hasattr(grid, 'update_cost_safety'):
                grid.update_cost_safety(net)

    # ============================================
    # Space Construction
    # ============================================

    def get_system_observation_space(self) -> Box:
        """Get observation space for the system.

        Returns:
            Box space for system observations
        """
        # Aggregate from all grids
        grid_obs_size = 0
        for grid in self.grids.values():
            if hasattr(grid, 'get_grid_observation_space'):
                space = grid.get_grid_observation_space()
                grid_obs_size += space.shape[0] if space.shape else 0

        # Add system state size
        system_state_size = len(self.state.vector()) if self.state else 3

        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(grid_obs_size + system_state_size,),
            dtype=np.float32
        )

    def get_system_action_space(self) -> Box:
        """Get action space for the system.

        Aggregates action spaces from all grids.

        Returns:
            Box space for system actions
        """
        return self.get_joint_action_space()

    # ============================================
    # Distributed Mode Configuration
    # ============================================

    def configure_for_distributed(self, message_broker=None) -> None:
        """Configure system and all grids for distributed execution mode.

        Args:
            message_broker: MessageBroker instance for inter-agent communication
        """
        if message_broker is not None:
            self.set_message_broker(message_broker)

        # Create message channel for system agent
        if self._message_broker is not None:
            from heron.messaging.base import ChannelManager
            env_id = self.env_id or "default"
            channel = ChannelManager.result_channel(env_id, self.agent_id)
            self._message_broker.create_channel(channel)

        # Propagate to all grids
        for grid in self.grids.values():
            if hasattr(grid, 'configure_for_distributed'):
                grid.configure_for_distributed(self._message_broker)

    def reset_all_grids(self, **kwargs) -> None:
        """Reset all subordinate grid agents.

        Args:
            **kwargs: Keyword arguments passed to each grid's reset
        """
        for grid in self.grids.values():
            grid.reset(**kwargs)

    # ============================================
    # Utility
    # ============================================

    def __repr__(self) -> str:
        num_grids = len(self.grids)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"GridSystemAgent(id={self.agent_id}, grids={num_grids}, protocol={protocol_name})"

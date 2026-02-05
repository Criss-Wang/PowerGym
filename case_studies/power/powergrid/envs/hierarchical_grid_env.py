"""HierarchicalGridEnv: Multi-agent environment with GridSystemAgent coordination.

This environment extends NetworkedGridEnv to use HERON's 3-level agent hierarchy:
- SystemAgent (L3): GridSystemAgent - Manages multiple grid coordinators
- CoordinatorAgent (L2): PowerGridAgent - Manages devices within a grid
- FieldAgent (L1): DeviceAgent - Individual controllable devices

The environment supports both execution modes:
- Option A (Training): Synchronous step() with CTDE pattern via SystemAgent
- Option B (Testing): Event-driven execution with heterogeneous tick rates
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandapower as pp
from gymnasium.spaces import Box

from heron.protocols.base import Protocol, NoProtocol
from heron.protocols.vertical import SetpointProtocol, SystemProtocol
from heron.scheduling import EventScheduler
from heron.utils.typing import AgentID
from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.agents.grid_system_agent import GridSystemAgent
from powergrid.envs.networked_grid_env import NetworkedGridEnv


class HierarchicalGridEnv(NetworkedGridEnv):
    """Multi-agent environment with GridSystemAgent as top-level coordinator.

    This environment provides a complete HERON hierarchical setup where:
    - GridSystemAgent observes aggregate state and coordinates PowerGridAgents
    - PowerGridAgents coordinate their DeviceAgents
    - Actions flow top-down, observations flow bottom-up

    The environment can be used in two modes:

    Training Mode (Option A):
        Standard RL training with synchronous step():
        ```python
        env = HierarchicalGridEnv(config)
        obs, _ = env.reset()
        # Get joint action from policy
        action = policy(obs)
        obs, rewards, done, truncated, info = env.step(action)
        ```

    Testing Mode (Option B):
        Event-driven simulation with realistic timing:
        ```python
        env = HierarchicalGridEnv(config)
        env.reset()
        events_processed = env.run_event_driven_simulation(t_end=3600.0)
        ```

    Attributes:
        grid_system: GridSystemAgent managing all PowerGridAgents
        system_protocol: Protocol for system-level coordination
    """

    def __init__(self, env_config: Dict[str, Any]):
        """Initialize hierarchical grid environment.

        Args:
            env_config: Configuration dictionary with keys:
                - system_protocol: str, protocol name for system coordination
                  ('setpoint', 'price_signal', or 'system'). Default: 'system'
                - system_tick_interval: float, tick interval for system agent (s)
                - grid_tick_interval: float, tick interval for grid agents (s)
                - device_tick_interval: float, tick interval for device agents (s)
                - Plus all keys from NetworkedGridEnv
        """
        # Store hierarchy config before parent init
        self._system_protocol_name = env_config.get('system_protocol', 'system')
        self._system_tick_interval = env_config.get('system_tick_interval', 300.0)
        self._grid_tick_interval = env_config.get('grid_tick_interval', 60.0)
        self._device_tick_interval = env_config.get('device_tick_interval', 1.0)

        # Initialize parent (builds agents and network)
        super().__init__(env_config)

        # Build GridSystemAgent after agents are created
        self._grid_system = self._build_grid_system_agent()

        # Register with HERON's SystemAgent integration
        self.set_system_agent(self._grid_system)

    @property
    def grid_system(self) -> GridSystemAgent:
        """Get the GridSystemAgent managing this environment."""
        return self._grid_system

    def _build_grid_system_agent(self) -> GridSystemAgent:
        """Build GridSystemAgent to manage all PowerGridAgents.

        Returns:
            Configured GridSystemAgent instance
        """
        # Create system-level protocol
        protocol = self._create_system_protocol()

        # Get all PowerGridAgents as grids for the system agent
        grids = [
            agent for agent in self.agent_dict.values()
            if isinstance(agent, PowerGridAgent)
        ]

        # Configure tick intervals on grid agents
        for grid in grids:
            grid.tick_interval = self._grid_tick_interval
            # Also configure device tick intervals
            for device in grid.devices.values():
                device.tick_interval = self._device_tick_interval

        # Create GridSystemAgent
        system_agent = GridSystemAgent(
            agent_id=f"{self._name}_system",
            protocol=protocol,
            env_id=self.env_id,
            grids=grids,
            tick_interval=self._system_tick_interval,
        )

        # Set message broker if available
        if self.message_broker is not None:
            system_agent.set_message_broker(self.message_broker)

        return system_agent

    def _create_system_protocol(self) -> Protocol:
        """Create protocol for system-level coordination.

        Returns:
            Protocol instance based on configuration
        """
        if self._system_protocol_name == 'setpoint':
            return SetpointProtocol()
        elif self._system_protocol_name == 'system':
            return SystemProtocol()
        elif self._system_protocol_name == 'none':
            return NoProtocol()
        else:
            # Default to SystemProtocol
            return SystemProtocol()

    # ============================================
    # Training Mode (Option A) - Synchronous Step
    # ============================================

    def step(self, action_n: Dict[str, Any]):
        """Execute one environment step using SystemAgent coordination.

        This method extends the parent step() to use SystemAgent for
        coordinating actions across the hierarchy.

        Args:
            action_n: Dictionary mapping agent IDs to actions.
                     Can be keyed by grid agent IDs or a single 'system' key.

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
        """
        # If action is provided at system level, distribute to grids
        if 'system' in action_n:
            system_action = action_n['system']
            # Use SystemAgent to distribute actions
            self._grid_system.act(
                self._grid_system.observe(self._get_global_state()),
                upstream_action=system_action
            )
            # Convert to per-grid actions for parent step
            action_n = self._get_grid_actions_from_system()

        # Use parent step logic
        return super().step(action_n)

    def _get_grid_actions_from_system(self) -> Dict[str, Any]:
        """Extract per-grid actions after system coordination.

        Returns:
            Dictionary mapping grid agent IDs to their actions
        """
        actions = {}
        for grid_id, grid in self._grid_system.grids.items():
            if hasattr(grid, 'action') and grid.action is not None:
                # Get the continuous action vector
                actions[grid_id] = grid.action.c.copy() if grid.action.c is not None else None
        return actions

    def reset(self, seed=None, options=None):
        """Reset environment and SystemAgent hierarchy.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (observations, info)
        """
        # Reset parent environment
        obs, info = super().reset(seed=seed, options=options)

        # Reset SystemAgent hierarchy
        self._grid_system.reset_system()
        self._grid_system.reset_all_grids()

        # Add system-level info
        info['system_state'] = self._grid_system.state.vector().tolist() if self._grid_system.state else []

        return obs, info

    # ============================================
    # Testing Mode (Option B) - Event-Driven
    # ============================================

    def setup_hierarchical_event_driven(self) -> EventScheduler:
        """Setup event-driven execution with full agent hierarchy.

        This method configures the EventScheduler with:
        - All device agents (L1) at fast tick rate
        - All grid agents (L2) at medium tick rate
        - System agent (L3) at slow tick rate

        Returns:
            Configured EventScheduler
        """
        # First, register all agents with HERON
        # Devices first (fastest tick)
        for grid in self._grid_system.grids.values():
            for device in grid.devices.values():
                self.register_agent(device)

        # Then grid agents
        for grid in self._grid_system.grids.values():
            self.register_agent(grid)

        # Finally system agent
        self.register_agent(self._grid_system)

        # Use HERON's setup_event_driven
        scheduler = self.setup_event_driven()

        return scheduler

    def run_event_driven_simulation(
        self,
        t_end: float,
        on_action_effect: Optional[Callable[[AgentID, Any], None]] = None,
        max_events: Optional[int] = None,
    ) -> int:
        """Run event-driven simulation with SystemAgent hierarchy.

        This is a convenience method that:
        1. Sets up the event scheduler with all agents
        2. Configures default handlers
        3. Runs the simulation

        Args:
            t_end: End time for simulation (seconds)
            on_action_effect: Optional callback when actions take effect
            max_events: Maximum number of events to process

        Returns:
            Number of events processed
        """
        # Setup scheduler if needed
        if self.scheduler is None:
            self.setup_hierarchical_event_driven()

        # Use HERON's convenience method
        return self.run_event_driven_with_system_agent(
            t_end=t_end,
            get_global_state=self._get_global_state,
            on_action_effect=on_action_effect or self._default_action_effect,
            max_events=max_events,
        )

    def _get_global_state(self) -> Dict[str, Any]:
        """Get current global state for event-driven observation.

        Returns:
            Dictionary containing network and timestep information
        """
        state = {
            'timestep': self._t,
            'converged': self.net.get('converged', False) if self.net else False,
            'frequency': 60.0,  # Nominal frequency
        }

        # Add aggregate metrics from network
        if self.net is not None and self.net.get('converged', False):
            state['total_generation'] = float(self.net.res_sgen['p_mw'].sum()) if 'res_sgen' in self.net else 0.0
            state['total_load'] = float(self.net.res_load['p_mw'].sum()) if 'res_load' in self.net else 0.0

        return state

    def _default_action_effect(self, agent_id: AgentID, action: Any) -> None:
        """Default handler for action effects in event-driven mode.

        Args:
            agent_id: Agent whose action is taking effect
            action: The action to apply
        """
        # Find the agent and apply action
        if agent_id in self.agent_dict:
            agent = self.agent_dict[agent_id]
            obs = agent.observe()
            agent.act(obs, upstream_action=action)

        # Check if we need to run power flow
        # (Run after all device actions in a timestep are applied)
        self._maybe_run_power_flow()

    def _maybe_run_power_flow(self) -> None:
        """Run power flow if needed after action effects."""
        # Simple heuristic: run power flow if network state changed
        # In practice, you might want more sophisticated batching
        try:
            pp.runpp(self.net)
        except Exception:
            self.net['converged'] = False

    # ============================================
    # Observation/Action Space for Hierarchy
    # ============================================

    def get_system_observation_space(self) -> Box:
        """Get observation space for system-level observations.

        Returns:
            Box space for system observations
        """
        return self._grid_system.get_system_observation_space()

    def get_system_action_space(self) -> Box:
        """Get action space for system-level actions.

        Returns:
            Box space for system actions (joint action of all grids)
        """
        return self._grid_system.get_system_action_space()

    def get_hierarchical_spaces(self) -> Dict[str, Dict[str, Box]]:
        """Get observation and action spaces for all hierarchy levels.

        Returns:
            Dictionary with 'observation_space' and 'action_space' keys,
            each containing per-level spaces.
        """
        return {
            'observation_space': {
                'system': self.get_system_observation_space(),
                'grids': {
                    grid_id: grid.observation_space
                    for grid_id, grid in self._grid_system.grids.items()
                },
            },
            'action_space': {
                'system': self.get_system_action_space(),
                'grids': {
                    grid_id: grid.action_space
                    for grid_id, grid in self._grid_system.grids.items()
                },
            },
        }

    # ============================================
    # System-Level Metrics
    # ============================================

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics from GridSystemAgent.

        Returns:
            Dictionary with system metrics including:
            - frequency: System frequency (Hz)
            - total_generation: Aggregate generation (MW)
            - total_load: Aggregate load (MW)
            - system_cost: Total cost across all grids
            - system_safety: Total safety violations
        """
        # Update system agent from environment state
        self._grid_system.update_from_environment(self._get_global_state())

        return {
            'frequency': self._grid_system.frequency,
            'total_generation': self._grid_system.total_generation,
            'total_load': self._grid_system.total_load,
            'system_cost': self._grid_system.cost,
            'system_safety': self._grid_system.safety,
            'num_grids': len(self._grid_system.grids),
            'system_state': self._grid_system.state.vector().tolist() if self._grid_system.state else [],
        }

    def get_hierarchy_info(self) -> Dict[str, Any]:
        """Get information about the agent hierarchy.

        Returns:
            Dictionary describing the hierarchy structure
        """
        hierarchy = {
            'system': {
                'agent_id': self._grid_system.agent_id,
                'protocol': self._grid_system.protocol.__class__.__name__,
                'tick_interval': self._grid_system.tick_interval,
            },
            'grids': {},
        }

        for grid_id, grid in self._grid_system.grids.items():
            hierarchy['grids'][grid_id] = {
                'agent_id': grid.agent_id,
                'protocol': grid.protocol.__class__.__name__ if grid.protocol else 'None',
                'tick_interval': grid.tick_interval,
                'num_devices': len(grid.devices),
                'devices': list(grid.devices.keys()),
            }

        return hierarchy

    # ============================================
    # Distributed Mode Configuration
    # ============================================

    def configure_for_distributed(self) -> None:
        """Configure entire hierarchy for distributed execution.

        Sets up message broker channels for all agents in the hierarchy.
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker not configured. Set centralized=False in config.")

        # Configure system agent and all subordinates
        self._grid_system.configure_for_distributed(self.message_broker)

        # Setup broker channels
        self.setup_broker_channels()

    def close(self) -> None:
        """Clean up environment and hierarchy resources."""
        super().close()

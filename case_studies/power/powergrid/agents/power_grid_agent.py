"""Grid-level coordinator agents for hierarchical control.

GridAgent manages a set of device agents, implementing coordination
protocols like price signals, setpoints, or consensus algorithms.

GridAgent extends CoordinatorAgent from the HERON framework, inheriting
standard coordinator-level capabilities while adding power-grid specific
functionality.
"""

from typing import Any, Dict as DictType, Iterable, List, Optional, Union

import gymnasium as gym
import numpy as np
import pandapower as pp
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from heron.agents.coordinator_agent import CoordinatorAgent, COORDINATOR_LEVEL
from heron.agents.base import Agent, AgentID
from heron.core.action import Action
from heron.core.observation import Observation
from powergrid.agents.device_agent import DeviceAgent
from heron.core.policies import Policy
from heron.protocols.base import NoProtocol, Protocol
from powergrid.core.state.state import GridState
from powergrid.agents.generator import Generator
from powergrid.agents.storage import ESS
from powergrid.core.features.network import BusVoltages, LineFlows, NetworkMetrics


GRID_LEVEL = COORDINATOR_LEVEL  # Level identifier for grid-level agents (same as coordinator)


class GridAgent(CoordinatorAgent):
    """Grid-level coordinator for managing device agents.

    GridAgent extends CoordinatorAgent from the HERON framework, coordinating
    multiple device agents using specified protocols and optionally a centralized
    policy for joint decision-making.

    Attributes:
        devices: Dictionary mapping device agent IDs to DeviceAgent instances
                 (alias for subordinate_agents)
        protocol: Coordination protocol for managing subordinate devices
        policy: Optional centralized policy for joint action computation
        centralized: If True, uses centralized policy; if False, devices act independently
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),

        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # GridAgent specific params
        devices: Optional[List[DeviceAgent]] = None,
        centralized: bool = True,
        grid_config: DictType[str, Any] = {},

        # timing params (for event-driven scheduling - Option B)
        tick_interval: float = 60.0,
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,
    ):
        """Initialize grid coordinator.

        Args:
            agent_id: Unique identifier
            policy: Optional centralized policy for joint action computation
            protocol: Protocol for coordinating devices
            upstream_id: Optional parent agent ID for hierarchy structure
            env_id: Optional environment ID for multi-environment isolation
            devices: List of device agents to manage (alternative to grid_config)
            centralized: If True, uses centralized policy; if False, devices act independently
            grid_config: Grid configuration dictionary
            tick_interval: Time between agent ticks (default 60s for coordinators)
            obs_delay: Observation delay
            act_delay: Action delay
            msg_delay: Message delay
        """
        self.centralized = centralized
        self._grid_config = grid_config

        # Build config for CoordinatorAgent
        # Convert device list to agent_configs if provided
        if devices is not None:
            # Store devices to be processed after super().__init__
            self._init_devices = devices
            config = {'agents': []}
        else:
            self._init_devices = None
            config = {'agents': grid_config.get('devices', [])}

        super().__init__(
            agent_id=agent_id,
            policy=policy,
            protocol=protocol,
            upstream_id=upstream_id,
            env_id=env_id,
            config=config,
            tick_interval=tick_interval,
            obs_delay=obs_delay,
            act_delay=act_delay,
            msg_delay=msg_delay,
        )

        # Use GridState instead of CoordinatorAgentState for power-grid domain
        self.state = GridState(
            owner_id=self.agent_id,
            owner_level=GRID_LEVEL
        )

        # If devices were provided directly, set them up now
        if self._init_devices is not None:
            self.subordinate_agents = {
                device.agent_id: device for device in self._init_devices
            }
            # Set upstream relationship for devices
            for device in self._init_devices:
                device.upstream_id = self.agent_id

    # ============================================
    # Backward Compatibility: devices <-> subordinate_agents
    # ============================================

    @property
    def devices(self) -> DictType[AgentID, DeviceAgent]:
        """Alias for subordinate_agents for backward compatibility."""
        return self.subordinate_agents

    @devices.setter
    def devices(self, value: DictType[AgentID, DeviceAgent]) -> None:
        """Setter for devices alias."""
        self.subordinate_agents = value

    def _build_subordinate_agents(
        self,
        agent_configs: List[DictType[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, DeviceAgent]:
        """Build device agents from configuration.

        Overrides CoordinatorAgent method to create DeviceAgent instances.

        Args:
            agent_configs: List of device configuration dictionaries
            env_id: Environment ID for multi-environment isolation
            upstream_id: Upstream agent ID (this grid agent)

        Returns:
            Dictionary mapping device IDs to DeviceAgent instances
        """
        return self._build_device_agents(
            agent_configs,
            env_id=env_id,
            upstream_id=upstream_id
        )

    def _build_device_agents(
        self,
        device_configs: List[DictType[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, DeviceAgent]:
        """Build device agents from configuration.

        Override in subclasses to create specific device types.

        Args:
            device_configs: List of device configuration dictionaries
            env_id: Environment ID for multi-environment isolation
            upstream_id: Upstream agent ID (this grid agent)

        Returns:
            Dictionary mapping device IDs to DeviceAgent instances
        """
        # Default implementation - override in subclasses
        return {}

    # ============================================
    # Cost/Safety Properties (Override to allow setting)
    # ============================================

    @property
    def cost(self) -> float:
        """Get total cost from all device agents.

        If _cost is set explicitly, return that value.
        Otherwise, aggregate from subordinate devices.
        """
        if hasattr(self, '_cost') and self._cost is not None:
            return self._cost
        return sum(agent.cost for agent in self.subordinate_agents.values())

    @cost.setter
    def cost(self, value: float) -> None:
        """Set cost value explicitly."""
        self._cost = value

    @property
    def safety(self) -> float:
        """Get total safety penalty from all device agents.

        If _safety is set explicitly, return that value.
        Otherwise, aggregate from subordinate devices.
        """
        if hasattr(self, '_safety') and self._safety is not None:
            return self._safety
        return sum(agent.safety for agent in self.subordinate_agents.values())

    @safety.setter
    def safety(self, value: float) -> None:
        """Set safety value explicitly."""
        self._safety = value

    def get_reward(self) -> DictType[str, float]:
        """Get aggregated reward from devices.

        Returns:
            Dict with total cost and safety values
        """
        return {"cost": self.cost, "safety": self.safety}

    # ============================================
    # Observation Methods (Override for device terminology)
    # ============================================

    def _build_local_observation(
        self,
        subordinate_obs: DictType[AgentID, Observation],
        *args,
        **kwargs
    ) -> Any:
        """Build local observation from device observations.

        Overrides CoordinatorAgent to use device-specific terminology.

        Args:
            subordinate_obs: Dictionary mapping device IDs to their observations
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Aggregated local observation dictionary
        """
        return {
            "device_obs": subordinate_obs,  # Use device_obs for backward compat
            "subordinate_obs": subordinate_obs,  # Also include standard name
            "grid_state": self.state.vector(),
            "coordinator_state": self.state.vector(),
        }

    # ============================================
    # Action Methods (Override for centralized flag)
    # ============================================

    def act(self, observation: Observation, upstream_action: Any = None) -> Optional[Action]:
        """Compute coordination action and distribute to devices.

        Extends CoordinatorAgent.act() with centralized mode check.

        Args:
            observation: Aggregated observation
            upstream_action: Pre-computed action (if any)

        Returns:
            Action object (if centralized mode with policy)

        Raises:
            NotImplementedError: If decentralized mode without policy
            RuntimeError: If centralized mode without action or policy
        """
        # Check centralized vs decentralized mode
        if not self.centralized:
            if self.policy is None and upstream_action is None:
                raise NotImplementedError(
                    "Decentralized mode requires devices to act independently. "
                    "Use device.act() directly or provide a policy."
                )

        # Get coordinator action from policy if available
        if upstream_action is not None:
            action = upstream_action
        elif self.policy is not None:
            action = self.policy.forward(observation)
        else:
            raise RuntimeError("No action or policy provided for GridAgent.")

        # Coordinate devices using unified method
        self.coordinate_devices(observation, action)

        return action

    # ============================================
    # Coordination Methods (Aliases for backward compatibility)
    # ============================================

    def coordinate_devices(
        self,
        observation: Observation,
        action: Any,
    ) -> None:
        """Unified coordination method using protocol.

        Alias for coordinate_subordinates() for backward compatibility.

        Args:
            observation: Current observation
            action: Computed action from coordinator
        """
        self.coordinate_subordinates(observation, action)

    def coordinate_device(
        self,
        observation: Observation,
        action: Any,
    ) -> None:
        """Coordinate a single device or all devices with given action.

        Alias for coordinate_devices for backward compatibility.

        Args:
            observation: Current observation
            action: Computed action from coordinator
        """
        self.coordinate_devices(observation, action)

    # ============================================
    # State Update Hooks
    # ============================================

    def _update_state_with_upstream_info(self, upstream_info: Optional[DictType[str, Any]]) -> None:
        """Update internal state based on info received from upstream agent.

        In distributed mode, this receives network state from ProxyAgent and
        updates internal features/attributes accordingly.

        Args:
            upstream_info: Info dict received from upstream agent (ProxyAgent in distributed mode)
        """
        if not upstream_info:
            return

        # In distributed mode, upstream_info contains network state from ProxyAgent
        # Update internal state features based on received network info
        self._update_from_network_state(upstream_info)

    def _update_from_network_state(self, network_state: DictType[str, Any]) -> None:
        """Update internal features from received network state.

        Override in subclasses to handle power-grid specific state updates.

        Args:
            network_state: Network state dict from ProxyAgent
        """
        pass

    # ============================================
    # Utility Methods
    # ============================================

    def _consume_network_state(self) -> Optional[DictType[str, Any]]:
        """Consume network state from message broker.

        In Option B (event-driven mode), agents retrieve network state from
        messages delivered via the message broker.

        Returns:
            Network state payload or None if no message available
        """
        # Check message broker for network state messages
        messages = self.receive_messages(clear=True)
        for msg in messages:
            if 'network_state' in msg:
                return msg['network_state']
        return None

    def __repr__(self) -> str:
        num_subs = len(self.devices)
        protocol_name = self.protocol.__class__.__name__
        return f"GridAgent(id={self.agent_id}, devices={num_subs}, protocol={protocol_name})"


class PowerGridAgent(GridAgent):
    """Grid agent for power system coordination with PandaPower integration.

    PowerGridAgent extends GridAgent with power system-specific functionality,
    including PandaPower network integration, device management, and state updates.

    Attributes:
        net: PandaPower network object
        name: Grid name (from network)
        config: Grid configuration dictionary
        sgen: Dictionary of renewable energy sources (Generator)
        base_power: Base power for normalization (MW)
        load_scale: Scaling factor for loads
    """

    def __init__(
        self,
        # Base class args
        protocol: Protocol = NoProtocol(),
        policy: Optional[Policy] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        grid_config: DictType[str, Any] = {},

        # GridAgent args
        devices: Optional[List[Agent]] = None,
        centralized: bool = True,

        # PowerGridAgent specific params
        net: Optional[pp.pandapowerNet] = None,

        # timing params (for event-driven scheduling - Option B)
        tick_interval: float = 60.0,
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        msg_delay: float = 0.0,
    ):
        """Initialize power grid agent.

        Args:
            protocol: Coordination protocol
            policy: Optional centralized policy
            upstream_id: Optional parent agent ID for hierarchy structure
            env_id: Optional environment ID for multi-environment isolation
            grid_config: Grid configuration dictionary
            devices: Optional list of device agents to manage
            centralized: If True, uses centralized policy for coordination
            net: PandaPower network object (required)
            tick_interval: Time between agent ticks (default 60s for coordinators)
            obs_delay: Observation delay
            act_delay: Action delay
            msg_delay: Message delay
        """
        if net is None:
            raise ValueError("PandaPower network 'net' must be provided to PowerGridAgent.")

        self.net = net
        self.name = net.name
        self.sgen: DictType[str, Generator] = {}
        self.storage: DictType[str, ESS] = {}
        self.base_power = grid_config.get("base_power", 1)
        self.load_scale = grid_config.get("load_scale", 1)
        self.load_rescaling(net, self.load_scale)

        super().__init__(
            agent_id=self.name,
            protocol=protocol,
            policy=policy,
            upstream_id=upstream_id,
            env_id=env_id,
            devices=devices,
            centralized=centralized,
            grid_config=grid_config,
            tick_interval=tick_interval,
            obs_delay=obs_delay,
            act_delay=act_delay,
            msg_delay=msg_delay,
        )

    def _build_device_agents(
        self,
        device_configs: List[DictType[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, DeviceAgent]:
        """Build device agents from configuration.

        Args:
            device_configs: List of device configuration dictionaries
            env_id: Environment ID for multi-environment isolation
            upstream_id: Upstream agent ID (this grid agent)

        Returns:
            Dictionary mapping device IDs to DeviceAgent instances
        """
        devices = {}
        generators = []
        ess_devices = []

        for device_config in device_configs:
            device_type = device_config.get('type', None)

            # Wrap YAML config in device_state_config if not already wrapped
            # Keep name and type at top level
            if 'device_state_config' not in device_config:
                wrapped_config = {
                    'name': device_config.get('name', 'device_agent'),
                    'type': device_type,
                    'device_state_config': device_config
                }
            else:
                wrapped_config = device_config

            if device_type == 'Generator':
                generator = Generator(
                    upstream_id=upstream_id,
                    env_id=env_id,
                    device_config=wrapped_config,
                )
                devices[generator.agent_id] = generator
                generators.append(generator)
            elif device_type == 'ESS':
                ess = ESS(
                    upstream_id=upstream_id,
                    env_id=env_id,
                    device_config=wrapped_config,
                )
                devices[ess.agent_id] = ess
                ess_devices.append(ess)
            else:
                # Other device types not yet implemented
                pass

        # Add devices to pandapower network
        if generators:
            self._add_sgen(generators)
        if ess_devices:
            self._add_storage(ess_devices)

        return devices

    # ============================================
    # Network Setup Methods
    # ============================================

    def _add_sgen(self, sgens: Iterable[Generator] | Generator):
        """Add renewable generators (solar/wind) to the network.

        Args:
            sgens: Single RES instance or iterable of RES instances
        """
        if not isinstance(sgens, Iterable):
            sgens = [sgens]

        for sgen in sgens:
            bus_id = pp.get_element_index(self.net, 'bus', self.name + ' ' + sgen.bus)
            pp.create_sgen(
                self.net,
                bus_id,
                name=self.name + ' ' + sgen.config.name,
                index=None,  # Let pandapower auto-assign to avoid collisions during fuse
                p_mw=sgen.electrical.P_MW,
                sn_mva=sgen.limits.s_rated_MVA,
                max_p_mw=sgen.limits.p_max_MW,
                min_p_mw=sgen.limits.p_min_MW,
                max_q_mvar=sgen.limits.q_max_MVAr,
                min_q_mvar=sgen.limits.q_min_MVAr
            )
            self.sgen[sgen.config.name] = sgen

    def _add_storage(self, storages: Iterable[ESS] | ESS):
        """Add energy storage systems to the network.

        Args:
            storages: Single ESS instance or iterable of ESS instances
        """
        if not isinstance(storages, Iterable):
            storages = [storages]

        for ess in storages:
            bus_id = pp.get_element_index(self.net, 'bus', self.name + ' ' + ess.bus)
            pp.create_storage(
                self.net,
                bus_id,
                name=self.name + ' ' + ess.config.name,
                index=None,  # Let pandapower auto-assign to avoid collisions during fuse
                p_mw=ess.electrical.P_MW,
                max_e_mwh=ess.storage.e_capacity_MWh,
                soc_percent=ess.storage.soc * 100,
                max_p_mw=ess.storage.p_ch_max_MW,
                min_p_mw=-ess.storage.p_dsc_max_MW,
                max_q_mvar=ess.limits.q_max_MVAr if ess.limits else 0.0,
                min_q_mvar=ess.limits.q_min_MVAr if ess.limits else 0.0,
            )
            self.storage[ess.config.name] = ess

    def add_sgen(self, sgens: Iterable[Generator] | Generator):
        """Add renewable generators (solar/wind) to the network.

        Public API for adding sgen devices after initialization.

        Args:
            sgens: Single Generator instance or iterable of Generator instances
        """
        self._add_sgen(sgens)

    def add_storage(self, storages: Iterable[ESS] | ESS):
        """Add energy storage systems to the network.

        Public API for adding ESS devices after initialization.

        Args:
            storages: Single ESS instance or iterable of ESS instances
        """
        self._add_storage(storages)

    def add_dataset(self, dataset):
        """Add time-series dataset for loads and renewables.

        Args:
            dataset: Dictionary containing 'load', 'solar', 'wind' time series
        """
        self.dataset = dataset

    def fuse_buses(self, ext_net, bus_name):
        """Merge this grid with an external network by fusing buses.

        Args:
            ext_net: External PandaPower network
            bus_name: Name of bus to fuse with external grid

        Returns:
            Merged PandaPower network
        """
        self.net.ext_grid.in_service = False
        net, index = pp.merge_nets(
            ext_net,
            self.net,
            validate=False,
            return_net2_reindex_lookup=True
        )
        substation = pp.get_element_index(net, 'bus', bus_name)
        ext_grid = index['bus'][self.net.ext_grid.bus.values[0]]
        pp.fuse_buses(net, ext_grid, substation)

        return net

    def load_rescaling(self, net, scale):
        """Apply scaling factor to local loads.

        Args:
            net: PandaPower network
            scale: Scaling multiplier
        """
        local_load_ids = pp.get_element_index(net, 'load', self.name, False)
        net.load.loc[local_load_ids, 'scaling'] *= scale

    # ============================================
    # Observation Methods
    # ============================================

    def _build_local_observation(self, device_obs: DictType[AgentID, Observation], *args, **kwargs) -> Any:
        """Build local observation including device states and network results.

        Args:
            device_obs: Device observations dictionary
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments (can include 'net')

        Returns:
            Local observation dictionary with device and network state
        """
        # Use provided net or fall back to self.net
        net = kwargs.get('net', self.net)

        # Update grid state from network results (only if net provided)
        if net is not None:
            self._update_grid_state(net)

        local = super()._build_local_observation(device_obs)
        local['state'] = self._get_obs(net, device_obs)
        return local

    def _get_obs(self, net, device_obs=None):
        """Extract numerical observation vector from network state.

        Args:
            net: PandaPower network (None in distributed mode)
            device_obs: Optional device observations (computed if not provided)

        Returns:
            Flattened observation array (float32)
        """
        if device_obs is None:
            device_obs = {
                agent_id: agent.observe()
                for agent_id, agent in self.devices.items()
            }
        obs = np.array([])
        for _, ob in device_obs.items():
            # P, Q, SoC of energy storage units
            # P, Q, UC status of generators
            obs = np.concatenate((obs, ob.local['state']))

        # P, Q at all buses (only in centralized mode when net is provided)
        if net is not None:
            local_load_ids = pp.get_element_index(net, 'load', self.name, False)
            load_pq = net.res_load.iloc[local_load_ids].values
            obs = np.concatenate([obs, load_pq.ravel() / self.base_power])

        return obs.astype(np.float32)

    def _update_grid_state(self, net) -> None:
        """Update GridState features from PandaPower network results.

        Args:
            net: PandaPower network with power flow results
        """
        if not net.get("converged", False):
            # If power flow didn't converge, keep previous state
            return

        # Extract local bus indices
        local_bus_ids = pp.get_element_index(net, 'bus', self.name, False)
        if len(local_bus_ids) == 0:
            return

        # Bus voltages
        bus_vm = net.res_bus.loc[local_bus_ids, 'vm_pu'].values
        bus_va = net.res_bus.loc[local_bus_ids, 'va_degree'].values
        bus_names = net.bus.loc[local_bus_ids, 'name'].tolist()

        # Line flows
        local_line_ids = pp.get_element_index(net, 'line', self.name, False)
        if len(local_line_ids) > 0:
            line_p = net.res_line.loc[local_line_ids, 'p_from_mw'].values
            line_q = net.res_line.loc[local_line_ids, 'q_from_mvar'].values
            line_loading = net.res_line.loc[local_line_ids, 'loading_percent'].values
            line_names = net.line.loc[local_line_ids, 'name'].tolist()
        else:
            line_p = np.array([])
            line_q = np.array([])
            line_loading = np.array([])
            line_names = []

        # Network metrics
        local_sgen_ids = pp.get_element_index(net, 'sgen', self.name, False)
        local_load_ids = pp.get_element_index(net, 'load', self.name, False)

        total_gen_mw = net.res_sgen.loc[local_sgen_ids, 'p_mw'].sum() if len(local_sgen_ids) > 0 else 0.0
        total_gen_mvar = net.res_sgen.loc[local_sgen_ids, 'q_mvar'].sum() if len(local_sgen_ids) > 0 else 0.0
        total_load_mw = net.res_load.loc[local_load_ids, 'p_mw'].sum() if len(local_load_ids) > 0 else 0.0
        total_load_mvar = net.res_load.loc[local_load_ids, 'q_mvar'].sum() if len(local_load_ids) > 0 else 0.0
        total_loss_mw = total_gen_mw - total_load_mw

        # Update GridState features
        self.state.features = [
            BusVoltages(
                vm_pu=bus_vm,
                va_deg=bus_va,
                bus_names=bus_names,
            ),
            LineFlows(
                p_from_mw=line_p,
                q_from_mvar=line_q,
                loading_percent=line_loading,
                line_names=line_names,
            ),
            NetworkMetrics(
                total_gen_mw=float(total_gen_mw),
                total_load_mw=float(total_load_mw),
                total_loss_mw=float(total_loss_mw),
                total_gen_mvar=float(total_gen_mvar),
                total_load_mvar=float(total_load_mvar),
            ),
        ]

    # ============================================
    # Space Construction Methods
    # ============================================

    def get_device_action_spaces(self) -> DictType[str, gym.Space]:
        """Get action spaces for all devices.

        Returns:
            Dictionary mapping device IDs to their action spaces
        """
        return {
            device.agent_id: device.action_space
            for device in self.devices.values()
        }

    def get_grid_action_space(self):
        """Construct combined action space for all devices.

        Returns:
            Gymnasium space representing joint action space of all devices
        """
        low, high, discrete_n = [], [], []
        for sp in self.get_device_action_spaces().values():
            if isinstance(sp, Box):
                low = np.append(low, sp.low)
                high = np.append(high, sp.high)
            elif isinstance(sp, Dict):
                # Handle Dict spaces (e.g., Generator with continuous + discrete)
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
            return Dict({"continuous": Box(low=low, high=high, dtype=np.float32),
                        'discrete': MultiDiscrete(discrete_n)})
        elif len(low):  # Continuous only
            return Box(low=low, high=high, dtype=np.float32)
        elif len(discrete_n):  # Discrete only
            return MultiDiscrete(discrete_n)
        else:  # No actionable agents
            return Discrete(1)

    def get_grid_observation_space(self, net):
        """Get observation space for this grid.

        Args:
            net: PandaPower network

        Returns:
            Gymnasium Box space for grid observations
        """
        # Ensure powerflow has run to get correct observation size
        try:
            pp.runpp(net, algorithm='nr', init='flat', max_iteration=100)
        except Exception:
            # If powerflow fails, still create space (may need adjustment later)
            pass

        return Box(
            low=-np.inf,
            high=np.inf,
            shape=self._get_obs(net).shape,
            dtype=np.float32
        )

    # ============================================
    # State Update Methods
    # ============================================

    def update_state(self, net, t):
        """Update grid state from dataset and device actions.

        Args:
            net: PandaPower network to update
            t: Timestep index in dataset
        """
        load_scaling = self.dataset['load'][t]

        local_ids = pp.get_element_index(net, 'load', self.name, False)
        net.load.loc[local_ids, 'scaling'] = load_scaling
        self.load_rescaling(net, self.load_scale)

        # Update all generators with their actions
        for name, generator in self.sgen.items():
            generator.update_state()

            # Update network with generator state
            local_ids = pp.get_element_index(net, 'sgen', self.name + ' ' + name)
            states = ['p_mw', 'q_mvar', 'in_service']
            # Ensure Q_MVAr is not None (use 0.0 if None)
            q_mvar = generator.electrical.Q_MVAr if generator.electrical.Q_MVAr is not None else 0.0
            values = [generator.electrical.P_MW, q_mvar, generator.status.in_service]
            net.sgen.loc[local_ids, states] = values


    def sync_global_state(self, net, t):
        """Sync global state from PandaPower network to devices.

        In centralized mode, syncs from pandapower network directly.
        In distributed mode, network state has already been synced via ProxyAgent
        in _update_from_network_state(), so we can skip this.

        Args:
            net: PandaPower network with power flow results (None in distributed mode)
            t: Current timestep
        """
        # In distributed mode, state is already synced via ProxyAgent
        if net is None:
            return

        # Centralized mode: sync from pandapower network
        for name, dg in self.sgen.items():
            local_ids = pp.get_element_index(net, 'sgen', self.name + ' ' + name)
            p_mw_val = net.res_sgen.loc[local_ids, 'p_mw']
            q_mvar_val = net.res_sgen.loc[local_ids, 'q_mvar']
            # Handle both scalar and array cases
            p_mw = p_mw_val if np.isscalar(p_mw_val) else p_mw_val.values[0]
            q_mvar = q_mvar_val if np.isscalar(q_mvar_val) else q_mvar_val.values[0]
            # Handle NaN values (can occur if power flow didn't converge properly)
            if not np.isnan(p_mw):
                dg.electrical.P_MW = p_mw
            if not np.isnan(q_mvar):
                dg.electrical.Q_MVAr = q_mvar

    def _update_from_network_state(self, network_state: DictType[str, Any]) -> None:
        """Update internal features from received network state.

        This method is called in distributed mode when network state is received
        from ProxyAgent. It updates the GridState features with the latest
        network information.

        Args:
            network_state: Network state dict from ProxyAgent containing:
                - converged: bool
                - bus_voltages: {vm_pu: [...], overvoltage: float, undervoltage: float}
                - line_loading: {loading_percent: [...], overloading: float}
                - device_results: {device_name: {p_mw: float, q_mvar: float}}
        """
        if not network_state or not network_state.get('converged', False):
            return

        # Extract bus voltages
        bus_voltages_data = network_state.get('bus_voltages', {})
        bus_vm = np.array(bus_voltages_data.get('vm_pu', []))

        # Extract line loading
        line_loading_data = network_state.get('line_loading', {})
        line_loading_percent = np.array(line_loading_data.get('loading_percent', []))

        # Update GridState features
        # Note: We don't have bus/line names in the message, so we use indices
        if len(bus_vm) > 0:
            from powergrid.core.features.network import BusVoltages
            self.state.features.append(
                BusVoltages(
                    vm_pu=bus_vm,
                    va_deg=np.zeros_like(bus_vm),  # Not transmitted in message
                    bus_names=[f"bus_{i}" for i in range(len(bus_vm))],
                )
            )

        if len(line_loading_percent) > 0:
            from powergrid.core.features.network import LineFlows
            self.state.features.append(
                LineFlows(
                    p_from_mw=np.zeros_like(line_loading_percent),  # Not transmitted
                    q_from_mvar=np.zeros_like(line_loading_percent),  # Not transmitted
                    loading_percent=line_loading_percent,
                    line_names=[f"line_{i}" for i in range(len(line_loading_percent))],
                )
            )

        # Update device results if available
        device_results = network_state.get('device_results', {})
        for device_name, result in device_results.items():
            if device_name in self.sgen:
                # Update device electrical state with power flow results
                device = self.sgen[device_name]
                p_mw = result.get('p_mw', device.electrical.P_MW)
                q_mvar = result.get('q_mvar', device.electrical.Q_MVAr)

                # Only update if values are valid (not NaN)
                if not np.isnan(p_mw):
                    device.electrical.P_MW = float(p_mw)
                if not np.isnan(q_mvar):
                    device.electrical.Q_MVAr = float(q_mvar)

    def update_cost_safety(self, net):
        """Update cost and safety metrics for the grid.

        This method computes both device-level costs (generation costs) and
        network-level safety metrics (voltage violations, line overloading).

        Execution modes:
        - Centralized (net != None): Directly accesses PandaPower network for
          voltage and line loading information
        - Distributed (net == None): Receives network state via messages from
          ProxyAgent, ensuring agents never access the network object directly

        Args:
            net: PandaPower network with power flow results, or None for distributed mode
        """
        self.cost, self.safety = 0, 0

        # Always update device-level costs (devices have local state)
        for dg in self.sgen.values():
            dg.update_cost_safety()
            self.cost += dg.cost
            self.safety += dg.safety

        # Network-level safety metrics
        if net is not None:
            # Centralized mode: access net directly
            if net.get("converged", False):
                local_bus_ids = pp.get_element_index(net, 'bus', self.name, False)
                local_vm = net.res_bus.loc[local_bus_ids].vm_pu.values
                overvoltage = np.maximum(local_vm - 1.05, 0).sum()
                undervoltage = np.maximum(0.95 - local_vm, 0).sum()

                local_line_ids = pp.get_element_index(net, 'line', self.name, False)
                local_line_loading = net.res_line.loc[local_line_ids].loading_percent.values
                overloading = np.maximum(local_line_loading - 100, 0).sum() * 0.01

                self.safety += overloading + overvoltage + undervoltage
        else:
            # Event-driven mode (Option B): check mailbox for network state
            network_state = self._consume_network_state()
            if network_state and network_state.get('converged', False):
                # Extract pre-computed safety metrics from message
                bus_voltages = network_state.get('bus_voltages', {})
                line_loading = network_state.get('line_loading', {})

                overvoltage = bus_voltages.get('overvoltage', 0)
                undervoltage = bus_voltages.get('undervoltage', 0)
                overloading = line_loading.get('overloading', 0)

                self.safety += overloading + overvoltage + undervoltage

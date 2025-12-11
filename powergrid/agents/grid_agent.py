"""Grid-level coordinator agents for hierarchical control.

GridAgent manages a set of device agents, implementing coordination
protocols like price signals, setpoints, or consensus algorithms.
"""

from typing import Any, Dict as DictType, Iterable, List, Optional

import gymnasium as gym
import numpy as np
import pandapower as pp
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from powergrid.agents.base import Agent, AgentID, Observation
from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.core.state import GridState
from powergrid.devices.generator import Generator
from powergrid.devices.storage import ESS
from powergrid.features.network import BusVoltages, LineFlows, NetworkMetrics
from powergrid.messaging.base import ChannelManager, MessageBroker


GRID_LEVEL = 2  # Level identifier for grid-level agents


class GridAgent(Agent):
    """Grid-level coordinator for managing device agents.

    GridAgent coordinates multiple device agents using specified protocols
    and optionally a centralized policy for joint decision-making.

    Attributes:
        devices: Dictionary mapping device agent IDs to DeviceAgent instances
        protocol: Coordination protocol for managing subordinate devices
        policy: Optional centralized policy for joint action computation
        centralized: If True, uses centralized policy; if False, devices act independently
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),

        # communication params
        message_broker: Optional[MessageBroker] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,

        # GridAgent specific params
        grid_config: DictType[str, Any] = {},
    ):
        """Initialize grid coordinator.

        Args:
            agent_id: Unique identifier
            policy: Optional centralized policy for joint action computation
            protocol: Protocol for coordinating devices
            message_broker: Optional message broker for hierarchical execution
            upstream_id: Optional parent agent ID for hierarchical execution
            env_id: Optional environment ID for multi-environment isolation
            grid_config: Grid configuration dictionary
        """
        self.protocol = protocol
        self.policy = policy
        self.state = GridState()

        # Build device agents
        device_configs = grid_config.get('devices', [])
        self.devices = self._build_device_agents(device_configs, message_broker)

        super().__init__(
            agent_id=agent_id,
            level=GRID_LEVEL,
            message_broker=message_broker,
            upstream_id=upstream_id,
            env_id=env_id,
            subordinates=self.devices,
        )

    def _build_device_agents(
        self,
        device_configs: List[DictType[str, Any]],
        message_broker: Optional[MessageBroker] = None,
    ) -> DictType[AgentID, DeviceAgent]:
        """Build device agents from configuration.

        Args:
            device_configs: List of device configuration dictionaries
            message_broker: Optional message broker for communication

        Returns:
            Dictionary mapping device IDs to DeviceAgent instances
        """
        pass

    # ============================================
    # Core Agent Lifecycle Methods
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset coordinator and all devices.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)

        # Reset devices
        for _, agent in self.devices.items():
            agent.reset(seed=seed, **kwargs)

        # Reset policy
        if self.policy is not None:
            self.policy.reset()

    def observe(self, global_state: Optional[DictType[str, Any]] = None, *args, **kwargs) -> Observation:
        """Collect observations from device agents.

        Args:
            global_state: Environment state

        Returns:
            Aggregated observation from all devices
        """
        # Collect device observations
        device_obs = {}
        for agent_id, agent in self.devices.items():
            device_obs[agent_id] = agent.observe(global_state)
        local_observation = self._build_local_observation(device_obs, *args, **kwargs)

        # TODO: update global info aggregation if needed
        global_info = global_state

        # TODO: update message aggregation if needed
        messages = []

        return Observation(
            timestamp=self._timestep,
            local=local_observation,
            global_info=global_info,
            messages=messages,
        )

    def _build_local_observation(self, device_obs: DictType[AgentID, Observation], *args, **kwargs) -> Any:
        """Build local observation from device observations.

        Args:
            device_obs: Dictionary mapping device IDs to their observations
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Aggregated local observation dictionary
        """
        return {
            "device_obs": device_obs,
            "grid_state": self.state.vector()
        }

    def act(self, observation: Observation, upstream_action: Any = None) -> None:
        """Compute coordination action and distribute to devices.

        Args:
            observation: Aggregated observation
            upstream_action: Pre-computed action (if any)

        Raises:
            NotImplementedError: If using decentralized mode (not yet implemented)
        """
        # Get coordinator action from policy if available
        if upstream_action is not None:
            action = upstream_action
        elif self.policy is not None:
            action = self.policy.forward(observation)
        else:
            raise RuntimeError("No action or policy provided for GridAgent.")

        self.coordinate_device(observation, action)
    
    # ============================================
    # Abstract Methods for Hierarchical Execution
    # ============================================

    def _derive_local_action(self, upstream_action: Optional[Any]) -> Optional[Any]:
        """Derive local action from upstream action.

        Currently, GridAgent is a pure coordinator with no local physical action.

        Args:
            upstream_action: Action received from upstream agent

        Returns:
            None - GridAgent has no local action to execute
        """
        return None

    def _derive_downstream_actions(
        self,
        upstream_action: Optional[Any]
    ) -> DictType[AgentID, Any]:
        """Derive actions for subordinates from upstream action.

        Decomposes the flat action vector from upstream (or from policy)
        into per-device actions by splitting based on device action dimensions.

        Args:
            upstream_action: Action received from upstream agent (flat numpy array or dict)

        Returns:
            Dict mapping subordinate IDs to their actions
        """
        downstream_actions = {}
        if not self.devices or upstream_action is None:
            return downstream_actions

        # Handle Dict action space (continuous + discrete)
        if isinstance(upstream_action, dict):
            # For Dict spaces with 'continuous' and 'discrete' keys
            if 'continuous' in upstream_action:
                action = np.asarray(upstream_action['continuous'])
            else:
                # Assume it's already a per-device dict
                return upstream_action
        else:
            action = np.asarray(upstream_action)

        # Decompose flat action vector into per-device actions
        offset = 0
        for agent_id, device in self.devices.items():
            # Get device action size
            action_size = device.action.dim_c + device.action.dim_d
            device_action = action[offset:offset + action_size]
            downstream_actions[agent_id] = device_action
            offset += action_size

        return downstream_actions

    def _execute_local_action(self, action: Optional[Any]) -> None:
        """Execute own action and update internal state.

        Subclasses should override this to implement their action execution.
        State updates should be published via _publish_state_updates().

        Args:
            action: Action to execute
        """
        # Currently, grid agent doesn't perform any local action
        pass

    # ============================================
    # State Update Hooks
    # ============================================

    def _update_state_with_upstream_info(self, upstream_info: Optional[DictType[str, Any]]) -> None:
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
    # Coordination Methods
    # ============================================

    def coordinate_device(self, observation: Observation, action: Any) -> None:
        """Coordinate device actions using the protocol.

        Args:
            observation: Current observation
            action: Computed action from coordinator
        """
        self.protocol.coordinate_action(self.devices, observation, action)
        self.protocol.coordinate_message(self.devices, observation, action)

    def get_device_actions(
        self,
        observations: DictType[AgentID, Observation],
    ) -> DictType[AgentID, Any]:
        """Get actions from all devices in decentralized mode.

        Args:
            observations: Dictionary mapping device IDs to observations

        Returns:
            Dictionary mapping device IDs to their computed actions

        Note:
            This function is intended for decentralized coordination where
            each device computes its own action independently.
        """
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = self.devices[agent_id].act(obs)
        return actions

    # ============================================
    # Utility Methods
    # ============================================

    def _consume_network_state(self) -> Optional[DictType[str, Any]]:
        """Consume network state from environment via message broker.

        Returns:
            Network state payload or None if no message available
        """
        if not self.message_broker or not self.env_id:
            return None

        channel = ChannelManager.power_flow_result_channel(self.env_id, self.agent_id)
        messages = self.message_broker.consume(
            channel,
            recipient_id=self.agent_id,
            env_id=self.env_id,
            clear=True
        )

        if messages:
            # Return the most recent message
            return messages[-1].payload

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
        message_broker: Optional[MessageBroker] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        grid_config: DictType[str, Any] = {},

        # PowerGridAgent specific params
        net: Optional[pp.pandapowerNet] = None,
    ):
        """Initialize power grid agent.

        Args:
            protocol: Coordination protocol
            policy: Optional centralized policy
            message_broker: Optional message broker for hierarchical execution
            upstream_id: Optional parent agent ID for hierarchical execution
            env_id: Optional environment ID for multi-environment isolation
            grid_config: Grid configuration dictionary
            net: PandaPower network object (required)
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
            message_broker=message_broker,
            upstream_id=upstream_id,
            env_id=env_id,
            grid_config=grid_config,
        )

    def _build_device_agents(
        self,
        device_configs: List[DictType[str, Any]],
        message_broker: Optional[MessageBroker] = None,
        env_id: Optional[str] = None,
        upstream_id: Optional[AgentID] = None,
    ) -> DictType[AgentID, DeviceAgent]:
        """Build device agents from configuration.

        Args:
            device_configs: List of device configuration dictionaries
            message_broker: Optional message broker for communication
            env_id: Environment ID for multi-environment isolation
            upstream_id: Upstream agent ID (this grid agent)

        Returns:
            Dictionary mapping device IDs to DeviceAgent instances
        """
        devices = {}
        for device_config in device_configs:
            device_type = device_config.get('type', None)

            if device_type == 'Generator':
                generator = Generator(
                    message_broker=message_broker,
                    upstream_id=upstream_id,
                    env_id=env_id,
                    device_config=device_config,
                )
                # Don't add to network here - that's done separately via _add_sgen()
                devices[generator.agent_id] = generator
            elif device_type == 'ESS':
                ess = ESS(
                    message_broker=message_broker,
                    upstream_id=upstream_id,
                    env_id=env_id,
                    device_config=device_config,
                )
                # Don't add to network here - that's done separately
                devices[ess.agent_id] = ess
            else:
                # Other device types not yet implemented
                pass

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
            self.devices[sgen.config.name] = sgen

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
            self.devices[ess.config.name] = ess

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

        # Update grid state from network results
        self._update_grid_state(net)

        local = super()._build_local_observation(device_obs)
        local['state'] = self._get_obs(net, device_obs)
        return local

    def _get_obs(self, net, device_obs=None):
        """Extract numerical observation vector from network state.

        Args:
            net: PandaPower network
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
        # P, Q at all buses
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

        Args:
            net: PandaPower network with power flow results
            t: Current timestep
        """
        for name, dg in self.sgen.items():
            local_ids = pp.get_element_index(net, 'sgen', self.name + ' ' + name)
            p_mw_val = net.res_sgen.loc[local_ids, 'p_mw']
            q_mvar_val = net.res_sgen.loc[local_ids, 'q_mvar']
            # Handle both scalar and array cases
            p_mw = p_mw_val if np.isscalar(p_mw_val) else p_mw_val.values[0]
            q_mvar = q_mvar_val if np.isscalar(q_mvar_val) else q_mvar_val.values[0]
            dg.electrical.P_MW = p_mw
            dg.electrical.Q_MVAr = q_mvar

    def update_cost_safety(self, net):
        """Update cost and safety metrics for the grid.

        This method computes both device-level costs (generation costs) and
        network-level safety metrics (voltage violations, line overloading).

        Execution modes:
        - Centralized (net != None): Directly accesses PandaPower network for
          voltage and line loading information
        - Distributed (net == None): Receives network state via messages from
          environment through message broker, ensuring agents never access the
          network object directly

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
            # Distributed mode: receive network state via messages
            if self.message_broker and self.env_id:
                network_state = self._consume_network_state()
                if network_state and network_state.get('converged', False):
                    # Extract pre-computed safety metrics from message
                    bus_voltages = network_state.get('bus_voltages', {})
                    line_loading = network_state.get('line_loading', {})

                    overvoltage = bus_voltages.get('overvoltage', 0)
                    undervoltage = bus_voltages.get('undervoltage', 0)
                    overloading = line_loading.get('overloading', 0)

                    self.safety += overloading + overvoltage + undervoltage

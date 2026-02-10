from typing import Any, Dict as DictType, Iterable, List, Optional, Union

import gymnasium as gym
import numpy as np
import pandapower as pp
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.base import Agent, AgentID
from heron.core.feature import FeatureProvider
from heron.core.observation import Observation
from powergrid.agents.device_agent import DeviceAgent
from heron.core.policies import Policy
from heron.protocols.base import NoProtocol, Protocol
from heron.scheduling.tick_config import TickConfig, JitterType
from powergrid.core.state.state import GridState
from powergrid.agents.generator import Generator
from powergrid.agents.storage import ESS
from powergrid.core.features.network import BusVoltages, LineFlows, NetworkMetrics


class PowerGridAgent(CoordinatorAgent):

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        features: List[FeatureProvider] = [],
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        subordinates: Optional[Dict[AgentID, "Agent"]] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None
    ):
        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            subordinates=subordinates,
            env_id=env_id,
            tick_config=tick_config,
            protocol=protocol,
            policy=policy
        )
        # if net is None:
        #     raise ValueError("PandaPower network 'net' must be provided to PowerGridAgent.")

        if grid_config is None:
            grid_config = {}

        # PowerGridAgent-specific initialization
        self.net = net
        self.name = net.name
        self.sgen: DictType[str, Generator] = {}
        self.storage: DictType[str, ESS] = {}
        self.base_power = grid_config.get("base_power", 1)
        self.load_scale = grid_config.get("load_scale", 1)
        self.load_rescaling(net, self.load_scale)

        # GridAgent initialization (merged)
        self.centralized = centralized
        self._grid_config = grid_config

        # Build config for CoordinatorAgent
        if devices is not None:
            self._init_devices = devices
            config = {'agents': []}
        else:
            self._init_devices = None
            config = {'agents': grid_config.get('devices', [])}

        super().__init__(
            agent_id=self.name,
            policy=policy,
            protocol=protocol,
            upstream_id=upstream_id,
            env_id=env_id,
            config=config,
            tick_config=tick_config,
        )

        # Use GridState instead of CoordinatorAgentState for power-grid domain
        self.state = GridState(
            owner_id=self.agent_id,
            owner_level=self.level
        )

        # If devices were provided directly, set them up now
        if self._init_devices is not None:
            self.subordinates = {
                device.agent_id: device for device in self._init_devices
            }
            for device in self._init_devices:
                device.upstream_id = self.agent_id

        # Initialize optional attributes
        self._cost: Optional[float] = None
        self._safety: Optional[float] = None
        self._cached_load_pq: Optional[np.ndarray] = None

    # ============================================
    # Subordinate Building (HERON Hook)
    # ============================================

    # def _build_subordinates(
    #     self,
    #     configs: List[DictType[str, Any]],
    #     env_id: Optional[str] = None,
    #     upstream_id: Optional[AgentID] = None,
    # ) -> DictType[AgentID, DeviceAgent]:
    #     if not configs:
    #         return {}

    #     devices = {}
    #     for config in configs:
    #         device_type = config.get('type', 'device').lower()
    #         device_id = config.get('name') or config.get('id')

    #         if device_type == 'generator':
    #             device = Generator(
    #                 agent_id=device_id,
    #                 env_id=env_id,
    #                 upstream_id=upstream_id,
    #                 generator_config=config,
    #             )
    #         elif device_type == 'ess' or device_type == 'storage':
    #             device = ESS(
    #                 agent_id=device_id,
    #                 env_id=env_id,
    #                 upstream_id=upstream_id,
    #                 ess_config=config,
    #             )
    #         else:
    #             device = DeviceAgent(
    #                 agent_id=device_id,
    #                 env_id=env_id,
    #                 upstream_id=upstream_id,
    #                 device_config=config,
    #             )

    #         devices[device.agent_id] = device

    #     return devices

    # ============================================
    # Backward Compatibility: devices <-> subordinates
    # ============================================

    # @property
    # def devices(self) -> DictType[AgentID, DeviceAgent]:
    #     return self.subordinates

    # @devices.setter
    # def devices(self, value: DictType[AgentID, DeviceAgent]) -> None:
    #     self.subordinates = value

    # ============================================
    # Cost/Safety Properties (Override to allow setting)
    # ============================================

    @property
    def cost(self) -> float:
        if self._cost is not None:
            return self._cost
        return sum(agent.cost for agent in self.devices.values())

    @cost.setter
    def cost(self, value: float) -> None:
        self._cost = value

    @property
    def safety(self) -> float:
        if self._safety is not None:
            return self._safety
        return sum(agent.safety for agent in self.devices.values())

    @safety.setter
    def safety(self, value: float) -> None:
        self._safety = value

    def get_reward(self) -> DictType[str, float]:
        return {"cost": self.cost, "safety": self.safety}

    # ============================================
    # Utility Methods
    # ============================================

    # def _consume_network_state(self) -> Optional[DictType[str, Any]]:
    #     messages = self.receive_messages(sender_id="proxy_agent", clear=True)
    #     for msg in messages:
    #         if 'converged' in msg or 'bus_voltages' in msg or 'line_loading' in msg:
    #             return msg
    #     return None

    # def __repr__(self) -> str:
    #     num_subs = len(self.devices)
    #     protocol_name = self.protocol.__class__.__name__
    #     return f"PowerGridAgent(id={self.agent_id}, devices={num_subs}, protocol={protocol_name})"



    # ============================================
    # Network Setup Methods
    # ============================================

    def _add_sgen(self, sgens: Iterable[Generator] | Generator):
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
        self._add_sgen(sgens)

    def add_storage(self, storages: Iterable[ESS] | ESS):
        self._add_storage(storages)

    def add_dataset(self, dataset):
        self.dataset = dataset

    def fuse_buses(self, ext_net, bus_name):
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
        try:
            local_load_ids = pp.get_element_index(net, 'load', self.name, False)
            if len(local_load_ids) > 0:
                net.load.loc[local_load_ids, 'scaling'] *= scale
        except (KeyError, ValueError):
            # No loads matching this agent's name, or names contain NaN
            pass

    # ============================================
    # Observation Methods
    # ============================================

    def observe(self, global_state: Optional[DictType[str, Any]] = None,
                visibility_level: str = 'system', **kwargs) -> Observation:
        obs = super().observe(global_state, **kwargs)

        # Apply visibility-based filtering to the state vector
        if visibility_level != 'system':
            # Store visibility level for use in _get_obs
            self._current_visibility_level = visibility_level
        else:
            self._current_visibility_level = 'system'

        return obs

    def _build_local_observation(self, device_obs: DictType[AgentID, Observation], *args, **kwargs) -> Any:
        return {
            "device_obs": device_obs,
            "grid_state": self.state.vector(),
            "state": self._get_obs(None, device_obs),
        }

    def _get_obs(self, net, device_obs=None):
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

        # Get current visibility level (set by observe())
        visibility_level = getattr(self, '_current_visibility_level', 'system')

        # P, Q at all buses - only include for system/upper_level visibility
        # (set via set_load_data() by the environment after power flow)
        if visibility_level in ('system', 'upper_level'):
            if self._cached_load_pq is not None:
                obs = np.concatenate([obs, self._cached_load_pq.ravel() / self.base_power])

        return obs.astype(np.float32)

    def set_load_data(self, load_pq: np.ndarray) -> None:
        self._cached_load_pq = load_pq

    def clear_load_data(self) -> None:
        self._cached_load_pq = None

    def _update_grid_state(self, net) -> None:
        if net is None or not net.get("converged", False):
            # If power flow didn't converge or no network, keep previous state
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
        return {
            device.agent_id: device.action_space
            for device in self.devices.values()
        }

    def get_grid_action_space(self):
        low_parts, high_parts, discrete_n = [], [], []
        for sp in self.get_device_action_spaces().values():
            if isinstance(sp, Box):
                low_parts.append(sp.low)
                high_parts.append(sp.high)
            elif isinstance(sp, Dict):
                # Handle Dict spaces (e.g., Generator with continuous + discrete)
                if 'continuous' in sp.spaces or 'c' in sp.spaces:
                    cont_space = sp.spaces.get('continuous', sp.spaces.get('c'))
                    low_parts.append(cont_space.low)
                    high_parts.append(cont_space.high)
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

        low = np.concatenate(low_parts) if low_parts else np.array([])
        high = np.concatenate(high_parts) if high_parts else np.array([])

        if len(low) and len(discrete_n):
            return Dict({"continuous": Box(low=low, high=high, dtype=np.float32),
                        'discrete': MultiDiscrete(discrete_n)})
        elif len(low):  # Continuous only
            return Box(low=low, high=high, dtype=np.float32)
        elif len(discrete_n):  # Discrete only
            return MultiDiscrete(discrete_n)
        else:  # No actionable agents
            return Discrete(1)

    def get_grid_observation_space(self, net=None, visibility_level: str = 'system'):
        local_net = self.net

        # Ensure powerflow has run on local network to get correct observation size
        try:
            pp.runpp(local_net, algorithm='nr', init='flat', max_iteration=100)
        except Exception:
            # If powerflow fails, still create space
            pass

        # Get device observation size from device state vectors
        # Each device's observation is its state.vector() (from observe().local['state'])
        device_obs_size = sum(
            device.state.vector().shape[0]
            for device in self.devices.values()
        )

        # Get load size from local network (before fusion)
        # Only include loads for system and upper_level visibility
        if visibility_level in ('system', 'upper_level'):
            try:
                local_load_ids = pp.get_element_index(local_net, 'load', self.name, False)
                load_obs_size = len(local_load_ids) * 2  # P and Q for each load
            except (KeyError, UserWarning):
                load_obs_size = 0
        else:
            # owner and public visibility: no load info
            load_obs_size = 0

        total_obs_size = device_obs_size + load_obs_size

        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )

    # ============================================
    # State Update Methods
    # ============================================

    def update_state(self, net, t):
        load_scaling = self.dataset['load'][t]

        local_ids = pp.get_element_index(net, 'load', self.name, False)
        net.load.loc[local_ids, 'scaling'] = load_scaling
        self.load_rescaling(net, self.load_scale)

        # Update all generators with their actions (apply_action updates internal state)
        for name, generator in self.sgen.items():
            generator.apply_action()

            # Update network with generator state
            local_ids = pp.get_element_index(net, 'sgen', self.name + ' ' + name)
            states = ['p_mw', 'q_mvar', 'in_service']
            # Ensure Q_MVAr is not None (use 0.0 if None)
            q_mvar = generator.electrical.Q_MVAr if generator.electrical.Q_MVAr is not None else 0.0
            values = [generator.electrical.P_MW, q_mvar, generator.status.in_service]
            net.sgen.loc[local_ids, states] = values

    def sync_global_state(self, net, t):
        if net is None:
            return

        # Centralized mode: sync from pandapower network
        # 1. Sync device states (sgen results)
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

        # 2. Cache load data for observations (agents should not access net directly)
        try:
            local_load_ids = pp.get_element_index(net, 'load', self.name, False)
            if len(local_load_ids) > 0:
                load_pq = net.res_load.loc[local_load_ids].values
                self.set_load_data(load_pq)
            else:
                self.set_load_data(np.array([]))
        except (KeyError, UserWarning):
            # No loads for this agent
            self.set_load_data(np.array([]))

        # 3. Update grid state features (for safety/cost calculations)
        self._update_grid_state(net)

    def _update_from_network_state(self, network_state: DictType[str, Any]) -> None:
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

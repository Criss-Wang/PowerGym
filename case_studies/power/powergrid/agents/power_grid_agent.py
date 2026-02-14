"""Power Grid Coordinator Agent following grid_age style.

This module implements a coordinator agent for power grid simulations with:
- Direct constructor with pre-initialized device agents
- PandaPower network integration
- Cost/safety aggregation from subordinates
"""

from typing import Any, Dict as DictType, Iterable, List, Optional

import gymnasium as gym
import numpy as np
import pandapower as pp
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.base import Agent
from heron.core.feature import FeatureProvider
from heron.core.observation import Observation
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig
from heron.utils.typing import AgentID
from powergrid.agents.device_agent import DeviceAgent
from powergrid.agents.generator import Generator
from powergrid.agents.storage import ESS
from powergrid.core.features.network import BusVoltages, LineFlows, NetworkMetrics
from heron.core.state import CoordinatorAgentState


class PowerGridAgent(CoordinatorAgent):
    """Power grid coordinator managing device agents.

    Follows the grid_age coordinator pattern:
    - Takes pre-initialized device agents as subordinates
    - Aggregates cost/safety from subordinates
    - Manages PandaPower network integration

    Example:
        >>> devices = {
        ...     "gen_1": Generator(agent_id="gen_1", bus="bus_1", ...),
        ...     "ess_1": ESS(agent_id="ess_1", bus="bus_2", ...),
        ... }
        >>> grid = PowerGridAgent(
        ...     agent_id="grid_1",
        ...     subordinates=devices,
        ...     net=pp_net,
        ... )
    """

    def __init__(
        self,
        agent_id: AgentID,
        subordinates: DictType[AgentID, DeviceAgent],
        net: Optional[Any] = None,  # PandaPower network
        features: List[FeatureProvider] = [],
        # Grid configuration
        base_power: float = 1.0,
        load_scale: float = 1.0,
        # Hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        tick_config: Optional[TickConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        """Initialize power grid coordinator.

        Args:
            agent_id: Unique identifier
            subordinates: Pre-initialized device agents (REQUIRED)
            net: PandaPower network object
            features: Coordinator-level features
            base_power: Base power for normalization
            load_scale: Load scaling factor
            upstream_id: Parent agent ID
            env_id: Environment ID
            tick_config: Timing configuration
            policy: Coordinator policy
            protocol: Coordination protocol
        """
        if subordinates is None or len(subordinates) == 0:
            raise ValueError(
                f"PowerGridAgent requires subordinates. "
                f"Create device agents externally and pass as subordinates dict."
            )

        # Store parameters
        self.net = net
        self.base_power = base_power
        self.load_scale = load_scale
        self._cached_load_pq: Optional[np.ndarray] = None
        self._cost: Optional[float] = None
        self._safety: Optional[float] = None

        # Organize devices by type
        self.sgen: DictType[str, Generator] = {}
        self.storage: DictType[str, ESS] = {}
        for device_id, device in subordinates.items():
            if isinstance(device, Generator):
                self.sgen[device_id] = device
            elif isinstance(device, ESS):
                self.storage[device_id] = device

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            subordinates=subordinates,
            env_id=env_id,
            tick_config=tick_config,
            policy=policy,
            protocol=protocol,
        )

        # Use CoordinatorAgentState for power-grid domain
        self.state = CoordinatorAgentState(
            owner_id=self.agent_id,
            owner_level=self.level
        )

        # Apply load scaling if network provided
        if net is not None and self.name:
            self._apply_load_scaling(net, self.load_scale)

    @property
    def name(self) -> str:
        """Get grid name."""
        return self.agent_id

    @property
    def devices(self) -> DictType[AgentID, DeviceAgent]:
        """Alias for subordinates."""
        return self.subordinates

    @property
    def cost(self) -> float:
        """Aggregate cost from all devices."""
        if self._cost is not None:
            return self._cost
        return sum(agent.cost for agent in self.devices.values())

    @cost.setter
    def cost(self, value: float) -> None:
        self._cost = value

    @property
    def safety(self) -> float:
        """Aggregate safety penalty from all devices."""
        if self._safety is not None:
            return self._safety
        return sum(agent.safety for agent in self.devices.values())

    @safety.setter
    def safety(self, value: float) -> None:
        self._safety = value

    def get_reward(self) -> DictType[str, float]:
        """Get cost and safety metrics."""
        return {"cost": self.cost, "safety": self.safety}

    def _apply_load_scaling(self, net, scale: float) -> None:
        """Apply load scaling to network loads matching this grid."""
        try:
            local_load_ids = pp.get_element_index(net, 'load', self.name, False)
            if len(local_load_ids) > 0:
                net.load.loc[local_load_ids, 'scaling'] *= scale
        except (KeyError, ValueError):
            pass

    def set_load_data(self, load_pq: np.ndarray) -> None:
        """Cache load data for observations."""
        self._cached_load_pq = load_pq

    def clear_load_data(self) -> None:
        """Clear cached load data."""
        self._cached_load_pq = None

    def observe(
        self,
        global_state: Optional[DictType[str, Any]] = None,
        visibility_level: str = 'system',
        **kwargs
    ) -> Observation:
        """Get observation with visibility filtering."""
        obs = super().observe(global_state, **kwargs)
        self._current_visibility_level = visibility_level
        return obs

    def _build_local_observation(
        self,
        device_obs: DictType[AgentID, Observation],
        *args,
        **kwargs
    ) -> Any:
        """Build local observation from device states."""
        return {
            "device_obs": device_obs,
            "grid_state": self.state.vector(),
            "state": self._get_obs(device_obs),
        }

    def _get_obs(self, device_obs: Optional[DictType[AgentID, Observation]] = None) -> np.ndarray:
        """Build flattened observation vector."""
        if device_obs is None:
            device_obs = {
                agent_id: agent.observe()
                for agent_id, agent in self.devices.items()
            }

        obs = np.array([])
        for _, ob in device_obs.items():
            obs = np.concatenate((obs, ob.local['state']))

        visibility_level = getattr(self, '_current_visibility_level', 'system')

        # Include load data for system/upper_level visibility
        if visibility_level in ('system', 'upper_level'):
            if self._cached_load_pq is not None:
                obs = np.concatenate([obs, self._cached_load_pq.ravel() / self.base_power])

        return obs.astype(np.float32)

    def update_state(self, net, t: int) -> None:
        """Update network state from device actions."""
        # Update all generators
        for name, generator in self.sgen.items():
            generator.apply_action()

            if net is not None:
                try:
                    local_ids = pp.get_element_index(net, 'sgen', self.name + ' ' + name)
                    q_mvar = generator.electrical.Q_MVAr or 0.0
                    net.sgen.loc[local_ids, ['p_mw', 'q_mvar', 'in_service']] = [
                        generator.electrical.P_MW,
                        q_mvar,
                        generator.status.in_service if generator.status else True
                    ]
                except (KeyError, ValueError):
                    pass

        # Update all storage
        for name, ess in self.storage.items():
            ess.apply_action()

            if net is not None:
                try:
                    local_ids = pp.get_element_index(net, 'storage', self.name + ' ' + name)
                    net.storage.loc[local_ids, ['p_mw', 'soc_percent']] = [
                        ess.electrical.P_MW,
                        ess.storage.soc * 100
                    ]
                except (KeyError, ValueError):
                    pass

    def sync_global_state(self, net, t: int) -> None:
        """Sync state from power flow results."""
        if net is None:
            return

        # Sync generator results
        for name, dg in self.sgen.items():
            try:
                local_ids = pp.get_element_index(net, 'sgen', self.name + ' ' + name)
                p_mw_val = net.res_sgen.loc[local_ids, 'p_mw']
                q_mvar_val = net.res_sgen.loc[local_ids, 'q_mvar']
                p_mw = p_mw_val if np.isscalar(p_mw_val) else p_mw_val.values[0]
                q_mvar = q_mvar_val if np.isscalar(q_mvar_val) else q_mvar_val.values[0]
                if not np.isnan(p_mw):
                    dg.electrical.P_MW = p_mw
                if not np.isnan(q_mvar):
                    dg.electrical.Q_MVAr = q_mvar
            except (KeyError, ValueError):
                pass

        # Cache load data
        try:
            local_load_ids = pp.get_element_index(net, 'load', self.name, False)
            if len(local_load_ids) > 0:
                self.set_load_data(net.res_load.loc[local_load_ids].values)
            else:
                self.set_load_data(np.array([]))
        except (KeyError, UserWarning):
            self.set_load_data(np.array([]))

        # Update grid state features
        self._update_grid_state(net)

    def _update_grid_state(self, net) -> None:
        """Update grid state features from network results."""
        if net is None or not net.get("converged", False):
            return

        try:
            local_bus_ids = pp.get_element_index(net, 'bus', self.name, False)
            if len(local_bus_ids) == 0:
                return

            bus_vm = net.res_bus.loc[local_bus_ids, 'vm_pu'].values
            bus_va = net.res_bus.loc[local_bus_ids, 'va_degree'].values
            bus_names = net.bus.loc[local_bus_ids, 'name'].tolist()

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

            local_sgen_ids = pp.get_element_index(net, 'sgen', self.name, False)
            local_load_ids = pp.get_element_index(net, 'load', self.name, False)

            total_gen_mw = net.res_sgen.loc[local_sgen_ids, 'p_mw'].sum() if len(local_sgen_ids) > 0 else 0.0
            total_gen_mvar = net.res_sgen.loc[local_sgen_ids, 'q_mvar'].sum() if len(local_sgen_ids) > 0 else 0.0
            total_load_mw = net.res_load.loc[local_load_ids, 'p_mw'].sum() if len(local_load_ids) > 0 else 0.0
            total_load_mvar = net.res_load.loc[local_load_ids, 'q_mvar'].sum() if len(local_load_ids) > 0 else 0.0
            total_loss_mw = total_gen_mw - total_load_mw

            self.state.features = [
                BusVoltages(vm_pu=bus_vm, va_deg=bus_va, bus_names=bus_names),
                LineFlows(p_from_mw=line_p, q_from_mvar=line_q, loading_percent=line_loading, line_names=line_names),
                NetworkMetrics(
                    total_gen_mw=float(total_gen_mw),
                    total_load_mw=float(total_load_mw),
                    total_loss_mw=float(total_loss_mw),
                    total_gen_mvar=float(total_gen_mvar),
                    total_load_mvar=float(total_load_mvar),
                ),
            ]
        except (KeyError, ValueError):
            pass

    def update_cost_safety(self, net=None) -> None:
        """Update cost and safety from devices and network."""
        self.cost = 0.0
        self.safety = 0.0

        # Device costs
        for dg in self.sgen.values():
            self.cost += dg.cost
            self.safety += dg.safety

        for ess in self.storage.values():
            self.cost += ess.cost
            self.safety += ess.safety

        # Network violations
        if net is not None and net.get("converged", False):
            try:
                local_bus_ids = pp.get_element_index(net, 'bus', self.name, False)
                local_vm = net.res_bus.loc[local_bus_ids].vm_pu.values
                overvoltage = np.maximum(local_vm - 1.05, 0).sum()
                undervoltage = np.maximum(0.95 - local_vm, 0).sum()

                local_line_ids = pp.get_element_index(net, 'line', self.name, False)
                local_line_loading = net.res_line.loc[local_line_ids].loading_percent.values
                overloading = np.maximum(local_line_loading - 100, 0).sum() * 0.01

                self.safety += overloading + overvoltage + undervoltage
            except (KeyError, ValueError):
                pass

    def get_device_action_spaces(self) -> DictType[str, gym.Space]:
        """Get action spaces for all devices."""
        return {
            device.agent_id: device.action_space
            for device in self.devices.values()
        }

    def get_grid_action_space(self) -> gym.Space:
        """Get combined action space for the grid."""
        low_parts, high_parts, discrete_n = [], [], []

        for sp in self.get_device_action_spaces().values():
            if isinstance(sp, Box):
                low_parts.append(sp.low)
                high_parts.append(sp.high)
            elif isinstance(sp, Dict):
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
            return Dict({
                "continuous": Box(low=low, high=high, dtype=np.float32),
                'discrete': MultiDiscrete(discrete_n)
            })
        elif len(low):
            return Box(low=low, high=high, dtype=np.float32)
        elif len(discrete_n):
            return MultiDiscrete(discrete_n)
        else:
            return Discrete(1)

    def get_grid_observation_space(self, visibility_level: str = 'system') -> Box:
        """Get observation space for the grid."""
        device_obs_size = sum(
            device.state.vector().shape[0]
            for device in self.devices.values()
        )

        load_obs_size = 0
        if visibility_level in ('system', 'upper_level') and self.net is not None:
            try:
                local_load_ids = pp.get_element_index(self.net, 'load', self.name, False)
                load_obs_size = len(local_load_ids) * 2
            except (KeyError, UserWarning):
                pass

        total_obs_size = device_obs_size + load_obs_size

        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )

    def __repr__(self) -> str:
        num_devices = len(self.devices)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"PowerGridAgent(id={self.agent_id}, devices={num_devices}, protocol={protocol_name})"

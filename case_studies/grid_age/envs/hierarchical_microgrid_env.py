"""Hierarchical multi-agent microgrid environment.

This module implements a fully hierarchical version of the microgrid environment:
- SystemAgent → MicrogridCoordinatorAgents → DeviceFieldAgents

Each device is its own field agent, and microgrids are coordinator agents.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandapower as pp

from heron.envs.base import MultiAgentEnv
from heron.agents.system_agent import SystemAgent
from case_studies.grid_age.envs.common import EnvState
from case_studies.grid_age.agents import (
    MicrogridCoordinatorAgent,
    ESSFieldAgent,
    DGFieldAgent,
    RESFieldAgent,
)
from case_studies.grid_age.utils.data_loader import get_data_loader


class HierarchicalMicrogridEnv(MultiAgentEnv):
    def __init__(
        self,
        system_agent: SystemAgent,  # REQUIRED: Always pass pre-initialized system agent
        episode_steps: int = 24,
        dt: float = 1.0,
        **kwargs
    ):
        self.episode_steps = episode_steps
        self.dt = dt
        self.num_microgrids = len(system_agent.subordinates)

        print(f"Initialized with {self.num_microgrids} microgrids")

        # Initialize episode statistics
        self._episode = 0
        self._timestep = 0
        self.data_loader = get_data_loader(split='train')
        self.current_episode_data = self.data_loader.get_episode_data(
            episode=self._episode,
            episode_length=self.episode_steps,
        )

        # Call parent init (registers agents)
        super().__init__(
            system_agent=system_agent,
            **kwargs
        )

        # Network will be created during first reset() when global state is available
        self.net = None

    def _create_network_from_agents(self, global_state: Dict) -> pp.pandapowerNet:
        """Create Pandapower network from registered agents.

        Note: Called during first reset after agents are registered.
        Uses agent structure to build network topology.

        Args:
            global_state: Global state dict (unused, for signature compatibility)

        Returns:
            Pandapower network matching agent structure
        """
        net = pp.create_empty_network(name="Hierarchical Microgrid Network")

        # Create main grid bus
        slack_bus = pp.create_bus(net, vn_kv=12.47, name="Grid")
        pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0)

        # Build network from registered microgrid coordinators
        microgrids = {}
        for agent_id, agent in self.registered_agents.items():
            if isinstance(agent, MicrogridCoordinatorAgent):
                microgrids[agent_id] = agent

        print(f"  Creating network for {len(microgrids)} microgrids")

        # Build network from microgrids
        for mg_id, mg_agent in microgrids.items():
            # Create microgrid bus
            mg_bus = pp.create_bus(net, vn_kv=0.48, name=f"{mg_id}_Bus")

            # Connect to grid
            pp.create_transformer_from_parameters(
                net, hv_bus=slack_bus, lv_bus=mg_bus,
                sn_mva=2.5, vn_hv_kv=12.47, vn_lv_kv=0.48,
                vkr_percent=1.0, vk_percent=5.0,
                pfe_kw=0.1, i0_percent=0.1,
                name=f"{mg_id}_Trafo"
            )

            # Get device agent IDs from coordinator's subordinates
            subordinate_ids = list(mg_agent.subordinates.keys()) if hasattr(mg_agent, 'subordinates') else []

            if not subordinate_ids:
                continue

            # Create devices based on subordinate agents
            for dev_id in subordinate_ids:
                dev_agent = mg_agent.subordinates.get(dev_id)
                if not dev_agent or not hasattr(dev_agent, 'state'):
                    continue

                # Get feature names from agent's state features
                feature_names = [f.feature_name for f in dev_agent.state.features]

                # Feature-based device detection
                if "SOCFeature" in feature_names:
                    # Storage device - get capacity from ESS feature
                    soc_feature = next((f for f in dev_agent.state.features if f.feature_name == "SOCFeature"), None)
                    capacity = soc_feature.capacity if soc_feature else 1.0
                    pp.create_storage(
                        net, bus=mg_bus, p_mw=0.0, max_e_mwh=capacity,
                        name=dev_id, controllable=True
                    )

                elif "UnitCommitmentFeature" in feature_names:
                    # Dispatchable generator - get max_p from power feature
                    power_feature = next((f for f in dev_agent.state.features if f.feature_name == "PowerFeature"), None)
                    max_p = power_feature.max_p if power_feature else 1.0
                    pp.create_sgen(
                        net, bus=mg_bus, p_mw=max_p * 0.5, q_mvar=0.0,
                        name=dev_id, controllable=True
                    )

                elif "AvailabilityFeature" in feature_names:
                    # Renewable generator - get max_p from power feature
                    power_feature = next((f for f in dev_agent.state.features if f.feature_name == "PowerFeature"), None)
                    max_p = power_feature.max_p if power_feature else 0.1
                    pp.create_sgen(
                        net, bus=mg_bus, p_mw=0.0, q_mvar=0.0,
                        name=dev_id, controllable=True
                    )

            # Add load
            pp.create_load(net, bus=mg_bus, p_mw=0.2, q_mvar=0.05, name=f"{mg_id}_Load")

        print(f"  ✅ Network: {len(net.bus)} buses, {len(net.storage)} storage, {len(net.sgen)} sgen")

        return net


    def _update_profiles(self, timestep: int) -> None:
        """Update profiles for current timestep using real data."""
        if self.current_episode_data is None:
            raise ValueError("Episode data not loaded. Call reset() to load episode data before updating profiles.")

        hour = timestep % len(self.current_episode_data['price'])

        current_price = float(self.current_episode_data['price'][hour])
        pv_avail = float(self.current_episode_data['solar'][hour])
        wind_avail = float(self.current_episode_data['wind'][hour])

        # Get microgrid coordinator IDs from global state
        # Use system agent ID to get full global state visibility
        from heron.agents.system_agent import SYSTEM_AGENT_ID
        global_state = self.proxy_agent.get_global_states(sender_id=SYSTEM_AGENT_ID, protocol=None)
        agent_states = global_state if isinstance(global_state, dict) else {}

        """
        TODO: 
        We shall never directly update the states of individual agents inside the env. 
        We shall only set the individual agents' local state in proxy via set_local_state
        Then, agents will retrieve from proxy
        a. observations when they compute their actions
        b. local states when they compute reward or when the execute/tick happens 
        """
        from case_studies.grid_age.agents import MicrogridCoordinatorAgent
        for agent_id, agent_state in agent_states.items():
            metadata = agent_state.get("metadata", {})
            agent_type = metadata.get("agent_type", "")

            # Check if this is a microgrid coordinator
            if "MicrogridCoordinator" in agent_type or "microgrid" in agent_id.lower():
                # Get the actual agent object to call methods on it
                agent = self.registered_agents.get(agent_id)
                if agent and isinstance(agent, MicrogridCoordinatorAgent):
                    agent.set_grid_price(current_price)
                    agent.set_renewable_availability(pv_avail, wind_avail)

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        """Convert global state from proxy to custom env state for simulation.

        Extracts device setpoints from agent states.

        Args:
            global_state: Dict from proxy with structure:
                {"agent_states": {agent_id: state_dict, ...}}

        Returns:
            EnvState with device setpoints for power flow
        """
        env_state = EnvState()

        agent_states = global_state.get("agent_states", {})

        # Extract device setpoints from PowerFeatures
        for agent_id, state_dict in agent_states.items():
            features = state_dict.get("features", {})

            # Extract power setpoints from PowerFeature (used in hierarchical devices)
            if "PowerFeature" in features:
                power_feature = features["PowerFeature"]
                env_state.set_device_setpoint(
                    agent_id,
                    P=power_feature.get("P", 0.0),
                    Q=power_feature.get("Q", 0.0),
                )

        return env_state

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Run Pandapower AC power flow simulation.

        Args:
            env_state: EnvState with device setpoints

        Returns:
            Updated EnvState with power flow results
        """
        device_setpoints = env_state.device_setpoints

        # Update Pandapower network with device setpoints
        for device_id, setpoint in device_setpoints.items():
            P = setpoint.get("P", 0.0)
            Q = setpoint.get("Q", 0.0)

            # Update storage devices
            storage_idx = self.net.storage[self.net.storage.name == device_id]
            if len(storage_idx) > 0:
                self.net.storage.at[storage_idx.index[0], "p_mw"] = P
                self.net.storage.at[storage_idx.index[0], "q_mvar"] = Q
                continue

            # Update generator devices
            sgen_idx = self.net.sgen[self.net.sgen.name == device_id]
            if len(sgen_idx) > 0:
                self.net.sgen.at[sgen_idx.index[0], "p_mw"] = P
                self.net.sgen.at[sgen_idx.index[0], "q_mvar"] = Q

        # Run AC power flow
        try:
            pp.runpp(self.net, algorithm="nr", calculate_voltage_angles=True)
            converged = True
        except Exception as e:
            print(f"Warning: Power flow failed: {e}")
            converged = False

        # Extract power flow results and update env_state
        if converged:
            results = {
                "converged": True,
                "voltage_min": float(self.net.res_bus.vm_pu.min()),
                "voltage_max": float(self.net.res_bus.vm_pu.max()),
                "voltage_avg": float(self.net.res_bus.vm_pu.mean()),
                "max_line_loading": float(self.net.res_line.loading_percent.max() / 100.0) if len(self.net.res_line) > 0 else 0.0,
                "grid_power": float(self.net.res_ext_grid.p_mw.values[0]) if len(self.net.res_ext_grid) > 0 else 0.0,
            }
        else:
            results = {
                "converged": False,
                "voltage_min": 1.0,
                "voltage_max": 1.0,
                "voltage_avg": 1.0,
                "max_line_loading": 0.0,
                "grid_power": 0.0,
            }

        env_state.update_power_flow_results(results)
        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        """Convert simulation results back to global state format.

        Updates agent states with power flow results.

        Args:
            env_state: EnvState with power flow results

        Returns:
            Updated global state dict:
                {"agent_states": {agent_id: updated_state_dict, ...}}
        """
        power_flow_results = env_state.power_flow_results

        # Get current global state from proxy
        from heron.agents.system_agent import SYSTEM_AGENT_ID
        global_state = self.proxy_agent.get_global_states(sender_id=SYSTEM_AGENT_ID, protocol=None)
        agent_states = global_state if isinstance(global_state, dict) else {}

        # Update microgrid coordinator voltage features with power flow results
        for agent_id, agent_state in agent_states.items():
            metadata = agent_state.get("metadata", {})
            agent_type = metadata.get("agent_type", "")

            # Check if this is a microgrid coordinator
            if "MicrogridCoordinator" in agent_type or "microgrid" in agent_id.lower():
                # Update voltage feature with power flow results
                if "features" in agent_state and "VoltageFeature" in agent_state["features"]:
                    agent_state["features"]["VoltageFeature"]["voltage"] = power_flow_results.get("voltage_avg", 1.0)

        return {"agent_states": agent_states}

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        """Reset environment."""
        self._episode += 1
        self._timestep = 0

        # Load episode-specific data
        self.current_episode_data = self.data_loader.get_episode_data(
            episode=self._episode,
            episode_length=self.episode_steps,
        )

        self._update_profiles(0)
        obs, info = super().reset(seed=seed, **kwargs)

        # Create network on first reset using global state
        if self.net is None:
            from heron.agents.system_agent import SYSTEM_AGENT_ID
            global_state = self.proxy_agent.get_global_states(sender_id=SYSTEM_AGENT_ID, protocol=None)
            self.net = self._create_network_from_agents(global_state)

        return obs, info


    """
    TODO: This impelementation seems unncessary:
    - self._update_profiles(self._timestep) shoud be done inside run_simulation
    - the part below can be done inside system_agent.execute
        self._timestep += 1
        if self._timestep >= self.episode_steps:
            terminated = {aid: True for aid in terminated}
            terminated["__all__"] = True
    """
    def step(self, actions: Dict[str, Any]):
        """Execute step."""
        self._update_profiles(self._timestep) 
        obs, rewards, terminated, truncated, infos = super().step(actions)


        self._timestep += 1
        if self._timestep >= self.episode_steps:
            terminated = {aid: True for aid in terminated}
            terminated["__all__"] = True

        return obs, rewards, terminated, truncated, infos
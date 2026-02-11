"""Multi-agent microgrid environment with Pandapower integration.

This module implements the GridAges multi-microgrid environment using Heron's
MultiAgentEnv framework with hierarchical agent control and AC power flow simulation.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandapower as pp
import pandapower.networks as pn

from heron.envs.base import MultiAgentEnv
from heron.agents.system_agent import SystemAgent
from case_studies.grid_age.agents import MicrogridFieldAgent


class EnvState:
    """Custom environment state for power flow simulation.

    Stores device setpoints and power flow results for each microgrid.
    """

    def __init__(self, num_microgrids: int = 3):
        self.num_microgrids = num_microgrids
        self.device_setpoints = {}  # {mg_id: {device: {P, Q, ...}}}
        self.power_flow_results = {}  # {mg_id: {voltages, line_flows, ...}}
        self.converged = False

    def update_setpoints(self, mg_id: str, device_states: Dict[str, Dict]) -> None:
        """Update device setpoints for a microgrid."""
        self.device_setpoints[mg_id] = device_states

    def update_power_flow_results(self, results: Dict) -> None:
        """Update power flow results."""
        self.power_flow_results = results
        self.converged = results.get("converged", False)


class MicrogridEnv(MultiAgentEnv):
    """Multi-agent microgrid environment with networked topology.

    Simulates multiple microgrids connected to a distribution system operator (DSO).
    Each microgrid is controlled by a MicrogridFieldAgent that manages:
    - Energy Storage System (ESS)
    - Distributed Generator (DG)
    - Solar PV and Wind turbines
    - Grid connection

    The environment uses Pandapower for AC power flow simulation to model
    network physics including voltage constraints and line flows.

    Attributes:
        num_microgrids: Number of microgrids in the system
        episode_steps: Number of time steps per episode (default: 24 hours)
        dt: Time step in hours (default: 1.0)
        net: Pandapower network object
        voltage_limits: (min, max) voltage limits in pu
        line_loading_limit: Maximum line loading in pu
    """

    def __init__(
        self,
        num_microgrids: int = 3,
        episode_steps: int = 24,
        dt: float = 1.0,
        voltage_limits: Tuple[float, float] = (0.95, 1.05),
        line_loading_limit: float = 1.0,
        # Environment configuration
        scheduler_config: Optional[Dict[str, Any]] = None,
        message_broker_config: Optional[Dict[str, Any]] = None,
        simulation_wait_interval: Optional[float] = None,
        **kwargs
    ):
        """Initialize microgrid environment.

        Args:
            num_microgrids: Number of microgrids (default: 3)
            episode_steps: Episode length in time steps (default: 24)
            dt: Time step duration in hours (default: 1.0)
            voltage_limits: (min_pu, max_pu) voltage constraints
            line_loading_limit: Maximum line loading fraction
            scheduler_config: Configuration for event scheduler
            message_broker_config: Configuration for message broker
            simulation_wait_interval: Wait time for simulation events
            **kwargs: Additional arguments for MultiAgentEnv
        """
        self.num_microgrids = num_microgrids
        self.episode_steps = episode_steps
        self.dt = dt
        self.voltage_limits = voltage_limits
        self.line_loading_limit = line_loading_limit

        # Create microgrid agents with different DG capacities
        # (matching GridAges: MG1=0.66MW, MG2=0.60MW, MG3=0.50MW)
        dg_capacities = {
            1: 0.66,
            2: 0.60,
            3: 0.50,
        }

        mg_agents = {}
        for i in range(1, num_microgrids + 1):
            mg_agents[f"MG{i}"] = MicrogridFieldAgent(
                agent_id=f"MG{i}",
                ess_capacity=2.0,
                ess_min_p=-0.5,
                ess_max_p=0.5,
                dg_max_p=dg_capacities.get(i, 0.66),
                dg_min_p=0.1,
                pv_max_p=0.1,
                wind_max_p=0.1,
            )

        # Create system agent (top-level coordinator)
        # Note: agent_id must be "system_agent" for event-driven mode
        system_agent = SystemAgent(
            agent_id="system_agent",  # Required by scheduler
            subordinates=mg_agents,
        )

        # Initialize Pandapower network
        self.net = self._create_network(num_microgrids)

        # Store bus mappings for each microgrid
        self.mg_bus_mappings = self._create_bus_mappings()

        # Initialize episode counter and timestep
        self._episode = 0
        self._timestep = 0

        # Initialize price and renewable profiles
        self._initialize_profiles()

        # Call parent constructor
        super().__init__(
            system_agent=system_agent,
            scheduler_config=scheduler_config,
            message_broker_config=message_broker_config,
            simulation_wait_interval=simulation_wait_interval,
            **kwargs
        )

    def _create_network(self, num_microgrids: int) -> pp.pandapowerNet:
        """Create networked microgrid topology using Pandapower.

        Creates a simplified network structure:
        - Main DSO grid (simplified from IEEE 34-bus)
        - Multiple microgrids (simplified from IEEE 13-bus)
        - Connection points between DSO and microgrids

        Args:
            num_microgrids: Number of microgrids to create

        Returns:
            Pandapower network object
        """
        # Create main network
        net = pp.create_empty_network(name="GridAges Multi-Microgrid")

        # Create DSO buses (simplified topology)
        dso_slack_bus = pp.create_bus(net, vn_kv=12.47, name="DSO_Slack")
        pp.create_ext_grid(net, bus=dso_slack_bus, vm_pu=1.0)

        # Create microgrid connection buses on DSO
        dso_mg_buses = []
        for i in range(num_microgrids):
            bus = pp.create_bus(net, vn_kv=12.47, name=f"DSO_MG{i+1}_Connection")
            dso_mg_buses.append(bus)
            # Connect to slack via line
            pp.create_line(net, from_bus=dso_slack_bus, to_bus=bus,
                           length_km=1.0, std_type="NAYY 4x50 SE")

        # Create each microgrid
        for i, dso_bus in enumerate(dso_mg_buses):
            self._create_microgrid(net, i + 1, dso_bus)

        return net

    def _create_microgrid(self, net: pp.pandapowerNet, mg_id: int,
                          connection_bus: int) -> None:
        """Create a single microgrid with devices.

        Args:
            net: Pandapower network
            mg_id: Microgrid ID (1, 2, 3, ...)
            connection_bus: DSO bus to connect to
        """
        # Microgrid main bus
        mg_main_bus = pp.create_bus(net, vn_kv=0.48, name=f"MG{mg_id}_Main")

        # Transformer connecting to DSO
        pp.create_transformer_from_parameters(
            net,
            hv_bus=connection_bus,
            lv_bus=mg_main_bus,
            sn_mva=2.5,
            vn_hv_kv=12.47,
            vn_lv_kv=0.48,
            vkr_percent=1.0,
            vk_percent=5.0,
            pfe_kw=0.1,
            i0_percent=0.1,
            name=f"MG{mg_id}_Trafo"
        )

        # Create device buses
        ess_bus = pp.create_bus(net, vn_kv=0.48, name=f"MG{mg_id}_ESS_Bus")
        dg_bus = pp.create_bus(net, vn_kv=0.48, name=f"MG{mg_id}_DG_Bus")
        res_bus = pp.create_bus(net, vn_kv=0.48, name=f"MG{mg_id}_RES_Bus")
        load_bus = pp.create_bus(net, vn_kv=0.48, name=f"MG{mg_id}_Load_Bus")

        # Connect device buses to main bus
        for device_bus in [ess_bus, dg_bus, res_bus, load_bus]:
            pp.create_line(net, from_bus=mg_main_bus, to_bus=device_bus,
                           length_km=0.1, std_type="NAYY 4x50 SE")

        # Create ESS as storage
        pp.create_storage(
            net,
            bus=ess_bus,
            p_mw=0.0,
            max_e_mwh=2.0,
            q_mvar=0.0,
            name=f"MG{mg_id}_ESS",
            controllable=True,
        )

        # Create DG as static generator
        pp.create_sgen(
            net,
            bus=dg_bus,
            p_mw=0.1,
            q_mvar=0.0,
            name=f"MG{mg_id}_DG",
            controllable=True,
        )

        # Create PV as static generator
        pp.create_sgen(
            net,
            bus=res_bus,
            p_mw=0.0,
            q_mvar=0.0,
            name=f"MG{mg_id}_PV",
            controllable=True,
        )

        # Create Wind as static generator
        pp.create_sgen(
            net,
            bus=res_bus,
            p_mw=0.0,
            q_mvar=0.0,
            name=f"MG{mg_id}_Wind",
            controllable=True,
        )

        # Create load
        pp.create_load(
            net,
            bus=load_bus,
            p_mw=0.2,  # Base load
            q_mvar=0.05,
            name=f"MG{mg_id}_Load"
        )

    def _create_bus_mappings(self) -> Dict[str, Dict[str, int]]:
        """Create mappings from microgrid IDs to Pandapower element indices.

        Returns:
            Dict mapping MG_ID to device indices
        """
        mappings = {}

        for mg_id in range(1, self.num_microgrids + 1):
            mg_name = f"MG{mg_id}"

            # Find indices by name
            ess_idx = self.net.storage[self.net.storage.name == f"{mg_name}_ESS"].index[0]
            dg_idx = self.net.sgen[self.net.sgen.name == f"{mg_name}_DG"].index[0]
            pv_idx = self.net.sgen[self.net.sgen.name == f"{mg_name}_PV"].index[0]
            wind_idx = self.net.sgen[self.net.sgen.name == f"{mg_name}_Wind"].index[0]
            load_idx = self.net.load[self.net.load.name == f"{mg_name}_Load"].index[0]

            mappings[mg_name] = {
                "ess": ess_idx,
                "dg": dg_idx,
                "pv": pv_idx,
                "wind": wind_idx,
                "load": load_idx,
            }

        return mappings

    def _initialize_profiles(self) -> None:
        """Initialize price and renewable availability profiles.

        Creates 24-hour profiles for:
        - Electricity price (time-of-use pricing)
        - Solar PV availability (solar irradiance)
        - Wind availability (wind speed)
        """
        # Price profile (higher during peak hours)
        base_price = 50.0  # $/MWh
        self.price_profile = np.array([
            base_price * 0.7,  # 0-1am
            base_price * 0.6,  # 1-2am
            base_price * 0.6,  # 2-3am
            base_price * 0.6,  # 3-4am
            base_price * 0.7,  # 4-5am
            base_price * 0.8,  # 5-6am
            base_price * 0.9,  # 6-7am
            base_price * 1.0,  # 7-8am
            base_price * 1.2,  # 8-9am (peak)
            base_price * 1.3,  # 9-10am (peak)
            base_price * 1.2,  # 10-11am
            base_price * 1.1,  # 11am-12pm
            base_price * 1.0,  # 12-1pm
            base_price * 1.0,  # 1-2pm
            base_price * 1.1,  # 2-3pm
            base_price * 1.2,  # 3-4pm
            base_price * 1.3,  # 4-5pm (peak)
            base_price * 1.4,  # 5-6pm (peak)
            base_price * 1.3,  # 6-7pm (peak)
            base_price * 1.2,  # 7-8pm
            base_price * 1.0,  # 8-9pm
            base_price * 0.9,  # 9-10pm
            base_price * 0.8,  # 10-11pm
            base_price * 0.7,  # 11pm-12am
        ])

        # Solar PV profile (daytime only)
        self.pv_profile = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 0-6am: no sun
            0.1, 0.3, 0.6, 0.8, 0.9, 1.0,  # 6am-12pm: sunrise to peak
            1.0, 0.9, 0.8, 0.6, 0.3, 0.1,  # 12pm-6pm: peak to sunset
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 6pm-12am: no sun
        ])

        # Wind profile (variable throughout day)
        np.random.seed(42)
        self.wind_profile = np.clip(0.5 + 0.3 * np.sin(np.linspace(0, 2*np.pi, 24)) +
                                     0.1 * np.random.randn(24), 0.0, 1.0)

    def _update_profiles(self, timestep: int) -> None:
        """Update environment profiles for current timestep.

        Args:
            timestep: Current episode timestep (0-23 for 24-hour episode)
        """
        hour = timestep % 24

        # Update price for all microgrids
        current_price = self.price_profile[hour]
        for agent_id, agent in self.registered_agents.items():
            if isinstance(agent, MicrogridFieldAgent):
                agent.set_grid_price(current_price)

        # Update renewable availability
        pv_avail = self.pv_profile[hour]
        wind_avail = self.wind_profile[hour]
        for agent_id, agent in self.registered_agents.items():
            if isinstance(agent, MicrogridFieldAgent):
                agent.set_renewable_availability(pv_avail, wind_avail)

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Run Pandapower AC power flow simulation.

        Updates Pandapower network with device setpoints from all microgrids,
        runs AC power flow, and extracts results.

        Args:
            env_state: Environment state with device setpoints

        Returns:
            Updated env_state with power flow results
        """
        # Update Pandapower network with device setpoints
        for mg_id, device_states in env_state.device_setpoints.items():
            if mg_id not in self.mg_bus_mappings:
                continue

            indices = self.mg_bus_mappings[mg_id]

            # Update ESS
            if "ess" in device_states:
                self.net.storage.at[indices["ess"], "p_mw"] = device_states["ess"]["P"]
                self.net.storage.at[indices["ess"], "q_mvar"] = device_states["ess"]["Q"]

            # Update DG
            if "dg" in device_states:
                self.net.sgen.at[indices["dg"], "p_mw"] = device_states["dg"]["P"]
                self.net.sgen.at[indices["dg"], "q_mvar"] = device_states["dg"]["Q"]

            # Update PV
            if "pv" in device_states:
                self.net.sgen.at[indices["pv"], "p_mw"] = device_states["pv"]["P"]
                self.net.sgen.at[indices["pv"], "q_mvar"] = device_states["pv"]["Q"]

            # Update Wind
            if "wind" in device_states:
                self.net.sgen.at[indices["wind"], "p_mw"] = device_states["wind"]["P"]
                self.net.sgen.at[indices["wind"], "q_mvar"] = device_states["wind"]["Q"]

        # Run AC power flow
        try:
            pp.runpp(self.net, algorithm="nr", calculate_voltage_angles=True)
            converged = True
        except Exception as e:
            print(f"Power flow failed to converge: {e}")
            converged = False

        # Extract results
        results = {"converged": converged}

        if converged:
            # Bus voltages
            results["bus_vm_pu"] = self.net.res_bus.vm_pu.values
            results["voltage_min"] = self.net.res_bus.vm_pu.min()
            results["voltage_max"] = self.net.res_bus.vm_pu.max()
            results["voltage_avg"] = self.net.res_bus.vm_pu.mean()

            # Line loading
            results["line_loading"] = self.net.res_line.loading_percent.values
            results["max_line_loading"] = self.net.res_line.loading_percent.max() / 100.0

            # Voltage violations
            v_min, v_max = self.voltage_limits
            results["voltage_violations"] = int(
                ((self.net.res_bus.vm_pu < v_min) | (self.net.res_bus.vm_pu > v_max)).sum()
            )

            # Overload violations
            results["overload_violations"] = int(
                (self.net.res_line.loading_percent > self.line_loading_limit * 100).sum()
            )

            # Power exchange with grid (from ext_grid results)
            if len(self.net.res_ext_grid) > 0:
                results["grid_p"] = self.net.res_ext_grid.p_mw.values[0]
                results["grid_q"] = self.net.res_ext_grid.q_mvar.values[0]
        else:
            # Power flow didn't converge - use default values
            results["voltage_min"] = 1.0
            results["voltage_max"] = 1.0
            results["voltage_avg"] = 1.0
            results["max_line_loading"] = 0.0
            results["voltage_violations"] = 0
            results["overload_violations"] = 0
            results["grid_p"] = 0.0
            results["grid_q"] = 0.0

        env_state.update_power_flow_results(results)
        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        """Convert simulation results to HERON global state dict format.

        Updates agent states with power flow results (voltages, line loading).

        Args:
            env_state: Environment state with power flow results

        Returns:
            Dict with structure: {"agent_states": {agent_id: state_dict, ...}}
        """
        agent_states = {}

        results = env_state.power_flow_results

        for agent_id, agent in self.registered_agents.items():
            if hasattr(agent, 'state') and agent.state and isinstance(agent, MicrogridFieldAgent):
                state_dict = agent.state.to_dict(include_metadata=True)

                # Update network state feature with power flow results
                if "features" in state_dict and "NetworkFeature" in state_dict["features"]:
                    state_dict["features"]["NetworkFeature"].update({
                        "voltage_min": results.get("voltage_min", 1.0),
                        "voltage_max": results.get("voltage_max", 1.0),
                        "voltage_avg": results.get("voltage_avg", 1.0),
                        "max_line_loading": results.get("max_line_loading", 0.0),
                        "voltage_violations": results.get("voltage_violations", 0),
                        "overload_violations": results.get("overload_violations", 0),
                    })

                # Update grid power exchange
                if "features" in state_dict and "GridFeature" in state_dict["features"]:
                    # Grid power is roughly divided among microgrids
                    grid_p = results.get("grid_p", 0.0) / self.num_microgrids
                    grid_q = results.get("grid_q", 0.0) / self.num_microgrids
                    state_dict["features"]["GridFeature"]["P"] = grid_p
                    state_dict["features"]["GridFeature"]["Q"] = grid_q

                agent_states[agent_id] = state_dict

        return {"agent_states": agent_states}

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        """Convert HERON global state to environment state for simulation.

        Extracts device setpoints from agent states.

        Args:
            global_state: Dict from proxy.state_cache["global"]

        Returns:
            EnvState for running power flow simulation
        """
        env_state = EnvState(num_microgrids=self.num_microgrids)

        agent_states = global_state.get("agent_states", {})

        for agent_id, state_dict in agent_states.items():
            # Get device states from features
            features = state_dict.get("features", {})

            device_states = {}
            if "ESSFeature" in features:
                device_states["ess"] = features["ESSFeature"]
            if "DGFeature" in features:
                device_states["dg"] = features["DGFeature"]
            if "RESFeature" in features:
                # Assuming first RES is PV, second is Wind
                # This is a simplification - better to use res_type field
                res_features = [v for k, v in features.items() if "RESFeature" in k]
                if len(res_features) >= 2:
                    device_states["pv"] = res_features[0]
                    device_states["wind"] = res_features[1]
                elif len(res_features) == 1:
                    device_states["pv"] = res_features[0]
                    device_states["wind"] = {"P": 0.0, "Q": 0.0}

            if "GridFeature" in features:
                device_states["grid"] = features["GridFeature"]

            env_state.update_setpoints(agent_id, device_states)

        return env_state

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        """Reset environment to initial state.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters

        Returns:
            Tuple of (observations, infos)
        """
        self._episode += 1
        self._timestep = 0

        # Update profiles for timestep 0
        self._update_profiles(self._timestep)

        # Call parent reset
        return super().reset(seed=seed, **kwargs)

    def step(self, actions: Dict[str, Any]):
        """Execute one environment step.

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        # Update profiles for current timestep
        self._update_profiles(self._timestep)

        # Call parent step (handles action execution, simulation, reward computation)
        obs, rewards, terminated, truncated, infos = super().step(actions)

        # Update device dynamics (ESS SOC, etc.) after simulation
        for agent_id, agent in self.registered_agents.items():
            if isinstance(agent, MicrogridFieldAgent):
                agent.update_device_dynamics()

        # Check episode termination
        self._timestep += 1
        if self._timestep >= self.episode_steps:
            terminated = {aid: True for aid in terminated}
            terminated["__all__"] = True

        return obs, rewards, terminated, truncated, infos

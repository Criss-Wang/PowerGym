"""Hierarchical multi-agent power grid environment.

This module implements a hierarchical microgrid environment following the grid_age style:
- SystemAgent -> PowerGridAgents -> DeviceAgents (Generator, ESS, Transformer)

Each device is its own field agent, and microgrids are coordinator agents.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandapower as pp

from heron.envs.base import MultiAgentEnv
from heron.agents.system_agent import SystemAgent, SYSTEM_AGENT_ID
from powergrid.envs.common import EnvState
from powergrid.agents import (
    PowerGridAgent,
    Generator,
    ESS,
    Transformer,
)
from powergrid.utils.loader import load_dataset


class HierarchicalMicrogridEnv(MultiAgentEnv):
    """Hierarchical multi-agent environment for networked microgrids.

    This environment supports CTDE (Centralized Training with Decentralized Execution):
    - Training: Agents share a collective reward to encourage cooperation
    - Execution: Agents can operate with limited communication
    """

    def __init__(
        self,
        system_agent: SystemAgent,  # REQUIRED: Always pass pre-initialized system agent
        dataset_path: str,
        episode_steps: int = 24,
        dt: float = 1.0,
        **kwargs,
    ):
        """Initialize hierarchical microgrid environment.

        Args:
            system_agent: Pre-initialized SystemAgent with agent hierarchy
            dataset_path: Path to dataset file
            episode_steps: Episode length in time steps (default: 24)
            dt: Time step duration in hours (default: 1.0)
            **kwargs: Additional arguments for MultiAgentEnv

        Note:
            Rewards are computed by individual agents via compute_local_reward().
            Safety penalties and reward sharing should be configured at the agent level.
        """
        self.episode_steps = episode_steps
        self.dt = dt
        self.num_microgrids = len(system_agent.subordinates)

        # Load dataset
        self._dataset = load_dataset(dataset_path)
        self._total_days = 0

        # Initialize episode state
        self._episode = 0
        self._timestep = 0
        self._train = True

        # Call parent init (registers agents)
        super().__init__(
            system_agent=system_agent,
            **kwargs,
        )

        # Network will be created during first reset()
        self.net = None

    def _read_data(self, load_area: str, renew_area: str) -> Dict[str, Any]:
        """Read data from dataset with train/test split support.

        Args:
            load_area: Load area identifier (e.g., 'AVA', 'BANC', 'BANCMID')
            renew_area: Renewable energy area identifier (e.g., 'NP15')

        Returns:
            Dict with load, solar, wind, price data
        """
        split = "train" if self._train else "test"
        data = self._dataset[split]

        return {
            "load": data["load"][load_area],
            "solar": data["solar"][renew_area],
            "wind": data["wind"][renew_area],
            "price": data["price"]["0096WD_7_N001"],
        }

    def _create_network_from_agents(self, global_state: Dict) -> pp.pandapowerNet:
        """Create Pandapower network from registered agents.

        Called during first reset after agents are registered.
        Uses agent structure to build network topology.

        Args:
            global_state: Global state dict (unused, for signature compatibility)

        Returns:
            Pandapower network matching agent structure
        """
        net = pp.create_empty_network(name="Hierarchical Microgrid Network")

        # Create main grid bus (slack)
        slack_bus = pp.create_bus(net, vn_kv=12.47, name="Grid")
        pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0)

        # Build network from registered microgrid coordinators
        microgrids = {}
        for agent_id, agent in self.registered_agents.items():
            if isinstance(agent, PowerGridAgent):
                microgrids[agent_id] = agent

        # Build network from microgrids
        for mg_id, mg_agent in microgrids.items():
            # Create microgrid bus
            mg_bus = pp.create_bus(net, vn_kv=0.48, name=f"{mg_id}_Bus")

            # Connect to grid via transformer
            pp.create_transformer_from_parameters(
                net,
                hv_bus=slack_bus,
                lv_bus=mg_bus,
                sn_mva=2.5,
                vn_hv_kv=12.47,
                vn_lv_kv=0.48,
                vkr_percent=1.0,
                vk_percent=5.0,
                pfe_kw=0.1,
                i0_percent=0.1,
                name=f"{mg_id}_Trafo",
            )

            # Get device agents from coordinator's subordinates
            subordinates = (
                mg_agent.subordinates if hasattr(mg_agent, "subordinates") else {}
            )

            # Create devices based on subordinate agent types
            for dev_id, dev_agent in subordinates.items():
                if isinstance(dev_agent, ESS):
                    # Storage device
                    capacity = dev_agent.capacity if hasattr(dev_agent, "capacity") else 1.0
                    pp.create_storage(
                        net,
                        bus=mg_bus,
                        p_mw=0.0,
                        max_e_mwh=capacity,
                        name=dev_id,
                        controllable=True,
                    )
                elif isinstance(dev_agent, Generator):
                    # Generator device
                    max_p = dev_agent.max_p if hasattr(dev_agent, "max_p") else 1.0
                    pp.create_sgen(
                        net,
                        bus=mg_bus,
                        p_mw=max_p * 0.5,
                        q_mvar=0.0,
                        name=dev_id,
                        controllable=True,
                    )
                elif isinstance(dev_agent, Transformer):
                    # Transformer is already created as connection
                    pass

            # Add load for this microgrid
            pp.create_load(
                net, bus=mg_bus, p_mw=0.2, q_mvar=0.05, name=f"{mg_id}_Load"
            )

        # Calculate total days from price data
        for agent in microgrids.values():
            if hasattr(agent, "dataset") and agent.dataset is not None:
                self._total_days = len(agent.dataset["price"]) // self.episode_steps
                break

        return net

    def _update_profiles(self, timestep: int) -> None:
        """Update profiles for current timestep using real data.

        Args:
            timestep: Current timestep index
        """
        global_state = self.proxy_agent.get_global_states(
            sender_id=SYSTEM_AGENT_ID, protocol=None
        )
        agent_states = global_state if isinstance(global_state, dict) else {}

        # Update each microgrid coordinator with current price and availability
        for agent_id, agent_state in agent_states.items():
            metadata = agent_state.get("metadata", {})
            agent_type = metadata.get("agent_type", "")

            # Check if this is a microgrid coordinator
            if "PowerGridAgent" in agent_type or "microgrid" in agent_id.lower():
                agent = self.registered_agents.get(agent_id)
                if agent and isinstance(agent, PowerGridAgent):
                    # Get dataset for this microgrid
                    if hasattr(agent, "dataset") and agent.dataset is not None:
                        hour = timestep % len(agent.dataset["price"])
                        price = float(agent.dataset["price"][hour])
                        solar = float(agent.dataset["solar"][hour])
                        wind = float(agent.dataset["wind"][hour])
                        load_scale = float(agent.dataset["load"][hour])

                        # Update agent state with current profiles
                        if hasattr(agent, "set_grid_price"):
                            agent.set_grid_price(price)
                        if hasattr(agent, "set_renewable_availability"):
                            agent.set_renewable_availability(solar, wind)
                        if hasattr(agent, "set_load_scale"):
                            agent.set_load_scale(load_scale)

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

        # Extract device setpoints from features
        for agent_id, state_dict in agent_states.items():
            features = state_dict.get("features", {})

            # Extract power setpoints from ElectricalBasePh
            if "ElectricalBasePh" in features:
                elec_feature = features["ElectricalBasePh"]
                env_state.set_device_setpoint(
                    agent_id,
                    P=elec_feature.get("P_MW", 0.0),
                    Q=elec_feature.get("Q_MVAr", 0.0),
                    in_service=elec_feature.get("in_service", True),
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
            in_service = setpoint.get("in_service", True)

            # Update storage devices
            storage_idx = self.net.storage[self.net.storage.name == device_id]
            if len(storage_idx) > 0:
                idx = storage_idx.index[0]
                self.net.storage.at[idx, "p_mw"] = P
                self.net.storage.at[idx, "q_mvar"] = Q
                self.net.storage.at[idx, "in_service"] = in_service
                continue

            # Update generator devices
            sgen_idx = self.net.sgen[self.net.sgen.name == device_id]
            if len(sgen_idx) > 0:
                idx = sgen_idx.index[0]
                self.net.sgen.at[idx, "p_mw"] = P
                self.net.sgen.at[idx, "q_mvar"] = Q
                self.net.sgen.at[idx, "in_service"] = in_service

        # Run AC power flow
        try:
            pp.runpp(self.net, algorithm="nr", calculate_voltage_angles=True)
            converged = True
        except Exception:
            converged = False

        # Extract power flow results
        if converged:
            results = {
                "converged": True,
                "voltage_min": float(self.net.res_bus.vm_pu.min()),
                "voltage_max": float(self.net.res_bus.vm_pu.max()),
                "voltage_avg": float(self.net.res_bus.vm_pu.mean()),
                "max_line_loading": (
                    float(self.net.res_line.loading_percent.max() / 100.0)
                    if len(self.net.res_line) > 0
                    else 0.0
                ),
                "grid_power": (
                    float(self.net.res_ext_grid.p_mw.values[0])
                    if len(self.net.res_ext_grid) > 0
                    else 0.0
                ),
                "overvoltage": float(
                    np.maximum(self.net.res_bus.vm_pu.values - 1.05, 0).sum()
                ),
                "undervoltage": float(
                    np.maximum(0.95 - self.net.res_bus.vm_pu.values, 0).sum()
                ),
            }
        else:
            results = {
                "converged": False,
                "voltage_min": 1.0,
                "voltage_max": 1.0,
                "voltage_avg": 1.0,
                "max_line_loading": 0.0,
                "grid_power": 0.0,
                "overvoltage": 0.0,
                "undervoltage": 0.0,
            }

        env_state.update_power_flow_results(results)
        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        """Convert simulation results back to global state format.

        Updates agent states with power flow results.

        Args:
            env_state: EnvState with power flow results

        Returns:
            Updated global state dict
        """
        power_flow_results = env_state.power_flow_results

        # Get current global state from proxy
        global_state = self.proxy_agent.get_global_states(
            sender_id=SYSTEM_AGENT_ID, protocol=None
        )
        agent_states = global_state if isinstance(global_state, dict) else {}

        # Update microgrid coordinator features with power flow results
        for agent_id, agent_state in agent_states.items():
            metadata = agent_state.get("metadata", {})
            agent_type = metadata.get("agent_type", "")

            # Check if this is a microgrid coordinator
            if "PowerGridAgent" in agent_type or "microgrid" in agent_id.lower():
                # Update network features with power flow results
                if "features" in agent_state:
                    features = agent_state["features"]
                    if "NetworkMetrics" in features:
                        features["NetworkMetrics"]["voltage_avg"] = power_flow_results.get(
                            "voltage_avg", 1.0
                        )
                        features["NetworkMetrics"]["converged"] = power_flow_results.get(
                            "converged", False
                        )

        return {"agent_states": agent_states}

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        """Reset environment.

        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional arguments

        Returns:
            Tuple of (observations, info)
        """
        if seed is not None:
            np.random.seed(seed)

        self._episode += 1
        self._timestep = 0

        # Select random day for training
        if self._train and self._total_days > 0:
            day = np.random.randint(0, self._total_days - 1)
            self._timestep = day * self.episode_steps

        # Update profiles for initial timestep
        if self.proxy_agent is not None:
            self._update_profiles(self._timestep)

        # Call parent reset
        obs, info = super().reset(seed=seed, **kwargs)

        # Create network on first reset
        if self.net is None:
            global_state = self.proxy_agent.get_global_states(
                sender_id=SYSTEM_AGENT_ID, protocol=None
            )
            self.net = self._create_network_from_agents(global_state)

        return obs, info

    def _pre_step(self, _actions: Dict[str, Any]) -> None:
        """Hook called before step execution.

        Updates external profiles (price, solar, wind, load) for agents.
        This is called before agents compute actions, so they observe
        the correct external conditions.

        Args:
            _actions: Dict mapping agent_id to action (unused here)
        """
        self._update_profiles(self._timestep)
        self._timestep += 1

    def get_power_grid_metrics(self) -> Dict[str, Any]:
        """Get power-grid specific metrics for evaluation.

        Returns:
            Dictionary containing power grid metrics
        """
        metrics = {}

        # Power balance metrics
        total_gen = 0.0
        total_load = 0.0

        for agent in self.registered_agents.values():
            if isinstance(agent, PowerGridAgent):
                # Sum generation from devices
                subordinates = (
                    agent.subordinates if hasattr(agent, "subordinates") else {}
                )
                for device in subordinates.values():
                    if isinstance(device, Generator):
                        p_mw = (
                            device.state.features.get("ElectricalBasePh", {}).get(
                                "P_MW", 0.0
                            )
                            if hasattr(device, "state")
                            else 0.0
                        )
                        total_gen += abs(p_mw)
                    elif isinstance(device, ESS):
                        p_mw = (
                            device.state.features.get("ElectricalBasePh", {}).get(
                                "P_MW", 0.0
                            )
                            if hasattr(device, "state")
                            else 0.0
                        )
                        if p_mw < 0:  # Discharging
                            total_gen += abs(p_mw)
                        else:  # Charging
                            total_load += p_mw

        # Network metrics
        if self.net is not None and self.net.get("converged", False):
            total_load += float(self.net.res_load["p_mw"].sum())
            vm = self.net.res_bus["vm_pu"].values
            voltage_violations = int(np.sum((vm > 1.05) | (vm < 0.95)))
            loading = self.net.res_line["loading_percent"].values
            line_overloads = int(np.sum(loading > 100))
        else:
            voltage_violations = 0
            line_overloads = 0

        metrics.update(
            {
                "total_generation_mw": float(total_gen),
                "total_load_mw": float(total_load),
                "power_balance_mw": float(total_gen - total_load),
                "voltage_violations": voltage_violations,
                "line_overloads": line_overloads,
                "convergence": (
                    bool(self.net.get("converged", False)) if self.net else False
                ),
                "timestep": self._timestep,
            }
        )

        return metrics

    def set_train_mode(self, train: bool = True) -> None:
        """Set training/evaluation mode.

        Args:
            train: True for training, False for evaluation
        """
        self._train = train

"""MultiAgentMicrogrids: Concrete environment for 3 networked microgrids.

This is a modernized version of the legacy MultiAgentMicrogrids that uses
PowerGridAgent instead of GridEnv while maintaining identical logic.

This environment supports CTDE (Centralized Training with Decentralized Execution):
- Training: Agents share a collective reward to encourage cooperation
- Execution: Agents can operate in event-driven mode with limited communication
"""

from typing import Any, Dict, List

import numpy as np
import pandapower as pp

from powergrid.agents.power_grid_agent import PowerGridAgent
from heron.protocols.vertical import SetpointProtocol
from powergrid.setups.loader import load_dataset
from powergrid.envs.networked_grid_env import NetworkedGridEnv
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.networks.ieee34 import IEEE34Bus
from heron.utils.typing import AgentID


class MultiAgentMicrogrids(NetworkedGridEnv):
    """
    Multi-agent environment with 3 networked microgrids connected to a DSO grid.
    """

    def __init__(self, env_config):
        """
        Initialize multi-agent microgrids environment.

        Args:
            env_config: Configuration dict with keys:
                - train: bool, training mode
                - penalty: float, safety penalty multiplier
                - share_reward: bool, share rewards across agents
        """
        if 'dataset_path' not in env_config:
            raise ValueError("env_config must include 'dataset_path' key.")
        self._dataset = load_dataset(env_config.get('dataset_path'))
        self._name = "MultiAgentMicrogrids"

        self._safety = env_config.get('penalty', 0.0)
        self._convergence_failure_reward = env_config.get('convergence_failure_reward', -200.0)
        self._convergence_failure_safety = env_config.get('convergence_failure_safety', 20.0)

        super().__init__(env_config)

    def _read_data(self, load_area, renew_area):
        """Read data from dataset with train/test split support.

        Args:
            load_area: Load area identifier (e.g., 'AVA', 'BANC', 'BANCMID')
            renew_area: Renewable energy area identifier (e.g., 'NP15')
        """
        split = 'train' if self.train else 'test'
        data = self._dataset[split]

        return {
            'load': data['load'][load_area],
            'solar': data['solar'][renew_area],
            'wind': data['wind'][renew_area],
            'price': data['price']['0096WD_7_N001']
        }
    
    def _build_agents(self) -> Dict[AgentID, PowerGridAgent]:
        """Build microgrid agents from environment config."""
        agents = {}
        for microgrid_config in self.env_config['microgrid_configs']:
            microgrid_agent = self._build_microgrid_agent(microgrid_config)
            agents[microgrid_agent.agent_id] = microgrid_agent
        return agents

    def _build_dso_net(self):
        """Build DSO main grid (non-actionable)."""
        net=IEEE34Bus('DSO')

        dso_config = self.env_config['dso_config']
        load_area = dso_config.get('load_area', 'BANC')
        renew_area = dso_config.get('renew_area', 'NP15')

        # No actionable devices in DSO, purely coordinating
        self.dso = PowerGridAgent(
            upstream_id=self._name,
            env_id=self.env_id,
            grid_config=dso_config,
            net=net,
        )
        # Set message broker if available (for distributed mode)
        if self.message_broker is not None:
            self.dso.set_message_broker(self.message_broker)
        self.dso.add_dataset(self._read_data(load_area, renew_area))

        return net

    def _build_microgrid_agent(self, microgrid_config) -> PowerGridAgent:
        """Build microgrid agent from config."""
        # Initialize protocol and policy
        # TODO: Make protocol configurable
        protocol = SetpointProtocol()
        policy = None
        
        # Create microgrid net
        microgrid_net = IEEE13Bus(microgrid_config['name'])

        # Initialize microgrid agent
        microgrid_agent = PowerGridAgent(
            protocol=protocol,
            policy=policy,
            upstream_id=self._name,
            env_id=self.env_id,
            grid_config=microgrid_config,
            net=microgrid_net,
        )
        # Set message broker if available (for distributed mode)
        if self.message_broker is not None:
            microgrid_agent.set_message_broker(self.message_broker)

        # Load dataset
        load_area = microgrid_config.get('load_area', 'AVA')
        renew_area = microgrid_config.get('renew_area', 'NP15')
        microgrid_agent.add_dataset(self._read_data(load_area, renew_area))

        return microgrid_agent


    def _build_net(self):
        """Build network with 3 microgrids connected to DSO grid."""
        # Create DSO main grid (non-actionable)
        net = self._build_dso_net()
        self._total_days = self.dso.dataset['price'].size // self.max_episode_steps

        # Create microgrids (actionable)
        microgrid_agents: List[PowerGridAgent] = []
        for microgrid_config in self.env_config['microgrid_configs']:
            microgrid_agent = self._build_microgrid_agent(microgrid_config)
            net = microgrid_agent.fuse_buses(net, microgrid_config['connection_bus'])
            microgrid_agents.append(microgrid_agent)

        # Run initial power flow
        pp.runpp(net)

        # Set network and agents
        self.net = net
        self.agent_dict.update({a.agent_id: a for a in microgrid_agents})
        # Update PettingZoo agent IDs via adapter method
        self._set_agent_ids(list(self.agent_dict.keys()))

        return net

    def _reward_and_safety(self):
        """
        Compute rewards and safety violations.

        Returns:
            rewards: Dict mapping agent_id → reward
            safety: Dict mapping agent_id → safety violation
        """
        if self.net["converged"]:
            # Reward and safety
            rewards = {n: -a.cost for n, a in self.agent_dict.items()}
            safety = {n: a.safety for n, a in self.agent_dict.items()}
        else:
            # Convergence failure penalty
            rewards = {n: self._convergence_failure_reward for n in self.agent_dict}
            safety = {n: self._convergence_failure_safety for n in self.agent_dict}

        # Apply safety penalty
        if self._safety:
            for name in self.agent_dict:
                rewards[name] -= safety[name] * self._safety

        return rewards, safety

    # ============================================
    # CTDE Collective Metrics
    # ============================================

    def get_power_grid_metrics(self) -> Dict[str, Any]:
        """Get power-grid specific metrics for CTDE evaluation.

        Returns:
            Dictionary containing:
                - total_generation_mw: Total active power generation
                - total_load_mw: Total load consumption
                - power_balance_mw: Generation - Load
                - voltage_violations: Number of buses with voltage violations
                - line_overloads: Number of overloaded lines
                - convergence_rate: Power flow convergence success rate
                - collective_metrics: From parent class get_collective_metrics()
        """
        metrics = self.get_collective_metrics()

        # Power balance metrics
        total_gen = 0.0
        total_load = 0.0

        for agent in self.agent_dict.values():
            if isinstance(agent, PowerGridAgent):
                # Sum generation from all devices
                for device in agent.sgen.values():
                    total_gen += abs(device.electrical.P_MW) if device.electrical else 0.0
                for device in agent.storage.values():
                    p_mw = device.electrical.P_MW if device.electrical else 0.0
                    if p_mw < 0:  # Discharging
                        total_gen += abs(p_mw)
                    else:  # Charging is load
                        total_load += p_mw

        # Get load from network
        if self.net is not None and 'converged' in self.net and self.net['converged']:
            total_load += float(self.net.res_load['p_mw'].sum())

        # Voltage and line violations
        voltage_violations = 0
        line_overloads = 0

        if self.net is not None and self.net.get('converged', False):
            vm = self.net.res_bus['vm_pu'].values
            voltage_violations = int(np.sum((vm > 1.05) | (vm < 0.95)))

            loading = self.net.res_line['loading_percent'].values
            line_overloads = int(np.sum(loading > 100))

        metrics.update({
            'total_generation_mw': float(total_gen),
            'total_load_mw': float(total_load),
            'power_balance_mw': float(total_gen - total_load),
            'voltage_violations': voltage_violations,
            'line_overloads': line_overloads,
            'convergence': bool(self.net.get('converged', False)) if self.net else False,
        })

        return metrics

    def get_ctde_training_info(self) -> Dict[str, Any]:
        """Get information useful for CTDE training monitoring.

        Returns:
            Dictionary with training metrics suitable for logging.
        """
        metrics = self.get_power_grid_metrics()

        # Per-agent breakdown
        agent_costs = {}
        agent_safety = {}

        for agent_id, agent in self.agent_dict.items():
            agent_costs[agent_id] = float(agent.cost)
            agent_safety[agent_id] = float(agent.safety)

        return {
            **metrics,
            'agent_costs': agent_costs,
            'agent_safety': agent_safety,
            'episode_step': self._episode_step,
            'timestep': self._t,
        }


if __name__ == '__main__':
    # Example usage
    # PUT YOUR TEST CODE HERE
    pass
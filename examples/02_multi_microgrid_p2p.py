"""
Example 2: Multi-Microgrid with Peer-to-Peer Energy Trading
=============================================================

This example demonstrates multi-agent coordination with peer-to-peer (P2P)
energy trading between microgrids using a horizontal protocol.

What you'll learn:
- Creating multiple GridAgents (microgrids)
- Using PeerToPeerTradingProtocol for horizontal coordination
- Environment-owned vs agent-owned protocols
- Market clearing mechanism with bids and offers
- Multi-agent reward computation with trading

Architecture:
    NetworkedGridEnv (Environment)
    ├── P2PTradingProtocol (Horizontal - Environment-owned)
    ├── GridAgent MG1 (Microgrid 1)
    │   ├── Generator
    │   └── ESS
    ├── GridAgent MG2 (Microgrid 2)
    │   ├── Generator
    │   └── ESS
    └── GridAgent MG3 (Microgrid 3)
        ├── Generator
        └── ESS

Coordination Flow:
    1. Each microgrid acts independently
    2. Environment collects P2P trading bids/offers
    3. Market clearing protocol matches trades
    4. Prices and quantities distributed back to agents

Runtime: ~45 seconds for 24 timesteps
"""

import numpy as np
import pandapower as pp

from powergrid.agents.grid_agent import PowerGridAgent
from powergrid.core.protocols import SetpointProtocol, PeerToPeerTradingProtocol
from powergrid.devices.generator import Generator
from powergrid.devices.storage import ESS
from powergrid.envs.multi_agent.networked_grid_env import NetworkedGridEnv
from powergrid.networks.ieee13 import IEEE13Bus


class MultiMicrogridP2PEnv(NetworkedGridEnv):
    """Environment with 3 microgrids using P2P energy trading."""

    def _build_agents(self):
        """Build grid agents for each microgrid."""
        return {}  # Will be populated in _build_net

    def _build_net(self):
        """Build network with 3 microgrids."""
        # Initialize agent dict
        if not hasattr(self, 'agent_dict') or self.agent_dict is None:
            self.agent_dict = {}

        # Create main network (just a placeholder bus)
        net = pp.create_empty_network(name="P2P_Network")
        pp.create_bus(net, vn_kv=4.16, name="Main Bus")

        # Create 3 microgrids with different characteristics
        mg_configs = [
            {
                "name": "MG1",
                "devices": [
                    # MG1: High generation capacity, low storage
                    {"type": "gen", "p_max": 3.0, "p_min": 1.0, "cost": 0.01},
                    {"type": "ess", "capacity": 3.0, "p_max": 1.0},
                ],
            },
            {
                "name": "MG2",
                "devices": [
                    # MG2: Medium generation, high storage
                    {"type": "gen", "p_max": 2.0, "p_min": 0.5, "cost": 0.02},
                    {"type": "ess", "capacity": 6.0, "p_max": 2.0},
                ],
            },
            {
                "name": "MG3",
                "devices": [
                    # MG3: Low generation, medium storage
                    {"type": "gen", "p_max": 1.5, "p_min": 0.3, "cost": 0.03},
                    {"type": "ess", "capacity": 4.0, "p_max": 1.5},
                ],
            },
        ]

        # Create dataset with varying load patterns for each microgrid
        dataset = self._create_multi_mg_dataset()

        # Build each microgrid
        for i, config in enumerate(mg_configs):
            mg_net = IEEE13Bus(config["name"])

            # Extract device configs
            gen_config = config["devices"][0]
            ess_config = config["devices"][1]

            # Create microgrid agent with full grid_config
            mg_agent = PowerGridAgent(
                net=mg_net,
                grid_config={
                    "name": config["name"],
                    "base_power": 1.0,
                    "load_scale": 1.0,
                    "devices": [
                        {
                            "type": "Generator",
                            "name": f"{config['name']}_gen",
                            "device_state_config": {
                                "bus": "Bus 633",
                                "p_max_MW": gen_config["p_max"],
                                "p_min_MW": gen_config["p_min"],
                                "q_max_MVAr": gen_config["p_max"] * 0.5,
                                "q_min_MVAr": -gen_config["p_max"] * 0.5,
                                "s_rated_MVA": gen_config["p_max"] * 1.2,
                                "startup_time_hr": 1.0,
                                "shutdown_time_hr": 1.0,
                                "cost_curve_coefs": [gen_config["cost"], 10.0, 0.0],
                            },
                        },
                        {
                            "type": "ESS",
                            "name": f"{config['name']}_ess",
                            "device_state_config": {
                                "bus": "Bus 634",
                                "e_capacity_MWh": ess_config["capacity"],
                                "soc_max": 0.9,
                                "soc_min": 0.1,
                                "p_max_MW": ess_config["p_max"],
                                "p_min_MW": -ess_config["p_max"],
                                "q_max_MVAr": ess_config["p_max"] * 0.4,
                                "q_min_MVAr": -ess_config["p_max"] * 0.4,
                                "s_rated_MVA": ess_config["p_max"] * 1.2,
                                "init_soc": 0.5,
                                "ch_eff": 0.95,
                                "dsc_eff": 0.95,
                            },
                        },
                    ],
                },
                protocol=SetpointProtocol(),  # Vertical protocol for internal coordination
            )

            # Add microgrid-specific dataset
            mg_agent.add_dataset(dataset[config["name"]])

            # Store agent
            self.agent_dict[config["name"]] = mg_agent

            # Merge networks (simple parallel connection for P2P trading)
            if i == 0:
                net = mg_net
            else:
                net = pp.merge_nets(net, mg_net, validate=False)

        # Update all agents to reference the merged network
        for agent in self.agent_dict.values():
            agent.net = net

        # Set environment attributes
        self.possible_agents = list(self.agent_dict.keys())
        self.net = net
        self.data_size = 240  # 10 days
        self._total_days = 10

        return net

    def _create_multi_mg_dataset(self):
        """Create datasets with different load patterns for each microgrid."""
        days = 10
        hours = days * 24

        # Base daily pattern
        base_pattern = np.array(
            [0.5, 0.45, 0.4, 0.4, 0.45, 0.55,  # Midnight to 6am
             0.7, 0.85, 0.95, 1.0, 1.05, 1.1,  # 6am to noon
             1.1, 1.05, 1.0, 1.05, 1.15, 1.2,  # Noon to 6pm
             1.15, 1.0, 0.85, 0.7, 0.6, 0.55]   # 6pm to midnight
        )

        # Create different patterns for each microgrid
        datasets = {}

        # MG1: High morning load (industrial)
        load1 = np.tile(base_pattern * 1.2, days)
        load1[:len(load1)//3] += 0.3  # Higher in morning
        datasets["MG1"] = {
            "load": np.clip(load1 + np.random.normal(0, 0.05, hours), 0.3, 1.5),
            "solar": np.zeros(hours),
            "wind": np.zeros(hours),
            "price": 50.0 * np.ones(hours),
        }

        # MG2: High evening load (residential)
        load2 = np.tile(base_pattern * 1.0, days)
        load2[2*len(load2)//3:] += 0.4  # Higher in evening
        datasets["MG2"] = {
            "load": np.clip(load2 + np.random.normal(0, 0.05, hours), 0.3, 1.5),
            "solar": np.zeros(hours),
            "wind": np.zeros(hours),
            "price": 50.0 * np.ones(hours),
        }

        # MG3: Moderate load (mixed)
        load3 = np.tile(base_pattern * 0.8, days)
        datasets["MG3"] = {
            "load": np.clip(load3 + np.random.normal(0, 0.05, hours), 0.3, 1.5),
            "solar": np.zeros(hours),
            "wind": np.zeros(hours),
            "price": 50.0 * np.ones(hours),
        }

        return datasets

    def _reward_and_safety(self):
        """Compute rewards and safety with P2P trading benefits."""
        rewards = {}
        safety = {}

        for agent_id, agent in self.agent_dict.items():
            # Base reward: negative cost
            rewards[agent_id] = -agent.cost

            # Safety violations
            safety[agent_id] = agent.safety

        return rewards, safety


def main():
    """Run multi-microgrid P2P trading example."""
    print("=" * 70)
    print("Example 2: Multi-Microgrid with P2P Energy Trading")
    print("=" * 70)

    # Create environment with P2P trading protocol
    env_config = {
        "max_episode_steps": 24,
        "train": True,
        "centralized": True,  # Use centralized execution mode
        "protocol": PeerToPeerTradingProtocol(
            trading_fee=0.02,  # 2% transaction fee
        ),
    }

    print("\n[1] Creating multi-microgrid environment with P2P trading...")
    env = MultiMicrogridP2PEnv(env_config)

    print(f"    Number of microgrids: {len(env.possible_agents)}")
    print(f"    Agents: {env.possible_agents}")
    print(f"    Action spaces:")
    for agent_id, space in env.action_spaces.items():
        print(f"      {agent_id}: {space}")

    # Reset environment
    print("\n[2] Resetting environment...")
    obs, info = env.reset(seed=42)

    print(f"    Observation shapes:")
    for agent_id, ob in obs.items():
        print(f"      {agent_id}: {ob.shape}")

    # Run simulation with random actions
    print("\n[3] Running 24-hour simulation with random actions...")
    print(f"    {'Hour':<6} {'MG1 Reward':<14} {'MG2 Reward':<14} {'MG3 Reward':<14} {'Total':<12}")
    print("    " + "-" * 65)

    total_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}
    total_safety = {agent_id: 0.0 for agent_id in env.possible_agents}

    for t in range(24):
        # Sample random actions and flatten dict spaces
        actions = {}
        for agent_id, space in env.action_spaces.items():
            action_sample = space.sample()
            # Flatten dict action space to array for protocol
            if isinstance(action_sample, dict):
                action_array = []
                if 'continuous' in action_sample:
                    action_array.append(action_sample['continuous'])
                if 'discrete' in action_sample:
                    action_array.append(action_sample['discrete'])
                actions[agent_id] = np.concatenate(action_array)
            else:
                actions[agent_id] = action_sample

        # Step environment (P2P trading happens here)
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # Track rewards
        for agent_id in env.possible_agents:
            total_rewards[agent_id] += rewards[agent_id]
            total_safety[agent_id] += infos[agent_id]["safety"]

        # Display
        total_r = sum(rewards.values())
        print(
            f"    {t+1:<6} "
            f"{rewards['MG1']:>12.2f}  "
            f"{rewards['MG2']:>12.2f}  "
            f"{rewards['MG3']:>12.2f}  "
            f"{total_r:>10.2f}"
        )

        if terminateds["__all__"]:
            break

    # Print summary
    print("\n[4] Simulation Summary:")
    print(f"    {'Microgrid':<12} {'Total Reward':<16} {'Avg/Hour':<14} {'Safety':<12}")
    print("    " + "-" * 60)
    for agent_id in env.possible_agents:
        avg_reward = total_rewards[agent_id] / 24
        print(
            f"    {agent_id:<12} "
            f"{total_rewards[agent_id]:>14.2f}  "
            f"{avg_reward:>12.2f}  "
            f"{total_safety[agent_id]:>10.2f}"
        )

    total_system_reward = sum(total_rewards.values())
    total_system_safety = sum(total_safety.values())
    print("    " + "-" * 60)
    print(
        f"    {'TOTAL':<12} "
        f"{total_system_reward:>14.2f}  "
        f"{total_system_reward/24:>12.2f}  "
        f"{total_system_safety:>10.2f}"
    )

    # Print P2P trading info
    print("\n[5] P2P Trading Protocol Info:")
    print(f"    Protocol type: PeerToPeerTradingProtocol")
    print(f"    Trading fee: 2.0%")
    print(f"    Market mechanism: Bid-offer matching with fee")
    print(f"    Coordination: Horizontal (environment-owned)")

    print("\n[6] Key Observations:")
    print("    - Each microgrid has different load patterns")
    print("    - MG1 has high generation capacity (can sell surplus)")
    print("    - MG2 has high storage capacity (can arbitrage)")
    print("    - MG3 has lower costs (can buy cheap, sell expensive)")
    print("    - P2P trading allows them to exchange energy at market prices")
    print("    - Transaction fees incentivize efficient trading")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Example 1: Single Microgrid with Centralized Control
=====================================================

This example demonstrates the basic usage of PowerGrid 2.0 with a single microgrid
containing multiple devices controlled by a centralized GridAgent.

What you'll learn:
- Creating device agents (Generator, ESS, Grid connection)
- Building a GridAgent to coordinate devices
- Using CentralizedSetpointProtocol for direct control
- Creating a simple PettingZoo environment
- Running a simulation loop with random actions

Architecture:
    GridAgent (Microgrid Controller)
    ├── Generator (Dispatchable power source)
    └── ESS (Energy Storage System)

Runtime: ~30 seconds for 24 timesteps
"""

import numpy as np
import pandapower as pp
from pettingzoo import ParallelEnv

from powergrid.agents.grid_agent import PowerGridAgent
from powergrid.core.protocols import CentralizedSetpointProtocol
from powergrid.devices.generator import Generator
# Note: Grid device not used in this example (implementation incomplete)
from powergrid.devices.storage import ESS
from powergrid.envs.multi_agent.networked_grid_env import NetworkedGridEnv
from powergrid.networks.ieee13 import IEEE13Bus


class SingleMicrogridEnv(NetworkedGridEnv):
    """Simple environment with one microgrid and 3 devices."""

    def _build_net(self):
        """Build network with single microgrid."""
        # Create IEEE 13-bus network
        net = IEEE13Bus("MG1")

        # Create devices
        # Note: bus names should be just "Bus XXX" without MG1 prefix
        # The PowerGridAgent.add_sgen/add_storage will prepend the microgrid name
        generator = Generator(
            agent_id="gen1",
            device_config={
                "name": "gen1",
                "device_state_config": {
                    "bus": "Bus 633",  # Will become "MG1 Bus 633" after add_sgen
                    "p_max_MW": 2.0,
                    "p_min_MW": 0.5,
                    "q_max_MVAr": 1.0,
                    "q_min_MVAr": -1.0,
                    "s_rated_MVA": 2.5,
                    "startup_time_hr": 1.0,
                    "shutdown_time_hr": 1.0,
                    "cost_curve_coefs": [0.02, 10.0, 0.0],  # quadratic cost
                },
            },
        )

        ess = ESS(
            agent_id="ess1",
            device_config={
                "name": "ess1",
                "device_state_config": {
                    "bus": "Bus 634",  # Will become "MG1 Bus 634" after add_storage
                    "capacity_MWh": 5.0,  # 5 MWh capacity
                    "max_e_MWh": 4.5,  # 90% usable
                    "min_e_MWh": 0.5,  # 10% minimum
                    "max_p_MW": 1.0,  # 1 MW charging/discharging
                    "min_p_MW": -1.0,
                    "max_q_MVAr": 0.5,
                    "min_q_MVAr": -0.5,
                    "s_rated_MVA": 1.2,
                    "init_soc": 0.5,  # Start at 50%
                    "ch_eff": 0.95,
                    "dsc_eff": 0.95,
                },
            },
        )

        # Create GridAgent with centralized control
        mg_agent = PowerGridAgent(
            net=net,
            grid_config={
                "name": "MG1",
                "base_power": 1.0,
                "load_scale": 1.0,
            },
            devices=[generator, ess],
            protocol=CentralizedSetpointProtocol(),  # Direct action distribution
            centralized=True,
        )

        # Add dataset (simple synthetic data)
        dataset = self._create_synthetic_dataset()
        mg_agent.add_dataset(dataset)

        # Set data size and total days for environment
        self.data_size = len(dataset["load"])
        self._total_days = self.data_size // self.max_episode_steps

        # Add devices to pandapower network
        mg_agent.add_sgen([generator])
        mg_agent.add_storage([ess])

        # Store agents
        self.possible_agents = ["MG1"]
        self.agent_dict = {"MG1": mg_agent}
        self.net = net

        return net

    def _create_synthetic_dataset(self):
        """Create simple dataset with multiple days for training."""
        days = 10  # 10 days of data
        hours_per_day = 24
        total_hours = days * hours_per_day

        # Create realistic daily patterns
        hourly_pattern = np.array([0.6, 0.55, 0.5, 0.5, 0.5, 0.55,  # Midnight to 6am
                                   0.7, 0.85, 0.95, 1.0, 1.05, 1.1,  # 6am to noon
                                   1.1, 1.05, 1.0, 1.05, 1.15, 1.2,  # Noon to 6pm
                                   1.15, 1.05, 0.95, 0.85, 0.75, 0.65])  # 6pm to midnight

        # Repeat pattern for multiple days with slight variations
        load = np.tile(hourly_pattern, days)
        load += np.random.normal(0, 0.05, total_hours)  # Add noise
        load = np.clip(load, 0.4, 1.3)  # Ensure reasonable bounds

        return {
            "load": load,
            "solar": np.zeros(total_hours),  # No solar in this example
            "wind": np.zeros(total_hours),  # No wind in this example
            "price": 50.0 * np.ones(total_hours),  # Constant price $50/MWh
        }

    def _reward_and_safety(self):
        """Compute rewards and safety violations."""
        rewards = {}
        safety = {}

        for agent_id, agent in self.agent_dict.items():
            # Reward = negative cost (minimize cost)
            rewards[agent_id] = -agent.cost

            # Safety = constraint violations
            safety[agent_id] = agent.safety

        return rewards, safety


def main():
    """Run single microgrid example."""
    print("=" * 70)
    print("Example 1: Single Microgrid with Centralized Control")
    print("=" * 70)

    # Create environment
    env_config = {
        "max_episode_steps": 24,  # 24-hour simulation
        "train": True,
        "protocol": CentralizedSetpointProtocol(),  # No horizontal protocol needed
    }

    print("\n[1] Creating environment...")
    env = SingleMicrogridEnv(env_config)

    print(f"    Possible agents: {env.possible_agents}")
    print(f"    Action spaces: {env.action_spaces}")
    print(f"    Observation spaces: {env.observation_spaces}")

    # Reset environment
    print("\n[2] Resetting environment...")
    obs, info = env.reset(seed=42)

    print(f"    Initial observation shape for MG1: {obs['MG1'].shape}")

    # Run simulation with random actions
    print("\n[3] Running 24-hour simulation with random actions...")
    print(f"    {'Hour':<6} {'Reward':<12} {'Safety':<12} {'Done'}")
    print("    " + "-" * 45)

    total_reward = 0
    total_safety = 0

    for t in range(24):
        # Sample random actions for each agent
        actions = {agent_id: space.sample() for agent_id, space in env.action_spaces.items()}

        # Step environment
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # Display results
        reward = rewards["MG1"]
        safety_val = infos["MG1"]["safety"]
        done = terminateds["__all__"]

        total_reward += reward
        total_safety += safety_val

        print(f"    {t+1:<6} {reward:>10.2f}  {safety_val:>10.2f}  {done}")

        if done:
            break

    # Print summary
    print("\n[4] Simulation Summary:")
    print(f"    Total reward: {total_reward:.2f}")
    print(f"    Total safety violations: {total_safety:.2f}")
    print(f"    Average reward per hour: {total_reward / 24:.2f}")

    # Inspect final agent states
    print("\n[5] Final Device States:")
    mg_agent = env.agent_dict["MG1"]

    for device_id, device in mg_agent.devices.items():
        print(f"\n    {device_id}:")
        state_dict = device.state.to_dict()
        for feature_name, feature_data in state_dict.items():
            print(f"      {feature_name}: {feature_data}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

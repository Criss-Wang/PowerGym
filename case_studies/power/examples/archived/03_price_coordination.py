"""
Example 3: Price-Based Coordination with Vertical Protocol
===========================================================

This example demonstrates hierarchical coordination using price signals.
A GridAgent broadcasts electricity prices to its subordinate devices, which
respond by adjusting their power output.

What you'll learn:
- Using PriceSignalProtocol for vertical coordination
- GridAgent → DeviceAgent communication via messages
- How devices respond to price signals
- Difference between vertical (agent-owned) and horizontal (env-owned) protocols
- Price-responsive behavior (economic dispatch)

Architecture:
    GridAgent (Microgrid Controller)
    ├── PriceSignalProtocol (Vertical - Agent-owned)
    ├── Generator (responds to price: high price → more generation)
    ├── ESS (responds to price: low price → charge, high price → discharge)
    └── Generator 2 (different cost curve → different response)

Coordination Flow:
    1. GridAgent observes system state
    2. GridAgent computes optimal price signal
    3. PriceSignalProtocol broadcasts price to all devices
    4. Each device receives price via message broker
    5. Devices adjust actions based on local objectives + price

Runtime: ~35 seconds for 24 timesteps
"""

import numpy as np
import pandapower as pp

from powergrid.agents.power_grid_agent import PowerGridAgent
from heron.protocols.vertical import PriceSignalProtocol
from powergrid.envs.networked_grid_env import NetworkedGridEnv
from powergrid.networks.ieee13 import IEEE13Bus


class PriceCoordinationEnv(NetworkedGridEnv):
    """Single microgrid with price-based device coordination."""

    def __init__(self, env_config):
        # Store price history for analysis
        self.price_history = []
        super().__init__(env_config)

    def _build_agents(self):
        """Build grid agents - populated in _build_net."""
        return {}

    def _build_net(self):
        """Build network with price-responsive devices."""
        net = IEEE13Bus("MG1")

        # Create GridAgent with price signal protocol
        # Initial price = $40/MWh (will vary during simulation)
        mg_agent = PowerGridAgent(
            net=net,
            message_broker=self.message_broker,
            upstream_id=self._name,  # Environment is the upstream in distributed mode
            env_id=self._env_id,
            grid_config={
                "name": "MG1",
                "base_power": 1.0,
                "load_scale": 1.0,
                "devices": [
                    # Generator 1: Low cost (will run at high prices)
                    {
                        "type": "Generator",
                        "name": "gen1_cheap",
                        "device_state_config": {
                            "bus": "Bus 633",
                            "p_max_MW": 2.5,
                            "p_min_MW": 0.5,
                            "q_max_MVAr": 1.2,
                            "q_min_MVAr": -1.2,
                            "s_rated_MVA": 3.0,
                            "startup_time_hr": 0.5,
                            "shutdown_time_hr": 0.5,
                            "cost_curve_coefs": [0.01, 8.0, 0.0],  # Low marginal cost
                        },
                    },
                    # Generator 2: High cost (will run only at very high prices)
                    {
                        "type": "Generator",
                        "name": "gen2_expensive",
                        "device_state_config": {
                            "bus": "Bus 645",
                            "p_max_MW": 1.5,
                            "p_min_MW": 0.3,
                            "q_max_MVAr": 0.8,
                            "q_min_MVAr": -0.8,
                            "s_rated_MVA": 2.0,
                            "startup_time_hr": 1.0,
                            "shutdown_time_hr": 1.0,
                            "cost_curve_coefs": [0.05, 15.0, 0.0],  # High marginal cost
                        },
                    },
                    # ESS: Charge at low prices, discharge at high prices
                    {
                        "type": "ESS",
                        "name": "ess1",
                        "device_state_config": {
                            "bus": "Bus 634",
                            "e_capacity_MWh": 6.0,
                            "max_e_MWh": 5.5,
                            "min_e_MWh": 0.5,
                            "p_max_MW": 1.5,
                            "p_min_MW": -1.5,
                            "q_max_MVAr": 0.7,
                            "q_min_MVAr": -0.7,
                            "s_rated_MVA": 1.8,
                            "init_soc": 0.5,
                            "ch_eff": 0.95,
                            "dsc_eff": 0.95,
                        },
                    },
                ],
            },
            protocol=PriceSignalProtocol(initial_price=40.0),
        )

        # Create dataset with varying prices
        dataset = self._create_price_varying_dataset()
        mg_agent.add_dataset(dataset)

        # Set environment attributes
        self.data_size = len(dataset["load"])
        self._total_days = self.data_size // self.max_episode_steps

        # Store agents
        self.possible_agents = ["MG1"]
        self.agent_dict = {"MG1": mg_agent}
        self.net = net

        return net

    def _create_price_varying_dataset(self):
        """Create dataset with realistic time-varying electricity prices."""
        days = 10
        hours = days * 24

        # Daily load pattern
        load_pattern = np.array(
            [0.5, 0.45, 0.4, 0.4, 0.45, 0.55,  # Midnight to 6am (low)
             0.7, 0.9, 1.0, 1.05, 1.1, 1.15,    # 6am to noon (ramping up)
             1.15, 1.1, 1.05, 1.1, 1.25, 1.3,   # Noon to 6pm (peak)
             1.25, 1.1, 0.95, 0.8, 0.65, 0.55]  # 6pm to midnight (declining)
        )
        load = np.tile(load_pattern, days) + np.random.normal(0, 0.05, hours)
        load = np.clip(load, 0.3, 1.5)

        # Price pattern (correlated with load)
        # Low prices at night, high prices during peak hours
        price_pattern = np.array(
            [25, 23, 22, 22, 23, 28,            # Midnight to 6am (off-peak)
             35, 45, 52, 55, 58, 60,            # 6am to noon (shoulder)
             62, 60, 58, 62, 75, 80,            # Noon to 6pm (peak)
             75, 60, 50, 42, 35, 28]            # 6pm to midnight (shoulder)
        )
        price = np.tile(price_pattern, days) + np.random.normal(0, 3.0, hours)
        price = np.clip(price, 20.0, 90.0)  # Price range: $20-90/MWh

        return {
            "load": load,
            "solar": np.zeros(hours),
            "wind": np.zeros(hours),
            "price": price,
        }

    def step(self, actions):
        """Override step to track price signals."""
        # Get current price from dataset
        current_price = self.agent_dict["MG1"].dataset["price"][self._t]
        self.price_history.append(current_price)

        # Update price in protocol
        self.agent_dict["MG1"].protocol.price = float(current_price)

        return super().step(actions)

    def _reward_and_safety(self):
        """Compute rewards and safety."""
        rewards = {}
        safety = {}

        for agent_id, agent in self.agent_dict.items():
            rewards[agent_id] = -agent.cost
            safety[agent_id] = agent.safety

        return rewards, safety


def main():
    """Run price-based coordination example."""
    print("=" * 70)
    print("Example 3: Price-Based Coordination with Vertical Protocol")
    print("=" * 70)

    # Create environment
    env_config = {
        "max_episode_steps": 24,
        "train": True,
    }

    print("\n[1] Creating environment with price-responsive devices...")
    env = PriceCoordinationEnv(env_config)

    print(f"    Microgrid: {env.possible_agents[0]}")
    print(f"    Devices: {list(env.agent_dict['MG1'].devices.keys())}")
    print(f"    Protocol: {env.agent_dict['MG1'].protocol.__class__.__name__}")
    print(f"    Action space: {env.action_spaces['MG1']}")

    # Reset
    print("\n[2] Resetting environment...")
    obs, info = env.reset(seed=42)

    # Run simulation
    print("\n[3] Running 24-hour simulation with price-responsive control...")
    print(f"    {'Hour':<6} {'Price':<10} {'Reward':<12} {'Safety':<10} {'Gen1':<10} {'Gen2':<10} {'ESS':<10}")
    print("    " + "-" * 75)

    total_reward = 0
    total_safety = 0
    device_powers = []

    for t in range(24):
        # Sample actions (in real use, policy would respond to prices)
        actions = {"MG1": env.action_spaces["MG1"].sample()}

        # Step
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # Track results
        reward = rewards["MG1"]
        safety_val = infos["MG1"]["safety"]
        price = env.price_history[-1]

        total_reward += reward
        total_safety += safety_val

        # Get device powers (from observations)
        mg_agent = env.agent_dict["MG1"]
        gen1_p = mg_agent.devices["gen1_cheap"].electrical.P_MW
        gen2_p = mg_agent.devices["gen2_expensive"].electrical.P_MW
        ess_p = mg_agent.devices["ess1"].electrical.P_MW

        device_powers.append({
            "hour": t + 1,
            "price": price,
            "gen1": gen1_p,
            "gen2": gen2_p,
            "ess": ess_p,
        })

        print(
            f"    {t+1:<6} "
            f"${price:>7.1f}  "
            f"{reward:>10.2f}  "
            f"{safety_val:>8.2f}  "
            f"{gen1_p:>8.2f}  "
            f"{gen2_p:>8.2f}  "
            f"{ess_p:>8.2f}"
        )

        if terminateds["__all__"]:
            break

    # Summary
    print("\n[4] Simulation Summary:")
    print(f"    Total reward: {total_reward:.2f}")
    print(f"    Average reward/hour: {total_reward / 24:.2f}")
    print(f"    Total safety violations: {total_safety:.2f}")

    # Analyze price-response behavior
    print("\n[5] Price-Response Analysis:")
    prices = [d["price"] for d in device_powers]
    gen1_powers = [d["gen1"] for d in device_powers]
    gen2_powers = [d["gen2"] for d in device_powers]
    ess_powers = [d["ess"] for d in device_powers]

    print(f"    Price range: ${min(prices):.1f} - ${max(prices):.1f}/MWh")
    print(f"    Average price: ${np.mean(prices):.1f}/MWh")
    print(f"\n    Device Behavior:")
    print(f"      Gen1 (cheap): Avg {np.mean(gen1_powers):.2f} MW")
    print(f"      Gen2 (expensive): Avg {np.mean(gen2_powers):.2f} MW")
    print(f"      ESS: Avg {np.mean(ess_powers):.2f} MW (+ discharge, - charge)")

    # Find price-response correlation
    high_price_hours = [i for i, p in enumerate(prices) if p > 60]
    low_price_hours = [i for i, p in enumerate(prices) if p < 35]

    if high_price_hours:
        avg_gen1_high = np.mean([gen1_powers[i] for i in high_price_hours])
        avg_ess_high = np.mean([ess_powers[i] for i in high_price_hours])
        print(f"\n    High Price Hours (>${60}/MWh):")
        print(f"      Gen1 output: {avg_gen1_high:.2f} MW")
        print(f"      ESS output: {avg_ess_high:.2f} MW (should be positive = discharging)")

    if low_price_hours:
        avg_gen1_low = np.mean([gen1_powers[i] for i in low_price_hours])
        avg_ess_low = np.mean([ess_powers[i] for i in low_price_hours])
        print(f"\n    Low Price Hours (<${35}/MWh):")
        print(f"      Gen1 output: {avg_gen1_low:.2f} MW")
        print(f"      ESS output: {avg_ess_low:.2f} MW (should be negative = charging)")

    print("\n[6] Protocol Details:")
    print(f"    Type: Vertical (Agent-owned)")
    print(f"    Owner: GridAgent (MG1)")
    print(f"    Mechanism: Broadcast price to subordinate devices")
    print(f"    Communication: Via MessageBroker")
    print(f"    Response: Each device optimizes locally given price signal")

    print("\n[7] Key Insights:")
    print("    - Price signals coordinate distributed decision-making")
    print("    - Cheap generators run more at high prices (profit motive)")
    print("    - Expensive generators run less (wait for higher prices)")
    print("    - ESS charges at low prices, discharges at high prices (arbitrage)")
    print("    - No central optimization needed - market clearing emerges")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

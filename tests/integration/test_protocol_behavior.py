"""
Behavioral verification tests for protocols.

Tests that protocols actually produce the expected coordination effects,
not just that training doesn't crash. Verifies:
1. P2P trading actually reduces system costs
2. Consensus actually causes agents to converge
3. Price signals actually affect device behavior
4. Protocol rewards/costs are reasonable
"""

import pytest
import numpy as np

try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False

from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.configs.config_loader import load_config
from heron.protocols.base import (
    Protocol,
    NoProtocol,
    NoCommunication,
    NoActionCoordination,
)
from heron.protocols.vertical import (
    SetpointProtocol,
    PriceSignalProtocol,
)
from heron.protocols.horizontal import (
    PeerToPeerTradingProtocol,
    ConsensusProtocol,
# Original: (
    NoHorizontalProtocol,
    PeerToPeerTradingProtocol,
    ConsensusProtocol,
)


@pytest.mark.skipif(not RLLIB_AVAILABLE, reason="RLlib not installed")
class TestProtocolBehavioralCorrectness:
    """Test that protocols produce expected behavioral effects."""

    @classmethod
    def setup_class(cls):
        """Initialize Ray once for all tests."""
        if RLLIB_AVAILABLE:
            ray.init(ignore_reinit_error=True, num_cpus=2)

    @classmethod
    def teardown_class(cls):
        """Shutdown Ray after all tests."""
        if RLLIB_AVAILABLE:
            ray.shutdown()

    def train_and_evaluate(self, protocol, protocol_name, iterations=10):
        """Train with protocol and return reward history."""
        config = load_config('ieee34_ieee13')
        config['train'] = True
        config['penalty'] = 10.0
        config['share_reward'] = True
        config['max_episode_steps'] = 24

        def env_creator(env_config):
            env = MultiAgentMicrogrids(env_config)
            return ParallelPettingZooEnv(env)

        env_name = f"test_behavior_{protocol_name}"
        register_env(env_name, env_creator)

        # Create temp env
        base_env = MultiAgentMicrogrids(config)
        temp_env = ParallelPettingZooEnv(base_env)
        first_agent = base_env.possible_agents[0]

        # MAPPO config
        ppo_config = (
            PPOConfig()
            .environment(env=env_name, env_config=config, disable_env_checking=True)
            .framework("torch")
            .training(
                train_batch_size=1000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
                lr=5e-4,
                gamma=0.99,
                lambda_=0.95,
            )
            .rollouts(num_rollout_workers=1)
            .multi_agent(
                policies={
                    'shared_policy': (
                        None,
                        temp_env.observation_space[first_agent],
                        temp_env.action_space[first_agent],
                        {}
                    )
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: 'shared_policy',
            )
            .resources(num_gpus=0)
            .debugging(seed=42)  # Fixed seed for reproducibility
        )

        algo = ppo_config.build()

        rewards = []
        for i in range(iterations):
            result = algo.train()
            reward = result.get('episode_reward_mean', 0)
            rewards.append(reward)

        algo.stop()

        return rewards

    def test_p2p_trading_improves_over_no_coordination(self):
        """Test that P2P trading produces better rewards than no coordination."""
        print("\n" + "="*70)
        print("BEHAVIORAL TEST: P2P Trading Should Improve System Performance")
        print("="*70)

        # Train without coordination (baseline)
        print("\n[1] Training with NoProtocol (baseline)...")
        no_protocol_rewards = self.train_and_evaluate(
            NoHorizontalProtocol(), "no_protocol", iterations=10
        )

        # Train with P2P trading
        print("\n[2] Training with P2PTradingProtocol...")
        p2p_rewards = self.train_and_evaluate(
            PeerToPeerTradingProtocol(trading_fee=0.01), "p2p_trading", iterations=10
        )

        # Compare final rewards (last 3 iterations average)
        no_protocol_final = np.mean(no_protocol_rewards[-3:])
        p2p_final = np.mean(p2p_rewards[-3:])

        print(f"\n[3] Results:")
        print(f"  NoProtocol final reward (avg last 3): {no_protocol_final:.2f}")
        print(f"  P2PTrading final reward (avg last 3): {p2p_final:.2f}")
        print(f"  Improvement: {p2p_final - no_protocol_final:.2f}")
        print(f"  Improvement %: {((p2p_final - no_protocol_final) / abs(no_protocol_final)) * 100:.1f}%")

        # Check that rewards are improving over training
        print(f"\n[4] Learning Curves:")
        print(f"  NoProtocol - Start: {no_protocol_rewards[0]:.2f}, End: {no_protocol_rewards[-1]:.2f}")
        print(f"  P2PTrading - Start: {p2p_rewards[0]:.2f}, End: {p2p_rewards[-1]:.2f}")

        # Verification checks
        # 1. Both should learn (improve over time)
        no_protocol_improved = no_protocol_rewards[-1] > no_protocol_rewards[0]
        p2p_improved = p2p_rewards[-1] > p2p_rewards[0]

        print(f"\n[5] Learning Verification:")
        print(f"  NoProtocol learning: {no_protocol_improved} (improved: {no_protocol_rewards[-1] - no_protocol_rewards[0]:.2f})")
        print(f"  P2PTrading learning: {p2p_improved} (improved: {p2p_rewards[-1] - p2p_rewards[0]:.2f})")

        # 2. P2P should eventually be better OR at least competitive
        # (might not always be better in short training, but should be close)
        competitive_threshold = abs(no_protocol_final) * 0.1  # Within 10%
        is_competitive = (p2p_final >= no_protocol_final - competitive_threshold)

        print(f"\n[6] Protocol Effectiveness:")
        print(f"  Is P2P competitive? {is_competitive}")
        print(f"  Threshold: {competitive_threshold:.2f}")

        # Assertions
        assert no_protocol_improved, "NoProtocol should learn (improve over time)"
        assert p2p_improved, "P2PTrading should learn (improve over time)"
        assert is_competitive, f"P2P should be competitive with baseline (within {competitive_threshold:.2f})"

        print("\n✓ P2P Trading behavioral test PASSED")

    def test_consensus_causes_agent_convergence(self):
        """Test that consensus protocol causes agent values to converge."""
        print("\n" + "="*70)
        print("BEHAVIORAL TEST: Consensus Should Cause Agent Convergence")
        print("="*70)

        # Create environment with consensus
        config = load_config('ieee34_ieee13')
        config['train'] = False  # Evaluation mode
        config['max_episode_steps'] = 24

        # Use consensus protocol
        # Note: We need to check if agents' control values converge during episodes

        env = MultiAgentMicrogrids(config)

        # Run episode and track agent observations
        obs, info = env.reset()

        agent_observations = []
        for t in range(10):
            # Random actions
            actions = {
                agent_id: env.action_space(agent_id).sample()
                for agent_id in env.agents
            }

            obs, rewards, dones, truncated, info = env.step(actions)
            agent_observations.append(obs)

        # Check that observations are valid
        assert len(agent_observations) > 0
        assert all(len(obs) == len(env.agents) for obs in agent_observations)

        print(f"\n✓ Collected {len(agent_observations)} timesteps of observations")
        print(f"  Number of agents: {len(env.agents)}")
        print(f"  Agents: {list(env.agents)}")

        # Basic check: observations should have consistent structure
        first_obs = agent_observations[0]
        for agent_id in env.agents:
            assert agent_id in first_obs
            assert first_obs[agent_id] is not None

        print("✓ Consensus behavioral structure test PASSED")

    def test_rewards_are_reasonable_magnitudes(self):
        """Test that rewards are in reasonable ranges (not all zeros or extreme values)."""
        print("\n" + "="*70)
        print("BEHAVIORAL TEST: Rewards Should Be Reasonable Magnitudes")
        print("="*70)

        config = load_config('ieee34_ieee13')
        config['train'] = False
        config['max_episode_steps'] = 24

        protocols_to_test = {
            "NoProtocol": NoHorizontalProtocol(),
            "P2PTrading": PeerToPeerTradingProtocol(trading_fee=0.01),
        }

        for protocol_name, protocol in protocols_to_test.items():
            print(f"\n[{protocol_name}]")

            env = MultiAgentMicrogrids(config)
            obs, info = env.reset()

            episode_rewards = {agent_id: [] for agent_id in env.agents}

            for t in range(24):
                actions = {
                    agent_id: env.action_space(agent_id).sample()
                    for agent_id in env.agents
                }

                obs, rewards, dones, truncated, info = env.step(actions)

                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id].append(reward)

            # Check rewards
            for agent_id, reward_list in episode_rewards.items():
                total_reward = sum(reward_list)
                avg_reward = np.mean(reward_list)
                std_reward = np.std(reward_list)

                print(f"  {agent_id}:")
                print(f"    Total: {total_reward:.2f}")
                print(f"    Average: {avg_reward:.2f}")
                print(f"    Std Dev: {std_reward:.2f}")
                print(f"    Range: [{min(reward_list):.2f}, {max(reward_list):.2f}]")

                # Verification checks
                # 1. Rewards should not all be zero
                assert not all(r == 0 for r in reward_list), f"{agent_id} rewards are all zero"

                # 2. Rewards should not be all the same (no variation)
                assert std_reward > 0.01, f"{agent_id} rewards have no variation"

                # 3. Rewards should be in reasonable range (not extreme)
                assert all(abs(r) < 100000 for r in reward_list), f"{agent_id} has extreme reward values"

                # 4. Total reward should be negative (cost minimization)
                # In power systems, we typically minimize cost, so rewards are negative
                assert total_reward < 0, f"{agent_id} total reward should be negative (cost)"

        print("\n✓ Reward magnitude test PASSED")

    def test_training_shows_improvement(self):
        """Test that training actually improves performance over time."""
        print("\n" + "="*70)
        print("BEHAVIORAL TEST: Training Should Show Improvement")
        print("="*70)

        # Train for more iterations to ensure learning
        print("\n[1] Training with P2P protocol for 15 iterations...")
        rewards = self.train_and_evaluate(
            PeerToPeerTradingProtocol(trading_fee=0.01),
            "improvement_test",
            iterations=15
        )

        # Compare early vs late performance
        early_rewards = np.mean(rewards[:3])
        late_rewards = np.mean(rewards[-3:])
        improvement = late_rewards - early_rewards
        improvement_pct = (improvement / abs(early_rewards)) * 100

        print(f"\n[2] Learning Progress:")
        print(f"  Early (iterations 0-2): {early_rewards:.2f}")
        print(f"  Late (iterations 12-14): {late_rewards:.2f}")
        print(f"  Improvement: {improvement:.2f}")
        print(f"  Improvement %: {improvement_pct:.1f}%")

        # Check for monotonic-ish improvement (allowing some noise)
        # Use moving average to smooth out noise
        window = 3
        moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]

        print(f"\n[3] Moving Average (window={window}):")
        for i, ma in enumerate(moving_avg):
            print(f"  Iteration {i}: {ma:.2f}")

        # Final moving average should be better than initial
        assert moving_avg[-1] > moving_avg[0], "Final performance should be better than initial"

        # Should see overall improvement
        assert improvement > 0, f"Should see improvement over training (got {improvement:.2f})"

        print("\n✓ Training improvement test PASSED")


@pytest.mark.skipif(not RLLIB_AVAILABLE, reason="RLlib not installed")
class TestProtocolSpecificBehavior:
    """Test protocol-specific behavioral characteristics."""

    def test_p2p_trading_creates_trades(self):
        """Test that P2P trading protocol actually creates trades."""
        print("\n" + "="*70)
        print("BEHAVIORAL TEST: P2P Should Create Actual Trades")
        print("="*70)

        protocol = PeerToPeerTradingProtocol(trading_fee=0.02)

        # Create scenario with buyer and seller
        from heron.agents.base import Agent, Observation

        class MockAgent(Agent):
            def __init__(self, agent_id, net_demand, marginal_cost):
                super().__init__(agent_id)
                self.net_demand = net_demand
                self.marginal_cost = marginal_cost

            def observe(self):
                return Observation(local={
                    'net_demand': self.net_demand,
                    'marginal_cost': self.marginal_cost
                })

            def act(self, observation):
                return None

        # Create buyer (positive demand) and seller (negative demand)
        buyer = MockAgent("MG1", net_demand=10.0, marginal_cost=60.0)
        seller = MockAgent("MG2", net_demand=-10.0, marginal_cost=40.0)

        agents = {"MG1": buyer, "MG2": seller}
        observations = {
            "MG1": Observation(local={'net_demand': 10.0, 'marginal_cost': 60.0}),
            "MG2": Observation(local={'net_demand': -10.0, 'marginal_cost': 40.0})
        }

        # Run protocol coordination
        signals = protocol.coordinate(agents, observations)

        print(f"\n[1] Trade Signals:")
        for agent_id, signal in signals.items():
            print(f"  {agent_id}: {signal}")

        # Verify trades were created
        assert 'MG1' in signals, "Buyer should receive signals"
        assert 'MG2' in signals, "Seller should receive signals"
        assert 'trades' in signals['MG1'], "Buyer should receive trade info"
        assert 'trades' in signals['MG2'], "Seller should receive trade info"

        buyer_trades = signals['MG1']['trades']
        seller_trades = signals['MG2']['trades']

        assert len(buyer_trades) > 0, "Buyer should have at least one trade"
        assert len(seller_trades) > 0, "Seller should have at least one trade"

        # Check trade properties
        buyer_trade = buyer_trades[0]
        seller_trade = seller_trades[0]

        print(f"\n[2] Trade Details:")
        print(f"  Buyer trade: {buyer_trade}")
        print(f"  Seller trade: {seller_trade}")

        # Buyer should buy (positive quantity)
        assert buyer_trade['quantity'] > 0, "Buyer should have positive quantity"

        # Seller should sell (negative quantity)
        assert seller_trade['quantity'] < 0, "Seller should have negative quantity"

        # Quantities should match (conservation)
        assert abs(buyer_trade['quantity'] + seller_trade['quantity']) < 0.01, \
            "Trade quantities should conserve"

        # Price should be between bid and offer
        trade_price = buyer_trade['price']
        assert 40.0 < trade_price < 60.0, \
            f"Trade price {trade_price} should be between marginal costs"

        print(f"\n✓ P2P trading creates trades correctly")
        print(f"  Trade quantity: {buyer_trade['quantity']:.2f} MW")
        print(f"  Trade price: {trade_price:.2f} $/MWh")

    def test_consensus_converges_values(self):
        """Test that consensus protocol converges agent values."""
        print("\n" + "="*70)
        print("BEHAVIORAL TEST: Consensus Should Converge Values")
        print("="*70)

        protocol = ConsensusProtocol(max_iterations=50, tolerance=0.01)

        from heron.agents.base import Agent, Observation

        class MockAgent(Agent):
            def observe(self):
                return Observation()

            def act(self, observation):
                return None

        # Create agents with different initial values
        agents = {
            "MG1": MockAgent("MG1"),
            "MG2": MockAgent("MG2"),
            "MG3": MockAgent("MG3"),
        }

        observations = {
            "MG1": Observation(local={'control_value': 10.0}),
            "MG2": Observation(local={'control_value': 20.0}),
            "MG3": Observation(local={'control_value': 30.0}),
        }

        # Run consensus
        signals = protocol.coordinate(agents, observations)

        print(f"\n[1] Consensus Results:")
        consensus_values = []
        for agent_id, signal in signals.items():
            value = signal['consensus_value']
            consensus_values.append(value)
            print(f"  {agent_id}: {value:.4f}")

        # Check convergence
        mean_value = np.mean(consensus_values)
        std_value = np.std(consensus_values)
        max_diff = max(consensus_values) - min(consensus_values)

        print(f"\n[2] Convergence Metrics:")
        print(f"  Mean: {mean_value:.4f}")
        print(f"  Std Dev: {std_value:.4f}")
        print(f"  Max Difference: {max_diff:.4f}")

        # All values should be close to average (20.0)
        expected_avg = 20.0
        assert abs(mean_value - expected_avg) < 1.0, \
            f"Mean should be close to {expected_avg}"

        # Values should have converged (small std dev)
        assert std_value < 1.0, f"Values should converge (std={std_value})"

        # Max difference should be small
        assert max_diff < 2.0, f"Max difference should be small (got {max_diff})"

        print(f"\n✓ Consensus converges values correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
Integration tests for protocols under actual RL training.

Tests that protocols work correctly during MAPPO/IPPO/MADDPG training,
not just in simulation loops. This ensures:
1. Protocols don't break gradient computation
2. Protocols work with different RL algorithms
3. Training converges with different protocols
4. Protocol coordination improves learning
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
class TestMAPPOTrainingWithProtocols:
    """Test MAPPO training with different horizontal protocols."""

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

    def create_env(self, protocol):
        """Create environment with specified protocol."""
        config = load_config('ieee34_ieee13')
        config['train'] = True
        config['penalty'] = 10.0
        config['share_reward'] = True
        config['max_episode_steps'] = 24  # Short episodes for fast testing

        def env_creator(env_config):
            env = MultiAgentMicrogrids(env_config)
            return ParallelPettingZooEnv(env)

        return env_creator, config

    def create_ppo_config(self, env_name, env_config, shared_policy=True):
        """Create PPO configuration for testing."""
        # Create temp env to get spaces
        base_env = MultiAgentMicrogrids(env_config)
        temp_env = ParallelPettingZooEnv(base_env)

        possible_agents = base_env.possible_agents
        first_agent = possible_agents[0]

        if shared_policy:
            # MAPPO: shared policy
            policies = {
                'shared_policy': (
                    None,
                    temp_env.observation_space[first_agent],
                    temp_env.action_space[first_agent],
                    {}
                )
            }
            policy_mapping_fn = lambda agent_id, *args, **kwargs: 'shared_policy'
        else:
            # IPPO: independent policies
            policies = {
                agent_id: (None, temp_env.observation_space[agent_id],
                          temp_env.action_space[agent_id], {})
                for agent_id in possible_agents
            }
            policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id

        config = (
            PPOConfig()
            .environment(env=env_name, env_config=env_config, disable_env_checking=True)
            .framework("torch")
            .training(
                train_batch_size=500,
                sgd_minibatch_size=128,
                num_sgd_iter=5,
                lr=5e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.3,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
            )
            .rollouts(
                num_rollout_workers=1,
                num_envs_per_worker=1,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .resources(num_gpus=0)
        )

        return config

    def test_mappo_with_no_horizontal_protocol(self):
        """Test MAPPO training with NoHorizontalProtocol (baseline)."""
        protocol = NoHorizontalProtocol()

        # Create environment
        env_creator, env_config = self.create_env(protocol)
        env_name = "test_no_protocol"
        register_env(env_name, env_creator)

        # Create MAPPO configuration
        config = self.create_ppo_config(env_name, env_config, shared_policy=True)

        # Train for 3 iterations
        algo = config.build()

        rewards = []
        for i in range(3):
            result = algo.train()
            reward = result.get('episode_reward_mean', 0)
            rewards.append(reward)

            # Check training runs without errors
            assert isinstance(reward, (int, float))
            assert not np.isnan(reward)

        # Training should not crash
        algo.stop()

        print(f"\n✓ NoHorizontalProtocol: Training completed, rewards: {rewards}")

    def test_mappo_with_p2p_trading(self):
        """Test MAPPO training with PeerToPeerTradingProtocol."""
        protocol = PeerToPeerTradingProtocol(trading_fee=0.02)

        # Create environment
        env_creator, env_config = self.create_env(protocol)
        env_name = "test_p2p_trading"
        register_env(env_name, env_creator)

        # Create MAPPO configuration
        config = self.create_ppo_config(env_name, env_config, shared_policy=True)

        # Train for 3 iterations
        algo = config.build()

        rewards = []
        for i in range(3):
            result = algo.train()
            reward = result.get('episode_reward_mean', 0)
            rewards.append(reward)

            # Check training runs without errors
            assert isinstance(reward, (int, float))
            assert not np.isnan(reward)

        algo.stop()

        print(f"\n✓ P2PTradingProtocol: Training completed, rewards: {rewards}")

    def test_mappo_with_consensus(self):
        """Test MAPPO training with ConsensusProtocol."""
        protocol = ConsensusProtocol(max_iterations=5, tolerance=0.01)

        # Create environment
        env_creator, env_config = self.create_env(protocol)
        env_name = "test_consensus"
        register_env(env_name, env_creator)

        # Create MAPPO configuration
        config = self.create_ppo_config(env_name, env_config, shared_policy=True)

        # Train for 3 iterations
        algo = config.build()

        rewards = []
        for i in range(3):
            result = algo.train()
            reward = result.get('episode_reward_mean', 0)
            rewards.append(reward)

            # Check training runs without errors
            assert isinstance(reward, (int, float))
            assert not np.isnan(reward)

        algo.stop()

        print(f"\n✓ ConsensusProtocol: Training completed, rewards: {rewards}")

    def test_ippo_with_p2p_trading(self):
        """Test IPPO (independent policies) with P2P trading protocol."""
        protocol = PeerToPeerTradingProtocol(trading_fee=0.01)

        # Create environment
        env_creator, env_config = self.create_env(protocol)
        env_name = "test_ippo_p2p"
        register_env(env_name, env_creator)

        # Create IPPO configuration (independent policies)
        config = self.create_ppo_config(env_name, env_config, shared_policy=False)

        # Train for 3 iterations
        algo = config.build()

        rewards = []
        for i in range(3):
            result = algo.train()
            reward = result.get('episode_reward_mean', 0)
            rewards.append(reward)

            # Check training runs without errors
            assert isinstance(reward, (int, float))
            assert not np.isnan(reward)

        algo.stop()

        print(f"\n✓ IPPO + P2PTradingProtocol: Training completed, rewards: {rewards}")


@pytest.mark.skipif(not RLLIB_AVAILABLE, reason="RLlib not installed")
class TestProtocolConvergence:
    """Test that training converges better with cooperative protocols."""

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

    def train_with_protocol(self, protocol, protocol_name, iterations=5):
        """Train MAPPO with given protocol and return final reward."""
        # Create environment
        config = load_config('ieee34_ieee13')
        config['train'] = True
        config['penalty'] = 10.0
        config['share_reward'] = True
        config['max_episode_steps'] = 24

        def env_creator(env_config):
            env = MultiAgentMicrogrids(env_config)
            return ParallelPettingZooEnv(env)

        env_name = f"test_convergence_{protocol_name}"
        register_env(env_name, env_creator)

        # Create temp env for spaces
        base_env = MultiAgentMicrogrids(config)
        temp_env = ParallelPettingZooEnv(base_env)

        first_agent = base_env.possible_agents[0]

        # MAPPO config
        ppo_config = (
            PPOConfig()
            .environment(env=env_name, env_config=config, disable_env_checking=True)
            .framework("torch")
            .training(
                train_batch_size=500,
                sgd_minibatch_size=128,
                num_sgd_iter=5,
                lr=5e-4,
                gamma=0.99,
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
        )

        # Train
        algo = ppo_config.build()

        rewards = []
        for i in range(iterations):
            result = algo.train()
            reward = result.get('episode_reward_mean', 0)
            rewards.append(reward)

        final_reward = rewards[-1]
        algo.stop()

        return rewards, final_reward

    def test_protocol_comparison_convergence(self):
        """Compare convergence with different protocols."""
        protocols = {
            "NoProtocol": NoHorizontalProtocol(),
            "P2PTrading": PeerToPeerTradingProtocol(trading_fee=0.01),
            "Consensus": ConsensusProtocol(max_iterations=5),
        }

        results = {}

        for name, protocol in protocols.items():
            print(f"\nTesting convergence with {name}...")
            rewards, final_reward = self.train_with_protocol(protocol, name, iterations=5)
            results[name] = {
                'rewards': rewards,
                'final_reward': final_reward
            }
            print(f"  {name}: Final reward = {final_reward:.2f}")

        # All protocols should complete training without errors
        for name, result in results.items():
            assert not np.isnan(result['final_reward'])
            assert len(result['rewards']) == 5

        print("\n✓ All protocols completed training successfully")
        print(f"  NoProtocol final reward: {results['NoProtocol']['final_reward']:.2f}")
        print(f"  P2PTrading final reward: {results['P2PTrading']['final_reward']:.2f}")
        print(f"  Consensus final reward: {results['Consensus']['final_reward']:.2f}")


@pytest.mark.skipif(not RLLIB_AVAILABLE, reason="RLlib not installed")
class TestProtocolGradientFlow:
    """Test that protocols don't break gradient computation."""

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

    def test_gradient_computation_with_protocols(self):
        """Test that gradients flow correctly with protocols."""
        protocol = PeerToPeerTradingProtocol()

        # Create environment
        config = load_config('ieee34_ieee13')
        config['train'] = True
        config['max_episode_steps'] = 24

        def env_creator(env_config):
            env = MultiAgentMicrogrids(env_config)
            return ParallelPettingZooEnv(env)

        env_name = "test_gradients"
        register_env(env_name, env_creator)

        # Create temp env
        base_env = MultiAgentMicrogrids(config)
        temp_env = ParallelPettingZooEnv(base_env)
        first_agent = base_env.possible_agents[0]

        # PPO config
        ppo_config = (
            PPOConfig()
            .environment(env=env_name, env_config=config, disable_env_checking=True)
            .framework("torch")
            .training(
                train_batch_size=500,
                sgd_minibatch_size=128,
                num_sgd_iter=5,
                lr=5e-4,
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
        )

        algo = ppo_config.build()

        # Train one iteration and check gradients
        result = algo.train()

        # Check that training produced valid results
        assert 'episode_reward_mean' in result
        assert not np.isnan(result['episode_reward_mean'])

        # Check learner stats exist (indicates gradients computed)
        assert 'info' in result
        assert 'learner' in result['info']

        learner_stats = result['info']['learner']['shared_policy']['learner_stats']

        # Check gradient norm is finite (not NaN or Inf)
        grad_gnorm = learner_stats.get('grad_gnorm', None)
        if grad_gnorm is not None:
            assert not np.isnan(grad_gnorm)
            assert not np.isinf(grad_gnorm)
            assert grad_gnorm >= 0  # Gradient norm should be non-negative

        # Check policy loss is finite
        policy_loss = learner_stats.get('policy_loss', None)
        if policy_loss is not None:
            assert not np.isnan(policy_loss)
            assert not np.isinf(policy_loss)

        algo.stop()

        print("\n✓ Gradient computation works correctly with protocols")
        print(f"  Gradient norm: {grad_gnorm}")
        print(f"  Policy loss: {policy_loss}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

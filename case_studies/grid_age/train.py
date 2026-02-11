"""Training script for GridAges multi-microgrid environment.

This script demonstrates CTDE (Centralized Training with Decentralized Execution)
for training microgrid control policies.
"""

from typing import Dict
import numpy as np

from case_studies.grid_age.envs import MicrogridEnv
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.core.observation import Observation


class SimpleMLP:
    """Simple MLP for value function approximation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)

    def update(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01):
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        d_out = (out - target) * (1 - out**2)
        self.W2 -= lr * np.outer(h, d_out)
        self.b2 -= lr * d_out
        d_h = d_out @ self.W2.T
        d_h[h <= 0] = 0
        self.W1 -= lr * np.outer(x, d_h)
        self.b1 -= lr * d_h


class ActorMLP(SimpleMLP):
    """Actor network with tanh output for bounded actions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        # Smaller weights for actor
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def update(self, x: np.ndarray, action_taken: np.ndarray,
               advantage: float, lr: float = 0.01):
        """Update actor using policy gradient."""
        # Forward pass
        h = np.maximum(0, x @ self.W1 + self.b1)
        current_action = np.tanh(h @ self.W2 + self.b2)

        # Policy gradient
        error = current_action - action_taken
        grad_scale = advantage * (1 - current_action**2)

        # Backprop
        d_W2 = np.outer(h, grad_scale * error)
        d_b2 = grad_scale * error
        d_h = (grad_scale * error) @ self.W2.T
        d_h[h <= 0] = 0
        d_W1 = np.outer(x, d_h)
        d_b1 = d_h

        # Update
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2.flatten()
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1.flatten()


class NeuralPolicy(Policy):
    """Neural network policy for microgrid control.

    Architecture:
        obs → hidden (64) → action (4D, tanh activation)

    The policy learns to:
    - Minimize operational cost (fuel, grid purchases, cycling)
    - Avoid safety violations (voltage, line overloads, SOC bounds)
    """

    def __init__(self, obs_dim: int, action_dim: int = 4,
                 hidden_dim: int = 64, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = (-1.0, 1.0)  # Required by vector_to_action decorator
        self.hidden_dim = hidden_dim

        # Actor network for policy
        self.actor = ActorMLP(obs_dim, hidden_dim, action_dim, seed)

        # Critic network for value estimation
        self.critic = SimpleMLP(obs_dim, hidden_dim, 1, seed + 1)

        # Exploration noise
        self.noise_scale = 0.2

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute action with exploration noise.

        Decorators auto-convert: observation → obs_vec → action_vec → Action
        """
        action_mean = self.actor.forward(obs_vec)
        action_vec = action_mean + np.random.normal(0, self.noise_scale, self.action_dim)
        return np.clip(action_vec, -1.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward_deterministic(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute action without exploration noise."""
        return self.actor.forward(obs_vec)

    @obs_to_vector
    def get_value(self, obs_vec: np.ndarray) -> float:
        """Estimate value of current state."""
        return float(self.critic.forward(obs_vec)[0])

    def update(self, obs: np.ndarray, action_taken: np.ndarray,
               advantage: float, lr: float = 0.01):
        """Update policy using policy gradient with advantage."""
        self.actor.update(obs, action_taken, advantage, lr)

    def update_critic(self, obs: np.ndarray, target: float, lr: float = 0.01):
        """Update critic to better estimate values."""
        self.critic.update(obs, np.array([target]), lr)

    def decay_noise(self, decay_rate: float = 0.995, min_noise: float = 0.05):
        """Decay exploration noise over training."""
        self.noise_scale = max(min_noise, self.noise_scale * decay_rate)


def train_microgrid_ctde(
    env: MicrogridEnv,
    num_episodes: int = 100,
    steps_per_episode: int = 24,
    gamma: float = 0.99,
    lr: float = 0.02,
    print_every: int = 10,
) -> Dict[str, NeuralPolicy]:
    """Train microgrid policies using CTDE with policy gradient.

    Args:
        env: MicrogridEnv instance
        num_episodes: Number of training episodes
        steps_per_episode: Steps per episode (24 for 24-hour episodes)
        gamma: Discount factor
        lr: Learning rate
        print_every: Print progress every N episodes

    Returns:
        Dict mapping agent IDs to trained policies
    """
    # Get agent IDs that have action spaces
    agent_ids = [
        aid for aid, agent in env.registered_agents.items()
        if agent.action_space is not None
    ]

    print(f"Training {len(agent_ids)} agents: {agent_ids}")

    # Initialize environment to get observation dimensions
    obs, _ = env.reset(seed=0)

    # Get observation dimension from first agent
    # Use the full observation vector from Observation.vector()
    first_obs = obs[agent_ids[0]]
    if isinstance(first_obs, Observation):
        obs_vec = first_obs.vector()  # Get full observation vector
    else:
        obs_vec = first_obs
    obs_dim = obs_vec.shape[0] if hasattr(obs_vec, 'shape') else len(obs_vec)

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: 4 (P_ess, P_dg, Q_pv, Q_wind)")

    # Initialize policies for each agent
    policies = {
        aid: NeuralPolicy(obs_dim=obs_dim, seed=42 + i)
        for i, aid in enumerate(agent_ids)
    }

    # Training metrics
    returns_history = []
    avg_rewards_history = []

    print("\nStarting training...")
    print("=" * 60)

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)

        # Collect trajectories
        trajectories = {
            aid: {"obs": [], "actions": [], "rewards": []}
            for aid in agent_ids
        }

        episode_return = 0.0

        # Run episode
        for step in range(steps_per_episode):
            actions = {}

            # Collect actions from each agent
            for aid in agent_ids:
                obs_value = obs[aid]

                # Extract full observation vector
                if isinstance(obs_value, Observation):
                    obs_vec = obs_value.vector()  # Get full concatenated observation
                    observation = obs_value
                else:
                    obs_vec = obs_value
                    observation = Observation(timestamp=step, local={"obs": obs_vec})

                # Get action from policy
                action = policies[aid].forward(observation)
                actions[aid] = action

                # Store for training
                trajectories[aid]["obs"].append(obs_vec)
                trajectories[aid]["actions"].append(action.c.copy())

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Store rewards
            for aid in agent_ids:
                if aid in rewards:
                    trajectories[aid]["rewards"].append(rewards[aid])
                    episode_return += rewards[aid]

            # Check termination
            if terminated.get("__all__", False) or all(
                terminated.get(aid, False) for aid in agent_ids
            ):
                break

        # Update policies using collected trajectories
        for aid, traj in trajectories.items():
            if not traj["rewards"]:
                continue

            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(traj["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)
            returns = np.array(returns)

            # Update policy with advantage
            for t in range(len(traj["obs"])):
                obs_t = traj["obs"][t]
                baseline = policies[aid].get_value(
                    Observation(timestamp=t, local={"obs": obs_t})
                )
                advantage = returns[t] - baseline
                policies[aid].update(obs_t, traj["actions"][t], advantage, lr=lr)
                policies[aid].update_critic(obs_t, returns[t], lr=lr)

            # Decay exploration noise
            policies[aid].decay_noise()

        # Track metrics
        returns_history.append(episode_return)
        avg_reward = episode_return / (len(agent_ids) * steps_per_episode)
        avg_rewards_history.append(avg_reward)

        # Print progress
        if (episode + 1) % print_every == 0:
            recent_returns = returns_history[-print_every:]
            recent_avg = np.mean(recent_returns)
            print(f"Episode {episode + 1:3d}/{num_episodes}: "
                  f"avg_return={recent_avg:.2f}, "
                  f"avg_reward={np.mean(avg_rewards_history[-print_every:]):.3f}")

    print("=" * 60)
    print("Training complete!")
    print(f"Initial avg return: {np.mean(returns_history[:10]):.2f}")
    print(f"Final avg return:   {np.mean(returns_history[-10:]):.2f}")

    return policies


def main():
    """Main training function."""
    # Create environment
    print("Creating microgrid environment...")
    env = MicrogridEnv(
        num_microgrids=3,
        episode_steps=24,
        dt=1.0,
    )

    # Train policies
    policies = train_microgrid_ctde(
        env,
        num_episodes=100,
        steps_per_episode=24,
        lr=0.02,
        print_every=10,
    )

    # Test trained policies
    print("\nTesting trained policies (deterministic)...")
    env.set_agent_policies(policies)

    obs, _ = env.reset(seed=999)
    total_reward = 0.0

    for step in range(24):
        actions = {}
        for aid, policy in policies.items():
            obs_value = obs[aid]
            # Use Observation object directly with forward_deterministic
            if isinstance(obs_value, Observation):
                action = policy.forward_deterministic(obs_value)
            else:
                action = policy.forward_deterministic(
                    Observation(timestamp=step, local={"obs": obs_value})
                )
            actions[aid] = action

        obs, rewards, terminated, truncated, info = env.step(actions)
        step_reward = sum(rewards.values())
        total_reward += step_reward

        if terminated.get("__all__", False):
            break

    print(f"Test episode total reward: {total_reward:.2f}")
    print(f"Test episode avg reward:   {total_reward / (len(policies) * 24):.3f}")


if __name__ == "__main__":
    main()

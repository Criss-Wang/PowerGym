"""
Action passing through agent hierarchy with protocols.

This script demonstrates and tests action coordination via protocols:
- Coordinator owns neural policy
- Coordinator computes joint action
- Protocol distributes actions to field agents (via .coordinate)
- Training with CTDE (centralized training, decentralized execution)
- Event-driven execution with asynchronous timing
"""

import numpy as np
from typing import Any, Dict, List, Optional

# HERON imports
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.core.observation import Observation
from heron.core.feature import FeatureProvider
from heron.core.action import Action
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.base import MultiAgentEnv
from heron.protocols.base import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
)
from heron.protocols.vertical import SetpointProtocol
from heron.utils.typing import AgentID
from heron.scheduling import EventScheduler, TickConfig, JitterType
from heron.scheduling.analysis import EventAnalyzer


# =============================================================================
# Custom Action Protocol - Proportional Distribution
# =============================================================================

class ProportionalActionProtocol(ActionProtocol):
    """Distributes coordinator action proportionally based on weights."""

    def __init__(self, distribution_weights: Optional[Dict[AgentID, float]] = None):
        self.distribution_weights = distribution_weights or {}

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Optional[Dict[AgentID, Any]] = None,
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        """Distribute coordinator action proportionally among subordinates."""
        if coordinator_action is None or subordinate_states is None:
            return {sub_id: None for sub_id in (subordinate_states or {})}

        # Extract action value from Action object or array
        if hasattr(coordinator_action, 'c'):
            total_action = float(coordinator_action.c[0]) if len(coordinator_action.c) > 0 else 0.0
        elif isinstance(coordinator_action, np.ndarray):
            total_action = float(coordinator_action[0]) if len(coordinator_action) > 0 else 0.0
        else:
            total_action = float(coordinator_action)

        # Compute weights (use subordinate IDs from subordinate_states)
        sub_ids = list(subordinate_states.keys())
        if not sub_ids:
            return {}

        if not self.distribution_weights:
            weights = {sub_id: 1.0 / len(sub_ids) for sub_id in sub_ids}
        else:
            total_weight = sum(self.distribution_weights.get(sub_id, 0.0) for sub_id in sub_ids)
            if total_weight == 0:
                # Equal distribution if weights sum to zero
                weights = {sub_id: 1.0 / len(sub_ids) for sub_id in sub_ids}
            else:
                weights = {
                    sub_id: self.distribution_weights.get(sub_id, 0.0) / total_weight
                    for sub_id in sub_ids
                }

        # Distribute action proportionally
        actions = {}
        for sub_id in sub_ids:
            proportional_action = total_action * weights[sub_id]
            actions[sub_id] = np.array([proportional_action])

        # Only print during event-driven (when we have context with subordinates)
        # During training, context might not have subordinates
        if context and "subordinates" in context:
            print(f"[ProportionalProtocol] Distributing action {total_action:.4f} -> {[(sid, f'{a[0]:.4f}') for sid, a in actions.items()]}")
            print(f"  Weights: {weights}")
        return actions


class ProportionalProtocol(Protocol):
    """Protocol with proportional action distribution."""

    def __init__(self, distribution_weights: Optional[Dict[AgentID, float]] = None):
        from heron.protocols.base import NoCommunication
        super().__init__(
            communication_protocol=NoCommunication(),
            action_protocol=ProportionalActionProtocol(distribution_weights)
        )

    def coordinate(self, coordinator_state, coordinator_action=None, info_for_subordinates=None, context=None):
        """Override to add debug output."""
        print(f"[ProportionalProtocol.coordinate] Called with action={coordinator_action}, subordinates={list(info_for_subordinates.keys()) if info_for_subordinates else []}")
        return super().coordinate(coordinator_state, coordinator_action, info_for_subordinates, context)


# =============================================================================
# Features and Agents
# =============================================================================

class DevicePowerFeature(FeatureProvider):
    """Power state feature for devices."""
    visibility = ["public"]

    power: float = 0.0
    capacity: float = 1.0

    def vector(self):
        """Return feature as vector [power, capacity]."""
        return np.array([self.power, self.capacity], dtype=np.float32)

    def set_values(self, **kwargs: Any) -> None:
        if "power" in kwargs:
            self.power = np.clip(kwargs["power"], -self.capacity, self.capacity)
        if "capacity" in kwargs:
            self.capacity = kwargs["capacity"]


class DeviceAgent(FieldAgent):
    """Device field agent - receives actions from coordinator via protocol."""

    @property
    def power(self) -> float:
        return self.state.features[0].power

    @property
    def capacity(self) -> float:
        return self.state.features[0].capacity

    def init_action(self, features: List[FeatureProvider] = []):
        """Initialize action (power control)."""
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def compute_local_reward(self, local_state: dict) -> float:
        """Reward = maintain power near zero (minimize deviation)."""
        if "DevicePowerFeature" in local_state:
            feature_vec = local_state["DevicePowerFeature"]
            power = float(feature_vec[0])
            return -power ** 2  # Penalize deviation from zero
        return 0.0

    def set_action(self, action: Any) -> None:
        """Set action from Action object or array."""
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        elif isinstance(action, np.ndarray):
            self.action.set_values(action)
        else:
            self.action.set_values(np.array([action]))

    def set_state(self) -> None:
        """Update power based on action (direct setpoint control)."""
        # Action directly sets power (not incremental change)
        # This makes it easier to learn: reward = -power^2, and action controls power directly
        new_power = self.action.c[0] * 0.5  # Scale action to reasonable power range
        self.state.features[0].set_values(power=new_power)

    def apply_action(self):
        self.set_state()


class ZoneCoordinator(CoordinatorAgent):
    """Coordinator - owns policy and distributes actions via protocol."""
    pass


class GridSystem(SystemAgent):
    """System agent."""
    pass


# =============================================================================
# Environment
# =============================================================================

class EnvState:
    def __init__(self):
        self.timestep = 0


class ActionPassingEnv(MultiAgentEnv):
    """Environment for testing action passing through protocols."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Simple physics simulation."""
        env_state.timestep += 1
        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        """Convert env state to global state."""
        agent_states = {}
        for agent_id, agent in self.registered_agents.items():
            if hasattr(agent, 'level') and agent.level == 1 and agent.state:
                state_dict = agent.state.to_dict(include_metadata=True)
                agent_states[agent_id] = state_dict
        return {"agent_states": agent_states}

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        """Convert global state to env state."""
        return EnvState()


# =============================================================================
# Neural Policy for Coordinator
# =============================================================================

class SimpleMLP:
    """Simple MLP for value function approximation."""
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)

    def update(self, x, target, lr=0.01):
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
    """Actor network with tanh output."""
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        # Initialize with zero bias so initial actions are near 0 (not biased positive/negative)
        self.b2 = np.zeros(output_dim)

    def update(self, x, action_taken, advantage, lr=0.01):
        """Update actor using policy gradient."""
        h = np.maximum(0, x @ self.W1 + self.b1)
        current_action = np.tanh(h @ self.W2 + self.b2)

        error = current_action - action_taken
        grad_scale = advantage * (1 - current_action**2)

        d_W2 = np.outer(h, grad_scale * error)
        d_b2 = grad_scale * error
        d_h = (grad_scale * error) @ self.W2.T
        d_h[h <= 0] = 0
        d_W1 = np.outer(x, d_h)
        d_b1 = d_h

        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2.flatten()
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1.flatten()


class CoordinatorNeuralPolicy(Policy):
    """Neural policy for coordinator that computes joint action.

    The coordinator observes all subordinate states (aggregated) and outputs
    a single action that will be distributed to subordinates via protocol.
    """
    def __init__(self, obs_dim, action_dim=1, hidden_dim=32, seed=42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = (-1.0, 1.0)
        self.hidden_dim = hidden_dim

        self.actor = ActorMLP(obs_dim, hidden_dim, action_dim, seed)
        self.critic = SimpleMLP(obs_dim, hidden_dim, 1, seed + 1)

        self.noise_scale = 0.15

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute joint action with exploration noise."""
        action_mean = self.actor.forward(obs_vec)
        action_vec = action_mean + np.random.normal(0, self.noise_scale, self.action_dim)
        action_clipped = np.clip(action_vec, -1.0, 1.0)
        return action_clipped

    @obs_to_vector
    @vector_to_action
    def forward_deterministic(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute joint action without exploration noise."""
        return self.actor.forward(obs_vec)

    @obs_to_vector
    def get_value(self, obs_vec: np.ndarray) -> float:
        """Estimate value of current state."""
        return float(self.critic.forward(obs_vec)[0])

    def update(self, obs, action_taken, advantage, lr=0.01):
        """Update policy using policy gradient with advantage."""
        self.actor.update(obs, action_taken, advantage, lr)

    def update_critic(self, obs, target, lr=0.01):
        """Update critic to better estimate values."""
        self.critic.update(obs, np.array([target]), lr)

    def decay_noise(self, decay_rate=0.995, min_noise=0.05):
        """Decay exploration noise over training."""
        self.noise_scale = max(min_noise, self.noise_scale * decay_rate)


# =============================================================================
# CTDE Training Loop
# =============================================================================

def train_ctde(env: MultiAgentEnv, num_episodes=100, steps_per_episode=50, gamma=0.99, lr=0.01):
    """Train coordinator policy using CTDE with action distribution via protocol.

    Key: Coordinator computes joint action, protocol distributes to field agents.
    """
    # Get field agent IDs (agents that receive distributed actions)
    agent_ids = [aid for aid, agent in env.registered_agents.items() if agent.action_space is not None]

    obs, _ = env.reset(seed=0)

    # Observation dimension: aggregate all field agent observations
    # Coordinator observes all subordinates
    first_obs = obs[agent_ids[0]]
    local_vec = list(first_obs.local.values())[0] if first_obs.local else np.array([])
    obs_dim_per_agent = local_vec.shape[0] if hasattr(local_vec, 'shape') else 0
    obs_dim = obs_dim_per_agent * len(agent_ids)  # Coordinator sees all agents

    print(f"Training coordinator with obs_dim={obs_dim} (aggregated from {len(agent_ids)} agents)")

    # Coordinator policy (not per-agent policies)
    coordinator_policy = CoordinatorNeuralPolicy(obs_dim=obs_dim, action_dim=1, seed=42)

    returns_history, power_history = [], []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)

        # Debug: Check initial power values
        if episode == 0:
            for aid in agent_ids:
                agent = env.registered_agents[aid]
                if hasattr(agent, 'power'):
                    print(f"  Initial power for {aid}: {agent.power:.4f}")

        # Track trajectories for coordinator (not individual agents)
        trajectories = {"obs": [], "actions": [], "rewards": []}
        episode_return = 0.0
        power_values = []

        for step in range(steps_per_episode):
            # Coordinator observes all subordinates (aggregate observation)
            aggregated_obs = []
            for aid in agent_ids:
                obs_value = obs[aid]
                if isinstance(obs_value, Observation):
                    local_features = list(obs_value.local.values())
                    obs_vec = local_features[0] if local_features else np.array([])
                else:
                    obs_vec = obs_value[:obs_dim_per_agent]
                aggregated_obs.append(obs_vec)

            aggregated_obs_vec = np.concatenate(aggregated_obs)
            coordinator_observation = Observation(timestamp=step, local={"obs": aggregated_obs_vec})

            # Coordinator computes joint action
            coordinator_action = coordinator_policy.forward(coordinator_observation)

            # Protocol distributes coordinator action to field agents
            # Get coordinator from environment
            coordinator_agent = env.registered_agents.get("coordinator")
            if coordinator_agent and coordinator_agent.protocol:
                # Use protocol to distribute actions (matching event-driven behavior)
                _, distributed_actions = coordinator_agent.protocol.coordinate(
                    coordinator_state=coordinator_agent.state,
                    coordinator_action=coordinator_action,
                    info_for_subordinates={aid: obs[aid] for aid in agent_ids},
                    context={"subordinates": coordinator_agent.subordinates}
                )
                actions = distributed_actions
            else:
                # Fallback: use coordinator action directly
                actions = {aid: coordinator_action for aid in agent_ids}

            # Store coordinator's action (not individual actions)
            trajectories["obs"].append(aggregated_obs_vec)
            trajectories["actions"].append(coordinator_action.c.copy())

            obs, rewards, terminated, _, info = env.step(actions)

            # Aggregate rewards from all field agents
            total_reward = sum(rewards.get(aid, 0) for aid in agent_ids)
            trajectories["rewards"].append(total_reward)
            episode_return += total_reward

            # Track power values
            for aid in agent_ids:
                if aid in obs:
                    obs_value = obs[aid]
                    if isinstance(obs_value, Observation):
                        power_values.append(obs_value.vector()[0])
                    else:
                        power_values.append(obs_value[0])

            if terminated.get("__all__", False) or all(terminated.get(aid, False) for aid in agent_ids):
                break

        # Update coordinator policy
        if trajectories["rewards"]:
            returns = []
            G = 0
            for r in reversed(trajectories["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)
            returns = np.array(returns)

            for t in range(len(trajectories["obs"])):
                obs_t = trajectories["obs"][t]
                baseline = coordinator_policy.get_value(Observation(timestamp=t, local={"obs": obs_t}))
                advantage = returns[t] - baseline
                coordinator_policy.update(obs_t, trajectories["actions"][t], advantage, lr=lr)
                coordinator_policy.update_critic(obs_t, returns[t], lr=lr)

            coordinator_policy.decay_noise()

        returns_history.append(episode_return)
        power_history.append(np.mean(power_values) if power_values else 0)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d}: return={episode_return:.1f}, avg_power={np.mean(power_values) if power_values else 0:.4f}")

    return coordinator_policy, returns_history, power_history


# =============================================================================
# Main Execution
# =============================================================================

# Create devices
device_1 = DeviceAgent(
    agent_id="device_1",
    features=[DevicePowerFeature(power=0.0, capacity=1.0)]
)
device_2 = DeviceAgent(
    agent_id="device_2",
    features=[DevicePowerFeature(power=0.0, capacity=1.0)]
)

# Coordinator with ProportionalProtocol
# Coordinator will compute action and distribute via protocol
weights = {"device_1": 1.0, "device_2": 1.0}  # Equal distribution
proportional_protocol = ProportionalProtocol(distribution_weights=weights)
coordinator = ZoneCoordinator(
    agent_id="coordinator",
    subordinates={"device_1": device_1, "device_2": device_2},
)
# WORKAROUND: CoordinatorAgent.__init__ doesn't pass protocol to super().__init__(),
# so base Agent overwrites it with None. We need to set it again after init.
coordinator.protocol = proportional_protocol

system = GridSystem(
    agent_id="system_agent",
    subordinates={"coordinator": coordinator}
)

# Configure environment
scheduler_config = {
    "start_time": 0.0,
    "time_step": 1.0,
}

message_broker_config = {
    "buffer_size": 1000,
    "max_queue_size": 100,
}

env = ActionPassingEnv(
    system_agent=system,
    scheduler_config=scheduler_config,
    message_broker_config=message_broker_config,
    simulation_wait_interval=0.01,
)

# Run training
print("="*80)
print("CTDE Training with Action Distribution via Protocol")
print("="*80)
print("Coordinator computes joint action → Protocol distributes to field agents")
print(f"Protocol: ProportionalProtocol with weights {weights}")
print()

coordinator_policy, returns, avg_powers = train_ctde(
    env,
    num_episodes=50,  # Reduced for faster testing
    steps_per_episode=30,
    lr=0.02
)

initial_avg_power = np.mean(avg_powers[:10])
final_avg_power = np.mean(avg_powers[-10:])
initial_return = np.mean(returns[:10])
final_return = np.mean(returns[-10:])

print(f"\nTraining Results:")
print(f"  Initial avg power: {initial_avg_power:.4f} (return: {initial_return:.2f})")
print(f"  Final avg power:   {final_avg_power:.4f} (return: {final_return:.2f})")
print(f"  Return improvement: {final_return - initial_return:.2f}")
print(f"  Power closer to zero: {abs(initial_avg_power) > abs(final_avg_power)}")

# Attach trained policy to coordinator
print(f"\nAttaching trained policy to coordinator for event-driven execution...")
coordinator.policy = coordinator_policy

# Configure tick timing for event-driven execution
field_tick_config = TickConfig.with_jitter(
    tick_interval=5.0,
    obs_delay=0.1,
    act_delay=0.2,
    msg_delay=0.1,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=42
)

coordinator_tick_config = TickConfig.with_jitter(
    tick_interval=10.0,
    obs_delay=0.2,
    act_delay=0.3,
    msg_delay=0.15,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=43
)

system_tick_config = TickConfig.with_jitter(
    tick_interval=15.0,
    obs_delay=0.3,
    act_delay=0.5,
    msg_delay=0.2,
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,
    seed=44
)

device_1.tick_config = field_tick_config
device_2.tick_config = field_tick_config
coordinator.tick_config = coordinator_tick_config
system.tick_config = system_tick_config

# Run event-driven simulation
print("\n" + "="*80)
print("Event-Driven Execution with Trained Policy")
print("="*80)
print("Coordinator uses trained policy to compute actions")
print("Protocol distributes actions to devices asynchronously\n")

event_analyzer = EventAnalyzer(verbose=True, track_data=True)
episode = env.run_event_driven(event_analyzer=event_analyzer, t_end=300.0)

print(f"\n{'='*80}")
print("Action Passing Test Complete")
print(f"{'='*80}")
print(f"\nKey Points:")
print(f"  1. Coordinator owns neural policy and computes joint actions")
print(f"  2. Protocol.coordinate() distributes actions to field agents")
print(f"  3. Actions flow through hierarchy: System → Coordinator → Devices")
print(f"  4. Event-driven execution with asynchronous timing and jitter")
print()

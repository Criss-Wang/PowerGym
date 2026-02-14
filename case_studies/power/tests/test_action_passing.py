"""Action passing test for PowerGrid case study.

This script tests action coordination via protocols in power grid hierarchy:
- PowerGridAgent (Coordinator) owns neural policy
- Coordinator computes joint action for all devices
- Protocol distributes actions to Generator and ESS agents
- Training with CTDE (centralized training, decentralized execution)
- Event-driven execution with asynchronous timing

Uses the grid_age style environment structure.
"""

import numpy as np
from typing import Any, Dict, List, Optional

# HERON imports
from heron.agents.system_agent import SystemAgent
from heron.core.observation import Observation
from heron.core.action import Action
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.base import MultiAgentEnv
from heron.protocols.base import Protocol, ActionProtocol, NoCommunication
from heron.protocols.vertical import VerticalProtocol
from heron.utils.typing import AgentID
from heron.scheduling import TickConfig, JitterType
from heron.scheduling.analysis import EventAnalyzer

# PowerGrid imports
from powergrid.agents import (
    PowerGridAgent,
    Generator,
    ESS,
)
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.storage import StorageBlock
from powergrid.core.features.metrics import CostSafetyMetrics
from powergrid.envs.common import EnvState


# =============================================================================
# Custom Action Protocol - Power Dispatch
# =============================================================================

class PowerDispatchProtocol(ActionProtocol):
    """Dispatches coordinator action to devices based on capacity weights.

    Used by PowerGridAgent to distribute power setpoints to generators and storage.
    """

    def __init__(self, capacity_weights: Optional[Dict[AgentID, float]] = None):
        self.capacity_weights = capacity_weights or {}

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Optional[Dict[AgentID, Any]] = None,
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        """Distribute coordinator action based on device capacities."""
        if coordinator_action is None or subordinate_states is None:
            # breakpoint()
            return {sub_id: None for sub_id in (subordinate_states or {})}

        # Extract action value
        if hasattr(coordinator_action, 'scalar'):
            total_action = coordinator_action.scalar()
        elif isinstance(coordinator_action, np.ndarray):
            total_action = float(coordinator_action[0]) if len(coordinator_action) > 0 else 0.0
        else:
            total_action = float(coordinator_action)

        sub_ids = list(subordinate_states.keys())
        if not sub_ids:
            return {}

        # Compute weights based on capacity
        if not self.capacity_weights:
            weights = {sub_id: 1.0 / len(sub_ids) for sub_id in sub_ids}
        else:
            total_weight = sum(self.capacity_weights.get(sub_id, 1.0) for sub_id in sub_ids)
            weights = {
                sub_id: self.capacity_weights.get(sub_id, 1.0) / total_weight
                for sub_id in sub_ids
            }

        # Distribute action
        actions = {}
        for sub_id in sub_ids:
            proportional_action = total_action * weights[sub_id]
            actions[sub_id] = np.array([proportional_action])

        if context and "subordinates" in context:
            print(f"[PowerDispatch] Total={total_action:.4f} -> {[(s, f'{a[0]:.4f}') for s, a in actions.items()]}")

        return actions


class MicrogridDispatchProtocol(Protocol):
    """Protocol for microgrid power dispatch."""

    def __init__(self, capacity_weights: Optional[Dict[AgentID, float]] = None):
        super().__init__(
            communication_protocol=NoCommunication(),
            action_protocol=PowerDispatchProtocol(capacity_weights)
        )


# =============================================================================
# Test Environment
# =============================================================================

class ActionPassingTestEnv(MultiAgentEnv):
    """Test environment for action passing in power grid hierarchy."""

    def __init__(self, system_agent: SystemAgent, **kwargs):
        self.episode_steps = kwargs.pop("episode_steps", 24)
        self._timestep = 0

        super().__init__(system_agent=system_agent, **kwargs)

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        """Convert global state to env state."""
        env_state = EnvState()

        agent_states = global_state.get("agent_states", {})
        for agent_id, state_dict in agent_states.items():
            features = state_dict.get("features", {})
            if "ElectricalBasePh" in features:
                elec = features["ElectricalBasePh"]
                env_state.set_device_setpoint(
                    agent_id,
                    P=elec.get("P_MW", 0.0),
                    Q=elec.get("Q_MVAr", 0.0),
                )

        return env_state

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Run power flow simulation."""
        total_power = 0.0
        for device_id, setpoint in env_state.device_setpoints.items():
            P = np.clip(setpoint.get("P", 0.0), -1.0, 1.0)
            total_power += P
            env_state.device_setpoints[device_id]["P"] = P

        # Power balance check
        imbalance = abs(total_power)
        env_state.update_power_flow_results({
            "converged": True,
            "total_power": total_power,
            "imbalance": imbalance,
        })

        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        """Convert env state back to global state."""
        from heron.agents.system_agent import SYSTEM_AGENT_ID

        global_state = self.proxy_agent.get_global_states(
            sender_id=SYSTEM_AGENT_ID, protocol=None
        )
        agent_states = global_state if isinstance(global_state, dict) else {}

        for agent_id, setpoint in env_state.device_setpoints.items():
            if agent_id in agent_states:
                if "features" not in agent_states[agent_id]:
                    agent_states[agent_id]["features"] = {}
                agent_states[agent_id]["features"]["ElectricalBasePh"] = {
                    "P_MW": setpoint.get("P", 0.0),
                    "Q_MVAr": setpoint.get("Q", 0.0),
                }

        return {"agent_states": agent_states}

    def _pre_step(self, _actions: Dict[str, Any]) -> None:
        self._timestep += 1


# =============================================================================
# Neural Policy for Coordinator
# =============================================================================

class SimpleMLP:
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
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def update(self, x, action_taken, advantage, lr=0.01):
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

    Outputs action vector that will be distributed to subordinates via protocol.
    """
    def __init__(self, obs_dim, action_dim=2, hidden_dim=32, seed=42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = (-1.0, 1.0)
        self.hidden_dim = hidden_dim
        self.actor = ActorMLP(obs_dim, hidden_dim, action_dim, seed)
        self.critic = SimpleMLP(obs_dim, hidden_dim, 1, seed + 1)
        self.noise_scale = 0.15

    def _normalize_obs(self, obs_vec: np.ndarray) -> np.ndarray:
        """Ensure observation vector matches expected dimension."""
        if len(obs_vec) > self.obs_dim:
            return obs_vec[:self.obs_dim]
        elif len(obs_vec) < self.obs_dim:
            return np.pad(obs_vec, (0, self.obs_dim - len(obs_vec)))
        return obs_vec

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        obs_vec = self._normalize_obs(obs_vec)
        action_mean = self.actor.forward(obs_vec)
        action_vec = action_mean + np.random.normal(0, self.noise_scale, self.action_dim)
        return np.clip(action_vec, -1.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward_deterministic(self, obs_vec: np.ndarray) -> np.ndarray:
        obs_vec = self._normalize_obs(obs_vec)
        return self.actor.forward(obs_vec)

    @obs_to_vector
    def get_value(self, obs_vec: np.ndarray) -> float:
        obs_vec = self._normalize_obs(obs_vec)
        return float(self.critic.forward(obs_vec)[0])

    def update(self, obs, action_taken, advantage, lr=0.01):
        self.actor.update(obs, action_taken, advantage, lr)

    def update_critic(self, obs, target, lr=0.01):
        self.critic.update(obs, np.array([target]), lr)

    def decay_noise(self, decay_rate=0.995, min_noise=0.05):
        self.noise_scale = max(min_noise, self.noise_scale * decay_rate)


# =============================================================================
# CTDE Training
# =============================================================================

def train_ctde_with_protocol(
    env: MultiAgentEnv,
    num_episodes=50,
    steps_per_episode=24,
    gamma=0.99,
    lr=0.01
):
    """Train coordinator policy with action distribution via protocol."""
    # Get field agent IDs
    agent_ids = [aid for aid, agent in env.registered_agents.items()
                 if agent.action_space is not None]

    obs, _ = env.reset(seed=0)

    # Observation dimension (aggregate all agents using full local_vector)
    obs_dims_per_agent = {}
    for aid in agent_ids:
        agent_obs = obs[aid]
        if isinstance(agent_obs, Observation):
            obs_dims_per_agent[aid] = agent_obs.local_vector().shape[0]
        else:
            obs_dims_per_agent[aid] = len(agent_obs)
    obs_dim = sum(obs_dims_per_agent.values())

    print(f"Training coordinator: obs_dim={obs_dim}, num_devices={len(agent_ids)}")
    print(f"  Per-agent dims: {obs_dims_per_agent}")

    # Coordinator policy outputs joint action (one component per device)
    coordinator_policy = CoordinatorNeuralPolicy(
        obs_dim=obs_dim,
        action_dim=len(agent_ids),
        seed=42
    )

    returns_history = []
    power_history = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        trajectories = {"obs": [], "actions": [], "rewards": []}
        episode_return = 0.0
        power_values = []

        for step in range(steps_per_episode):
            # Coordinator observes all subordinates using full local vectors
            aggregated_obs = []
            for aid in agent_ids:
                obs_value = obs[aid]
                agent_dim = obs_dims_per_agent[aid]
                if isinstance(obs_value, Observation):
                    obs_vec = obs_value.local_vector()
                else:
                    obs_vec = np.asarray(obs_value, dtype=np.float32)
                # Ensure consistent dimensions
                if len(obs_vec) > agent_dim:
                    obs_vec = obs_vec[:agent_dim]
                elif len(obs_vec) < agent_dim:
                    obs_vec = np.pad(obs_vec, (0, agent_dim - len(obs_vec)))
                aggregated_obs.append(obs_vec)

            aggregated_obs_vec = np.concatenate(aggregated_obs)
            coordinator_observation = Observation(timestamp=step, local={"obs": aggregated_obs_vec})

            # Coordinator computes joint action
            coordinator_action = coordinator_policy.forward(coordinator_observation)

            # Get coordinator from env and use protocol to distribute actions
            coordinator_agent = env.registered_agents.get("MG1")
            if coordinator_agent and coordinator_agent.protocol:
                # Protocol already has subordinate info registered at setup time
                _, distributed_actions = coordinator_agent.protocol.coordinate(
                    coordinator_state=coordinator_agent.state,
                    coordinator_action=coordinator_action,
                    info_for_subordinates={aid: obs[aid] for aid in agent_ids},
                )
                actions = distributed_actions
            else:
                # Fallback: use VerticalProtocol's vector decomposition
                actions = {}
                for i, aid in enumerate(agent_ids):
                    if i < len(coordinator_action.c):
                        actions[aid] = Action()
                        actions[aid].set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
                        actions[aid].set_values(np.array([coordinator_action.c[i]]))
                    else:
                        actions[aid] = coordinator_action

            trajectories["obs"].append(aggregated_obs_vec)
            trajectories["actions"].append(coordinator_action.c.copy())

            obs, rewards, terminated, _, info = env.step(actions)

            # Aggregate rewards
            total_reward = sum(rewards.get(aid, 0) for aid in agent_ids)
            trajectories["rewards"].append(total_reward)
            episode_return += total_reward

            # Track action values (power setpoints)
            for i, aid in enumerate(agent_ids):
                if i < len(coordinator_action.c):
                    power_values.append(coordinator_action.c[i])

            if terminated.get("__all__", False):
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
                baseline = coordinator_policy.get_value(
                    Observation(timestamp=t, local={"obs": obs_t})
                )
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

def main():
    print("=" * 80)
    print("PowerGrid Action Passing Test")
    print("=" * 80)
    print("Testing action coordination via protocols:")
    print("  - PowerGridAgent computes joint action")
    print("  - Protocol distributes to Generator and ESS")
    print()

    # Create device agents
    gen1 = Generator(
        agent_id="MG1_Gen1",
        bus="MG1_bus",
        p_min_MW=0.1,
        p_max_MW=1.0,
        cost_curve_coefs=(0.01, 0.5, 0.0),  # Quadratic cost curve for meaningful rewards
    )

    ess1 = ESS(
        agent_id="MG1_ESS1",
        bus="MG1_bus",
        capacity_MWh=2.0,
        p_min_MW=-0.5,
        p_max_MW=0.5,
        degr_cost_per_MWh=0.1,  # Degradation cost for meaningful rewards
    )

    # Configure tick timing
    field_tick_config = TickConfig.with_jitter(
        tick_interval=2.0,
        obs_delay=0.05,
        act_delay=0.1,
        msg_delay=0.05,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=42
    )

    coordinator_tick_config = TickConfig.with_jitter(
        tick_interval=4.0,
        obs_delay=0.1,
        act_delay=0.15,
        msg_delay=0.075,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=43
    )

    gen1.tick_config = field_tick_config
    ess1.tick_config = field_tick_config

    # Create coordinator with VerticalProtocol
    vertical_protocol = VerticalProtocol()
    mg1 = PowerGridAgent(
        agent_id="MG1",
        subordinates={"MG1_Gen1": gen1, "MG1_ESS1": ess1},
        tick_config=coordinator_tick_config,
        protocol=vertical_protocol,  # Pass protocol during init for proper registration
    )

    # Create system agent
    system_agent = SystemAgent(
        agent_id="system_agent",
        subordinates={"MG1": mg1},
    )

    # Create environment
    env = ActionPassingTestEnv(
        system_agent=system_agent,
        episode_steps=24,
    )

    print(f"Environment created with agents:")
    for aid in env.registered_agents:
        print(f"  - {aid}")

    # Run CTDE training
    print("\n" + "-" * 40)
    print("CTDE Training with Protocol-Based Action Distribution")
    print("-" * 40)

    coordinator_policy, returns, powers = train_ctde_with_protocol(
        env,
        num_episodes=30,
        steps_per_episode=24,
        lr=0.02
    )

    print(f"\nTraining Results:")
    print(f"  Initial avg return: {np.mean(returns[:5]):.1f}")
    print(f"  Final avg return:   {np.mean(returns[-5:]):.1f}")
    print(f"  Initial avg power:  {np.mean(powers[:5]):.4f}")
    print(f"  Final avg power:    {np.mean(powers[-5:]):.4f}")

    # Attach policy and run event-driven
    print("\n" + "-" * 40)
    print("Event-Driven Execution with Action Distribution")
    print("-" * 40)

    # Set coordinator policy
    env_coordinator = env.registered_agents.get("MG1")
    if env_coordinator:
        env_coordinator.policy = coordinator_policy
        print(f"Policy attached to coordinator: {env_coordinator.agent_id}")

    event_analyzer = EventAnalyzer(verbose=False, track_data=True)
    env.run_event_driven(event_analyzer=event_analyzer, t_end=50.0)

    print(f"\nEvent-Driven Statistics:")
    print(f"  Observations: {event_analyzer.observation_count}")
    print(f"  State updates: {event_analyzer.state_update_count}")
    print(f"  Action results: {event_analyzer.action_result_count}")

    # Validation
    print("\n" + "-" * 40)
    print("Validation")
    print("-" * 40)

    print(f"Action Passing Flow:")
    print(f"  1. SystemAgent -> PowerGridAgent (MG1)")
    print(f"  2. MG1 computes joint action via neural policy")
    print(f"  3. VerticalProtocol decomposes: [a1, a2] -> {{Gen1: a1, ESS1: a2}}")
    print(f"  4. Devices receive and apply individual actions")

    # Count rewards collected
    reward_history = event_analyzer.get_reward_history()
    for agent_id in ["MG1", "MG1_Gen1", "MG1_ESS1"]:
        if agent_id in reward_history:
            print(f"  {agent_id}: {len(reward_history[agent_id])} rewards collected")

    print("\n" + "=" * 80)
    print("PowerGrid Action Passing Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

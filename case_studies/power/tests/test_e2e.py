"""End-to-end test for PowerGrid case study.

This script tests the complete hierarchical microgrid environment:
- SystemAgent -> PowerGridAgents -> DeviceAgents (Generator, ESS)
- Environment simulation with power flow
- CTDE training loop
- Event-driven execution

Uses the grid_age style environment structure.
"""

import numpy as np
from typing import Any, Dict, List

# HERON imports
from heron.agents.system_agent import SystemAgent
from heron.core.observation import Observation
from heron.core.action import Action
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.base import MultiAgentEnv
from heron.scheduling import TickConfig, JitterType
from heron.scheduling.analysis import EventAnalyzer

# PowerGrid imports
from powergrid.agents import (
    PowerGridAgent,
    Generator,
    ESS,
)
from powergrid.envs.common import EnvState


class TestPowerGridEnv(MultiAgentEnv):
    """Test environment for power grid simulation.

    Simplified version for testing without external dataset dependency.
    """

    def __init__(self, system_agent: SystemAgent, **kwargs):
        self.episode_steps = kwargs.pop("episode_steps", 24)
        self.dt = kwargs.pop("dt", 1.0)
        self._timestep = 0
        self._episode = 0

        super().__init__(system_agent=system_agent, **kwargs)

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        """Convert global state from proxy to env state for simulation."""
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
        """Run simplified power flow simulation."""
        # Simple physics: clip power values and compute basic metrics
        total_gen = 0.0
        for device_id, setpoint in env_state.device_setpoints.items():
            P = np.clip(setpoint.get("P", 0.0), -1.0, 1.0)
            total_gen += abs(P)
            env_state.device_setpoints[device_id]["P"] = P

        # Simulate power flow result
        env_state.update_power_flow_results({
            "converged": True,
            "voltage_avg": 1.0,
            "voltage_min": 0.98,
            "voltage_max": 1.02,
            "total_generation": total_gen,
        })

        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        """Convert simulation results back to global state."""
        from heron.agents.system_agent import SYSTEM_AGENT_ID
        from heron.agents.constants import FIELD_LEVEL

        global_state = self.proxy_agent.get_global_states(
            sender_id=SYSTEM_AGENT_ID, protocol=None
        )
        agent_states = global_state if isinstance(global_state, dict) else {}

        # Update device states with simulation results
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
        """Hook called before step execution."""
        self._timestep += 1


# Simple MLP for policy
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


class DeviceNeuralPolicy(Policy):
    """Neural policy for device agents."""
    observation_mode = "local"

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
        action_mean = self.actor.forward(obs_vec)
        action_vec = action_mean + np.random.normal(0, self.noise_scale, self.action_dim)
        return np.clip(action_vec, -1.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward_deterministic(self, obs_vec: np.ndarray) -> np.ndarray:
        return self.actor.forward(obs_vec)

    @obs_to_vector
    def get_value(self, obs_vec: np.ndarray) -> float:
        return float(self.critic.forward(obs_vec)[0])

    def update(self, obs, action_taken, advantage, lr=0.01):
        self.actor.update(obs, action_taken, advantage, lr)

    def update_critic(self, obs, target, lr=0.01):
        self.critic.update(obs, np.array([target]), lr)

    def decay_noise(self, decay_rate=0.995, min_noise=0.05):
        self.noise_scale = max(min_noise, self.noise_scale * decay_rate)


def train_ctde(env: MultiAgentEnv, num_episodes=50, steps_per_episode=24, gamma=0.99, lr=0.01):
    """Train policies using CTDE."""
    agent_ids = [aid for aid, agent in env.registered_agents.items()
                 if agent.action_space is not None]

    obs, _ = env.reset(seed=0)

    # Create policies with per-agent observation dimensions
    policies = {}
    for i, aid in enumerate(agent_ids):
        agent_obs = obs[aid]
        if isinstance(agent_obs, Observation):
            obs_dim = agent_obs.local_vector().shape[0]
        else:
            obs_dim = 2
        obs_dim = max(obs_dim, 2)  # Ensure minimum dimension
        print(f"Agent {aid}: obs_dim={obs_dim}")
        policies[aid] = DeviceNeuralPolicy(obs_dim=obs_dim, seed=42 + i)
    returns_history = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        trajectories = {aid: {"obs": [], "actions": [], "rewards": []} for aid in agent_ids}
        episode_return = 0.0

        for step in range(steps_per_episode):
            actions = {}
            for aid in agent_ids:
                obs_value = obs[aid]
                policy_obs_dim = policies[aid].obs_dim

                if isinstance(obs_value, Observation):
                    obs_vec = obs_value.local_vector()
                    observation = obs_value
                else:
                    # obs_value is numpy array - wrap it properly
                    obs_vec = np.asarray(obs_value, dtype=np.float32)
                    # Ensure obs_vec matches expected dimension
                    if len(obs_vec) > policy_obs_dim:
                        obs_vec = obs_vec[:policy_obs_dim]
                    elif len(obs_vec) < policy_obs_dim:
                        obs_vec = np.pad(obs_vec, (0, policy_obs_dim - len(obs_vec)))
                    observation = Observation(timestamp=step, local={"obs": obs_vec})

                action = policies[aid].forward(observation)
                actions[aid] = action
                trajectories[aid]["obs"].append(obs_vec[:policy_obs_dim].copy())
                trajectories[aid]["actions"].append(action.c.copy())

            obs, rewards, terminated, _, info = env.step(actions)

            for aid in agent_ids:
                if aid in rewards:
                    trajectories[aid]["rewards"].append(rewards[aid])
                    episode_return += rewards[aid]

            if terminated.get("__all__", False):
                break

        # Update policies
        for aid, traj in trajectories.items():
            if not traj["rewards"]:
                continue

            returns = []
            G = 0
            for r in reversed(traj["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)
            returns = np.array(returns)

            for t in range(len(traj["obs"])):
                obs_t = traj["obs"][t]
                baseline = policies[aid].get_value(Observation(timestamp=t, local={"obs": obs_t}))
                advantage = returns[t] - baseline
                policies[aid].update(obs_t, traj["actions"][t], advantage, lr=lr)
                policies[aid].update_critic(obs_t, returns[t], lr=lr)
            policies[aid].decay_noise()

        returns_history.append(episode_return)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d}: return={episode_return:.1f}")

    return policies, returns_history


def main():
    print("=" * 80)
    print("PowerGrid E2E Test")
    print("=" * 80)

    # Create device agents
    gen1 = Generator(
        agent_id="MG1_Gen1",
        bus="MG1_bus",
        p_min_MW=0.1,
        p_max_MW=1.0,
        cost_curve_coefs=(0.01, 0.5, 0.0),  # Quadratic cost: 0.01*P^2 + 0.5*P
    )

    ess1 = ESS(
        agent_id="MG1_ESS1",
        bus="MG1_bus",
        capacity_MWh=2.0,
        p_min_MW=-0.5,
        p_max_MW=0.5,
        degr_cost_per_MWh=0.1,  # Degradation cost per MWh throughput
    )

    # Create microgrid coordinator
    mg1 = PowerGridAgent(
        agent_id="MG1",
        subordinates={"MG1_Gen1": gen1, "MG1_ESS1": ess1},
    )

    # Create system agent
    system_agent = SystemAgent(
        agent_id="system_agent",
        subordinates={"MG1": mg1},
    )

    # Create environment
    env = TestPowerGridEnv(
        system_agent=system_agent,
        episode_steps=24,
    )

    print(f"\nEnvironment created with {len(env.registered_agents)} registered agents:")
    for aid in env.registered_agents:
        print(f"  - {aid}")

    # Run CTDE training
    print("\n" + "-" * 40)
    print("CTDE Training")
    print("-" * 40)

    policies, returns = train_ctde(env, num_episodes=30, steps_per_episode=24, lr=0.02)

    print(f"\nTraining Results:")
    print(f"  Initial avg return: {np.mean(returns[:5]):.1f}")
    print(f"  Final avg return:   {np.mean(returns[-5:]):.1f}")

    # Attach policies and run event-driven
    print("\n" + "-" * 40)
    print("Event-Driven Execution")
    print("-" * 40)

    env.set_agent_policies(policies)

    # Configure tick timing
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

    # Apply configs
    gen1.tick_config = field_tick_config
    ess1.tick_config = field_tick_config
    mg1.tick_config = coordinator_tick_config

    event_analyzer = EventAnalyzer(verbose=False, track_data=True)
    env.run_event_driven(event_analyzer=event_analyzer, t_end=100.0)

    print(f"\nEvent-Driven Statistics:")
    print(f"  Observations: {event_analyzer.observation_count}")
    print(f"  State updates: {event_analyzer.state_update_count}")
    print(f"  Action results: {event_analyzer.action_result_count}")

    print("\n" + "=" * 80)
    print("PowerGrid E2E Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

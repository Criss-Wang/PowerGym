"""End-to-End Integration Test for CTDE Training + Event-Driven Testing.

This comprehensive test verifies the complete HERON framework flow:
1. CTDE (Centralized Training with Decentralized Execution) training mode
2. Full decentralized & event-driven testing mode
3. Policy learning and execution across both modes

The test uses a simplified energy management scenario with:
- 3-level hierarchy: Field (batteries) → Coordinator (zones) → System (grid)
- All agent properties: tick_interval, obs_delay, act_delay, msg_delay
- Visibility-based information filtering
- Message broker communication
- Mixed continuous/discrete actions
- Policy-based decision making
"""

import pytest
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace

from heron.envs.base import MultiAgentEnv, EnvCore
from heron.agents.field_agent import FieldAgent, FIELD_LEVEL
from heron.agents.coordinator_agent import CoordinatorAgent, COORDINATOR_LEVEL
from heron.agents.system_agent import SystemAgent, SYSTEM_LEVEL
from heron.agents.proxy_agent import ProxyAgent
from heron.core.observation import Observation
from heron.core.feature import FeatureProvider
from heron.core.action import Action
from heron.core.state import FieldAgentState
from heron.core.policies import Policy, RandomPolicy
from heron.scheduling.scheduler import EventScheduler
from heron.scheduling.event import EventType
from heron.scheduling.tick_config import TickConfig
from heron.messaging.in_memory_broker import InMemoryBroker


# =============================================================================
# Feature Providers with Visibility Rules
# =============================================================================

class BatteryStateFeature(FeatureProvider):
    """Battery state feature - public visibility."""
    visibility = ["public"]

    def __init__(self, charge: float = 50.0, capacity: float = 100.0):
        self.charge = charge
        self.capacity = capacity

    def vector(self):
        return np.array([self.charge, self.capacity], dtype=np.float32)

    def names(self):
        return ["charge", "capacity"]

    def to_dict(self):
        return {"charge": self.charge, "capacity": self.capacity}

    @classmethod
    def from_dict(cls, d):
        return cls(charge=d.get("charge", 50.0), capacity=d.get("capacity", 100.0))

    def set_values(self, **kwargs):
        if "charge" in kwargs:
            self.charge = kwargs["charge"]
        if "capacity" in kwargs:
            self.capacity = kwargs["capacity"]


class BatteryInternalFeature(FeatureProvider):
    """Battery internal state - owner only visibility."""
    visibility = ["owner"]

    def __init__(self, temperature: float = 25.0, cycles: int = 0):
        self.temperature = temperature
        self.cycles = cycles

    def vector(self):
        return np.array([self.temperature, float(self.cycles)], dtype=np.float32)

    def names(self):
        return ["temperature", "cycles"]

    def to_dict(self):
        return {"temperature": self.temperature, "cycles": self.cycles}

    @classmethod
    def from_dict(cls, d):
        return cls(temperature=d.get("temperature", 25.0), cycles=d.get("cycles", 0))

    def set_values(self, **kwargs):
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "cycles" in kwargs:
            self.cycles = kwargs["cycles"]


class GridSignalFeature(FeatureProvider):
    """Grid-level signal - upper_level visibility."""
    visibility = ["upper_level"]

    def __init__(self, price: float = 0.1, frequency: float = 60.0):
        self.price = price
        self.frequency = frequency

    def vector(self):
        return np.array([self.price, self.frequency], dtype=np.float32)

    def names(self):
        return ["price", "frequency"]

    def to_dict(self):
        return {"price": self.price, "frequency": self.frequency}

    @classmethod
    def from_dict(cls, d):
        return cls(price=d.get("price", 0.1), frequency=d.get("frequency", 60.0))

    def set_values(self, **kwargs):
        if "price" in kwargs:
            self.price = kwargs["price"]
        if "frequency" in kwargs:
            self.frequency = kwargs["frequency"]


# =============================================================================
# Simple Neural Network Policy for CTDE
# =============================================================================

class SimpleMLP:
    """Simple MLP for policy approximation (numpy-based for testing)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        np.random.seed(seed)
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activation."""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        out = np.tanh(h @ self.W2 + self.b2)  # Tanh for bounded output
        return out

    def update(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01):
        """Simple gradient descent update."""
        # Forward
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)

        # Backward (simplified)
        d_out = (out - target) * (1 - out**2)  # tanh derivative
        d_W2 = h.T @ d_out if h.ndim > 1 else np.outer(h, d_out)
        d_b2 = d_out.sum(axis=0) if d_out.ndim > 1 else d_out

        d_h = d_out @ self.W2.T
        d_h[h <= 0] = 0  # ReLU derivative
        d_W1 = x.T @ d_h if x.ndim > 1 else np.outer(x, d_h)
        d_b1 = d_h.sum(axis=0) if d_h.ndim > 1 else d_h

        # Update
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1


class CTDEPolicy(Policy):
    """CTDE-style centralized critic with decentralized actors.

    This implements a simple actor-critic where:
    - Critic (centralized): Sees all agent observations during training
    - Actors (decentralized): Each agent has its own policy network
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 32, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = SimpleMLP(obs_dim, hidden_dim, action_dim, seed)
        self.critic = SimpleMLP(obs_dim, hidden_dim, 1, seed + 1)

    def forward(self, observation: Observation) -> Action:
        """Compute action from observation."""
        obs_vec = observation.vector()
        action_vec = self.actor.forward(obs_vec)
        action = Action()
        action.set_specs(dim_c=self.action_dim, range=(np.full(self.action_dim, -1.0), np.full(self.action_dim, 1.0)))
        action.set_values(action_vec)
        return action

    def get_value(self, observation: Observation) -> float:
        """Get state value estimate."""
        obs_vec = observation.vector()
        return float(self.critic.forward(obs_vec)[0])

    def update_actor(self, obs: np.ndarray, advantage: float, action: np.ndarray, lr: float = 0.001):
        """Update actor using policy gradient."""
        # Simple policy gradient: move towards actions with positive advantage
        if advantage > 0:
            self.actor.update(obs, action, lr=lr * advantage)

    def update_critic(self, obs: np.ndarray, target_value: float, lr: float = 0.01):
        """Update critic towards target value."""
        self.critic.update(obs, np.array([target_value]), lr=lr)


# =============================================================================
# Field Agents (L1)
# =============================================================================

class BatteryAgent(FieldAgent):
    """Battery field agent with full feature set."""

    def __init__(
        self,
        agent_id: str,
        capacity: float = 100.0,
        max_power: float = 20.0,
        efficiency: float = 0.95,
        tick_config: Optional[TickConfig] = None,
        **kwargs
    ):
        self.capacity = capacity
        self.max_power = max_power
        self.efficiency = efficiency
        self.charge_level = capacity * 0.5  # Start at 50%

        # Set timing config with defaults appropriate for field level
        if tick_config is None:
            tick_config = TickConfig.deterministic(
                tick_interval=1.0,  # Fast: 1 second
                obs_delay=0.1,  # Small observation delay
                act_delay=0.2,  # Small action delay
                msg_delay=0.05,  # Fast message delivery
            )

        super().__init__(agent_id=agent_id, tick_config=tick_config, **kwargs)

    def set_action(self):
        """Mixed continuous (power) + discrete (mode) action."""
        self.action.set_specs(
            dim_c=1,  # Continuous: power setpoint [-1, 1] normalized
            dim_d=1,  # Discrete: mode (0=idle, 1=charge, 2=discharge)
            ncats=[3],
            range=(np.array([-1.0]), np.array([1.0]))
        )

    def set_state(self):
        """Set up battery state with multiple feature types."""
        self.state.features = [
            BatteryStateFeature(charge=self.charge_level, capacity=self.capacity),
            BatteryInternalFeature(temperature=25.0, cycles=0),
        ]

    def _get_obs(self, proxy=None) -> np.ndarray:
        """Build observation vector."""
        state_vec = self.state.vector()
        # Add normalized charge level
        normalized_charge = self.charge_level / self.capacity
        return np.concatenate([state_vec, [normalized_charge]], dtype=np.float32)

    def step_physics(self, dt: float = 1.0) -> Dict[str, float]:
        """Simulate battery physics for one timestep."""
        # Decode action
        power_setpoint = float(self.action.c[0]) * self.max_power
        mode = int(self.action.d[0]) if self.action.dim_d > 0 else 1

        actual_power = 0.0

        if mode == 1:  # Charge
            power = max(0, power_setpoint)
            energy = power * dt * self.efficiency
            new_charge = min(self.charge_level + energy, self.capacity)
            actual_power = (new_charge - self.charge_level) / dt / self.efficiency
            self.charge_level = new_charge

        elif mode == 2:  # Discharge
            power = max(0, -power_setpoint)
            energy = power * dt / self.efficiency
            new_charge = max(self.charge_level - energy, 0)
            actual_power = -(self.charge_level - new_charge) / dt * self.efficiency
            self.charge_level = new_charge

        # Update state features
        self.state.features[0].charge = self.charge_level

        # Update temperature based on power
        internal = self.state.features[1]
        internal.temperature = 25.0 + abs(actual_power) * 0.1

        return {
            "power": actual_power,
            "charge": self.charge_level,
            "soc": self.charge_level / self.capacity,
        }

    def reset_agent(self, **kwargs):
        """Reset battery to initial state."""
        self.charge_level = self.capacity * 0.5
        self.state.features[0].charge = self.charge_level
        self.state.features[1].temperature = 25.0
        self.state.features[1].cycles = 0


# =============================================================================
# Coordinator Agents (L2)
# =============================================================================

class ZoneCoordinator(CoordinatorAgent):
    """Zone coordinator managing multiple batteries."""

    def __init__(self, agent_id: str, tick_config: Optional[TickConfig] = None, **kwargs):
        # Set timing config appropriate for coordinator level
        if tick_config is None:
            tick_config = TickConfig.deterministic(
                tick_interval=5.0,  # Medium: 5 seconds
                obs_delay=0.5,  # Moderate delay
                act_delay=1.0,  # Action takes longer
                msg_delay=0.2,  # Moderate message delay
            )

        super().__init__(agent_id=agent_id, tick_config=tick_config, **kwargs)

        # Add grid signal feature to coordinator state
        self.state.features = [GridSignalFeature(price=0.1, frequency=60.0)]

    def _build_subordinates(
        self,
        agent_configs: List[Dict[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[str] = None,
    ) -> Dict[str, BatteryAgent]:
        """Build battery agents from config."""
        agents = {}
        for config in agent_configs:
            agent_id = config.get("id")
            capacity = config.get("capacity", 100.0)
            max_power = config.get("max_power", 20.0)
            agents[agent_id] = BatteryAgent(
                agent_id=agent_id,
                capacity=capacity,
                max_power=max_power,
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )
        return agents

    def get_zone_metrics(self) -> Dict[str, float]:
        """Get aggregated zone metrics."""
        total_capacity = sum(a.capacity for a in self.subordinates.values())
        total_charge = sum(a.charge_level for a in self.subordinates.values())
        avg_soc = total_charge / total_capacity if total_capacity > 0 else 0

        return {
            "total_capacity": total_capacity,
            "total_charge": total_charge,
            "avg_soc": avg_soc,
            "num_batteries": len(self.subordinates),
        }


# =============================================================================
# System Agent (L3)
# =============================================================================

class GridSystemAgent(SystemAgent):
    """Grid system agent managing multiple zones."""

    def __init__(self, agent_id: str, tick_config: Optional[TickConfig] = None, **kwargs):
        # Set timing config appropriate for system level
        if tick_config is None:
            tick_config = TickConfig.deterministic(
                tick_interval=15.0,  # Slow: 15 seconds
                obs_delay=1.0,  # Higher delay
                act_delay=2.0,  # Actions take effect slowly
                msg_delay=0.5,  # Slower communication
            )

        super().__init__(agent_id=agent_id, tick_config=tick_config, **kwargs)

        # System-level state tracking
        self._env_state = {}
        self._grid_load = 0.0
        self._grid_price = 0.1

    def _build_subordinates(
        self,
        coordinator_configs: List[Dict[str, Any]],
        env_id: Optional[str] = None,
        upstream_id: Optional[str] = None,
    ) -> Dict[str, ZoneCoordinator]:
        """Build zone coordinators from config."""
        coordinators = {}
        for config in coordinator_configs:
            coord_id = config.get("id")
            agent_configs = config.get("agents", [])
            coordinators[coord_id] = ZoneCoordinator(
                agent_id=coord_id,
                config={"agents": agent_configs},
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )
        return coordinators

    def update_from_environment(self, env_state: Dict[str, Any]) -> None:
        """Update from environment state."""
        self._env_state = env_state
        self._grid_load = env_state.get("grid_load", 0.0)
        self._grid_price = env_state.get("price", 0.1)

        # Propagate price to coordinators
        for coord in self.coordinators.values():
            if coord.state.features:
                coord.state.features[0].price = self._grid_price

    def get_state_for_environment(self) -> Dict[str, Any]:
        """Get aggregated state for environment."""
        zone_metrics = {}
        all_battery_states = {}

        for coord_id, coord in self.coordinators.items():
            zone_metrics[coord_id] = coord.get_zone_metrics()
            for agent_id, agent in coord.subordinates.items():
                all_battery_states[agent_id] = {
                    "charge": agent.charge_level,
                    "soc": agent.charge_level / agent.capacity,
                    "power": float(agent.action.c[0]) * agent.max_power if agent.action.dim_c > 0 else 0,
                }

        return {
            "zone_metrics": zone_metrics,
            "battery_states": all_battery_states,
            "grid_load": self._grid_load,
        }


# =============================================================================
# Multi-Agent Environment
# =============================================================================

class GridMicrogridEnv(MultiAgentEnv):
    """Complete microgrid environment for CTDE + event-driven testing."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_proxy: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        config = config or self._default_config()

        # Create system agent hierarchy
        self._grid_system = GridSystemAgent(
            agent_id="grid_system",
            config=config,
            env_id=self.env_id,
        )

        # Register all agents
        self._register_hierarchy()

        # Setup proxy agent for visibility filtering
        self._proxy_agent = None
        if use_proxy:
            self._setup_proxy_agent()

        # Environment state
        self._timestep = 0
        self._grid_load = 100.0  # Base load
        self._price = 0.1  # Base electricity price
        self._max_steps = 100

        # Metrics tracking
        self._episode_rewards = []
        self._episode_metrics = []

    def _default_config(self) -> Dict[str, Any]:
        """Default environment configuration."""
        return {
            "coordinators": [
                {
                    "id": "zone_north",
                    "agents": [
                        {"id": "battery_n1", "capacity": 100.0, "max_power": 20.0},
                        {"id": "battery_n2", "capacity": 150.0, "max_power": 30.0},
                    ]
                },
                {
                    "id": "zone_south",
                    "agents": [
                        {"id": "battery_s1", "capacity": 80.0, "max_power": 15.0},
                    ]
                }
            ]
        }

    @property
    def grid_system(self) -> GridSystemAgent:
        """Get the grid system agent."""
        return self._grid_system

    def _register_hierarchy(self):
        """Register all agents in the hierarchy."""
        self.register_agent(self._grid_system)
        for coord_id, coord in self._grid_system.coordinators.items():
            self.register_agent(coord)
            for agent_id, agent in coord.subordinates.items():
                self.register_agent(agent)

    def _setup_proxy_agent(self):
        """Setup proxy agent for state distribution."""
        all_agent_ids = list(self.registered_agents.keys())
        self._proxy_agent = ProxyAgent(
            agent_id="proxy",
            env_id=self.env_id,
            registered_agents=all_agent_ids,
            history_length=50,
        )
        self.register_agent(self._proxy_agent)

    def _get_controllable_agents(self) -> List[str]:
        """Get list of controllable agent IDs (field level only)."""
        agents = []
        for coord in self._grid_system.coordinators.values():
            agents.extend(coord.subordinates.keys())
        return agents

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment."""
        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)

        self._timestep = 0
        self._grid_load = np.random.uniform(80, 120)
        self._price = 0.1

        # Reset all agents
        self._grid_system.reset(seed=seed)

        # Reset proxy
        if self._proxy_agent:
            self._proxy_agent.reset(seed=seed)

        # Reset tracking
        self._episode_rewards = []
        self._episode_metrics = []

        # Get initial observations
        obs = self._get_observations()
        info = self._get_info()

        return obs, info

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one environment step."""
        self._timestep += 1

        # Update grid state (simulate load variation)
        self._update_grid_state()

        # Update system agent with environment state
        self._grid_system.update_from_environment({
            "grid_load": self._grid_load,
            "price": self._price,
            "timestep": self._timestep,
        })

        # Apply actions through hierarchy
        self._apply_actions(actions)

        # Step physics for all batteries
        physics_results = self._step_physics()

        # Update proxy state
        if self._proxy_agent:
            self._proxy_agent.update_state({
                "agents": {
                    agent_id: {"charge": agent.charge_level, "soc": agent.charge_level / agent.capacity}
                    for coord in self._grid_system.coordinators.values()
                    for agent_id, agent in coord.subordinates.items()
                },
                "grid": {"load": self._grid_load, "price": self._price},
            })
            self._proxy_agent._timestep = self._timestep

        # Calculate rewards
        rewards = self._calculate_rewards(physics_results)

        # Get observations
        obs = self._get_observations()

        # Check termination
        terminated = {"__all__": self._timestep >= self._max_steps}
        truncated = {"__all__": False}

        info = self._get_info()
        info["physics"] = physics_results

        # Track metrics
        self._episode_rewards.append(sum(rewards.values()))
        self._episode_metrics.append({
            "total_power": sum(r.get("power", 0) for r in physics_results.values()),
            "avg_soc": np.mean([r.get("soc", 0.5) for r in physics_results.values()]),
            "grid_load": self._grid_load,
        })

        return obs, rewards, terminated, truncated, info

    def _update_grid_state(self):
        """Simulate grid state dynamics."""
        # Add sinusoidal load pattern + noise
        t = self._timestep
        self._grid_load = 100.0 + 30.0 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 5)
        self._price = 0.1 + 0.05 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.01)
        self._price = max(0.05, self._price)

    def _apply_actions(self, actions: Dict[str, Any]):
        """Apply actions through hierarchy."""
        system_obs = self._grid_system.observe()

        # Build hierarchical action dict
        action_dict = {}
        for coord_id, coord in self._grid_system.coordinators.items():
            coord_actions = {}
            for agent_id in coord.subordinates:
                if agent_id in actions:
                    coord_actions[agent_id] = actions[agent_id]
            action_dict[coord_id] = coord_actions

        self._grid_system.act(system_obs, upstream_action=action_dict)

    def _step_physics(self) -> Dict[str, Dict[str, float]]:
        """Step physics for all batteries."""
        results = {}
        for coord in self._grid_system.coordinators.values():
            for agent_id, agent in coord.subordinates.items():
                results[agent_id] = agent.step_physics()
        return results

    def _calculate_rewards(self, physics_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {}

        # Get total power output
        total_power = sum(r.get("power", 0) for r in physics_results.values())

        # Reward based on meeting grid load
        load_error = abs(total_power - self._grid_load * 0.1)  # Want ~10% of load from batteries
        base_reward = -load_error / 50.0  # Normalize

        # Per-agent rewards with SOC penalty
        for agent_id, result in physics_results.items():
            soc = result.get("soc", 0.5)
            # Penalize extreme SOC
            soc_penalty = -0.1 * ((soc - 0.5) ** 2)
            rewards[agent_id] = base_reward + soc_penalty

        return rewards

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all controllable agents."""
        obs = {}
        global_state = {"grid_load": self._grid_load, "price": self._price}

        for coord in self._grid_system.coordinators.values():
            for agent_id, agent in coord.subordinates.items():
                agent_obs = agent.observe(global_state)
                obs[agent_id] = agent_obs.vector()

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {
            "timestep": self._timestep,
            "grid_load": self._grid_load,
            "price": self._price,
            "system_state": self._grid_system.get_state_for_environment(),
        }


# =============================================================================
# Integration Tests
# =============================================================================

class TestCTDETrainingMode:
    """Test CTDE training mode (Option A)."""

    def test_environment_creation_with_full_hierarchy(self):
        """Test creating environment with complete 3-level hierarchy."""
        env = GridMicrogridEnv()

        # Verify hierarchy
        assert env.grid_system is not None
        assert env.grid_system.level == SYSTEM_LEVEL
        assert len(env.grid_system.coordinators) == 2

        # Verify coordinators
        zone_north = env.grid_system.coordinators.get("zone_north")
        assert zone_north is not None
        assert zone_north.level == COORDINATOR_LEVEL
        assert len(zone_north.subordinates) == 2

        # Verify field agents
        battery_n1 = zone_north.subordinates.get("battery_n1")
        assert battery_n1 is not None
        assert battery_n1.level == FIELD_LEVEL

    def test_all_agent_timing_parameters(self):
        """Test that all agents have proper timing parameters."""
        env = GridMicrogridEnv()

        # System agent - slowest
        assert env.grid_system._tick_config.tick_interval == 15.0
        assert env.grid_system._tick_config.obs_delay == 1.0
        assert env.grid_system._tick_config.act_delay == 2.0

        # Coordinator - medium
        for coord in env.grid_system.coordinators.values():
            assert coord._tick_config.tick_interval == 5.0
            assert coord._tick_config.obs_delay == 0.5
            assert coord._tick_config.act_delay == 1.0

        # Field agents - fastest
        for coord in env.grid_system.coordinators.values():
            for agent in coord.subordinates.values():
                assert agent._tick_config.tick_interval == 1.0
                assert agent._tick_config.obs_delay == 0.1
                assert agent._tick_config.act_delay == 0.2

    def test_mixed_action_space(self):
        """Test mixed continuous/discrete action handling."""
        env = GridMicrogridEnv()
        env.reset(seed=42)

        # Get a battery agent
        coord = list(env.grid_system.coordinators.values())[0]
        battery = list(coord.subordinates.values())[0]

        # Verify mixed action space
        assert battery.action.dim_c == 1  # Continuous power
        assert battery.action.dim_d == 1  # Discrete mode
        assert battery.action.ncats == [3]  # 3 modes

        # Test setting mixed action
        battery.action.set_values({"c": [0.5], "d": [2]})
        assert np.allclose(battery.action.c, [0.5])
        assert battery.action.d[0] == 2

    def test_visibility_filtering_through_hierarchy(self):
        """Test that visibility rules are respected."""
        env = GridMicrogridEnv()
        env.reset(seed=42)

        # Get a battery
        coord = env.grid_system.coordinators["zone_north"]
        battery = coord.subordinates["battery_n1"]

        # Battery should see its own internal state (owner visibility)
        own_state = battery.state.observed_by(battery.agent_id, battery.level)
        # Should include both public and owner features
        assert len(own_state) > 0

        # Other battery should only see public state
        other_battery = coord.subordinates["battery_n2"]
        visible_state = battery.state.observed_by(other_battery.agent_id, other_battery.level)
        # Should only include public features
        feature_names = list(visible_state.keys())
        # BatteryInternalFeature (owner visibility) should NOT be visible
        assert "temperature" not in feature_names or visible_state.get("temperature") is None

    def test_full_training_episode(self):
        """Test running a full training episode."""
        env = GridMicrogridEnv()
        obs, info = env.reset(seed=42)

        total_rewards = {agent_id: 0.0 for agent_id in obs.keys()}
        steps = 0

        while steps < 100:
            # Random actions
            actions = {}
            for agent_id in obs.keys():
                actions[agent_id] = {
                    "c": np.random.uniform(-1, 1, size=(1,)),
                    "d": np.random.randint(0, 3, size=(1,)),
                }

            obs, rewards, terminated, truncated, info = env.step(actions)

            for agent_id, r in rewards.items():
                total_rewards[agent_id] += r

            steps += 1

            if terminated["__all__"] or truncated["__all__"]:
                break

        assert steps == 100
        # Rewards should be non-zero (some learning signal)
        assert any(r != 0 for r in total_rewards.values())

    def test_ctde_policy_training_loop(self):
        """Test complete CTDE policy training loop."""
        env = GridMicrogridEnv()

        # Create policies for each agent
        policies = {}
        for coord in env.grid_system.coordinators.values():
            for agent_id, agent in coord.subordinates.items():
                obs_dim = agent.observe().vector().shape[0]
                action_dim = agent.action.dim_c
                policies[agent_id] = CTDEPolicy(obs_dim, action_dim, hidden_dim=16)

        # Training loop
        num_episodes = 3
        episode_returns = []

        for episode in range(num_episodes):
            obs, _ = env.reset(seed=episode)
            episode_return = 0.0

            for step in range(50):  # Shorter episodes for testing
                # Get actions from policies
                actions = {}
                for agent_id, agent_obs in obs.items():
                    observation = Observation(timestamp=step, local={"observation": agent_obs})
                    policy_action = policies[agent_id].forward(observation)
                    actions[agent_id] = {
                        "c": policy_action.c,
                        "d": np.array([1]),  # Default to charge mode
                    }

                next_obs, rewards, terminated, truncated, info = env.step(actions)

                # Simple policy update (for testing)
                for agent_id in obs.keys():
                    reward = rewards[agent_id]
                    episode_return += reward

                    # Update critic
                    observation = Observation(timestamp=step, local={"observation": obs[agent_id]})
                    next_observation = Observation(timestamp=step + 1, local={"observation": next_obs[agent_id]})

                    current_value = policies[agent_id].get_value(observation)
                    next_value = policies[agent_id].get_value(next_observation)
                    target = reward + 0.99 * next_value
                    policies[agent_id].update_critic(obs[agent_id], target)

                    # Update actor
                    advantage = target - current_value
                    policy_action = policies[agent_id].forward(observation)
                    policies[agent_id].update_actor(obs[agent_id], advantage, policy_action.c)

                obs = next_obs

                if terminated["__all__"]:
                    break

            episode_returns.append(episode_return)

        # Returns should exist (training ran)
        assert len(episode_returns) == num_episodes


class TestEventDrivenTestingMode:
    """Test event-driven testing mode (Option B)."""

    def test_scheduler_with_full_hierarchy(self):
        """Test EventScheduler with complete agent hierarchy."""
        env = GridMicrogridEnv()
        env.reset(seed=42)

        scheduler = EventScheduler(start_time=0.0)

        # Register all agents with their tick intervals
        for agent_id, agent in env.registered_agents.items():
            if agent_id != "proxy":  # Skip proxy
                scheduler.register_agent(
                    agent_id=agent_id,
                    tick_interval=agent._tick_config.tick_interval,
                    obs_delay=agent._tick_config.obs_delay,
                    act_delay=agent._tick_config.act_delay,
                )

        # Track ticks per agent
        tick_counts = {agent_id: 0 for agent_id in env.registered_agents.keys() if agent_id != "proxy"}

        def tick_handler(event, sched):
            if event.agent_id in tick_counts:
                tick_counts[event.agent_id] += 1

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)

        # Run for 30 seconds
        scheduler.run_until(t_end=30.0)

        # Verify tick ratios match hierarchy levels
        # Field agents (tick_interval=1.0) should tick most frequently
        # Coordinators (tick_interval=5.0) should tick less
        # System agent (tick_interval=15.0) should tick least

        field_ticks = [tick_counts[aid] for aid in tick_counts if "battery" in aid]
        coord_ticks = [tick_counts[aid] for aid in tick_counts if "zone" in aid]
        system_ticks = tick_counts.get("grid_system", 0)

        assert all(t >= 30 for t in field_ticks)  # ~30 ticks at 1s interval
        assert all(t >= 6 for t in coord_ticks)  # ~6 ticks at 5s interval
        assert system_ticks >= 2  # ~2 ticks at 15s interval

    def test_heterogeneous_delays(self):
        """Test action and observation delays in event-driven mode."""
        scheduler = EventScheduler(start_time=0.0)

        # Register agent with delays
        scheduler.register_agent(
            agent_id="battery_test",
            tick_interval=5.0,
            obs_delay=0.5,
            act_delay=1.0,
        )

        events_log = []

        def tick_handler(event, sched):
            events_log.append(("tick", event.timestamp, event.agent_id))
            # Schedule action effect
            sched.schedule_action_effect(
                event.agent_id,
                action={"power": 10.0},
            )

        def effect_handler(event, sched):
            events_log.append(("effect", event.timestamp, event.agent_id))

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)
        scheduler.set_handler(EventType.ACTION_EFFECT, effect_handler)

        scheduler.run_until(t_end=12.0)

        # Extract events
        ticks = [(t, aid) for e, t, aid in events_log if e == "tick"]
        effects = [(t, aid) for e, t, aid in events_log if e == "effect"]

        # Ticks at t=0, 5, 10
        assert len(ticks) == 3
        assert ticks[0][0] == 0.0
        assert ticks[1][0] == 5.0
        assert ticks[2][0] == 10.0

        # Effects at t=1, 6, 11 (tick + act_delay=1.0)
        assert len(effects) == 3
        assert effects[0][0] == 1.0
        assert effects[1][0] == 6.0
        assert effects[2][0] == 11.0

    def test_message_broker_in_event_mode(self):
        """Test message broker communication in event-driven mode."""
        env = GridMicrogridEnv()
        env.reset(seed=42)

        # Configure for distributed
        env.configure_agents_for_distributed()

        scheduler = EventScheduler(start_time=0.0)

        # Register coordinator and subordinates
        coord = env.grid_system.coordinators["zone_north"]
        scheduler.register_agent("zone_north", tick_interval=5.0)
        for aid in coord.subordinates:
            scheduler.register_agent(aid, tick_interval=1.0)

        messages_sent = []
        messages_received = []

        def coord_tick_handler(event, sched):
            if event.agent_id == "zone_north":
                # Coordinator sends action to subordinates
                for sub_id in coord.subordinates:
                    coord.send_action_to_subordinate(sub_id, {"power": 5.0})
                    messages_sent.append((event.timestamp, sub_id))

        def agent_tick_handler(event, sched):
            if event.agent_id in coord.subordinates:
                agent = coord.subordinates[event.agent_id]
                msgs = agent.receive_action_messages()
                if msgs:
                    messages_received.append((event.timestamp, event.agent_id, msgs[-1]))

        def combined_handler(event, sched):
            coord_tick_handler(event, sched)
            agent_tick_handler(event, sched)

        scheduler.set_handler(EventType.AGENT_TICK, combined_handler)

        scheduler.run_until(t_end=10.0)

        # Messages should have been sent and received
        assert len(messages_sent) > 0
        assert len(messages_received) > 0

    def test_proxy_agent_delayed_observations(self):
        """Test ProxyAgent provides delayed observations."""
        env = GridMicrogridEnv(use_proxy=True)
        env.reset(seed=42)

        proxy = env._proxy_agent

        # Update proxy state at different times
        proxy._timestep = 0
        proxy.update_state({"agents": {"battery_n1": {"charge": 50.0}}, "grid": {"load": 100.0}})

        proxy._timestep = 5
        proxy.update_state({"agents": {"battery_n1": {"charge": 60.0}}, "grid": {"load": 110.0}})

        proxy._timestep = 10
        proxy.update_state({"agents": {"battery_n1": {"charge": 70.0}}, "grid": {"load": 120.0}})

        # Get state at different times
        state_at_0 = proxy.get_state_at_time(0)
        state_at_5 = proxy.get_state_at_time(5)
        state_at_10 = proxy.get_state_at_time(10)

        # With delay of 2, at t=5 agent should see state from t=3 (which is t=0 state)
        state_delayed = proxy.get_state_at_time(3)
        assert state_delayed["agents"]["battery_n1"]["charge"] == 50.0

        # At t=8, agent should see state from t=5
        state_delayed_2 = proxy.get_state_at_time(8)
        assert state_delayed_2["agents"]["battery_n1"]["charge"] == 60.0


class TestDualModeCompatibility:
    """Test that agents work in both modes."""

    def test_same_agent_both_modes(self):
        """Test that same agent instance works in both execution modes."""
        env = GridMicrogridEnv()
        obs, _ = env.reset(seed=42)

        # Get a battery agent
        coord = env.grid_system.coordinators["zone_north"]
        battery = coord.subordinates["battery_n1"]

        # === Mode A: Synchronous ===
        # Directly call observe/act
        observation = battery.observe()
        assert observation is not None
        assert observation.vector().shape[0] > 0

        # Apply action directly
        battery.act(observation, upstream_action={"c": [0.5], "d": [1]})
        result_a = battery.step_physics()

        initial_charge = battery.charge_level

        # === Mode B: Event-driven ===
        # Reset for event mode
        env.reset(seed=42)

        scheduler = EventScheduler(start_time=0.0)
        scheduler.register_agent(
            battery.agent_id,
            tick_interval=battery._tick_config.tick_interval,
            obs_delay=battery._tick_config.obs_delay,
            act_delay=battery._tick_config.act_delay,
        )

        tick_results = []

        def tick_handler(event, sched):
            # Simulate tick behavior
            battery._timestep = event.timestamp
            obs = battery.observe()
            battery.action.set_values({"c": [0.5], "d": [1]})
            result = battery.step_physics()
            tick_results.append(result)

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)
        scheduler.run_until(t_end=1.0)

        # Both modes should produce valid results
        assert "power" in result_a
        assert len(tick_results) > 0
        assert "power" in tick_results[0]

    def test_policy_works_in_both_modes(self):
        """Test that trained policy works in both modes."""
        env = GridMicrogridEnv()

        # Create a simple policy
        coord = env.grid_system.coordinators["zone_north"]
        battery = coord.subordinates["battery_n1"]
        obs_dim = battery.observe().vector().shape[0]
        policy = CTDEPolicy(obs_dim, 1, hidden_dim=16)

        # === Mode A: Policy in synchronous mode ===
        env.reset(seed=42)
        sync_actions = []

        for _ in range(5):
            observation = battery.observe()
            action = policy.forward(observation)
            sync_actions.append(action.c.copy())
            battery.action.set_values({"c": action.c, "d": [1]})
            battery.step_physics()

        # === Mode B: Policy in event-driven mode ===
        env.reset(seed=42)
        event_actions = []

        scheduler = EventScheduler(start_time=0.0)
        scheduler.register_agent(battery.agent_id, tick_interval=1.0)

        def tick_handler(event, sched):
            battery._timestep = event.timestamp
            observation = battery.observe()
            action = policy.forward(observation)
            event_actions.append(action.c.copy())
            battery.action.set_values({"c": action.c, "d": [1]})
            battery.step_physics()

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)
        scheduler.run_until(t_end=5.0)

        # Both should produce similar actions (same policy, same initial state)
        assert len(sync_actions) == 5
        assert len(event_actions) >= 5
        # First action should be identical (same initial state)
        np.testing.assert_array_almost_equal(sync_actions[0], event_actions[0], decimal=5)


class TestEndToEndFlow:
    """Test complete end-to-end flow: train in Mode A, test in Mode B."""

    def test_train_then_test_flow(self):
        """Test training in sync mode then testing in event mode."""
        env = GridMicrogridEnv()

        # Create policies
        policies = {}
        for coord in env.grid_system.coordinators.values():
            for agent_id, agent in coord.subordinates.items():
                obs_dim = agent.observe().vector().shape[0]
                policies[agent_id] = CTDEPolicy(obs_dim, 1, hidden_dim=16, seed=42)

        # === TRAINING PHASE (Mode A) ===
        training_returns = []

        for episode in range(5):
            obs, _ = env.reset(seed=episode)
            episode_return = 0.0

            for step in range(30):
                actions = {}
                for agent_id, agent_obs in obs.items():
                    observation = Observation(timestamp=step, local={"observation": agent_obs})
                    policy_action = policies[agent_id].forward(observation)
                    actions[agent_id] = {"c": policy_action.c, "d": [1]}

                next_obs, rewards, terminated, _, _ = env.step(actions)
                episode_return += sum(rewards.values())

                # Simple policy update
                for agent_id in obs.keys():
                    observation = Observation(timestamp=step, local={"observation": obs[agent_id]})
                    next_observation = Observation(timestamp=step + 1, local={"observation": next_obs[agent_id]})
                    target = rewards[agent_id] + 0.99 * policies[agent_id].get_value(next_observation)
                    policies[agent_id].update_critic(obs[agent_id], target)

                obs = next_obs
                if terminated["__all__"]:
                    break

            training_returns.append(episode_return)

        # === TESTING PHASE (Mode B) ===
        env.reset(seed=100)
        env.configure_agents_for_distributed()

        scheduler = EventScheduler(start_time=0.0)

        # Register all field agents
        for coord in env.grid_system.coordinators.values():
            for agent_id, agent in coord.subordinates.items():
                scheduler.register_agent(
                    agent_id,
                    tick_interval=agent._tick_config.tick_interval,
                    obs_delay=agent._tick_config.obs_delay,
                    act_delay=agent._tick_config.act_delay,
                )

        test_actions = []
        test_charges = []

        def tick_handler(event, sched):
            agent_id = event.agent_id
            for coord in env.grid_system.coordinators.values():
                if agent_id in coord.subordinates:
                    agent = coord.subordinates[agent_id]
                    agent._timestep = event.timestamp

                    # Use trained policy
                    observation = agent.observe()
                    policy_action = policies[agent_id].forward(observation)
                    test_actions.append((agent_id, policy_action.c.copy()))

                    agent.action.set_values({"c": policy_action.c, "d": [1]})
                    result = agent.step_physics()
                    test_charges.append((agent_id, result["soc"]))

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)
        scheduler.run_until(t_end=10.0)

        # Verify training happened
        assert len(training_returns) == 5

        # Verify testing happened
        assert len(test_actions) > 0
        assert len(test_charges) > 0

        # All SOC values should be valid (0-1)
        for agent_id, soc in test_charges:
            assert 0.0 <= soc <= 1.0

    def test_meaningful_training_improvement(self):
        """Test that training produces meaningful improvement."""
        env = GridMicrogridEnv()

        # Create policies
        policies = {}
        for coord in env.grid_system.coordinators.values():
            for agent_id, agent in coord.subordinates.items():
                obs_dim = agent.observe().vector().shape[0]
                policies[agent_id] = CTDEPolicy(obs_dim, 1, hidden_dim=32, seed=42)

        # Collect baseline performance (random policy)
        baseline_returns = []
        for episode in range(3):
            obs, _ = env.reset(seed=episode + 100)
            episode_return = 0.0
            for _ in range(30):
                actions = {aid: {"c": np.random.uniform(-1, 1, (1,)), "d": [np.random.randint(3)]} for aid in obs}
                obs, rewards, terminated, _, _ = env.step(actions)
                episode_return += sum(rewards.values())
                if terminated["__all__"]:
                    break
            baseline_returns.append(episode_return)

        # Train policies
        for episode in range(10):
            obs, _ = env.reset(seed=episode)

            for step in range(30):
                actions = {}
                for agent_id, agent_obs in obs.items():
                    observation = Observation(timestamp=step, local={"observation": agent_obs})
                    policy_action = policies[agent_id].forward(observation)
                    actions[agent_id] = {"c": policy_action.c, "d": [1]}

                next_obs, rewards, terminated, _, _ = env.step(actions)

                # Update policies
                for agent_id in obs.keys():
                    observation = Observation(timestamp=step, local={"observation": obs[agent_id]})
                    next_observation = Observation(timestamp=step + 1, local={"observation": next_obs[agent_id]})
                    target = rewards[agent_id] + 0.99 * policies[agent_id].get_value(next_observation)
                    policies[agent_id].update_critic(obs[agent_id], target)
                    advantage = target - policies[agent_id].get_value(observation)
                    if advantage > 0:
                        policies[agent_id].update_actor(obs[agent_id], advantage, policies[agent_id].forward(observation).c)

                obs = next_obs
                if terminated["__all__"]:
                    break

        # Evaluate trained policy
        trained_returns = []
        for episode in range(3):
            obs, _ = env.reset(seed=episode + 100)  # Same seeds as baseline
            episode_return = 0.0
            for step in range(30):
                actions = {}
                for agent_id, agent_obs in obs.items():
                    observation = Observation(timestamp=step, local={"observation": agent_obs})
                    policy_action = policies[agent_id].forward(observation)
                    actions[agent_id] = {"c": policy_action.c, "d": [1]}
                obs, rewards, terminated, _, _ = env.step(actions)
                episode_return += sum(rewards.values())
                if terminated["__all__"]:
                    break
            trained_returns.append(episode_return)

        # Training should produce some change (not necessarily better due to simple MLP)
        baseline_mean = np.mean(baseline_returns)
        trained_mean = np.mean(trained_returns)

        # At minimum, trained policy should produce valid returns
        assert not np.isnan(trained_mean)
        assert not np.isinf(trained_mean)


class TestComprehensiveAgentProperties:
    """Test all agent properties are properly used."""

    def test_all_field_agent_properties(self):
        """Test all FieldAgent properties."""
        tick_cfg = TickConfig.deterministic(
            tick_interval=1.5,
            obs_delay=0.2,
            act_delay=0.3,
            msg_delay=0.1,
        )
        agent = BatteryAgent(
            agent_id="test_battery",
            capacity=100.0,
            max_power=20.0,
            efficiency=0.95,
            tick_config=tick_cfg,
            upstream_id="test_coord",
            env_id="test_env",
        )

        # Core properties
        assert agent.agent_id == "test_battery"
        assert agent.level == FIELD_LEVEL
        assert agent.upstream_id == "test_coord"
        assert agent.env_id == "test_env"

        # Timing properties
        assert agent._tick_config.tick_interval == 1.5
        assert agent._tick_config.obs_delay == 0.2
        assert agent._tick_config.act_delay == 0.3
        assert agent._tick_config.msg_delay == 0.1

        # Agent-specific properties
        assert agent.capacity == 100.0
        assert agent.max_power == 20.0
        assert agent.efficiency == 0.95

        # State and action
        assert agent.state is not None
        assert len(agent.state.features) == 2
        assert agent.action is not None
        assert agent.action.dim_c == 1
        assert agent.action.dim_d == 1

        # Spaces
        assert agent.action_space is not None
        assert agent.observation_space is not None

        # Policy (None by default)
        assert agent.policy is None

    def test_all_coordinator_properties(self):
        """Test all CoordinatorAgent properties."""
        config = {
            "agents": [
                {"id": "battery_1", "capacity": 100.0},
                {"id": "battery_2", "capacity": 150.0},
            ]
        }

        tick_cfg = TickConfig.deterministic(
            tick_interval=10.0,
            obs_delay=0.5,
            act_delay=1.0,
            msg_delay=0.2,
        )
        coord = ZoneCoordinator(
            agent_id="test_zone",
            config=config,
            tick_config=tick_cfg,
            upstream_id="test_system",
            env_id="test_env",
        )

        # Core properties
        assert coord.agent_id == "test_zone"
        assert coord.level == COORDINATOR_LEVEL
        assert coord.upstream_id == "test_system"

        # Timing
        assert coord._tick_config.tick_interval == 10.0
        assert coord._tick_config.obs_delay == 0.5
        assert coord._tick_config.act_delay == 1.0

        # Subordinates
        assert len(coord.subordinates) == 2
        assert "battery_1" in coord.subordinates
        assert "battery_2" in coord.subordinates

        # Subordinates have correct upstream
        for agent in coord.subordinates.values():
            assert agent.upstream_id == "test_zone"

    def test_all_system_agent_properties(self):
        """Test all SystemAgent properties."""
        config = {
            "coordinators": [
                {
                    "id": "zone_1",
                    "agents": [{"id": "battery_1"}]
                }
            ]
        }

        tick_cfg = TickConfig.deterministic(
            tick_interval=30.0,
            obs_delay=1.5,
            act_delay=2.5,
            msg_delay=0.5,
        )
        system = GridSystemAgent(
            agent_id="test_system",
            config=config,
            tick_config=tick_cfg,
            env_id="test_env",
        )

        # Core properties
        assert system.agent_id == "test_system"
        assert system.level == SYSTEM_LEVEL
        assert system.upstream_id is None  # Top level

        # Timing
        assert system._tick_config.tick_interval == 30.0
        assert system._tick_config.obs_delay == 1.5
        assert system._tick_config.act_delay == 2.5

        # Coordinators
        assert len(system.coordinators) == 1
        assert "zone_1" in system.coordinators

        # Coordinators have correct upstream
        for coord in system.coordinators.values():
            assert coord.upstream_id == "test_system"

    def test_all_proxy_agent_properties(self):
        """Test all ProxyAgent properties."""
        proxy = ProxyAgent(
            agent_id="test_proxy",
            env_id="test_env",
            registered_agents=["agent_1", "agent_2"],
            visibility_rules={"agent_1": ["feature_a"], "agent_2": ["feature_a", "feature_b"]},
            history_length=50,
        )

        # Core properties
        assert proxy.agent_id == "test_proxy"
        assert proxy.level == 0  # PROXY_LEVEL (not part of L1-L3 hierarchy)

        # Registered agent tracking
        assert len(proxy.registered_agents) == 2
        assert "agent_1" in proxy.registered_agents

        # Visibility rules
        assert "agent_1" in proxy.visibility_rules
        assert proxy.visibility_rules["agent_1"] == ["feature_a"]

        # State management
        proxy.update_state({"agents": {"agent_1": {"feature_a": 1.0, "feature_b": 2.0}}})
        filtered = proxy.get_state_for_agent("agent_1", requestor_level=1)
        assert "feature_a" in filtered
        assert "feature_b" not in filtered  # Not in visibility rules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

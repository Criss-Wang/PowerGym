"""Integration tests for multi-agent environment.

Tests the complete HERON multi-agent environment including:
- Option A: Synchronous training mode with gymnasium interface
- Option B: Event-driven testing mode with realistic timing
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace

from heron.envs.base import MultiAgentEnv, EnvCore
from heron.agents.field_agent import FieldAgent, FIELD_LEVEL
from heron.agents.coordinator_agent import CoordinatorAgent, COORDINATOR_LEVEL
from heron.agents.system_agent import SystemAgent, SYSTEM_LEVEL
from heron.core.observation import Observation
from heron.core.feature import FeatureProvider
from heron.scheduling.scheduler import EventScheduler
from heron.scheduling.event import EventType
from heron.messaging.in_memory_broker import InMemoryBroker


# =============================================================================
# Test Fixtures - Environment and Agents
# =============================================================================

class DeviceFeature(FeatureProvider):
    """Device state feature."""
    visibility = ["public"]

    def __init__(self, power: float = 0.0, capacity: float = 100.0):
        self.power = power
        self.capacity = capacity

    def vector(self):
        return np.array([self.power, self.capacity], dtype=np.float32)

    def names(self):
        return ["power", "capacity"]

    def to_dict(self):
        return {"power": self.power, "capacity": self.capacity}

    @classmethod
    def from_dict(cls, d):
        return cls(power=d.get("power", 0.0), capacity=d.get("capacity", 100.0))

    def set_values(self, **kwargs):
        if "power" in kwargs:
            self.power = kwargs["power"]
        if "capacity" in kwargs:
            self.capacity = kwargs["capacity"]


class BatteryAgent(FieldAgent):
    """Battery device agent."""

    def __init__(self, agent_id, capacity=100.0, **kwargs):
        self.capacity = capacity
        self.charge_level = 50.0  # 50% initial charge
        super().__init__(agent_id=agent_id, **kwargs)

    def set_action(self):
        # Action: charge/discharge rate normalized to [-1, 1]
        self.action.set_specs(
            dim_c=1,
            range=(np.array([-1.0]), np.array([1.0]))
        )

    def set_state(self):
        self.state.features = [DeviceFeature(power=0.0, capacity=self.capacity)]

    def _get_obs(self, proxy=None):
        return np.array([self.charge_level / self.capacity], dtype=np.float32)

    def step_physics(self, dt: float = 1.0):
        """Simulate battery charge/discharge."""
        # action.c[0] is charge rate: positive = charge, negative = discharge
        rate = float(self.action.c[0]) * 10.0  # Scale to +/- 10 kW
        self.charge_level = np.clip(
            self.charge_level + rate * dt,
            0.0,
            self.capacity
        )
        self.state.features[0].power = rate


class MicrogridCoordinator(CoordinatorAgent):
    """Microgrid zone coordinator."""

    def _build_subordinates(self, agent_configs, env_id=None, upstream_id=None):
        agents = {}
        for config in agent_configs:
            agent_id = config.get("id")
            capacity = config.get("capacity", 100.0)
            agents[agent_id] = BatteryAgent(
                agent_id=agent_id,
                capacity=capacity,
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )
        return agents


class GridSystem(SystemAgent):
    """Grid system agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_state = {}
        self._total_power = 0.0

    def _build_subordinates(self, coordinator_configs, env_id=None, upstream_id=None):
        coordinators = {}
        for config in coordinator_configs:
            coord_id = config.get("id")
            agent_configs = config.get("agents", [])
            coordinators[coord_id] = MicrogridCoordinator(
                agent_id=coord_id,
                config={"agents": agent_configs},
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )
        return coordinators

    def update_from_environment(self, env_state):
        self._env_state = env_state

    def get_state_for_environment(self):
        # Collect all battery states
        actions = {}
        for coord_id, coord in self.coordinators.items():
            for agent_id, agent in coord.subordinates.items():
                actions[agent_id] = {
                    "power": float(agent.state.features[0].power),
                    "charge": agent.charge_level,
                }
        return {"device_states": actions}


class MicrogridEnv(MultiAgentEnv):
    """Multi-agent microgrid environment."""

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)

        config = config or {
            "coordinators": [
                {
                    "id": "zone_1",
                    "agents": [
                        {"id": "battery_1", "capacity": 100.0},
                        {"id": "battery_2", "capacity": 150.0},
                    ]
                }
            ]
        }

        # Create system agent with hierarchy
        self._grid_system = GridSystem(
            agent_id="grid_system",
            config=config,
            env_id=self.env_id,
        )

        # Register all agents
        self._register_all_agents()

        # Environment state
        self._load_demand = 0.0
        self._timestep = 0

    @property
    def grid_system(self):
        """Get the grid system agent."""
        return self._grid_system

    def _register_all_agents(self):
        """Register all agents in hierarchy with environment."""
        self.register_agent(self._grid_system)

        for coord_id, coordinator in self._grid_system.coordinators.items():
            self.register_agent(coordinator)
            for agent_id, agent in coordinator.subordinates.items():
                self.register_agent(agent)

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset environment and all agents."""
        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)

        self._timestep = 0
        self._load_demand = np.random.uniform(50, 150)

        # Reset all agents
        self.grid_system.reset(seed=seed)

        # Reset agent states
        self.reset_agents()

        # Get initial observations
        obs = self._get_agent_observations()
        info = {"load_demand": self._load_demand}

        return obs, info

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one environment step."""
        self._timestep += 1

        # Apply actions to agents
        self._apply_agent_actions(actions)

        # Step physics for all devices
        self._step_physics()

        # Calculate rewards
        rewards = self._calculate_rewards()

        # Get observations
        obs = self._get_agent_observations()

        # Check termination
        terminated = {"__all__": self._timestep >= 100}
        truncated = {"__all__": False}

        info = {
            "load_demand": self._load_demand,
            "total_power": self._get_total_power(),
        }

        return obs, rewards, terminated, truncated, info

    def _get_agent_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all controllable agents."""
        obs = {}
        global_state = {"load_demand": self._load_demand}

        # Use system agent's observe to get hierarchical observations
        system_obs = self.grid_system.observe(global_state=global_state)

        # Extract field agent observations
        for coord_id, coord in self.grid_system.coordinators.items():
            for agent_id, agent in coord.subordinates.items():
                agent_obs = agent.observe()
                obs[agent_id] = agent_obs.vector()

        return obs

    def _apply_agent_actions(self, actions: Dict[str, Any]):
        """Apply actions to agents through hierarchy."""
        # Build action dict for system agent
        system_obs = self.grid_system.observe()

        action_dict = {}
        for coord_id, coord in self.grid_system.coordinators.items():
            coord_actions = {}
            for agent_id in coord.subordinates:
                if agent_id in actions:
                    coord_actions[agent_id] = actions[agent_id]
            action_dict[coord_id] = coord_actions

        self.grid_system.act(system_obs, upstream_action=action_dict)

    def _step_physics(self):
        """Step physics for all devices."""
        for coord in self.grid_system.coordinators.values():
            for agent in coord.subordinates.values():
                if hasattr(agent, 'step_physics'):
                    agent.step_physics()

    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {}
        total_power = self._get_total_power()

        # Reward for meeting load demand
        power_deficit = abs(total_power - self._load_demand)
        base_reward = -power_deficit / 100.0  # Normalize

        for coord in self.grid_system.coordinators.values():
            for agent_id in coord.subordinates:
                rewards[agent_id] = base_reward

        return rewards

    def _get_total_power(self) -> float:
        """Get total power from all devices."""
        total = 0.0
        for coord in self.grid_system.coordinators.values():
            for agent in coord.subordinates.values():
                total += agent.state.features[0].power
        return total


# =============================================================================
# Integration Tests - Option A: Synchronous Training
# =============================================================================

class TestSynchronousTrainingMode:
    """Test synchronous training mode (Option A)."""

    def test_env_reset(self):
        """Test environment reset."""
        env = MicrogridEnv()
        obs, info = env.reset(seed=42)

        assert "battery_1" in obs
        assert "battery_2" in obs
        assert isinstance(obs["battery_1"], np.ndarray)
        assert "load_demand" in info

    def test_env_step(self):
        """Test environment step."""
        env = MicrogridEnv()
        env.reset(seed=42)

        actions = {
            "battery_1": np.array([0.5]),
            "battery_2": np.array([-0.3]),
        }

        obs, rewards, terminated, truncated, info = env.step(actions)

        assert "battery_1" in obs
        assert "battery_1" in rewards
        assert isinstance(rewards["battery_1"], float)
        assert "__all__" in terminated

    def test_full_episode(self):
        """Test running a full episode."""
        env = MicrogridEnv()
        obs, _ = env.reset(seed=42)

        total_rewards = {"battery_1": 0.0, "battery_2": 0.0}
        done = False
        steps = 0

        while not done and steps < 100:
            # Random actions
            actions = {
                "battery_1": np.random.uniform(-1, 1, size=(1,)),
                "battery_2": np.random.uniform(-1, 1, size=(1,)),
            }

            obs, rewards, terminated, truncated, info = env.step(actions)

            for agent_id in rewards:
                total_rewards[agent_id] += rewards[agent_id]

            done = terminated["__all__"] or truncated["__all__"]
            steps += 1

        assert steps == 100  # Episode should run for 100 steps

    def test_action_distribution_through_hierarchy(self):
        """Test that actions propagate correctly through hierarchy."""
        env = MicrogridEnv()
        env.reset(seed=42)

        # Set specific actions
        actions = {
            "battery_1": np.array([0.8]),
            "battery_2": np.array([-0.5]),
        }

        env.step(actions)

        # Verify actions reached field agents
        battery_1 = env.grid_system.coordinators["zone_1"].subordinates["battery_1"]
        battery_2 = env.grid_system.coordinators["zone_1"].subordinates["battery_2"]

        np.testing.assert_array_almost_equal(battery_1.action.c, [0.8])
        np.testing.assert_array_almost_equal(battery_2.action.c, [-0.5])

    def test_observation_aggregation(self):
        """Test that observations aggregate correctly."""
        env = MicrogridEnv()
        env.reset(seed=42)

        # Get system-level observation
        global_state = {"load_demand": 100.0}
        system_obs = env.grid_system.observe(global_state=global_state)

        assert isinstance(system_obs, Observation)
        assert "coordinator_obs" in system_obs.local
        assert "zone_1" in system_obs.local["coordinator_obs"]


class TestMultipleEnvironments:
    """Test multiple environment instances."""

    def test_environments_are_independent(self):
        """Test that multiple environments run independently."""
        env1 = MicrogridEnv(env_id="env_1")
        env2 = MicrogridEnv(env_id="env_2")

        env1.reset(seed=1)
        env2.reset(seed=2)

        # Run different actions in each
        actions_1 = {"battery_1": np.array([1.0]), "battery_2": np.array([1.0])}
        actions_2 = {"battery_1": np.array([-1.0]), "battery_2": np.array([-1.0])}

        env1.step(actions_1)
        env2.step(actions_2)

        # States should be different
        b1_env1 = env1.grid_system.coordinators["zone_1"].subordinates["battery_1"]
        b1_env2 = env2.grid_system.coordinators["zone_1"].subordinates["battery_1"]

        assert b1_env1.charge_level != b1_env2.charge_level


class TestMessageBrokerIntegration:
    """Test message broker integration with environment."""

    def test_agents_have_message_broker(self):
        """Test that all agents are configured with message broker."""
        env = MicrogridEnv()
        env.reset()

        # Configure for distributed
        env.configure_agents_for_distributed()

        # All agents should have broker
        for agent in env.registered_agents.values():
            assert agent.message_broker is not None
            assert agent.message_broker is env.message_broker


# =============================================================================
# Integration Tests - Option B: Event-Driven Testing
# =============================================================================

class TestEventDrivenMode:
    """Test event-driven testing mode (Option B)."""

    def test_scheduler_with_agent_hierarchy(self):
        """Test EventScheduler coordinating agent hierarchy."""
        scheduler = EventScheduler(start_time=0.0)

        # Register agents with different tick rates
        # Field level: fast (1s)
        scheduler.register_agent("battery_1", tick_interval=1.0)
        scheduler.register_agent("battery_2", tick_interval=1.0)
        # Coordinator level: medium (5s)
        scheduler.register_agent("zone_1", tick_interval=5.0)
        # System level: slow (15s)
        scheduler.register_agent("grid_system", tick_interval=15.0)

        tick_counts = {
            "battery_1": 0, "battery_2": 0,
            "zone_1": 0, "grid_system": 0
        }

        def handler(event, sched):
            if event.agent_id in tick_counts:
                tick_counts[event.agent_id] += 1

        scheduler.set_handler(EventType.AGENT_TICK, handler)

        # Run for 15 seconds
        scheduler.run_until(t_end=15.0)

        # Verify tick ratios match hierarchy levels
        assert tick_counts["battery_1"] > tick_counts["zone_1"]
        assert tick_counts["zone_1"] > tick_counts["grid_system"]
        assert tick_counts["battery_1"] == 16  # t=0,1,...,15
        assert tick_counts["zone_1"] == 4  # t=0,5,10,15
        assert tick_counts["grid_system"] == 2  # t=0,15

    def test_action_and_observation_delays(self):
        """Test realistic delays in event-driven mode."""
        scheduler = EventScheduler(start_time=0.0)

        # Agent with observation and action delays
        scheduler.register_agent(
            "controller",
            tick_interval=5.0,
            obs_delay=0.5,
            act_delay=1.0,
        )

        events_timeline = []

        def tick_handler(event, sched):
            events_timeline.append(("tick", event.timestamp))
            # Schedule observation delay (simulated)
            # After observing, schedule action effect
            sched.schedule_action_effect(
                event.agent_id,
                action={"power": 50.0},
            )

        def effect_handler(event, sched):
            events_timeline.append(("action_effect", event.timestamp))

        scheduler.set_handler(EventType.AGENT_TICK, tick_handler)
        scheduler.set_handler(EventType.ACTION_EFFECT, effect_handler)

        scheduler.run_until(t_end=12.0)

        # Extract tick and effect times
        tick_times = [t for e, t in events_timeline if e == "tick"]
        effect_times = [t for e, t in events_timeline if e == "action_effect"]

        # Ticks at t=0, 5, 10
        assert tick_times == [0.0, 5.0, 10.0]

        # Effects at t=1, 6, 11 (tick + act_delay=1.0)
        assert effect_times == [1.0, 6.0, 11.0]


class TestDualModeCompatibility:
    """Test that same agents work in both modes."""

    def test_agent_works_in_sync_mode(self):
        """Test agent in synchronous training mode."""
        env = MicrogridEnv()
        env.reset(seed=42)

        # Run sync mode steps
        for _ in range(10):
            actions = {
                "battery_1": np.array([0.5]),
                "battery_2": np.array([-0.5]),
            }
            env.step(actions)

        # Check state changed
        battery_1 = env.grid_system.coordinators["zone_1"].subordinates["battery_1"]
        assert battery_1.charge_level != 50.0  # Initial was 50

    def test_same_hierarchy_in_event_mode(self):
        """Test same hierarchy structure works with EventScheduler."""
        config = {
            "coordinators": [
                {
                    "id": "zone_1",
                    "agents": [
                        {"id": "battery_1", "capacity": 100.0},
                    ]
                }
            ]
        }

        system = GridSystem(agent_id="grid_system", config=config)
        scheduler = EventScheduler(start_time=0.0)

        # Register all agents in hierarchy
        scheduler.register_agent("grid_system", tick_interval=10.0)
        scheduler.register_agent("zone_1", tick_interval=5.0)
        scheduler.register_agent("battery_1", tick_interval=1.0)

        # Track observations
        observations_made = []

        def handler(event, sched):
            if event.agent_id == "battery_1":
                agent = system.coordinators["zone_1"].subordinates["battery_1"]
                agent._timestep = event.timestamp
                obs = agent.observe()
                observations_made.append({
                    "time": event.timestamp,
                    "obs_dim": len(obs.vector()),
                })

        scheduler.set_handler(EventType.AGENT_TICK, handler)
        scheduler.run_until(t_end=5.0)

        # Battery should have ticked 6 times (0,1,2,3,4,5)
        assert len(observations_made) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

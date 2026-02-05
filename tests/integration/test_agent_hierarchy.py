"""Integration tests for hierarchical agent structure.

Tests the core HERON agent hierarchy: Field (L1) → Coordinator (L2) → System (L3).
This is a key contribution of the framework enabling scalable multi-agent control.
"""

import pytest
import numpy as np
from gymnasium.spaces import Box

from heron.agents.field_agent import FieldAgent, FIELD_LEVEL
from heron.agents.coordinator_agent import CoordinatorAgent, COORDINATOR_LEVEL
from heron.agents.system_agent import SystemAgent, SYSTEM_LEVEL
from heron.core.observation import Observation
from heron.core.action import Action
from heron.core.state import FieldAgentState
from heron.core.feature import FeatureProvider
from heron.core.policies import Policy


# =============================================================================
# Test Fixtures - Concrete Implementations
# =============================================================================

class DeviceFeature(FeatureProvider):
    """Feature representing a device's state."""
    visibility = ["public"]

    def __init__(self, power: float = 0.0, status: float = 1.0):
        self.power = power
        self.status = status

    def vector(self):
        return np.array([self.power, self.status], dtype=np.float32)

    def names(self):
        return ["power", "status"]

    def to_dict(self):
        return {"power": self.power, "status": self.status}

    @classmethod
    def from_dict(cls, d):
        return cls(power=d.get("power", 0.0), status=d.get("status", 1.0))

    def set_values(self, **kwargs):
        if "power" in kwargs:
            self.power = kwargs["power"]
        if "status" in kwargs:
            self.status = kwargs["status"]


class DeviceAgent(FieldAgent):
    """Concrete field agent representing a controllable device."""

    def __init__(self, agent_id, device_type="generic", max_power=100.0, **kwargs):
        self.device_type = device_type
        self.max_power = max_power
        super().__init__(agent_id=agent_id, **kwargs)

    def set_action(self):
        # Continuous action: power setpoint normalized to [-1, 1]
        self.action.set_specs(
            dim_c=1,
            range=(np.array([-1.0]), np.array([1.0]))
        )

    def set_state(self):
        self.state.features = [DeviceFeature(power=0.0, status=1.0)]

    def _get_obs(self, proxy=None):
        return self.state.vector()


class ZoneCoordinator(CoordinatorAgent):
    """Coordinator managing a zone of devices."""

    def _build_subordinates(self, configs, env_id=None, upstream_id=None):
        agents = {}
        for config in configs:
            agent_id = config.get("id")
            device_type = config.get("device_type", "generic")
            max_power = config.get("max_power", 100.0)
            agents[agent_id] = DeviceAgent(
                agent_id=agent_id,
                device_type=device_type,
                max_power=max_power,
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )
        return agents


class GridSystemAgent(SystemAgent):
    """System agent managing the entire grid."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_state = {}
        self._actions_collected = {}

    def _build_subordinates(self, configs, env_id=None, upstream_id=None):
        coordinators = {}
        for config in configs:
            coord_id = config.get("id")
            agent_configs = config.get("agents", [])
            coordinators[coord_id] = ZoneCoordinator(
                agent_id=coord_id,
                config={"agents": agent_configs},
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )
        return coordinators

    def update_from_environment(self, env_state):
        self._env_state = env_state
        super().update_from_environment(env_state)

    def get_state_for_environment(self):
        result = super().get_state_for_environment()
        result["actions"] = self._actions_collected
        return result


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentHierarchyStructure:
    """Test the hierarchical structure of agents."""

    def test_three_level_hierarchy_creation(self):
        """Test creating a complete 3-level hierarchy."""
        config = {
            "coordinators": [
                {
                    "id": "zone_north",
                    "agents": [
                        {"id": "battery_1", "device_type": "battery", "max_power": 500},
                        {"id": "solar_1", "device_type": "solar", "max_power": 200},
                    ]
                },
                {
                    "id": "zone_south",
                    "agents": [
                        {"id": "battery_2", "device_type": "battery", "max_power": 300},
                    ]
                },
            ]
        }

        system = GridSystemAgent(agent_id="grid_system", config=config)

        # Verify hierarchy levels
        assert system.level == SYSTEM_LEVEL
        assert len(system.coordinators) == 2

        # Check coordinators
        zone_north = system.coordinators["zone_north"]
        assert zone_north.level == COORDINATOR_LEVEL
        assert len(zone_north.subordinates) == 2

        zone_south = system.coordinators["zone_south"]
        assert len(zone_south.subordinates) == 1

        # Check field agents
        battery_1 = zone_north.subordinates["battery_1"]
        assert battery_1.level == FIELD_LEVEL
        assert battery_1.device_type == "battery"

    def test_upstream_downstream_relationships(self):
        """Test that upstream/downstream relationships are correctly set."""
        config = {
            "coordinators": [
                {
                    "id": "zone_1",
                    "agents": [{"id": "device_1"}]
                }
            ]
        }

        system = GridSystemAgent(agent_id="grid_system", config=config)

        # System has no upstream
        assert system.upstream_id is None

        # Coordinator's upstream is system
        coordinator = system.coordinators["zone_1"]
        assert coordinator.upstream_id == "grid_system"

        # Field agent's upstream is coordinator
        device = coordinator.subordinates["device_1"]
        assert device.upstream_id == "zone_1"


class TestHierarchicalObservations:
    """Test observation aggregation through the hierarchy."""

    def test_observation_flows_bottom_up(self):
        """Test that observations aggregate from field to system level."""
        config = {
            "coordinators": [
                {
                    "id": "zone_1",
                    "agents": [
                        {"id": "device_1"},
                        {"id": "device_2"},
                    ]
                }
            ]
        }

        system = GridSystemAgent(agent_id="grid_system", config=config)

        # Set some state on field agents
        coordinator = system.coordinators["zone_1"]
        coordinator.subordinates["device_1"].state.features[0].power = 50.0
        coordinator.subordinates["device_2"].state.features[0].power = 75.0

        # Get system-level observation
        system_obs = system.observe()

        # Verify observation structure
        assert isinstance(system_obs, Observation)
        assert "coordinator_obs" in system_obs.local

        coord_obs = system_obs.local["coordinator_obs"]["zone_1"]
        assert isinstance(coord_obs, Observation)
        assert "subordinate_obs" in coord_obs.local

        # Verify field agent observations are included
        sub_obs = coord_obs.local["subordinate_obs"]
        assert "device_1" in sub_obs
        assert "device_2" in sub_obs

    def test_global_state_propagates_down(self):
        """Test that global state is passed down through hierarchy."""
        config = {
            "coordinators": [
                {"id": "zone_1", "agents": [{"id": "device_1"}]}
            ]
        }

        system = GridSystemAgent(agent_id="grid_system", config=config)
        global_state = {"grid_frequency": 60.0, "total_load": 1000.0}

        system_obs = system.observe(global_state=global_state)

        assert system_obs.global_info == global_state


class TestHierarchicalActions:
    """Test action distribution through the hierarchy."""

    def test_action_flows_top_down(self):
        """Test that actions distribute from system to field level."""
        config = {
            "coordinators": [
                {
                    "id": "zone_1",
                    "agents": [
                        {"id": "device_1"},
                        {"id": "device_2"},
                    ]
                }
            ]
        }

        system = GridSystemAgent(agent_id="grid_system", config=config)

        # Get observation first
        system_obs = system.observe()

        # Define actions for each device
        actions = {
            "zone_1": {
                "device_1": np.array([0.5]),
                "device_2": np.array([-0.3]),
            }
        }

        # Distribute actions
        system.act(system_obs, upstream_action=actions)

        # Verify actions reached field agents
        coordinator = system.coordinators["zone_1"]
        device_1 = coordinator.subordinates["device_1"]
        device_2 = coordinator.subordinates["device_2"]

        np.testing.assert_array_almost_equal(device_1.action.c, [0.5])
        np.testing.assert_array_almost_equal(device_2.action.c, [-0.3])

    def test_coordinator_distributes_flat_action(self):
        """Test coordinator can distribute a flat action array to subordinates."""
        config = {
            "agents": [
                {"id": "device_1"},
                {"id": "device_2"},
            ]
        }

        coordinator = ZoneCoordinator(agent_id="zone_1", config=config)

        # Get observation
        obs = coordinator.observe()

        # Flat action array (2 devices x 1 action dim each = 2 total)
        flat_action = np.array([0.7, -0.4])

        # Distribute to subordinates
        coordinator.act(obs, upstream_action=flat_action)

        # Verify each device got its portion
        np.testing.assert_array_almost_equal(
            coordinator.subordinates["device_1"].action.c, [0.7]
        )
        np.testing.assert_array_almost_equal(
            coordinator.subordinates["device_2"].action.c, [-0.4]
        )


class TestHierarchicalReset:
    """Test reset propagation through hierarchy."""

    def test_reset_cascades_through_hierarchy(self):
        """Test that reset propagates to all agents."""
        config = {
            "coordinators": [
                {
                    "id": "zone_1",
                    "agents": [
                        {"id": "device_1"},
                        {"id": "device_2"},
                    ]
                }
            ]
        }

        system = GridSystemAgent(agent_id="grid_system", config=config)

        # Modify timesteps at all levels
        system._timestep = 100.0
        system.coordinators["zone_1"]._timestep = 100.0
        system.coordinators["zone_1"].subordinates["device_1"]._timestep = 100.0

        # Reset from top
        system.reset()

        # Verify all levels are reset
        assert system._timestep == 0.0
        assert system.coordinators["zone_1"]._timestep == 0.0
        assert system.coordinators["zone_1"].subordinates["device_1"]._timestep == 0.0


class TestHierarchicalSpaces:
    """Test action/observation space construction."""

    def test_joint_action_space_construction(self):
        """Test that joint action space is correctly constructed."""
        config = {
            "agents": [
                {"id": "device_1"},
                {"id": "device_2"},
                {"id": "device_3"},
            ]
        }

        coordinator = ZoneCoordinator(agent_id="zone_1", config=config)

        # Get joint action space
        joint_space = coordinator.get_joint_action_space()

        # Each device has 1D continuous action
        assert isinstance(joint_space, Box)
        assert joint_space.shape == (3,)  # 3 devices x 1 dim each


class TestScalability:
    """Test scalability of the hierarchical structure."""

    def test_large_hierarchy(self):
        """Test creating a larger hierarchy with many agents."""
        # Create config with multiple zones, each with multiple devices
        config = {
            "coordinators": [
                {
                    "id": f"zone_{i}",
                    "agents": [
                        {"id": f"device_{i}_{j}"} for j in range(5)
                    ]
                }
                for i in range(4)
            ]
        }

        system = GridSystemAgent(agent_id="grid_system", config=config)

        # Verify structure
        assert len(system.coordinators) == 4

        total_devices = sum(
            len(coord.subordinates)
            for coord in system.coordinators.values()
        )
        assert total_devices == 20  # 4 zones x 5 devices

        # Verify observation collection works
        obs = system.observe()
        assert len(obs.local["coordinator_obs"]) == 4

        # Verify action distribution works
        actions = {
            f"zone_{i}": {
                f"device_{i}_{j}": np.array([0.1 * j])
                for j in range(5)
            }
            for i in range(4)
        }
        system.act(obs, upstream_action=actions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

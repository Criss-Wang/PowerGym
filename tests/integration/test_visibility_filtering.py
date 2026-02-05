"""Integration tests for visibility-based information filtering.

Tests the HERON FeatureProvider visibility system that enables realistic
information constraints in multi-agent coordination.
"""

import pytest
import numpy as np

from heron.core.feature import FeatureProvider
from heron.core.state import State, FieldAgentState
from heron.agents.field_agent import FieldAgent, FIELD_LEVEL
from heron.agents.coordinator_agent import CoordinatorAgent, COORDINATOR_LEVEL
from heron.agents.system_agent import SystemAgent, SYSTEM_LEVEL


# =============================================================================
# Test Features with Different Visibility Levels
# =============================================================================

class PublicMeasurement(FeatureProvider):
    """Publicly visible measurement (e.g., power output)."""
    visibility = ["public"]

    def __init__(self, power: float = 0.0):
        self.power = power

    def vector(self):
        return np.array([self.power], dtype=np.float32)

    def names(self):
        return ["power"]

    def to_dict(self):
        return {"power": self.power}

    @classmethod
    def from_dict(cls, d):
        return cls(power=d.get("power", 0.0))

    def set_values(self, **kwargs):
        if "power" in kwargs:
            self.power = kwargs["power"]


class OwnerOnlyState(FeatureProvider):
    """State only visible to the owning agent (e.g., internal temperature)."""
    visibility = ["owner"]

    def __init__(self, temperature: float = 25.0):
        self.temperature = temperature

    def vector(self):
        return np.array([self.temperature], dtype=np.float32)

    def names(self):
        return ["temperature"]

    def to_dict(self):
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(cls, d):
        return cls(temperature=d.get("temperature", 25.0))

    def set_values(self, **kwargs):
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]


class UpperLevelInfo(FeatureProvider):
    """Information only visible to immediate upstream coordinator."""
    visibility = ["upper_level"]

    def __init__(self, efficiency: float = 0.95):
        self.efficiency = efficiency

    def vector(self):
        return np.array([self.efficiency], dtype=np.float32)

    def names(self):
        return ["efficiency"]

    def to_dict(self):
        return {"efficiency": self.efficiency}

    @classmethod
    def from_dict(cls, d):
        return cls(efficiency=d.get("efficiency", 0.95))

    def set_values(self, **kwargs):
        if "efficiency" in kwargs:
            self.efficiency = kwargs["efficiency"]


class SystemOnlyInfo(FeatureProvider):
    """Information only visible at system level and above."""
    visibility = ["system"]

    def __init__(self, cost: float = 0.0):
        self.cost = cost

    def vector(self):
        return np.array([self.cost], dtype=np.float32)

    def names(self):
        return ["cost"]

    def to_dict(self):
        return {"cost": self.cost}

    @classmethod
    def from_dict(cls, d):
        return cls(cost=d.get("cost", 0.0))

    def set_values(self, **kwargs):
        if "cost" in kwargs:
            self.cost = kwargs["cost"]


class PrivateSecret(FeatureProvider):
    """Completely private information not visible to anyone."""
    visibility = ["private"]

    def __init__(self, secret: float = 42.0):
        self.secret = secret

    def vector(self):
        return np.array([self.secret], dtype=np.float32)

    def names(self):
        return ["secret"]

    def to_dict(self):
        return {"secret": self.secret}

    @classmethod
    def from_dict(cls, d):
        return cls(secret=d.get("secret", 42.0))

    def set_values(self, **kwargs):
        if "secret" in kwargs:
            self.secret = kwargs["secret"]


# =============================================================================
# Test Agents
# =============================================================================

class DeviceWithMultipleFeatures(FieldAgent):
    """Device with features at multiple visibility levels."""

    def set_action(self):
        self.action.set_specs(dim_c=1, range=(np.array([0.0]), np.array([1.0])))

    def set_state(self):
        self.state.features = [
            PublicMeasurement(power=100.0),
            OwnerOnlyState(temperature=35.0),
            UpperLevelInfo(efficiency=0.92),
            SystemOnlyInfo(cost=50.0),
            PrivateSecret(secret=999.0),
        ]

    def _get_obs(self, proxy=None):
        return self.state.vector()


class ZoneCoordinatorWithVisibility(CoordinatorAgent):
    """Coordinator that respects visibility when aggregating observations."""

    def _build_subordinates(self, agent_configs, env_id=None, upstream_id=None):
        agents = {}
        for config in agent_configs:
            agent_id = config.get("id")
            agents[agent_id] = DeviceWithMultipleFeatures(
                agent_id=agent_id,
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )
        return agents


class GridSystemWithVisibility(SystemAgent):
    """System agent that can filter state based on visibility."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_state = {}

    def _build_subordinates(self, coordinator_configs, env_id=None, upstream_id=None):
        coordinators = {}
        for config in coordinator_configs:
            coord_id = config.get("id")
            agent_configs = config.get("agents", [])
            coordinators[coord_id] = ZoneCoordinatorWithVisibility(
                agent_id=coord_id,
                config={"agents": agent_configs},
                env_id=env_id,
                upstream_id=upstream_id or self.agent_id,
            )
        return coordinators

    def update_from_environment(self, env_state):
        self._env_state = env_state

    def get_state_for_environment(self):
        return {}


# =============================================================================
# Integration Tests
# =============================================================================

class TestFeatureVisibilityRules:
    """Test that visibility rules are correctly applied."""

    def test_public_visible_to_all(self):
        """Test that public features are visible to everyone."""
        state = FieldAgentState(owner_id="device_1", owner_level=FIELD_LEVEL)
        state.features = [PublicMeasurement(power=100.0)]

        # Visible to same-level peer
        obs_peer = state.observed_by("device_2", FIELD_LEVEL)
        assert "PublicMeasurement" in obs_peer
        np.testing.assert_array_equal(obs_peer["PublicMeasurement"], [100.0])

        # Visible to coordinator (higher level)
        obs_coord = state.observed_by("coord_1", COORDINATOR_LEVEL)
        assert "PublicMeasurement" in obs_coord

        # Visible to system (even higher level)
        obs_system = state.observed_by("system", SYSTEM_LEVEL)
        assert "PublicMeasurement" in obs_system

    def test_owner_only_visibility(self):
        """Test that owner-only features are only visible to owner."""
        state = FieldAgentState(owner_id="device_1", owner_level=FIELD_LEVEL)
        state.features = [OwnerOnlyState(temperature=35.0)]

        # Visible to owner
        obs_owner = state.observed_by("device_1", FIELD_LEVEL)
        assert "OwnerOnlyState" in obs_owner

        # NOT visible to peer
        obs_peer = state.observed_by("device_2", FIELD_LEVEL)
        assert "OwnerOnlyState" not in obs_peer

        # NOT visible to coordinator
        obs_coord = state.observed_by("coord_1", COORDINATOR_LEVEL)
        assert "OwnerOnlyState" not in obs_coord

    def test_upper_level_visibility(self):
        """Test that upper_level features are only visible to immediate upstream."""
        state = FieldAgentState(owner_id="device_1", owner_level=FIELD_LEVEL)
        state.features = [UpperLevelInfo(efficiency=0.92)]

        # NOT visible to owner (same level)
        obs_owner = state.observed_by("device_1", FIELD_LEVEL)
        assert "UpperLevelInfo" not in obs_owner

        # Visible to immediate upstream (coordinator, level 2)
        obs_coord = state.observed_by("coord_1", COORDINATOR_LEVEL)
        assert "UpperLevelInfo" in obs_coord

        # NOT visible to system (two levels up)
        obs_system = state.observed_by("system", SYSTEM_LEVEL)
        assert "UpperLevelInfo" not in obs_system

    def test_system_level_visibility(self):
        """Test that system features are only visible at system level."""
        state = FieldAgentState(owner_id="device_1", owner_level=FIELD_LEVEL)
        state.features = [SystemOnlyInfo(cost=50.0)]

        # NOT visible to owner
        obs_owner = state.observed_by("device_1", FIELD_LEVEL)
        assert "SystemOnlyInfo" not in obs_owner

        # NOT visible to coordinator
        obs_coord = state.observed_by("coord_1", COORDINATOR_LEVEL)
        assert "SystemOnlyInfo" not in obs_coord

        # Visible to system agent
        obs_system = state.observed_by("system", SYSTEM_LEVEL)
        assert "SystemOnlyInfo" in obs_system

    def test_private_visibility(self):
        """Test that private features are not visible to anyone."""
        state = FieldAgentState(owner_id="device_1", owner_level=FIELD_LEVEL)
        state.features = [PrivateSecret(secret=999.0)]

        # NOT visible to owner
        obs_owner = state.observed_by("device_1", FIELD_LEVEL)
        assert "PrivateSecret" not in obs_owner

        # NOT visible to peer
        obs_peer = state.observed_by("device_2", FIELD_LEVEL)
        assert "PrivateSecret" not in obs_peer

        # NOT visible to coordinator
        obs_coord = state.observed_by("coord_1", COORDINATOR_LEVEL)
        assert "PrivateSecret" not in obs_coord

        # NOT visible to system
        obs_system = state.observed_by("system", SYSTEM_LEVEL)
        assert "PrivateSecret" not in obs_system


class TestHierarchicalVisibilityFiltering:
    """Test visibility filtering in hierarchical agent structure."""

    def test_coordinator_sees_appropriate_features(self):
        """Test coordinator sees public and upper_level features from subordinates."""
        config = {"agents": [{"id": "device_1"}]}
        coordinator = ZoneCoordinatorWithVisibility(
            agent_id="coord_1",
            config=config,
        )

        device = coordinator.subordinates["device_1"]
        device_state = device.state

        # Get what coordinator can see
        filtered = device_state.observed_by("coord_1", COORDINATOR_LEVEL)

        # Should see public
        assert "PublicMeasurement" in filtered

        # Should see upper_level (it's the immediate upstream)
        assert "UpperLevelInfo" in filtered

        # Should NOT see owner-only
        assert "OwnerOnlyState" not in filtered

        # Should NOT see system-only
        assert "SystemOnlyInfo" not in filtered

        # Should NOT see private
        assert "PrivateSecret" not in filtered

    def test_system_agent_visibility_filtering(self):
        """Test system agent can filter state for different requestors."""
        config = {
            "coordinators": [
                {"id": "zone_1", "agents": [{"id": "device_1"}]}
            ]
        }

        system = GridSystemWithVisibility(agent_id="grid_system", config=config)

        device = system.coordinators["zone_1"].subordinates["device_1"]

        # System filters state for a coordinator
        coord_view = system.filter_state_for_agent(
            device.state,
            requestor_id="zone_1",
            requestor_level=COORDINATOR_LEVEL,
        )

        # Coordinator should see public and upper_level
        assert "PublicMeasurement" in coord_view
        assert "UpperLevelInfo" in coord_view
        assert "SystemOnlyInfo" not in coord_view

        # System filters state for system level
        system_view = system.filter_state_for_agent(
            device.state,
            requestor_id="grid_system",
            requestor_level=SYSTEM_LEVEL,
        )

        # System should see public and system-only
        assert "PublicMeasurement" in system_view
        assert "SystemOnlyInfo" in system_view
        # Upper level is specifically for one level up, not system
        assert "UpperLevelInfo" not in system_view


class TestInformationConstraintsInCoordination:
    """Test that information constraints affect coordination behavior."""

    def test_peer_cannot_see_internal_state(self):
        """Test that peer agents cannot observe each other's internal state."""
        config = {
            "agents": [
                {"id": "device_1"},
                {"id": "device_2"},
            ]
        }

        coordinator = ZoneCoordinatorWithVisibility(agent_id="coord_1", config=config)

        device_1 = coordinator.subordinates["device_1"]
        device_2 = coordinator.subordinates["device_2"]

        # Device 1 tries to observe Device 2's state
        device_2_view_by_1 = device_2.state.observed_by("device_1", FIELD_LEVEL)

        # Can see public measurement
        assert "PublicMeasurement" in device_2_view_by_1

        # Cannot see internal temperature
        assert "OwnerOnlyState" not in device_2_view_by_1

    def test_visibility_affects_observation_vector_size(self):
        """Test that visibility filtering affects observation vector size."""
        state = FieldAgentState(owner_id="device_1", owner_level=FIELD_LEVEL)
        state.features = [
            PublicMeasurement(power=100.0),      # 1 dim, public
            OwnerOnlyState(temperature=35.0),    # 1 dim, owner only
            UpperLevelInfo(efficiency=0.92),     # 1 dim, upper_level
            SystemOnlyInfo(cost=50.0),           # 1 dim, system
        ]

        # Owner sees 2 features (public + owner)
        owner_obs = state.observed_by("device_1", FIELD_LEVEL)
        owner_total_dims = sum(len(v) for v in owner_obs.values())
        assert owner_total_dims == 2

        # Coordinator sees 2 features (public + upper_level)
        coord_obs = state.observed_by("coord_1", COORDINATOR_LEVEL)
        coord_total_dims = sum(len(v) for v in coord_obs.values())
        assert coord_total_dims == 2

        # System sees 2 features (public + system)
        system_obs = state.observed_by("system", SYSTEM_LEVEL)
        system_total_dims = sum(len(v) for v in system_obs.values())
        assert system_total_dims == 2


class TestVisibilityWithMultipleAgents:
    """Test visibility across multiple agents in hierarchy."""

    def test_full_hierarchy_visibility(self):
        """Test visibility constraints across full 3-level hierarchy."""
        config = {
            "coordinators": [
                {
                    "id": "zone_1",
                    "agents": [
                        {"id": "device_1"},
                        {"id": "device_2"},
                    ]
                },
                {
                    "id": "zone_2",
                    "agents": [
                        {"id": "device_3"},
                    ]
                }
            ]
        }

        system = GridSystemWithVisibility(agent_id="grid_system", config=config)

        # Get device 1's state
        device_1_state = system.coordinators["zone_1"].subordinates["device_1"].state

        # Device 2 (same zone peer) view
        peer_view = device_1_state.observed_by("device_2", FIELD_LEVEL)
        assert "PublicMeasurement" in peer_view
        assert "OwnerOnlyState" not in peer_view

        # Device 3 (different zone peer) view - same as any peer
        other_zone_view = device_1_state.observed_by("device_3", FIELD_LEVEL)
        assert "PublicMeasurement" in other_zone_view
        assert "OwnerOnlyState" not in other_zone_view

        # Zone 1 coordinator view
        coord_1_view = device_1_state.observed_by("zone_1", COORDINATOR_LEVEL)
        assert "PublicMeasurement" in coord_1_view
        assert "UpperLevelInfo" in coord_1_view

        # Zone 2 coordinator (not direct upstream) - sees same as any coordinator
        coord_2_view = device_1_state.observed_by("zone_2", COORDINATOR_LEVEL)
        assert "PublicMeasurement" in coord_2_view

        # System view
        system_view = device_1_state.observed_by("grid_system", SYSTEM_LEVEL)
        assert "PublicMeasurement" in system_view
        assert "SystemOnlyInfo" in system_view


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

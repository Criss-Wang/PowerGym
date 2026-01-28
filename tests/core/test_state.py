"""Tests for core.state module."""

import numpy as np
import pytest

from heron.core.state import FieldAgentState, CoordinatorAgentState
from heron.core.feature import FeatureProvider


class MockFeature(FeatureProvider):
    """Mock feature provider for testing."""

    feature_name = "MockFeature"

    def __init__(self, value1=0.0, value2=0.0, visibility=None):
        self.value1 = value1
        self.value2 = value2
        self._visibility = visibility or ["owner"]
        self._reset_value1 = value1
        self._reset_value2 = value2

    def vector(self) -> np.ndarray:
        return np.array([self.value1, self.value2], dtype=np.float32)

    def names(self) -> list:
        return ["value1", "value2"]

    def reset(self):
        self.value1 = self._reset_value1
        self.value2 = self._reset_value2

    def set_values(self, **kwargs):
        if "value1" in kwargs:
            self.value1 = kwargs["value1"]
        if "value2" in kwargs:
            self.value2 = kwargs["value2"]

    def to_dict(self) -> dict:
        return {"value1": self.value1, "value2": self.value2}

    @classmethod
    def from_dict(cls, d: dict) -> "MockFeature":
        return cls(value1=d.get("value1", 0.0), value2=d.get("value2", 0.0))

    def is_observable_by(
        self,
        requestor_id: str,
        requestor_level: int,
        owner_id: str,
        owner_level: int
    ) -> bool:
        """Check if this feature is observable by the requestor."""
        if "owner" in self._visibility and requestor_id == owner_id:
            return True
        return False


class TestFieldAgentState:
    """Test FieldAgentState (DeviceState) class."""

    def test_state_initialization_empty(self):
        """Test state initialization with no features."""
        state = FieldAgentState(owner_id="agent1", owner_level=1)

        assert state.owner_id == "agent1"
        assert state.owner_level == 1
        assert len(state.features) == 0

    def test_state_with_features(self):
        """Test state initialization with features."""
        feature1 = MockFeature(value1=1.0, value2=2.0)
        feature2 = MockFeature(value1=3.0, value2=4.0)

        state = FieldAgentState(
            owner_id="agent1",
            owner_level=1,
            features=[feature1, feature2]
        )

        assert len(state.features) == 2
        assert state.features[0] is feature1
        assert state.features[1] is feature2

    def test_vector_concatenation(self):
        """Test that vector() concatenates all feature vectors."""
        feature1 = MockFeature(value1=1.0, value2=2.0)
        feature2 = MockFeature(value1=3.0, value2=4.0)

        state = FieldAgentState(
            owner_id="agent1",
            owner_level=1,
            features=[feature1, feature2]
        )

        vec = state.vector()

        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        np.testing.assert_array_equal(vec, [1.0, 2.0, 3.0, 4.0])

    def test_vector_empty_features(self):
        """Test vector with no features."""
        state = FieldAgentState(owner_id="agent1", owner_level=1)

        vec = state.vector()

        assert isinstance(vec, np.ndarray)
        assert len(vec) == 0

    def test_reset_features(self):
        """Test that reset() resets all features."""
        feature = MockFeature(value1=1.0, value2=2.0)
        state = FieldAgentState(
            owner_id="agent1",
            owner_level=1,
            features=[feature]
        )

        # Modify feature
        feature.value1 = 5.0
        feature.value2 = 6.0

        # Reset
        state.reset()

        assert feature.value1 == 1.0  # Reset value
        assert feature.value2 == 2.0  # Reset value

    def test_update_feature(self):
        """Test updating a specific feature."""
        feature = MockFeature(value1=1.0, value2=2.0)
        state = FieldAgentState(
            owner_id="agent1",
            owner_level=1,
            features=[feature]
        )

        state.update_feature("MockFeature", value1=10.0, value2=20.0)

        assert feature.value1 == 10.0
        assert feature.value2 == 20.0

    def test_update_batch(self):
        """Test batch update via update() method."""
        feature1 = MockFeature(value1=1.0, value2=2.0)
        feature2 = MockFeature(value1=3.0, value2=4.0)
        feature2.feature_name = "MockFeature2"

        state = FieldAgentState(
            owner_id="agent1",
            owner_level=1,
            features=[feature1, feature2]
        )

        state.update({
            "MockFeature": {"value1": 10.0},
            "MockFeature2": {"value2": 40.0}
        })

        assert feature1.value1 == 10.0
        assert feature1.value2 == 2.0  # Unchanged
        assert feature2.value1 == 3.0  # Unchanged
        assert feature2.value2 == 40.0

    def test_observed_by_owner(self):
        """Test that owner can observe their own state."""
        feature = MockFeature(value1=1.0, value2=2.0, visibility=["owner"])
        state = FieldAgentState(
            owner_id="agent1",
            owner_level=1,
            features=[feature]
        )

        obs = state.observed_by("agent1", requestor_level=1)

        assert "MockFeature" in obs
        np.testing.assert_array_equal(obs["MockFeature"], [1.0, 2.0])

    def test_to_dict(self):
        """Test serialization to dict."""
        feature = MockFeature(value1=1.0, value2=2.0)
        state = FieldAgentState(
            owner_id="agent1",
            owner_level=1,
            features=[feature]
        )

        d = state.to_dict()

        assert "MockFeature" in d
        assert d["MockFeature"]["value1"] == 1.0
        assert d["MockFeature"]["value2"] == 2.0


class TestCoordinatorAgentState:
    """Test CoordinatorAgentState class."""

    def test_coordinator_state_initialization(self):
        """Test coordinator state initialization."""
        state = CoordinatorAgentState(owner_id="coordinator1", owner_level=2)

        assert state.owner_id == "coordinator1"
        assert state.owner_level == 2
        assert len(state.features) == 0

    def test_coordinator_state_with_features(self):
        """Test coordinator state with features."""
        feature = MockFeature(value1=5.0, value2=6.0)
        state = CoordinatorAgentState(
            owner_id="coordinator1",
            owner_level=2,
            features=[feature]
        )

        assert len(state.features) == 1
        vec = state.vector()
        np.testing.assert_array_equal(vec, [5.0, 6.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

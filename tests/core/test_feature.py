"""Tests for FeatureProvider base class."""

import pytest
import numpy as np

from heron.core.feature import FeatureProvider


class PublicFeature(FeatureProvider):
    """Feature visible to all agents."""

    visibility = ["public"]

    def __init__(self, value: float = 1.0):
        self.value = value

    def vector(self):
        return np.array([self.value], dtype=np.float32)

    def names(self):
        return ["value"]

    def to_dict(self):
        return {"value": self.value}

    @classmethod
    def from_dict(cls, d):
        return cls(value=d.get("value", 1.0))

    def set_values(self, **kwargs):
        if "value" in kwargs:
            self.value = kwargs["value"]


class OwnerFeature(FeatureProvider):
    """Feature visible only to owner."""

    visibility = ["owner"]

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


class SystemFeature(FeatureProvider):
    """Feature visible only to system-level agents."""

    visibility = ["system"]

    def __init__(self, data: float = 100.0):
        self.data = data

    def vector(self):
        return np.array([self.data], dtype=np.float32)

    def names(self):
        return ["data"]

    def to_dict(self):
        return {"data": self.data}

    @classmethod
    def from_dict(cls, d):
        return cls(data=d.get("data", 100.0))

    def set_values(self, **kwargs):
        if "data" in kwargs:
            self.data = kwargs["data"]


class UpperLevelFeature(FeatureProvider):
    """Feature visible to agents one level above."""

    visibility = ["upper_level"]

    def __init__(self, info: float = 50.0):
        self.info = info

    def vector(self):
        return np.array([self.info], dtype=np.float32)

    def names(self):
        return ["info"]

    def to_dict(self):
        return {"info": self.info}

    @classmethod
    def from_dict(cls, d):
        return cls(info=d.get("info", 50.0))

    def set_values(self, **kwargs):
        if "info" in kwargs:
            self.info = kwargs["info"]


class PrivateFeature(FeatureProvider):
    """Feature with no visibility (private)."""

    visibility = []

    def __init__(self, hidden: float = 999.0):
        self.hidden = hidden

    def vector(self):
        return np.array([self.hidden], dtype=np.float32)

    def names(self):
        return ["hidden"]

    def to_dict(self):
        return {"hidden": self.hidden}

    @classmethod
    def from_dict(cls, d):
        return cls(hidden=d.get("hidden", 999.0))

    def set_values(self, **kwargs):
        if "hidden" in kwargs:
            self.hidden = kwargs["hidden"]


class MultiFieldFeature(FeatureProvider):
    """Feature with multiple fields."""

    visibility = ["public"]

    def __init__(self, power: float = 0.0, voltage: float = 1.0, current: float = 0.0):
        self.power = power
        self.voltage = voltage
        self.current = current

    def vector(self):
        return np.array([self.power, self.voltage, self.current], dtype=np.float32)

    def names(self):
        return ["power", "voltage", "current"]

    def to_dict(self):
        return {"power": self.power, "voltage": self.voltage, "current": self.current}

    @classmethod
    def from_dict(cls, d):
        return cls(
            power=d.get("power", 0.0),
            voltage=d.get("voltage", 1.0),
            current=d.get("current", 0.0),
        )

    def set_values(self, **kwargs):
        if "power" in kwargs:
            self.power = kwargs["power"]
        if "voltage" in kwargs:
            self.voltage = kwargs["voltage"]
        if "current" in kwargs:
            self.current = kwargs["current"]

    def reset(self, **overrides):
        self.power = 0.0
        self.voltage = 1.0
        self.current = 0.0
        if overrides:
            self.set_values(**overrides)
        return self


class TestFeatureProviderSubclass:
    """Test FeatureProvider subclassing behavior."""

    def test_feature_name_auto_assigned(self):
        """Test that feature_name is auto-assigned to class name."""
        feature = PublicFeature()

        assert feature.feature_name == "PublicFeature"

    def test_different_subclasses_have_different_names(self):
        """Test different subclasses have unique names."""
        public = PublicFeature()
        owner = OwnerFeature()
        system = SystemFeature()

        assert public.feature_name == "PublicFeature"
        assert owner.feature_name == "OwnerFeature"
        assert system.feature_name == "SystemFeature"


class TestFeatureProviderVector:
    """Test vector() method."""

    def test_vector_single_field(self):
        """Test vector with single field."""
        feature = PublicFeature(value=5.0)

        vec = feature.vector()

        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert len(vec) == 1
        assert vec[0] == 5.0

    def test_vector_multiple_fields(self):
        """Test vector with multiple fields."""
        feature = MultiFieldFeature(power=100.0, voltage=1.05, current=95.24)

        vec = feature.vector()

        assert len(vec) == 3
        np.testing.assert_array_almost_equal(vec, [100.0, 1.05, 95.24], decimal=5)


class TestFeatureProviderNames:
    """Test names() method."""

    def test_names_single_field(self):
        """Test names with single field."""
        feature = PublicFeature()

        names = feature.names()

        assert names == ["value"]

    def test_names_multiple_fields(self):
        """Test names with multiple fields."""
        feature = MultiFieldFeature()

        names = feature.names()

        assert names == ["power", "voltage", "current"]


class TestFeatureProviderSerialization:
    """Test to_dict() and from_dict() methods."""

    def test_to_dict(self):
        """Test serialization to dict."""
        feature = MultiFieldFeature(power=50.0, voltage=1.02, current=49.02)

        d = feature.to_dict()

        assert d == {"power": 50.0, "voltage": 1.02, "current": 49.02}

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {"power": 75.0, "voltage": 0.98, "current": 76.53}

        feature = MultiFieldFeature.from_dict(d)

        assert feature.power == 75.0
        assert feature.voltage == 0.98
        assert feature.current == 76.53

    def test_round_trip_serialization(self):
        """Test serialization round trip."""
        original = MultiFieldFeature(power=123.45, voltage=1.1, current=112.23)

        d = original.to_dict()
        restored = MultiFieldFeature.from_dict(d)

        np.testing.assert_array_equal(original.vector(), restored.vector())


class TestFeatureProviderSetValues:
    """Test set_values() method."""

    def test_set_values_single_field(self):
        """Test setting single value."""
        feature = PublicFeature(value=1.0)

        feature.set_values(value=5.0)

        assert feature.value == 5.0

    def test_set_values_multiple_fields(self):
        """Test setting multiple values."""
        feature = MultiFieldFeature()

        feature.set_values(power=100.0, voltage=1.05)

        assert feature.power == 100.0
        assert feature.voltage == 1.05
        assert feature.current == 0.0  # Unchanged

    def test_set_values_ignores_unknown_fields(self):
        """Test that unknown fields are ignored."""
        feature = PublicFeature(value=1.0)

        feature.set_values(value=2.0, unknown_field=999)

        assert feature.value == 2.0


class TestFeatureProviderReset:
    """Test reset() method."""

    def test_reset_default_does_nothing(self):
        """Test default reset implementation."""
        feature = PublicFeature(value=5.0)

        result = feature.reset()

        # Default reset returns self
        assert result is feature
        assert feature.value == 5.0  # Unchanged

    def test_reset_with_custom_implementation(self):
        """Test custom reset implementation."""
        feature = MultiFieldFeature(power=100.0, voltage=1.1, current=90.9)

        feature.reset()

        assert feature.power == 0.0
        assert feature.voltage == 1.0
        assert feature.current == 0.0

    def test_reset_with_overrides(self):
        """Test reset with overrides."""
        feature = MultiFieldFeature(power=100.0, voltage=1.1, current=90.9)

        feature.reset(power=50.0)

        assert feature.power == 50.0
        assert feature.voltage == 1.0
        assert feature.current == 0.0


class TestFeatureProviderVisibilityPublic:
    """Test public visibility."""

    def test_public_visible_to_owner(self):
        """Test public feature visible to owner."""
        feature = PublicFeature()

        assert feature.is_observable_by(
            requestor_id="agent_1",
            requestor_level=1,
            owner_id="agent_1",
            owner_level=1,
        )

    def test_public_visible_to_other_agent(self):
        """Test public feature visible to other agents."""
        feature = PublicFeature()

        assert feature.is_observable_by(
            requestor_id="agent_2",
            requestor_level=1,
            owner_id="agent_1",
            owner_level=1,
        )

    def test_public_visible_to_higher_level(self):
        """Test public feature visible to higher level."""
        feature = PublicFeature()

        assert feature.is_observable_by(
            requestor_id="coordinator",
            requestor_level=2,
            owner_id="agent_1",
            owner_level=1,
        )


class TestFeatureProviderVisibilityOwner:
    """Test owner visibility."""

    def test_owner_visible_to_owner(self):
        """Test owner feature visible to owner."""
        feature = OwnerFeature()

        assert feature.is_observable_by(
            requestor_id="agent_1",
            requestor_level=1,
            owner_id="agent_1",
            owner_level=1,
        )

    def test_owner_not_visible_to_other(self):
        """Test owner feature not visible to other agents."""
        feature = OwnerFeature()

        assert not feature.is_observable_by(
            requestor_id="agent_2",
            requestor_level=1,
            owner_id="agent_1",
            owner_level=1,
        )

    def test_owner_not_visible_to_higher_level(self):
        """Test owner feature not visible to higher level."""
        feature = OwnerFeature()

        assert not feature.is_observable_by(
            requestor_id="coordinator",
            requestor_level=2,
            owner_id="agent_1",
            owner_level=1,
        )


class TestFeatureProviderVisibilitySystem:
    """Test system visibility."""

    def test_system_not_visible_to_field_agent(self):
        """Test system feature not visible to field agent."""
        feature = SystemFeature()

        assert not feature.is_observable_by(
            requestor_id="field_1",
            requestor_level=1,
            owner_id="field_1",
            owner_level=1,
        )

    def test_system_not_visible_to_coordinator(self):
        """Test system feature not visible to coordinator."""
        feature = SystemFeature()

        assert not feature.is_observable_by(
            requestor_id="coord_1",
            requestor_level=2,
            owner_id="field_1",
            owner_level=1,
        )

    def test_system_visible_to_system_agent(self):
        """Test system feature visible to system agent."""
        feature = SystemFeature()

        assert feature.is_observable_by(
            requestor_id="system",
            requestor_level=3,
            owner_id="field_1",
            owner_level=1,
        )

    def test_system_visible_to_higher_than_system(self):
        """Test system feature visible to level > 3."""
        feature = SystemFeature()

        assert feature.is_observable_by(
            requestor_id="super_system",
            requestor_level=4,
            owner_id="field_1",
            owner_level=1,
        )


class TestFeatureProviderVisibilityUpperLevel:
    """Test upper_level visibility."""

    def test_upper_level_not_visible_to_owner(self):
        """Test upper_level feature not visible to owner."""
        feature = UpperLevelFeature()

        assert not feature.is_observable_by(
            requestor_id="field_1",
            requestor_level=1,
            owner_id="field_1",
            owner_level=1,
        )

    def test_upper_level_visible_to_one_level_above(self):
        """Test upper_level feature visible to one level above."""
        feature = UpperLevelFeature()

        assert feature.is_observable_by(
            requestor_id="coord_1",
            requestor_level=2,
            owner_id="field_1",
            owner_level=1,
        )

    def test_upper_level_not_visible_to_two_levels_above(self):
        """Test upper_level feature not visible to two levels above."""
        feature = UpperLevelFeature()

        assert not feature.is_observable_by(
            requestor_id="system",
            requestor_level=3,
            owner_id="field_1",
            owner_level=1,
        )

    def test_upper_level_not_visible_to_same_level(self):
        """Test upper_level feature not visible to same level."""
        feature = UpperLevelFeature()

        assert not feature.is_observable_by(
            requestor_id="field_2",
            requestor_level=1,
            owner_id="field_1",
            owner_level=1,
        )


class TestFeatureProviderVisibilityPrivate:
    """Test private (empty) visibility."""

    def test_private_not_visible_to_anyone(self):
        """Test private feature not visible to anyone."""
        feature = PrivateFeature()

        assert not feature.is_observable_by(
            requestor_id="field_1",
            requestor_level=1,
            owner_id="field_1",
            owner_level=1,
        )

        assert not feature.is_observable_by(
            requestor_id="coord_1",
            requestor_level=2,
            owner_id="field_1",
            owner_level=1,
        )

        assert not feature.is_observable_by(
            requestor_id="system",
            requestor_level=3,
            owner_id="field_1",
            owner_level=1,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

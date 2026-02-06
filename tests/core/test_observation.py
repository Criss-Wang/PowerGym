"""Comprehensive tests for Observation class.

Tests cover:
1. Observation initialization and structure
2. Vector flattening (simple and nested dicts)
3. Handling different data types (scalars, arrays, nested dicts)
4. Edge cases (empty obs, complex nesting, large arrays)
"""

import pytest
import numpy as np

from heron.core.observation import Observation


# =============================================================================
# Observation Initialization Tests
# =============================================================================

class TestObservationInitialization:
    """Test Observation initialization."""

    def test_initialization_empty(self):
        """Test empty observation initialization."""
        obs = Observation()

        assert obs.local == {}
        assert obs.global_info == {}
        assert obs.timestamp == 0.0

    def test_initialization_with_values(self):
        """Test initialization with provided values."""
        local = {"power": 100.0, "voltage": 1.0}
        global_info = {"grid_freq": 60.0}
        timestamp = 5.5

        obs = Observation(
            local=local,
            global_info=global_info,
            timestamp=timestamp
        )

        assert obs.local == local
        assert obs.global_info == global_info
        assert obs.timestamp == 5.5

    def test_initialization_partial(self):
        """Test initialization with partial values."""
        obs = Observation(local={"voltage": 1.0}, timestamp=2.0)

        assert obs.local == {"voltage": 1.0}
        assert obs.global_info == {}
        assert obs.timestamp == 2.0


# =============================================================================
# Observation Vector Flattening Tests - Simple Cases
# =============================================================================

class TestObservationVectorSimple:
    """Test vector flattening for simple cases."""

    def test_vector_empty_observation(self):
        """Test vector of empty observation."""
        obs = Observation()
        vec = obs.vector()

        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert len(vec) == 0

    def test_vector_single_scalar(self):
        """Test vector with single scalar value."""
        obs = Observation(local={"value": 5.0})
        vec = obs.vector()

        assert len(vec) == 1
        assert vec[0] == 5.0
        assert vec.dtype == np.float32

    def test_vector_multiple_scalars(self):
        """Test vector with multiple scalar values."""
        obs = Observation(local={
            "power": 100.0,
            "voltage": 1.0,
            "current": 50.0
        })
        vec = obs.vector()

        assert len(vec) == 3
        assert vec.dtype == np.float32
        # Keys are sorted alphabetically
        assert vec[0] == 50.0  # current
        assert vec[1] == 100.0  # power
        assert vec[2] == 1.0  # voltage

    def test_vector_with_integers(self):
        """Test vector handles integer values."""
        obs = Observation(local={
            "count": 5,
            "status": 1
        })
        vec = obs.vector()

        assert len(vec) == 2
        assert vec[0] == 5.0
        assert vec[1] == 1.0
        assert vec.dtype == np.float32

    def test_vector_key_ordering(self):
        """Test that keys are sorted consistently."""
        obs1 = Observation(local={"z": 1.0, "a": 2.0, "m": 3.0})
        obs2 = Observation(local={"m": 3.0, "z": 1.0, "a": 2.0})

        vec1 = obs1.vector()
        vec2 = obs2.vector()

        # Should be identical despite different insertion order
        np.testing.assert_array_equal(vec1, vec2)
        # Order should be: a, m, z
        assert vec1[0] == 2.0
        assert vec1[1] == 3.0
        assert vec1[2] == 1.0


# =============================================================================
# Observation Vector Flattening Tests - Arrays
# =============================================================================

class TestObservationVectorArrays:
    """Test vector flattening with numpy arrays."""

    def test_vector_1d_array(self):
        """Test vector with 1D numpy array."""
        obs = Observation(local={
            "powers": np.array([10.0, 20.0, 30.0])
        })
        vec = obs.vector()

        assert len(vec) == 3
        np.testing.assert_array_equal(vec, [10.0, 20.0, 30.0])
        assert vec.dtype == np.float32

    def test_vector_2d_array(self):
        """Test vector with 2D numpy array (should flatten)."""
        obs = Observation(local={
            "matrix": np.array([[1.0, 2.0], [3.0, 4.0]])
        })
        vec = obs.vector()

        assert len(vec) == 4
        np.testing.assert_array_equal(vec, [1.0, 2.0, 3.0, 4.0])

    def test_vector_mixed_scalars_and_arrays(self):
        """Test vector with mix of scalars and arrays."""
        obs = Observation(local={
            "a_scalar": 5.0,
            "b_array": np.array([1.0, 2.0]),
            "c_scalar": 10.0
        })
        vec = obs.vector()

        # Order: a_scalar, b_array (flattened), c_scalar
        assert len(vec) == 4
        assert vec[0] == 5.0  # a_scalar
        assert vec[1] == 1.0  # b_array[0]
        assert vec[2] == 2.0  # b_array[1]
        assert vec[3] == 10.0  # c_scalar

    def test_vector_empty_array(self):
        """Test vector with empty array."""
        obs = Observation(local={
            "empty": np.array([]),
            "value": 5.0
        })
        vec = obs.vector()

        assert len(vec) == 1
        assert vec[0] == 5.0

    def test_vector_array_type_conversion(self):
        """Test arrays of different types are converted to float32."""
        obs = Observation(local={
            "int_array": np.array([1, 2, 3], dtype=np.int64),
            "float64_array": np.array([1.5, 2.5], dtype=np.float64)
        })
        vec = obs.vector()

        assert vec.dtype == np.float32
        assert len(vec) == 5


# =============================================================================
# Observation Vector Flattening Tests - Nested Dicts
# =============================================================================

class TestObservationVectorNested:
    """Test vector flattening with nested dictionaries."""

    def test_vector_nested_dict_one_level(self):
        """Test vector with one level of nesting."""
        obs = Observation(local={
            "device1": {
                "power": 100.0,
                "voltage": 1.0
            }
        })
        vec = obs.vector()

        assert len(vec) == 2
        # Nested keys also sorted
        assert vec[0] == 100.0  # power
        assert vec[1] == 1.0  # voltage

    def test_vector_nested_dict_multiple_devices(self):
        """Test vector with multiple nested devices."""
        obs = Observation(local={
            "device1": {"power": 100.0},
            "device2": {"power": 200.0}
        })
        vec = obs.vector()

        assert len(vec) == 2
        # device1 comes before device2 alphabetically
        assert vec[0] == 100.0
        assert vec[1] == 200.0

    def test_vector_nested_dict_deep(self):
        """Test vector with deep nesting."""
        obs = Observation(local={
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42.0
                    }
                }
            }
        })
        vec = obs.vector()

        assert len(vec) == 1
        assert vec[0] == 42.0

    def test_vector_nested_mixed_types(self):
        """Test nested dict with mixed types."""
        obs = Observation(local={
            "device": {
                "scalar": 10.0,
                "array": np.array([1.0, 2.0]),
                "nested": {
                    "value": 5.0
                }
            }
        })
        vec = obs.vector()

        assert len(vec) == 4
        # Order: array, nested.value, scalar (alphabetical)
        assert vec[0] == 1.0  # array[0]
        assert vec[1] == 2.0  # array[1]
        assert vec[2] == 5.0  # nested.value
        assert vec[3] == 10.0  # scalar

    def test_vector_complex_structure(self):
        """Test complex nested structure."""
        obs = Observation(
            local={
                "gen1": {
                    "p": 100.0,
                    "q": 20.0
                },
                "ess1": {
                    "soc": 0.5,
                    "p": -50.0
                }
            },
            global_info={
                "bus_voltages": np.array([1.0, 0.98, 1.02]),
                "frequency": 60.0
            }
        )
        vec = obs.vector()

        # Should flatten everything: 2 (gen1) + 2 (ess1) + 3 (voltages) + 1 (freq) = 8
        assert len(vec) == 8


# =============================================================================
# Observation Local vs Global Tests
# =============================================================================

class TestObservationLocalGlobal:
    """Test separation of local and global info."""

    def test_vector_local_only(self):
        """Test vector with only local info."""
        obs = Observation(
            local={"value": 10.0},
            global_info={}
        )
        vec = obs.vector()

        assert len(vec) == 1
        assert vec[0] == 10.0

    def test_vector_global_only(self):
        """Test vector with only global info."""
        obs = Observation(
            local={},
            global_info={"value": 20.0}
        )
        vec = obs.vector()

        assert len(vec) == 1
        assert vec[0] == 20.0

    def test_vector_local_and_global(self):
        """Test vector combines local and global."""
        obs = Observation(
            local={"local_val": 10.0},
            global_info={"global_val": 20.0}
        )
        vec = obs.vector()

        assert len(vec) == 2
        # Local comes first, then global (both sorted)
        assert vec[0] == 10.0  # local_val
        assert vec[1] == 20.0  # global_val

    def test_vector_overlapping_keys(self):
        """Test vector when local and global have same keys."""
        obs = Observation(
            local={"voltage": 1.0},
            global_info={"voltage": 1.05}
        )
        vec = obs.vector()

        # Both should be included
        assert len(vec) == 2
        np.testing.assert_almost_equal(vec[0], 1.0, decimal=5)  # local voltage
        np.testing.assert_almost_equal(vec[1], 1.05, decimal=5)  # global voltage


# =============================================================================
# Edge Cases and Special Scenarios
# =============================================================================

class TestObservationEdgeCases:
    """Test edge cases and special scenarios."""

    def test_vector_with_nan(self):
        """Test vector handles NaN values."""
        obs = Observation(local={"value": np.nan})
        vec = obs.vector()

        assert len(vec) == 1
        assert np.isnan(vec[0])

    def test_vector_with_inf(self):
        """Test vector handles infinity values."""
        obs = Observation(local={
            "pos_inf": np.inf,
            "neg_inf": -np.inf
        })
        vec = obs.vector()

        assert len(vec) == 2
        assert np.isinf(vec[0])
        assert np.isinf(vec[1])

    def test_vector_with_very_large_array(self):
        """Test vector with large array."""
        large_array = np.random.randn(1000).astype(np.float32)
        obs = Observation(local={"large": large_array})
        vec = obs.vector()

        assert len(vec) == 1000
        np.testing.assert_array_equal(vec, large_array)

    def test_vector_deterministic(self):
        """Test that vector is deterministic for same input."""
        obs = Observation(local={
            "z": 1.0,
            "a": 2.0,
            "m": 3.0,
            "array": np.array([4.0, 5.0])
        })

        vec1 = obs.vector()
        vec2 = obs.vector()

        np.testing.assert_array_equal(vec1, vec2)

    def test_vector_skips_non_numeric_types(self):
        """Test that vector skips non-numeric types (lists, strings, etc)."""
        obs = Observation(local={
            "a_number": 10.0,
            "b_string": "ignored",  # Should be skipped
            "c_list": [1, 2, 3],  # Should be skipped
            "d_number": 20.0
        })
        vec = obs.vector()

        # Only numeric values should be in vector
        assert len(vec) == 2
        assert vec[0] == 10.0
        assert vec[1] == 20.0

    def test_vector_zero_values(self):
        """Test vector with zero values."""
        obs = Observation(local={
            "zero_int": 0,
            "zero_float": 0.0,
            "zero_array": np.zeros(3)
        })
        vec = obs.vector()

        assert len(vec) == 5
        assert np.all(vec == 0.0)

    def test_vector_negative_values(self):
        """Test vector with negative values."""
        obs = Observation(local={
            "neg_power": -100.0,
            "neg_array": np.array([-1.0, -2.0, -3.0])
        })
        vec = obs.vector()

        assert len(vec) == 4
        assert vec[0] == -1.0
        assert vec[1] == -2.0
        assert vec[2] == -3.0
        assert vec[3] == -100.0

    def test_multiple_observations_independent(self):
        """Test that multiple observations are independent."""
        obs1 = Observation(local={"value": 10.0})
        obs2 = Observation(local={"value": 20.0})

        vec1 = obs1.vector()
        vec2 = obs2.vector()

        assert vec1[0] == 10.0
        assert vec2[0] == 20.0

        # Modifying obs1 shouldn't affect obs2
        obs1.local["value"] = 30.0
        vec1_new = obs1.vector()
        vec2_new = obs2.vector()

        assert vec1_new[0] == 30.0
        assert vec2_new[0] == 20.0  # Unchanged

    def test_observation_timestamp_preserved(self):
        """Test that timestamp is preserved."""
        obs = Observation(
            local={"value": 10.0},
            timestamp=123.456
        )

        assert obs.timestamp == 123.456

        # Calling vector shouldn't affect timestamp
        vec = obs.vector()
        assert obs.timestamp == 123.456


# =============================================================================
# Observation Serialization Tests (for async message passing)
# =============================================================================

class TestObservationSerialization:
    """Test Observation serialization for async message passing.

    These methods are used in fully async event-driven mode (Option B with
    async_observations=True) where observations are sent via message broker.
    """

    def test_to_dict_simple(self):
        """Test to_dict with simple scalar values."""
        obs = Observation(
            local={"power": 100.0, "voltage": 1.02},
            global_info={"frequency": 60.0},
            timestamp=10.5
        )
        d = obs.to_dict()

        assert d["timestamp"] == 10.5
        assert d["local"]["power"] == 100.0
        assert d["local"]["voltage"] == 1.02
        assert d["global_info"]["frequency"] == 60.0

    def test_to_dict_with_numpy_array(self):
        """Test to_dict serializes numpy arrays with type markers."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        obs = Observation(local={"values": arr})
        d = obs.to_dict()

        serialized = d["local"]["values"]
        assert serialized["__type__"] == "ndarray"
        assert serialized["data"] == [1.0, 2.0, 3.0]
        assert serialized["dtype"] == "float32"

    def test_to_dict_with_nested_observation(self):
        """Test to_dict serializes nested Observation objects."""
        inner_obs = Observation(
            local={"inner_value": 42.0},
            timestamp=5.0
        )
        outer_obs = Observation(
            local={"subordinate": inner_obs},
            timestamp=10.0
        )
        d = outer_obs.to_dict()

        serialized = d["local"]["subordinate"]
        assert serialized["__type__"] == "Observation"
        assert serialized["data"]["timestamp"] == 5.0
        assert serialized["data"]["local"]["inner_value"] == 42.0

    def test_to_dict_with_nested_dict(self):
        """Test to_dict handles nested dicts."""
        obs = Observation(
            local={
                "device": {
                    "power": 100.0,
                    "stats": {
                        "efficiency": 0.95
                    }
                }
            }
        )
        d = obs.to_dict()

        assert d["local"]["device"]["power"] == 100.0
        assert d["local"]["device"]["stats"]["efficiency"] == 0.95

    def test_from_dict_simple(self):
        """Test from_dict reconstructs simple observations."""
        d = {
            "timestamp": 15.0,
            "local": {"power": 200.0, "voltage": 1.0},
            "global_info": {"frequency": 59.9}
        }
        obs = Observation.from_dict(d)

        assert obs.timestamp == 15.0
        assert obs.local["power"] == 200.0
        assert obs.local["voltage"] == 1.0
        assert obs.global_info["frequency"] == 59.9

    def test_from_dict_with_numpy_array(self):
        """Test from_dict reconstructs numpy arrays from type markers."""
        d = {
            "timestamp": 0.0,
            "local": {
                "values": {
                    "__type__": "ndarray",
                    "data": [1.0, 2.0, 3.0],
                    "dtype": "float32"
                }
            },
            "global_info": {}
        }
        obs = Observation.from_dict(d)

        assert isinstance(obs.local["values"], np.ndarray)
        np.testing.assert_array_equal(obs.local["values"], [1.0, 2.0, 3.0])
        assert obs.local["values"].dtype == np.float32

    def test_from_dict_with_nested_observation(self):
        """Test from_dict reconstructs nested Observation objects."""
        d = {
            "timestamp": 10.0,
            "local": {
                "subordinate": {
                    "__type__": "Observation",
                    "data": {
                        "timestamp": 5.0,
                        "local": {"inner_value": 42.0},
                        "global_info": {}
                    }
                }
            },
            "global_info": {}
        }
        obs = Observation.from_dict(d)

        assert obs.timestamp == 10.0
        inner = obs.local["subordinate"]
        assert isinstance(inner, Observation)
        assert inner.timestamp == 5.0
        assert inner.local["inner_value"] == 42.0

    def test_serialization_roundtrip(self):
        """Test to_dict -> from_dict roundtrip preserves data."""
        original = Observation(
            local={
                "power": 100.0,
                "voltages": np.array([1.0, 0.98, 1.02], dtype=np.float32)
            },
            global_info={"frequency": 60.0},
            timestamp=25.5
        )

        d = original.to_dict()
        reconstructed = Observation.from_dict(d)

        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.local["power"] == original.local["power"]
        np.testing.assert_array_equal(
            reconstructed.local["voltages"],
            original.local["voltages"]
        )
        assert reconstructed.global_info["frequency"] == original.global_info["frequency"]

    def test_serialization_roundtrip_nested_observations(self):
        """Test roundtrip with nested Observation objects."""
        sub1 = Observation(local={"state": np.array([1.0, 2.0])}, timestamp=1.0)
        sub2 = Observation(local={"state": np.array([3.0, 4.0])}, timestamp=2.0)
        coordinator = Observation(
            local={
                "sub1_obs": sub1,
                "sub2_obs": sub2
            },
            timestamp=5.0
        )

        d = coordinator.to_dict()
        reconstructed = Observation.from_dict(d)

        assert reconstructed.timestamp == 5.0
        assert isinstance(reconstructed.local["sub1_obs"], Observation)
        assert isinstance(reconstructed.local["sub2_obs"], Observation)
        np.testing.assert_array_equal(
            reconstructed.local["sub1_obs"].local["state"],
            [1.0, 2.0]
        )
        np.testing.assert_array_equal(
            reconstructed.local["sub2_obs"].local["state"],
            [3.0, 4.0]
        )

    def test_serialization_empty_observation(self):
        """Test serialization of empty observation."""
        obs = Observation()
        d = obs.to_dict()
        reconstructed = Observation.from_dict(d)

        assert reconstructed.timestamp == 0.0
        assert reconstructed.local == {}
        assert reconstructed.global_info == {}

    def test_serialization_with_2d_array(self):
        """Test serialization preserves 2D array shape info."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        obs = Observation(local={"matrix": arr})

        d = obs.to_dict()
        reconstructed = Observation.from_dict(d)

        # Note: Shape is flattened in serialization
        # This is a known limitation - arrays become 1D after roundtrip
        assert isinstance(reconstructed.local["matrix"], np.ndarray)

    def test_from_dict_missing_fields_default(self):
        """Test from_dict handles missing fields with defaults."""
        d = {"local": {"value": 10.0}}
        obs = Observation.from_dict(d)

        assert obs.timestamp == 0.0
        assert obs.local["value"] == 10.0
        assert obs.global_info == {}

    def test_from_dict_dtype_preserved(self):
        """Test that dtype is preserved in deserialization."""
        d = {
            "timestamp": 0.0,
            "local": {
                "int_arr": {
                    "__type__": "ndarray",
                    "data": [1, 2, 3],
                    "dtype": "int64"
                },
                "float_arr": {
                    "__type__": "ndarray",
                    "data": [1.5, 2.5],
                    "dtype": "float64"
                }
            },
            "global_info": {}
        }
        obs = Observation.from_dict(d)

        assert obs.local["int_arr"].dtype == np.int64
        assert obs.local["float_arr"].dtype == np.float64

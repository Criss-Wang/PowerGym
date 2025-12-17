"""Comprehensive tests for Observation and Message classes.

Tests cover:
1. Observation initialization and structure
2. Vector flattening (simple and nested dicts)
3. Handling different data types (scalars, arrays, nested dicts)
4. Message structure and attributes
5. Edge cases (empty obs, complex nesting, large arrays)
"""

import pytest
import numpy as np

from powergrid.core.observation import Observation, Message


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
        assert obs.messages == []
        assert obs.timestamp == 0.0

    def test_initialization_with_values(self):
        """Test initialization with provided values."""
        local = {"power": 100.0, "voltage": 1.0}
        global_info = {"grid_freq": 60.0}
        messages = []
        timestamp = 5.5

        obs = Observation(
            local=local,
            global_info=global_info,
            messages=messages,
            timestamp=timestamp
        )

        assert obs.local == local
        assert obs.global_info == global_info
        assert obs.messages == messages
        assert obs.timestamp == 5.5

    def test_initialization_partial(self):
        """Test initialization with partial values."""
        obs = Observation(local={"voltage": 1.0}, timestamp=2.0)

        assert obs.local == {"voltage": 1.0}
        assert obs.global_info == {}
        assert obs.messages == []
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
# Message Tests
# =============================================================================

class TestMessage:
    """Test Message dataclass."""

    def test_message_initialization_basic(self):
        """Test basic message initialization."""
        msg = Message(
            sender="agent_1",
            content={"price": 50.0}
        )

        assert msg.sender == "agent_1"
        assert msg.content == {"price": 50.0}
        assert msg.recipient is None  # Broadcast
        assert msg.timestamp == 0.0

    def test_message_initialization_full(self):
        """Test message with all fields."""
        msg = Message(
            sender="agent_1",
            content={"setpoint": [100.0, 20.0]},
            recipient="agent_2",
            timestamp=5.5
        )

        assert msg.sender == "agent_1"
        assert msg.content == {"setpoint": [100.0, 20.0]}
        assert msg.recipient == "agent_2"
        assert msg.timestamp == 5.5

    def test_message_broadcast(self):
        """Test broadcast message (no recipient)."""
        msg = Message(
            sender="controller",
            content={"price": 60.0}
        )

        assert msg.recipient is None

    def test_message_multiple_recipients(self):
        """Test message with multiple recipients."""
        msg = Message(
            sender="coordinator",
            content={"action": "start"},
            recipient=["agent_1", "agent_2", "agent_3"]
        )

        assert len(msg.recipient) == 3
        assert "agent_1" in msg.recipient

    def test_message_complex_content(self):
        """Test message with complex content."""
        content = {
            "type": "coordination",
            "actions": {
                "gen1": [100.0, 20.0],
                "ess1": [-50.0, 10.0]
            },
            "metadata": {
                "cost": 150.5,
                "timestamp": 1.0
            }
        }

        msg = Message(sender="controller", content=content)

        assert msg.content["type"] == "coordination"
        assert msg.content["actions"]["gen1"] == [100.0, 20.0]
        assert msg.content["metadata"]["cost"] == 150.5

    def test_message_empty_content(self):
        """Test message with empty content."""
        msg = Message(sender="agent", content={})
        assert msg.content == {}


# =============================================================================
# Observation with Messages Tests
# =============================================================================

class TestObservationWithMessages:
    """Test Observation containing messages."""

    def test_observation_with_single_message(self):
        """Test observation containing one message."""
        msg = Message(sender="agent_1", content={"price": 50.0})
        obs = Observation(
            local={"power": 100.0},
            messages=[msg]
        )

        assert len(obs.messages) == 1
        assert obs.messages[0].sender == "agent_1"
        assert obs.messages[0].content["price"] == 50.0

    def test_observation_with_multiple_messages(self):
        """Test observation with multiple messages."""
        msg1 = Message(sender="agent_1", content={"price": 50.0})
        msg2 = Message(sender="agent_2", content={"setpoint": 100.0})
        msg3 = Message(sender="agent_3", content={"constraint": "max_power"})

        obs = Observation(messages=[msg1, msg2, msg3])

        assert len(obs.messages) == 3
        assert obs.messages[0].sender == "agent_1"
        assert obs.messages[1].sender == "agent_2"
        assert obs.messages[2].sender == "agent_3"

    def test_observation_vector_ignores_messages(self):
        """Test that vector() doesn't include message content."""
        msg = Message(sender="agent_1", content={"value": 999.0})
        obs = Observation(
            local={"value": 10.0},
            messages=[msg]
        )

        vec = obs.vector()

        # Should only contain local value, not message content
        assert len(vec) == 1
        assert vec[0] == 10.0


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

"""Tests for heron.utils.array_utils module."""

import pytest
import numpy as np

from heron.utils.array_utils import cat_f32, as_f32, one_hot


class TestCatF32:
    """Test cat_f32 function."""

    def test_concatenate_single_array(self):
        """Test concatenating a single array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = cat_f32([arr])

        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_concatenate_multiple_arrays(self):
        """Test concatenating multiple arrays."""
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([3.0, 4.0])
        arr3 = np.array([5.0])

        result = cat_f32([arr1, arr2, arr3])

        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_concatenate_empty_list(self):
        """Test concatenating empty list returns empty float32 array."""
        result = cat_f32([])

        assert result.dtype == np.float32
        assert len(result) == 0

    def test_concatenate_converts_to_float32(self):
        """Test that integer arrays are converted to float32."""
        arr1 = np.array([1, 2], dtype=np.int32)
        arr2 = np.array([3, 4], dtype=np.float64)

        result = cat_f32([arr1, arr2])

        assert result.dtype == np.float32

    def test_concatenate_preserves_values(self):
        """Test that values are preserved during concatenation."""
        arr1 = np.array([0.123456, 0.789012])
        result = cat_f32([arr1])

        np.testing.assert_array_almost_equal(result, [0.123456, 0.789012], decimal=5)


class TestAsF32:
    """Test as_f32 function."""

    def test_convert_int_to_float32(self):
        """Test converting int to float32."""
        result = as_f32(5)

        assert isinstance(result, np.float32)
        assert result == 5.0

    def test_convert_float_to_float32(self):
        """Test converting float to float32."""
        result = as_f32(3.14)

        assert isinstance(result, np.float32)
        assert abs(result - 3.14) < 1e-6

    def test_convert_negative_value(self):
        """Test converting negative value."""
        result = as_f32(-2.5)

        assert isinstance(result, np.float32)
        assert result == -2.5

    def test_convert_zero(self):
        """Test converting zero."""
        result = as_f32(0)

        assert isinstance(result, np.float32)
        assert result == 0.0


class TestOneHot:
    """Test one_hot function."""

    def test_one_hot_basic(self):
        """Test basic one-hot encoding."""
        result = one_hot(0, 3)

        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

    def test_one_hot_middle_index(self):
        """Test one-hot with middle index."""
        result = one_hot(1, 3)

        np.testing.assert_array_equal(result, [0.0, 1.0, 0.0])

    def test_one_hot_last_index(self):
        """Test one-hot with last index."""
        result = one_hot(2, 3)

        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_one_hot_negative_index_clipped(self):
        """Test that negative index is clipped to 0."""
        result = one_hot(-1, 3)

        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

    def test_one_hot_overflow_index_clipped(self):
        """Test that overflow index is clipped to n-1."""
        result = one_hot(10, 3)

        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_one_hot_single_element(self):
        """Test one-hot with single element."""
        result = one_hot(0, 1)

        np.testing.assert_array_equal(result, [1.0])

    def test_one_hot_empty(self):
        """Test one-hot with n=0 returns empty array."""
        result = one_hot(0, 0)

        assert len(result) == 0
        assert result.dtype == np.float32

    def test_one_hot_large_array(self):
        """Test one-hot with larger array."""
        result = one_hot(5, 10)

        expected = np.zeros(10, dtype=np.float32)
        expected[5] = 1.0
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

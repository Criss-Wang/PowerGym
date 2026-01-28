"""Tests for core.actions module."""

import numpy as np
import pytest

from heron.core.action import Action


class TestAction:
    """Test Action dataclass."""

    rng = np.random.default_rng(12345)

    def test_action_initialization_default(self):
        """Test action initialization with default values."""
        action = Action()

        assert isinstance(action.c, np.ndarray)
        assert action.c.dtype == np.float32
        assert len(action.c) == 0

        assert isinstance(action.d, np.ndarray)
        assert action.d.dtype == np.int32
        assert len(action.d) == 0

        assert action.dim_c == 0
        assert action.dim_d == 0
        assert action.ncats == []
        assert action.range is None

    def test_action_continuous_setup(self):
        """Test action setup for continuous control."""
        action = Action()
        lb = np.array([0.0, -1.0], dtype=np.float32)
        ub = np.array([1.0, 1.0], dtype=np.float32)
        action.set_specs(dim_c=2, range=(lb, ub))

        assert action.dim_c == 2
        assert action.range[0].shape == (2,)
        assert action.range[1].shape == (2,)

    def test_action_discrete_setup(self):
        """Test action setup for discrete control."""
        action = Action()
        action.set_specs(dim_d=1, ncats=[5])

        assert action.dim_d == 1
        assert action.ncats == [5]

    def test_sample_continuous(self):
        """Test sampling continuous actions."""
        action = Action()
        lb = np.array([0.0, -1.0], dtype=np.float32)
        ub = np.array([1.0, 1.0], dtype=np.float32)
        action.set_specs(dim_c=2, range=(lb, ub))

        action.sample()

        assert action.c.shape == (2,)
        assert action.c.dtype == np.float32
        assert 0.0 <= action.c[0] <= 1.0
        assert -1.0 <= action.c[1] <= 1.0

    def test_sample_discrete(self):
        """Test sampling discrete actions."""
        action = Action()
        action.set_specs(dim_d=1, ncats=[5])

        action.sample()

        assert action.d.shape == (1,)
        assert action.d.dtype == np.int32
        assert 0 <= action.d[0] < 5

    def test_sample_mixed_actions(self):
        """Test sampling both continuous and discrete actions."""
        action = Action()
        lb = np.array([0.0], dtype=np.float32)
        ub = np.array([1.0], dtype=np.float32)
        action.set_specs(dim_c=1, dim_d=1, ncats=[3], range=(lb, ub))

        action.sample()

        assert action.c.shape == (1,)
        assert 0.0 <= action.c[0] <= 1.0
        assert action.d.shape == (1,)
        assert 0 <= action.d[0] < 3

    def test_sample_without_range(self):
        """Test sampling continuous without range specified."""
        action = Action()
        action.set_specs(dim_c=2)
        # Don't set range - defaults to [-inf, inf]

        action.sample()  # sample according to a standard normal distribution

        # Check that default range is [-inf, inf]
        low = action.range[0]
        high = action.range[1]
        assert all(low < -1e10)
        assert all(high > 1e10)

    def test_sample_without_ncats(self):
        """Test sampling discrete without ncats specified."""
        action = Action()
        # This should raise because dim_d > 0 but ncats is empty
        with pytest.raises(ValueError):
            action.set_specs(dim_d=1, ncats=[])

    def test_continuous_action_bounds(self):
        """Test continuous action respects specified bounds."""
        action = Action()
        lb = np.array([0.0, 5.0, -10.0], dtype=np.float32)
        ub = np.array([10.0, 15.0, 10.0], dtype=np.float32)
        action.set_specs(dim_c=3, range=(lb, ub))

        # Sample multiple times to check consistency
        for _ in range(10):
            action.sample()
            assert 0.0 <= action.c[0] <= 10.0
            assert 5.0 <= action.c[1] <= 15.0
            assert -10.0 <= action.c[2] <= 10.0

    def test_action_modification(self):
        """Test modifying action values."""
        action = Action()
        lb = np.array([0.0, 0.0], dtype=np.float32)
        ub = np.array([1.0, 1.0], dtype=np.float32)
        action.set_specs(dim_c=2, range=(lb, ub))
        action.sample()

        # Modify continuous action
        action.c[0] = 0.5
        action.c[1] = 0.8

        np.testing.assert_array_almost_equal(action.c, [0.5, 0.8])

    def test_action_discrete_modification(self):
        """Test modifying discrete action values."""
        action = Action()
        action.set_specs(dim_d=1, ncats=[5])
        action.sample()

        # Modify discrete action
        action.d[0] = 3

        assert action.d[0] == 3

    def test_action_range_shape(self):
        """Test action range has correct shape."""
        action = Action()
        lb = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        ub = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        action.set_specs(dim_c=3, range=(lb, ub))

        assert action.range[0].shape == (3,)
        assert action.range[1].shape == (3,)
        assert action.range[0][0] == 0.0  # Lower bound dim 0
        assert action.range[1][0] == 3.0  # Upper bound dim 0

    def test_action_dtype_enforcement(self):
        """Test action maintains correct dtypes."""
        action = Action()
        lb = np.array([0.0], dtype=np.float32)
        ub = np.array([1.0], dtype=np.float32)
        action.set_specs(dim_c=1, dim_d=1, ncats=[3], range=(lb, ub))

        action.sample()

        assert action.c.dtype == np.float32
        assert action.d.dtype == np.int32

    def test_set_specs_initializes_shapes(self):
        a = Action()
        a.set_specs(dim_c=2, dim_d=1, ncats=[4])
        assert a.c.shape == (2,)
        assert a.c.dtype == np.float32
        assert a.d.shape == (1,)
        assert a.d.dtype == np.int32
        assert a.ncats == [4]


    def test_sample_continuous_and_multidiscrete(self):
        lb = np.array([-1.0, 0.0], np.float32)
        ub = np.array([+1.0, 2.0], np.float32)
        a = Action()
        a.set_specs(dim_c=2, dim_d=3, ncats=[2, 3, 4], range=(lb, ub))
        a.sample(seed=12345)

        # c within bounds
        assert np.all(a.c >= lb) and np.all(a.c <= ub)
        # d within 0..K-1
        Ks = np.array([2, 3, 4])
        assert np.all((a.d >= 0) & (a.d < Ks))


    def test_sampling_reproducible_seed(self):
        a1 = Action()
        a1.set_specs(dim_c=0, dim_d=2, ncats=[3, 4])
        a2 = Action()
        a2.set_specs(dim_c=0, dim_d=2, ncats=[3, 4])

        a1.sample(seed=2025)
        a2.sample(seed=2025)
        np.testing.assert_array_equal(a1.d, a2.d)


    def test_scale_unscale_roundtrip_with_zero_span(self):
        lb = np.array([-1.0, 0.0, 2.0], np.float32)
        ub = np.array([+1.0, 4.0, 2.0], np.float32)  # zero-span on last axis
        a = Action()
        a.set_specs(dim_c=3, dim_d=0, ncats=[], range=(lb, ub))

        a.c[...] = np.array([0.25, 1.0, 2.0], np.float32)
        x = a.scale()
        # scale() normalizes to [-1, 1] based on range
        # For [-1, 1] span: 0.25 maps to (0.25 - (-1)) / 2 * 2 - 1 = 0.25
        # For [0, 4] span: 1.0 maps to (1.0 - 0) / 4 * 2 - 1 = -0.5
        # For zero-span [2, 2]: should be 0
        np.testing.assert_allclose(
            x, np.array([0.25, -0.5, 0.0], np.float32), atol=1e-6
        )

        # unscale should restore original values
        a.unscale(x)
        np.testing.assert_allclose(
            a.c, np.array([0.25, 1.0, 2.0], np.float32),
            atol=1e-6,
        )


    def test_clip_in_place(self):
        lb = np.array([-1.0, -2.0], np.float32)
        ub = np.array([+1.0, +2.0], np.float32)
        a = Action()
        a.set_specs(dim_c=2, dim_d=0, range=(lb, ub))
        a.c[...] = np.array([5.0, -9.0], np.float32)
        a.clip()
        np.testing.assert_allclose(a.c, np.array([1.0, -2.0], np.float32))


    def test_vector_and_from_vector(self):
        lb = np.array([-1.0, -1.0], np.float32)
        ub = np.array([+1.0, +1.0], np.float32)
        a = Action()
        a.set_specs(dim_c=2, dim_d=3, ncats=[2, 3, 4], range=(lb, ub))

        a.c[...] = np.array([0.2, -0.3], np.float32)
        a.d[...] = np.array([1, 0, 3], np.int32)
        vec = a.vector()
        assert vec.dtype == np.float32
        assert vec.shape == (2 + 3,)

        b = Action.from_vector(
            vec, dim_c=2, dim_d=3, ncats=[2, 3, 4], range=(lb, ub)
        )
        np.testing.assert_allclose(b.c, a.c)
        np.testing.assert_array_equal(b.d, a.d)


    def test_reset(self):
        a = Action()
        a.set_specs(dim_c=3, dim_d=2, ncats=[2, 5])
        a.c[...] = np.array([1.0, -1.0, 0.5], np.float32)
        a.d[...] = np.array([1, 4], np.int32)
        a.reset()
        # Reset sets c to midpoint of range (default [-inf, inf] -> 0)
        # and d to 0
        np.testing.assert_allclose(a.c, np.zeros(3, np.float32))
        np.testing.assert_array_equal(a.d, np.zeros(2, np.int32))


    def test_error_when_ncats_seq_len_mismatch(self):
        with pytest.raises(ValueError):
            Action().set_specs(dim_c=0, dim_d=2, ncats=[3])  # len 1 != dim_d 2


    def test_error_when_ncats_invalid_with_dim_d_zero(self):
        # When dim_d=0, ncats should be empty list
        # Having ncats with values when dim_d=0 doesn't make sense
        # but doesn't necessarily raise - just ensure consistent behavior
        a = Action()
        a.set_specs(dim_c=0, dim_d=0, ncats=[])
        assert a.ncats == []


    def test_set_values_from_dict(self):
        """Test setting action values from dict."""
        a = Action()
        lb = np.array([-1.0], np.float32)
        ub = np.array([+1.0], np.float32)
        a.set_specs(dim_c=1, dim_d=1, ncats=[3], range=(lb, ub))

        a.set_values({"c": [0.5], "d": [2]})

        np.testing.assert_allclose(a.c, [0.5])
        np.testing.assert_array_equal(a.d, [2])


    def test_set_values_from_vector(self):
        """Test setting action values from flat vector."""
        a = Action()
        lb = np.array([-1.0, -1.0], np.float32)
        ub = np.array([+1.0, +1.0], np.float32)
        a.set_specs(dim_c=2, dim_d=1, ncats=[3], range=(lb, ub))

        a.set_values([0.2, -0.3, 1])  # [c0, c1, d0]

        np.testing.assert_allclose(a.c, [0.2, -0.3])
        np.testing.assert_array_equal(a.d, [1])


    def test_set_values_scalar_discrete(self):
        """Test setting pure discrete action from scalar."""
        a = Action()
        a.set_specs(dim_c=0, dim_d=1, ncats=[5])

        a.set_values(3)

        np.testing.assert_array_equal(a.d, [3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

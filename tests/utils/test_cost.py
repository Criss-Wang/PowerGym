"""Comprehensive tests for cost utility functions.

Tests cover:
1. Quadratic cost function
2. Polynomial cost function
3. Piecewise linear cost
4. Cost curve convenience function
5. Ramping cost
6. Switching cost
7. Tap change cost
8. Energy cost
9. Edge cases
"""

import pytest
import numpy as np

from powergrid.utils.cost import (
    quadratic_cost,
    polynomial_cost,
    piecewise_linear_cost,
    cost_from_curve,
    ramping_cost,
    switching_cost,
    tap_change_cost,
    energy_cost
)


# =============================================================================
# Quadratic Cost Tests
# =============================================================================

class TestQuadraticCost:
    """Test quadratic cost function."""

    def test_quadratic_basic(self):
        """Test basic quadratic cost calculation."""
        # Cost = 0.01*P^2 + 10*P + 50
        cost = quadratic_cost(P=100.0, a=0.01, b=10.0, c=50.0)
        expected = 0.01 * (100**2) + 10 * 100 + 50
        assert cost == expected
        assert cost == 1150.0

    def test_quadratic_zero_power(self):
        """Test quadratic cost at zero power."""
        cost = quadratic_cost(P=0.0, a=0.01, b=10.0, c=50.0)
        assert cost == 50.0  # Only constant term

    def test_quadratic_negative_power(self):
        """Test quadratic with negative power."""
        cost = quadratic_cost(P=-50.0, a=0.01, b=10.0, c=50.0)
        expected = 0.01 * (50**2) - 10 * 50 + 50
        assert cost == expected

    def test_quadratic_no_quadratic_term(self):
        """Test quadratic with a=0 (linear)."""
        cost = quadratic_cost(P=100.0, a=0.0, b=5.0, c=10.0)
        assert cost == 5.0 * 100 + 10
        assert cost == 510.0

    def test_quadratic_no_constant(self):
        """Test quadratic with c=0."""
        cost = quadratic_cost(P=10.0, a=0.1, b=2.0, c=0.0)
        expected = 0.1 * 100 + 2.0 * 10
        assert cost == expected


# =============================================================================
# Polynomial Cost Tests
# =============================================================================

class TestPolynomialCost:
    """Test polynomial cost function."""

    def test_polynomial_constant(self):
        """Test constant polynomial (0-th order)."""
        cost = polynomial_cost(P=100.0, coeffs=[50.0])
        assert cost == 50.0

    def test_polynomial_linear(self):
        """Test linear polynomial."""
        cost = polynomial_cost(P=10.0, coeffs=[5.0, 2.0])  # 5 + 2*P
        assert cost == 5.0 + 2.0 * 10
        assert cost == 25.0

    def test_polynomial_quadratic(self):
        """Test quadratic polynomial."""
        cost = polynomial_cost(P=5.0, coeffs=[1.0, 2.0, 3.0])  # 1 + 2*P + 3*P^2
        expected = 1.0 + 2.0 * 5 + 3.0 * (5**2)
        assert cost == expected
        assert cost == 86.0

    def test_polynomial_higher_order(self):
        """Test higher order polynomial."""
        # P^0 + P^1 + P^2 + P^3
        cost = polynomial_cost(P=2.0, coeffs=[1.0, 1.0, 1.0, 1.0])
        expected = 1 + 2 + 4 + 8
        assert cost == expected
        assert cost == 15.0

    def test_polynomial_zero_power(self):
        """Test polynomial at zero power."""
        cost = polynomial_cost(P=0.0, coeffs=[10.0, 5.0, 3.0])
        assert cost == 10.0  # Only constant term

    def test_polynomial_negative_power(self):
        """Test polynomial with negative power."""
        cost = polynomial_cost(P=-2.0, coeffs=[1.0, 2.0, 1.0])
        expected = 1 + 2 * (-2) + 1 * 4
        assert cost == expected
        assert cost == 1.0


# =============================================================================
# Piecewise Linear Cost Tests
# =============================================================================

class TestPiecewiseLinearCost:
    """Test piecewise linear cost function."""

    def test_piecewise_within_segment(self):
        """Test cost within a segment."""
        # Knots: (0,0), (50,100), (100,250)
        coefs = [0, 0, 50, 100, 100, 250]

        # Test at P=25 (midpoint of first segment)
        cost = piecewise_linear_cost(P=25.0, coefs=coefs)
        # Linear interpolation: 0 + (25-0)*(100-0)/(50-0) = 50
        assert cost == 50.0

    def test_piecewise_at_knot(self):
        """Test cost exactly at a knot."""
        coefs = [0, 0, 50, 100, 100, 250]
        cost = piecewise_linear_cost(P=50.0, coefs=coefs)
        assert cost == 100.0

    def test_piecewise_below_range(self):
        """Test cost below first knot (clamped)."""
        coefs = [10, 50, 20, 100, 30, 150]
        cost = piecewise_linear_cost(P=5.0, coefs=coefs)
        # Should clamp to first segment
        # Extrapolate from (10,50) to (20,100): slope = 5
        expected = 50 + (5 - 10) * (100 - 50) / (20 - 10)
        assert cost == expected

    def test_piecewise_above_range(self):
        """Test cost above last knot (clamped)."""
        coefs = [0, 0, 50, 100, 100, 250]
        cost = piecewise_linear_cost(P=150.0, coefs=coefs)
        # Should clamp to last segment
        # Extrapolate from (50,100) to (100,250): slope = 3
        expected = 100 + (150 - 50) * (250 - 100) / (100 - 50)
        assert cost == expected

    def test_piecewise_multiple_segments(self):
        """Test with many segments."""
        # (0,0), (25,50), (50,150), (75,300), (100,500)
        coefs = [0, 0, 25, 50, 50, 150, 75, 300, 100, 500]

        # Test in third segment: P=60 between (50,150) and (75,300)
        cost = piecewise_linear_cost(P=60.0, coefs=coefs)
        expected = 150 + (60 - 50) * (300 - 150) / (75 - 50)
        assert cost == expected


# =============================================================================
# Cost From Curve Tests
# =============================================================================

class TestCostFromCurve:
    """Test cost_from_curve convenience function."""

    def test_cost_curve_quadratic(self):
        """Test that 3 coefficients triggers quadratic."""
        coefs = [0.01, 10.0, 50.0]
        cost = cost_from_curve(P=100.0, coefs=coefs)
        expected = quadratic_cost(100.0, 0.01, 10.0, 50.0)
        assert cost == expected

    def test_cost_curve_piecewise(self):
        """Test that >3 coefficients triggers piecewise."""
        coefs = [0, 0, 50, 100, 100, 250]
        cost = cost_from_curve(P=25.0, coefs=coefs)
        expected = piecewise_linear_cost(25.0, coefs)
        assert cost == expected

    def test_cost_curve_consistency(self):
        """Test both methods give same result at knots."""
        # Quadratic
        quad_coefs = [0.01, 5.0, 10.0]
        quad_cost = cost_from_curve(P=50.0, coefs=quad_coefs)

        # Piecewise (different approach)
        pw_coefs = [0, 10, 50, 285, 100, 1060]
        pw_cost = cost_from_curve(P=50.0, coefs=pw_coefs)

        # Should match at P=50
        assert pw_cost == 285.0


# =============================================================================
# Ramping Cost Tests
# =============================================================================

class TestRampingCost:
    """Test ramping cost function."""

    def test_ramping_increase(self):
        """Test cost for power increase."""
        cost = ramping_cost(P_prev=50.0, P_curr=80.0, up_cost=2.0, down_cost=1.0)
        # Increase of 30 MW * 2.0 = 60
        assert cost == 60.0

    def test_ramping_decrease(self):
        """Test cost for power decrease."""
        cost = ramping_cost(P_prev=80.0, P_curr=50.0, up_cost=2.0, down_cost=1.5)
        # Decrease of 30 MW * 1.5 = 45
        assert cost == 45.0

    def test_ramping_no_change(self):
        """Test zero cost when no change."""
        cost = ramping_cost(P_prev=50.0, P_curr=50.0, up_cost=2.0, down_cost=1.0)
        assert cost == 0.0

    def test_ramping_zero_costs(self):
        """Test with zero ramping costs."""
        cost = ramping_cost(P_prev=50.0, P_curr=100.0, up_cost=0.0, down_cost=0.0)
        assert cost == 0.0

    def test_ramping_asymmetric_costs(self):
        """Test asymmetric up/down costs."""
        up_cost = ramping_cost(P_prev=50.0, P_curr=70.0, up_cost=5.0, down_cost=1.0)
        down_cost = ramping_cost(P_prev=70.0, P_curr=50.0, up_cost=5.0, down_cost=1.0)

        assert up_cost == 20 * 5.0  # 20 MW increase
        assert down_cost == 20 * 1.0  # 20 MW decrease
        assert up_cost > down_cost


# =============================================================================
# Switching Cost Tests
# =============================================================================

class TestSwitchingCost:
    """Test switching cost function."""

    def test_switching_changed(self):
        """Test cost when change occurred."""
        cost = switching_cost(changed=True, cost_per_change=100.0)
        assert cost == 100.0

    def test_switching_no_change(self):
        """Test zero cost when no change."""
        cost = switching_cost(changed=False, cost_per_change=100.0)
        assert cost == 0.0

    def test_switching_zero_cost(self):
        """Test with zero cost per change."""
        cost = switching_cost(changed=True, cost_per_change=0.0)
        assert cost == 0.0


# =============================================================================
# Tap Change Cost Tests
# =============================================================================

class TestTapChangeCost:
    """Test tap change cost function."""

    def test_tap_change_positive(self):
        """Test cost for positive tap change."""
        cost = tap_change_cost(delta_steps=3, cost_per_step=5.0)
        assert cost == 15.0

    def test_tap_change_negative(self):
        """Test cost for negative tap change (absolute value)."""
        cost = tap_change_cost(delta_steps=-3, cost_per_step=5.0)
        assert cost == 15.0  # Absolute value

    def test_tap_change_zero(self):
        """Test zero cost for no tap change."""
        cost = tap_change_cost(delta_steps=0, cost_per_step=5.0)
        assert cost == 0.0

    def test_tap_change_large_steps(self):
        """Test with large number of steps."""
        cost = tap_change_cost(delta_steps=10, cost_per_step=2.5)
        assert cost == 25.0


# =============================================================================
# Energy Cost Tests
# =============================================================================

class TestEnergyCost:
    """Test energy cost function."""

    def test_energy_cost_import(self):
        """Test cost for importing energy (P > 0)."""
        cost = energy_cost(P_mw=10.0, price_per_mwh=50.0, discount=0.9)
        assert cost == 10.0 * 50.0
        assert cost == 500.0

    def test_energy_cost_export(self):
        """Test credit for exporting energy (P < 0)."""
        cost = energy_cost(P_mw=-10.0, price_per_mwh=50.0, discount=0.9)
        assert cost == -10.0 * 50.0 * 0.9
        assert cost == -450.0  # Negative = credit

    def test_energy_cost_zero(self):
        """Test zero cost at zero power."""
        cost = energy_cost(P_mw=0.0, price_per_mwh=50.0, discount=0.9)
        assert cost == 0.0

    def test_energy_cost_no_discount(self):
        """Test with discount=1.0 (no discount)."""
        cost_import = energy_cost(P_mw=10.0, price_per_mwh=50.0, discount=1.0)
        cost_export = energy_cost(P_mw=-10.0, price_per_mwh=50.0, discount=1.0)

        assert cost_import == 500.0
        assert cost_export == -500.0  # Symmetric

    def test_energy_cost_high_discount(self):
        """Test with low export price (high discount)."""
        cost = energy_cost(P_mw=-10.0, price_per_mwh=50.0, discount=0.5)
        assert cost == -10.0 * 50.0 * 0.5
        assert cost == -250.0


# =============================================================================
# Edge Cases
# =============================================================================

class TestCostEdgeCases:
    """Test edge cases for cost functions."""

    def test_very_large_power(self):
        """Test with very large power values."""
        cost = quadratic_cost(P=10000.0, a=0.001, b=1.0, c=100.0)
        expected = 0.001 * (10000**2) + 10000 + 100
        assert cost == expected

    def test_very_small_power(self):
        """Test with very small power values."""
        cost = quadratic_cost(P=0.001, a=1.0, b=1.0, c=1.0)
        expected = 1.0 * (0.001**2) + 1.0 * 0.001 + 1.0
        np.testing.assert_almost_equal(cost, expected, decimal=10)

    def test_polynomial_empty_coeffs(self):
        """Test polynomial with no coefficients."""
        cost = polynomial_cost(P=100.0, coeffs=[])
        assert cost == 0.0

    def test_piecewise_minimum_knots(self):
        """Test piecewise with minimum 4 values (2 knots)."""
        coefs = [0, 0, 100, 100]
        cost = piecewise_linear_cost(P=50.0, coefs=coefs)
        # Linear between (0,0) and (100,100)
        assert cost == 50.0

    def test_negative_costs(self):
        """Test functions can return negative values (subsidies/credits)."""
        # Polynomial with negative coefficients
        cost = polynomial_cost(P=10.0, coeffs=[-100.0, 5.0])
        assert cost < 0

        # Energy export credit
        cost = energy_cost(P_mw=-50.0, price_per_mwh=100.0, discount=1.0)
        assert cost < 0

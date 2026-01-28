"""Comprehensive tests for safety utility functions.

Tests cover:
1. Apparent power overload (s_over_rating)
2. Power factor penalty
3. Voltage deviation
4. SOC bounds penalty
5. Loading over percentage
6. Rate of change penalty
7. SafetySpec dataclass
8. Total safety composition
9. Edge cases
"""

import pytest
import math

from powergrid.utils.safety import (
    s_over_rating,
    pf_penalty,
    voltage_deviation,
    soc_bounds_penalty,
    loading_over_pct,
    rate_of_change_penalty,
    SafetySpec,
    total_safety
)


# =============================================================================
# Apparent Power Overload Tests
# =============================================================================

class TestApparentPowerOverload:
    """Test s_over_rating function."""

    def test_no_overload(self):
        """Test when S is within rating."""
        # S = sqrt(100^2 + 50^2) = 111.8 < 150
        penalty = s_over_rating(P=100.0, Q=50.0, sn_mva=150.0)
        assert penalty == 0.0

    def test_overload(self):
        """Test when S exceeds rating."""
        # S = sqrt(100^2 + 50^2) = 111.8 > 100
        penalty = s_over_rating(P=100.0, Q=50.0, sn_mva=100.0)
        expected = (111.8 - 100.0) / 100.0
        assert pytest.approx(penalty, rel=0.01) == expected

    def test_none_rating(self):
        """Test with None rating returns 0."""
        penalty = s_over_rating(P=1000.0, Q=500.0, sn_mva=None)
        assert penalty == 0.0

    def test_zero_rating(self):
        """Test with zero rating returns 0."""
        penalty = s_over_rating(P=100.0, Q=50.0, sn_mva=0.0)
        assert penalty == 0.0

    def test_negative_rating(self):
        """Test with negative rating returns 0."""
        penalty = s_over_rating(P=100.0, Q=50.0, sn_mva=-100.0)
        assert penalty == 0.0

    def test_zero_power(self):
        """Test with zero power."""
        penalty = s_over_rating(P=0.0, Q=0.0, sn_mva=100.0)
        assert penalty == 0.0

    def test_reactive_only(self):
        """Test with only reactive power."""
        # S = Q = 120 > 100
        penalty = s_over_rating(P=0.0, Q=120.0, sn_mva=100.0)
        expected = (120.0 - 100.0) / 100.0
        assert penalty == expected
        assert penalty == 0.2


# =============================================================================
# Power Factor Penalty Tests
# =============================================================================

class TestPowerFactorPenalty:
    """Test pf_penalty function."""

    def test_good_power_factor(self):
        """Test when power factor is above minimum."""
        # pf = 100/sqrt(100^2 + 50^2) = 0.894 > 0.8
        penalty = pf_penalty(P=100.0, Q=50.0, min_pf=0.8)
        assert penalty == 0.0

    def test_bad_power_factor(self):
        """Test when power factor is below minimum."""
        # pf = 50/sqrt(50^2 + 100^2) = 0.447 < 0.9
        penalty = pf_penalty(P=50.0, Q=100.0, min_pf=0.9)
        expected = 0.9 - 0.447
        assert pytest.approx(penalty, rel=0.01) == expected

    def test_none_min_pf(self):
        """Test with None min_pf returns 0."""
        penalty = pf_penalty(P=10.0, Q=100.0, min_pf=None)
        assert penalty == 0.0

    def test_zero_power(self):
        """Test with zero power returns 0."""
        penalty = pf_penalty(P=0.0, Q=0.0, min_pf=0.95)
        assert penalty == 0.0

    def test_unity_power_factor(self):
        """Test with unity power factor (Q=0)."""
        penalty = pf_penalty(P=100.0, Q=0.0, min_pf=0.95)
        # pf = 1.0 >= 0.95
        assert penalty == 0.0

    def test_negative_power(self):
        """Test with negative active power."""
        # Should use abs(P/S) for pf
        penalty = pf_penalty(P=-100.0, Q=50.0, min_pf=0.8)
        # pf = |-100/111.8| = 0.894 > 0.8
        assert penalty == 0.0


# =============================================================================
# Voltage Deviation Tests
# =============================================================================

class TestVoltageDeviation:
    """Test voltage_deviation function."""

    def test_voltage_in_range(self):
        """Test when voltage is within limits."""
        penalty = voltage_deviation(V_pu=1.0, vmin_pu=0.95, vmax_pu=1.05)
        assert penalty == 0.0

    def test_voltage_too_low(self):
        """Test when voltage is below minimum."""
        penalty = voltage_deviation(V_pu=0.90, vmin_pu=0.95, vmax_pu=1.05)
        expected = (0.95 - 0.90) / 0.95
        assert pytest.approx(penalty, rel=0.01) == expected

    def test_voltage_too_high(self):
        """Test when voltage is above maximum."""
        penalty = voltage_deviation(V_pu=1.10, vmin_pu=0.95, vmax_pu=1.05)
        expected = (1.10 - 1.05) / (1.5 - 1.05)
        assert pytest.approx(penalty, rel=0.01) == expected

    def test_voltage_at_min_limit(self):
        """Test voltage exactly at minimum limit."""
        penalty = voltage_deviation(V_pu=0.95, vmin_pu=0.95, vmax_pu=1.05)
        assert penalty == 0.0

    def test_voltage_at_max_limit(self):
        """Test voltage exactly at maximum limit."""
        penalty = voltage_deviation(V_pu=1.05, vmin_pu=0.95, vmax_pu=1.05)
        assert penalty == 0.0

    def test_voltage_severely_low(self):
        """Test with very low voltage."""
        penalty = voltage_deviation(V_pu=0.70, vmin_pu=0.95, vmax_pu=1.05)
        expected = (0.95 - 0.70) / 0.95
        assert pytest.approx(penalty, rel=0.01) == expected
        assert penalty > 0


# =============================================================================
# SOC Bounds Penalty Tests
# =============================================================================

class TestSOCBoundsPenalty:
    """Test soc_bounds_penalty function."""

    def test_soc_in_range(self):
        """Test when SOC is within bounds."""
        penalty = soc_bounds_penalty(soc=0.5, min_soc=0.2, max_soc=0.8)
        assert penalty == 0.0

    def test_soc_too_high(self):
        """Test when SOC exceeds maximum."""
        penalty = soc_bounds_penalty(soc=0.9, min_soc=0.2, max_soc=0.8)
        assert penalty == pytest.approx(0.9 - 0.8)
        assert penalty == pytest.approx(0.1)

    def test_soc_too_low(self):
        """Test when SOC is below minimum."""
        penalty = soc_bounds_penalty(soc=0.15, min_soc=0.2, max_soc=0.8)
        assert penalty == pytest.approx(0.2 - 0.15)
        assert penalty == pytest.approx(0.05)

    def test_soc_at_min_bound(self):
        """Test SOC exactly at minimum."""
        penalty = soc_bounds_penalty(soc=0.2, min_soc=0.2, max_soc=0.8)
        assert penalty == 0.0

    def test_soc_at_max_bound(self):
        """Test SOC exactly at maximum."""
        penalty = soc_bounds_penalty(soc=0.8, min_soc=0.2, max_soc=0.8)
        assert penalty == 0.0

    def test_soc_extreme_violation(self):
        """Test with extreme SOC violation."""
        penalty = soc_bounds_penalty(soc=1.5, min_soc=0.1, max_soc=0.9)
        assert penalty == 1.5 - 0.9
        assert penalty == 0.6


# =============================================================================
# Loading Over Percentage Tests
# =============================================================================

class TestLoadingOverPercentage:
    """Test loading_over_pct function."""

    def test_loading_under_100(self):
        """Test when loading is under 100%."""
        penalty = loading_over_pct(loading_pct=80.0)
        assert penalty == 0.0

    def test_loading_exactly_100(self):
        """Test when loading is exactly 100%."""
        penalty = loading_over_pct(loading_pct=100.0)
        assert penalty == 0.0

    def test_loading_over_100(self):
        """Test when loading exceeds 100%."""
        penalty = loading_over_pct(loading_pct=120.0)
        assert penalty == 0.2  # (120-100)/100

    def test_loading_severe_overload(self):
        """Test with severe overloading."""
        penalty = loading_over_pct(loading_pct=250.0)
        assert penalty == 1.5  # (250-100)/100

    def test_loading_zero(self):
        """Test with zero loading."""
        penalty = loading_over_pct(loading_pct=0.0)
        assert penalty == 0.0


# =============================================================================
# Rate of Change Penalty Tests
# =============================================================================

class TestRateOfChangePenalty:
    """Test rate_of_change_penalty function."""

    def test_roc_within_limit(self):
        """Test when rate of change is within limit."""
        penalty = rate_of_change_penalty(prev=50.0, curr=60.0, limit=15.0)
        # delta = 10 < 15
        assert penalty == 0.0

    def test_roc_exceeds_limit(self):
        """Test when rate of change exceeds limit."""
        penalty = rate_of_change_penalty(prev=50.0, curr=80.0, limit=20.0)
        # delta = 30 > 20
        expected = (30.0 - 20.0) / 20.0
        assert penalty == expected
        assert penalty == 0.5

    def test_roc_zero_limit(self):
        """Test with zero limit returns 0."""
        penalty = rate_of_change_penalty(prev=50.0, curr=100.0, limit=0.0)
        assert penalty == 0.0

    def test_roc_negative_limit(self):
        """Test with negative limit returns 0."""
        penalty = rate_of_change_penalty(prev=50.0, curr=100.0, limit=-10.0)
        assert penalty == 0.0

    def test_roc_decrease(self):
        """Test rate of change with decrease."""
        penalty = rate_of_change_penalty(prev=100.0, curr=50.0, limit=30.0)
        # delta = 50 > 30
        expected = (50.0 - 30.0) / 30.0
        assert pytest.approx(penalty, rel=0.01) == expected

    def test_roc_no_change(self):
        """Test with no change."""
        penalty = rate_of_change_penalty(prev=50.0, curr=50.0, limit=10.0)
        assert penalty == 0.0


# =============================================================================
# SafetySpec Tests
# =============================================================================

class TestSafetySpec:
    """Test SafetySpec dataclass."""

    def test_safetyspec_defaults(self):
        """Test SafetySpec default weights."""
        spec = SafetySpec()

        assert spec.s_over_rating_w == 1.0
        assert spec.pf_w == 1.0
        assert spec.voltage_w == 0.0  # Disabled by default
        assert spec.soc_w == 1.0
        assert spec.loading_w == 1.0
        assert spec.roc_w == 0.0  # Disabled by default

    def test_safetyspec_custom(self):
        """Test SafetySpec with custom weights."""
        spec = SafetySpec(
            s_over_rating_w=2.0,
            pf_w=0.0,
            voltage_w=1.5,
            soc_w=1.0,
            loading_w=0.5,
            roc_w=0.0
        )

        assert spec.s_over_rating_w == 2.0
        assert spec.pf_w == 0.0
        assert spec.voltage_w == 1.5


# =============================================================================
# Total Safety Composition Tests
# =============================================================================

class TestTotalSafety:
    """Test total_safety composition function."""

    def test_total_safety_single_term(self):
        """Test with only one term enabled."""
        spec = SafetySpec(
            s_over_rating_w=1.0,
            pf_w=0.0,
            voltage_w=0.0,
            soc_w=0.0,
            loading_w=0.0,
            roc_w=0.0
        )

        # S overload only
        safety = total_safety(
            spec=spec,
            P=100.0,
            Q=50.0,
            sn_mva=100.0
        )

        # S = 111.8, overload = 11.8/100 = 0.118
        assert safety > 0
        assert pytest.approx(safety, rel=0.01) == 0.118

    def test_total_safety_multiple_terms(self):
        """Test with multiple terms enabled."""
        spec = SafetySpec(
            s_over_rating_w=1.0,
            pf_w=0.0,
            voltage_w=0.0,
            soc_w=1.0,
            loading_w=0.0,
            roc_w=0.0
        )

        safety = total_safety(
            spec=spec,
            P=100.0,
            Q=50.0,
            sn_mva=100.0,
            soc=0.95,
            min_soc=0.1,
            max_soc=0.9
        )

        # Should include both S overload and SOC penalty
        s_penalty = s_over_rating(100.0, 50.0, 100.0)
        soc_penalty = soc_bounds_penalty(0.95, 0.1, 0.9)
        expected = s_penalty + soc_penalty

        assert pytest.approx(safety, rel=0.01) == expected

    def test_total_safety_all_terms(self):
        """Test with all terms enabled."""
        spec = SafetySpec(
            s_over_rating_w=1.0,
            pf_w=1.0,
            voltage_w=1.0,
            soc_w=1.0,
            loading_w=1.0,
            roc_w=1.0
        )

        safety = total_safety(
            spec=spec,
            P=100.0,
            Q=50.0,
            sn_mva=100.0,
            min_pf=0.95,
            V_pu=1.08,
            vmin_pu=0.95,
            vmax_pu=1.05,
            soc=0.92,
            min_soc=0.1,
            max_soc=0.9,
            loading_pct=110.0,
            prev=50.0,
            curr=90.0,
            limit=30.0
        )

        # Should be sum of all penalties (all have violations)
        assert safety > 0

    def test_total_safety_weighted(self):
        """Test that weights are applied correctly."""
        spec = SafetySpec(
            s_over_rating_w=2.0,  # Double weight
            pf_w=0.0,
            voltage_w=0.0,
            soc_w=0.0,
            loading_w=0.0,
            roc_w=0.0
        )

        safety = total_safety(
            spec=spec,
            P=100.0,
            Q=50.0,
            sn_mva=100.0
        )

        # Should be 2x the base penalty
        base_penalty = s_over_rating(100.0, 50.0, 100.0)
        expected = 2.0 * base_penalty

        assert pytest.approx(safety, rel=0.01) == expected

    def test_total_safety_all_ok(self):
        """Test when all constraints are satisfied."""
        spec = SafetySpec(
            s_over_rating_w=1.0,
            pf_w=1.0,
            voltage_w=1.0,
            soc_w=1.0,
            loading_w=1.0,
            roc_w=1.0
        )

        safety = total_safety(
            spec=spec,
            P=50.0,
            Q=25.0,
            sn_mva=100.0,  # S = 55.9 < 100
            min_pf=0.8,  # pf = 0.894 > 0.8
            V_pu=1.0,  # In range [0.95, 1.05]
            vmin_pu=0.95,
            vmax_pu=1.05,
            soc=0.5,  # In range [0.2, 0.8]
            min_soc=0.2,
            max_soc=0.8,
            loading_pct=80.0,  # < 100%
            prev=50.0,
            curr=55.0,  # delta = 5 < 10
            limit=10.0
        )

        assert safety == 0.0

    def test_total_safety_optional_terms_none(self):
        """Test with optional terms set to None."""
        spec = SafetySpec(
            s_over_rating_w=1.0,
            pf_w=0.0,
            voltage_w=1.0,  # Enabled but V_pu is None
            soc_w=1.0,  # Enabled but soc is None
            loading_w=0.0,
            roc_w=0.0
        )

        # V_pu and soc are None - should skip those terms
        safety = total_safety(
            spec=spec,
            P=100.0,
            Q=50.0,
            sn_mva=100.0,
            V_pu=None,  # Should be skipped
            soc=None  # Should be skipped
        )

        # Should only include S overload
        expected = s_over_rating(100.0, 50.0, 100.0)
        assert pytest.approx(safety, rel=0.01) == expected


# =============================================================================
# Edge Cases
# =============================================================================

class TestSafetyEdgeCases:
    """Test edge cases for safety functions."""

    def test_zero_values(self):
        """Test with all zero values."""
        spec = SafetySpec()
        safety = total_safety(
            spec=spec,
            P=0.0,
            Q=0.0,
            sn_mva=100.0,
            min_pf=0.9,
            V_pu=1.0,
            soc=0.5,
            loading_pct=0.0,
            prev=0.0,
            curr=0.0,
            limit=10.0
        )

        assert safety == 0.0

    def test_extreme_violations(self):
        """Test with extreme violations."""
        spec = SafetySpec(
            s_over_rating_w=1.0,
            pf_w=1.0,
            voltage_w=1.0,
            soc_w=1.0,
            loading_w=1.0,
            roc_w=1.0
        )

        safety = total_safety(
            spec=spec,
            P=1000.0,
            Q=500.0,
            sn_mva=10.0,  # Huge overload
            min_pf=0.99,  # Impossible pf
            V_pu=1.5,  # Way over
            vmin_pu=0.95,
            vmax_pu=1.05,
            soc=2.0,  # Over 100%!
            min_soc=0.1,
            max_soc=0.9,
            loading_pct=500.0,  # 5x overload
            prev=0.0,
            curr=1000.0,  # Huge change
            limit=10.0
        )

        # Should have very large safety penalty
        assert safety > 10.0

    def test_nan_handling(self):
        """Test that NaN in sn_mva returns 0."""
        penalty = s_over_rating(P=100.0, Q=50.0, sn_mva=float('nan'))
        assert penalty == 0.0

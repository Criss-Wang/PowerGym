"""Tests for device feature providers."""

import pytest
import numpy as np

from case_studies.grid_age.features import (
    ESSFeature,
    DGFeature,
    RESFeature,
    GridFeature,
    NetworkFeature,
)


class TestESSFeature:
    """Tests for Energy Storage System feature."""

    def test_initialization(self):
        """Test ESS feature initialization."""
        ess = ESSFeature(capacity=2.0, soc=0.5)
        assert ess.capacity == 2.0
        assert ess.soc == 0.5
        assert ess.P == 0.0
        assert ess.Q == 0.0

    def test_set_values(self):
        """Test setting ESS values with constraints."""
        ess = ESSFeature(capacity=2.0, min_p=-0.5, max_p=0.5)

        # Test power limits
        ess.set_values(P=0.7)
        assert ess.P == 0.5  # Clipped to max_p

        ess.set_values(P=-0.7)
        assert ess.P == -0.5  # Clipped to min_p

        # Test SOC limits
        ess.set_values(soc=1.5)
        assert ess.soc == 1.0  # Clipped to 1.0

        ess.set_values(soc=-0.1)
        assert ess.soc == 0.0  # Clipped to 0.0

    def test_update_soc_charging(self):
        """Test SOC update during charging."""
        ess = ESSFeature(capacity=2.0, soc=0.5, ch_eff=0.95)
        ess.set_values(P=0.5)  # 0.5 MW charging
        ess.update_soc(dt=1.0)  # 1 hour

        # Expected: soc += 0.5 * 0.95 * 1.0 / 2.0 = 0.2375
        expected_soc = 0.5 + 0.2375
        assert abs(ess.soc - expected_soc) < 1e-6

    def test_update_soc_discharging(self):
        """Test SOC update during discharging."""
        ess = ESSFeature(capacity=2.0, soc=0.5, dsc_eff=0.95)
        ess.set_values(P=-0.5)  # 0.5 MW discharging
        ess.update_soc(dt=1.0)

        # Expected: soc += -0.5 / 0.95 * 1.0 / 2.0 = -0.2632
        expected_soc = 0.5 - 0.2632
        assert abs(ess.soc - expected_soc) < 1e-3

    def test_get_feasible_power_range(self):
        """Test feasible power range based on SOC."""
        ess = ESSFeature(
            capacity=2.0,
            min_p=-0.5,
            max_p=0.5,
            min_soc=0.2,
            max_soc=0.9,
            soc=0.5
        )

        min_p, max_p = ess.get_feasible_power_range()

        # Should be limited by both physical limits and SOC bounds
        assert min_p <= 0  # Discharge
        assert max_p >= 0  # Charge


class TestDGFeature:
    """Tests for Distributed Generator feature."""

    def test_initialization(self):
        """Test DG feature initialization."""
        dg = DGFeature(max_p=0.66, min_p=0.1)
        assert dg.max_p == 0.66
        assert dg.min_p == 0.1
        assert dg.on == 1
        assert dg.P == 0.0

    def test_set_values_on(self):
        """Test setting DG values when on."""
        dg = DGFeature(max_p=0.66, min_p=0.1, on=1)

        # Power should be clipped to [min_p, max_p]
        dg.set_values(P=0.8)
        assert dg.P == 0.66

        dg.set_values(P=0.05)
        assert dg.P == 0.1

    def test_set_values_off(self):
        """Test setting DG values when off."""
        dg = DGFeature(max_p=0.66, min_p=0.1, on=0)

        # Power should be 0 when off
        dg.set_values(P=0.5)
        assert dg.P == 0.0

    def test_compute_fuel_cost(self):
        """Test fuel cost computation."""
        dg = DGFeature(
            max_p=0.66,
            on=1,
            fuel_cost_a=10.0,
            fuel_cost_b=5.0,
            fuel_cost_c=1.0
        )
        dg.set_values(P=0.5)

        # Cost = (10 * 0.5^2 + 5 * 0.5 + 1) * dt = 6.0 * 1.0 = 6.0
        cost = dg.compute_fuel_cost(dt=1.0)
        expected_cost = 10.0 * 0.5**2 + 5.0 * 0.5 + 1.0
        assert abs(cost - expected_cost) < 1e-6


class TestRESFeature:
    """Tests for Renewable Energy Source feature."""

    def test_initialization(self):
        """Test RES feature initialization."""
        res = RESFeature(max_p=0.1)
        assert res.max_p == 0.1
        assert res.availability == 1.0

    def test_set_availability(self):
        """Test setting availability."""
        res = RESFeature(max_p=0.1)
        res.set_values(P=0.1)

        res.set_availability(0.5)
        assert res.availability == 0.5
        # Power should be capped at max_p * availability = 0.05
        assert res.P == 0.05

    def test_set_values_with_availability(self):
        """Test setting power with availability constraint."""
        res = RESFeature(max_p=0.1, availability=0.5)

        # Power should be clipped to max_p * availability
        res.set_values(P=0.1)
        assert res.P == 0.05


class TestGridFeature:
    """Tests for Grid connection feature."""

    def test_initialization(self):
        """Test Grid feature initialization."""
        grid = GridFeature()
        assert grid.P == 0.0
        assert grid.Q == 0.0
        assert grid.price == 50.0

    def test_compute_energy_cost_buying(self):
        """Test energy cost when buying from grid."""
        grid = GridFeature(price=50.0)
        grid.set_values(P=1.0)  # Buying 1 MW

        # Cost = 1.0 * 50.0 * 1.0 = 50.0
        cost = grid.compute_energy_cost(dt=1.0)
        assert cost == 50.0

    def test_compute_energy_cost_selling(self):
        """Test energy cost when selling to grid."""
        grid = GridFeature(price=50.0, sell_discount=0.9)
        grid.set_values(P=-1.0)  # Selling 1 MW

        # Cost = -1.0 * 50.0 * 0.9 * 1.0 = -45.0 (revenue)
        cost = grid.compute_energy_cost(dt=1.0)
        assert cost == -45.0

    def test_set_price(self):
        """Test setting price."""
        grid = GridFeature()
        grid.set_price(75.0)
        assert grid.price == 75.0

        # Negative price should be clamped to 0
        grid.set_price(-10.0)
        assert grid.price == 0.0


class TestNetworkFeature:
    """Tests for Network state feature."""

    def test_initialization(self):
        """Test Network feature initialization."""
        net = NetworkFeature()
        assert net.voltage_min == 1.0
        assert net.voltage_max == 1.0
        assert net.voltage_violations == 0
        assert net.overload_violations == 0

    def test_set_values(self):
        """Test setting network values."""
        net = NetworkFeature()
        net.set_values(
            voltage_min=0.95,
            voltage_max=1.05,
            max_line_loading=0.8,
            voltage_violations=2,
            overload_violations=1
        )

        assert net.voltage_min == 0.95
        assert net.voltage_max == 1.05
        assert net.max_line_loading == 0.8
        assert net.voltage_violations == 2
        assert net.overload_violations == 1

    def test_compute_safety_penalty_no_violations(self):
        """Test safety penalty with no violations."""
        net = NetworkFeature()
        net.set_values(
            voltage_min=0.98,
            voltage_max=1.02,
            max_line_loading=0.8
        )

        penalty = net.compute_safety_penalty(
            voltage_limit=0.05,
            loading_limit=1.0
        )
        assert penalty == 0.0

    def test_compute_safety_penalty_voltage_low(self):
        """Test safety penalty with low voltage."""
        net = NetworkFeature()
        net.set_values(voltage_min=0.90, voltage_max=1.00)

        penalty = net.compute_safety_penalty(voltage_limit=0.05)
        # Penalty = (0.95 - 0.90) * 10 = 0.5
        assert abs(penalty - 0.5) < 1e-6

    def test_compute_safety_penalty_voltage_high(self):
        """Test safety penalty with high voltage."""
        net = NetworkFeature()
        net.set_values(voltage_min=1.00, voltage_max=1.10)

        penalty = net.compute_safety_penalty(voltage_limit=0.05)
        # Penalty = (1.10 - 1.05) * 10 = 0.5
        assert abs(penalty - 0.5) < 1e-6

    def test_compute_safety_penalty_overload(self):
        """Test safety penalty with line overload."""
        net = NetworkFeature()
        net.set_values(max_line_loading=1.2)

        penalty = net.compute_safety_penalty(loading_limit=1.0)
        # Penalty = (1.2 - 1.0) * 20 = 4.0
        assert abs(penalty - 4.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

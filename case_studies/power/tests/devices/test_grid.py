"""
Unit tests for Grid device.

Tests cover:
- Initialization
- State tracking (P, Q, price)
- Cost computation (energy purchase/sale)
- Reset functionality
- Price updates

Note: Grid device has a simplified implementation as it represents
the external grid connection (not a controllable asset).
"""

import pytest
import numpy as np

# Skip entire module as Grid device is not yet implemented
pytest.importorskip("powergrid.agents.grid", reason="Grid device not yet implemented")

from powergrid.agents.grid import Grid


class TestGridInitialization:
    """Test Grid device initialization."""

    def test_basic_initialization(self):
        """Test basic grid creation."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
            sell_discount=0.8,
            dt=1.0,
        )

        assert grid.name == "grid1"
        assert grid.agent_id == "grid1"
        assert grid.bus == 650
        assert grid.sn_mva == 5.0
        assert grid.sell_discount == 0.8
        assert grid.action_callback == True  # Grid doesn't have direct control

    def test_device_level(self):
        """Test that Grid is a device-level agent."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
        )

        assert grid.level == 1  # Device level


class TestGridStateTracking:
    """Test grid state P, Q, price tracking."""

    def test_initial_state(self):
        """Test initial state after creation."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
        )

        grid.reset_agent()

        # Check initial values
        assert grid.state.P == 0.0
        assert grid.state.Q == 0.0
        assert grid.state.price == 0.0

    def test_update_power(self):
        """Test updating P and Q."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
        )

        grid.reset_agent()

        # Update P and Q
        grid.update_state(P=2.5, Q=1.0)

        assert grid.state.P == 2.5
        assert grid.state.Q == 1.0

    def test_update_price(self):
        """Test updating electricity price."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
        )

        grid.reset_agent()

        # Update price
        grid.update_state(price=50.0)

        assert grid.state.price == 50.0

    def test_partial_updates(self):
        """Test that None values don't overwrite existing state."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
        )

        grid.reset_agent()

        # Set initial values
        grid.update_state(P=2.0, Q=1.0, price=40.0)

        # Update only price
        grid.update_state(price=50.0)

        assert grid.state.P == 2.0  # Unchanged
        assert grid.state.Q == 1.0  # Unchanged
        assert grid.state.price == 50.0  # Updated


class TestGridCost:
    """Test grid cost computation (energy purchase/sale)."""

    def test_cost_buying_power(self):
        """Test cost when buying power (P > 0)."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
            sell_discount=0.8,
            dt=1.0,
        )

        grid.reset_agent()

        # Buy 3 MW at $50/MWh for 1 hour
        grid.update_state(P=3.0, price=50.0)
        grid.update_cost_safety()

        # Expected cost: 3 MW * $50/MWh * 1 hr = $150
        expected_cost = 3.0 * 50.0 * 1.0
        assert np.isclose(grid.cost, expected_cost)

    def test_revenue_selling_power(self):
        """Test revenue when selling power (P < 0)."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
            sell_discount=0.8,
            dt=1.0,
        )

        grid.reset_agent()

        # Sell 2 MW at $50/MWh for 1 hour
        # Revenue = 2 * 50 * 0.8 (sell discount) * 1 hr = $80
        # Cost is negative (revenue)
        grid.update_state(P=-2.0, price=50.0)
        grid.update_cost_safety()

        # Expected revenue (negative cost): -2 * 50 * 0.8 * 1 = -80
        expected_cost = -2.0 * 50.0 * 0.8 * 1.0
        assert np.isclose(grid.cost, expected_cost)

    def test_zero_cost_no_power(self):
        """Test zero cost when P=0."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
            dt=1.0,
        )

        grid.reset_agent()

        grid.update_state(P=0.0, price=50.0)
        grid.update_cost_safety()

        assert grid.cost == 0.0

    def test_cost_over_time(self):
        """Test cost accumulation over multiple timesteps."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
            dt=1.0,
        )

        grid.reset_agent()

        total_cost = 0.0

        # Hour 1: Buy 2 MW at $40/MWh
        grid.update_state(P=2.0, price=40.0)
        grid.update_cost_safety()
        total_cost += grid.cost
        assert np.isclose(total_cost, 80.0)

        # Hour 2: Buy 3 MW at $50/MWh
        grid.update_state(P=3.0, price=50.0)
        grid.update_cost_safety()
        total_cost += grid.cost
        assert np.isclose(total_cost, 230.0)  # 80 + 150

    def test_sell_discount_effect(self):
        """Test that sell discount reduces revenue."""
        grid1 = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
            sell_discount=1.0,  # 100% (full price)
            dt=1.0,
        )

        grid2 = Grid(
            name="grid2",
            bus=650,
            sn_mva=5.0,
            sell_discount=0.8,  # 80% (discounted)
            dt=1.0,
        )

        # Both sell same amount at same price
        grid1.update_state(P=-2.0, price=50.0)
        grid1.update_cost_safety()

        grid2.update_state(P=-2.0, price=50.0)
        grid2.update_cost_safety()

        # Grid1 should have higher revenue (more negative cost)
        assert grid1.cost < grid2.cost
        assert np.isclose(grid1.cost, -100.0)  # 2 * 50 * 1.0
        assert np.isclose(grid2.cost, -80.0)   # 2 * 50 * 0.8


class TestGridSafety:
    """Test grid safety (always zero for Grid device)."""

    def test_safety_always_zero(self):
        """Test that Grid device has no safety violations."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
        )

        grid.reset_agent()

        # Even with high power
        grid.update_state(P=10.0, Q=5.0)
        grid.update_cost_safety()

        assert grid.safety == 0.0  # Grid has no local safety constraints


class TestGridReset:
    """Test grid reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
        )

        # Set some values
        grid.update_state(P=5.0, Q=2.0, price=60.0)
        grid.update_cost_safety()

        assert grid.state.P != 0.0
        assert grid.cost != 0.0

        # Reset
        grid.reset_agent()

        # Should be cleared
        assert grid.state.P == 0.0
        assert grid.state.Q == 0.0
        assert grid.state.price == 0.0
        assert grid.cost == 0.0
        assert grid.safety == 0.0


class TestGridConventions:
    """Test Grid device conventions."""

    def test_positive_p_is_buying(self):
        """Test convention: P > 0 means buying from grid."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
            dt=1.0,
        )

        grid.reset_agent()
        grid.update_state(P=2.0, price=50.0)
        grid.update_cost_safety()

        # Positive cost = spending money (buying)
        assert grid.cost > 0.0

    def test_negative_p_is_selling(self):
        """Test convention: P < 0 means selling to grid."""
        grid = Grid(
            name="grid1",
            bus=650,
            sn_mva=5.0,
            dt=1.0,
        )

        grid.reset_agent()
        grid.update_state(P=-2.0, price=50.0)
        grid.update_cost_safety()

        # Negative cost = earning money (selling)
        assert grid.cost < 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

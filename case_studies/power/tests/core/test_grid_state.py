"""Tests for GridState (CoordinatorAgentState) with network features."""

import numpy as np
import pytest

from heron.core.state import GridState, CoordinatorAgentState
from powergrid.features.network import BusVoltages, LineFlows, NetworkMetrics


def test_grid_state_alias():
    """Test that GridState is an alias for CoordinatorAgentState."""
    assert GridState is CoordinatorAgentState


def test_grid_state_empty():
    """Test GridState with no features."""
    gs = GridState(owner_id="grid1", owner_level=2)
    vec = gs.vector()

    assert vec.shape == (0,)
    assert len(gs.features) == 0


def test_grid_state_bus_voltages():
    """Test GridState with BusVoltages feature."""
    bus_voltages = BusVoltages(
        vm_pu=np.array([1.0, 1.02, 0.98]),
        va_deg=np.array([0.0, -5.0, -10.0]),
        bus_names=["Bus1", "Bus2", "Bus3"],
    )

    gs = GridState(
        owner_id="grid1",
        owner_level=2,
        features=[bus_voltages]
    )
    vec = gs.vector()
    names = bus_voltages.names()

    # 3 buses * 2 features (vm, va) = 6 values
    assert vec.shape == (6,)
    assert len(names) == 6
    assert names[0] == "vm_pu_Bus1"
    assert names[3] == "va_deg_Bus1"


def test_grid_state_line_flows():
    """Test GridState with LineFlows feature."""
    line_flows = LineFlows(
        p_from_mw=np.array([10.0, 20.0]),
        q_from_mvar=np.array([5.0, 8.0]),
        loading_percent=np.array([45.0, 67.0]),
        line_names=["Line1", "Line2"],
    )

    gs = GridState(
        owner_id="grid1",
        owner_level=2,
        features=[line_flows]
    )
    vec = gs.vector()
    names = line_flows.names()

    # 2 lines * 3 features (p, q, loading) = 6 values
    assert vec.shape == (6,)
    assert len(names) == 6
    assert names[0] == "p_from_mw_Line1"
    assert names[2] == "q_from_mvar_Line1"


def test_grid_state_network_metrics():
    """Test GridState with NetworkMetrics feature."""
    metrics = NetworkMetrics(
        total_gen_mw=100.0,
        total_load_mw=95.0,
        total_loss_mw=5.0,
        total_gen_mvar=30.0,
        total_load_mvar=28.0,
    )

    gs = GridState(
        owner_id="grid1",
        owner_level=2,
        features=[metrics]
    )
    vec = gs.vector()
    names = metrics.names()

    assert vec.shape == (5,)
    assert len(names) == 5
    assert np.allclose(vec, [100.0, 95.0, 5.0, 30.0, 28.0])


def test_grid_state_combined():
    """Test GridState with multiple features."""
    bus_voltages = BusVoltages(
        vm_pu=np.array([1.0, 1.02]),
        va_deg=np.array([0.0, -5.0]),
        bus_names=["Bus1", "Bus2"],
    )
    line_flows = LineFlows(
        p_from_mw=np.array([10.0]),
        q_from_mvar=np.array([5.0]),
        loading_percent=np.array([45.0]),
        line_names=["Line1"],
    )
    metrics = NetworkMetrics(
        total_gen_mw=100.0,
        total_load_mw=95.0,
        total_loss_mw=5.0,
    )

    gs = GridState(
        owner_id="grid1",
        owner_level=2,
        features=[bus_voltages, line_flows, metrics]
    )
    vec = gs.vector()

    # 2 buses * 2 + 1 line * 3 + 5 metrics = 12 values
    assert vec.shape == (12,)


def test_grid_state_feature_names():
    """Test that feature names are accessible from features."""
    metrics = NetworkMetrics(total_gen_mw=100.0)

    gs = GridState(
        owner_id="grid1",
        owner_level=2,
        features=[metrics]
    )

    # Names come from the feature provider
    names = metrics.names()
    assert "total_gen_mw" in names


def test_grid_state_update_feature():
    """Test updating a feature via update_feature."""
    metrics = NetworkMetrics(total_gen_mw=100.0)

    gs = GridState(
        owner_id="grid1",
        owner_level=2,
        features=[metrics]
    )

    gs.update_feature("NetworkMetrics", total_gen_mw=200.0)

    assert metrics.total_gen_mw == 200.0


def test_grid_state_to_dict():
    """Test GridState serialization."""
    bus_voltages = BusVoltages(
        vm_pu=np.array([1.0, 1.02]),
        va_deg=np.array([0.0, -5.0]),
        bus_names=["Bus1", "Bus2"],
    )
    metrics = NetworkMetrics(total_gen_mw=100.0)

    gs = GridState(
        owner_id="grid1",
        owner_level=2,
        features=[bus_voltages, metrics]
    )

    d = gs.to_dict()

    assert "BusVoltages" in d
    assert "NetworkMetrics" in d
    assert np.allclose(d["BusVoltages"]["vm_pu"], [1.0, 1.02])
    assert d["NetworkMetrics"]["total_gen_mw"] == 100.0


def test_grid_state_reset():
    """Test GridState reset calls reset on features."""
    metrics = NetworkMetrics(total_gen_mw=100.0)

    gs = GridState(
        owner_id="grid1",
        owner_level=2,
        features=[metrics]
    )

    # Reset should not raise error
    gs.reset()

    # Verify features still exist after reset
    assert len(gs.features) == 1
    assert gs.features[0] is metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

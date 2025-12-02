"""Tests for GridState feature aggregation."""

import numpy as np
import pytest

from powergrid.core.state import GridState
from powergrid.features.network import BusVoltages, LineFlows, NetworkMetrics


def test_grid_state_empty():
    """Test GridState with no features."""
    gs = GridState()
    vec = gs.vector()
    names = gs.names()

    assert vec.shape == (0,)
    assert len(names) == 0


def test_grid_state_bus_voltages():
    """Test GridState with BusVoltages only."""
    bus_voltages = BusVoltages(
        vm_pu=np.array([1.0, 1.02, 0.98]),
        va_deg=np.array([0.0, -5.0, -10.0]),
        bus_names=["Bus1", "Bus2", "Bus3"],
    )

    gs = GridState(features=[bus_voltages])
    vec = gs.vector()
    names = gs.names()

    # 3 buses * 2 features (vm, va) = 6 values
    assert vec.shape == (6,)
    assert len(names) == 6
    assert names[0] == "vm_pu_Bus1"
    assert names[3] == "va_deg_Bus1"


def test_grid_state_line_flows():
    """Test GridState with LineFlows only."""
    line_flows = LineFlows(
        p_from_mw=np.array([10.0, 20.0]),
        q_from_mvar=np.array([5.0, 8.0]),
        loading_percent=np.array([45.0, 67.0]),
        line_names=["Line1", "Line2"],
    )

    gs = GridState(features=[line_flows])
    vec = gs.vector()
    names = gs.names()

    # 2 lines * 3 features (p, q, loading) = 6 values
    assert vec.shape == (6,)
    assert len(names) == 6
    assert names[0] == "p_from_mw_Line1"
    assert names[2] == "q_from_mvar_Line1"


def test_grid_state_network_metrics():
    """Test GridState with NetworkMetrics only."""
    metrics = NetworkMetrics(
        total_gen_mw=100.0,
        total_load_mw=95.0,
        total_loss_mw=5.0,
        total_gen_mvar=30.0,
        total_load_mvar=28.0,
    )

    gs = GridState(features=[metrics])
    vec = gs.vector()
    names = gs.names()

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

    gs = GridState(features=[bus_voltages, line_flows, metrics])
    vec = gs.vector()
    names = gs.names()

    # 2 buses * 2 + 1 line * 3 + 5 metrics = 12 values
    assert vec.shape == (12,)
    assert len(names) == 12


def test_grid_state_prefix_names():
    """Test GridState with prefix_names enabled."""
    metrics = NetworkMetrics(total_gen_mw=100.0)

    gs = GridState(features=[metrics], prefix_names=True)
    names = gs.names()

    assert all(name.startswith("NetworkMetrics.") for name in names)


def test_grid_state_clamp():
    """Test GridState clamp_ propagation."""
    bus_voltages = BusVoltages(
        vm_pu=np.array([3.0, -1.0]),  # Out of valid range
        va_deg=np.array([0.0, 0.0]),
    )

    gs = GridState(features=[bus_voltages])
    gs.clamp_()

    vec = gs.vector()
    # vm_pu should be clamped to [0, 2]
    assert vec[0] == 2.0
    assert vec[1] == 0.0


def test_grid_state_to_from_dict():
    """Test GridState serialization and deserialization."""
    bus_voltages = BusVoltages(
        vm_pu=np.array([1.0, 1.02]),
        va_deg=np.array([0.0, -5.0]),
        bus_names=["Bus1", "Bus2"],
    )
    metrics = NetworkMetrics(total_gen_mw=100.0)

    gs1 = GridState(features=[bus_voltages, metrics], prefix_names=True)
    d = gs1.to_dict()

    # Verify structure
    assert "features" in d
    assert len(d["features"]) == 2
    assert d["features"][0]["kind"] == "BusVoltages"
    assert d["features"][1]["kind"] == "NetworkMetrics"

    # Reconstruct
    from powergrid.utils.registry import ProviderRegistry
    gs2 = GridState.from_dict(d, registry=ProviderRegistry.all())

    # Verify equality
    assert np.allclose(gs1.vector(), gs2.vector())
    assert gs1.names() == gs2.names()

"""Tests for DeviceState with power domain features."""

import numpy as np
import pytest

from heron.core.state import DeviceState, FieldAgentState
from powergrid.utils.phase import PhaseModel, PhaseSpec
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.status import StatusBlock
from powergrid.features.connection import PhaseConnection


def assert_f32(vec: np.ndarray):
    """Assert vector is float32."""
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32, f"expected float32, got {vec.dtype}"


@pytest.fixture
def spec_3ph():
    """Create 3-phase spec."""
    return PhaseSpec("ABC", has_neutral=False, earth_bond=True)


def make_electrical_feature_balanced():
    """Create balanced electrical feature."""
    return ElectricalBasePh(
        phase_model=PhaseModel.BALANCED_1PH,
        P_MW=1.0,
        Q_MVAr=0.5,
    )


def make_status_feature():
    """Create status feature."""
    return StatusBlock(
        in_service=True,
        state="online",
        states_vocab=["offline", "online", "ramp_up", "ramp_down"],
        emit_state_one_hot=True,
        emit_state_index=False,
    )


def test_device_state_with_electrical():
    """Test DeviceState with ElectricalBasePh feature."""
    elec = make_electrical_feature_balanced()

    ds = FieldAgentState(
        owner_id="gen1",
        owner_level=1,
        features=[elec]
    )

    vec = ds.vector()
    assert_f32(vec)
    # ElectricalBasePh should contribute at least P and Q
    assert vec.size >= 2


def test_device_state_with_status():
    """Test DeviceState with StatusBlock feature."""
    status = make_status_feature()

    ds = FieldAgentState(
        owner_id="gen1",
        owner_level=1,
        features=[status]
    )

    vec = ds.vector()
    names = status.names()

    assert_f32(vec)
    assert vec.size == len(names)


def test_device_state_combined_features():
    """Test DeviceState with multiple features."""
    elec = make_electrical_feature_balanced()
    status = make_status_feature()

    ds = FieldAgentState(
        owner_id="gen1",
        owner_level=1,
        features=[elec, status]
    )

    vec = ds.vector()
    assert_f32(vec)

    # Vector should be concatenation of all feature vectors
    elec_vec = elec.vector()
    status_vec = status.vector()
    assert vec.size == elec_vec.size + status_vec.size


def test_device_state_feature_update():
    """Test updating features in DeviceState."""
    elec = make_electrical_feature_balanced()

    ds = FieldAgentState(
        owner_id="gen1",
        owner_level=1,
        features=[elec]
    )

    # Update electrical feature
    ds.update_feature("ElectricalBasePh", P_MW=2.0)

    assert elec.P_MW == 2.0


def test_device_state_reset():
    """Test DeviceState reset."""
    elec = make_electrical_feature_balanced()
    status = make_status_feature()

    ds = FieldAgentState(
        owner_id="gen1",
        owner_level=1,
        features=[elec, status]
    )

    # Reset should not raise error
    ds.reset()

    # Features should still be present
    assert len(ds.features) == 2


def test_device_state_to_dict():
    """Test DeviceState serialization."""
    elec = make_electrical_feature_balanced()
    status = make_status_feature()

    ds = FieldAgentState(
        owner_id="gen1",
        owner_level=1,
        features=[elec, status]
    )

    d = ds.to_dict()

    # Should contain feature names as keys
    assert "ElectricalBasePh" in d
    assert "StatusBlock" in d


def test_status_toggle_preserves_vector_shape():
    """Test that changing status state preserves vector shape."""
    status = make_status_feature()

    ds = FieldAgentState(
        owner_id="gen1",
        owner_level=1,
        features=[status]
    )

    v0 = ds.vector()

    # Change status state
    status.state = "ramp_up"

    v1 = ds.vector()

    assert v0.shape == v1.shape, "Vector shape should be stable under value changes"
    assert_f32(v1)


def test_three_phase_electrical():
    """Test 3-phase electrical feature."""
    spec = PhaseSpec("ABC", has_neutral=False, earth_bond=False)
    elec = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=spec,
        P_MW_ph=[0.5, 0.6, 0.7],
    )

    ds = FieldAgentState(
        owner_id="gen1",
        owner_level=1,
        features=[elec]
    )

    vec = ds.vector()
    assert_f32(vec)
    # 3-phase should have per-phase values
    assert vec.size >= 3


def test_device_state_alias():
    """Test DeviceState is an alias for FieldAgentState."""
    assert DeviceState is FieldAgentState


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

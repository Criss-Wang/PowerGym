import numpy as np
import pytest

from heron.core.state import DeviceState
from powergrid.utils.phase import PhaseModel, PhaseSpec

from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.status import StatusBlock
from powergrid.features.connection import PhaseConnection


def assert_f32(vec: np.ndarray):
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32, f"expected float32, got {vec.dtype}"


@pytest.fixture
def spec_3ph():
    return PhaseSpec("ABC", has_neutral=False, earth_bond=True)


# ------------------------------------------------------------
# DG (Distributed Generator) as a composition of features
# ------------------------------------------------------------

def make_dg_features_three_phase(spec: PhaseSpec):
    """A 3φ DG has per-phase electrical telemetry, a status block, and a connection."""
    elec = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=spec,
        # THREE_PHASE requires *at least one* per-phase vector:
        P_MW_ph=[0.5, 0.6, 0.7],
        # others optional; add more if you want (Q_MVAr_ph, V_pu_ph, theta_rad_ph, …)
    )
    status = StatusBlock(
        online=True,
        state="online",
        states_vocab=["offline", "online", "ramp_up", "ramp_down"],
        emit_state_one_hot=True,   # export one-hot states
        emit_state_index=False,
    )
    conn = PhaseConnection(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=spec,
        connection="ABC",          # fully connected DG
    )
    return [elec, status, conn]


def make_dg_features_balanced():
    """A 1φ balanced DG has scalar electrical telemetry, status, and a presence connection."""
    elec = ElectricalBasePh(
        phase_model=PhaseModel.BALANCED_1PH,
        # BALANCED_1PH requires *at least one* scalar:
        P_MW=1.2,
        # optional: Q_MVAr=..., V_pu=..., theta_rad=...
    )
    status = StatusBlock(
        online=True,
        state="online",
        states_vocab=["offline", "online", "ramp_up", "ramp_down"],
        emit_state_one_hot=True,
    )
    # For BALANCED_1PH, PhaseConnection may legitimately be empty (presence only)
    conn = PhaseConnection(
        phase_model=PhaseModel.BALANCED_1PH,
        connection="ON",  # or None (your feature allows empty vector for balanced if absent)
    )
    return [elec, status, conn]


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def test_dg_three_phase_into_device_state(spec_3ph):
    feats = make_dg_features_three_phase(spec_3ph)
    ds = DeviceState(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=spec_3ph,
        features=feats,
        prefix_names=True,   # ensure names are prefixed with real class names
    )

    vec, names = ds.vector(), ds.names()
    assert len(names) == vec.size
    assert_f32(vec)
    # Connection should contribute per-phase presence flags (length equals nph)
    assert any(n.startswith("PhaseConnection.") for n in names)


def test_dg_balanced_into_device_state():
    feats = make_dg_features_balanced()
    ds = DeviceState(
        phase_model=PhaseModel.BALANCED_1PH,
        phase_spec=None,
        features=feats,
        prefix_names=True,
    )

    vec, names = ds.vector(), ds.names()
    assert len(names) == vec.size
    assert_f32(vec)
    # ElectricalBasePh contributes scalars in balanced mode
    assert any(n.startswith("ElectricalBasePh.") for n in names)


def test_dg_status_toggle_keeps_shape(spec_3ph):
    """Status and connection may change values but should not change the DG observation length."""
    feats = make_dg_features_three_phase(spec_3ph)
    ds = DeviceState(phase_model=PhaseModel.THREE_PHASE, phase_spec=spec_3ph, features=feats)

    v0, n0 = ds.vector(), ds.names()
    assert_f32(v0)
    # Toggle status and connection values (same phase spec → same shape)
    sb = next(x for x in feats if isinstance(x, StatusBlock))
    pc = next(x for x in feats if isinstance(x, PhaseConnection))
    sb.state = "ramp_up"
    pc.connection = "BC"   # still 3φ, spec unchanged

    v1, n1 = ds.vector(), ds.names()
    assert v0.size == v1.size, "DG vector length should remain stable under value changes"
    assert n0 == n1, "DG names should remain stable under value changes"
    assert_f32(v1)


def test_dg_clamp_fanout(spec_3ph):
    """Clamp should fan out into features without altering shape/dtype."""
    feats = make_dg_features_three_phase(spec_3ph)
    # Push something out-of-range to verify clamp
    sb = next(x for x in feats if isinstance(x, StatusBlock))
    sb.progress_frac = 2.0  # invalid (>1), should be clamped

    ds = DeviceState(phase_model=PhaseModel.THREE_PHASE, phase_spec=spec_3ph, features=feats)
    v0 = ds.vector()
    ds.clamp_()
    v1 = ds.vector()
    assert v0.size == v1.size
    assert_f32(v1)


def test_dg_to_from_dict_roundtrip(spec_3ph):
    """Full serialization path using your registry (class name → class)."""
    feats = make_dg_features_three_phase(spec_3ph)
    ds0 = DeviceState(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=spec_3ph,
        features=feats,
        prefix_names=True,
    )

    registry = {
        "ElectricalBasePh": ElectricalBasePh,
        "StatusBlock": StatusBlock,
        "PhaseConnection": PhaseConnection,
    }

    d = ds0.to_dict()
    ds1 = DeviceState.from_dict(d, registry=registry)

    v0, n0 = ds0.vector(), ds0.names()
    v1, n1 = ds1.vector(), ds1.names()
    assert v0.shape == v1.shape
    assert np.allclose(v0, v1)
    assert n0 == n1


def test_dg_context_switch_three_to_balanced(spec_3ph):
    """
    When the *device context* changes (3φ → 1φ), DeviceState re-applies phase context to features.
    We expect a different vector length across contexts (that’s OK), but still consistent parity/dtype.
    """
    feats = make_dg_features_three_phase(spec_3ph)
    ds = DeviceState(phase_model=PhaseModel.THREE_PHASE, phase_spec=spec_3ph, features=feats)
    v0, n0 = ds.vector(), ds.names()
    assert_f32(v0)
    assert len(n0) == v0.size

    # Switch context on the DeviceState and re-apply
    ds.phase_model = PhaseModel.BALANCED_1PH
    ds.phase_spec = None
    ds._apply_phase_context_to_features_()

    v1, n1 = ds.vector(), ds.names()
    assert_f32(v1)
    assert len(n1) == v1.size
    assert v1.size != v0.size, "context change may legitimately change observation length"

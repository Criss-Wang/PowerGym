"""Tests for StorageBlock feature provider."""

import json
import numpy as np
import pytest

from powergrid.features.storage import StorageBlock


def _assert_vec_names_consistent(b: StorageBlock):
    """Assert vector and names have same length."""
    v = b.vector().ravel()
    n = b.names()
    assert v.ndim == 1
    assert len(v) == len(n), f"len(vector)={len(v)} vs len(names)={len(n)}"


# ----------------- GOOD EXAMPLES -----------------

def test_good_minimal_required_fields():
    """Test StorageBlock with all required fields."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )
    _assert_vec_names_consistent(b)
    assert "soc" in b.names()
    assert b.soc == 0.5
    assert b.soc_min == 0.0
    assert b.soc_max == 1.0
    assert b.e_capacity_MWh == 10.0


def test_good_capacity_and_power_limits():
    """Test StorageBlock with capacity and power limits."""
    b = StorageBlock(
        soc=0.6,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=8.0,
        p_ch_max_MW=2.0,
        p_dsc_max_MW=3.0,
    )
    _assert_vec_names_consistent(b)
    assert b.p_ch_max_MW == 2.0
    assert b.p_dsc_max_MW == 3.0


def test_good_with_efficiency():
    """Test StorageBlock with efficiency values."""
    b = StorageBlock(
        soc=0.4,
        soc_min=0.1,
        soc_max=0.9,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
        ch_eff=0.9,
        dsc_eff=0.95,
    )
    assert b.ch_eff == 0.9
    assert b.dsc_eff == 0.95
    _assert_vec_names_consistent(b)


def test_good_with_degradation_params():
    """Test StorageBlock with degradation parameters."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
        degr_cost_per_MWh=0.1,
        degr_cost_per_cycle=1.0,
    )
    assert b.degr_cost_per_MWh == 0.1
    assert b.degr_cost_per_cycle == 1.0
    _assert_vec_names_consistent(b)


def test_good_roundtrip_serialization():
    """Test serialization and deserialization."""
    b0 = StorageBlock(
        soc=0.5,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )
    d = b0.to_dict()
    s = json.dumps(d)
    d2 = json.loads(s)
    b1 = StorageBlock.from_dict(d2)
    _assert_vec_names_consistent(b1)
    assert len(b0.names()) == len(b1.names())
    assert np.allclose(b0.vector(), b1.vector())


# ----------------- BAD EXAMPLES -----------------

def test_bad_missing_soc():
    """Test that missing soc raises error."""
    with pytest.raises(ValueError, match="soc is None"):
        StorageBlock(
            soc=None,
            soc_min=0.0,
            soc_max=1.0,
            e_capacity_MWh=10.0,
            p_ch_max_MW=5.0,
            p_dsc_max_MW=5.0,
        )


def test_bad_missing_soc_min():
    """Test that missing soc_min raises error."""
    with pytest.raises(ValueError, match="soc_min is None"):
        StorageBlock(
            soc=0.5,
            soc_min=None,
            soc_max=1.0,
            e_capacity_MWh=10.0,
            p_ch_max_MW=5.0,
            p_dsc_max_MW=5.0,
        )


def test_bad_missing_soc_max():
    """Test that missing soc_max raises error."""
    with pytest.raises(ValueError, match="soc_max is None"):
        StorageBlock(
            soc=0.5,
            soc_min=0.0,
            soc_max=None,
            e_capacity_MWh=10.0,
            p_ch_max_MW=5.0,
            p_dsc_max_MW=5.0,
        )


def test_bad_missing_capacity():
    """Test that missing e_capacity_MWh raises error."""
    with pytest.raises(ValueError, match="e_capacity_MWh is None"):
        StorageBlock(
            soc=0.5,
            soc_min=0.0,
            soc_max=1.0,
            e_capacity_MWh=None,
            p_ch_max_MW=5.0,
            p_dsc_max_MW=5.0,
        )


def test_bad_missing_p_ch_max():
    """Test that missing p_ch_max_MW raises error."""
    with pytest.raises(ValueError, match="p_ch_max_MW is None"):
        StorageBlock(
            soc=0.5,
            soc_min=0.0,
            soc_max=1.0,
            e_capacity_MWh=10.0,
            p_ch_max_MW=None,
            p_dsc_max_MW=5.0,
        )


def test_bad_missing_p_dsc_max():
    """Test that missing p_dsc_max_MW raises error."""
    with pytest.raises(ValueError, match="p_dsc_max_MW is None"):
        StorageBlock(
            soc=0.5,
            soc_min=0.0,
            soc_max=1.0,
            e_capacity_MWh=10.0,
            p_ch_max_MW=5.0,
            p_dsc_max_MW=None,
        )


def test_bad_soc_bounds_reversed():
    """Test that soc_min > soc_max raises error."""
    with pytest.raises(ValueError, match="soc_min cannot be greater than soc_max"):
        StorageBlock(
            soc=0.5,
            soc_min=0.8,
            soc_max=0.2,
            e_capacity_MWh=10.0,
            p_ch_max_MW=5.0,
            p_dsc_max_MW=5.0,
        )


def test_bad_soc_out_of_range():
    """Test that soc out of [0, 1] raises error."""
    with pytest.raises(ValueError, match="soc must be in"):
        StorageBlock(
            soc=1.5,
            soc_min=0.0,
            soc_max=1.0,
            e_capacity_MWh=10.0,
            p_ch_max_MW=5.0,
            p_dsc_max_MW=5.0,
        )


def test_bad_negative_capacity():
    """Test that negative capacity raises error."""
    with pytest.raises(ValueError, match="e_capacity_MWh must be > 0"):
        StorageBlock(
            soc=0.5,
            soc_min=0.0,
            soc_max=1.0,
            e_capacity_MWh=-5.0,
            p_ch_max_MW=5.0,
            p_dsc_max_MW=5.0,
        )


def test_bad_negative_power_limit():
    """Test that negative power limit raises error."""
    with pytest.raises(ValueError, match="p_ch_max_MW must be >= 0"):
        StorageBlock(
            soc=0.5,
            soc_min=0.0,
            soc_max=1.0,
            e_capacity_MWh=10.0,
            p_ch_max_MW=-2.0,
            p_dsc_max_MW=5.0,
        )


def test_bad_efficiency_out_of_range():
    """Test that efficiency out of (0, 1] raises error."""
    with pytest.raises(ValueError, match="ch_eff must be in"):
        StorageBlock(
            soc=0.5,
            soc_min=0.0,
            soc_max=1.0,
            e_capacity_MWh=10.0,
            p_ch_max_MW=5.0,
            p_dsc_max_MW=5.0,
            ch_eff=1.5,
        )


# ----------------- CLIP & BOUNDS -----------------

def test_clip_clamps_soc():
    """Test clip_() clamps SOC to bounds."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.3,
        soc_max=0.7,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )
    # Manually set soc outside bounds and clip
    b.soc = 0.1
    b.clip_()
    assert b.soc == 0.3  # Clamped to soc_min


def test_clip_clamps_efficiency():
    """Test clip_() clamps efficiency to valid range."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )
    # Test that clip keeps efficiency in valid range
    b.ch_eff = 1.5
    b.clip_()
    assert b.ch_eff == 1.0


# ----------------- VECTOR & NAMES -----------------

def test_vector_names_alignment():
    """Test vector and names are aligned."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )
    _assert_vec_names_consistent(b)
    # Current implementation includes: soc, e_throughput_MWh, equiv_full_cycles
    names = b.names()
    assert "soc" in names
    assert "e_throughput_MWh" in names
    assert "equiv_full_cycles" in names


# ----------------- DEGRADATION -----------------

def test_accumulate_throughput():
    """Test degradation throughput accumulation."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
        degr_cost_per_MWh=0.1,
        degr_cost_per_cycle=1.0,
    )

    assert b.e_throughput_MWh == 0.0
    assert b.equiv_full_cycles == 0.0

    # Accumulate 5 MWh throughput
    cost = b.accumulate_throughput(5.0)

    assert b.e_throughput_MWh == 5.0
    assert b.equiv_full_cycles == 0.5  # 5 / 10 capacity
    assert cost > 0  # Should have some cost


def test_soc_violation():
    """Test SOC violation computation."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.2,
        soc_max=0.8,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )

    # No violation when soc is within bounds
    assert b.soc_violation() == 0.0

    # Violation when below min
    b.soc = 0.1
    assert b.soc_violation() == 0.1  # 0.2 - 0.1


# ----------------- RESET -----------------

def test_reset_basic():
    """Test basic reset functionality."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )

    # Add some throughput
    b.accumulate_throughput(5.0)
    assert b.e_throughput_MWh > 0

    # Reset
    b.reset(soc=0.3, reset_degradation=True)

    assert b.soc == 0.3
    assert b.e_throughput_MWh == 0.0
    assert b.degr_cost_cum == 0.0


def test_reset_random():
    """Test random reset functionality."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.2,
        soc_max=0.8,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )

    b.reset(random_init=True, seed=42)

    assert 0.2 <= b.soc <= 0.8


# ----------------- SET_VALUES -----------------

def test_set_values():
    """Test set_values method."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )

    b.set_values(soc=0.7, p_ch_max_MW=3.0)

    assert b.soc == 0.7
    assert b.p_ch_max_MW == 3.0


def test_set_values_unknown_field():
    """Test set_values rejects unknown fields."""
    b = StorageBlock(
        soc=0.5,
        soc_min=0.0,
        soc_max=1.0,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dsc_max_MW=5.0,
    )

    with pytest.raises(AttributeError, match="unknown fields"):
        b.set_values(unknown_field=1.0)


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))

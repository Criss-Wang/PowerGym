import math
import numpy as np
import pytest

from powergrid.features.generator_limits import GeneratorLimits


def assert_f32(v: np.ndarray):
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float32, f"expected float32, got {v.dtype}"


def test_vector_names_parity_and_dtype():
    gl = GeneratorLimits(
        s_rated_MVA=10.0,
        derate_frac=0.75,
        p_min_MW=1.0, p_max_MW=8.0,
        q_min_MVAr=-4.0, q_max_MVAr=3.0,
        pf_min_abs=0.9,
    )
    v, n = gl.vector(), gl.names()
    assert len(n) == v.size
    assert_f32(v)
    # names should include all provided fields in order
    assert n == [
        "s_rated_MVA",
        "derate_frac",
        "p_min_MW",
        "p_max_MW",
        "q_min_MVAr",
        "q_max_MVAr",
        "pf_min_abs",
    ]


def test_clamp_swapped_bounds_and_derate():
    gl = GeneratorLimits(
        s_rated_MVA=5.0,
        derate_frac=1.5,       # will clamp to 1.0
        p_min_MW=5.0, p_max_MW=2.0,   # swapped
        q_min_MVAr=3.0, q_max_MVAr=-1.0,  # swapped
    )
    gl.clamp_()
    assert gl.derate_frac == 1.0
    assert gl.p_min_MW == 2.0 and gl.p_max_MW == 5.0
    assert gl.q_min_MVAr == -1.0 and gl.q_max_MVAr == 3.0


def test_pf_min_abs_validation():
    with pytest.raises(ValueError):
        GeneratorLimits(pf_min_abs=0.0).clamp_()
    with pytest.raises(ValueError):
        GeneratorLimits(pf_min_abs=1.01).clamp_()
    # valid edge
    GeneratorLimits(pf_min_abs=1.0).clamp_()


def test_effective_q_bounds_with_S_only():
    gl = GeneratorLimits(s_rated_MVA=10.0)  # no derate, no static Q bounds
    # at P=6, |Q| <= sqrt(10^2 - 6^2) = 8
    qmin, qmax = gl.effective_q_bounds(6.0)
    assert pytest.approx(qmin, rel=1e-6) == -8.0
    assert pytest.approx(qmax, rel=1e-6) == 8.0


def test_effective_q_bounds_with_PF_only():
    pf = 0.8
    gl = GeneratorLimits(pf_min_abs=pf)
    # |Q| <= |P| * tan(arccos(pf))
    tanphi = math.sqrt(1.0 / (pf*pf) - 1.0)
    qmin, qmax = gl.effective_q_bounds(5.0)
    assert pytest.approx(qmin, rel=1e-6) == -5.0 * tanphi
    assert pytest.approx(qmax, rel=1e-6) ==  5.0 * tanphi


def test_effective_q_bounds_intersection_S_and_PF_and_static():
    gl = GeneratorLimits(
        s_rated_MVA=5.0,
        pf_min_abs=0.9,
        q_min_MVAr=-1.0,
        q_max_MVAr=1.5
    )
    tanphi = math.sqrt(1.0 / (0.9*0.9) - 1.0)
    q_wedge = 3.0 * tanphi  # ≈ 1.452 < 1.5

    qmin, qmax = gl.effective_q_bounds(3.0)

    # Lower bound: max(-1.0, -4.0, -1.452) = -1.0 (static dominates)
    assert pytest.approx(qmin, rel=1e-6) == -1.0

    # Upper bound: min(1.5, 4.0, 1.452) = q_wedge (PF wedge dominates)
    assert pytest.approx(qmax, rel=1e-6) == q_wedge

    # sanity
    assert q_wedge < 1.5                      # PF wedge is tighter than static here
    assert math.sqrt(25 - 9) > 1.5            # S-circle is looser than static here


def test_feasible_reports_violations():
    gl = GeneratorLimits(
        s_rated_MVA=5.0,
        p_min_MW=1.0,
        p_max_MW=4.0,
        q_min_MVAr=-2.0,
        q_max_MVAr=2.0,
        pf_min_abs=0.9,
    )
    # Choose a clearly infeasible point
    P, Q = 5.5, 3.0
    viol = gl.feasible(P, Q)
    assert viol["p_violation"] > 0.0  # P above p_max
    assert viol["q_violation"] > 0.0  # Q above q_max
    assert viol["s_excess"] > 0.0     # exceeds S circle
    # PF violation may or may not be >0 depending on S; ensure key exists and non-negative
    assert viol["pf_violation"] >= 0.0


def test_project_pq_clips_static_ranges():
    gl = GeneratorLimits(p_min_MW=1.0, p_max_MW=4.0, q_min_MVAr=-2.0, q_max_MVAr=2.0)
    # Below P min, above Q max -> clip both
    P2, Q2 = gl.project_pq(0.5, 3.0)
    assert P2 == 1.0
    assert Q2 == 2.0
    # Above P max, below Q min -> clip both
    P2, Q2 = gl.project_pq(5.0, -5.0)
    assert P2 == 4.0
    assert Q2 == -2.0


def test_project_pq_respects_S_circle_and_pf_wedge():
    gl = GeneratorLimits(s_rated_MVA=5.0, pf_min_abs=0.8)
    # pick P inside S but Q outside PF wedge
    tanphi = math.sqrt(1.0 / (0.8*0.8) - 1.0)
    P, Q = 3.0, 3.0  # this Q is likely above PF wedge at P=3
    P2, Q2 = gl.project_pq(P, Q)
    # S-circle at P=3: |Q|<=4; PF wedge at P=3: |Q|<=3*tanphi
    assert abs(P2 - P) < 1e-9
    assert abs(Q2) <= 3.0 * tanphi + 1e-6  # projected into PF wedge
    # Now violate S circle strongly
    P, Q = 5.0, 5.0
    P2, Q2 = gl.project_pq(P, Q)
    # After projection, hypot(P2,Q2) <= S
    S = gl.s_rated_MVA
    assert math.hypot(P2, Q2) <= S + 1e-6


def test_roundtrip_to_from_dict():
    gl = GeneratorLimits(
        s_rated_MVA=12.5,
        derate_frac=0.6,
        p_min_MW=1.5, p_max_MW=10.0,
        q_min_MVAr=-4.5, q_max_MVAr=3.2,
        pf_min_abs=0.85,
    )
    d = gl.to_dict()
    gl2 = GeneratorLimits.from_dict(d)
    # Same configuration after round-trip
    assert gl2.to_dict() == d
    # Vectors/names identical
    v1, v2 = gl.vector(), gl2.vector()
    n1, n2 = gl.names(), gl2.names()
    assert n1 == n2
    assert v1.shape == v2.shape and np.allclose(v1, v2)


def test_effective_q_bounds_returns_none_when_unconstrained():
    gl = GeneratorLimits()  # no S, no PF, no static Q
    qmin, qmax = gl.effective_q_bounds(0.0)
    assert qmin is None and qmax is None
    # project should then leave Q unchanged (except P clipping if set)
    P2, Q2 = gl.project_pq(100.0, -100.0)
    assert P2 == 100.0 and Q2 == -100.0


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    # You can pass pytest args like -q, -v, etc.
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))
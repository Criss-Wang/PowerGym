import math
import numpy as np
import pytest

from powergrid.agents.generator import Generator
from powergrid.utils.phase import PhaseModel


def make_config(
    *,
    with_q=True,
    with_uc=True,
    s_mva=10.0,
    pmin=1.0,
    pmax=8.0,
    qmin=-3.0,
    qmax=3.0,
    pfmin=0.8,
    dt_h=1.0,
):
    cfg = {
        "name": "G1",
        "device_state_config": {
            "phase_model": PhaseModel.BALANCED_1PH.value,
            "s_rated_MVA": s_mva,
            "derate_frac": 1.0,
            "p_min_MW": pmin,
            "p_max_MW": pmax,
            "cost_curve_coefs": (0.01, 1.0, 0.0),
            "dt_h": dt_h,
            "min_pf": pfmin,
        },
    }
    if with_q:
        cfg["device_state_config"]["q_min_MVAr"] = qmin
        cfg["device_state_config"]["q_max_MVAr"] = qmax
        cfg["device_state_config"]["pf_min_abs"] = pfmin
    if with_uc:
        cfg["device_state_config"]["startup_time_hr"] = 2
        cfg["device_state_config"]["shutdown_time_hr"] = 2
        cfg["device_state_config"]["startup_cost"] = 5.0
        cfg["device_state_config"]["shutdown_cost"] = 1.0
    return cfg


def test_features_and_observation_vector_parity():
    dg = Generator(agent_id="G1", device_config=make_config())
    # Feature types present
    kinds = {type(f).__name__ for f in dg.state.features}
    assert "ElectricalBasePh" in kinds
    assert "StatusBlock" in kinds
    assert "PowerLimits" in kinds  # Not "GeneratorLimits"

    # Observation space shape == state.vector().shape
    vec = dg.state.vector()
    obs = dg.observe()
    assert vec.dtype == np.float32
    assert obs.local["state"].shape == vec.shape


def test_action_space_continuous_only_and_mixed_uc():
    # P + Q + UC; DeviceAgent action space is built from continuous part (Box)
    dg = Generator(
        agent_id="G1",
        device_config=make_config(with_q=True, with_uc=True)
    )
    assert dg.action.dim_c == 2  # Control both P and Q
    assert dg.action.dim_d == 1  # Control when to startup or shutdown
    low, high = dg.action.range
    assert low.shape == (2,) and high.shape == (2,)
    # UC head exists with 2 categories (off=0, on=1)
    assert dg.action.ncats == [2]


def test_projection_applies_limits():
    dg = Generator(
        agent_id="G1",
        device_config=make_config(with_q=True, with_uc=False)
    )
    # ask for an over-limit action: P>pmax and Q>qmax → must be projected
    # Set action directly
    dg.action.c[:] = np.array([9.0, 4.0], dtype=np.float32)
    dg.apply_action()  # Modern pattern: apply_action() updates state
    e = dg.electrical
    lim = dg.limits

    # within static bounds
    assert lim.p_min_MW <= e.P_MW <= lim.p_max_MW
    assert lim.q_min_MVAr <= e.Q_MVAr <= lim.q_max_MVAr
    # within S circle if S is set
    if lim.s_rated_MVA is not None:
        assert math.hypot(e.P_MW, e.Q_MVAr) <= lim.s_rated_MVA + 1e-6
    # within PF wedge if pf_min_abs set
    if lim.pf_min_abs is not None and abs(e.P_MW) > 1e-8:
        tanphi = math.sqrt(1.0 / (lim.pf_min_abs**2) - 1.0)
        assert abs(e.Q_MVAr) <= abs(e.P_MW) * tanphi + 1e-6


def test_uc_shutdown_then_startup_costs_and_states():
    dg = Generator(
        agent_id="G1",
        device_config=make_config(with_q=True, with_uc=True)
    )
    # State is initialized during __init__, no need for reset_agent()
    # dg.status.state == "online"

    # Request OFF: d=0; set continuous c as well
    dg.action.c[:] = np.array([5.0, 0.0], dtype=np.float32)
    dg.action.d[:] = np.array([0], dtype=np.int32)  # UC command: turn off
    dg.apply_action()  # Modern pattern: apply_action() updates state
    # first step: entering "shutdown"
    assert dg.status.state == "shutdown"

    # advance one more step to complete shutdown (shutdown_time_hr=2)
    dg.action.d[:] = np.array([0], dtype=np.int32)
    dg.apply_action()
    dg.update_cost_safety()
    assert dg.status.state == "offline"
    # shutdown cost applied on completion
    assert dg.cost >= dg._shutdown_cost * dg._dt_h

    # Now request ON: d=1, two steps to start (startup_time_hr=2)
    dg.action.d[:] = np.array([1], dtype=np.int32)  # UC command: turn on
    dg.apply_action()
    assert dg.status.state == "startup"

    dg.action.d[:] = np.array([1], dtype=np.int32)
    dg.apply_action()
    dg.update_cost_safety()
    assert dg.status.state == "online"
    # startup cost applied on completion
    assert dg.cost >= dg._startup_cost * dg._dt_h


def test_uc_timers_startup_shutdown():
    dg = Generator(
        agent_id="G1",
        device_config=make_config(with_q=True, with_uc=True)
    )
    # State is initialized during __init__, no need for reset_agent()

    # Request OFF (d=0) → enter shutdown → next step complete → offline
    dg.action.c[:] = np.array([5.0, 0.0], dtype=np.float32)
    dg.action.d[:] = np.array([0], dtype=np.int32)
    dg.apply_action()  # Modern pattern: apply_action() updates state
    assert dg.status.state == "shutdown"

    dg.action.d[:] = np.array([0], dtype=np.int32)
    dg.apply_action()
    dg.update_cost_safety()
    assert dg.status.state == "offline"
    assert dg.cost >= dg._shutdown_cost * dg._dt_h

    # Request ON (d=1) → startup spans 2 steps → online and startup_cost
    dg.action.d[:] = np.array([1], dtype=np.int32)
    dg.apply_action()
    assert dg.status.state == "startup"

    dg.action.d[:] = np.array([1], dtype=np.int32)
    dg.apply_action()
    dg.update_cost_safety()
    assert dg.status.state == "online"
    assert dg.cost >= dg._startup_cost * dg._dt_h


def test_cost_and_safety_accounting():
    dg = Generator(
        agent_id="G1",
        device_config=make_config(with_q=True, with_uc=False)
    )
    # Set a feasible but nonzero (P,Q)
    dg.action.c[:] = np.array([6.0, 2.0], dtype=np.float32)
    dg.apply_action()  # Modern pattern: apply_action() updates state
    dg.update_cost_safety()
    # Cost should be positive when online and P>0
    assert dg.cost > 0.0
    # Safety non-negative; positive if S and PF penalties kick in
    assert dg.safety >= 0.0


def test_feasible_action_preclips_action_vector():
    dg = Generator(
        agent_id="G1",
        device_config=make_config(with_q=True, with_uc=False)
    )
    # Action far outside bounds
    dg.action.c[:] = np.array([999.0, 999.0], dtype=np.float32)
    # Call clip to get into range (feasible_action is for ESS, not generator)
    dg.action.clip()
    P, Q = dg.action.c.tolist()
    lim = dg.limits
    assert lim.p_min_MW <= P <= lim.p_max_MW
    assert lim.q_min_MVAr <= Q <= lim.q_max_MVAr


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    # You can pass pytest args like -q, -v, etc.
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))

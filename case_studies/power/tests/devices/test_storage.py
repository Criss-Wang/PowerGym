"""Tests for devices.storage module (ESS)."""

import numpy as np
import pytest

from powergrid.agents.storage import ESS
from heron.core.policies import Policy
from heron.agents.base import Observation
from powergrid.utils.phase import PhaseModel


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, action_value=0.5):
        self.action_value = action_value

    def forward(self, observation):
        """Return fixed action."""
        if isinstance(self.action_value, (list, np.ndarray)):
            return np.array(self.action_value, dtype=np.float32)
        return np.array([self.action_value], dtype=np.float32)


def make_ess_config(
    bus="Bus1",
    p_min_MW=-2.0,
    p_max_MW=2.0,
    e_capacity_MWh=10.0,
    soc_min=0.0,
    soc_max=1.0,
    init_soc=0.5,
    q_min_MVAr=None,
    q_max_MVAr=None,
    s_rated_MVA=None,
    ch_eff=0.98,
    dsc_eff=0.98,
    dt_h=1.0,
):
    """Helper to create ESS device config.

    Field names match ESS implementation's expected config keys:
    - p_min_MW: minimum power (negative for discharge)
    - p_max_MW: maximum power (positive for charge)
    - e_capacity_MWh: nameplate energy capacity
    - soc_min/soc_max: SOC bounds [0, 1]
    - init_soc: initial SOC [0, 1]
    """
    config = {
        "device_state_config": {
            "phase_model": PhaseModel.BALANCED_1PH.value,
            "bus": bus,
            "p_min_MW": p_min_MW,
            "p_max_MW": p_max_MW,
            "e_capacity_MWh": e_capacity_MWh,
            "soc_min": soc_min,
            "soc_max": soc_max,
            "init_soc": init_soc,
            "ch_eff": ch_eff,
            "dsc_eff": dsc_eff,
            "dt_h": dt_h,
        }
    }
    if s_rated_MVA is not None:
        config["device_state_config"]["s_rated_MVA"] = s_rated_MVA
    if q_min_MVAr is not None:
        config["device_state_config"]["q_min_MVAr"] = q_min_MVAr
    if q_max_MVAr is not None:
        config["device_state_config"]["q_max_MVAr"] = q_max_MVAr
    return config


class TestESS:
    """Test ESS (Energy Storage System) device."""

    def test_ess_initialization(self):
        """Test ESS initialization."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config()
        )

        assert ess.agent_id == "ESS1"
        assert ess.bus == "Bus1"
        assert ess._storage_config.p_min_MW == -2.0
        assert ess._storage_config.p_max_MW == 2.0
        assert ess._storage_config.e_capacity_MWh == 10.0

    def test_ess_with_q_control(self):
        """Test ESS with reactive power control."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config(
                s_rated_MVA=3.0,
                q_min_MVAr=-1.5,
                q_max_MVAr=1.5
            )
        )

        # Should have Q limits
        assert ess._storage_config.q_min_MVAr == -1.5
        assert ess._storage_config.q_max_MVAr == 1.5
        assert ess.action.dim_c == 2  # P and Q

    def test_ess_action_space_p_only(self):
        """Test action space with P control only."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config()
        )

        assert ess.action.dim_c == 1
        assert ess.action.range.shape == (2, 1)
        np.testing.assert_array_almost_equal(ess.action.range[0], [-2.0])
        np.testing.assert_array_almost_equal(ess.action.range[1], [2.0])

    def test_ess_soc_initialization(self):
        """Test SOC initialization."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config(init_soc=0.7)
        )

        assert ess.storage.soc == 0.7

    def test_ess_update_state_charging(self):
        """Test state update during charging (P > 0)."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config(
                init_soc=0.5,
                ch_eff=0.95,
                dt_h=1.0
            )
        )

        # Charge at 1 MW for 1 hour
        ess.action.c = np.array([1.0], dtype=np.float32)
        ess.update_state()

        # SOC should increase: 0.5 + (1.0 * 0.95 * 1.0 / 10.0) = 0.595
        assert ess.electrical.P_MW == 1.0
        np.testing.assert_almost_equal(ess.storage.soc, 0.595)

    def test_ess_update_state_discharging(self):
        """Test state update during discharging (P < 0)."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config(
                init_soc=0.5,
                dsc_eff=0.95,
                dt_h=1.0
            )
        )

        # Discharge at -1 MW for 1 hour
        ess.action.c = np.array([-1.0], dtype=np.float32)
        ess.update_state()

        # SOC should decrease: 0.5 + (-1.0 / 0.95 * 1.0 / 10.0) = 0.3947
        assert ess.electrical.P_MW == -1.0
        np.testing.assert_almost_equal(ess.storage.soc, 0.3947, decimal=4)

    def test_ess_update_state_with_q(self):
        """Test state update with P and Q control."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy([1.0, 0.5]),
            device_config=make_ess_config(
                s_rated_MVA=3.0,
                q_min_MVAr=-1.5,
                q_max_MVAr=1.5
            )
        )

        ess.action.c = np.array([1.0, 0.5], dtype=np.float32)
        ess.update_state()

        assert ess.electrical.P_MW == 1.0
        assert ess.electrical.Q_MVAr == 0.5

    def test_ess_update_cost_safety(self):
        """Test cost and safety updates."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config(dt_h=1.0)
        )

        ess.electrical.P_MW = 1.0
        ess.storage.soc = 0.5
        ess.update_cost_safety()

        # Cost should be calculated (may include degradation cost)
        assert ess.cost >= 0
        assert ess.safety >= 0

    def test_ess_feasible_action(self):
        """Test feasible action clamping based on SOC."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config(
                soc_min=0.1,
                soc_max=0.9,
                ch_eff=1.0,
                dsc_eff=1.0,
                dt_h=1.0
            )
        )

        # Set SOC near minimum
        ess.storage.soc = 0.15  # 1.5 MWh out of 10 MWh capacity

        # Try to discharge at max power - feasible_action returns clipped value
        P_req = -2.0
        P_clipped = ess.feasible_action(P_req)

        # Should be clamped to prevent going below soc_min
        # Available discharge: (0.15 - 0.1) * 10 = 0.5 MWh over 1 hour = 0.5 MW
        assert P_clipped >= -0.5

    def test_ess_feasible_action_charging(self):
        """Test feasible action clamping during charging."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config(
                soc_max=0.9,
                ch_eff=1.0,
                dt_h=1.0
            )
        )

        # Set SOC near maximum
        ess.storage.soc = 0.85  # 8.5 MWh out of 10 MWh capacity

        # Try to charge at max power - feasible_action returns clipped value
        P_req = 2.0
        P_clipped = ess.feasible_action(P_req)

        # Should be clamped to prevent exceeding soc_max
        # Available charge: (0.9 - 0.85) * 10 = 0.5 MWh over 1 hour = 0.5 MW
        np.testing.assert_almost_equal(P_clipped, 0.5, decimal=5)

    def test_ess_reset(self):
        """Test ESS reset."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config(
                soc_min=0.2,
                soc_max=0.8
            )
        )

        # Modify state
        ess.electrical.P_MW = 1.5
        ess.storage.soc = 0.3
        ess.cost = 100.0

        # Reset with specific SOC
        ess.reset_device(soc=0.6)

        assert ess.electrical.P_MW == 0.0
        assert ess.storage.soc == 0.6
        assert ess.cost == 0.0
        assert ess.safety == 0.0

    def test_ess_reset_random_soc(self):
        """Test ESS reset with random SOC."""
        ess = ESS(
            agent_id="ESS1",
            policy=MockPolicy(),
            device_config=make_ess_config(
                soc_min=0.2,
                soc_max=0.8
            )
        )

        # Reset with random init
        ess.reset_device(random_init_soc=True, seed=42)

        # SOC should be between soc_min and soc_max
        assert ess._storage_config.soc_min <= ess.storage.soc <= ess._storage_config.soc_max

    def test_ess_full_lifecycle(self):
        """Test full ESS lifecycle."""
        policy = MockPolicy(action_value=1.0)
        ess = ESS(
            agent_id="ESS1",
            policy=policy,
            device_config=make_ess_config(init_soc=0.5)
        )

        # Reset
        ess.reset()
        initial_soc = ess.storage.soc

        # Observe
        obs = ess.observe()
        assert isinstance(obs, Observation)

        # Act - charge
        ess.act(obs)
        np.testing.assert_array_almost_equal(ess.action.c, [1.0])

        # Update
        ess.update_state()
        assert ess.storage.soc > initial_soc

        # Update cost/safety
        ess.update_cost_safety()
        assert ess.cost >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

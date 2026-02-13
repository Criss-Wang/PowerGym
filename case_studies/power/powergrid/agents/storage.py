from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from powergrid.agents.device_agent import DeviceAgent, CostSafetyMetrics
from heron.core.action import Action
from heron.core.feature import FeatureProvider
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig
from heron.utils.typing import AgentID
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.power_limits import PowerLimits
from powergrid.core.features.status import StatusBlock
from powergrid.core.features.storage import StorageBlock
from powergrid.utils.phase import PhaseModel, PhaseSpec, check_phase_model_consistency


@dataclass
class StorageConfig:
    """Configuration for an Energy Storage System.

    Note: Most storage parameters (SOC limits, capacity, efficiencies, degradation)
    are stored in the StorageBlock feature. Reactive power limits (if Q control is
    enabled) are in the PowerLimits feature.
    """
    bus: str

    # Phase model configuration
    phase_model: str = "balanced_1ph"
    phase_spec: Optional[Dict[str, Any]] = None

    # Power bounds for action space and feasibility checks
    p_min_MW: float = 0.0          # allowed negative (discharge)
    p_max_MW: float = 0.0          # positive (charge)

    # Time step for dynamics calculations
    dt_h: float = 1.0

class ESS(DeviceAgent):
    """
    Energy Storage System following DeviceAgent's lifecycle.

    Example device_config:
    device_config = {
        "phase_model": "balanced_1ph",
        "phase_spec":  {
            "phases": "", 
            "has_neutral": False, 
            "earth_bond": False
        },
        "device_state_config": {
            "bus": "bus_1",
            "p_min_MW": -10.0,
            "p_max_MW": 10.0,
            "capacity_MWh": 20.0,
            "max_e_MWh": 18.0,
            "min_e_MWh": 2.0,
            "init_soc": 0.5,

            "q_min_MVAr": -5.0,
            "q_max_MVAr": 5.0,
            "s_rated_MVA": 12.0,

            "ch_eff": 0.98,
            "dsc_eff": 0.98,

            e_throughput_MWh = 0.0
            degr_cost_per_MWh = 0.5
            degr_cost_per_cycle = 2.8
            degr_cost_cum = 0.0

            "dt_h": 1.0
        }
    }
    """

    def __init__(
        self,
        agent_id: str,
        storage_config: StorageConfig,
        features: List[FeatureProvider] = [],
        # hierarchy params
        upstream_id: Optional[str] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None,
    ):
        # Extract config fields into instance variables
        self._initialize_from_config(storage_config)

        super().__init__(
            agent_id=agent_id,
            features=features,
            policy=policy,
            protocol=protocol,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
        )

    def _initialize_from_config(self, config: StorageConfig) -> None:
        """Extract and store needed config fields as instance variables.

        This avoids persisting the entire config object and makes explicit
        which fields are actually used by the storage system.

        Args:
            config: StorageConfig object to extract fields from
        """
        # Metadata
        self._bus = config.bus

        # Power bounds (for action space and feasibility)
        self._p_min_MW = config.p_min_MW
        self._p_max_MW = config.p_max_MW

        # Time step (for dynamics calculations)
        self._dt_h = config.dt_h

        # Phase model & spec
        self.phase_model = PhaseModel(config.phase_model)
        self.phase_spec = PhaseSpec().from_dict(config.phase_spec or {})
        check_phase_model_consistency(self.phase_model, self.phase_spec)

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """Initialize Action space for energy storage.

        Action:
            - c[0]: P_MW in [p_min_MW, p_max_MW]
            - c[1]: Q_MVAr (if Q control enabled via PowerLimits feature)

        Args:
            features: List of feature providers (may include PowerLimits for Q control)

        Returns:
            Initialized Action object
        """
        action = Action()

        # Check for PowerLimits feature for Q control
        limits = None
        for f in features:
            if isinstance(f, PowerLimits):
                limits = f
                break

        lows = [self._p_min_MW]
        highs = [self._p_max_MW]

        # Q control if PowerLimits feature is present with Q bounds
        if limits and limits.q_min_MVAr is not None and limits.q_max_MVAr is not None:
            lows.append(limits.q_min_MVAr)
            highs.append(limits.q_max_MVAr)

        action.set_specs(
            dim_c=len(lows),
            dim_d=0,
            ncats=0,
            range=(
                np.asarray(lows, dtype=np.float32),
                np.asarray(highs, dtype=np.float32),
            ),
        )

        # Initialize with zeros
        if len(lows) > 0:
            action.set_values(c=np.zeros(len(lows), dtype=np.float32))

        return action

    def set_action(self, action: Any) -> None:
        """Set action from Action object or compatible format.

        Args:
            action: Action object or numpy array/dict with action values
        """
        if isinstance(action, Action):
            # Extract action vector from Action object
            if action.c.size > 0:
                self.action.set_values(c=action.c)
            if action.d.size > 0:
                self.action.set_values(d=action.d)
        else:
            # Direct value (numpy array or dict)
            self.action.set_values(action)

    def set_state(self) -> None:
        """Apply action to update storage state.

        Modern HERON pattern: called by apply_action() after action is set.
        Updates power outputs, storage dynamics, and cost/safety metrics.
        """
        P_eff, Q_eff = self._update_power_outputs()
        self._update_storage_dynamics(P_eff)
        self._update_cost_safety()

    def apply_action(self) -> None:
        """Apply current action to update state.

        Overrides DeviceAgent.apply_action() to follow modern HERON pattern.
        """
        self.set_state()

    def sync_state_from_observed(self, observed_state: Any) -> None:
        """Synchronize state from external observations, then update cost/safety.

        Called after state features are updated from observations (e.g., from power flow).
        Recalculates cost and safety metrics based on the synchronized state.

        Args:
            observed_state: External observations to sync from
        """
        # Call parent to sync state features (ElectricalBasePh, StorageBlock, etc.)
        super().sync_state_from_observed(observed_state)

        # Update cost and safety based on new state
        self._update_cost_safety()

    def _update_power_outputs(self) -> Tuple[float, float]:
        """Read action.c, project through limits, update ElectricalBasePh.

        Returns:
            Tuple of (P_eff, Q_eff): effective active/reactive power after limits
        """
        electrical = self.electrical

        # Requested P/Q from continuous action; fall back to current electrical
        P_req = self.action.c[0] if self.action.c.size >= 1 else electrical.P_MW
        Q_req = self.action.c[1] if self.action.c.size >= 2 else electrical.Q_MVAr

        # If we have PowerLimits, project into feasible region; otherwise pass through
        if self.limits is not None:
            P_req = self.feasible_action(P_req)
            P_eff, Q_eff = self.limits.project_pq(P_req, Q_req)
        else:
            P_eff, Q_eff = P_req, Q_req

        # Apply updates to ElectricalBasePh
        elec_updates: Dict[str, Any] = {"P_MW": P_eff}
        if self.action.c.size >= 2:
            elec_updates["Q_MVAr"] = Q_eff

        self.state.update_feature(ElectricalBasePh.feature_name, **elec_updates)
        return P_eff, Q_eff

    def _update_storage_dynamics(self, P_MW: float) -> None:
        """Update SOC and degradation in StorageBlock based on active power.

        Args:
            P_MW: Active power (positive=charging, negative=discharging)
        """
        storage = self.storage
        dt = self._dt_h

        # SOC update: normalize energy change by capacity
        if P_MW >= 0.0:
            # Charging
            delta_e = P_MW * storage.ch_eff * dt
        else:
            # Discharging
            delta_e = P_MW / max(storage.dsc_eff, 1e-9) * dt

        delta_soc = delta_e / storage.e_capacity_MWh
        new_soc = np.clip(
            storage.soc + delta_soc,
            storage.soc_min,
            storage.soc_max,
        )
        storage.set_values(soc=new_soc)

    def _update_cost_safety(self) -> None:
        """Update per-step cost and safety penalties for the ESS.

        Cost: Degradation from energy throughput and equivalent cycles
        Safety: Apparent-power overload and SOC bounds violations
        """
        P = self.electrical.P_MW or 0.0
        Q = self.electrical.Q_MVAr or 0.0
        dt = self._dt_h

        if P >= 0.0:
            # Charging
            delta_e = P * self.storage.ch_eff * dt
        else:
            # Discharging
            delta_e = Q / max(self.storage.dsc_eff, 1e-9) * dt

        # Cost: degradation
        degr_cost_inc = 0.0
        e_throughput = self.storage.e_throughput_MWh + abs(delta_e)
        degr_cost_inc += self.storage.degr_cost_per_MWh * delta_e

        # Approximate equivalent full cycles from throughput
        eq_cycles = delta_e / self.storage.e_capacity_MWh
        degr_cost_inc += self.storage.degr_cost_per_cycle * eq_cycles

        degr_cost_cum = self.storage.degr_cost_cum + degr_cost_inc

        self.storage.set_values(
            e_throughput_MWh=e_throughput,
            degr_cost_cum=degr_cost_cum
        )

        cost = degr_cost_inc

        # Safety: SOC violations + power limit violations
        safety = self.storage.soc_violation()
        if self.limits is not None:
            violations = self.limits.feasible(P, Q)
            safety += np.sum(list(violations.values())) * dt

        # Sync cost and safety to the CostSafetyMetrics feature
        self.state.update_feature(
            CostSafetyMetrics.feature_name,
            cost=cost,
            safety=safety
        )

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward for ESS based on SOC, cost, and safety.

        Reward = SOC - cost - safety
        (maximize SOC while minimizing operating cost and safety violations)

        Args:
            local_state: State dict from proxy.get_local_state() with structure:
                {"StorageBlock": np.array([soc, ...]),
                 "CostSafetyMetrics": np.array([cost, safety]), ...}

        Returns:
            Reward value (higher is better)
        """
        soc_reward = 0.0
        cost_penalty = 0.0
        safety_penalty = 0.0

        # Extract SOC from StorageBlock (first element)
        if "StorageBlock" in local_state:
            storage_vec = local_state["StorageBlock"]
            soc_reward = float(storage_vec[0])

        # Extract cost/safety from CostSafetyMetrics
        if "CostSafetyMetrics" in local_state:
            metrics_vec = local_state["CostSafetyMetrics"]
            cost_penalty = float(metrics_vec[0])
            safety_penalty = float(metrics_vec[1])

        return soc_reward - cost_penalty - safety_penalty

    def feasible_action(self, P_req: float) -> float:
        """Clip action to feasible set based on SOC window and power limits.

        Args:
            P_req: Requested active power (positive=charging, negative=discharging)

        Returns:
            Clipped active power respecting SOC and power constraints
        """
        storage = self.storage
        dt = self._dt_h

        # Charging headroom (P > 0)
        if storage.soc < storage.soc_max:
            e_room_ch = (storage.soc_max - storage.soc) * storage.e_capacity_MWh
            p_max_soc = e_room_ch / max(storage.ch_eff * dt, 1e-9)
        else:
            p_max_soc = 0.0

        # Discharging headroom (P < 0)
        if storage.soc > storage.soc_min:
            e_room_dsc = (storage.soc - storage.soc_min) * storage.e_capacity_MWh
            p_min_soc = -e_room_dsc * storage.dsc_eff / max(dt, 1e-9)
        else:
            p_min_soc = 0.0

        p_min = max(p_min_soc, self._p_min_MW)
        p_max = min(p_max_soc, self._p_max_MW)

        return np.clip(P_req, p_min, p_max)

    @property
    def electrical(self) -> ElectricalBasePh:
        for f in self.state.features:
            if isinstance(f, ElectricalBasePh):
                return f

    @property
    def storage(self) -> StorageBlock:
        for f in self.state.features:
            if isinstance(f, StorageBlock):
                return f

    @property
    def status(self) -> StatusBlock:
        for f in self.state.features:
            if isinstance(f, StatusBlock):
                return f

    @property
    def limits(self) -> PowerLimits:
        for f in self.state.features:
            if isinstance(f, PowerLimits):
                return f

    @property
    def bus(self) -> str:
        return self._bus

    @property
    def name(self) -> str:
        return self.agent_id

    @property
    def capacity(self) -> float:
        return self.storage.e_capacity_MWh

    @property
    def p_ch_max(self) -> float:
        return self.storage.p_ch_max_MW

    @property
    def p_dsc_max(self) -> float:
        return self.storage.p_dsc_max_MW

    @property
    def soc(self) -> float:
        return self.storage.soc

    # ============================================
    # Distributed Mode Overrides
    # ============================================

    def _get_pandapower_device_type(self) -> str:
        """Get the PandaPower element type for this storage device.

        Returns:
            String identifier for PandaPower element type ('storage')
        """
        return 'storage'

    def _get_power_output(self) -> float:
        """Get current real power output in MW.

        Returns:
            Real power output (positive = charging, negative = discharging)
        """
        return float(self.electrical.P_MW) if self.electrical else 0.0

    def _get_reactive_power(self) -> float:
        """Get current reactive power output in MVAr.

        Returns:
            Reactive power output
        """
        return float(self.electrical.Q_MVAr or 0.0) if self.electrical else 0.0

    def _is_in_service(self) -> bool:
        """Check if storage device is currently in service.

        Returns:
            True if device is operational
        """
        # ESS doesn't track in_service via StatusBlock, always operational
        return True

    def __repr__(self) -> str:
        name = self.agent_id
        cap = self.storage.e_capacity_MWh if self.storage else 0.0
        pmin = self._p_min_MW
        pmax = self._p_max_MW
        return f"ESS(name={name}, capacity={cap}MWh, P∈[{pmin},{pmax}]MW)"

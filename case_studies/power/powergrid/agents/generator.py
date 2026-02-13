import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from powergrid.agents.device_agent import DeviceAgent, CostSafetyMetrics
from heron.core.action import Action
from heron.core.feature import FeatureProvider
from heron.core.policies import Policy
from heron.protocols.base import NoProtocol, Protocol
from heron.scheduling.tick_config import TickConfig
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.power_limits import PowerLimits
from powergrid.core.features.status import StatusBlock
from powergrid.utils.cost import cost_from_curve
from heron.utils.typing import AgentID
from powergrid.utils.phase import PhaseModel, PhaseSpec, check_phase_model_consistency


@dataclass
class GeneratorConfig:
    """Configuration for a dispatchable Generator.

    Note: Power limits (s_rated_MVA, p_min_MW, p_max_MW, q_min_MVAr, q_max_MVAr,
    pf_min_abs, derate_frac) are stored in the PowerLimits feature, not here.
    """
    bus: str

    # Phase model configuration
    phase_model: str = "balanced_1ph"
    phase_spec: Optional[Dict[str, Any]] = None

    # Voltage bounds for PV/SLACK control modes (not in features)
    vm_min_pu: Optional[float] = None
    vm_max_pu: Optional[float] = None
    va_min_deg: Optional[float] = None
    va_max_deg: Optional[float] = None

    # UC timings and costs
    startup_time_hr: Optional[int] = None
    shutdown_time_hr: Optional[int] = None
    startup_cost: float = 0.0
    shutdown_cost: float = 0.0

    # Economic parameters
    cost_curve_coefs: Sequence[float] = (0.0, 0.0, 0.0)  # a,b,c for a*(P^2)+b*P+c
    dt_h: float = 1.0  # hours per step (for costs)
    min_pf: Optional[float] = None   # for safety penalty

    # Metadata
    type: str = "fossil"
    source: Optional[str] = None  # "solar", "wind", or None for dispatchable
    control_mode: str = "PQ"      # "PQ", "PV", or "SLACK"


class Generator(DeviceAgent):
    """
    Dispatchable Generator following DeviceAgent's lifecycle.

    Example device_config:
    device_config = {
        "phase_model": "balanced_1ph",
        "phase_spec":  {
            "phases": "", 
            "has_neutral": False, 
            "earth_bond": False
        },
        "device_state_config": {
            "control_mode": "PQ",

            "s_rated_MVA": 10.0,
            "derate_frac": 1.0,
            "p_min_MW": 1.0,
            "p_max_MW": 8.0,
            "q_min_MVAr": -3.0,
            "q_max_MVAr":  3.0,
            "pf_min_abs":  0.8,
            "vm_min_pu": 0.95,
            "vm_max_pu": 1.05,
            "va_min_deg": -180.0,
            "va_max_deg": 180.0,

            "startup_time_hr":  2,
            "shutdown_time_hr": 1,
            "startup_cost":  5.0,
            "shutdown_cost": 1.0,

            "cost_curve_coefs": [0.01, 1.0, 0.0],
            "dt_h": 1.0,
            "min_pf": 0.8
        }
    }
    """

    def __init__(
        self,
        agent_id: AgentID,
        generator_config: GeneratorConfig,
        features: List[FeatureProvider] = [],
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None,
    ):
        # Extract config fields into instance variables
        self._initialize_from_config(generator_config)

        super().__init__(
            agent_id=agent_id,
            features=features,
            policy=policy,
            protocol=protocol,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
        )

    def _initialize_from_config(self, config: GeneratorConfig) -> None:
        """Extract and store needed config fields as instance variables.

        This avoids persisting the entire config object and makes explicit
        which fields are actually used by the generator.

        Args:
            config: GeneratorConfig object to extract fields from
        """
        # Metadata
        self._bus = config.bus
        self._control_mode = config.control_mode
        self._type = config.type
        self._source = config.source

        # Voltage bounds (for PV/SLACK action space)
        self._vm_min_pu = config.vm_min_pu
        self._vm_max_pu = config.vm_max_pu
        self._va_min_deg = config.va_min_deg
        self._va_max_deg = config.va_max_deg

        # Unit commitment parameters
        self._startup_time_hr = config.startup_time_hr
        self._shutdown_time_hr = config.shutdown_time_hr
        self._startup_cost = config.startup_cost
        self._shutdown_cost = config.shutdown_cost

        # Economic parameters
        self._cost_curve_coefs = config.cost_curve_coefs
        self._dt_h = config.dt_h
        self._min_pf = config.min_pf

        # Phase model & spec
        self.phase_model = PhaseModel(config.phase_model)
        self.phase_spec = PhaseSpec().from_dict(config.phase_spec or {})
        check_phase_model_consistency(self.phase_model, self.phase_spec)

        # Unit commitment cost tracking (updated during _update_uc_status)
        self._uc_cost_step = 0.0

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """
        Initialize Action space depending on generator control model.

        PQ model:
        (1a) P-only:
            - control_mode == "PQ"
            - q_min_MVAr and q_max_MVAr are None
            - pf_min_abs is None or s_rated_MVA is None
            => action.c = [P]

        (1b) P+Q with explicit limits:
            - control_mode == "PQ"
            - q_min_MVAr and q_max_MVAr are BOTH not None
            => action.c = [P, Q] with box limits

        (1c) P+Q with PF/S capability:
            - control_mode == "PQ"
            - q_min_MVAr and q_max_MVAr are None
            - pf_min_abs and s_rated_MVA are BOTH not None
            => action.c = [P, Q], Q limits derived from S & pf_min_abs
                (further refined by PowerLimits.project_pq)

        PV model:
        (2) control_mode == "PV"
            => action.c = [P, Vm_pu]

        Slack model:
        (3) control_mode == "SLACK"
            => action.c = [Vm_pu, Va_deg]

        UC head d[0] in {0=turn_off, 1=turn_on}.
        """
        action = Action()

        # Need to get limits from features parameter
        limits = None
        for f in features:
            if isinstance(f, PowerLimits):
                limits = f
                break

        if limits is None:
            raise ValueError("PowerLimits feature required for action initialization")

        mode = (self._control_mode or "PQ").upper()

        lows: list[float] = []
        highs: list[float] = []

        if mode == "PQ":
            # --- P bounds (always present in PQ) ---
            lows.append(limits.p_min_MW)
            highs.append(limits.p_max_MW)

            has_q_box = (
                limits.q_min_MVAr is not None and
                limits.q_max_MVAr is not None
            )
            has_pf_cap = (
                limits.pf_min_abs is not None and
                limits.s_rated_MVA is not None
            )

            if has_q_box:
                # (1b) P+Q with explicit limits
                lows.append(limits.q_min_MVAr)
                highs.append(limits.q_max_MVAr)

            elif has_pf_cap:
                # (1c) P+Q with PF capability curve:
                # approximate symmetric Q box from S & pf_min_abs
                S = limits.s_rated_MVA * (limits.derate_frac or 1.0)
                pf = limits.pf_min_abs
                q_max_mag = S * math.sqrt(max(0.0, 1.0 - pf * pf))
                lows.append(-q_max_mag)
                highs.append(q_max_mag)

            else:
                # (1a) P-only: Q is not a control variable (e.g., fixed or 0)
                pass

        elif mode == "PV":
            # (2) PV: action = [P, Vm]
            lows.append(limits.p_min_MW)
            highs.append(limits.p_max_MW)

            vm_min = float(self._vm_min_pu)
            vm_max = float(self._vm_max_pu)
            lows.append(vm_min)
            highs.append(vm_max)

        elif mode == "SLACK":
            # (3) Slack: action = [Vm, Va]
            lows.append(self._vm_min_pu)
            highs.append(self._vm_max_pu)
            lows.append(np.deg2rad(self._va_min_deg))
            highs.append(np.deg2rad(self._va_max_deg))

        else:
            raise ValueError(f"Unknown generator control_mode={mode!r}")

        # --- Unit commitment discrete head ---
        have_uc = (
            self._startup_time_hr is not None or
            self._shutdown_time_hr is not None
        )

        action.set_specs(
            dim_c=len(lows),
            dim_d=(1 if have_uc else 0),
            ncats=(2 if have_uc else 0),
            range=(np.asarray(lows, np.float32), np.asarray(highs, np.float32)),
        )

        # Initialize with zeros
        if len(lows) > 0:
            action.set_values(c=np.zeros(len(lows), dtype=np.float32))
        if have_uc:
            action.set_values(d=np.array([1], dtype=np.int32))  # Default to "on"

        return action

    def set_action(self, action: Any) -> None:
        """Set action from Action object or compatible format.

        Args:
            action: Action object or numpy array with action values
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
        """Apply action to update generator state.

        Modern HERON pattern: called by apply_action() after action is set.
        Updates UC status, power outputs, and applies cost/safety metrics.
        """
        self._update_uc_status()
        self._update_power_outputs()
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
        # Call parent to sync state features (ElectricalBasePh, StatusBlock, etc.)
        super().sync_state_from_observed(observed_state)

        # Update cost and safety based on new state
        self._update_cost_safety()

    def _update_uc_status(self) -> None:
        """Advance UC lifecycle in StatusBlock using d[0] ∈ {off, on}.
        When a transition completes, apply the corresponding UC cost for
        this step.
        """
        self._uc_cost_step = 0.0

        # 1) Base timing: advance time in current state
        t_in_state_s = self.status.t_in_state_s or 0.0
        t_to_next_s = self.status.t_to_next_s
        progress_frac = self.status.progress_frac
        dt = self._dt_h
        t_in_state_s += dt

        # 2) UC command: 0=request_off, 1=request_on
        uc_cmd = None
        if self.action.dim_d > 0 and self.action.d.size > 0:
            uc_cmd = int(self.action.d[0])

        t_start = self._startup_time_hr
        t_stop = self._shutdown_time_hr

        # 3) Handle UC command
        #    - If not in transition: react to UC
        #    - startup_time_hr / shutdown_time_hr == 0 => immediate transition
        state = self.status.state
        if uc_cmd is not None and t_to_next_s is None:
            # Request OFF from ONLINE
            if state == "online" and uc_cmd == 0:
                if t_stop <= 0:
                    # Immediate shutdown
                    state = "offline"
                    t_in_state_s = 0.0
                    t_to_next_s = None
                    progress_frac = None
                    self._uc_cost_step = self._shutdown_cost
                else:
                    # Begin shutdown transition
                    state = "shutdown"
                    t_in_state_s = 0.0
                    t_to_next_s = t_stop * dt
                    progress_frac = 0.0

            # Request ON from OFFLINE
            elif state == "offline" and uc_cmd == 1:
                if t_start <= 0:
                    # Immediate startup
                    state = "online"
                    t_in_state_s = 0.0
                    t_to_next_s = None
                    progress_frac = None
                    self._uc_cost_step = self._startup_cost
                else:
                    # Begin startup transition
                    state = "startup"
                    t_in_state_s = 0.0
                    t_to_next_s = t_start * dt
                    progress_frac = 0.0

        # 4) Progress transitional states (startup/shutdown with timers)
        if t_to_next_s is not None and state in ("startup", "shutdown"):
            t_to_next_s = max(0.0, t_to_next_s - dt)

            total = t_start if state == "startup" else t_stop
            if total > 0:
                denom = max(total * dt, 1e-9)
                progress_frac = 1.0 - t_to_next_s / denom

            # Finish transition
            if t_to_next_s == 0.0:
                if state == "startup":
                    state = "online"
                    t_in_state_s = 0.0
                    t_to_next_s = None
                    progress_frac = None
                    self._uc_cost_step = self._startup_cost
                elif state == "shutdown":
                    state = "offline"
                    t_in_state_s = 0.0
                    t_to_next_s = None
                    progress_frac = None
                    self._uc_cost_step = self._shutdown_cost

        # Apply updates to features
        status_updates: Dict[str, Any] = {
            "state": state,
            "t_in_state_s": t_in_state_s,
            "t_to_next_s": t_to_next_s,
            "progress_frac": progress_frac,
        }

        self.state.update_feature(StatusBlock.feature_name, **status_updates)

    def _update_power_outputs(self) -> None:
        """Apply continuous P/Q control from action.c, projected to limits.
        """
        electrical = self.electrical

        # Continuous P/Q control
        P_req = self.action.c[0] if self.action.c.size >=1 else electrical.P_MW
        Q_req = self.action.c[1] if self.action.c.size >=2 else electrical.Q_MVAr
        P_eff, Q_eff = self.limits.project_pq(P_req, Q_req)

        # Apply updates to features
        elec_updates: Dict[str, Any] = {"P_MW": P_eff}
        if self.action.c.size >= 2:
            elec_updates["Q_MVAr"] = Q_eff

        self.state.update_feature(ElectricalBasePh.feature_name, **elec_updates)

    def _update_cost_safety(self) -> None:
        """Economic cost + S/PF penalties + UC start/stop cost."""
        P = self.electrical.P_MW or 0.0
        Q = self.electrical.Q_MVAr or 0.0
        on = 1.0 if self.status.state == "online" else 0.0
        dt = self._dt_h

        # Cost
        fuel_cost = cost_from_curve(P, self._cost_curve_coefs)
        uc_cost = self._uc_cost_step
        cost = fuel_cost * on * dt + uc_cost

        # Safety violations
        violations = self.limits.feasible(P, Q)
        safety = np.sum(list(violations.values())) * on * dt

        # Sync cost and safety to the CostSafetyMetrics feature
        self.state.update_feature(
            CostSafetyMetrics.feature_name,
            cost=cost,
            safety=safety
        )

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward for Generator based on cost and safety.

        Reward = -cost - safety
        (minimize fuel cost, UC cost, and safety violations)

        Args:
            local_state: State dict from proxy.get_local_state() with structure:
                {"ElectricalBasePh": np.array([P_MW, Q_MVAr, ...]),
                 "StatusBlock": np.array([...]),
                 "PowerLimits": np.array([...]),
                 "CostSafetyMetrics": np.array([cost, safety]), ...}

        Returns:
            Reward value (higher is better)
        """
        cost_penalty = 0.0
        safety_penalty = 0.0

        # Extract cost/safety from CostSafetyMetrics
        if "CostSafetyMetrics" in local_state:
            metrics_vec = local_state["CostSafetyMetrics"]
            cost_penalty = float(metrics_vec[0])
            safety_penalty = float(metrics_vec[1])

        return -cost_penalty - safety_penalty

    @property
    def electrical(self) -> ElectricalBasePh:
        for f in self.state.features:
            if isinstance(f, ElectricalBasePh):
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
    def source(self) -> Optional[str]:
        """Get renewable source type ('solar', 'wind', or None for dispatchable)."""
        return self._source

    # ============================================
    # Distributed Mode Overrides
    # ============================================

    def _get_pandapower_device_type(self) -> str:
        """Get the PandaPower element type for this generator."""
        return 'sgen'

    def _get_power_output(self) -> float:
        """Get current real power output in MW."""
        return float(self.electrical.P_MW) if self.electrical else 0.0

    def _get_reactive_power(self) -> float:
        """Get current reactive power output in MVAr."""
        return float(self.electrical.Q_MVAr or 0.0) if self.electrical else 0.0

    def _is_in_service(self) -> bool:
        """Check if generator is currently in service."""
        return bool(self.status.in_service) if self.status else True

    def __repr__(self) -> str:
        s = self.limits.s_rated_MVA
        pmin = self.limits.p_min_MW
        pmax = self.limits.p_max_MW
        return f"DG(name={self.agent_id}, S={s}MVA, P∈[{pmin},{pmax}]MW)"

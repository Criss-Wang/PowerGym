import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.policies import Policy
from powergrid.core.protocols import Protocol, NoProtocol
from powergrid.core.state import DeviceState, PhaseModel, PhaseSpec
from powergrid.core.action import Action
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.connection import PhaseConnection
from powergrid.features.status import StatusBlock
from powergrid.features.generator_limits import GeneratorLimits
from powergrid.utils.cost import cost_from_curve
from powergrid.utils.safety import pf_penalty, s_over_rating
from powergrid.utils.typing import float_if_not_none


@dataclass
class GeneratorConfig:
    """Configuration for a dispatchable Generator."""
    bus: str

    # capability & constraints
    s_rated_MVA: Optional[float] = None
    p_min_MW: float = 0.0
    p_max_MW: float = 0.0
    q_min_MVAr: Optional[float] = None
    q_max_MVAr: Optional[float] = None
    pf_min_abs: Optional[float] = None
    derate_frac: float = 1.0

    # UC timings (in timesteps); if both None -> no UC head
    startup_time: Optional[int] = None
    shutdown_time: Optional[int] = None
    startup_cost: float = 0.0
    shutdown_cost: float = 0.0

    # economics & sim
    cost_curve_coefs: Sequence[float] = (0.0, 0.0, 0.0)  # a,b,c for a*(P^2)+b*P+c
    dt_h: float = 1.0  # hours per step (for costs)
    min_pf: Optional[float] = None   # for safety penalty
    type: str = "fossil"

    # phase context
    phase_model: str = "balanced_1ph"
    phase_spec: Optional[Dict[str, Any]] = None


class Generator(DeviceAgent):
    """
    Dispatchable Generator following DeviceAgent's lifecycle.

    Example device_config:
    {
      "device_state_config": {
        "phase_model": "BALANCED_1PH" | "THREE_PHASE",
        "phase_spec":  None or {
            "phases": "ABC", "has_neutral": False, "earth_bond": True
        },

        "s_rated_MVA": 10.0,
        "derate_frac": 1.0,
        "p_min_MW": 1.0,
        "p_max_MW": 8.0,
        "q_min_MVAr": -3.0,
        "q_max_MVAr":  3.0,
        "pf_min_abs":  0.8,

        "startup_time":  2,
        "shutdown_time": 1,
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
        *,
        agent_id: Optional[str] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any],
    ):
        super().__init__(
            agent_id=agent_id,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

        generator_config = self.config.device_state_config
        self._generator_config = GeneratorConfig(
            bus=generator_config.get("bus", ""),
            s_rated_MVA=float_if_not_none(generator_config.get("s_rated_MVA", None)),
            p_min_MW=float_if_not_none(generator_config.get("p_min_MW", 0.0)),
            p_max_MW=float_if_not_none(generator_config.get("p_max_MW", 0.0)),
            q_min_MVAr=float_if_not_none(generator_config.get("q_min_MVAr", None)),
            q_max_MVAr=float_if_not_none(generator_config.get("q_max_MVAr", None)),
            pf_min_abs=float_if_not_none(generator_config.get("pf_min_abs", None)),
            derate_frac=float_if_not_none(generator_config.get("derate_frac", 1.0)),

            # economics & UC params
            cost_curve_coefs=generator_config.get("cost_curve_coefs", (0.0, 0.0, 0.0)),
            dt_h=float_if_not_none(generator_config.get("dt_h", 1.0)),
            startup_time=float_if_not_none(generator_config.get("startup_time", None)),
            shutdown_time=float_if_not_none(generator_config.get("shutdown_time", None)),
            startup_cost=float_if_not_none(generator_config.get("startup_cost", 0.0)),
            shutdown_cost=float_if_not_none(generator_config.get("shutdown_cost", 0.0)),
            min_pf=float_if_not_none(generator_config.get("min_pf", None)),

            phase_model=generator_config.get("phase_model", None),
            phase_spec=generator_config.get("phase_spec", None),
        )

    def set_action_space(self) -> None:
        """
        Define Action:
          - continuous P (and optionally Q),
          - optional discrete UC command d[0] in {off, on}.
        """
        # determine action ranges
        lows = [self._generator_config.p_min_MW]
        highs = [self._generator_config.p_max_MW]
        # if control reactive power
        if self._generator_config.q_min_MVAr is not None and self._generator_config.q_max_MVAr is not None:
            lows.append(self._generator_config.q_min_MVAr)
            highs.append(self._generator_config.q_max_MVAr)

        # if control unit committement
        have_uc = self._generator_config.startup_time is not None or self._generator_config.shutdown_time is not None

        # Set action specs & sample
        self.action.set_specs(
            dim_c=len(lows),
            dim_d=(1 if have_uc else 0),
            ncats=(2 if have_uc else 0),
            range=(np.asarray(lows, np.float32), np.asarray(highs, np.float32)),
            masks=(None if not have_uc else [np.array([True, True], bool)]),
        )
        self.action.sample()

    def set_device_state(self) -> None:
        # phase model & spec
        pm = PhaseModel(self._generator_config.phase_model)
        ps = PhaseSpec(
            self._generator_config.phase_spec.get("phases", "ABC"),
            self._generator_config.phase_spec.get("has_neutral", False),
            self._generator_config.phase_spec.get("earth_bond", True),
        ) if self._generator_config.phase_spec is not None else None

        # Electrical telemetry
        eletrical_telemetry = ElectricalBasePh(
            P_MW=0.0,
            Q_MVAr=(
                0.0
                if self._generator_config.q_min_MVAr is not None and self._generator_config.q_max_MVAr is not None
                else None
            ),
        )

        # Status / UC lifecycle
        status = StatusBlock(
            in_service=True,
            out_service=None,
            state="online",
            states_vocab=["offline", "startup", "online", "shutdown", "fault"],
            emit_state_one_hot=True,
            emit_state_index=False,
        )

        # Connection
        conn = "ON" if pm == PhaseModel.BALANCED_1PH else "ABC"
        connection = PhaseConnection(
            phase_model=pm, 
            phase_spec=ps, 
            connection=conn
        )

        # Capability / limits
        generatorlimits = GeneratorLimits(
            s_rated_MVA=self._generator_config.s_rated_MVA,
            derate_frac=self._generator_config.derate_frac,
            p_min_MW=self._generator_config.p_min_MW,
            p_max_MW=self._generator_config.p_max_MW,
            q_min_MVAr=self._generator_config.q_min_MVAr,
            q_max_MVAr=self._generator_config.q_max_MVAr,
            pf_min_abs=self._generator_config.pf_min_abs,
        )


        self.state = DeviceState(
            phase_model=pm,
            phase_spec=ps,
            features=[
                eletrical_telemetry,
                status,
                connection,
                generatorlimits,
            ],
            prefix_names=False,
        )
        self.state._apply_phase_context_to_features_()


    def reset_device(self, *args, **kwargs) -> None:
        """Zero P/Q, set online (or offline if you pass online=False)."""
        in_service = bool(kwargs.get("in_service", True))
        online = bool(kwargs.get("online", True))

        # reset electricals
        electrical = self.electrical
        electrical.P_MW = 0.0
        if self.action.dim_c >= 2:
            electrical.Q_MVAr = 0.0

        # reset status block
        status_block = self.status
        status_block.state = "online" if online else "offline"
        status_block.in_service = in_service
        status_block.out_service = None
        status_block.t_in_state_s = 0.0
        status_block.t_to_next_s = None
        status_block.progress_frac = None

        # reset cost and safety
        self.cost = 0.0
        self.safety = 0.0
        self._uc_cost_step = 0.0

        if self.action.dim_d:
            self.action.masks = [np.array([True, True], dtype=bool)]

    def update_state(self, *args, **kwargs) -> None:
        """
        UC via discrete head (d[0]: 0=off, 1=on) with simple timers, then apply P/Q.
        If not ONLINE, the unit produces no power (P=Q=0). When a transition
        completes, apply the corresponding UC cost for this step.
        """
        status_block = self.status
        electrical = self.electrical
        generatorlimits = self.limits

        # UC timing & state progression (startup/shutdown only)
        self._uc_cost_step = 0.0

        # Advance time-in-state (reuse seconds field as "steps * dt_h" proxy)
        status_block.t_in_state_s = (
            0.0 
            if status_block.t_in_state_s is None 
            else float(status_block.t_in_state_s + self._generator_config.dt_h)
        )

        # Read discrete UC command if present
        uc_cmd = None
        if self.action.dim_d > 0 and self.action.d.size > 0:
            uc_cmd = int(self.action.d[0])  # 0=request_off, 1=request_on

        t_start = int(self._generator_config.startup_time or 0)
        t_stop = int(self._generator_config.shutdown_time or 0)

        # Trigger transitions only when not already transitioning
        if uc_cmd is not None and status_block.t_to_next_s is None:
            if status_block.state == "online" and uc_cmd == 0 and t_stop > 0:
                status_block.state = "shutdown"
                status_block.t_to_next_s = t_stop * self._generator_config.dt_h
                status_block.progress_frac = 0.0
            elif status_block.state == "offline" and uc_cmd == 1 and t_start > 0:
                status_block.state = "startup"
                status_block.t_to_next_s = t_start * self._generator_config.dt_h
                status_block.progress_frac = 0.0

        # Count down transitional states
        if status_block.t_to_next_s is not None:
            status_block.t_to_next_s = max(0.0, float(status_block.t_to_next_s - self._generator_config.dt_h))
            total = t_start if status_block.state == "startup" else t_stop
            if total > 0:
                denom = max(total * self._generator_config.dt_h, 1e-9)
                status_block.progress_frac = float(1.0 - status_block.t_to_next_s / denom)

            # Finish transition at zero
            if status_block.state == "startup" and status_block.t_to_next_s == 0.0:
                status_block.state = "online"
                status_block.t_in_state_s = 0.0
                status_block.t_to_next_s = None
                status_block.progress_frac = None
                self._uc_cost_step = self._generator_config.startup_cost

            elif status_block.state == "shutdown" and status_block.t_to_next_s == 0.0:
                status_block.state = "offline"
                status_block.t_in_state_s = 0.0
                status_block.t_to_next_s = None
                status_block.progress_frac = None
                self._uc_cost_step = self._generator_config.shutdown_cost

        # Continuous P/Q control (projected); zero when not online
        P_req = (
            float(self.action.c[0])
            if self.action.c.size >= 1 else float(electrical.P_MW or 0.0)
        )
        Q_req = (
            float(self.action.c[1])
            if self.action.c.size >= 2 else float(electrical.Q_MVAr or 0.0)
        )

        if status_block.state != "online":
            P_req, Q_req = 0.0, 0.0

        P_eff, Q_eff = generatorlimits.project_pq(P_req, Q_req)
        electrical.P_MW = P_eff
        if self.action.c.size >= 2:
            electrical.Q_MVAr = Q_eff

        # Defensive clamp on all child features
        self.state.clamp_()

    def update_cost_safety(self, *args, **kwargs) -> None:
        """Economic cost + S/PF penalties + UC start/stop cost."""
        electrical = self.electrical
        P = float(electrical.P_MW or 0.0)
        Q = float(electrical.Q_MVAr or 0.0)

        on = 1.0 if self.status.state == "online" else 0.0
        self.cost = (
            on * cost_from_curve(P, self._generator_config.cost_curve_coefs) * self._generator_config.dt_h
            + getattr(self, "_uc_cost_step", 0.0) * self._generator_config.dt_h
        )

        safety = 0.0
        S = self.limits.s_rated_MVA
        if self.action.dim_c >= 2 and S is not None:
            safety += s_over_rating(P, Q, S)
            safety += pf_penalty(P, Q, self._generator_config.min_pf)
        self.safety = safety * self._generator_config.dt_h

    def feasible_action(self) -> None:
        """Optionally clip action.c to feasible set before use."""
        if self.action.dim_c:
            P = float(self.action.c[0])
            Q = float(self.action.c[1]) if self.action.c.size >= 2 else 0.0
            P2, Q2 = self.limits.project_pq(P, Q)
            self.action.c[0] = P2
            if self.action.c.size >= 2:
                self.action.c[1] = Q2

    def _advance_uc(self) -> None:
        """Advance UC lifecycle in StatusBlock using d[0] ∈ {off, on}."""
        status_block = self.status
        a = int(self.action.d[0]) if self.action.d.size else 1
        self._uc_cost_step = 0.0

        status_block.t_in_state_s = (
            0.0 
            if status_block.t_in_state_s is None 
            else float(status_block.t_in_state_s + self._generator_config.dt_h)
        )

        t_start = int(self._generator_config.startup_time or 0)
        t_stop = int(self._generator_config.shutdown_time or 0)

        if status_block.state == "online" and a == 0 and t_stop > 0:
            status_block.state = "shutdown"
            status_block.t_to_next_s = t_stop * self._generator_config.dt_h
            status_block.progress_frac = 0.0
        elif status_block.state == "offline" and a == 1 and t_start > 0:
            status_block.state = "startup"
            status_block.t_to_next_s = t_start * self._generator_config.dt_h
            status_block.progress_frac = 0.0

        if status_block.t_to_next_s is not None:
            status_block.t_to_next_s = max(0.0, status_block.t_to_next_s - self._generator_config.dt_h)
            total = t_start if status_block.state == "startup" else t_stop
            if total > 0:
                denom = max(total * self._generator_config.dt_h, 1e-9)
                status_block.progress_frac = float(1.0 - status_block.t_to_next_s / denom)

        if status_block.state == "startup" and status_block.t_to_next_s == 0.0:
            status_block.state = "online"
            status_block.t_in_state_s = 0.0
            status_block.t_to_next_s = None
            status_block.progress_frac = None
            self._uc_cost_step = self._generator_config.startup_cost

        if status_block.state == "shutdown" and status_block.t_to_next_s == 0.0:
            status_block.state = "offline"
            status_block.t_in_state_s = 0.0
            status_block.t_to_next_s = None
            status_block.progress_frac = None
            self._uc_cost_step = self._generator_config.shutdown_cost

    @property
    def electrical(self) -> ElectricalBasePh:
        for f in self.state.features:
            if isinstance(f, ElectricalBasePh):
                return f
        raise ValueError("ElectricalBasePh feature not found")

    @property
    def status(self) -> StatusBlock:
        for f in self.state.features:
            if isinstance(f, StatusBlock):
                return f
        raise ValueError("StatusBlock feature not found")

    @property
    def limits(self) -> GeneratorLimits:
        for f in self.state.features:
            if isinstance(f, GeneratorLimits):
                return f
        raise ValueError("GeneratorLimits feature not found")

    @property
    def bus(self) -> str:
        return self._generator_config.bus

    def __repr__(self) -> str:
        name = self.config.name
        s = self.limits.s_rated_MVA
        pmin = self.limits.p_min_MW
        pmax = self.limits.p_max_MW
        return f"DG(name={name}, S={s}MVA, P∈[{pmin},{pmax}]MW)"

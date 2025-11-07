import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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


@dataclass
class GeneratorConfig:
    """Configuration for a dispatchable Generator."""
    name: str
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
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: Optional[PhaseSpec] = None



class Generator(DeviceAgent):
    """
    Dispatchable Generator following DeviceAgent's lifecycle.

    Example device_config:
    {
      "name": "G1",
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
        device_config: GeneratorConfig,
    ):
        super().__init__(
            agent_id=agent_id,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def set_action_space(self) -> None:
        """
        Define Action:
          - continuous P (and optionally Q),
          - optional discrete UC command d[0] in {off, on}.
        """
        cfg = self.config.get("device_state_config", {})
        p_min = float(cfg.get("p_min_MW", 0.0))
        p_max = float(cfg.get("p_max_MW", 0.0))

        # if control reactive power
        has_q = (
            ("q_min_MVAr" in cfg)
            and ("q_max_MVAr" in cfg)
            and (cfg["q_min_MVAr"] is not None)
            and (cfg["q_max_MVAr"] is not None)
        )

        lows = [p_min]
        highs = [p_max]
        if has_q:
            lows.append(float(cfg["q_min_MVAr"]))
            highs.append(float(cfg["q_max_MVAr"]))

        # if control unit committement
        have_uc = (
            cfg.get("startup_time") is not None
            or cfg.get("shutdown_time") is not None
        )

        self.action = Action().set_specs(
            dim_c=len(lows),
            dim_d=(1 if have_uc else 0),
            ncats=(2 if have_uc else 0),
            range=(np.asarray(lows, np.float32), np.asarray(highs, np.float32)),
            masks=(None if not have_uc else [np.array([True, True], bool)]),
        )

        # seed a valid sample so gym space construction is happy
        self.action.sample()

    def set_device_state(self, config: Dict[str, Any]) -> None:
        """Compose features, install into DeviceState, enforce phase context."""
        pm_raw = config.get("phase_model", PhaseModel.BALANCED_1PH)
        pm = pm_raw if isinstance(pm_raw, PhaseModel) else PhaseModel(pm_raw)

        psd = config.get("phase_spec", None)
        if psd is None or isinstance(psd, PhaseSpec):
            ps = psd
        else:
            ps = PhaseSpec(
                psd.get("phases", "ABC"),
                psd.get("has_neutral", False),
                psd.get("earth_bond", True),
            )

        feats: List[object] = []

        # Electrical telemetry
        feats.append(
            ElectricalBasePh(
                P_MW=0.0,
                Q_MVAr=(
                    0.0
                    if (
                        "q_min_MVAr" in config
                        and "q_max_MVAr" in config
                        and config["q_min_MVAr"] is not None
                        and config["q_max_MVAr"] is not None
                    )
                    else None
                ),
            )
        )

        # Status / UC lifecycle
        feats.append(
            StatusBlock(
                in_service=True,
                out_service=None,
                state="online",
                states_vocab=["offline", "startup", "online", "shutdown", "fault"],
                emit_state_one_hot=True,
                emit_state_index=False,
            )
        )

        # Connection
        conn = "ON" if pm == PhaseModel.BALANCED_1PH else "ABC"
        feats.append(
            PhaseConnection(
                phase_model=pm, 
                phase_spec=ps, 
                connection=conn
            )
        )

        # Capability / limits
        feats.append(
            GeneratorLimits(
                s_rated_MVA=config.get("s_rated_MVA"),
                derate_frac=float(config.get("derate_frac", 1.0)),
                p_min_MW=float(config.get("p_min_MW", 0.0)),
                p_max_MW=float(config.get("p_max_MW", 0.0)),
                q_min_MVAr=(
                    None
                    if config.get("q_min_MVAr") is None
                    else float(config["q_min_MVAr"])
                ),
                q_max_MVAr=(
                    None
                    if config.get("q_max_MVAr") is None
                    else float(config["q_max_MVAr"])
                ),
                pf_min_abs=(
                    None
                    if config.get("pf_min_abs") is None
                    else float(config["pf_min_abs"])
                ),
            )
        )

        self.state = DeviceState(
            phase_model=pm,
            phase_spec=ps,
            features=feats,
            prefix_names=False,
        )
        self.state._apply_phase_context_to_features_()

        # economics & UC params
        self._cost_coefs: Sequence[float] = config.get(
            "cost_curve_coefs", (0.0, 0.0, 0.0)
        )
        self._dt_h: float = float(config.get("dt_h", 1.0))
        self._startup_time: Optional[int] = config.get("startup_time")
        self._shutdown_time: Optional[int] = config.get("shutdown_time")
        self._startup_cost: float = float(config.get("startup_cost", 0.0))
        self._shutdown_cost: float = float(config.get("shutdown_cost", 0.0))
        self._min_pf: Optional[float] = config.get("min_pf")

    def reset_device(self, *args, **kwargs) -> None:
        """Zero P/Q, set online (or offline if you pass online=False)."""
        in_service = bool(kwargs.get("in_service", True))
        online = bool(kwargs.get("online", True))

        elec = self._electrical()
        elec.P_MW = 0.0
        if self.action.dim_c >= 2:
            elec.Q_MVAr = 0.0

        s = self._status()
        s.state = "online" if online else "offline"
        s.in_service = in_service
        s.out_service = None
        s.t_in_state_s = 0.0
        s.t_to_next_s = None
        s.progress_frac = None

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
        s = self._status()
        elec = self._electrical()
        limits = self._limits()

        # UC timing & state progression (startup/shutdown only)
        self._uc_cost_step = 0.0

        # Advance time-in-state (reuse seconds field as "steps * dt_h" proxy)
        s.t_in_state_s = (
            0.0 if s.t_in_state_s is None else float(s.t_in_state_s + self._dt_h)
        )

        # Read discrete UC command if present
        uc_cmd = None
        if self.action.dim_d > 0 and self.action.d.size > 0:
            uc_cmd = int(self.action.d[0])  # 0=request_off, 1=request_on

        t_start = int(self._startup_time or 0)
        t_stop = int(self._shutdown_time or 0)

        # Trigger transitions only when not already transitioning
        if uc_cmd is not None and s.t_to_next_s is None:
            if s.state == "online" and uc_cmd == 0 and t_stop > 0:
                s.state = "shutdown"
                s.t_to_next_s = t_stop * self._dt_h
                s.progress_frac = 0.0
            elif s.state == "offline" and uc_cmd == 1 and t_start > 0:
                s.state = "startup"
                s.t_to_next_s = t_start * self._dt_h
                s.progress_frac = 0.0

        # Count down transitional states
        if s.t_to_next_s is not None:
            s.t_to_next_s = max(0.0, float(s.t_to_next_s - self._dt_h))
            total = t_start if s.state == "startup" else t_stop
            if total > 0:
                denom = max(total * self._dt_h, 1e-9)
                s.progress_frac = float(1.0 - s.t_to_next_s / denom)

            # Finish transition at zero
            if s.state == "startup" and s.t_to_next_s == 0.0:
                s.state = "online"
                s.t_in_state_s = 0.0
                s.t_to_next_s = None
                s.progress_frac = None
                self._uc_cost_step = float(self._startup_cost)

            elif s.state == "shutdown" and s.t_to_next_s == 0.0:
                s.state = "offline"
                s.t_in_state_s = 0.0
                s.t_to_next_s = None
                s.progress_frac = None
                self._uc_cost_step = float(self._shutdown_cost)

        # Continuous P/Q control (projected); zero when not online
        P_req = (
            float(self.action.c[0])
            if self.action.c.size >= 1 else float(elec.P_MW or 0.0)
        )
        Q_req = (
            float(self.action.c[1])
            if self.action.c.size >= 2 else float(elec.Q_MVAr or 0.0)
        )

        if s.state != "online":
            P_req, Q_req = 0.0, 0.0

        P_eff, Q_eff = limits.project_pq(P_req, Q_req)
        elec.P_MW = P_eff
        if self.action.c.size >= 2:
            elec.Q_MVAr = Q_eff

        # Defensive clamp on all child features
        self.state.clamp_()

    def update_cost_safety(self, *args, **kwargs) -> None:
        """Economic cost + S/PF penalties + UC start/stop cost."""
        elec = self._electrical()
        P = float(elec.P_MW or 0.0)
        Q = float(elec.Q_MVAr or 0.0)

        on = 1.0 if self._status().state == "online" else 0.0
        self.cost = (
            on * cost_from_curve(P, self._cost_coefs) * self._dt_h
            + getattr(self, "_uc_cost_step", 0.0) * self._dt_h
        )

        safety = 0.0
        S = self._limits().s_rated_MVA
        if self.action.dim_c >= 2 and S is not None:
            safety += s_over_rating(P, Q, S)
            safety += pf_penalty(P, Q, self._min_pf)
        self.safety = safety * self._dt_h

    def feasible_action(self) -> None:
        """Optionally clip action.c to feasible set before use."""
        if self.action.dim_c:
            P = float(self.action.c[0])
            Q = float(self.action.c[1]) if self.action.c.size >= 2 else 0.0
            P2, Q2 = self._limits().project_pq(P, Q)
            self.action.c[0] = P2
            if self.action.c.size >= 2:
                self.action.c[1] = Q2

    def _advance_uc(self) -> None:
        """Advance UC lifecycle in StatusBlock using d[0] ∈ {off, on}."""
        s = self._status()
        a = int(self.action.d[0]) if self.action.d.size else 1
        self._uc_cost_step = 0.0

        s.t_in_state_s = (
            0.0 
            if s.t_in_state_s is None 
            else float(s.t_in_state_s + self._dt_h)
        )

        t_start = int(self._startup_time or 0)
        t_stop = int(self._shutdown_time or 0)

        if s.state == "online" and a == 0 and t_stop > 0:
            s.state = "shutdown"
            s.t_to_next_s = t_stop * self._dt_h
            s.progress_frac = 0.0
        elif s.state == "offline" and a == 1 and t_start > 0:
            s.state = "startup"
            s.t_to_next_s = t_start * self._dt_h
            s.progress_frac = 0.0

        if s.t_to_next_s is not None:
            s.t_to_next_s = max(0.0, s.t_to_next_s - self._dt_h)
            total = t_start if s.state == "startup" else t_stop
            if total > 0:
                denom = max(total * self._dt_h, 1e-9)
                s.progress_frac = float(1.0 - s.t_to_next_s / denom)

        if s.state == "startup" and s.t_to_next_s == 0.0:
            s.state = "online"
            s.t_in_state_s = 0.0
            s.t_to_next_s = None
            s.progress_frac = None
            self._uc_cost_step = float(self._startup_cost)

        if s.state == "shutdown" and s.t_to_next_s == 0.0:
            s.state = "offline"
            s.t_in_state_s = 0.0
            s.t_to_next_s = None
            s.progress_frac = None
            self._uc_cost_step = float(self._shutdown_cost)

    def _electrical(self) -> ElectricalBasePh:
        for f in self.state.features:
            if isinstance(f, ElectricalBasePh):
                return f
        raise ValueError("ElectricalBasePh feature not found")

    def _status(self) -> StatusBlock:
        for f in self.state.features:
            if isinstance(f, StatusBlock):
                return f
        raise ValueError("StatusBlock feature not found")

    def _limits(self) -> GeneratorLimits:
        for f in self.state.features:
            if isinstance(f, GeneratorLimits):
                return f
        raise ValueError("GeneratorLimits feature not found")

    def __repr__(self) -> str:
        name = self.config.get("name", "<unnamed>")
        cfg = self.config.get("device_state_config", {})
        s = cfg.get("s_rated_MVA", None)
        pmin = cfg.get("p_min_MW", None)
        pmax = cfg.get("p_max_MW", None)
        return f"DG(name={name}, S={s}MVA, P∈[{pmin},{pmax}]MW)"

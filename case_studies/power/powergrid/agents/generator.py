from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Sequence

import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from heron.core.policies import Policy
from heron.protocols.base import NoProtocol, Protocol
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.power_limits import PowerLimits
from powergrid.features.status import StatusBlock
from heron.messaging.base import ChannelManager, Message, MessageType
from powergrid.utils.cost import cost_from_curve
from heron.utils.typing import float_if_not_none
from powergrid.utils.phase import PhaseModel, PhaseSpec, check_phase_model_consistency


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
    vm_min_pu: Optional[float] = None
    vm_max_pu: Optional[float] = None
    va_min_deg: Optional[float] = None
    va_max_deg: Optional[float] = None
    derate_frac: float = 1.0

    # UC timings (in timesteps); if both None -> no UC head
    startup_time_hr: Optional[int] = None
    shutdown_time_hr: Optional[int] = None
    startup_cost: float = 0.0
    shutdown_cost: float = 0.0

    # economics & sim
    cost_curve_coefs: Sequence[float] = (0.0, 0.0, 0.0)  # a,b,c for a*(P^2)+b*P+c
    dt_h: float = 1.0  # hours per step (for costs)
    min_pf: Optional[float] = None   # for safety penalty
    type: str = "fossil"
    source: Optional[str] = None  # "solar", "wind", or None for dispatchable

    # PQ / PV / SLACK control mode
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
        *,
        agent_id: Optional[str] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        message_broker: Optional['MessageBroker'] = None,
        upstream_id: Optional[str] = None,
        env_id: Optional[str] = None,
        device_config: Dict[str, Any] = {},
    ):
        state_config = device_config.get("device_state_config", {})

        # phase model & spec
        self.phase_model = PhaseModel(device_config.get("phase_model", "balanced_1ph"))
        self.phase_spec = PhaseSpec().from_dict(device_config.get("phase_spec", {}))
        check_phase_model_consistency(self.phase_model, self.phase_spec)

        self._generator_config = GeneratorConfig(
            bus=state_config.get("bus", ""),
            s_rated_MVA=float_if_not_none(state_config.get("s_rated_MVA", None)),
            p_min_MW=float_if_not_none(state_config.get("p_min_MW", 0.0)),
            p_max_MW=float_if_not_none(state_config.get("p_max_MW", 0.0)),
            q_min_MVAr=float_if_not_none(state_config.get("q_min_MVAr", None)),
            q_max_MVAr=float_if_not_none(state_config.get("q_max_MVAr", None)),
            pf_min_abs=float_if_not_none(state_config.get("pf_min_abs", None)),
            vm_min_pu=float_if_not_none(state_config.get("vm_min_pu", 0.95)),
            vm_max_pu=float_if_not_none(state_config.get("vm_max_pu", 1.05)),
            va_min_deg=float_if_not_none(state_config.get("va_min_deg", -180.0)),
            va_max_deg=float_if_not_none(state_config.get("va_max_deg", 180.0)),
            derate_frac=float_if_not_none(state_config.get("derate_frac", 1.0)),

            # economics & UC params
            cost_curve_coefs=state_config.get("cost_curve_coefs", (0.0, 0.0, 0.0)),
            dt_h=float_if_not_none(state_config.get("dt_h", 1.0)),
            startup_time_hr=float_if_not_none(state_config.get("startup_time_hr", None)),
            shutdown_time_hr=float_if_not_none(state_config.get("shutdown_time_hr", None)),
            startup_cost=float_if_not_none(state_config.get("startup_cost", 0.0)),
            shutdown_cost=float_if_not_none(state_config.get("shutdown_cost", 0.0)),
            min_pf=float_if_not_none(state_config.get("min_pf", None)),
            type=state_config.get("type", "fossil"),
            source=state_config.get("source", None),

            # control mode
            control_mode=state_config.get("control_mode", "PQ"),
        )
        super().__init__(
            agent_id=agent_id,
            policy=policy,
            protocol=protocol,
            message_broker=message_broker,
            upstream_id=upstream_id,
            env_id=env_id,
            device_config=device_config,
        )

    def set_device_action(self) -> None:
        """
        Define Action depending on generator control model.

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
        cfg = self._generator_config
        mode = (cfg.control_mode or "PQ").upper()

        lows: list[float] = []
        highs: list[float] = []

        if mode == "PQ":
            # --- P bounds (always present in PQ) ---
            lows.append(cfg.p_min_MW)
            highs.append(cfg.p_max_MW)

            has_q_box = (
                cfg.q_min_MVAr is not None and
                cfg.q_max_MVAr is not None
            )
            has_pf_cap = (
                cfg.pf_min_abs is not None and
                cfg.s_rated_MVA is not None
            )

            if has_q_box:
                # (1b) P+Q with explicit limits
                lows.append(cfg.q_min_MVAr)
                highs.append(cfg.q_max_MVAr)

            elif has_pf_cap:
                # (1c) P+Q with PF capability curve:
                # approximate symmetric Q box from S & pf_min_abs
                import math
                S = cfg.s_rated_MVA * (cfg.derate_frac or 1.0)
                pf = cfg.pf_min_abs
                q_max_mag = S * math.sqrt(max(0.0, 1.0 - pf * pf))
                lows.append(-q_max_mag)
                highs.append(q_max_mag)

            else:
                # (1a) P-only: Q is not a control variable (e.g., fixed or 0)
                pass

        elif mode == "PV":
            # (2) PV: action = [P, Vm]
            lows.append(cfg.p_min_MW)
            highs.append(cfg.p_max_MW)

            vm_min = float(cfg.vm_min_pu)
            vm_max = float(cfg.vm_max_pu)
            lows.append(vm_min)
            highs.append(vm_max)

        elif mode == "SLACK":
            # (3) Slack: action = [Vm, Va]
            lows.append(cfg.vm_min_pu)
            highs.append(cfg.vm_max_pu)
            lows.append(np.deg2rad(cfg.va_min_deg))
            highs.append(np.deg2rad(cfg.va_max_deg))

        else:
            raise ValueError(f"Unknown generator control_mode={mode!r}")

        # --- Unit commitment discrete head ---
        have_uc = (
            cfg.startup_time_hr is not None or
            cfg.shutdown_time_hr is not None
        )

        self.action.set_specs(
            dim_c=len(lows),
            dim_d=(1 if have_uc else 0),
            ncats=(2 if have_uc else 0),
            range=(np.asarray(lows, np.float32), np.asarray(highs, np.float32)),
        )

    def set_device_state(self) -> None:
        # Electrical telemetry
        eletrical_telemetry = ElectricalBasePh(
            phase_model=self.phase_model,
            phase_spec=self.phase_spec,
            P_MW=0.0,
            Q_MVAr=0.0,
            Vm_pu=1.0,
            Va_rad=0.0,
            visibility=["owner"],
        )

        # Status / UC lifecycle
        status = StatusBlock(
            in_service=True,
            out_service=False,
            state="online",
            states_vocab=["offline", "startup", "online", "shutdown", "fault"],
            emit_state_one_hot=True,
            emit_state_index=False,
            visibility=["owner"],
            t_in_state_s=0.0,
            t_to_next_s=None,  # None means not in transition
            progress_frac=None,
        )

        # Capability / limits
        power_limits = PowerLimits(
            s_rated_MVA=self._generator_config.s_rated_MVA,
            derate_frac=self._generator_config.derate_frac,
            p_min_MW=self._generator_config.p_min_MW,
            p_max_MW=self._generator_config.p_max_MW,
            q_min_MVAr=self._generator_config.q_min_MVAr,
            q_max_MVAr=self._generator_config.q_max_MVAr,
            pf_min_abs=self._generator_config.pf_min_abs,
        )

        self.state.features=[
            eletrical_telemetry,
            status,
            power_limits,
        ]
        self.state.owner_id = self.agent_id
        self.state.owner_level = self.level

    def reset_device(self) -> None:
        """Reset generator to a neutral operating point."""
        self.state.reset()
        self.action.reset()

        # Cost / safety bookkeeping
        self.cost = 0.0
        self.safety = 0.0
        self._uc_cost_step = 0.0

    def update_state(self, **kwargs) -> None:
        """Update generator state with optional kwargs for feature updates.

        Args:
            **kwargs: Optional keyword arguments to update specific feature fields
        """
        self._update_uc_status()
        self._update_power_outputs()

        # Apply any additional kwargs to respective features
        if kwargs:
            self._update_by_kwargs(**kwargs)

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
        dt = self._generator_config.dt_h
        t_in_state_s += dt

        # 2) UC command: 0=request_off, 1=request_on
        uc_cmd = None
        if self.action.dim_d > 0 and self.action.d.size > 0:
            uc_cmd = int(self.action.d[0])

        t_start = self._generator_config.startup_time_hr
        t_stop = self._generator_config.shutdown_time_hr

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
                    self._uc_cost_step = self._generator_config.shutdown_cost
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
                    self._uc_cost_step = self._generator_config.startup_cost
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
                    self._uc_cost_step = self._generator_config.startup_cost
                elif state == "shutdown":
                    state = "offline"
                    t_in_state_s = 0.0
                    t_to_next_s = None
                    progress_frac = None
                    self._uc_cost_step = self._generator_config.shutdown_cost

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

    def _update_by_kwargs(self, **kwargs) -> None:
        """Update features based on provided kwargs.

        Args:
            **kwargs: Keyword arguments mapping to feature fields
        """
        electrical_keys = {f.name for f in fields(ElectricalBasePh)}
        status_keys = {f.name for f in fields(StatusBlock)}
        power_limits_keys = {f.name for f in fields(PowerLimits)}

        elec_updates = {k: v for k, v in kwargs.items() if k in electrical_keys}
        status_updates = {k: v for k, v in kwargs.items() if k in status_keys}
        limits_updates = {k: v for k, v in kwargs.items() if k in power_limits_keys}

        if elec_updates:
            self.state.update_feature(ElectricalBasePh.feature_name, **elec_updates)
        if status_updates:
            self.state.update_feature(StatusBlock.feature_name, **status_updates)
        if limits_updates:
            self.state.update_feature(PowerLimits.feature_name, **limits_updates)
    def update_cost_safety(self) -> None:
        """Economic cost + S/PF penalties + UC start/stop cost."""
        P = self.electrical.P_MW or 0.0
        Q = self.electrical.Q_MVAr or 0.0
        on = 1.0 if self.status.state == "online" else 0.0
        dt = self._generator_config.dt_h

        # Cost
        fuel_cost = cost_from_curve(P, self._generator_config.cost_curve_coefs)
        uc_cost = self._uc_cost_step
        self.cost = fuel_cost * on * dt + uc_cost

        # Safety violations
        violations = self.limits.feasible(P, Q)
        self.safety = np.sum(list(violations.values())) * on * dt

    def _publish_state_updates(self) -> None:
        """Publish electrical state to environment for pandapower sync.

        This method is called during hierarchical execution to send device
        state updates to the environment via message broker.
        """
        if not self.message_broker:
            return

        channel = ChannelManager.state_update_channel(self.env_id)
        message = Message(
            env_id=self.env_id,
            sender_id=self.agent_id,
            recipient_id="environment",
            timestamp=self._timestep,
            message_type=MessageType.STATE_UPDATE,
            payload={
                'agent_id': self.agent_id,
                'device_type': 'sgen',
                'P_MW': float(self.electrical.P_MW),
                'Q_MVAr': float(self.electrical.Q_MVAr or 0.0),
                'in_service': bool(self.status.in_service),
            }
        )
        self.message_broker.publish(channel, message)

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
        return self._generator_config.bus

    @property
    def name(self) -> str:
        return self.agent_id

    @property
    def source(self) -> Optional[str]:
        """Get renewable source type ('solar', 'wind', or None for dispatchable)."""
        return self._generator_config.source

    def __repr__(self) -> str:
        name = self.config.name
        s = self.limits.s_rated_MVA
        pmin = self.limits.p_min_MW
        pmax = self.limits.p_max_MW
        return f"DG(name={name}, S={s}MVA, P∈[{pmin},{pmax}]MW)"

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.power_limits import PowerLimits
from powergrid.features.status import StatusBlock
from powergrid.features.storage import StorageBlock
from powergrid.messaging.base import ChannelManager, Message, MessageType
from powergrid.utils.typing import float_if_not_none
from powergrid.utils.phase import PhaseModel, PhaseSpec, check_phase_model_consistency


@dataclass
class StorageConfig:
    """Configuration for an Energy Storage System."""
    bus: str

    # Power & energy constraints
    p_min_MW: float = 0.0          # allowed negative (discharge)
    p_max_MW: float = 0.0          # positive (charge)
    e_capacity_MWh: float = 0.0    # nameplate energy capacity

    # SOC limits (fractions of capacity)
    soc_min: float = 0.0           # lower bound in [0,1]
    soc_max: float = 1.0           # upper bound in [0,1]
    init_soc: float = 0.5          # initial SOC in [0,1]

    # Reactive power
    q_min_MVAr: Optional[float] = None
    q_max_MVAr: Optional[float] = None
    s_rated_MVA: Optional[float] = None

    # Efficiency
    ch_eff: float = 0.98
    dsc_eff: float = 0.98

    # Degradation economics
    e_throughput_MWh: float = 0.0
    degr_cost_per_MWh: float = 0.0
    degr_cost_per_cycle: float = 0.0
    degr_cost_cum: float = 0.0

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
        *,
        agent_id: Optional[str] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        message_broker: Optional['MessageBroker'] = None,
        upstream_id: Optional[str] = None,
        env_id: Optional[str] = None,
        device_config: Dict[str, Any],
    ):
        config = device_config.get("device_state_config", {})

        # phase model & spec
        self.phase_model = PhaseModel(device_config.get("phase_model", "balanced_1ph"))
        self.phase_spec = PhaseSpec().from_dict(device_config.get("phase_spec", {}))
        check_phase_model_consistency(self.phase_model, self.phase_spec)

        self._storage_config = StorageConfig(
            bus=config.get("bus", ""),
            p_min_MW=config.get("p_min_MW", 0.0),
            p_max_MW=config.get("p_max_MW", 0.0),
            e_capacity_MWh=config.get("e_capacity_MWh", 0.0),

            soc_min=config.get("soc_min", 0.0),
            soc_max=config.get("soc_max", 1.0),
            init_soc=config.get("init_soc", 0.5),

            q_min_MVAr=float_if_not_none(config.get("q_min_MVAr", None)),
            q_max_MVAr=float_if_not_none(config.get("q_max_MVAr", None)),
            s_rated_MVA=float_if_not_none(config.get("s_rated_MVA", None)),

            ch_eff=config.get("ch_eff", 0.98),
            dsc_eff=config.get("dsc_eff", 0.98),

            e_throughput_MWh=config.get("e_throughput_MWh", 0.0),
            degr_cost_per_MWh=config.get("degr_cost_per_MWh", 0.0),
            degr_cost_per_cycle=config.get("degr_cost_per_cycle", 0.0),
            degr_cost_cum=config.get("degr_cost_cum", 0.0),

            dt_h=config.get("dt_h", 1.0),
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
        Define Action:
            - c[0]: P_MW in [p_min_MW, p_max_MW]
            - c[1]: Q_MVAr in [q_min_MVAr, q_max_MVAr] if Q control enabled
        """
        cfg = self._storage_config

        lows = [cfg.p_min_MW]
        highs = [cfg.p_max_MW]

        has_q_control = (
            cfg.q_min_MVAr is not None and 
            cfg.q_max_MVAr is not None
        )
        if has_q_control:
            lows.append(cfg.q_min_MVAr)
            highs.append(cfg.q_max_MVAr)

        self.action.set_specs(
            dim_c=len(lows),
            dim_d=0,
            ncats=0,
            range=(
                np.asarray(lows, dtype=np.float32),
                np.asarray(highs, dtype=np.float32),
            ),
        )

    def set_device_state(self) -> None:
        cfg = self._storage_config

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

        storage_block = StorageBlock(
            soc=cfg.init_soc,
            soc_min=cfg.soc_min,
            soc_max=cfg.soc_max,
            e_capacity_MWh=cfg.e_capacity_MWh,
            p_ch_max_MW=cfg.p_max_MW,
            p_dsc_max_MW=-cfg.p_min_MW,
            ch_eff=cfg.ch_eff,
            dsc_eff=cfg.dsc_eff,
            e_throughput_MWh=cfg.e_throughput_MWh,
            degr_cost_per_MWh=cfg.degr_cost_per_MWh,
            degr_cost_per_cycle=cfg.degr_cost_per_cycle,
            degr_cost_cum=cfg.degr_cost_cum,
            visibility=["owner"],
        )

        self.state.features = [eletrical_telemetry, storage_block]
        self.state.owner_id = self.agent_id
        self.state.owner_level = self.level

        has_q_control = (
            cfg.q_min_MVAr is not None and 
            cfg.q_max_MVAr is not None
        )
        if has_q_control:
            # Capability / limits
            power_limits = PowerLimits(
                s_rated_MVA=self._storage_config.s_rated_MVA,
                p_min_MW=self._storage_config.p_min_MW,
                p_max_MW=self._storage_config.p_max_MW,
                q_min_MVAr=self._storage_config.q_min_MVAr,
                q_max_MVAr=self._storage_config.q_max_MVAr,
            )
            self.state.features.append(power_limits)

    def reset_device(self, **kwargs) -> None:
        """Reset ESS to a neutral operating point.

        Args:
            **kwargs: Optional keyword arguments for StorageBlock.reset:
                soc: Explicit SOC target (fraction in [0, 1])
                random_init_soc: Sample SOC uniformly in [soc_min, soc_max]
                seed: RNG seed for random SOC initialization
                reset_degradation: Clear degradation accounting
        """
        # Extract optional overrides for StorageBlock.reset
        soc = kwargs.get("soc", None)
        random_init_soc = kwargs.get("random_init_soc", False)
        seed = kwargs.get("seed", None)
        reset_degradation = kwargs.get("reset_degradation", True)

        # Reset all feature providers and the action to their neutral state
        self.state.reset()
        self.action.reset()

        # If any SOC/degradation overrides are provided, explicitly re-reset
        # the StorageBlock with those options on top of the neutral reset
        if any(k in kwargs for k in ("soc", "random_init_soc", "seed", "reset_degradation")):
            self.storage.reset(
                soc=soc,
                random_init=random_init_soc,
                seed=seed,
                reset_degradation=reset_degradation,
            )

        # Cost / safety bookkeeping
        self.cost = 0.0
        self.safety = 0.0

    def update_state(self, **kwargs) -> None:
        """Apply P/Q from action, update SOC/degradation, apply extra updates.

        Args:
            **kwargs: Optional keyword arguments for feature updates:
                - ElectricalBasePh fields (e.g. P_MW, Q_MVAr, Vm_pu, Va_rad)
                - StorageBlock fields (e.g. soc, e_throughput_MWh)
                - PowerLimits fields (if PowerLimits is present)
        """
        P_eff, Q_eff = self._update_power_outputs()
        self._update_storage_dynamics(P_eff)

        if kwargs:
            self._update_by_kwargs(**kwargs)

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

        self.state.update_feature(ElectricalBasePh, **elec_updates)
        return P_eff, Q_eff

    def _update_storage_dynamics(self, P_MW: float) -> None:
        """Update SOC and degradation in StorageBlock based on active power.

        Args:
            P_MW: Active power (positive=charging, negative=discharging)
        """
        storage = self.storage
        dt = self._storage_config.dt_h

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

    def _update_by_kwargs(self, **kwargs) -> None:
        """Route keyword updates to appropriate FeatureProviders.

        Args:
            **kwargs: Keyword arguments mapping to feature fields
        """
        electrical_keys = {f.name for f in fields(ElectricalBasePh)}
        storage_keys = {f.name for f in fields(StorageBlock)}
        power_limits_keys = {f.name for f in fields(PowerLimits)}

        elec_updates = {k: v for k, v in kwargs.items() if k in electrical_keys}
        storage_updates = {k: v for k, v in kwargs.items() if k in storage_keys}
        limits_updates = {k: v for k, v in kwargs.items() if k in power_limits_keys}

        if elec_updates:
            self.state.update_feature(ElectricalBasePh, **elec_updates)
        if storage_updates:
            self.state.update_feature(StorageBlock, **storage_updates)
        if limits_updates:
            self.state.update_feature(PowerLimits, **limits_updates)

    def update_cost_safety(self) -> None:
        """Update per-step cost and safety penalties for the ESS.

        Cost: Degradation from energy throughput and equivalent cycles
        Safety: Apparent-power overload and SOC bounds violations
        """
        P = self.electrical.P_MW or 0.0
        Q = self.electrical.Q_MVAr or 0.0
        dt = self._storage_config.dt_h

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

        self.cost = degr_cost_inc

        # Safety: SOC violations + power limit violations
        safety = self.storage.soc_violation()
        if self.limits is not None:
            violations = self.limits.feasible(P, Q)
            safety += np.sum(list(violations.values())) * dt

        self.safety = safety

    def feasible_action(self, P_req: float) -> float:
        """Clip action to feasible set based on SOC window and power limits.

        Args:
            P_req: Requested active power (positive=charging, negative=discharging)

        Returns:
            Clipped active power respecting SOC and power constraints
        """
        storage = self.storage
        dt = self._storage_config.dt_h

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

        p_min = max(p_min_soc, self._storage_config.p_min_MW)
        p_max = min(p_max_soc, self._storage_config.p_max_MW)

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
        return self._storage_config.bus

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

    def __repr__(self) -> str:
        name = self.agent_id
        cap = self._storage_config.e_capacity_MWh
        pmin = self._storage_config.p_min_MW
        pmax = self._storage_config.p_max_MW
        return f"ESS(name={name}, capacity={cap}MWh, P∈[{pmin},{pmax}]MW)"

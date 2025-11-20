from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.core.state import DeviceState, PhaseModel, PhaseSpec
from powergrid.features.connection import PhaseConnection
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.storage import StorageBlock
from powergrid.utils.cost import cost_from_curve
from powergrid.utils.safety import s_over_rating, soc_bounds_penalty
from powergrid.utils.typing import float_if_not_none


@dataclass
class StorageConfig:
    """Configuration for an Energy Storage System."""
    bus: str

    # power & energy constraints
    min_p_MW: float = 0.0
    max_p_MW: float = 0.0
    capacity_MWh: float = 0.0
    max_e_MWh: Optional[float] = None
    min_e_MWh: float = 0.0
    init_soc: float = 0.5

    # reactive power
    min_q_MVAr: Optional[float] = None
    max_q_MVAr: Optional[float] = None
    s_rated_MVA: Optional[float] = None

    # efficiency
    ch_eff: float = 0.98
    dsc_eff: float = 0.98

    # economics & sim
    cost_curve_coefs: tuple = (0.0, 0.0, 0.0)
    dt_h: float = 1.0

    # phase context
    phase_model: str = "balanced_1ph"
    phase_spec: Optional[Dict[str, Any]] = None


class ESS(DeviceAgent):
    """
    Energy Storage System following DeviceAgent's lifecycle.

    Example device_config:
    {
      "device_state_config": {
        "phase_model": "BALANCED_1PH" | "THREE_PHASE",
        "phase_spec": None or {
            "phases": "ABC", "has_neutral": False, "earth_bond": True
        },

        "bus": "bus_1",
        "min_p_MW": -10.0,
        "max_p_MW": 10.0,
        "capacity_MWh": 20.0,
        "max_e_MWh": 18.0,
        "min_e_MWh": 2.0,
        "init_soc": 0.5,

        "min_q_MVAr": -5.0,
        "max_q_MVAr": 5.0,
        "s_rated_MVA": 12.0,

        "ch_eff": 0.98,
        "dsc_eff": 0.98,

        "cost_curve_coefs": [0.01, 0.5, 0.0],
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
        device_config: Dict[str, Any],
    ):
        config = device_config.get("device_state_config", {})

        # compute max_e_MWh and capacity
        capacity = float_if_not_none(config.get("capacity_MWh", 0.0))
        max_e = config.get("max_e_MWh", None)
        max_e_MWh = float_if_not_none(max_e) if max_e is not None else capacity

        self._storage_config = StorageConfig(
            bus=config.get("bus", ""),
            min_p_MW=float_if_not_none(config.get("min_p_MW", 0.0)),
            max_p_MW=float_if_not_none(config.get("max_p_MW", 0.0)),
            capacity_MWh=capacity,
            max_e_MWh=max_e_MWh,
            min_e_MWh=float_if_not_none(config.get("min_e_MWh", 0.0)),
            init_soc=float_if_not_none(config.get("init_soc", 0.5)),

            min_q_MVAr=float_if_not_none(config.get("min_q_MVAr", None)),
            max_q_MVAr=float_if_not_none(config.get("max_q_MVAr", None)),
            s_rated_MVA=float_if_not_none(config.get("s_rated_MVA", None)),

            ch_eff=float_if_not_none(config.get("ch_eff", 0.98)),
            dsc_eff=float_if_not_none(config.get("dsc_eff", 0.98)),

            cost_curve_coefs=config.get("cost_curve_coefs", (0.0, 0.0, 0.0)),
            dt_h=float_if_not_none(config.get("dt_h", 1.0)),

            phase_model=config.get("phase_model", "balanced_1ph"),
            phase_spec=config.get("phase_spec", None),
        )

        # compute Q limits from S rating if provided
        if self._storage_config.s_rated_MVA is not None:
            P_max = self._storage_config.max_p_MW
            S = self._storage_config.s_rated_MVA
            Q_max = float(np.sqrt(max(0.0, S**2 - P_max**2)))
            self._storage_config.min_q_MVAr = -Q_max
            self._storage_config.max_q_MVAr = Q_max

        super().__init__(
            agent_id=agent_id,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def _init_action_space(self) -> None:
        """Initialize action space - calls set_action_space()."""
        self.set_action_space()

    def _init_device_state(self) -> None:
        """Initialize device state - calls set_device_state()."""
        self.set_device_state()

    def set_action_space(self) -> None:
        """
        Define Action:
          - continuous P (and optionally Q).
        """
        lows = [self._storage_config.min_p_MW]
        highs = [self._storage_config.max_p_MW]

        # if control reactive power
        if self._storage_config.min_q_MVAr is not None and self._storage_config.max_q_MVAr is not None:
            lows.append(self._storage_config.min_q_MVAr)
            highs.append(self._storage_config.max_q_MVAr)

        # Set action specs & sample
        self.action.set_specs(
            dim_c=len(lows),
            dim_d=0,
            ncats=0,
            range=(np.asarray(lows, np.float32), np.asarray(highs, np.float32)),
            masks=None,
        )
        self.action.sample()

    def set_device_state(self) -> None:
        # phase model & spec
        pm = PhaseModel(self._storage_config.phase_model)
        ps = PhaseSpec(
            self._storage_config.phase_spec.get("phases", "ABC"),
            self._storage_config.phase_spec.get("has_neutral", False),
            self._storage_config.phase_spec.get("earth_bond", True),
        ) if self._storage_config.phase_spec is not None else None

        # Electrical telemetry
        electrical_telemetry = ElectricalBasePh(
            P_MW=0.0,
            Q_MVAr=(
                0.0
                if self._storage_config.min_q_MVAr is not None and self._storage_config.max_q_MVAr is not None
                else None
            ),
        )

        # Connection
        conn = "ON" if pm == PhaseModel.BALANCED_1PH else "ABC"
        connection = PhaseConnection(
            phase_model=pm,
            phase_spec=ps,
            connection=conn
        )

        # Storage block
        min_soc = self._storage_config.min_e_MWh / self._storage_config.capacity_MWh
        max_soc = self._storage_config.max_e_MWh / self._storage_config.capacity_MWh

        storage_block = StorageBlock(
            soc=self._storage_config.init_soc,
            soc_min=min_soc,
            soc_max=max_soc,
            e_capacity_MWh=self._storage_config.capacity_MWh,
            p_ch_max_MW=self._storage_config.max_p_MW,
            p_dis_max_MW=-self._storage_config.min_p_MW,
            eta_ch=self._storage_config.ch_eff,
            eta_dis=self._storage_config.dsc_eff,
        )

        self.state = DeviceState(
            phase_model=pm,
            phase_spec=ps,
            features=[
                electrical_telemetry,
                connection,
                storage_block,
            ],
            prefix_names=False,
        )
        self.state._apply_phase_context_to_features_()

    def reset_device(self, *args, **kwargs) -> None:
        """Reset P/Q and SOC."""
        rnd = kwargs.get("rnd", np.random)
        init_soc = kwargs.get("init_soc", None)

        # reset electricals
        electrical = self.electrical
        electrical.P_MW = 0.0
        if self.action.dim_c >= 2:
            electrical.Q_MVAr = 0.0

        # reset storage SOC
        storage = self.storage
        min_soc = self._storage_config.min_e_MWh / self._storage_config.capacity_MWh
        max_soc = self._storage_config.max_e_MWh / self._storage_config.capacity_MWh
        storage.soc = float(init_soc) if init_soc is not None else float(
            rnd.uniform(min_soc, max_soc)
        )

        # reset cost and safety
        self.cost = 0.0
        self.safety = 0.0

    def update_state(self, *args, **kwargs) -> None:
        """
        Apply P/Q from action and update SOC dynamics.
        P > 0: charging, P < 0: discharging.
        """
        electrical = self.electrical
        storage = self.storage

        # Read action
        P = float(self.action.c[0]) if self.action.c.size >= 1 else 0.0
        Q = float(self.action.c[1]) if self.action.c.size >= 2 else 0.0

        # Update electrical
        electrical.P_MW = P
        if self.action.dim_c >= 2:
            electrical.Q_MVAr = Q

        # Update SOC (P >= 0: charging, P < 0: discharging)
        if P >= 0:
            storage.soc += P * self._storage_config.ch_eff * self._storage_config.dt_h / self._storage_config.capacity_MWh
        else:
            storage.soc += P / self._storage_config.dsc_eff * self._storage_config.dt_h / self._storage_config.capacity_MWh

        # Defensive clamp on all child features
        self.state.clamp_()

    def update_cost_safety(self, *args, **kwargs) -> None:
        """Economic cost + S/SOC penalties."""
        electrical = self.electrical
        storage = self.storage

        P = float(electrical.P_MW or 0.0)
        Q = float(electrical.Q_MVAr or 0.0)

        # Cost
        self.cost = cost_from_curve(P, self._storage_config.cost_curve_coefs) * self._storage_config.dt_h

        # Safety
        safety = 0.0
        S = self._storage_config.s_rated_MVA
        if self.action.dim_c >= 2 and S is not None:
            safety += s_over_rating(P, Q, S)

        min_soc = self._storage_config.min_e_MWh / self._storage_config.capacity_MWh
        max_soc = self._storage_config.max_e_MWh / self._storage_config.capacity_MWh
        safety += soc_bounds_penalty(storage.soc, min_soc, max_soc)

        self.safety = safety * self._storage_config.dt_h

    def feasible_action(self) -> None:
        """Clip action to feasible set based on current SOC."""
        storage = self.storage

        min_soc = self._storage_config.min_e_MWh / self._storage_config.capacity_MWh
        max_soc = self._storage_config.max_e_MWh / self._storage_config.capacity_MWh

        # compute instantaneous feasible P based on available energy windows
        max_dsc_power = (storage.soc - min_soc) * self._storage_config.capacity_MWh * self._storage_config.dsc_eff / self._storage_config.dt_h
        max_dsc_power = min(max_dsc_power, -self._storage_config.min_p_MW)

        max_ch_power = (max_soc - storage.soc) * self._storage_config.capacity_MWh / self._storage_config.ch_eff / self._storage_config.dt_h
        max_ch_power = min(max_ch_power, self._storage_config.max_p_MW)

        low = -max_dsc_power
        high = max_ch_power

        if self.action.c.size >= 1:
            self.action.c[0] = np.clip(self.action.c[0], low, high)
        if self.action.c.size >= 2:
            self.action.c[1] = np.clip(
                self.action.c[1],
                self._storage_config.min_q_MVAr,
                self._storage_config.max_q_MVAr
            )

    @property
    def electrical(self) -> ElectricalBasePh:
        for f in self.state.features:
            if isinstance(f, ElectricalBasePh):
                return f
        raise ValueError("ElectricalBasePh feature not found")

    @property
    def storage(self) -> StorageBlock:
        for f in self.state.features:
            if isinstance(f, StorageBlock):
                return f
        raise ValueError("StorageBlock feature not found")

    @property
    def bus(self) -> str:
        return self._storage_config.bus

    @property
    def name(self) -> str:
        return self.agent_id

    @property
    def max_e_mwh(self) -> float:
        return self._storage_config.max_e_MWh

    @property
    def min_e_mwh(self) -> float:
        return self._storage_config.min_e_MWh

    @property
    def max_p_mw(self) -> float:
        return self._storage_config.max_p_MW

    @property
    def min_p_mw(self) -> float:
        return self._storage_config.min_p_MW

    @property
    def max_q_mvar(self) -> Optional[float]:
        return self._storage_config.max_q_MVAr

    @property
    def min_q_mvar(self) -> Optional[float]:
        return self._storage_config.min_q_MVAr

    @property
    def sn_mva(self) -> Optional[float]:
        return self._storage_config.s_rated_MVA

    def __repr__(self) -> str:
        name = self.agent_id
        cap = self._storage_config.capacity_MWh
        pmin = self._storage_config.min_p_MW
        pmax = self._storage_config.max_p_MW
        return f"ESS(name={name}, capacity={cap}MWh, P∈[{pmin},{pmax}]MW)"

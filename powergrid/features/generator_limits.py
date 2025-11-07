from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import math
import numpy as np

from powergrid.features.base import FeatureProvider
from powergrid.utils.registry import provider
from powergrid.utils.typing import Array


@provider()
@dataclass(slots=True)
class GeneratorLimits(FeatureProvider):
    """
    Generator capability / constraints.

    Fields
    ------
    # Apparent-power (S) capability
    s_rated_MVA:   rated apparent power (nameplate)
    derate_frac:   optional fraction [0..1] to uniformly derate capability (env/weather/thermal)

    # Static bounds (optional overrides / intersections with S, PF)
    p_min_MW:      minimum active power
    p_max_MW:      maximum active power
    q_min_MVAr:    minimum reactive power (absorption if negative)
    q_max_MVAr:    maximum reactive power (injection if positive)

    # Power factor constraint (symmetric lead/lag)
    pf_min_abs:    minimum |pf| in [0,1] (if provided, applies on both lead/lag)

    Notes
    -----
    - The feasible set is the intersection of:
        * P range:      p_min_MW <= P <= p_max_MW   (if provided)
        * Q range:      q_min_MVAr <= Q <= q_max_MVAr (if provided)
        * S circle:     P^2 + Q^2 <= (derate * s_rated_MVA)^2 (if s_rated_MVA provided)
        * PF wedge:     |PF| >= pf_min_abs  ->  |Q| <= |P| * tan(acos(pf_min_abs))  (if pf_min_abs provided)
    - Helper methods:
        * feasible(P,Q) -> dict of violations
        * project_pq(P,Q) -> (P*, Q*) projected into feasible region
        * effective_q_bounds(P) -> (qmin_eff, qmax_eff) from all active constraints
    """

    # S capability
    s_rated_MVA: Optional[float] = None
    derate_frac: float = 1.0

    # static P/Q
    p_min_MW: Optional[float] = None
    p_max_MW: Optional[float] = None
    q_min_MVAr: Optional[float] = None
    q_max_MVAr: Optional[float] = None

    # symmetric PF
    pf_min_abs: Optional[float] = None

    def _S_avail(self) -> Optional[float]:
        if self.s_rated_MVA is None:
            return None
        s = float(self.s_rated_MVA) * float(np.clip(self.derate_frac, 0.0, 1.0))
        return max(0.0, s)

    def _tan_phi(self) -> Optional[float]:
        """Return tan(phi_min) from pf_min_abs; None if no PF constraint."""
        if self.pf_min_abs is None:
            return None
        pf = float(self.pf_min_abs)
        if not (0.0 < pf <= 1.0):
            raise ValueError("pf_min_abs must be in (0,1].")
        # pf = cos(phi) -> tan(phi) = sqrt(1/pf^2 - 1)
        return math.sqrt(max(1.0 / (pf * pf) - 1.0, 0.0))

    def vector(self) -> Array:
        parts: List[np.ndarray] = []

        def add(x: Optional[float]) -> None:
            if x is not None:
                parts.append(np.array([float(x)], np.float32))

        add(self.s_rated_MVA)
        parts.append(np.array([np.clip(self.derate_frac, 0.0, 1.0)], np.float32))

        add(self.p_min_MW)
        add(self.p_max_MW)
        add(self.q_min_MVAr)
        add(self.q_max_MVAr)
        add(self.pf_min_abs)

        if not parts:
            return np.zeros(0, np.float32)

        return np.concatenate(
            [p.astype(np.float32, copy=False) for p in parts]
        ).astype(np.float32, copy=False)

    def names(self) -> list[str]:
        n: list[str] = []
        if self.s_rated_MVA is not None: n.append("s_rated_MVA")
        n.append("derate_frac")
        if self.p_min_MW is not None:    n.append("p_min_MW")
        if self.p_max_MW is not None:    n.append("p_max_MW")
        if self.q_min_MVAr is not None:  n.append("q_min_MVAr")
        if self.q_max_MVAr is not None:  n.append("q_max_MVAr")
        if self.pf_min_abs is not None:  n.append("pf_min_abs")
        return n

    def clamp_(self) -> None:
        # normalize derate
        self.derate_frac = float(np.clip(self.derate_frac, 0.0, 1.0))
        # fix swapped bounds if provided
        if (
            self.p_min_MW is not None
            and self.p_max_MW is not None
            and self.p_min_MW > self.p_max_MW
        ):
            self.p_min_MW, self.p_max_MW = self.p_max_MW, self.p_min_MW
        if (
            self.q_min_MVAr is not None
            and self.q_max_MVAr is not None
            and self.q_min_MVAr > self.q_max_MVAr
        ):
            self.q_min_MVAr, self.q_max_MVAr = self.q_max_MVAr, self.q_min_MVAr
        # sanity on pf
        if self.pf_min_abs is not None:
            if not (0.0 < float(self.pf_min_abs) <= 1.0):
                raise ValueError("pf_min_abs must be in (0,1].")

    def to_dict(self) -> Dict:
        return {
            "s_rated_MVA": self.s_rated_MVA,
            "derate_frac": self.derate_frac,
            "p_min_MW": self.p_min_MW,
            "p_max_MW": self.p_max_MW,
            "q_min_MVAr": self.q_min_MVAr,
            "q_max_MVAr": self.q_max_MVAr,
            "pf_min_abs": self.pf_min_abs,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GeneratorLimits":
        return cls(
            s_rated_MVA=d.get("s_rated_MVA"),
            derate_frac=d.get("derate_frac", 1.0),
            p_min_MW=d.get("p_min_MW"),
            p_max_MW=d.get("p_max_MW"),
            q_min_MVAr=d.get("q_min_MVAr"),
            q_max_MVAr=d.get("q_max_MVAr"),
            pf_min_abs=d.get("pf_min_abs"),
        )

    def effective_q_bounds(self, P_MW: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute effective (qmin,qmax) at a given P by intersecting:
        - static q_min/q_max (if set)
        - S circle:   P^2 + Q^2 <= S_avail^2
        - PF wedge:   |Q| <= |P| * tan(phi_min)
        Returns (qmin_eff, qmax_eff), either may be None if unconstrained.
        """
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []

        # static
        if self.q_min_MVAr is not None:
            lower_bounds.append(float(self.q_min_MVAr))
        if self.q_max_MVAr is not None:
            upper_bounds.append(float(self.q_max_MVAr))

        # S circle
        S = self._S_avail()
        if S is not None:
            rad2 = max(S*S - float(P_MW)*float(P_MW), 0.0)
            q_cap = math.sqrt(rad2)
            lower_bounds.append(-q_cap)
            upper_bounds.append(+q_cap)

        # PF wedge
        tphi = self._tan_phi()
        if tphi is not None:
            bound = abs(float(P_MW)) * tphi
            lower_bounds.append(-bound)
            upper_bounds.append(+bound)

        # Intersection = tightest lower/upper
        qmin_eff = max(lower_bounds) if lower_bounds else None
        qmax_eff = min(upper_bounds) if upper_bounds else None

        # If intersection is empty numerically, collapse to (0,0)
        if (qmin_eff is not None and qmax_eff is not None) and qmin_eff > qmax_eff:
            qmin_eff, qmax_eff = 0.0, 0.0
        return qmin_eff, qmax_eff

    def feasible(self, P_MW: float, Q_MVAr: float) -> Dict[str, float]:
        """
        Return violation magnitudes (>=0). Zero means feasible.
        Keys: 'p_violation', 'q_violation', 's_excess', 'pf_violation'
        """
        self.clamp_()
        P = float(P_MW); Q = float(Q_MVAr)

        p_violation = 0.0
        if self.p_min_MW is not None:
            p_violation += max(0.0, self.p_min_MW - P)
        if self.p_max_MW is not None:
            p_violation += max(0.0, P - self.p_max_MW)

        q_violation = 0.0
        if self.q_min_MVAr is not None:
            q_violation += max(0.0, self.q_min_MVAr - Q)
        if self.q_max_MVAr is not None:
            q_violation += max(0.0, Q - self.q_max_MVAr)

        s_excess = 0.0
        S = self._S_avail()
        if S is not None:
            mag = math.hypot(P, Q)
            s_excess = max(0.0, mag - S)

        pf_violation = 0.0
        if self.pf_min_abs is not None and (abs(P) > 1e-9 or abs(Q) > 1e-9):
            pf = abs(P) / max(math.hypot(P, Q), 1e-9)
            pf_violation = max(0.0, float(self.pf_min_abs) - pf)

        return {
            "p_violation": p_violation,
            "q_violation": q_violation,
            "s_excess": s_excess,
            "pf_violation": pf_violation,
        }

    def project_pq(self, P_MW: float, Q_MVAr: float) -> Tuple[float, float]:
        """
        Project (P,Q) into the feasible set with a simple, deterministic strategy:
        1) Clip P to [p_min, p_max] if given.
        2) Limit Q by PF wedge and S circle (take the tightest), then by static q_min/max.
        """
        self.clamp_()
        P = P_MW
        Q = Q_MVAr

        # 1) clip P
        if self.p_min_MW is not None:
            P = max(P, self.p_min_MW)
        if self.p_max_MW is not None:
            P = min(P, self.p_max_MW)

        # 2) PF + S derived bound at this P
        qmin_eff, qmax_eff = self.effective_q_bounds(P)

        if qmin_eff is not None:
            Q = max(Q, qmin_eff)
        if qmax_eff is not None:
            Q = min(Q, qmax_eff)

        return float(P), float(Q)

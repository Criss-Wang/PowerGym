from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Type, Tuple, Optional, Union

import numpy as np

from powergrid.features.base import FeatureProvider, AgentLike
from powergrid.utils.typing import Array
from powergrid.utils.registry import provider
from powergrid.utils.array_utils import cat_f32

KNOWN_FEATURES: Dict[str, Type[FeatureProvider]] = {}

def _vec_names(feat: FeatureProvider) -> Tuple[np.ndarray, List[str]]:
    v = np.asarray(feat.vector(), np.float32).ravel()
    n = feat.names()
    if len(n) != v.size:
        raise ValueError(
            f"{feat.__class__.__name__}: names ({len(n)}) != vector size ({v.size})."
        )
    return v, n


@provider()
@dataclass(slots=True)
class State(ABC):
    """Generic agent state, defined by a list of feature providers."""

    # Which agent "owns" this state (typically the agent that holds it)
    owner_id: str = ""
    owner_level: int = 0

    # Raw features that make up this state
    features: List[FeatureProvider] = field(default_factory=list)

    def _iter_features(self) -> Iterator[FeatureProvider]:
        """Hook for subclasses that want to filter or reorder features."""
        for f in self.features:
            yield f

    def as_vector(self) -> Array:
        """Concatenate all feature vectors into an array."""
        vecs: List[np.ndarray] = []
        for f in self._iter_features():
            vecs.append(f.as_vector())
        if not vecs:
            return np.zeros(0, np.float32)

        return np.concatenate(vecs, dtype=np.float32)

    def vector(self) -> Array:  # pragma: no cover
        return self.as_vector()

    def reset(self, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Reset all feature providers to their initial state, if they
        implement a `reset()` method.

        This does *not* recreate the feature list; it just forwards
        the reset to each feature that supports it.
        """
        for feat in self._iter_features():
            feat.reset()

        if overrides is not None:
            self.update(overrides)

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Abstract state-update hook.

        Subclasses (DeviceState, GridState, ...) must implement this with
        their own semantics, e.g.:

            - DeviceState.update(...)        -> device-level feature updates
            - GridState.update(...)          -> grid-wide feature updates
            - GridState.update_local_device(...)

        For convenience, subclasses can reuse `_apply_feature_updates(...)`
        to apply batched `FeatureProvider.set_values(...)` calls.
        """
        raise NotImplementedError

    def update_feature(self, feature_type: Type[FeatureProvider], **values: Any) -> None:
        """
        Update a single feature identified by its class.

        Example:
            state.update_feature(ElectricalBasePh, P_MW=10.0, Q_MVAr=2.0)
        """
        for feat in self._iter_features():
            if isinstance(feat, feature_type):
                feat.set_values(**values)
                # If you might have multiple of the same type, either:
                # - break after first, or
                # - remove this break to update all of them.
                break

    def observe_by(self, agent: AgentLike) -> Tuple[np.ndarray, List[str]]:
        """
        Build an observation (vector, names) as visible to `agent`,
        based on each feature's observability and the agent's identity/authority.

        Contract with features:
            - If a feature implements `is_observable_by(agent, owner_id=...) -> bool`,
              that is used.
            - Otherwise, default policy:
                • owner can see it (agent.agent_id == owner_id or owner_id is None)
                • non-owner cannot.

        Returns:
            (obs_vector, obs_names)
        """
        vecs: List[np.ndarray] = []

        for feat in self._iter_features():
            if "public" in feat.visibility:
                vecs.append(feat.vector())
            if "owner" in feat.visibility:
                if agent.agent_id == self.owner_id:
                    vecs.append(feat.vector())
            if "system" in feat.visibility:
                if agent.level >= 3:
                    vecs.append(feat.vector())
            if "upper_level" in feat.visibility:
                if agent.level == self.owner_level + 1:
                    vecs.append(feat.vector())

        return cat_f32(vecs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the State into a plain dict.

        Structure:
            {
                "owner_id": str,
                "owner_level": int,
                "features": {
                    "<FeatureClassName>": { ... feature.to_dict() ... },
                    ...
                }
            }

        Note:
            Assumes each FeatureProvider in `features` implements `to_dict()`.
        """
        feat_dict: Dict[str, Any] = {}

        for feat in self._iter_features():
            key = type(feat).__name__
            if not hasattr(feat, "to_dict"):
                raise AttributeError(
                    f"Feature {key} does not implement to_dict()."
                )
            feat_dict[key] = feat.to_dict()

        return feat_dict


@dataclass(slots=True)
class DeviceState(State):
    def update(self, updates: Dict[Union[str, Type[FeatureProvider]], Dict[str, Any]]) -> None:
        """
        Apply a batch of updates to features.

        `updates` is a mapping:

            {
                "ElectricalBasePh": {"P_MW": 5.0, "Q_MVAr": 1.0},
                "StatusBlock":      {"state": "online"},
                ...
            }

        For each feature in `self.features`, if its class appears as a key in
        `updates`, we forward the corresponding dict to feature.set_values(**...).
        """
        for feat in self._iter_features():
            feature_type = type(feat)
            values = {}
            if feature_type in updates:
                values = updates.get(feature_type)
            if feature_type.__name__ in updates:
                values = updates.get(feature_type.__name__)
            self.update_feature(feature_type, **values)

@provider()
@dataclass(slots=True)
class GridState:
    """
    GridState — container that aggregates grid-level feature providers
    (BusVoltages, LineFlows, NetworkMetrics) into a unified observation
    vector for grid agents.

    Unlike DeviceState, GridState does not enforce phase context since
    grid-level features are typically aggregated network observables.

    Vector & names:
      • vectors are concatenated in feature order; empty vectors are skipped.
      • names are concatenated in the same order; 1:1 parity enforced per feature.
      • prefix_names=True prepends '<ClassName>.' to each child's names.
    """
    features: List[FeatureProvider] = field(default_factory=list)
    prefix_names: bool = False

    def _iter_ready_features(self) -> Iterator[FeatureProvider]:
        for f in self.features:
            yield f

    def vector(self) -> Array:
        vecs: List[np.ndarray] = []
        for f in self._iter_ready_features():
            v, _ = _vec_names(f)
            if v.size:
                vecs.append(v)
        if not vecs:
            return np.zeros(0, np.float32)
        return np.concatenate(vecs, dtype=np.float32)

    def names(self) -> List[str]:
        out: List[str] = []
        for f in self._iter_ready_features():
            _, n = _vec_names(f)
            if self.prefix_names and n:
                pref = f.__class__.__name__ + "."
                n = [pref + s for s in n]
            out += n
        return out

    def clamp_(self) -> None:
        for f in self._iter_ready_features():
            if hasattr(f, "clamp_"):
                f.clamp_()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prefix_names": self.prefix_names,
            "features": [
                {
                    "kind": f.__class__.__name__,
                    "payload": (
                        f.to_dict() if hasattr(f, "to_dict") else asdict(f)
                    ),
                }
                for f in self.features
            ],
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        registry: Optional[Dict[str, Type[FeatureProvider]]] = None,
    ) -> "GridState":
        reg = registry or KNOWN_FEATURES
        feats: List[FeatureProvider] = []
        for item in d.get("features", []):
            kind = item.get("kind")
            payload = item.get("payload", {})
            cls_ = reg.get(kind)
            if cls_ is None:
                raise ValueError(
                    f"Unknown feature kind '{kind}'. Provide a registry mapping."
                )
            if hasattr(cls_, "from_dict"):
                feats.append(cls_.from_dict(payload))  # type: ignore
            else:
                feats.append(cls_(**payload))          # type: ignore

        return cls(
            features=feats,
            prefix_names=d.get("prefix_names", False),
        )

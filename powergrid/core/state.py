from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Type, Tuple, Optional

import numpy as np

from powergrid.features.base import FeatureProvider, AgentLike
from powergrid.utils.typing import Array
from powergrid.utils.array_utils import _cat_f32


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

    def update(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """
        Apply a batch of updates to features.

        `updates` is a mapping:

            {
                ElectricalBasePh: {"P_MW": 5.0, "Q_MVAr": 1.0},
                StatusBlock:      {"state": "online"},
                ...
            }

        For each feature in `self.features`, if its class appears as a key in
        `updates`, we forward the corresponding dict to feature.set_values(**...).
        """
        for feat in self.features:
            data = updates.get(type(feat).__name__)
            if data:
                # FeatureProvider is assumed to implement set_values(**kwargs)
                feat.set_values(**data)

    def update_feature(self, feature_type: Type[FeatureProvider], **values: Any) -> None:
        """
        Update a single feature identified by its class.

        Example:
            state.update_feature(ElectricalBasePh, P_MW=10.0, Q_MVAr=2.0)
        """
        for feat in self.features:
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

        for feat in self.features:
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

        return _cat_f32(vecs)

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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Type, Tuple, Optional, Union

import numpy as np

from powergrid.agents.base import Agent
from powergrid.features.base import FeatureProvider
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

@dataclass(slots=True)
class State(ABC):
    """Generic agent state, defined by a list of feature providers."""
    owner_id: str
    owner_level: int

    # Raw features that make up this state
    features: List[FeatureProvider] = field(default_factory=list)

    def vector(self) -> np.ndarray:
        """Concatenate all feature vectors into an array."""
        feature_vectors: List[np.ndarray] = []
        for feature in self.features:
            feature_vectors.append(feature.vector())

        return cat_f32(feature_vectors)

    def reset(self, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Reset all feature providers to their initial state, if they
        implement a `reset()` method.

        This does *not* recreate the feature list; it just forwards
        the reset to each feature that supports it.
        """
        for feature in self.features:
            feature.reset()

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

    def update_feature(self, feature_name: str, **values: Any) -> None:
        """
        Update a single feature identified by its class.

        Example:
            state.update_feature(ElectricalBasePh, P_MW=10.0, Q_MVAr=2.0)
        """
        for feature in self.features:
            if feature.feature_name == feature_name:
                feature.set_values(**values)
                # If you might have multiple of the same type, either:
                # - break after first, or
                # - remove this break to update all of them.
                break

    def observed_by(self, requestor_id: str, requestor_level: int) -> Dict[str, np.ndarray]:
        """Build observation dictionary visible to the requesting agent.

        Filters state features based on visibility rules. Each feature's `is_observable_by()`
        method determines if the requestor can see it.

        Args:
            requestor_id: ID of agent requesting observation
            requestor_level: Hierarchy level of requesting agent (1=device, 2=grid, 3=system)

        Returns:
            Dict mapping feature names to observation vectors (float32 numpy arrays).
            Only includes features the requestor is allowed to observe.

        Example:
            >>> state = DeviceState(owner_id="device1", owner_level=1)
            >>> # Owner observes own state
            >>> obs = state.observed_by("device1", requestor_level=1)
            >>> # {"ElectricalBasePh": array([1.0, 0.5, ...]), ...}
            >>>
            >>> # Non-owner with insufficient permissions
            >>> obs = state.observed_by("device2", requestor_level=1)
            >>> # {} (empty - no observable features)
        """
        observable_feature_dict = {}

        for feature in self.features:
            if feature.is_observable_by(requestor_id, requestor_level, self.owner_id, self.owner_level):
                observable_feature_dict[feature.feature_name] = cat_f32(feature.vector())
        
        self.validate_observation_dict(observable_feature_dict)
        return observable_feature_dict

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Serialize the State into a plain dict.

        Structure:
            {
                "features": {
                    "<FeatureClassName>": { ... feature.to_dict() ... },
                    ...
                }
            }
        """
        feature_dict: Dict[str, Any] = {}

        for feature in self.features:
            feature_dict[feature.feature_name] = feature.to_dict()

        return feature_dict

    def validate_observation_dict(self, obs_dict: Dict[str, np.ndarray]) -> None:
        """Validate the collected feature vectors for consistency."""
        pass

@dataclass(slots=True)
class DeviceState(State):
    def update(self, updates: Dict[str, Dict[str, Any]]) -> None:
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
        for feature in self.features:
            if feature.feature_name in updates:
                values = updates.get(feature.feature_name)
                self.update_feature(feature.feature_name, **values)

    def validate_observation_dict(self, obs_dict: Dict[str, np.ndarray]) -> None:
        for vector in obs_dict.values():
            if not (isinstance(vector, np.ndarray) and vector.ndim == 1):
                raise NotImplementedError(
                    "Only 1D vector observations supported. "
                    "Got: " + str(type(vector))
                )

@dataclass(slots=True)
class GridState(State):
    def update(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """Apply batch updates to grid-level features.

        Args:
            updates: Dict mapping feature names to field updates:
                {
                    "ElectricalBasePh": {"P_MW": 5.0, "Q_MVAr": 1.0},
                    "StatusBlock": {"state": "online"},
                    ...
                }

        Each feature in self.features that matches a key in updates will have
        its corresponding field values updated via feature.set_values(**values).
        """
        for feature in self.features:
            if feature.feature_name in updates:
                values = updates.get(feature.feature_name)
                self.update_feature(feature.feature_name, **values)
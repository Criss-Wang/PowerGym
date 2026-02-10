"""State abstractions for agent state management.

This module provides generic state containers that compose FeatureProviders
and support visibility-based observation filtering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from heron.core.feature import FeatureProvider
from heron.utils.array_utils import cat_f32


@dataclass(slots=True)
class State(ABC):
    """Generic agent state, defined by a list of feature providers.

    States aggregate multiple FeatureProviders and provide:
    - Vector representation for ML
    - Visibility-filtered observations
    - Batch update operations
    - Serialization

    Attributes:
        owner_id: ID of the agent that owns this state
        owner_level: Hierarchy level of owning agent (1=field, 2=coordinator, 3=system)
        features: List of feature providers composing this state

    Example:
        Create a field agent state with custom features::

            import numpy as np
            from heron.core.state import FieldAgentState
            from heron.core.feature import FeatureProvider

            # Define a custom feature (feature_name auto-set to class name)
            class BatteryFeature(FeatureProvider):
                visibility = ["owner"]

                def __init__(self):
                    self.soc = 0.5  # State of charge
                    self.capacity = 100.0

                def vector(self):
                    return np.array([self.soc, self.capacity], dtype=np.float32)

                def names(self):
                    return ["soc", "capacity"]

                def to_dict(self):
                    return {"soc": self.soc, "capacity": self.capacity}

                @classmethod
                def from_dict(cls, d):
                    f = cls()
                    f.soc = d.get("soc", 0.5)
                    f.capacity = d.get("capacity", 100.0)
                    return f

                def set_values(self, **kwargs):
                    for k, v in kwargs.items():
                        if hasattr(self, k):
                            setattr(self, k, v)

            # Create state and add feature
            state = FieldAgentState(owner_id="battery_1", owner_level=1)
            state.features.append(BatteryFeature())

            # Get full state vector
            vec = state.vector()  # array([0.5, 100.0])

            # Update specific feature
            state.update_feature("BatteryFeature", soc=0.8)
    """
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
        """Reset all feature providers to their initial state.

        This does not recreate the feature list; it just forwards
        the reset to each feature that supports it.

        Args:
            overrides: Optional dict mapping feature names to override values
        """
        for feature in self.features:
            feature.reset()

        if overrides is not None:
            self.update(overrides)

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Abstract state-update hook.

        Subclasses must implement this with their own semantics.
        For convenience, subclasses can use `update_feature(...)` for
        individual feature updates.
        """
        raise NotImplementedError

    def update_feature(self, feature_name: str, **values: Any) -> None:
        """Update a single feature identified by its class name.

        Args:
            feature_name: Feature class name string
            **values: Field values to update

        Example:
            state.update_feature("SomeFeature", field1=10.0, field2=2.0)
        """
        for feature in self.features:
            if feature.feature_name == feature_name:
                feature.set_values(**values)
                break

    def observed_by(self, requestor_id: str, requestor_level: int) -> Dict[str, np.ndarray]:
        """Build observation dictionary visible to the requesting agent.

        Filters state features based on visibility rules. Each feature's
        `is_observable_by()` method determines if the requestor can see it.

        Args:
            requestor_id: ID of agent requesting observation
            requestor_level: Hierarchy level of requesting agent
                           (1=field, 2=coordinator, 3=system)

        Returns:
            Dict mapping feature names to observation vectors (float32 numpy arrays).
            Only includes features the requestor is allowed to observe.

        Example:
            >>> state = FieldAgentState(owner_id="agent1", owner_level=1)
            >>> # Owner observes own state
            >>> obs = state.observed_by("agent1", requestor_level=1)
            >>> # {"SomeFeature": array([1.0, 0.5, ...]), ...}
            >>>
            >>> # Non-owner with insufficient permissions
            >>> obs = state.observed_by("agent2", requestor_level=1)
            >>> # {} (empty - no observable features)
        """
        observable_feature_dict = {}

        for feature in self.features:
            if feature.is_observable_by(
                requestor_id, requestor_level, self.owner_id, self.owner_level
            ):
                observable_feature_dict[feature.feature_name] = cat_f32([feature.vector()])

        self.validate_observation_dict(observable_feature_dict)
        return observable_feature_dict

    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Serialize the State into a plain dict.

        Args:
            include_metadata: If True, includes _owner_id and _owner_level for reconstruction

        Returns:
            Dict mapping feature names to their serialized representations.
            If include_metadata=True, includes "_owner_id", "_owner_level", "_state_type"
        """
        feature_dict: Dict[str, Any] = {}

        for feature in self.features:
            feature_dict[feature.feature_name] = feature.to_dict()

        if include_metadata:
            return {
                "_owner_id": self.owner_id,
                "_owner_level": self.owner_level,
                "_state_type": self.__class__.__name__,
                "features": feature_dict
            }

        return feature_dict

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "State":
        """Reconstruct State object from serialized dict.

        Args:
            state_dict: Serialized state with metadata format:
                {"_owner_id": str, "_owner_level": int, "_state_type": str, "features": {...}}
                OR simple format: {"FeatureName": {...}, ...}

        Returns:
            Reconstructed State object with FeatureProvider instances

        Raises:
            ValueError: If feature class not found in registry
        """
        from heron.core.feature import get_feature_class

        # Extract metadata if present
        if "_owner_id" in state_dict and "_owner_level" in state_dict:
            owner_id = state_dict["_owner_id"]
            owner_level = state_dict["_owner_level"]
            state_type = state_dict.get("_state_type")
            features_dict = state_dict.get("features", {})
        else:
            # Fallback: no metadata, assume features are at top level
            owner_id = state_dict.get("owner_id", "unknown")
            owner_level = state_dict.get("owner_level", 1)
            state_type = None
            features_dict = {k: v for k, v in state_dict.items() if not k.startswith("_") and k not in ["owner_id", "owner_level"]}

        # Determine which State class to instantiate
        if state_type and cls == State:
            # If called on base State class, use the type from metadata
            # Use globals() for late binding (classes defined later in file)
            import sys
            current_module = sys.modules[__name__]
            state_class = getattr(current_module, state_type, None)
            if state_class is None:
                raise ValueError(f"Unknown state type: {state_type}")
        else:
            # Called on concrete subclass, use it directly
            state_class = cls

        # Create State instance
        state = state_class(owner_id=owner_id, owner_level=owner_level)

        # Reconstruct features using registry
        for feature_name, feature_data in features_dict.items():
            if feature_name.startswith("_"):
                continue  # Skip internal metadata fields

            try:
                feature_class = get_feature_class(feature_name)
                feature_obj = feature_class.from_dict(feature_data)
                state.features.append(feature_obj)
            except ValueError as e:
                # Feature not registered - skip with warning
                print(f"Warning: Skipping unregistered feature '{feature_name}': {e}")

        return state

    def validate_observation_dict(self, obs_dict: Dict[str, np.ndarray]) -> None:
        """Validate the collected feature vectors for consistency.

        Override in subclasses to add custom validation.
        """
        pass


@dataclass(slots=True)
class FieldAgentState(State):
    """State for field-level (L1) agents.

    Field agents are the lowest level in the hierarchy, typically
    representing individual devices or units.
    """
    def update(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """Apply batch updates to features.

        Args:
            updates: Mapping of feature names to field updates:
                {
                    "FeatureA": {"field1": 5.0, "field2": 1.0},
                    "FeatureB": {"state": "active"},
                    ...
                }
        """
        for feature in self.features:
            if feature.feature_name in updates:
                values = updates.get(feature.feature_name)
                if values is not None:
                    self.update_feature(feature.feature_name, **values)

    def validate_observation_dict(self, obs_dict: Dict[str, np.ndarray]) -> None:
        """Validate that all observation vectors are 1D."""
        for vector in obs_dict.values():
            if not (isinstance(vector, np.ndarray) and vector.ndim == 1):
                raise NotImplementedError(
                    "Only 1D vector observations supported. "
                    "Got: " + str(type(vector))
                )


@dataclass(slots=True)
class CoordinatorAgentState(State):
    """State for coordinator-level (L2) agents.

    Coordinator agents manage groups of field agents and aggregate
    information from their subordinates.
    """
    def update(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """Apply batch updates to coordinator-level features.

        Args:
            updates: Mapping of feature names to field updates
        """
        for feature in self.features:
            if feature.feature_name in updates:
                values = updates.get(feature.feature_name)
                if values is not None:
                    self.update_feature(feature.feature_name, **values)


@dataclass(slots=True)
class SystemAgentState(State):
    """State for system-level (L3) agents.

    System agents are the top level in the hierarchy, typically
    representing system-wide state like market conditions, grid frequency,
    or aggregate metrics across multiple coordinators.

    Example:
        Create a system agent state with system-level features::

            from heron.core.state import SystemAgentState
            from heron.core.feature import FeatureProvider

            class SystemFrequency(FeatureProvider):
                visibility = ["system"]

                def __init__(self, frequency_hz: float = 60.0):
                    self.frequency_hz = frequency_hz
                    self.nominal_hz = 60.0

                def vector(self):
                    deviation = self.frequency_hz - self.nominal_hz
                    return np.array([self.frequency_hz, deviation], dtype=np.float32)

                def names(self):
                    return ["frequency_hz", "frequency_deviation"]

                def to_dict(self):
                    return {"frequency_hz": self.frequency_hz}

                @classmethod
                def from_dict(cls, d):
                    return cls(frequency_hz=d.get("frequency_hz", 60.0))

                def set_values(self, **kwargs):
                    if "frequency_hz" in kwargs:
                        self.frequency_hz = kwargs["frequency_hz"]

            # Create state and add features
            state = SystemAgentState(owner_id="system_1", owner_level=3)
            state.features.append(SystemFrequency(frequency_hz=60.0))
    """
    def update(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """Apply batch updates to system-level features.

        Args:
            updates: Mapping of feature names to field updates:
                {
                    "SystemFrequency": {"frequency_hz": 59.95},
                    "AggregateLoad": {"total_mw": 1500.0},
                    ...
                }
        """
        for feature in self.features:
            if feature.feature_name in updates:
                values = updates.get(feature.feature_name)
                if values is not None:
                    self.update_feature(feature.feature_name, **values)



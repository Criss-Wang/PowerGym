"""Base abstract class for feature providers.

Feature providers represent observable/controllable attributes of agents
(e.g., internal state, sensor readings, parameters).

The HERON framework supports 4 visibility levels:
- public: Visible to all agents
- owner: Only visible to the owning agent
- upper_level: Visible to agents one level above in hierarchy
- system: Only visible to system-level (L3) agents
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Sequence

import numpy as np


class FeatureProvider(ABC):
    """Abstract base class for feature providers used in agent states.

    Feature providers encapsulate observable state attributes with:
    - Vectorization for ML observations
    - Observability rules for multi-agent visibility
    - Serialization for communication/logging

    Subclasses must implement:
        - vector(): Convert state to numpy array
        - names(): Get field names corresponding to vector elements
        - to_dict(): Serialize to dictionary
        - from_dict(): Deserialize from dictionary
        - set_values(): Update state from keyword arguments

    Attributes:
        visibility: Class variable defining who can observe this feature
                   Options: "public", "owner", "system", "upper_level"
        feature_name: Auto-set to class name for registration/lookup

    Example:
        Define a custom feature for battery state::

            import numpy as np
            from heron.core.feature import FeatureProvider

            class BatteryState(FeatureProvider):
                # Visible to owner and upper-level agents
                visibility = ["owner", "upper_level"]

                def __init__(self, capacity: float = 100.0):
                    self.soc = 0.5  # State of charge [0, 1]
                    self.capacity = capacity
                    self.power = 0.0

                def vector(self) -> np.ndarray:
                    return np.array([self.soc, self.power], dtype=np.float32)

                def names(self):
                    return ["soc", "power"]

                def to_dict(self):
                    return {"soc": self.soc, "capacity": self.capacity, "power": self.power}

                @classmethod
                def from_dict(cls, d):
                    f = cls(d.get("capacity", 100.0))
                    f.soc = d.get("soc", 0.5)
                    f.power = d.get("power", 0.0)
                    return f

                def set_values(self, **kwargs):
                    for k, v in kwargs.items():
                        if hasattr(self, k):
                            setattr(self, k, v)

            # Usage
            battery = BatteryState(capacity=200.0)
            battery.set_values(soc=0.8, power=50.0)
            print(battery.vector())  # [0.8, 50.0]
    """

    visibility: ClassVar[Sequence[str]]
    feature_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.feature_name = cls.__name__

    @abstractmethod
    def vector(self) -> np.ndarray:
        """Convert feature state to numpy array for observations.

        Returns:
            1D numpy array of feature values
        """
        pass

    @abstractmethod
    def names(self) -> List[str]:
        """Get names of fields in the vector representation.

        Returns:
            List of field names corresponding to vector elements
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize feature to dictionary.

        Returns:
            Dictionary representation of feature state
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureProvider":
        """Deserialize feature from dictionary.

        Args:
            d: Dictionary representation

        Returns:
            Feature provider instance
        """
        pass

    @abstractmethod
    def set_values(self, **kwargs: Any) -> None:
        """Update feature values from keyword arguments.

        Args:
            **kwargs: Field names and values to update
        """
        pass

    def reset(self, **overrides: Any) -> "FeatureProvider":
        """Reset feature to initial/neutral state.

        Default implementation does nothing. Subclasses should override
        to provide meaningful reset behavior.

        Args:
            **overrides: Optional overrides to apply after reset

        Returns:
            self for chaining
        """
        return self

    def is_observable_by(
        self,
        requestor_id: str,
        requestor_level: int,
        owner_id: str,
        owner_level: int
    ) -> bool:
        """Check if this feature is observable by the requesting agent.

        Visibility rules are OR-ed together. A feature with visibility
        ["owner", "upper_level"] is visible to both the owner AND agents
        one level above.

        Visibility options:
            - "public": All agents can observe
            - "owner": Owner (requestor_id == owner_id) can observe
            - "system": System-level agents (level >= 3) can observe
            - "upper_level": Agents one level above owner can observe

        Args:
            requestor_id: ID of agent requesting observation
            requestor_level: Hierarchy level of requesting agent
                           (1=field, 2=coordinator, 3=system)
            owner_id: ID of agent that owns this feature
            owner_level: Hierarchy level of owning agent

        Returns:
            True if requestor can observe this feature, False otherwise
        """
        # OR logic: any matching visibility grants access
        if "public" in self.visibility:
            return True
        if "owner" in self.visibility and requestor_id == owner_id:
            return True
        if "system" in self.visibility and requestor_level >= 3:
            return True
        if "upper_level" in self.visibility and requestor_level == owner_level + 1:
            return True
        # Default: no visibility rules matched
        return False

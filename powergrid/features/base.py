"""Base abstract class for feature providers.

Feature providers represent observable/controllable attributes of agents
(e.g., electrical state, power limits, storage parameters).
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Sequence

import numpy as np

from powergrid.agents.base import Agent


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
    """

    visibility: ClassVar[Sequence[str]]

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

    def is_observable_by(self, requestor_id: str, requestor_level: int, owner_id: str, owner_level: int) -> bool:
        """Check if this feature is observable by the requesting agent.

        Visibility rules (checked in order):
            - "public": All agents can observe
            - "owner": Only owner (requestor_id == owner_id) can observe
            - "system": System-level agents (level >= 3) can observe
            - "upper_level": Agents one level above owner can observe
            - Default: No one can observe (private)

        Args:
            requestor_id: ID of agent requesting observation
            requestor_level: Hierarchy level of requesting agent (1=device, 2=grid, 3=system)
            owner_id: ID of agent that owns this feature
            owner_level: Hierarchy level of owning agent

        Returns:
            True if requestor can observe this feature, False otherwise
        """
        if "public" in self.visibility:
            return True
        if "owner" in self.visibility:
            return requestor_id == owner_id
        if "system" in self.visibility:
            return requestor_level >= 3
        if "upper_level" in self.visibility:
            return requestor_level == owner_level + 1
        # Default: treat as private
        return False

    @property
    def feature_name(self) -> str:
        """Get the feature type name (class name).

        Returns:
            String name of the feature class
        """
        return type(self).__name__  
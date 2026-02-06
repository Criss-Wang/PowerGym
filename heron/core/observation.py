"""Observation abstraction for agent observations.

This module defines the core data structure for agent observations
in the HERON framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


# =============================================================================
# Observation.local key constants
# =============================================================================

# FieldAgent local keys
OBS_KEY_STATE = "state"  # Agent's state vector
OBS_KEY_OBSERVATION = "observation"  # Agent's observation vector
OBS_KEY_PROXY_STATE = "proxy_state"  # State from proxy (delayed observations)

# CoordinatorAgent local keys
OBS_KEY_SUBORDINATE_OBS = "subordinate_obs"  # Dict of subordinate observations
OBS_KEY_COORDINATOR_STATE = "coordinator_state"  # Coordinator's state vector

# SystemAgent local keys
OBS_KEY_COORDINATOR_OBS = "coordinator_obs"  # Dict of coordinator observations
OBS_KEY_SYSTEM_STATE = "system_state"  # System agent's state vector


@dataclass
class Observation:
    """Structured observation for an agent.

    Observations separate local and global information, supporting both
    centralized training (full visibility) and decentralized execution
    (local-only).

    Attributes:
        local: Local agent state (e.g., device measurements, internal state)
        global_info: Global information visible to this agent (e.g., shared metrics)
        timestamp: Current simulation time

    Example:
        Create an observation with local sensor data::

            import numpy as np
            from heron.core import Observation

            obs = Observation(
                local={"voltage": 1.02, "power": np.array([100.0, 50.0])},
                global_info={"grid_frequency": 60.0},
                timestamp=10.5
            )

            # Convert to flat vector for RL algorithms
            vec = obs.vector()  # array([100., 50., 1.02, 60.], dtype=float32)

        Nested observations are flattened recursively::

            obs = Observation(
                local={
                    "sensors": {"temp": 25.0, "pressure": 101.3},
                    "status": 1
                }
            )
            vec = obs.vector()  # Flattens all numeric values
    """
    local: Dict[str, Any] = field(default_factory=dict)
    global_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def vector(self) -> np.ndarray:
        """Convert observation to flat numpy array for RL algorithms.

        Returns:
            Flattened observation vector
        """
        parts: list = []

        # Flatten local state
        self._flatten_dict_to_list(self.local, parts)

        # Flatten global info
        self._flatten_dict_to_list(self.global_info, parts)

        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def as_vector(self) -> np.ndarray:
        """Alias for vector() method.

        Returns:
            Flattened observation vector
        """
        return self.vector()

    def _flatten_dict_to_list(self, d: Dict, parts: list) -> None:
        """Recursively flatten a dictionary into a list of arrays.

        Args:
            d: Dictionary to flatten
            parts: List to append arrays to (modified in place)
        """
        for key in sorted(d.keys()):
            val = d[key]
            if isinstance(val, (int, float)):
                parts.append(np.array([val], dtype=np.float32))
            elif isinstance(val, np.ndarray):
                parts.append(val.ravel().astype(np.float32))
            elif isinstance(val, dict):
                # Recursively flatten nested dicts
                self._flatten_dict_to_list(val, parts)

    # =========================================================================
    # Serialization Methods (for async message passing)
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize observation to dictionary for message passing.

        Used in fully async event-driven mode (Option B with async_observations=True)
        where observations are sent via message broker instead of direct method calls.

        **Tricky Part - Nested Observation Serialization**:
        Subordinate observations in coordinator's local dict may themselves be
        Observation objects. These are recursively serialized. When deserializing,
        the receiver must know the structure to reconstruct nested Observations.

        **Follow-up Work Needed**:
        - Add schema/type hints to payload so receiver knows which fields are
          nested Observations vs plain dicts
        - Consider using a more robust serialization format (e.g., pickle, msgpack)
          for complex observation structures

        Returns:
            Serialized observation as dictionary
        """
        return {
            "timestamp": self.timestamp,
            "local": self._serialize_nested(self.local),
            "global_info": self._serialize_nested(self.global_info),
        }

    def _serialize_nested(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize nested structures for message passing.

        Args:
            data: Dictionary to serialize

        Returns:
            Serialized dictionary with numpy arrays converted to lists
        """
        result = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                result[k] = {"__type__": "ndarray", "data": v.tolist(), "dtype": str(v.dtype)}
            elif isinstance(v, Observation):
                result[k] = {"__type__": "Observation", "data": v.to_dict()}
            elif isinstance(v, dict):
                result[k] = self._serialize_nested(v)
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        """Deserialize observation from dictionary.

        Used to reconstruct Observation from message broker payload.

        **Tricky Part - Type Reconstruction**:
        The serialized dict uses "__type__" markers to indicate special types
        (ndarray, Observation). Without these markers, arrays would remain as
        lists and nested Observations would remain as dicts.

        **Follow-up Work Needed**:
        - Handle missing "__type__" markers gracefully (backward compatibility)
        - Add validation for expected observation structure

        Args:
            d: Serialized observation dictionary

        Returns:
            Reconstructed Observation object
        """
        return cls(
            timestamp=d.get("timestamp", 0.0),
            local=cls._deserialize_nested(d.get("local", {})),
            global_info=cls._deserialize_nested(d.get("global_info", {})),
        )

    @classmethod
    def _deserialize_nested(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively deserialize nested structures from message payload.

        Args:
            data: Serialized dictionary

        Returns:
            Deserialized dictionary with types reconstructed
        """
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                if v.get("__type__") == "ndarray":
                    result[k] = np.array(v["data"], dtype=v.get("dtype", "float32"))
                elif v.get("__type__") == "Observation":
                    result[k] = cls.from_dict(v["data"])
                else:
                    result[k] = cls._deserialize_nested(v)
            else:
                result[k] = v
        return result

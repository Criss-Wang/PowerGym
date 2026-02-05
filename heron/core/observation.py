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

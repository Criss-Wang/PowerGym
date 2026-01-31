"""Observation abstraction for agent observations.

This module defines the core data structure for agent observations
in the HERON framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class Observation:
    """Structured observation for an agent.

    Attributes:
        local: Local agent state (e.g., device measurements, internal state)
        global_info: Global information visible to this agent (e.g., shared metrics)
        timestamp: Current simulation time
    """
    local: Dict[str, Any] = field(default_factory=dict)
    global_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def vector(self) -> np.ndarray:
        """Convert observation to flat numpy array for RL algorithms.

        Returns:
            Flattened observation vector
        """
        vec = np.array([], dtype=np.float32)

        # Flatten local state
        vec = self._flatten_dict(self.local, vec)

        # Flatten global info
        vec = self._flatten_dict(self.global_info, vec)

        return vec.astype(np.float32)

    def as_vector(self) -> np.ndarray:
        """Alias for vector() method.

        Returns:
            Flattened observation vector
        """
        return self.vector()

    def _flatten_dict(self, d: Dict, vec: np.ndarray) -> np.ndarray:
        """Recursively flatten a dictionary into a numpy array.

        Args:
            d: Dictionary to flatten
            vec: Current vector to append to

        Returns:
            Updated vector
        """
        for key in sorted(d.keys()):
            val = d[key]
            if isinstance(val, (int, float)):
                vec = np.append(vec, np.float32(val))
            elif isinstance(val, np.ndarray):
                vec = np.concatenate([vec, val.ravel().astype(np.float32)])
            elif isinstance(val, dict):
                # Recursively flatten nested dicts
                vec = self._flatten_dict(val, vec)
        return vec

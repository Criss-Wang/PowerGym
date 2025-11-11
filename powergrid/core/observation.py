"""Observation and message abstractions for agent communication.

This module defines the core data structures for agent observations and
inter-agent communication in the PowerGrid platform.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from powergrid.utils.typing import AgentID, Array


@dataclass
class Observation:
    """Structured observation for an agent.

    Attributes:
        local: Local agent state (e.g., device P, Q, SOC)
        global_info: Global information visible to this agent (e.g., bus voltages)
        messages: Communication messages from other agents
        timestamp: Current simulation time
    """
    local: Dict[str, Any] = field(default_factory=dict)
    global_info: Dict[str, Any] = field(default_factory=dict)
    messages: List['Message'] = field(default_factory=list)
    timestamp: float = 0.0

    def as_vector(self) -> Array:
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


@dataclass
class Message:
    """Inter-agent communication message.

    Attributes:
        sender: ID of sending agent
        content: Message payload (e.g., price signals, setpoints, constraints)
        timestamp: Time when message was sent
    """
    sender: AgentID
    content: Dict[str, Any]
    recipient: Optional[Union[AgentID, List[AgentID]]] = None  # None = broadcast
    timestamp: float = 0.0

    # TODO: add more attributes like expiration, priority, etc.

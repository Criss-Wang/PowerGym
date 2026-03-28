"""Policy interfaces for agent decision-making.

This module provides abstract policy interfaces and common implementations
for agent control in multi-agent systems.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Callable
from functools import wraps

import numpy as np

from heron.core.action import Action
from heron.core.observation import Observation


def obs_to_vector(method: Callable) -> Callable:
    """Decorator that converts observation to vector before calling method.

    The decorated method receives obs_vec (np.ndarray) instead of observation.
    Requires self to have obs_dim and extract_obs_vector() method.
    """
    @wraps(method)
    def wrapper(self, observation, *args, **kwargs):
        obs_vec = self.extract_obs_vector(observation, self.obs_dim)
        return method(self, obs_vec, *args, **kwargs)
    return wrapper


def vector_to_action(method: Callable) -> Callable:
    """Decorator that converts returned vector to Action object.

    The decorated method returns np.ndarray, which gets converted to Action.
    Requires self to have action_dim, action_range, and vec_to_action() method.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        action_vec = method(self, *args, **kwargs)
        return self.vec_to_action(action_vec, self.action_dim, self.action_range)
    return wrapper


class Policy(ABC):
    """Abstract policy interface for agent decision-making.

    Policies can be:
    - Learned (RL algorithms)
    - Rule-based (heuristics, classical control)
    - Optimization-based (MPC, optimal control)

    For vector-based policies (most RL algorithms), subclasses can override
    _compute_action_vector() instead of forward() to get automatic observation
    extraction and action conversion.

    Attributes:
        observation_mode: Controls which observation components to use:
            - "full": Use both local and global (default, for centralized training)
            - "local": Use only local observations (for decentralized policies)
            - "global": Use only global information
    """
    observation_mode: str = "full"  # Default to full observations

    @abstractmethod
    def forward(self, observation: Observation) -> Optional[Action]:
        """Compute action from observation.

        Args:
            observation: Agent observation

        Returns:
            Action object, or None if no action is computed
        """
        pass

    def reset(self) -> None:
        """Reset policy state (e.g., hidden states for RNNs)."""
        pass

    # ============================================
    # Helper Methods for Policy Implementations
    # ============================================

    def extract_obs_vector(self, observation: Any, obs_dim: int) -> np.ndarray:
        """Extract observation vector from various formats.

        This is a common helper for policies that work with vector observations.
        Handles multiple formats for compatibility between training and deployment modes.

        Uses self.observation_mode to determine which observation components to extract:
        - "full": local + global (default)
        - "local": local only (for decentralized policies)
        - "global": global only

        Args:
            observation: Observation in various formats (Observation object, dict, or array)
            obs_dim: Expected observation dimension

        Returns:
            Numpy array of shape (obs_dim,)
        """
        if isinstance(observation, Observation):
            # Use appropriate vectorization based on observation_mode
            if self.observation_mode == "local":
                return observation.local_vector()
            elif self.observation_mode == "global":
                return observation.global_vector()
            else:  # "full" or any other value
                return observation.vector()
        elif isinstance(observation, dict):
            # Event-driven mode: observation from proxy.get_observation()
            # Structure: {"local": {"FeatureName": array([...])}, "global_info": ...}
            parts: list = []

            # Collect local features (sorted by key, matching Observation convention)
            include_local = self.observation_mode in ("full", "local")
            if include_local and "local" in observation and observation["local"]:
                self._flatten_obs_dict(observation["local"], parts)

            # Collect global features when mode requires it
            include_global = self.observation_mode in ("full", "global")
            if include_global and "global_info" in observation and observation["global_info"]:
                self._flatten_obs_dict(observation["global_info"], parts)

            if not parts:
                return np.zeros(obs_dim, dtype=np.float32)
            vec = np.concatenate(parts).astype(np.float32)
            return vec[:obs_dim] if len(vec) > obs_dim else vec
        elif isinstance(observation, np.ndarray) and observation.size > 0:
            # Ensure array matches expected dimension
            return observation[:obs_dim] if len(observation) > obs_dim else observation
        else:
            # Fallback to zeros
            return np.zeros(obs_dim, dtype=np.float32)

    @staticmethod
    def _flatten_obs_dict(d: dict, parts: list) -> None:
        """Recursively flatten a dict of observation features into a list of arrays.

        Mirrors Observation._flatten_dict_to_list: iterates keys in sorted order
        so the vector layout is deterministic across calls.
        """
        for key in sorted(d.keys()):
            val = d[key]
            if isinstance(val, (int, float, np.number)):
                parts.append(np.array([val], dtype=np.float32))
            elif isinstance(val, np.ndarray):
                parts.append(val.ravel().astype(np.float32))
            elif isinstance(val, (list, tuple)):
                parts.append(np.array(val, dtype=np.float32).ravel())
            elif isinstance(val, dict):
                Policy._flatten_obs_dict(val, parts)

    def vec_to_action(self, action_vec: np.ndarray, action_dim: int,
                     action_range: tuple = (-1.0, 1.0)) -> Action:
        """Convert action vector to Action object.

        Args:
            action_vec: Action values as numpy array
            action_dim: Dimension of continuous action
            action_range: Tuple of (min, max) bounds for actions

        Returns:
            Action object with specified values and specs
        """
        action = Action()
        action.set_specs(
            dim_c=action_dim,
            range=(np.full(action_dim, action_range[0]), np.full(action_dim, action_range[1]))
        )
        action.set_values(action_vec)
        return action
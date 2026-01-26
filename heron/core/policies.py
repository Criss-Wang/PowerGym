"""Policy interfaces for agent decision-making.

This module provides abstract policy interfaces and common implementations
for agent control in multi-agent systems.
"""

from abc import ABC, abstractmethod
from typing import Any

from heron.core.observation import Observation


class Policy(ABC):
    """Abstract policy interface for agent decision-making.

    Policies can be:
    - Learned (RL algorithms)
    - Rule-based (heuristics, classical control)
    - Optimization-based (MPC, optimal control)
    """

    @abstractmethod
    def forward(self, observation: Observation) -> Any:
        """Compute action from observation.

        Args:
            observation: Agent observation

        Returns:
            Action
        """
        pass

    def reset(self) -> None:
        """Reset policy state (e.g., hidden states for RNNs)."""
        pass


class RandomPolicy(Policy):
    """Random policy that samples uniformly from action space.

    Useful for testing and baseline comparisons.
    """

    def __init__(self, action_space, seed=None):
        """Initialize random policy.

        Args:
            action_space: Gymnasium action space
            seed: Optional random seed for reproducibility
        """
        self.action_space = action_space
        if seed is not None:
            self.action_space.seed(seed)

    def forward(self, observation: Observation) -> Any:
        """Sample random action from action space.

        Args:
            observation: Agent observation (ignored)

        Returns:
            Random action sampled from action space
        """
        return self.action_space.sample()

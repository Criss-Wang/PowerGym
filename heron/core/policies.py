"""Policy interfaces for agent decision-making.

This module provides abstract policy interfaces and common implementations
for agent control in multi-agent systems.
"""

from abc import ABC, abstractmethod
from typing import Optional

from heron.core.action import Action
from heron.core.observation import Observation


class Policy(ABC):
    """Abstract policy interface for agent decision-making.

    Policies can be:
    - Learned (RL algorithms)
    - Rule-based (heuristics, classical control)
    - Optimization-based (MPC, optimal control)
    """

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
        # Create Action template from gym space
        self._action = Action.from_gym_space(action_space)

    def forward(self, observation: Observation) -> Optional[Action]:
        """Sample random action from action space.

        Args:
            observation: Agent observation (ignored)

        Returns:
            Action object with randomly sampled values
        """
        self._action.sample()
        return self._action.copy()

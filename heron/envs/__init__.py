"""Environment classes for the multi-agent framework.

This module provides environment implementations:
- EnvCore: Core environment functionality mixin
- MultiAgentEnv: Multi-agent environment base
- PettingZooParallelEnv: PettingZoo parallel env adapter
"""

from heron.envs.base import EnvCore, MultiAgentEnv
from heron.envs.adapters import PettingZooParallelEnv

__all__ = [
    "EnvCore",
    "MultiAgentEnv",
    "PettingZooParallelEnv",
]

"""Environment classes for the multi-agent framework.

This module provides environment implementations:
- BaseEnv: Abstract base environment
- DefaultHeronEnv: Default environment with automatic state bridge
"""

from heron.envs.base import BaseEnv
from heron.envs.builder import EnvBuilder
from heron.envs.simple import DefaultHeronEnv

__all__ = [
    "BaseEnv",
    "DefaultHeronEnv",
    "EnvBuilder",
]

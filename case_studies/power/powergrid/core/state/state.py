"""Power grid specific state types.

This module provides power-grid domain aliases for the generic HERON state types.
"""

from heron.core.state import FieldAgentState, CoordinatorAgentState, SystemAgentState

# Power grid domain aliases
DeviceState = FieldAgentState
GridState = CoordinatorAgentState
GridSystemState = SystemAgentState

__all__ = ["DeviceState", "GridState", "GridSystemState"]

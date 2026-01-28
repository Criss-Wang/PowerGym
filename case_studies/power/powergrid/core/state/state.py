"""Power grid specific state types.

This module provides power-grid domain aliases for the generic HERON state types.
"""

from heron.core.state import FieldAgentState, CoordinatorAgentState

# Power grid domain aliases
DeviceState = FieldAgentState
GridState = CoordinatorAgentState

__all__ = ["DeviceState", "GridState"]

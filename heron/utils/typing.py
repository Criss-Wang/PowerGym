"""Type aliases for the HERON framework."""

from typing import Any


# Agent ID type
AgentID = str


def float_if_not_none(x: Any) -> Any:
    """Convert to float if not None."""
    return None if x is None else float(x)

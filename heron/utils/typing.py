"""Type aliases for the HERON framework."""

from typing import Any, Literal


# Agent and control types
AgentID = str

# Control mode for inverter-based sources
CtrlMode = Literal["q_set", "pf_set", "volt_var", "off"]


def float_if_not_none(x: Any) -> Any:
    """Convert to float if not None."""
    return None if x is None else float(x)

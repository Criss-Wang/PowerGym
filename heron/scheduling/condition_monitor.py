"""Condition monitor for event-driven scheduling.

A ConditionMonitor watches a state predicate and fires a CONDITION_TRIGGER
event when the predicate becomes true. Evaluated by SystemAgent after
SIMULATION and ENV_UPDATE events.
"""

import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from heron.utils.typing import AgentID


def _extract_nested(state: Dict[str, Any], key_path: List[str]) -> Any:
    """Safely traverse a nested dict by key path. Returns None on missing key."""
    current = state
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


@dataclass
class ConditionMonitor:
    """Monitors a state condition and fires CONDITION_TRIGGER when met.

    Evaluated after every SIMULATION and ENV_UPDATE event by SystemAgent.

    Attributes:
        monitor_id: Unique identifier for this monitor.
        agent_id: Agent to wake when condition is met.
        condition_fn: Function ``(state_dict) -> bool``.
        cooldown: Minimum seconds between consecutive triggers (prevents spam).
        one_shot: If True, deregister after first trigger.
        preempt_next_tick: If True, cancel the agent's next scheduled
            AGENT_TICK when this condition fires.
    """

    monitor_id: str
    agent_id: AgentID
    condition_fn: Callable[[Dict[str, Any]], bool]
    cooldown: float = 0.0
    one_shot: bool = False
    preempt_next_tick: bool = False
    _last_triggered: float = field(default=-float("inf"), repr=False)

    @classmethod
    def threshold(
        cls,
        monitor_id: str,
        agent_id: AgentID,
        key_path: List[str],
        threshold: float,
        direction: str = "below",
        cooldown: float = 0.0,
        one_shot: bool = False,
        preempt_next_tick: bool = False,
    ) -> "ConditionMonitor":
        """Create a monitor that fires when a state value crosses a threshold.

        Avoids the need for raw lambdas in common threshold-check patterns.

        Args:
            monitor_id: Unique identifier.
            agent_id: Agent to wake.
            key_path: List of keys to traverse in the state dict.
                Example: ``["agent_states", "bus_5", "features",
                "VoltageFeature", "vm_pu"]``
            threshold: Numeric threshold value.
            direction: ``"below"`` fires when value < threshold,
                ``"above"`` fires when value > threshold.
            cooldown: Minimum seconds between triggers.
            one_shot: Deregister after first trigger.
            preempt_next_tick: Cancel agent's next AGENT_TICK.

        Example::

            ConditionMonitor.threshold(
                monitor_id="undervoltage_bus_5",
                agent_id="field_agent_bus_5",
                key_path=["agent_states", "bus_5", "features",
                          "VoltageFeature", "vm_pu"],
                threshold=0.95,
                direction="below",
                cooldown=5.0,
            )
        """
        op = operator.lt if direction == "below" else operator.gt

        def _check(state: Dict[str, Any]) -> bool:
            value = _extract_nested(state, key_path)
            if value is None:
                return False
            return op(float(value), threshold)

        return cls(
            monitor_id=monitor_id,
            agent_id=agent_id,
            condition_fn=_check,
            cooldown=cooldown,
            one_shot=one_shot,
            preempt_next_tick=preempt_next_tick,
        )

    @classmethod
    def noop(cls, monitor_id: str = "noop", agent_id: str = "system_agent") -> "ConditionMonitor":
        """Create a no-op monitor that never fires. Useful as a placeholder."""
        return cls(
            monitor_id=monitor_id,
            agent_id=agent_id,
            condition_fn=lambda _: False,
        )

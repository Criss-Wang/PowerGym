"""Condition monitor for event-driven scheduling.

A ConditionMonitor watches a state predicate and fires a CONDITION_TRIGGER
event when the predicate becomes true. Evaluated by the scheduler after
SIMULATION and ENV_UPDATE events.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from heron.utils.typing import AgentID


@dataclass
class ConditionMonitor:
    """Monitors a state condition and fires CONDITION_TRIGGER when met.

    Evaluated after every SIMULATION and ENV_UPDATE event by the scheduler.

    Attributes:
        monitor_id: Unique identifier for this monitor.
        agent_id: Agent to wake when condition is met.
        condition_fn: Function (proxy_state_dict) -> bool.
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

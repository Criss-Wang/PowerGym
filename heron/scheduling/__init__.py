"""Event-driven scheduling for HERON.

This module provides discrete-event simulation capabilities with:
- Priority-queue based event scheduling
- Configurable latency modeling
- Heterogeneous agent tick rates
"""

from heron.scheduling.event import Event, EventType
from heron.scheduling.scheduler import EventScheduler

__all__ = ["Event", "EventType", "EventScheduler"]

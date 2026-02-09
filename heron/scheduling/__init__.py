"""Event-driven scheduling for HERON.

This module provides discrete-event simulation capabilities with:
- Priority-queue based event scheduling
- Configurable latency modeling
- Heterogeneous agent tick rates
- Tick configuration with optional jitter for testing
"""

from heron.scheduling.event import Event, EventType, EVENT_TYPE_FROM_STRING
from heron.scheduling.scheduler import DefaultScheduler, EventScheduler
from heron.scheduling.tick_config import JitterType, TickConfig

__all__ = ["DefaultScheduler", "Event", "EventType", "EventScheduler", "JitterType", "TickConfig", "EVENT_TYPE_FROM_STRING"]

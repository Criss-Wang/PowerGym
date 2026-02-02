"""Event-driven scheduling for HERON.

This module provides discrete-event simulation capabilities with:
- Priority-queue based event scheduling
- Configurable latency modeling
- Heterogeneous agent tick rates
- Tick configuration with optional jitter for testing
"""

from heron.scheduling.event import Event, EventType
from heron.scheduling.scheduler import EventScheduler
from heron.scheduling.tick_config import JitterType, TickConfig

__all__ = ["Event", "EventType", "EventScheduler", "JitterType", "TickConfig"]

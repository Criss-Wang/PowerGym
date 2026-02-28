"""Event-driven scheduling for HERON.

This module provides discrete-event simulation capabilities with:
- Priority-queue based event scheduling
- Configurable latency modeling
- Heterogeneous agent tick rates
- Tick configuration with optional jitter for testing
- Event analysis and episode result tracking
"""

from heron.scheduling.event import Event, EventType, EVENT_TYPE_FROM_STRING
from heron.scheduling.scheduler import DefaultScheduler, EventScheduler
from heron.scheduling.tick_config import (
    DEFAULT_COORDINATOR_AGENT_TICK_CONFIG,
    DEFAULT_FIELD_AGENT_TICK_CONFIG,
    DEFAULT_SYSTEM_AGENT_TICK_CONFIG,
    JitterType,
    TickConfig,
)
from heron.scheduling.analysis import EventAnalyzer, EpisodeResult, EventAnalysis

__all__ = [
    "DEFAULT_COORDINATOR_AGENT_TICK_CONFIG",
    "DEFAULT_FIELD_AGENT_TICK_CONFIG",
    "DefaultScheduler",
    "DEFAULT_SYSTEM_AGENT_TICK_CONFIG",
    "Event",
    "EventType",
    "EventScheduler",
    "JitterType",
    "TickConfig",
    "EVENT_TYPE_FROM_STRING",
    "EventAnalyzer",
    "EpisodeResult",
    "EventAnalysis",
]

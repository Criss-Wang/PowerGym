"""Event scheduler with priority queue for discrete-event simulation.

The EventScheduler is the core of HERON's event-driven execution model.
It processes events in timestamp order, supporting:
- Heterogeneous agent tick rates
- Configurable observation/action/message latency
- Deterministic replay for reproducibility
"""

import heapq
from typing import Any, Callable, Dict, List, Optional, Set

from heron.scheduling.event import Event, EventType
from heron.utils.typing import AgentID


class EventScheduler:
    """Priority-queue based event scheduler.

    The scheduler maintains a min-heap of events ordered by timestamp.
    Events are processed in order, and handlers can schedule new events.

    Example usage:
        scheduler = EventScheduler()

        # Register agents with tick intervals
        scheduler.register_agent("battery_1", tick_interval=1.0)
        scheduler.register_agent("solar_1", tick_interval=5.0)

        # Register handlers
        scheduler.set_handler(EventType.AGENT_TICK, agent_tick_handler)

        # Run simulation
        scheduler.run_until(t_end=100.0)

    Attributes:
        current_time: Current simulation time
        event_queue: Min-heap of pending events
        agent_intervals: Dict mapping agent IDs to their tick intervals
        handlers: Dict mapping event types to handler functions
    """

    def __init__(self, start_time: float = 0.0):
        """Initialize scheduler.

        Args:
            start_time: Initial simulation time
        """
        self.current_time: float = start_time
        self.event_queue: List[Event] = []
        self._sequence_counter: int = 0

        # Agent configuration
        self.agent_intervals: Dict[AgentID, float] = {}
        self.agent_obs_delays: Dict[AgentID, float] = {}
        self.agent_act_delays: Dict[AgentID, float] = {}

        # Event handlers: EventType -> Callable[[Event, EventScheduler], None]
        self.handlers: Dict[EventType, Callable[[Event, "EventScheduler"], None]] = {}

        # Tracking
        self._processed_count: int = 0
        self._active_agents: Set[AgentID] = set()

    def register_agent(
        self,
        agent_id: AgentID,
        tick_interval: float = 1.0,
        obs_delay: float = 0.0,
        act_delay: float = 0.0,
        first_tick: Optional[float] = None,
    ) -> None:
        """Register an agent with the scheduler.

        Args:
            agent_id: Unique agent identifier
            tick_interval: Time between agent ticks (steps)
            obs_delay: Observation delay (agent sees state from t - obs_delay)
            act_delay: Action delay (action takes effect at t + act_delay)
            first_tick: Time of first tick (defaults to current_time)
        """
        self.agent_intervals[agent_id] = tick_interval
        self.agent_obs_delays[agent_id] = obs_delay
        self.agent_act_delays[agent_id] = act_delay
        self._active_agents.add(agent_id)

        # Schedule first tick
        first_time = first_tick if first_tick is not None else self.current_time
        self.schedule(Event(
            timestamp=first_time,
            event_type=EventType.AGENT_TICK,
            agent_id=agent_id,
            payload={}
        ))

    def unregister_agent(self, agent_id: AgentID) -> None:
        """Remove an agent from the scheduler.

        Args:
            agent_id: Agent to remove
        """
        self._active_agents.discard(agent_id)
        self.agent_intervals.pop(agent_id, None)
        self.agent_obs_delays.pop(agent_id, None)
        self.agent_act_delays.pop(agent_id, None)

    def schedule(self, event: Event) -> None:
        """Schedule an event for future processing.

        Args:
            event: Event to schedule
        """
        # Assign sequence number for deterministic ordering
        event.sequence = self._sequence_counter
        self._sequence_counter += 1
        heapq.heappush(self.event_queue, event)

    def schedule_agent_tick(
        self,
        agent_id: AgentID,
        timestamp: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Schedule an agent tick event.

        Args:
            agent_id: Agent to tick
            timestamp: When to tick (defaults to current_time + interval)
            payload: Optional event payload
        """
        if timestamp is None:
            interval = self.agent_intervals.get(agent_id, 1.0)
            timestamp = self.current_time + interval

        self.schedule(Event(
            timestamp=timestamp,
            event_type=EventType.AGENT_TICK,
            agent_id=agent_id,
            payload=payload or {}
        ))

    def schedule_action_effect(
        self,
        agent_id: AgentID,
        action: Any,
        delay: Optional[float] = None,
    ) -> None:
        """Schedule a delayed action effect.

        Args:
            agent_id: Agent whose action this is
            action: The action to apply
            delay: Delay before action takes effect (defaults to agent's act_delay)
        """
        if delay is None:
            delay = self.agent_act_delays.get(agent_id, 0.0)

        self.schedule(Event(
            timestamp=self.current_time + delay,
            event_type=EventType.ACTION_EFFECT,
            agent_id=agent_id,
            payload={"action": action}
        ))

    def schedule_message_delivery(
        self,
        sender_id: AgentID,
        recipient_id: AgentID,
        message: Any,
        delay: float = 0.0,
    ) -> None:
        """Schedule a delayed message delivery.

        Args:
            sender_id: Sending agent
            recipient_id: Receiving agent
            message: Message content
            delay: Communication delay
        """
        self.schedule(Event(
            timestamp=self.current_time + delay,
            event_type=EventType.MESSAGE_DELIVERY,
            agent_id=recipient_id,
            priority=1,  # Lower priority than ticks
            payload={"sender": sender_id, "message": message}
        ))

    def set_handler(
        self,
        event_type: EventType,
        handler: Callable[[Event, "EventScheduler"], None]
    ) -> None:
        """Set handler for an event type.

        Args:
            event_type: Type of event to handle
            handler: Function called with (event, scheduler) when event is processed
        """
        self.handlers[event_type] = handler

    def peek(self) -> Optional[Event]:
        """Peek at the next event without removing it.

        Returns:
            Next event, or None if queue is empty
        """
        return self.event_queue[0] if self.event_queue else None

    def pop(self) -> Optional[Event]:
        """Pop and return the next event.

        Returns:
            Next event, or None if queue is empty
        """
        if not self.event_queue:
            return None
        return heapq.heappop(self.event_queue)

    def process_next(self) -> bool:
        """Process the next event in the queue.

        Returns:
            True if an event was processed, False if queue is empty
        """
        event = self.pop()
        if event is None:
            return False

        # Advance simulation time
        self.current_time = event.timestamp
        self._processed_count += 1

        # Skip events for unregistered agents
        if event.agent_id and event.agent_id not in self._active_agents:
            return True

        # Dispatch to handler
        handler = self.handlers.get(event.event_type)
        if handler:
            handler(event, self)

        # Auto-schedule next tick for AGENT_TICK events
        if (
            event.event_type == EventType.AGENT_TICK
            and event.agent_id
            and event.agent_id in self._active_agents
        ):
            self.schedule_agent_tick(event.agent_id)

        return True

    def run_until(
        self,
        t_end: float,
        max_events: Optional[int] = None,
    ) -> int:
        """Run simulation until time limit or event limit.

        Args:
            t_end: Stop when current_time exceeds this
            max_events: Optional maximum number of events to process

        Returns:
            Number of events processed
        """
        count = 0
        while self.event_queue:
            # Check time limit
            if self.peek().timestamp > t_end:
                break

            # Check event limit
            if max_events is not None and count >= max_events:
                break

            if self.process_next():
                count += 1

        return count

    def run_steps(self, n_steps: int) -> int:
        """Run exactly n_steps events.

        Args:
            n_steps: Number of events to process

        Returns:
            Number of events actually processed (may be less if queue empties)
        """
        count = 0
        for _ in range(n_steps):
            if not self.process_next():
                break
            count += 1
        return count

    def clear(self) -> None:
        """Clear all pending events."""
        self.event_queue.clear()
        self._sequence_counter = 0

    def reset(self, start_time: float = 0.0) -> None:
        """Reset scheduler to initial state.

        Args:
            start_time: New simulation start time
        """
        self.current_time = start_time
        self.clear()
        self._processed_count = 0

        # Re-schedule first ticks for all agents
        for agent_id in self._active_agents:
            self.schedule(Event(
                timestamp=start_time,
                event_type=EventType.AGENT_TICK,
                agent_id=agent_id,
                payload={}
            ))

    @property
    def pending_count(self) -> int:
        """Number of pending events in queue."""
        return len(self.event_queue)

    @property
    def processed_count(self) -> int:
        """Total number of events processed."""
        return self._processed_count

    def __repr__(self) -> str:
        return (
            f"EventScheduler(t={self.current_time:.3f}, "
            f"pending={self.pending_count}, agents={len(self._active_agents)})"
        )

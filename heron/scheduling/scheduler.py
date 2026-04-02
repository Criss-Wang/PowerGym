import heapq
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Iterable

if TYPE_CHECKING:
    from heron.agents import Agent

from heron.scheduling.condition_monitor import ConditionMonitor
from heron.scheduling.event import Event, EventType
from heron.utils.typing import AgentID
from heron.scheduling.schedule_config import ScheduleConfig
from heron.agents.constants import SYSTEM_AGENT_ID

_DEFAULT_SIMULATION_DELAY = 0.0

class EventScheduler:
    def __init__(self, start_time: float = 0.0):
        self.current_time: float = start_time
        self.event_queue: List[Event] = []
        self._sequence_counter: int = 0

        # Agent configuration - ScheduleConfig storage
        self._agent_schedule_configs: Dict[AgentID, "ScheduleConfig"] = {}

        self.handlers: Dict[
            AgentID, Dict[EventType, Callable[[Event, "EventScheduler"], None]]
        ] = {}

        # Condition monitors (evaluated after SIMULATION and ENV_UPDATE events)
        self._condition_monitors: List[ConditionMonitor] = []

        # Tracking
        self._processed_count: int = 0
        self._active_agents: Set[AgentID] = set()

    # ===============================
    # Scheduler Initialization and Configuration
    # ===============================
    @staticmethod
    def init(config: Optional[Dict[str, Any]] = None) -> "EventScheduler":
        if not config:
            return EventScheduler(start_time=0.0)

        start_time = config.get("start_time", 0.0)
        return EventScheduler(start_time)

    @property
    def agent_schedule_configs(self) -> Dict[AgentID, "ScheduleConfig"]:
        """Get the tick configs for all agents."""
        return self._agent_schedule_configs

    def attach(self, agents: Dict[AgentID, "Agent"]) -> None:
        """Register agents with the scheduler and schedule the initial system tick.

        Stores each agent's ScheduleConfig, marks them as active, registers their
        event handlers, and schedules the first AGENT_TICK for the system agent.

        Args:
            agents: Dict mapping agent IDs to Agent instances
        """
        for agent_id, agent in agents.items():
            self._agent_schedule_configs[agent_id] = agent.schedule_config
            self._active_agents.add(agent_id)
            self.set_handlers_for_agent(agent_id, agent.get_handlers())
            if agent_id == SYSTEM_AGENT_ID:
                # Only schedule first tick for system agent
                self.schedule(
                    Event(
                        timestamp=self.current_time,
                        event_type=EventType.AGENT_TICK,
                        agent_id=agent_id,
                        payload={},
                    )
                )

    # ===============================
    # Event Scheduling Methods
    # ===============================
    def schedule(self, event: Event) -> str:
        """Schedule an event for future processing.

        Args:
            event: Event to schedule

        Returns:
            The event_id assigned to this event.
        """
        # Assign sequence number for deterministic ordering
        event.sequence = self._sequence_counter
        self._sequence_counter += 1
        heapq.heappush(self.event_queue, event)
        return event.event_id

    def cancel_event(self, event_id: str) -> bool:
        """Mark a specific event as cancelled. Returns True if found."""
        for event in self.event_queue:
            if event.event_id == event_id and not event.cancelled:
                event.cancelled = True
                return True
        return False

    def cancel_events(self, agent_id: AgentID, event_type: EventType) -> int:
        """Cancel all pending events matching agent_id and event_type.

        Returns count of cancelled events.
        """
        count = 0
        for event in self.event_queue:
            if (
                event.agent_id == agent_id
                and event.event_type == event_type
                and not event.cancelled
            ):
                event.cancelled = True
                count += 1
        return count

    def schedule_agent_tick(
        self,
        agent_id: AgentID,
        timestamp: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Schedule an agent tick event.

        Uses jittered interval from ScheduleConfig if available.

        Args:
            agent_id: Agent to tick
            timestamp: When to tick (defaults to current_time + agent's jittered interval)
            payload: Optional event payload
        """
        if timestamp is None:
            timestamp = self.current_time + self.get_tick_interval(agent_id)

        self.schedule(
            Event(
                timestamp=timestamp,
                event_type=EventType.AGENT_TICK,
                agent_id=agent_id,
                payload=payload or {},
            )
        )

    def schedule_action_effect(
        self,
        agent_id: AgentID,
        delay: Optional[float] = None,
    ) -> None:
        """Schedule a delayed action effect.

        Uses jittered delay from ScheduleConfig if available.

        Args:
            agent_id: Agent whose action this is
            delay: Delay before action takes effect (defaults to jittered act_delay)
        """
        if delay is None:
            delay = self.get_act_delay(agent_id)

        self.schedule(
            Event(
                timestamp=self.current_time + delay,
                event_type=EventType.ACTION_EFFECT,
                agent_id=agent_id,
                priority=0,  # Agent-level events (applies action effects on agent state)
            )
        )

    def schedule_message_delivery(
        self,
        sender_id: AgentID,
        recipient_id: AgentID,
        message: Any,
        delay: Optional[float] = None,
    ) -> None:
        """Schedule a delayed message delivery.

        Uses jittered delay from sender's ScheduleConfig if delay not provided.

        Args:
            sender_id: Sending agent
            recipient_id: Receiving agent
            message: Message content
            delay: Communication delay (defaults to sender's jittered msg_delay)
        """
        if delay is None:
            delay = self.get_msg_delay(sender_id)

        self.schedule(
            Event(
                timestamp=self.current_time + delay,
                event_type=EventType.MESSAGE_DELIVERY,
                agent_id=recipient_id,
                priority=2,  # Communication-level events (deliver after state changes)
                payload={"sender": sender_id, "message": message},
            )
        )

    def schedule_simulation(
        self,
        agent_id: AgentID,
        delay: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        """Schedule a simulation (physics) event.

        Used both for periodic physics (called by SystemAgent each tick) and
        for reactive/exogenous physics triggers (called by any agent needing
        an out-of-band physics step).

        Args:
            agent_id: Agent that handles the simulation event (typically SystemAgent)
            delay: Delay before simulation fires (default: 0.0)
            payload: Optional metadata (e.g. requesting agent for reactive triggers)
        """
        if delay is None:
            delay = _DEFAULT_SIMULATION_DELAY

        self.schedule(
            Event(
                timestamp=self.current_time + delay,
                event_type=EventType.SIMULATION,
                agent_id=agent_id,
                priority=1,  # Environment-level events (runs physics after actions)
                payload=payload or {},
            )
        )

    # ===============================
    # Condition Monitor Management
    # ===============================
    def register_condition(self, monitor: ConditionMonitor) -> None:
        """Register a condition monitor. Conditions are evaluated after
        SIMULATION and ENV_UPDATE events."""
        self._condition_monitors.append(monitor)

    def deregister_condition(self, monitor_id: str) -> bool:
        """Remove a condition monitor by ID. Returns True if found."""
        for i, m in enumerate(self._condition_monitors):
            if m.monitor_id == monitor_id:
                self._condition_monitors.pop(i)
                return True
        return False

    def evaluate_conditions(self, proxy_state: Dict[str, Any]) -> List[Event]:
        """Evaluate all registered conditions against current state.

        Schedules CONDITION_TRIGGER events for any that fire.
        Returns list of triggered events.
        """
        triggered: List[Event] = []
        to_remove: List[str] = []

        for monitor in self._condition_monitors:
            if not monitor.condition_fn(proxy_state):
                continue
            if self.current_time - monitor._last_triggered < monitor.cooldown:
                continue

            # Condition fired — schedule CONDITION_TRIGGER
            event = Event(
                timestamp=self.current_time,
                event_type=EventType.CUSTOM,  # Renamed to CONDITION_TRIGGER in Phase 2
                agent_id=monitor.agent_id,
                priority=2,
                payload={
                    "monitor_id": monitor.monitor_id,
                    "condition": monitor.monitor_id,
                },
            )
            self.schedule(event)
            triggered.append(event)

            monitor._last_triggered = self.current_time

            if monitor.preempt_next_tick:
                self.cancel_events(monitor.agent_id, EventType.AGENT_TICK)

            if monitor.one_shot:
                to_remove.append(monitor.monitor_id)

        for mid in to_remove:
            self.deregister_condition(mid)

        return triggered

    # ===============================
    # Delay and Interval Accessors with Jitter Support
    # ===============================
    def get_obs_delay(self, agent_id: AgentID) -> float:
        """Get (possibly jittered) observation delay for agent.

        Args:
            agent_id: Agent ID

        Returns:
            Observation delay (jittered if config has jitter enabled)
        """
        config = self._agent_schedule_configs.get(agent_id)
        if config:
            return config.get_obs_delay()
        return 0.0

    def get_act_delay(self, agent_id: AgentID) -> float:
        """Get (possibly jittered) action delay for agent.

        Args:
            agent_id: Agent ID

        Returns:
            Action delay (jittered if config has jitter enabled)
        """
        config = self._agent_schedule_configs.get(agent_id)
        if config:
            return config.get_act_delay()
        return 0.0

    def get_msg_delay(self, agent_id: AgentID) -> float:
        """Get (possibly jittered) message delay for agent.

        Args:
            agent_id: Agent ID

        Returns:
            Message delay (jittered if config has jitter enabled)
        """
        config = self._agent_schedule_configs.get(agent_id)
        if config:
            return config.get_msg_delay()
        return 0.0

    def get_tick_interval(self, agent_id: AgentID) -> float:
        """Get (possibly jittered) tick interval for agent.

        Args:
            agent_id: Agent ID

        Returns:
            Tick interval (jittered if config has jitter enabled)
        """
        config = self._agent_schedule_configs.get(agent_id)
        if config:
            return config.get_tick_interval()
        return 1.0

    # ===============================
    # Handler Management
    # ===============================
    def set_handler(
        self,
        event_type: EventType,
        handler: Callable[[Event, "EventScheduler"], None],
        agent_id: AgentID,
    ) -> None:
        """Set handler for an event type for a specific agent.

        Args:
            event_type: Type of event to handle
            handler: Function called with (event, scheduler) when event is processed
            agent_id: Agent ID - handler only applies to this agent's events
        """
        if agent_id not in self.handlers:
            self.handlers[agent_id] = {}
        self.handlers[agent_id][event_type] = handler

    def set_handlers_for_agent(
        self,
        agent_id: AgentID,
        handlers: Dict[EventType, Callable[[Event, "EventScheduler"], None]],
    ) -> None:
        """Set multiple handlers for a specific agent.

        Args:
            agent_id: Agent to set handlers for
            handlers: Dict mapping event types to handler functions
        """
        if agent_id not in self.handlers:
            self.handlers[agent_id] = {}
        self.handlers[agent_id].update(handlers)

    def get_handler(
        self,
        event_type: EventType,
        agent_id: AgentID,
    ) -> Optional[Callable[[Event, "EventScheduler"], None]]:
        """Get handler for an event type for a specific agent.

        Args:
            event_type: Type of event
            agent_id: Agent ID for handler lookup

        Returns:
            Handler function or None if no handler registered
        """
        if agent_id in self.handlers:
            return self.handlers[agent_id].get(event_type)
        return None

    # ===============================
    # Core Event Loop Methods
    # - peek: Look at next event without removing
    # - pop: Remove and return next event
    # - process_next: Process next event (advance time, call handler)
    # - run_until: Run until time limit or event limit
    #
    # To be implemented (upon roadmap):
    # - run: Run indefinitely until no events left (with optional max_events limit)
    # - run_steps: Run a fixed number of events (Not used yet, but useful for step-based execution mode)
    # ===============================
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

    def process_next(self) -> Optional[Event]:
        """Process the next event in the queue.

        Cancelled events are silently discarded.

        Returns:
            The processed Event, or None if queue is empty
        """
        # Skip cancelled events
        while self.event_queue:
            event = self.pop()
            if event is None:
                return None
            if not event.cancelled:
                break
        else:
            return None

        # Advance simulation time
        self.current_time = event.timestamp
        self._processed_count += 1

        # check for valid agent id
        if event.agent_id and event.agent_id not in self._active_agents:
            raise ValueError(
                f"{event.agent_id} is not registered, check your environment setup again"
            )

        # Dispatch to handler (agent-specific first, then global fallback)
        handler = self.get_handler(event.event_type, event.agent_id)
        if handler is None:
            raise ValueError(
                f"No handler registered for event_type={event.event_type}, agent_id={event.agent_id}"
            )
        handler(event, self)

        return event

    def run_until(
        self,
        t_end: float,
        max_events: Optional[int] = None,
    ) -> Iterable[Event]:
        """Run simulation until time limit or event limit.

        Args:
            t_end: Stop when current_time exceeds this
            max_events: Optional maximum number of events to process

        Yields:
            Event objects as they are processed
        """
        count = 0
        while self.event_queue:
            # Check time limit
            if self.peek().timestamp > t_end:
                break

            # Check event limit
            if max_events is not None and count >= max_events:
                break

            if event := self.process_next():
                count += 1
                yield event

    def clear(self) -> None:
        """Clear all pending events."""
        self.event_queue.clear()
        self._sequence_counter = 0

    def sync_schedule_configs(self, agents: Dict[AgentID, "Agent"]) -> None:
        """Re-sync cached tick configs from agents.

        Call this after modifying agents' tick configs (e.g. changing
        tick_interval or enabling jitter) so the scheduler uses the
        updated values for delay/interval calculations.

        Args:
            agents: Dict mapping agent IDs to Agent instances
        """
        for agent_id, agent in agents.items():
            if agent_id in self._active_agents:
                self._agent_schedule_configs[agent_id] = agent.schedule_config

    def reset(self, start_time: float = 0.0) -> None:
        """Reset scheduler to initial state.

        Args:
            start_time: New simulation start time
        """
        self.current_time = start_time
        self.clear()
        self._processed_count = 0
        # Reset condition monitor cooldowns (keep registrations)
        for monitor in self._condition_monitors:
            monitor._last_triggered = -float("inf")

        # Re-schedule first tick only for system agent (matches attach() behavior)
        # System agent will cascade ticks to subordinates
        if SYSTEM_AGENT_ID not in self._active_agents:
            raise ValueError(
                f"System agent (ID={SYSTEM_AGENT_ID}) not registered, check your environment setup again"
            )
        self.schedule(
            Event(
                timestamp=start_time,
                event_type=EventType.AGENT_TICK,
                agent_id=SYSTEM_AGENT_ID,
                payload={},
            )
        )

    # ===============================
    # Properties and Utility Methods
    # ===============================
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



from collections import deque
from typing import Any, Dict, List, Optional, Set

from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.agents.proxy_agent import Proxy
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.observation import Observation
from heron.core.state import CoordinatorAgentState, State
from heron.core.policies import Policy
from heron.utils.typing import AgentID
from heron.protocols.base import Protocol
from heron.scheduling.schedule_config import DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG, ScheduleConfig
from heron.scheduling.scheduler import EventScheduler, Event
from heron.agents.constants import (
    COORDINATOR_LEVEL,
    PROXY_AGENT_ID,
    MSG_GET_INFO,
    MSG_SET_TICK_RESULT,
    MSG_PHYSICS_COMPLETED,
    INFO_TYPE_OBS,
    INFO_TYPE_LOCAL_STATE,
    MSG_KEY_BODY,
    MSG_KEY_PROTOCOL,
    MSG_GET_OBS_RESPONSE,
    MSG_GET_LOCAL_STATE_RESPONSE,
    MSG_KEY_ENV_CONTEXT,
)


class CoordinatorAgent(Agent):
    """Level-2 agent that coordinates subordinate field agents.

    **Reward timing contract (event-driven mode)**:

    1. Physics completes → coordinator forwards ``MSG_PHYSICS_COMPLETED``
       to reactive subordinates.
    2. Each subordinate computes its reward and sends ``sub_reward_complete``.
    3. When **all** subordinates finish, coordinator requests its own local
       state and calls ``compute_local_reward()``.
    4. Multiple physics cycles can overlap — each is tracked independently
       via a FIFO queue (``_pending_reward_cycles``).

    If the system tick interval is shorter than the estimated reward cascade
    time (~6 × max msg_delay), cascades may overlap and coordinator rewards
    will be computed against the latest (not cycle-specific) post-physics
    state.  A warning is emitted at env construction in this case.
    """

    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        features: Optional[List[Feature]] = None,
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        subordinates: Optional[Dict[AgentID, "Agent"]] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        schedule_config: Optional[ScheduleConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None
    ):

        super().__init__(
            agent_id=agent_id,
            level=COORDINATOR_LEVEL,
            features=features,
            upstream_id=upstream_id,
            subordinates=subordinates,
            env_id=env_id,
            schedule_config=schedule_config or DEFAULT_COORDINATOR_AGENT_SCHEDULE_CONFIG,
            policy=policy,
            protocol=protocol,
        )
        self._pending_reward_cycles: deque = deque()  # deque[Set[AgentID]]

    def init_state(self, features: List[Feature] = []) -> State:
        """Initialize a CoordinatorAgentState from the provided features."""
        return CoordinatorAgentState(
            owner_id=self.agent_id,
            owner_level=COORDINATOR_LEVEL,
            features={f.feature_name: f for f in features}
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        """Initialize an empty Action (coordinators delegate actions to subordinates)."""
        return Action()

    def set_state(self, *args, **kwargs) -> None:
        pass

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Store action for protocol-based distribution to subordinates."""
        if isinstance(action, Action):
            self.action = action
        elif self.action.is_valid():
            self.action.set_values(action)

    # ============================================
    # Core Lifecycle Methods Overrides (see heron/agents/base.py for more details)
    # ============================================
    # execute() inherited from base class - uses default implementation

    def reset(self, *, seed=None, proxy=None, **kwargs):
        super().reset(seed=seed, proxy=proxy, **kwargs)
        self._pending_reward_cycles = deque()

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
        reschedule: bool = True,
    ) -> None:
        """Action phase: observe → decide → coordinate subordinates. [Event-Driven Mode]

        Periodic coordinators self-reschedule here (R5). Reactive subordinate
        ticks are scheduled after action coordination (in obs response handler).

        Args:
            scheduler: Event scheduler.
            current_time: Current simulation time.
            reschedule: If False, skip self-reschedule. Used by
                ``condition_trigger_handler`` so reactive wakeups don't
                create duplicate periodic cycles.
        """
        super().tick(scheduler, current_time)  # Update internal timestep and check for upstream actions

        # Schedule subordinate ticks -> initiate action process
        for subordinate_id in self.subordinates:
            scheduler.schedule_agent_tick(subordinate_id)

        # Always request obs from proxy first for state sync.
        # Upstream action (if any) will be applied after sync in get_obs_response handler.
        # Uses obs_delay (not msg_delay) to model sensor/telemetry latency.
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={MSG_GET_INFO: INFO_TYPE_OBS, MSG_KEY_PROTOCOL: self.protocol},
            delay=scheduler.get_obs_delay(self.agent_id),
        )

        # R5: periodic agents self-reschedule immediately in tick()
        if reschedule:
            self._self_reschedule(scheduler)

    # ============================================
    # Custom Handlers for Event-Driven Execution
    # ============================================
    @Agent.handler("agent_tick")
    def agent_tick_handler(self, event: Event, scheduler: EventScheduler) -> None:
        self.tick(scheduler, event.timestamp)

    @Agent.handler("condition_trigger")
    def condition_trigger_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Handle a condition-triggered wakeup (e.g., voltage alarm).

        Runs the same observe→decide→coordinate cycle as a regular tick,
        but does NOT self-reschedule. Condition-triggered wakeups are
        reactive one-offs — the coordinator's periodic schedule is unaffected.

        Override for custom reactive logic.
        Payload contains monitor_id identifying which condition fired.
        """
        self.tick(scheduler, event.timestamp, reschedule=False)

    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Coordinator actions don't update local state (they coordinate subordinates).

        No-op handler to handle action_effect events scheduled by compute_action.
        """
        pass

    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Handle messages for action coordination and reward phases.

        Four cases:
        a. MSG_GET_OBS_RESPONSE → sync state, compute action, cache (obs, action),
           schedule reactive subordinate ticks
        b. MSG_PHYSICS_COMPLETED → forward to reactive subordinates, wait for their
           rewards before computing own
        c. sub_reward_complete → track reactive subordinate completion, trigger own
           reward when all done
        d. MSG_GET_LOCAL_STATE_RESPONSE → compute reward at physics boundary (R7)
        """
        recipient_id = event.agent_id
        assert recipient_id == self.agent_id
        message_content = event.payload.get("message", {})

        if MSG_GET_OBS_RESPONSE in message_content:
            assert isinstance(message_content, dict)
            response_data = message_content[MSG_GET_OBS_RESPONSE]
            body = response_data[MSG_KEY_BODY]

            obs_dict = body["obs"]
            local_state = body["local_state"]
            obs = Observation.from_dict(obs_dict)

            self.sync_state_from_observed(local_state)
            self.compute_action(obs, scheduler)

            # R7: cache (obs, action) for deferred reward at physics boundary.
            # Safe to cache here (unlike FieldAgent) because the coordinator's
            # action effect is coordination — already dispatched synchronously
            # in compute_action() above.
            self._cache_obs_action(obs, self.action)

            # R3: schedule reactive subordinate ticks after action coordination
            if self._should_send_subordinate_actions():
                for sub_id, sub in self.subordinates.items():
                    if not sub.is_periodic:
                        scheduler.schedule_agent_tick(sub_id)

        elif MSG_PHYSICS_COMPLETED in message_content:
            # Forward to reactive subordinates first (bottom-up reward cascade).
            reactive_subs = {
                sub_id for sub_id, sub in self.subordinates.items()
                if not sub.is_periodic
            }
            for sub_id in reactive_subs:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=sub_id,
                    message={MSG_PHYSICS_COMPLETED: "success"},
                )
            # Queue a per-cycle pending set so overlapping physics cycles
            # cascade independently (no overwrite, no reward loss).
            if not reactive_subs:
                # No reactive subordinates — compute own reward immediately
                self._request_post_physics_state(scheduler)
            else:
                self._pending_reward_cycles.append(set(reactive_subs))

        elif "sub_reward_complete" in message_content:
            sub_id = message_content["sub_reward_complete"]
            # FIFO match: find the oldest cycle that still expects this sub
            for cycle in self._pending_reward_cycles:
                if sub_id in cycle:
                    cycle.discard(sub_id)
                    if not cycle:
                        # This cycle's cascade is complete — pop and trigger reward
                        self._pending_reward_cycles.popleft()
                        self._request_post_physics_state(scheduler)
                    break

        elif MSG_GET_LOCAL_STATE_RESPONSE in message_content:
            # R7: compute reward at physics boundary
            response_data = message_content[MSG_GET_LOCAL_STATE_RESPONSE]
            local_state = response_data[MSG_KEY_BODY]
            env_context = response_data.get(MSG_KEY_ENV_CONTEXT)

            self.sync_state_from_observed(local_state)

            reward = self.compute_local_reward(local_state, self._prev_post_physics_state)
            obs_action_pairs = self._flush_obs_action_cache()

            tick_result = {
                "reward": reward,
                "terminated": self.is_terminated(local_state, env_context),
                "truncated": self.is_truncated(local_state, env_context),
                "info": self.get_local_info(local_state),
                "obs_action_pairs": obs_action_pairs,
            }
            self._prev_post_physics_state = local_state

            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_SET_TICK_RESULT: INFO_TYPE_LOCAL_STATE, MSG_KEY_BODY: tick_result},
            )

            # Reactive coordinators: notify upstream that reward is done
            if not self.is_periodic and self.upstream_id:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=self.upstream_id,
                    message={"sub_reward_complete": self.agent_id},
                )

        else:
            raise NotImplementedError(f"CoordinatorAgent received unknown message: {list(message_content.keys())}")

    def compute_local_reward(
        self,
        local_state: dict,
        prev_post_physics_state: Optional[dict] = None,
    ) -> float:
        """Compute coordinator reward at a physics boundary.

        Called only after **all** reactive subordinate rewards have
        completed (bottom-up cascade).  In event-driven mode,
        ``local_state`` reflects the post-physics global state, so
        team-level metrics (total throughput, grid stability) are
        already available via visibility-scoped features.

        Override to implement domain-specific coordinator reward::

            def compute_local_reward(self, local_state, prev=None):
                throughput = local_state.get("TeamFeature", [0.0])[0]
                return throughput

        The default returns 0.0 (coordinator produces no reward).
        """
        from heron.agents.constants import EMPTY_REWARD
        return EMPTY_REWARD

    def _request_post_physics_state(self, scheduler: EventScheduler) -> None:
        """Request local state from proxy for physics-boundary reward computation."""
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: self.protocol},
        )
    

    # ============================================
    # Convenience Property
    # ============================================
    @property
    def field_agents(self) -> Dict[AgentID, FieldAgent]:
        """Alias for subordinates - more descriptive for CoordinatorAgent context."""
        return self.subordinates

    def __repr__(self) -> str:
        num_fields = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"CoordinatorAgent(id={self.agent_id}, field_agents={num_fields}, protocol={protocol_name})"

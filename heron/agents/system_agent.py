import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

from heron.agents.base import Agent
from heron.agents.proxy_agent import Proxy
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.observation import Observation
from heron.core.state import State, SystemAgentState
from heron.core.env_context import compute_all_done
from heron.utils.typing import AgentID, MultiAgentDict
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.event import EventType
from heron.scheduling.scheduler import Event, EventScheduler
from heron.scheduling.schedule_config import DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG, ScheduleConfig
from gymnasium.spaces import Box, Space
from heron.agents.constants import (
    SYSTEM_LEVEL,
    SYSTEM_AGENT_ID,
    PROXY_AGENT_ID,
    MSG_GET_INFO,
    MSG_SET_STATE,
    MSG_SET_STATE_COMPLETION,
    MSG_PHYSICS_COMPLETED,
    INFO_TYPE_GLOBAL_STATE,
    STATE_TYPE_GLOBAL,
    MSG_KEY_BODY,
    MSG_KEY_PROTOCOL,
    MSG_GET_GLOBAL_STATE_RESPONSE,
)

logger = logging.getLogger(__name__)


class SystemAgent(Agent):
    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        features: Optional[List[Feature]] = None,
        # hierarchy params
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
            agent_id=agent_id or SYSTEM_AGENT_ID,
            level=SYSTEM_LEVEL,
            features=features,
            subordinates=subordinates,
            upstream_id=None,  # System agent has no upstream
            env_id=env_id,
            schedule_config=schedule_config or DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG,
            policy=policy,
            protocol=protocol
        )
        self._simulation_configured: bool = False
        self._subordinates_kicked_off: bool = False
        self._last_post_physics_state: Optional[Dict] = None
        self._apply_disturbance_func: Optional[Callable] = None
        # FIFO queue tracking whether each in-flight simulation was
        # disturbance-triggered. Pushed in simulation_handler, popped
        # in message_delivery_handler at MSG_SET_STATE_COMPLETION.
        from collections import deque
        self._sim_origin_queue: deque = deque()

        # Resolve which agents are periodic vs reactive (R2-R4)
        self._periodic_agents: List[AgentID] = []
        self.refresh_periodic_agents()

    def refresh_periodic_agents(self) -> None:
        """Re-resolve periodic children and cache the result."""
        self._periodic_agents = self.resolve_periodic_children()

    def init_state(self, features: List[Feature] = []) -> State:
        """Initialize a SystemAgentState from the provided features."""
        return SystemAgentState(
            owner_id=self.agent_id,
            owner_level=SYSTEM_LEVEL,
            features={f.feature_name: f for f in features}
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        """Initialize an empty Action (system agent manages simulation, not direct actions)."""
        return Action()

    
    def set_state(self, *args, **kwargs) -> None:
        pass

    def set_action(self, action: Any, *args, **kwargs) -> None:
        pass

    # ============================================
    # Core Lifecycle Methods Overrides (see heron/agents/base.py for more details)
    # ============================================
    def reset(self, *, seed: Optional[int] = None, proxy: Optional[Proxy] = None, **kwargs) -> Any:
        """Reset system agent and all subordinates. [Both Modes]

        Returns vectorized observations (``np.ndarray``) for subordinate
        agents only (R1: system agent does not observe).

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed, proxy=proxy, **kwargs)
        self._subordinates_kicked_off = False

        # R1: collect obs from subordinates only, not self
        obs: Dict[AgentID, Any] = {}
        for subordinate in self.subordinates.values():
            obs.update(subordinate.observe(proxy=proxy))
        obs_vectorized = {
            aid: o.vector() if isinstance(o, Observation) else o
            for aid, o in obs.items()
        }
        return obs_vectorized, {}
    
    def execute(
        self,
        actions: Dict[AgentID, Any],
        proxy: Optional[Proxy] = None,
        env_context: "Optional[EnvContext]" = None,
    ) -> None:
        """Execute actions with hierarchical coordination and simulation. [Training Mode]

        R1: SystemAgent is a pure orchestrator — it does not observe, act, or
        compute reward for itself.  All returned dicts contain only subordinate
        agent entries.

        Args:
            actions: Per-agent actions dict.
            proxy: Proxy for state management.
            env_context: Environment context for termination decisions.
        """
        if not proxy:
            raise ValueError("We still require a valid proxy agent so far")
        if not self._simulation_configured:
            raise RuntimeError("Simulation not configured. Call set_simulation() before execute().")

        # Run pre-step hook (e.g., update profiles for current timestep)
        if self._pre_step_func is not None:
            self._pre_step_func()

        # Layer actions for hierarchical structure and act (delegates to subordinates)
        actions = self.layer_actions(actions)
        self.act(actions, proxy)

        # get latest global state in dict format for simulation pipeline
        global_state = proxy.get_global_states(self.agent_id, self.protocol, for_simulation=True)
        # run external environment simulation step (upon action -> agent state update)
        updated_global_state = self.simulate(global_state)

        # broadcast updated global state via proxy
        proxy.set_global_state(updated_global_state)

        # R1: collect step statistics from subordinates only, not self
        obs: Dict[AgentID, Any] = {}
        rewards: Dict[AgentID, float] = {}
        infos: Dict[AgentID, Dict] = {}
        terminateds: Dict[AgentID, bool] = {}
        truncateds: Dict[AgentID, bool] = {}
        for subordinate in self.subordinates.values():
            obs.update(subordinate.observe(proxy=proxy))
            rewards.update(subordinate.compute_rewards(proxy))
            infos.update(subordinate.get_info(proxy))
            terminateds.update(subordinate.get_terminateds(proxy, env_context))
            truncateds.update(subordinate.get_truncateds(proxy, env_context))

        # Compute __all__ using configurable semantics
        agent_ids = {k for k in terminateds if k != "__all__"}
        semantics = env_context.all_semantics if env_context else "all"

        terminateds["__all__"] = compute_all_done(terminateds, agent_ids, semantics)
        truncateds["__all__"] = compute_all_done(truncateds, agent_ids, semantics)

        # Environment-level max_steps truncation
        if env_context and env_context.max_steps is not None:
            if env_context.step_count >= env_context.max_steps:
                for k in truncateds:
                    if k != "__all__":
                        truncateds[k] = True
                truncateds["__all__"] = True

        # set step results in proxy agent
        proxy.set_step_result(obs, rewards, terminateds, truncateds, infos)


    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        """Orchestrate periodic agent ticks and physics. [Event-Driven Mode]

        R1: SystemAgent does not observe, act, or compute reward.
        It only:
        1. Runs the pre-step hook
        2. Kicks off periodic agents on the first cycle (they self-reschedule after)
        3. Schedules the next periodic physics simulation
        """
        if not self._simulation_configured:
            raise RuntimeError("Simulation not configured. Call set_simulation() before tick().")

        self._timestep = current_time

        # Run pre-step hook (e.g., update profiles for current timestep)
        if self._pre_step_func is not None:
            self._pre_step_func()

        # First cycle only: kick off periodic agents. After that, they self-reschedule (R5).
        if not self._subordinates_kicked_off:
            for agent_id in self._periodic_agents:
                scheduler.schedule_agent_tick(agent_id)
            self._subordinates_kicked_off = True

        # Schedule periodic physics
        scheduler.schedule_simulation(self.agent_id, self._simulation_wait_interval)


    # ============================================
    # Custom Handlers for Event-Driven Execution
    # ============================================
    @Agent.handler("agent_tick")
    def agent_tick_handler(self, event: Event, scheduler: EventScheduler) -> None:
        self.tick(scheduler, event.timestamp)

    @Agent.handler("condition_trigger")
    def condition_trigger_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """No-op: SystemAgent is an orchestrator and does not react to conditions.

        Condition monitors should target field or coordinator agents instead.
        """
        logger.warning(
            "SystemAgent received CONDITION_TRIGGER (monitor_id=%s). "
            "Conditions should target field/coordinator agents.",
            event.payload.get("monitor_id"),
        )

    @Agent.handler("env_update")
    def env_update_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Handle an exogenous disturbance event (Class 4).

        Causal chain when ``requires_physics=True`` (default, recommended):
          1. ``apply_disturbance()`` mutates environment state
          2. Immediate SIMULATION scheduled (delay=0)
          3. Physics re-solves with post-disturbance state
          4. Condition monitors evaluated with fresh post-physics state
             (via existing ``MSG_SET_STATE_COMPLETION`` path)

        When ``requires_physics=False`` (lightweight disturbances only):
          1. ``apply_disturbance()`` mutates environment state
          2. Condition monitors evaluated against **cached** post-physics
             state from the most recent SIMULATION.

        .. note::
           ``requires_physics=False`` evaluates conditions against stale
           state because the event-driven architecture uses async messaging
           and there is no synchronous proxy access. If the disturbance
           changes state that condition monitors watch, use
           ``requires_physics=True`` to ensure fresh evaluation.
        """
        disturbance = event.payload.get("disturbance")
        if disturbance is None:
            logger.warning("ENV_UPDATE event has no 'disturbance' in payload. Ignoring.")
            return

        if self._apply_disturbance_func is None:
            raise RuntimeError(
                "Received ENV_UPDATE but no apply_disturbance_func is configured. "
                "Override apply_disturbance() in your BaseEnv subclass."
            )

        self._apply_disturbance_func(disturbance)
        logger.info(
            "Applied disturbance: type=%s at t=%.3f",
            disturbance.disturbance_type,
            event.timestamp,
        )

        if disturbance.requires_physics:
            scheduler.schedule_simulation(
                agent_id=self.agent_id, delay=0.0,
                payload={"triggered_by": "disturbance"},
            )
        else:
            # Best-effort: evaluate against cached state. See docstring note.
            if self._last_post_physics_state is not None:
                self.evaluate_conditions(self._last_post_physics_state, scheduler)

    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """No-op: SystemAgent does not apply actions (R1)."""
        pass

    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Handle messages for physics orchestration.

        Two cases:
        a. MSG_GET_GLOBAL_STATE_RESPONSE → run simulation, send updated state to proxy
        b. MSG_SET_STATE_COMPLETION → physics done, broadcast MSG_PHYSICS_COMPLETED
           to periodic agents and self-reschedule
        """
        recipient_id = event.agent_id
        assert recipient_id == self.agent_id
        message_content = event.payload.get("message", {})
        assert isinstance(message_content, dict)

        if MSG_GET_GLOBAL_STATE_RESPONSE in message_content:
            # Run simulation with global state from proxy
            response_data = message_content[MSG_GET_GLOBAL_STATE_RESPONSE]
            global_state = response_data[MSG_KEY_BODY]
            updated_global_state = self.simulate(global_state)
            # Cache for condition evaluation after proxy confirms write
            self._last_post_physics_state = updated_global_state
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_SET_STATE: STATE_TYPE_GLOBAL, MSG_KEY_BODY: updated_global_state},
            )

        elif MSG_SET_STATE_COMPLETION in message_content:
            if message_content[MSG_SET_STATE_COMPLETION] != "success":
                raise ValueError("State update failed in proxy, cannot proceed")

            # Evaluate condition monitors against post-physics state (Class 3).
            if self._last_post_physics_state is not None:
                self.evaluate_conditions(self._last_post_physics_state, scheduler)
                self._last_post_physics_state = None

            # Pop the origin of this simulation from the FIFO queue.
            # The queue ensures correct pairing even when multiple
            # simulations interleave in the message pipeline.
            is_disturbance = (
                self._sim_origin_queue.popleft()
                if self._sim_origin_queue
                else False
            )

            if is_disturbance:
                # Disturbance-triggered SIMULATION: only evaluate conditions
                # (done above). Do NOT broadcast PHYSICS_COMPLETED or
                # self-reschedule — that would create a duplicate periodic
                # cycle and generate empty-pair rewards.
                pass
            else:
                # Periodic SIMULATION: broadcast PHYSICS_COMPLETED to all
                # periodic agents so they compute reward (R7), then
                # self-reschedule for next physics cycle.
                for agent_id in self._periodic_agents:
                    scheduler.schedule_message_delivery(
                        sender_id=self.agent_id,
                        recipient_id=agent_id,
                        message={MSG_PHYSICS_COMPLETED: "success"},
                    )
                scheduler.schedule_agent_tick(self.agent_id)

        else:
            raise NotImplementedError(f"SystemAgent received unknown message: {list(message_content.keys())}")

    @Agent.handler("simulation")
    def simulation_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Request global state from proxy to begin simulation cycle."""
        is_disturbance = event.payload.get("triggered_by") == "disturbance"
        self._sim_origin_queue.append(is_disturbance)
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={MSG_GET_INFO: INFO_TYPE_GLOBAL_STATE, MSG_KEY_PROTOCOL: self.protocol},
        )


    # ============================================
    # Condition Evaluation (Class 3)
    # ============================================
    def evaluate_conditions(
        self,
        post_physics_state: Dict[str, Any],
        scheduler: EventScheduler,
    ) -> List[Event]:
        """Evaluate registered condition monitors against post-physics state.

        This is the canonical evaluation point. Called after every SIMULATION
        completion (both periodic and disturbance-triggered) and after
        ``requires_physics=False`` disturbances.

        The scheduler stores the monitors (for reset/cooldown management),
        but SystemAgent owns the evaluation logic because condition
        evaluation is domain coordination, not time management.

        Args:
            post_physics_state: Global state dict from ``simulate()``.
            scheduler: Scheduler for scheduling CONDITION_TRIGGER events
                and cancelling preempted ticks.

        Returns:
            List of triggered CONDITION_TRIGGER events.
        """
        triggered: List[Event] = []
        to_remove: List[str] = []

        for monitor in scheduler.condition_monitors:
            if not monitor.condition_fn(post_physics_state):
                continue
            if scheduler.current_time - monitor._last_triggered < monitor.cooldown:
                continue

            event = Event(
                timestamp=scheduler.current_time,
                event_type=EventType.CONDITION_TRIGGER,
                agent_id=monitor.agent_id,
                priority=2,
                payload={
                    "monitor_id": monitor.monitor_id,
                    "condition": monitor.monitor_id,
                },
            )
            scheduler.schedule(event)
            triggered.append(event)
            monitor._last_triggered = scheduler.current_time

            if monitor.preempt_next_tick:
                scheduler.cancel_events(monitor.agent_id, EventType.AGENT_TICK)
            if monitor.one_shot:
                to_remove.append(monitor.monitor_id)

        for mid in to_remove:
            scheduler.deregister_condition(mid)

        return triggered

    # ============================================
    # Simulation related functions - SystemAgent-specific
    # ============================================
    def set_simulation(
        self,
        simulation_func: Callable,
        env_state_to_global_state: Callable,
        global_state_to_env_state: Callable,
        wait_interval: Optional[float] = None,
        pre_step_func: Optional[Callable] = None,
        apply_disturbance_func: Optional[Callable] = None,
    ):
        """Configure simulation pipeline callables from the environment.

        Args:
            simulation_func: Physics simulation function.
            env_state_to_global_state: Converts env state to global state dict.
            global_state_to_env_state: Converts global state dict to env state.
            wait_interval: Waiting time between action kick-off and simulation.
                If None, derives from current schedule_config.tick_interval.
            pre_step_func: Optional hook called at the start of each step.
            apply_disturbance_func: Optional callable to apply exogenous
                disturbances to the environment (Class 4).
                Signature: ``(disturbance: Disturbance) -> None``.
        """
        self._simulation_func = simulation_func
        self._env_state_to_global_state = env_state_to_global_state
        self._global_state_to_env_state = global_state_to_env_state
        self._explicit_wait_interval = wait_interval  # None means derive from schedule_config
        self._pre_step_func = pre_step_func
        self._apply_disturbance_func = apply_disturbance_func
        self._simulation_configured = True

    @property
    def _simulation_wait_interval(self) -> float:
        if self._explicit_wait_interval is not None:
            return self._explicit_wait_interval
        return self.schedule_config.tick_interval

    def simulate(self, global_state: Dict[AgentID, Any]) -> Any:
        # proxy.get_global_states() returns a flat {agent_id: state_dict} dict,
        # but global_state_to_env_state() expects {"agent_states": {agent_id: ...}}.
        # Wrap if needed so both execute() and event-driven paths work correctly.
        if "agent_states" not in global_state:
            global_state = {"agent_states": global_state}
        env_state = self._global_state_to_env_state(global_state)
        updated_env_state = self._simulation_func(env_state)
        updated_global_state = self._env_state_to_global_state(updated_env_state)
        return updated_global_state


    # ============================================
    # Utility Methods
    # ============================================
    def __repr__(self) -> str:
        num_subs = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"SystemAgent(id={self.agent_id}, subordinates={num_subs}, protocol={protocol_name})"


def build_system_agent(
    schedule_config: Optional[ScheduleConfig] = None,
    **kwargs: Any,
) -> SystemAgent:
    """Build a SystemAgent.

    Parameters
    ----------
    schedule_config : ScheduleConfig, optional
        Custom schedule config.  Defaults to ``DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG``.
    **kwargs
        Additional keyword arguments forwarded to ``SystemAgent.__init__``.
    """
    sys_kwargs: dict = {"agent_id": SYSTEM_AGENT_ID}
    if schedule_config is not None:
        sys_kwargs["schedule_config"] = schedule_config
    sys_kwargs.update(kwargs)
    return SystemAgent(**sys_kwargs)

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.proxy_agent import Proxy
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.observation import Observation
from heron.core.state import State, SystemAgentState
from heron.utils.typing import AgentID, MultiAgentDict
from heron.core.policies import Policy
from heron.protocols.base import Protocol
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

        # Resolve which agents are periodic vs reactive (R2-R4)
        self._periodic_agents: List[AgentID] = self.resolve_periodic_children()

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
    
    def execute(self, actions: Dict[AgentID, Any], proxy: Optional[Proxy] = None) -> None:
        """Execute actions with hierarchical coordination and simulation. [Training Mode]

        R1: SystemAgent is a pure orchestrator — it does not observe, act, or
        compute reward for itself.  All returned dicts contain only subordinate
        agent entries.
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
            terminateds.update(subordinate.get_terminateds(proxy))
            truncateds.update(subordinate.get_truncateds(proxy))

        terminateds["__all__"] = all(terminateds.get(k, False) for k in terminateds if k != "__all__")
        truncateds["__all__"] = all(truncateds.get(k, False) for k in truncateds if k != "__all__")

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
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_SET_STATE: STATE_TYPE_GLOBAL, MSG_KEY_BODY: updated_global_state},
            )

        elif MSG_SET_STATE_COMPLETION in message_content:
            if message_content[MSG_SET_STATE_COMPLETION] != "success":
                raise ValueError("State update failed in proxy, cannot proceed")

            # Physics is done. Broadcast MSG_PHYSICS_COMPLETED to all periodic agents
            # so they compute reward from post-physics state (R7).
            for agent_id in self._periodic_agents:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=agent_id,
                    message={MSG_PHYSICS_COMPLETED: "success"},
                )

            # Self-reschedule for next physics cycle
            scheduler.schedule_agent_tick(self.agent_id)

        else:
            raise NotImplementedError(f"SystemAgent received unknown message: {list(message_content.keys())}")

    @Agent.handler("simulation")
    def simulation_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Request global state from proxy to begin simulation cycle."""
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={MSG_GET_INFO: INFO_TYPE_GLOBAL_STATE, MSG_KEY_PROTOCOL: self.protocol},
        )


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
    ):
        """
        simulation_func: simulation function passed from environment
        wait_interval: waiting time between action kick-off and simulation starts.
            If None, derives from current schedule_config.tick_interval at runtime.
        pre_step_func: optional hook called at the start of each step (before actions)
        """
        self._simulation_func = simulation_func
        self._env_state_to_global_state = env_state_to_global_state
        self._global_state_to_env_state = global_state_to_env_state
        self._explicit_wait_interval = wait_interval  # None means derive from schedule_config
        self._pre_step_func = pre_step_func
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
    @property
    def coordinators(self) -> Dict[AgentID, CoordinatorAgent]:
        """Alias for subordinates - more descriptive for SystemAgent context."""
        return self.subordinates

    def __repr__(self) -> str:
        num_coords = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"SystemAgent(id={self.agent_id}, coordinators={num_coords}, protocol={protocol_name})"


def build_system_agent(
    coordinator_agents: List[CoordinatorAgent],
    schedule_config: Optional[ScheduleConfig] = None,
    **kwargs: Any,
) -> SystemAgent:
    """Build a SystemAgent from a list of coordinator agents.

    This is the recommended way to create a SystemAgent with a custom
    ``schedule_config``.  For the default schedule config, you can simply
    pass ``coordinator_agents`` directly to ``BaseEnv`` and it will
    create one automatically.

    Parameters
    ----------
    coordinator_agents : list[CoordinatorAgent]
        The coordinator agents that will be subordinates of the system agent.
    schedule_config : ScheduleConfig, optional
        Custom schedule config.  Defaults to ``DEFAULT_SYSTEM_AGENT_SCHEDULE_CONFIG``.
    **kwargs
        Additional keyword arguments forwarded to ``SystemAgent.__init__``.
    """
    sys_kwargs: dict = {
        "agent_id": SYSTEM_AGENT_ID,
        "subordinates": {agent.agent_id: agent for agent in coordinator_agents},
    }
    if schedule_config is not None:
        sys_kwargs["schedule_config"] = schedule_config
    sys_kwargs.update(kwargs)
    return SystemAgent(**sys_kwargs)



from abc import abstractmethod
from typing import Any, Dict, Optional, List

import numpy as np
from gymnasium.spaces import Box, Space

from heron.agents.base import Agent
from heron.core.feature import Feature
from heron.core.observation import Observation
from heron.core.state import FieldAgentState, State
from heron.core.action import Action
from heron.core.policies import Policy
from heron.utils.typing import AgentID
from heron.protocols.base import Protocol
from heron.scheduling.schedule_config import DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG, ScheduleConfig
from heron.scheduling.scheduler import EventScheduler, Event
from heron.agents.proxy_agent import Proxy
from heron.agents.constants import (
    FIELD_LEVEL,
    PROXY_AGENT_ID,
    MSG_GET_INFO,
    MSG_SET_STATE,
    STATE_TYPE_LOCAL,
    MSG_SET_TICK_RESULT,
    MSG_SET_STATE_COMPLETION,
    MSG_PHYSICS_COMPLETED,
    INFO_TYPE_OBS,
    INFO_TYPE_LOCAL_STATE,
    MSG_KEY_BODY,
    MSG_KEY_PROTOCOL,
    MSG_GET_OBS_RESPONSE,
    MSG_GET_LOCAL_STATE_RESPONSE,
)


class FieldAgent(Agent):
    def __init__(
        self,
        agent_id: AgentID,
        features: Optional[List[Feature]] = None,
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        schedule_config: Optional[ScheduleConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None
    ):

        self.protocol = protocol
        self.policy = policy

        super().__init__(
            agent_id=agent_id,
            level=FIELD_LEVEL,
            features=features,
            upstream_id=upstream_id,
            subordinates=None, # L1 agent has no subordinate
            env_id=env_id,
            schedule_config=schedule_config or DEFAULT_FIELD_AGENT_SCHEDULE_CONFIG,
            policy=policy,
            protocol=protocol,
        )

    # ============================================
    # Functions requiring overriding:
    # 
    # [Must]
    # 1. compute_local_reward
    #   - applying specific reward computation mechanism
    #   - may use additional info via proxy (for additional info) + protocol (for visibility and info control)
    # 2. set_action
    #   - set self.action
    # 3. set_state
    #   - update self.state
    # 4. apply_action
    #   - given self.action, update self.state
    # 
    # [Optional]
    # 1. init_state & init_action
    #   - by default, state is initialized with features from parent coordinator, and action is
    #     initialized as empty Action. Override if you want to add additional features or custom initialization logic.
    # 2. get_local_info
    # 3. is_terminated
    # 4. is_truncated
    # 5. post_proxy_attach
    # ============================================

    def init_state(self, features: List[Feature] = []) -> State:
        """Initialize a FieldAgentState from the provided features."""
        return FieldAgentState(
            owner_id=self.agent_id,
            owner_level=FIELD_LEVEL,
            features={f.feature_name: f for f in features}
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        """Initialize an empty Action (override to define custom action structure)."""
        return Action()

    @abstractmethod
    def set_state(self, *args, **kwargs) -> None:
        """Define how to update state for this agent."""
        ...

    @abstractmethod
    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Define how to set action for this agent."""
        ...

    def post_proxy_attach(self, proxy: "Proxy") -> None:
        """Hook for any additional setup after proxy attachment and global state initialization."""
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space(proxy)

    @abstractmethod
    def compute_local_reward(self, local_state: dict, prev_post_physics_state: Optional[dict] = None) -> float:
        """Define reward computation mechanism for this agent.

        Args:
            local_state: Current post-physics local state.
            prev_post_physics_state: Previous post-physics local state
                (event-driven mode). None on first call or in step-based mode.
        """
        ...

    @abstractmethod
    def apply_action(self):
        """Define action effect mechanism for this agent."""
        ...

    def get_action_space(self) -> Space:
        """Get action space based on agent action. [Both Modes]

        Returns:
            Gymnasium Space object
        """
        return self.action.space

    def get_observation_space(self, proxy: Optional[Proxy] = None) -> Space:
        """Get observation space based on agent state. [Both Modes]

        Returns:
            Gymnasium Space object
        """
        if hasattr(self, "observation_space") and self.observation_space is not None:
            return self.observation_space
        if not proxy:
            raise ValueError("[get_observation_space]: We need proxy agent to retrieve observations for field agent")

        sample_obs = self.observe(proxy=proxy)
        obs_vector = sample_obs[self.agent_id]
        if len(obs_vector.shape) == 1:  # Vector observation
            return Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_vector.shape,
                dtype=np.float32,
            )
        else:
            raise NotImplementedError(
                "Extend _get_observation_space to handle images, discrete, "
                "or structured observations."
            )
    
    # ============================================
    # Core Lifecycle Methods Overrides (see heron/agents/base.py for more details)
    # ============================================
    def reset(self, *, seed: Optional[int] = None, proxy: Optional[Proxy] = None, **kwargs) -> Any:
        super().reset(seed=seed, proxy=proxy, **kwargs)

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space(proxy)
            
    # execute() inherited from base class - uses default implementation

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
        reschedule: bool = True,
    ) -> None:
        """Action phase: observe → decide → act. [Event-Driven Mode]

        Periodic agents self-reschedule here (R5). Reactive agents do not —
        they are ticked by their upstream coordinator after action coordination.

        Args:
            scheduler: Event scheduler.
            current_time: Current simulation time.
            reschedule: If False, skip self-reschedule. Used by
                ``condition_trigger_handler`` so reactive wakeups don't
                create duplicate periodic cycles.
        """
        super().tick(scheduler, current_time)  # Update internal timestep and check for upstream actions

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

        Runs the same observe→decide→act cycle as a regular tick, but
        does NOT self-reschedule. Condition-triggered wakeups are reactive
        one-offs — the agent's periodic schedule is unaffected.

        Override for custom reactive logic.
        Payload contains monitor_id identifying which condition fired.
        """
        self.tick(scheduler, event.timestamp, reschedule=False)

    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        self.apply_action()

        # R7: cache (obs, action) only after action has landed on state.
        if self._pending_obs_queue:
            obs = self._pending_obs_queue.popleft()
            self._cache_obs_action(obs, self.action)

        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={MSG_SET_STATE: STATE_TYPE_LOCAL, "body": self.state.to_dict(include_metadata=True)},
        )

    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Handle messages for action and reward phases.

        Three cases:
        a. MSG_GET_OBS_RESPONSE → sync state, compute action, cache (obs, action)
        b. MSG_PHYSICS_COMPLETED → request post-physics local state for reward
        c. MSG_GET_LOCAL_STATE_RESPONSE → compute reward at physics boundary (R7)
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

            # R7: queue obs for caching at action_effect time (not here).
            # Caching here would create fake rewards if physics runs before
            # the action_effect lands.
            if self.action:
                self._pending_obs_queue.append(obs)

        elif MSG_SET_STATE_COMPLETION in message_content:
            # No-op: proxy acknowledges local state update from action_effect.
            # Reward computation happens later at physics boundary.
            pass

        elif MSG_PHYSICS_COMPLETED in message_content:
            # Physics just completed. Request post-physics local state for reward.
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: self.protocol},
            )

        elif MSG_GET_LOCAL_STATE_RESPONSE in message_content:
            # R7: compute reward at physics boundary
            response_data = message_content[MSG_GET_LOCAL_STATE_RESPONSE]
            local_state = response_data[MSG_KEY_BODY]

            self.sync_state_from_observed(local_state)

            reward = self.compute_local_reward(local_state, self._prev_post_physics_state)
            obs_action_pairs = self._flush_obs_action_cache()

            tick_result = {
                "reward": reward,
                "terminated": self.is_terminated(local_state),
                "truncated": self.is_truncated(local_state),
                "info": self.get_local_info(local_state),
                "obs_action_pairs": obs_action_pairs,
            }
            self._prev_post_physics_state = local_state

            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_SET_TICK_RESULT: INFO_TYPE_LOCAL_STATE, MSG_KEY_BODY: tick_result},
            )

            # Reactive agents: notify upstream coordinator that reward is done
            if not self.is_periodic and self.upstream_id:
                scheduler.schedule_message_delivery(
                    sender_id=self.agent_id,
                    recipient_id=self.upstream_id,
                    message={"sub_reward_complete": self.agent_id},
                )

        else:
            raise NotImplementedError(f"FieldAgent received unknown message: {list(message_content.keys())}")
    

    # ============================================
    # Utility Methods (Both Modes)
    # ============================================
    def __repr__(self) -> str:
        return f"FieldAgent(id={self.agent_id}, policy={self.policy}, protocol={self.protocol})"


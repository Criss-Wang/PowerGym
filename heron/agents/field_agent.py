

from typing import Any, Dict, Optional, List

import numpy as np
from gymnasium.spaces import Box, Space

from heron.agents.base import Agent
from heron.core.feature import FeatureProvider
from heron.core.observation import Observation
from heron.core.state import FieldAgentState, State
from heron.core.action import Action
from heron.core.policies import Policy
from heron.utils.typing import AgentID
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig
from heron.scheduling.scheduler import EventScheduler, Event
from heron.agents.proxy_agent import ProxyAgent
from heron.agents.constants import (
    FIELD_LEVEL,
    PROXY_AGENT_ID,
    DEFAULT_FIELD_TICK_INTERVAL,
    MSG_GET_INFO,
    MSG_SET_STATE_COMPLETION,
    MSG_SET_TICK_RESULT,
    INFO_TYPE_OBS,
    INFO_TYPE_LOCAL_STATE,
    MSG_KEY_BODY,
    MSG_KEY_PROTOCOL,
)


class FieldAgent(Agent):
    def __init__(
        self,
        agent_id: AgentID,
        features: List[FeatureProvider] = [],
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
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
            tick_config=tick_config or TickConfig.deterministic(tick_interval=DEFAULT_FIELD_TICK_INTERVAL),
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

    def init_state(self, features: List[FeatureProvider] = []) -> State:
        return FieldAgentState(
            owner_id=self.agent_id,
            owner_level=FIELD_LEVEL,
            features={f.feature_name: f for f in features}
        )

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        return Action()

    def set_state(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "No implementation of set_state found, you need to define how to update state for this agent"
        )

    def set_action(self, action: Any, *args, **kwargs) -> None:
        raise NotImplementedError(
            "No implementation of set_action found, you need to define how to set action for this agent"
        )

    def post_proxy_attach(self, proxy: "ProxyAgent") -> None:
        """Hook for any additional setup after proxy attachment and global state initialization."""
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space(proxy)

    def compute_local_reward(self, local_state: dict) -> float:
        raise NotImplementedError(
            "No implementation of compute_local_reward found, you need to define reward computation mechanism for this agent"
        )

    def apply_action(self):
        raise NotImplementedError(
            "No implementation of apply_action found, you need to define action effect mechanism for this agent"
        )

    def get_action_space(self) -> Space:
        """Get action space based on agent action. [Both Modes]

        Returns:
            Gymnasium Space object
        """
        return self.action.space

    def get_observation_space(self, proxy: Optional[ProxyAgent] = None) -> Space:
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
    def reset(self, *, seed: Optional[int] = None, proxy: Optional[ProxyAgent] = None, **kwargs) -> Any:
        super().reset(seed=seed, proxy=proxy, **kwargs)

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space(proxy)
            
    # execute() inherited from base class - uses default implementation

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        """
        Action phase - equivalent to `self.act`

        Note: 
        - FieldAgent ticks only upon CoordinatorAgent.tick (see heron/agents/coordinator_agent.py)
        - Upstream actions checked in base.Agent.tick() via _check_for_upstream_action()
        """
        super().tick(scheduler, current_time)  # Update internal timestep and check for upstream actions

        # If we received an upstream action (from coordinator), apply it
        if self._upstream_action is not None:
            self.set_action(self._upstream_action)
            self._upstream_action = None  # Clear after use
            # Schedule action effect
            scheduler.schedule_action_effect(
                agent_id=self.agent_id,
                delay=self._tick_config.act_delay,
            )
        elif self.policy:
            # Compute & execute self action
            # Ask proxy_agent for global state to compute local action
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_GET_INFO: INFO_TYPE_OBS, MSG_KEY_PROTOCOL: self.protocol},
                delay=self._tick_config.msg_delay,
            )
        else:
            print(f"{self} doesn't act iself, becase there's no action policy")
    
    # ============================================
    # Custom Handlers for Event-Driven Execution
    # ============================================
    @Agent.handler("agent_tick")
    def agent_tick_handler(self, event: Event, scheduler: EventScheduler) -> None:
        self.tick(scheduler, event.timestamp)

    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        self.apply_action()
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={"set_state": "local", "body": self.state.to_dict(include_metadata=True)},
            delay=self._tick_config.msg_delay,
        )

    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Deliver message via message broker."""
        recipient_id = event.agent_id
        assert recipient_id == self.agent_id
        message_content = event.payload.get("message", {})

        # Publish message via broker
        if "get_obs_response" in message_content:
            assert isinstance(message_content, dict)
            response_data = message_content["get_obs_response"]
            body = response_data[MSG_KEY_BODY]

            # Proxy sends both obs and local_state (design principle: agent asks for obs, proxy gives both)
            obs_dict = body["obs"]
            local_state = body["local_state"]

            # Deserialize observation dict back to Observation object
            obs = Observation.from_dict(obs_dict)

            # Sync state first (proxy gives both obs & state)
            self.sync_state_from_observed(local_state)

            # Compute action - policy decides which parts of observation to use
            self.compute_action(obs, scheduler)
        elif "get_local_state_response" in message_content:
            response_data = message_content["get_local_state_response"]
            local_state = response_data[MSG_KEY_BODY]

            # Sync internal state with what's stored in proxy (may have been modified by simulation)
            self.sync_state_from_observed(local_state)

            tick_result = {
                "reward": self.compute_local_reward(local_state),
                "terminated": self.is_terminated(local_state),
                "truncated": self.is_truncated(local_state),
                "info": self.get_local_info(local_state)
            }

            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_SET_TICK_RESULT: INFO_TYPE_LOCAL_STATE, MSG_KEY_BODY: tick_result},
                delay=self._tick_config.msg_delay,
            )

        elif MSG_SET_STATE_COMPLETION in message_content:
            if message_content[MSG_SET_STATE_COMPLETION] != "success":
                raise ValueError(f"State update failed in proxy, cannot proceed with reward computation")

            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={MSG_GET_INFO: INFO_TYPE_LOCAL_STATE, MSG_KEY_PROTOCOL: self.protocol},
                delay=self._tick_config.msg_delay,
            )
        else:
            raise NotImplementedError
    

    # ============================================
    # Utility Methods (Both Modes)
    # ============================================
    def __repr__(self) -> str:
        return f"FieldAgent(id={self.agent_id}, policy={self.policy}, protocol={self.protocol})"


    # def act(self, observation: Observation, upstream_action: Any = None) -> None:
    #     """Compute and apply action. [Training Only - Direct Call]

    #     Note: In Testing (Option B), tick() computes actions directly via
    #     _handle_coordinator_action()/_handle_local_action() instead
    #     of calling this method.

    #     Routes to coordinator-directed or self-directed action computation based on
    #     whether upstream_action is provided.

    #     Args:
    #         observation: Structured observation
    #         upstream_action: Optional action from coordinator (coordinator-directed)
    #     """
    #     if upstream_action is not None:
    #         # Coordinator-directed: use the action provided by upstream coordinator
    #         action = self._handle_coordinator_action(upstream_action, observation)
    #     else:
    #         # Self-directed: compute action using local policy
    #         action = self._handle_local_action(observation)

    #     self.action.set_values(action)

    #     # Phase 1: Update action-dependent features immediately after action is set
    #     self._update_action_features(action, observation)

    # ============================================
    # Action Handling (Both Modes)
    # ============================================

    # def _handle_coordinator_action(
    #     self,
    #     upstream_action: Any,
    #     observation: Observation
    # ) -> Action | None:
    #     """Handle coordinator-directed action. [Both Modes]

    #     Args:
    #         upstream_action: Action assigned by coordinator
    #         observation: Current observation (unused in coordinator-directed mode)

    #     Returns:
    #         Action to execute (passthrough of upstream_action)
    #     """
    #     return upstream_action

    # def _handle_local_action(self, observation: Observation) -> Action | None:
    #     """Handle self-directed action computation using local policy. [Both Modes]

    #     Args:
    #         observation: Current observation including messages

    #     Returns:
    #         Action computed by local policy or None if no action

    #     Raises:
    #         ValueError: If no policy is defined for self-directed mode
    #     """
    #     if self.policy is None:
    #         raise ValueError(
    #             "No policy defined for FieldAgent. "
    #             "Agent requires either upstream_action (coordinator-directed) or policy (self-directed)."
    #         )

    #     action = self.policy.forward(observation)
    #     return action

    # ============================================
    # Event-Driven Execution (Option B - Testing)
    # ============================================

    # def tick(
    #     self,
    #     scheduler: EventScheduler,
    #     current_time: float,
    #     global_state: Optional[Dict[str, Any]] = None,
    #     proxy: Optional[Agent] = None,
    # ) -> None:
    #     """Execute one tick in event-driven mode. [Testing Only]

    #     Workflow:
    #     1. Check message broker for upstream action from coordinator
    #     2. Get observation (potentially delayed via ProxyAgent)
    #     3. Send observation to upstream coordinator (if has upstream)
    #     4. Compute action using upstream action or own policy
    #     5. Schedule ACTION_EFFECT with act_delay

    #     Args:
    #         scheduler: EventScheduler for scheduling future events
    #         current_time: Current simulation time
    #         global_state: Optional global state for observation
    #         proxy: Optional ProxyAgent for delayed observations
    #     """
    #     self._timestep = current_time

    #     # Check message broker for upstream action from coordinator
    #     upstream_action = None
    #     actions = self.receive_action_messages()
    #     if actions:
    #         upstream_action = actions[-1]  # Use most recent action

    #     # Get observation (with delay if proxy provided and obs_delay > 0)
    #     if proxy is not None and self._tick_config.obs_delay > 0:
    #         # Use proxy for delayed observations
    #         delayed_time = current_time - self._tick_config.obs_delay
    #         proxy_state = self.request_state_from_proxy(proxy, at_time=delayed_time)
    #         observation = self._build_observation_from_proxy(proxy_state)
    #     else:
    #         # Direct observation (proxy used for neighbor states if available)
    #         observation = self.observe(global_state, proxy=proxy)
    #     self._last_observation = observation

    #     # Send observation to upstream coordinator
    #     if self.upstream_id is not None:
    #         self.send_observation_to_upstream(observation, scheduler=scheduler)

    #     # Compute action (coordinator-directed if upstream provided, else self-directed)
    #     if upstream_action is not None:
    #         action = self._handle_coordinator_action(upstream_action, observation)
    #     elif self.policy is not None:
    #         action = self._handle_local_action(observation)
    #     else:
    #         # No action to take - agent is passive
    #         return

    #     self.action.set_values(action)

    #     # Phase 1: Update action-dependent features immediately after action is set
    #     self._update_action_features(action, observation)

    #     # Schedule delayed action effect
    #     if self._tick_config.act_delay > 0:
    #         scheduler.schedule_action_effect(
    #             agent_id=self.agent_id,
    #             action=self.action.vector(),
    #             delay=self._tick_config.act_delay,
    #         )

    # def _build_observation_from_proxy(
    #     self, proxy_state: Dict[str, Any]
    # ) -> Observation:
    #     """Build observation from proxy state. [Testing Only]

    #     Override in subclasses for custom observation building from proxy state.

    #     Args:
    #         proxy_state: Filtered state dict from ProxyAgent

    #     Returns:
    #         Observation built from proxy state
    #     """
    #     return Observation(
    #         timestamp=self._timestep,
    #         local={
    #             OBS_KEY_STATE: self.state.vector(),
    #             OBS_KEY_PROXY_STATE: proxy_state,
    #         }
    #     )


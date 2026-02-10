

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.protocols.base import Protocol
from heron.messaging import ChannelManager, Message, MessageBroker, MessageType
from heron.utils.typing import AgentID
from heron.scheduling.scheduler import Event, EventScheduler
from heron.agents.constants import (
    PROXY_LEVEL,
    PROXY_AGENT_ID,
    DEFAULT_HISTORY_LENGTH,
    MSG_GET_INFO,
    MSG_SET_STATE,
    MSG_SET_TICK_RESULT,
    MSG_SET_STATE_COMPLETION,
    INFO_TYPE_OBS,
    INFO_TYPE_GLOBAL_STATE,
    INFO_TYPE_LOCAL_STATE,
    STATE_TYPE_GLOBAL,
    STATE_TYPE_LOCAL,
    MSG_KEY_BODY,
    MSG_KEY_PROTOCOL,
)


if TYPE_CHECKING:
    from heron.core.state import State

class ProxyAgent(Agent):

    def __init__(
        self,
        agent_id: AgentID = PROXY_AGENT_ID,
        env_id: Optional[str] = None,
        registered_agents: Optional[List[AgentID]] = None,
        visibility_rules: Optional[Dict[AgentID, List[str]]] = None,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        message_broker: Optional[MessageBroker] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            level=PROXY_LEVEL,
            upstream_id=None,
            subordinates={},
            env_id=env_id,
        )

        self.registered_agents: List[AgentID] = registered_agents or []
        self.visibility_rules: Dict[AgentID, List[str]] = visibility_rules or {}
        self.state_cache: Dict[str, Any] = {}
        self._agent_levels: Dict[AgentID, int] = {}  # Track agent hierarchy levels for visibility
        self.history_length = history_length

        if message_broker is not None:
            raise ValueError("No message broker detected!")
        self.set_message_broker(message_broker)

    # ============================================
    # Initialization and State/Action Management Overrides
    # Note that ProxyAgent does not maintain its own state or action in the traditional sense, so these methods are either no-ops or can be used to set up any necessary internal structures for
    # managing the proxy's functionality (e.g. state cache, visibility rules, etc.)
    # ============================================
    def init_state(self) -> None:
        pass

    def init_action(self) -> None:
        pass

    def set_state(self, *args, **kwargs) -> None:
        pass

    def set_action(self, action: Any, *args, **kwargs) -> None:
        pass

    def register_agent(self, agent_id: AgentID, agent_state: Optional["State"] = None) -> None:
        """Register a new agent that can request state.

        Args:
            agent_id: Agent ID to register
            agent_state: Optional initial state of the agent
        """
        if agent_id == self.agent_id:
            print("Proxy agent doesn't register itself.")
            return

        if agent_id not in self.registered_agents:
            self.registered_agents.append(agent_id)

        # Always update state if provided (supports both init and reset)
        if agent_state is not None:
            # Track agent level for visibility checks
            self._agent_levels[agent_id] = agent_state.owner_level
            # Store State object directly (no serialization!)
            self.set_local_state(agent_id, agent_state)

    def init_global_state(self) -> None:
        """Initialize global state by compiling all registered agent states.

        Should be called after all agents are registered and their initial states are set.
        """
        if "agents" not in self.state_cache or not self.state_cache["agents"]:
            print("Warning: No agent states to compile into global state")
            return

        # Compile all agent states into global state
        if "global" not in self.state_cache:
            self.state_cache["global"] = {}

        # Aggregate relevant global information from all agents
        # (Override this method in subclasses for custom aggregation logic)
        for agent_id, agent_state in self.state_cache["agents"].items():
            # Store agent states as part of global state for now
            # Subclasses can implement more sophisticated aggregation
            if "agent_states" not in self.state_cache["global"]:
                self.state_cache["global"]["agent_states"] = {}
            self.state_cache["global"]["agent_states"][agent_id] = agent_state

    def _setup_channels(self) -> None:
        if self._message_broker is None:
            raise ValueError("Message broker is required to setup channels in ProxyAgent")

        # Create proxy->agent channels for distributing state
        for agent_id in self.registered_agents:
            agent_channel = ChannelManager.info_channel(
                self.agent_id, agent_id, self.env_id
            )
            self._message_broker.create_channel(agent_channel)

    # ============================================
    # Core Agent Lifecycle Methods Overrides (see heron/agents/base.py for more details)
    # Note that ProxyAgent does not follow the standard observe-decide-act loop, so execute and tick are empty logic.
    # ============================================
    def initialize(self, proxy = None):
        # Doesn't need to register with itself with proxy.
        self._setup_channels()

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset proxy agent state.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        # ProxyAgent doesn't need parent reset - it manages its own state cache
        self.state_cache = {}

    def execute(self, actions: Dict[AgentID, Any], proxy: Optional["ProxyAgent"] = None) -> None:
        pass

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        pass

    # ============================================
    # Custom Handlers for Event-Driven Execution (see heron/scheduling/scheduler.py for more details on event handling)
    # ============================================
    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        print("using proxy agent message delivery handler")
        recipient_id = event.agent_id
        if recipient_id != self.agent_id:
            raise ValueError(f"Event {event} sent to {event.agent_id} is handled in proxy_agent!")
        sender_id = event.payload.get("sender")
        message_content = event.payload.get("message", {})

        assert self._message_broker
        if MSG_GET_INFO in message_content:
            info_type = message_content[MSG_GET_INFO]
            protocol = message_content.get(MSG_KEY_PROTOCOL)
            info = self._handle_get_info_request(sender_id, info_type, protocol)

            # Serialize Observation objects before sending via message
            # In async/distributed systems, objects must be serialized for message passing
            # Observation.to_dict() converts to {"timestamp": float, "local": dict, "global_info": dict}
            info_type_key = "get_" + info_type + "_response" # e.g. obs -> get_obs_response

            # Convert to dict: Observation objects need .to_dict(), state dicts are already dicts
            if isinstance(info, Observation):
                info_data = info.to_dict()
            else:
                info_data = info  # Already a dict (from get_global_states or get_local_state)

            scheduler.schedule_message_delivery(
                sender_id=recipient_id, # same as self.agent_id
                recipient_id=sender_id,
                message={
                    info_type_key: {
                        MSG_KEY_BODY: info_data
                    }
                },
                delay=self._tick_config.msg_delay,
            )
        elif MSG_SET_STATE in message_content:
            from heron.core.state import State

            state_type = message_content[MSG_SET_STATE]
            if state_type == STATE_TYPE_GLOBAL:
                # Global state is dict of {agent_id: state_dict}
                global_state_payload = message_content.get(MSG_KEY_BODY, {})
                agent_states = global_state_payload.get("agent_states", {})
                for agent_id, state_dict in agent_states.items():
                    # Deserialize dict → State object
                    state_obj = State.from_dict(state_dict)
                    self.set_local_state(agent_id, state_obj)
            elif state_type == STATE_TYPE_LOCAL:
                state_dict = message_content.get(MSG_KEY_BODY, {})
                # Deserialize dict → State object
                state_obj = State.from_dict(state_dict)
                self.set_local_state(state_obj.owner_id, state_obj)
            else:
                raise NotImplementedError(f"Unknown state type {state_type} in {MSG_SET_STATE} message")
            scheduler.schedule_message_delivery(
                sender_id=recipient_id, # same as self.agent_id
                recipient_id=sender_id,
                message={MSG_SET_STATE_COMPLETION: "success"},
                delay=self._tick_config.msg_delay,
            )
        elif MSG_SET_TICK_RESULT in message_content:
            result_type = message_content[MSG_SET_TICK_RESULT]
            tick_result = message_content.get(MSG_KEY_BODY, {})
            # TODO: save tick result per-agent in proxy for later retrieval
            print(f"Received tick result from {sender_id}: {tick_result}")
        else:
            raise NotImplementedError(f"Unknown message content {message_content} in message_delivery to proxy_agent")

    def _handle_get_info_request(self, sender_id: AgentID, request_type: str, protocol: Optional[Protocol] = None):
        if request_type == INFO_TYPE_OBS:
            return self.get_observation(sender_id, protocol)
        elif request_type == INFO_TYPE_GLOBAL_STATE:
            return self.get_global_states(sender_id, protocol)
        elif request_type == INFO_TYPE_LOCAL_STATE:
            return self.get_local_state(sender_id, protocol)
        else:
            raise NotImplementedError("not yet complete")
    
    # ============================================
    # Core Logic Methods for Proxy Functionality
    # These methods define the core logic of how the proxy agent computes observations, manages state, and handles requests from other agents. They are designed to be overridden in subclasses to implement specific proxy behaviors (e.g. different state aggregation methods, visibility rules, reward computation logic, etc.)
    # ============================================
    def get_observation(self, sender_id: AgentID, protocol: Optional[Protocol] = None) -> Observation:
        """
        Compute observation for particular sender_id.
        A protocol may be given to specify the format of the observation, but the content is determined by the proxy agent's logic.

        if sender_id is SYSTEM_AGENT_ID -> Full global obs + agent-specific local obs (id-based)
        otherwise -> partial global obs + agent-specific local obs (id-based)

        Returns:
            Observation object that can be automatically converted to np.ndarray via __array__()
        """
        from heron.agents.system_agent import SYSTEM_AGENT_ID

        # Get global and local components
        global_state = self.get_global_states(sender_id, protocol)
        local_state = self.get_local_state(sender_id, protocol)

        # Return Observation object
        return Observation(
            local=local_state,
            global_info=global_state,
            timestamp=self._timestep
        )

    def get_global_states(self, sender_id: AgentID, protocol: Optional[Protocol] = None) -> Dict:
        """Get global state information visible to the requesting agent with feature-level visibility filtering.

        NEW: Applies feature-level visibility using state.observed_by()

        Args:
            sender_id: ID of agent requesting global state
            protocol: Optional protocol for formatting

        Returns:
            Dict containing global state information (filtered by visibility rules)
        """
        # Apply feature-level visibility filtering to all agents' states
        global_filtered = {}
        requestor_level = self._agent_levels.get(sender_id, 1)

        for agent_id, state_obj in self.state_cache.get("agents", {}).items():
            if agent_id == sender_id:
                continue  # Don't include own state in global (it's in local)

            # Apply visibility filtering
            observable = state_obj.observed_by(sender_id, requestor_level)
            if observable:  # Only include if agent can see something
                global_filtered[agent_id] = observable

        return global_filtered

    def get_local_state(self, sender_id: AgentID, protocol: Optional[Protocol] = None) -> Dict:
        """Get local state information for the requesting agent with visibility filtering.

        NEW: Applies feature-level visibility filtering via state.observed_by()

        Args:
            sender_id: ID of agent requesting local state
            protocol: Optional protocol for formatting

        Returns:
            Dict containing agent-specific local state (filtered by visibility rules)
        """
        agents_cache = self.state_cache.get("agents", {})
        state_obj = agents_cache.get(sender_id)

        if state_obj is None:
            return {}

        # Apply visibility filtering using observed_by()
        requestor_level = self._agent_levels.get(sender_id, 1)
        return state_obj.observed_by(sender_id, requestor_level)

    def set_global_state(self, global_state: Dict) -> None:
        """Update the global state in cache.

        Args:
            global_state: Global state dictionary to cache
        """
        if "global" not in self.state_cache:
            self.state_cache["global"] = {}
        self.state_cache["global"].update(global_state)

    def set_local_state(self, agent_id: str, state: "State") -> None:
        """Update local state for agents in cache.

        NEW: Stores State objects directly for visibility filtering.

        Args:
            agent_id: ID of the agent owning this state
            state: State object (FieldAgentState, CoordinatorAgentState, etc.)
        """
        if "agents" not in self.state_cache:
            self.state_cache["agents"] = {}

        # Store State object directly!
        self.state_cache["agents"][agent_id] = state

    def set_step_result(self, obs: Dict[AgentID, Observation], rewards, terminateds, truncateds, infos):
        """Cache the step results from environment execution.

        Args:
            obs: Observations dict mapping agent IDs to Observation objects
            rewards: Rewards dict mapping agent IDs to reward values
            terminateds: Terminated flags dict
            truncateds: Truncated flags dict
            infos: Info dicts mapping agent IDs to info dicts
        """
        self._step_results = {
            "obs": obs,
            "rewards": rewards,
            "terminateds": terminateds,
            "truncateds": truncateds,
            "infos": infos,
        }

    def get_step_results(self) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, bool], Dict[AgentID, Dict]]:
        """Retrieve cached step results.

        Automatically converts Observation objects to np.ndarray for RL algorithms.

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
            - observations: Dict[AgentID, np.ndarray] - vectorized observations for RL
        """
        if not hasattr(self, "_step_results") or self._step_results is None:
            raise RuntimeError("No step results available. Call set_step_result() first.")

        results = self._step_results
        obs: Dict[AgentID, Observation] = results["obs"]

        # Convert all Observation objects to np.ndarray for RL algorithms
        obs_vectorized = {
            agent_id: observation.vector()
            for agent_id, observation in obs.items()
        }

        return (
            obs_vectorized,
            results["rewards"],
            results["terminateds"],
            results["truncateds"],
            results["infos"],
        )
    
    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        num_registered = len(self.registered_agents)
        has_broker = self._message_broker is not None
        return f"ProxyAgent(id={self.agent_id}, registered_agents={num_registered}, broker={has_broker})"



    # # ============================================
    # # Legacy Code (for reference, may not be meaningful in current version)
    # # ============================================
    # def update_state(self, state: Dict[str, Any]) -> None:
    #     """Update cached state from environment.

    #     Should be called by the environment after each step.

    #     Args:
    #         state: Current environment state
    #     """
    #     self.state_cache = state.copy()

    #     # Add to history for delayed observations
    #     self.state_history.append({
    #         'timestamp': self._timestep,
    #         'state': state.copy()
    #     })

    #     # Trim history if exceeds limit
    #     if len(self.state_history) > self.history_length:
    #         self.state_history = self.state_history[-self.history_length:]

    # def get_state_at_time(self, target_time: float) -> Dict[str, Any]:
    #     """Get state from a specific time for delayed observations.

    #     Used in Option B to simulate observation delays.

    #     Args:
    #         target_time: Timestamp to retrieve state for

    #     Returns:
    #         State at or before target_time, or current cache if not available
    #     """
    #     if not self.state_history:
    #         return self.state_cache

    #     # Find the most recent state at or before target_time
    #     for entry in reversed(self.state_history):
    #         if entry['timestamp'] <= target_time:
    #             return entry['state']

    #     # If target_time is before all history, return oldest
    #     return self.state_history[0]['state'] if self.state_history else self.state_cache

    # def _filter_state_by_keys(
    #     self,
    #     agent_id: AgentID,
    #     agent_state: Dict[str, Any],
    # ) -> Dict[str, Any]:
    #     """Filter state based on visibility rules for a specific agent.

    #     Args:
    #         agent_id: Agent requesting the state
    #         agent_state: Agent-specific state to filter

    #     Returns:
    #         Filtered state dict containing only allowed keys
    #     """
    #     if agent_id not in self.visibility_rules:
    #         # No specific rules, return full agent state
    #         return agent_state.copy() if agent_state else {}

    #     allowed_keys = self.visibility_rules[agent_id]
    #     return {key: agent_state[key] for key in allowed_keys if key in agent_state}

    # def get_state_for_agent(
    #     self,
    #     agent_id: AgentID,
    #     requestor_level: int,
    #     owner_id: Optional[AgentID] = None,
    #     owner_level: Optional[int] = None,
    #     state: Optional["State"] = None,
    #     at_time: Optional[float] = None,
    # ) -> Dict[str, Any]:
    #     """Get filtered state respecting visibility rules.

    #     This is the primary method for agents to access state. It integrates
    #     with FeatureProvider.is_observable_by() for fine-grained visibility control.

    #     Args:
    #         agent_id: ID of the agent requesting state
    #         requestor_level: Hierarchy level of requesting agent (1=field, 2=coord, 3=system)
    #         owner_id: ID of the agent whose state is being requested (defaults to agent_id)
    #         owner_level: Hierarchy level of owner (defaults to requestor_level)
    #         state: Optional State object with FeatureProviders for visibility checking
    #         at_time: Optional timestamp for delayed observations (Option B)

    #     Returns:
    #         Filtered state dict based on visibility rules
    #     """
    #     owner_id = owner_id or agent_id
    #     owner_level = owner_level or requestor_level

    #     # Get base state (from history if time specified, otherwise current)
    #     base_state = self.get_state_at_time(at_time) if at_time is not None else self.state_cache

    #     agents_state = base_state.get('agents', {})
    #     agent_state = agents_state.get(owner_id, {})

    #     # Apply key-based visibility rules
    #     filtered = self._filter_state_by_keys(agent_id, agent_state)

    #     # If State object provided, apply FeatureProvider visibility rules
    #     if state is not None:
    #         return {
    #             feature.feature_name: filtered[feature.feature_name]
    #             for feature in state.features
    #             if feature.is_observable_by(agent_id, requestor_level, owner_id, owner_level)
    #             and feature.feature_name in filtered
    #         }

    #     return filtered

    # def get_observable_features(
    #     self,
    #     requestor_id: AgentID,
    #     requestor_level: int,
    #     owner_id: AgentID,
    #     owner_level: int,
    #     state: "State",
    # ) -> List[str]:
    #     """Get list of feature names observable by the requesting agent.

    #     Args:
    #         requestor_id: ID of agent requesting observation
    #         requestor_level: Hierarchy level of requestor
    #         owner_id: ID of agent that owns the state
    #         owner_level: Hierarchy level of owner
    #         state: State object containing FeatureProviders

    #     Returns:
    #         List of feature names the requestor can observe
    #     """
    #     return [
    #         feature.feature_name
    #         for feature in state.features
    #         if feature.is_observable_by(requestor_id, requestor_level, owner_id, owner_level)
    #     ]

    # def set_visibility_rules(
    #     self,
    #     agent_id: AgentID,
    #     allowed_keys: List[str],
    # ) -> None:
    #     """Set visibility rules for an agent.

    #     Args:
    #         agent_id: Agent to set rules for
    #         allowed_keys: List of state keys the agent can access
    #     """
    #     self.visibility_rules[agent_id] = allowed_keys

    # def receive_state_from_environment(self) -> Optional[Dict[str, Any]]:
    #     """Receive state from environment via message broker.

    #     Consumes messages from the result channel and updates the
    #     state cache with the latest state.

    #     Returns:
    #         Latest state payload, or None if no messages or no broker configured
    #     """
    #     if self._message_broker is None:
    #         return None

    #     env_id = self.env_id or "default"
    #     channel = ChannelManager.custom_channel(
    #         self._result_channel_type, env_id, self.agent_id
    #     )

    #     messages = self._message_broker.consume(
    #         channel=channel,
    #         recipient_id=self.agent_id,
    #         env_id=env_id,
    #         clear=True,
    #     )

    #     if not messages:
    #         return None

    #     # Use the latest message
    #     latest_msg = messages[-1]
    #     self.update_state(latest_msg.payload)
    #     return latest_msg.payload

    # def distribute_state_to_agents(self) -> None:
    #     """Distribute cached state to registered agents via message broker.

    #     Sends filtered state to each registered agent based on visibility rules.
    #     Only works if message_broker was provided during initialization.
    #     """
    #     if self._message_broker is None:
    #         return

    #     if not self.state_cache:
    #         return

    #     env_id = self.env_id or "default"
    #     agents_state = self.state_cache.get("agents", {})

    #     for agent_id in self.registered_agents:
    #         # Get agent-specific state from aggregated state
    #         agent_state = agents_state.get(agent_id, {})

    #         # Apply visibility filtering
    #         filtered_state = self._filter_state_by_keys(agent_id, agent_state)

    #         # Send to agent via info channel
    #         channel = ChannelManager.info_channel(self.agent_id, agent_id, env_id)
    #         msg = Message(
    #             env_id=env_id,
    #             sender_id=self.agent_id,
    #             recipient_id=agent_id,
    #             timestamp=self._timestep,
    #             message_type=MessageType.INFO,
    #             payload=filtered_state,
    #         )
    #         self._message_broker.publish(channel, msg)


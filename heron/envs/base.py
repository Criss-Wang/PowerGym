from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import gymnasium as gym

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.core.action import Action
from heron.core.policies import Policy
from heron.messaging import MessageBroker, ChannelManager, Message, MessageType
from heron.utils.typing import AgentID, MultiAgentDict
from heron.scheduling import EventScheduler, DefaultScheduler, Event, EventAnalyzer, EpisodeResult
from heron.agents.system_agent import SystemAgent
from heron.agents.proxy_agent import ProxyAgent
from heron.agents.constants import SYSTEM_AGENT_ID, PROXY_AGENT_ID


class EnvCore:
    def __init__(
        self,
        env_id: Optional[str] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        message_broker_config: Optional[Dict[str, Any]] = None,
        # agents
        system_agent: Optional[SystemAgent] = None,
        coordinator_agents: Optional[List[CoordinatorAgent]] = None,
        # simulation-related params
        simulation_wait_interval: Optional[float] = None,
    ) -> None:
        # environment attributes
        self.env_id = env_id or f"env_{uuid.uuid4().hex[:8]}"
        self.simulation_wait_interval = simulation_wait_interval

        # agent-specific fields
        self.registered_agents: Dict[AgentID, Agent] = {}
        self._register_agents(system_agent, coordinator_agents)

        # initialize proxy agent (singleton) for state access and action dispatch
        self.proxy_agent = ProxyAgent(agent_id=PROXY_AGENT_ID)
        self._register_agent(self.proxy_agent)

        # setup message broker (before proxy attach - proxy needs it for channels)
        self.message_broker = MessageBroker.init(message_broker_config)
        self.message_broker.attach(self.registered_agents)

        # attach message broker to proxy agent for communication
        self.proxy_agent.set_message_broker(self.message_broker)
        # establish direction link between registered agents and proxy for state access
        self.proxy_agent.attach(self.registered_agents)

        # setup scheduler (before initialization - agents need it)
        self.scheduler = EventScheduler.init(scheduler_config)
        self.scheduler.attach(self.registered_agents)

    # ============================================
    # Agent Management Methods
    # ============================================
    def _register_agents(
        self,
        system_agent: Optional[SystemAgent],
        coordinator_agents: Optional[List[CoordinatorAgent]],
    ) -> None:
        """Internal method to register agents during initialization."""
        # register system agent (singleton) & its subordinates
        if system_agent and coordinator_agents:
            raise ValueError("Cannot provide both SystemAgent and List[CoordinatorAgent]. Provide one or the other.")
        self._system_agent = None
        if system_agent:
            self._system_agent = system_agent
        else:
            print("No system agent provided, using default system agent")
            self._system_agent = SystemAgent(
                agent_id=SYSTEM_AGENT_ID,
                subordinates={agent.agent_id: agent for agent in coordinator_agents}
            )
        self._system_agent.set_simulation(
            self.run_simulation,
            self.env_state_to_global_state,
            self.global_state_to_env_state,
            self.simulation_wait_interval
        )
        self._register_agent(self._system_agent)
        

    def get_agent(self, agent_id: AgentID) -> Optional[Agent]:
        """Get a registered agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent instance or None if not found
        """
        return self.registered_agents.get(agent_id)

    def _register_agent(self, agent: Agent) -> None:
        """Register an agent with the environment.

        Args:
            agent: Agent to register
        """
        agent.env_id = self.env_id
        self.registered_agents[agent.agent_id] = agent
        for subordinate in agent.subordinates.values():
            self._register_agent(subordinate)

    # ===========================================
    # Environment Interaction Methods
    # ==========================================
    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset all registered agents.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        # reset scheduler and clear messages before resetting agents to ensure a clean slate
        self.scheduler.reset(seed)
        self.clear_broker_environment()

        # reset agents (system agent will reset subordinates)
        self.proxy_agent.reset(seed=seed)
        obs = self._system_agent.reset(seed=seed, proxy=self.proxy_agent)
        self.proxy_agent.init_global_state()  # Cache initial state in proxy after reset
        return obs
    
    def step(self, actions: Dict[AgentID, Any]) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict],
    ]:
        """Execute one environment step.

        The system_agent in the environment is responsible for entire 
        simulation step

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
            - observations: Dict mapping agent IDs to observation arrays
            - rewards: Dict mapping agent IDs to reward floats
            - terminated: Dict with agent IDs and "__all__" key
            - truncated: Dict with agent IDs and "__all__" key
            - infos: Dict mapping agent IDs to info dicts
        """
        self._system_agent.execute(actions, self.proxy_agent)
        return self.proxy_agent.get_step_results()
    
    def run_event_driven(
        self,
        event_analyzer: EventAnalyzer,
        t_end: float,
        max_events: Optional[int] = None,
    ) -> EpisodeResult:
        """Run event-driven simulation until time limit. 

        Args:
            t_end: Stop when simulation time exceeds this
            max_events: Optional maximum number of events to process

        Returns:
            Number of events processed

        Raises:
            RuntimeError: If scheduler not configured
        """
        result = EpisodeResult()
        for event in self.scheduler.run_until(t_end=t_end, max_events=max_events):
            result.add_event_analysis(event_analyzer.parser_event(event))
        return result

    # ============================================
    # Simulation-related Methods
    # ============================================
    @abstractmethod
    def run_simulation(self, env_state: Any, *args, **kwargs) -> Any:
        """ Custom simulation logic post-system_agent.act and before system_agent.update_from_environment().

        In the long run, this can be eventually turned into a static SimulatorAgent.
        """
        pass

    @abstractmethod
    def env_state_to_global_state(self, env_state: Any) -> Dict[str, Any]:
        """Convert custom environment state to HERON global state dict format.

        This method is called after run_simulation() to convert the updated
        environment state back into the global state dict structure that will
        be stored in proxy.state_cache["global"].

        Args:
            env_state: Custom environment state after simulation

        Returns:
            Dict that will be merged into proxy.state_cache["global"] via .update()
            Typically includes "agent_states" dict with updated agent state dicts.

        Example:
            return {
                "agent_states": {
                    "agent_1": {"FeatureName": {"field": value}},
                    ...
                }
            }
        """
        pass

    @abstractmethod
    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> Any:
        """Convert HERON global state dict to custom environment state format.

        This method is called before run_simulation() to extract relevant info
        from proxy.state_cache["global"] and convert it to the custom format
        your simulation function expects.

        Args:
            global_state: Dict from proxy.state_cache["global"] with structure:
                {
                    "agent_states": {agent_id: state_dict, ...},
                    ... other global fields ...
                }
                Note: state_dict is the dict representation from State.to_dict()

        Returns:
            Custom environment state object for your simulation

        Example:
            agent_states = global_state.get("agent_states", {})
            battery_soc = agent_states["battery_1"]["BatteryFeature"]["soc"]
            return CustomEnvState(battery_soc=battery_soc)
        """
        pass


    # ============================================
    # Utility Methods
    # ============================================
    def get_all_policies(self) -> Dict[AgentID, Policy]:
        return {
            agent_id: agent.policy
            for agent_id, agent in self.registered_agents.items()
            if agent.policy
        }
    
    def set_agent_policies(self, policies: Dict[AgentID, Policy]) -> None:
        for agent_id, policy in policies.items():
            agent = self.registered_agents.get(agent_id)
            if agent:
                agent.policy = policy

    def clear_broker_environment(self) -> None:
        """Clear all messages for this environment from the broker.

        Useful for resetting the environment.
        """
        if self.message_broker is not None:
            self.message_broker.clear_environment(self.env_id)

    def close_core(self) -> None:
        """Clean up core resources. [Both Modes]"""
        if self.message_broker is not None:
            self.message_broker.close()
    

#     # ============================================
#     # Synchronous Execution (Option A - Training)
#     # ============================================

#     def get_observations(
#         self, global_state: Optional[Dict[str, Any]] = None
#     ) -> Dict[AgentID, Observation]:
#         """Collect observations from all agents. [Training Only]

#         Args:
#             global_state: Optional global state to pass to agents

#         Returns:
#             Dictionary mapping agent IDs to observations
#         """
#         observations = {}
#         for agent_id, agent in self._registered_agents.items():
#             observations[agent_id] = agent.observe(global_state, self.proxy_agent)
#         return observations

#     def apply_actions(
#         self,
#         actions: Dict[AgentID, Any],
#         observations: Optional[Dict[AgentID, Observation]] = None,
#     ) -> None:
#         """Apply actions to agents. [Training Only]

#         Args:
#             actions: Dictionary mapping agent IDs to actions
#             observations: Optional observations to pass to agents
#         """
#         for agent_id, action in actions.items():
#             if agent_id in self._registered_agents:
#                 obs = observations.get(agent_id) if observations else None
#                 self._registered_agents[agent_id].act(obs, upstream_action=action)


#     def collect_agent_states(self) -> Dict[AgentID, Dict[str, Any]]:
#         """Collect post-action states from all agents. [Phase 1 - Both Modes]

#         Called after actions are applied to gather agent states (including
#         action-dependent features updated via _update_action_features())
#         before running simulation.

#         If a SystemAgent is configured, delegates to it for hierarchical collection.

#         Returns:
#             Dictionary mapping agent IDs to their current state dicts
#         """
#         if self._system_agent is not None:
#             return self._system_agent.get_state_for_environment()

#         return {
#             agent_id: agent.get_state_for_environment()
#             for agent_id, agent in self._registered_agents.items()
#         }

#     def distribute_environment_results(
#         self,
#         env_results: Dict[AgentID, Dict[str, Any]]
#     ) -> None:
#         """Distribute simulation results back to agents. [Phase 2 - Both Modes]

#         Called after simulation to push environment results to agents,
#         triggering update_from_environment() on each agent.

#         If a SystemAgent is configured, delegates to it for hierarchical distribution.

#         Args:
#             env_results: Dictionary mapping agent IDs to their environment results
#         """
#         if self._system_agent is not None:
#             self._system_agent.update_from_environment(env_results)
#         else:
#             for agent_id, result in env_results.items():
#                 agent = self._registered_agents.get(agent_id)
#                 if agent:
#                     agent.update_from_environment(result)

#     def get_agent_action_spaces(self) -> Dict[AgentID, gym.Space]:
#         """Get action spaces for all agents. [Both Modes]

#         Returns:
#             Dictionary mapping agent IDs to their action spaces
#         """
#         return {
#             agent_id: agent.action_space
#             for agent_id, agent in self._registered_agents.items()
#             if agent.action_space is not None
#         }

#     def get_agent_observation_spaces(self) -> Dict[AgentID, gym.Space]:
#         """Get observation spaces for all agents. [Both Modes]

#         Returns:
#             Dictionary mapping agent IDs to their observation spaces
#         """
#         return {
#             agent_id: agent.observation_space
#             for agent_id, agent in self._registered_agents.items()
#             if agent.observation_space is not None
#         }

#     # ============================================
#     # Event-Driven Execution (Option B - Testing)
#     # ============================================

#     def setup_event_driven(
#         self,
#         scheduler: Optional["EventScheduler"] = None,
#     ) -> "EventScheduler":
#         """Setup event-driven execution with scheduler. [Testing Only]

#         Registers all agents with the scheduler using their timing parameters.
#         Creates a new scheduler if none provided.

#         Args:
#             scheduler: Optional existing scheduler (creates new if None)

#         Returns:
#             The configured EventScheduler
#         """
#         from heron.scheduling import EventScheduler

#         if scheduler is None:
#             scheduler = EventScheduler(start_time=0.0)

#         self.scheduler = scheduler

#         # Register all agents with their timing parameters
#         for agent_id, agent in self._registered_agents.items():
#             tick_config = getattr(agent, '_tick_config', None)
#             scheduler.register_agent(
#                 agent_id=agent_id,
#                 tick_interval=tick_config.tick_interval if tick_config else 1.0,
#                 obs_delay=tick_config.obs_delay if tick_config else 0.0,
#                 act_delay=tick_config.act_delay if tick_config else 0.0,
#             )

#         return scheduler

#     def set_event_handlers(
#         self,
#         on_agent_tick: Optional[Callable[["Event", "EventScheduler"], None]] = None,
#         on_action_effect: Optional[Callable[["Event", "EventScheduler"], None]] = None,
#         on_message_delivery: Optional[Callable[["Event", "EventScheduler"], None]] = None,
#     ) -> None:
#         """Set event handlers for event-driven execution. [Testing Only]

#         Args:
#             on_agent_tick: Handler for AGENT_TICK events
#             on_action_effect: Handler for ACTION_EFFECT events
#             on_message_delivery: Handler for MESSAGE_DELIVERY events
#         """
#         if self.scheduler is None:
#             raise RuntimeError("Call setup_event_driven() first")

#         from heron.scheduling import EventType

#         if on_agent_tick:
#             self.scheduler.set_handler(EventType.AGENT_TICK, on_agent_tick)
#         if on_action_effect:
#             self.scheduler.set_handler(EventType.ACTION_EFFECT, on_action_effect)
#         if on_message_delivery:
#             self.scheduler.set_handler(EventType.MESSAGE_DELIVERY, on_message_delivery)

#     def setup_default_handlers(
#         self,
#         global_state_fn: Optional[Callable[[], Dict[str, Any]]] = None,
#         on_action_effect: Optional[Callable[[AgentID, Any], None]] = None,
#     ) -> None:
#         """Setup default event handlers for event-driven execution. [Testing Only]

#         This convenience method sets up standard handlers that:
#         - AGENT_TICK: Calls agent.tick() with scheduler and current time
#         - ACTION_EFFECT: Calls the provided callback to apply actions
#         - MESSAGE_DELIVERY: Publishes messages via message broker

#         Args:
#             global_state_fn: Optional function returning current global state
#                             for agent.tick(). If None, passes None to tick().
#             on_action_effect: Optional callback(agent_id, action) to apply actions.
#                             Override this to implement domain-specific action application.
#         """
#         if self.scheduler is None:
#             raise RuntimeError("Call setup_event_driven() first")

#         from heron.scheduling import EventType

#         # Create closures that capture self and callbacks
#         def agent_tick_handler(event: "Event", scheduler: "EventScheduler") -> None:
#             agent = self._registered_agents.get(event.agent_id)
#             if agent is not None:
#                 global_state = global_state_fn() if global_state_fn else None
#                 # Pass proxy_agent to enable delayed observations (Option B)
#                 proxy = getattr(self, '_proxy_agent', None)
#                 agent.tick(scheduler, event.timestamp, global_state, proxy)

#         def action_effect_handler(event: "Event", scheduler: "EventScheduler") -> None:
#             agent_id = event.agent_id
#             action = event.payload.get("action")
#             if on_action_effect and action is not None:
#                 on_action_effect(agent_id, action)

#         def message_delivery_handler(event: "Event", scheduler: "EventScheduler") -> None:
#             """Deliver message via message broker."""
#             recipient_id = event.agent_id
#             sender_id = event.payload.get("sender")
#             message_content = event.payload.get("message", {})

#             # Publish message via message broker
#             if self.message_broker is not None and sender_id is not None:
#                 if "action" in message_content:
#                     self.publish_action(
#                         sender_id=sender_id,
#                         recipient_id=recipient_id,
#                         action=message_content.get("action"),
#                     )
#                 else:
#                     self.publish_info(
#                         sender_id=sender_id,
#                         recipient_id=recipient_id,
#                         info=message_content,
#                     )

#         self.scheduler.set_handler(EventType.AGENT_TICK, agent_tick_handler)
#         self.scheduler.set_handler(EventType.ACTION_EFFECT, action_effect_handler)
#         self.scheduler.set_handler(EventType.MESSAGE_DELIVERY, message_delivery_handler)

#     def setup_batched_handlers(
#         self,
#         global_state_fn: Optional[Callable[[], Dict[str, Any]]] = None,
#         on_simulation_step: Optional[Callable[[Dict[AgentID, Dict[str, Any]]], Dict[AgentID, Dict[str, Any]]]] = None,
#     ) -> None:
#         """Setup event handlers for batched event-driven execution. [Testing Only - Option B Batched]

#         Unlike setup_default_handlers() which processes each ACTION_EFFECT individually,
#         this method batches action effects and processes them together at ENV_UPDATE events.

#         This enables the two-phase update flow in event-driven mode:
#         1. AGENT_TICK: Agents tick and update Phase 1 features via _update_action_features()
#         2. ACTION_EFFECT: Actions are accumulated (not processed immediately)
#         3. ENV_UPDATE: Batch simulation runs with all pending actions:
#            a. collect_agent_states() gathers states with Phase 1 features
#            b. on_simulation_step() runs global simulation
#            c. distribute_environment_results() updates Phase 2 features

#         Args:
#             global_state_fn: Optional function returning current global state for agent.tick()
#             on_simulation_step: Callback(agent_states) -> env_results that runs the simulation.
#                 Receives dict of agent states (with Phase 1 features), returns dict of results.

#         Example:
#             def run_simulation(agent_states):
#                 # Run physics with all agent states
#                 results = physics_engine.step(agent_states)
#                 return {aid: results[aid] for aid in agent_states}

#             env.setup_batched_handlers(
#                 global_state_fn=lambda: env.get_state(),
#                 on_simulation_step=run_simulation,
#             )
#             env.schedule_simulation_steps(interval=1.0, t_end=100.0)
#             env.run_event_driven(t_end=100.0)
#         """
#         if self.scheduler is None:
#             raise RuntimeError("Call setup_event_driven() first")

#         from heron.scheduling import EventType

#         # Clear pending actions
#         self._pending_actions = {}

#         def agent_tick_handler(event: "Event", scheduler: "EventScheduler") -> None:
#             agent = self._registered_agents.get(event.agent_id)
#             if agent is not None:
#                 global_state = global_state_fn() if global_state_fn else None
#                 proxy = getattr(self, '_proxy_agent', None)
#                 agent.tick(scheduler, event.timestamp, global_state, proxy)

#         def action_effect_handler(event: "Event", scheduler: "EventScheduler") -> None:
#             """Accumulate actions instead of processing immediately."""
#             agent_id = event.agent_id
#             action = event.payload.get("action")
#             if action is not None:
#                 self._pending_actions[agent_id] = action

#         def env_update_handler(event: "Event", scheduler: "EventScheduler") -> None:
#             """Process all pending actions in batch."""
#             if not self._pending_actions and on_simulation_step is None:
#                 return

#             # Phase 1 complete: Collect states (includes action-updated features)
#             agent_states = self.collect_agent_states()

#             # Run global simulation
#             if on_simulation_step is not None:
#                 env_results = on_simulation_step(agent_states)

#                 # Phase 2: Distribute results
#                 if env_results:
#                     self.distribute_environment_results(env_results)

#             # Clear pending actions
#             self._pending_actions = {}

#         def message_delivery_handler(event: "Event", scheduler: "EventScheduler") -> None:
#             """Deliver message via message broker."""
#             recipient_id = event.agent_id
#             sender_id = event.payload.get("sender")
#             message_content = event.payload.get("message", {})

#             if self.message_broker is not None and sender_id is not None:
#                 if "action" in message_content:
#                     self.publish_action(
#                         sender_id=sender_id,
#                         recipient_id=recipient_id,
#                         action=message_content.get("action"),
#                     )
#                 else:
#                     self.publish_info(
#                         sender_id=sender_id,
#                         recipient_id=recipient_id,
#                         info=message_content,
#                     )

#         self.scheduler.set_handler(EventType.AGENT_TICK, agent_tick_handler)
#         self.scheduler.set_handler(EventType.ACTION_EFFECT, action_effect_handler)
#         self.scheduler.set_handler(EventType.ENV_UPDATE, env_update_handler)
#         self.scheduler.set_handler(EventType.MESSAGE_DELIVERY, message_delivery_handler)

#     def schedule_simulation_steps(
#         self,
#         interval: float,
#         t_end: float,
#         t_start: float = 0.0,
#         priority: int = 10,
#     ) -> None:
#         """Schedule periodic ENV_UPDATE events for batched simulation. [Testing Only]

#         Call this after setup_batched_handlers() to schedule when batch simulations run.

#         Args:
#             interval: Time between simulation steps
#             t_end: Stop scheduling after this time
#             t_start: Start time for first simulation step (default: 0.0)
#             priority: Event priority (higher = later at same timestamp, default: 10)
#                       Use higher priority than AGENT_TICK to ensure all agents
#                       have ticked before simulation runs.

#         Example:
#             # Simulate every 1.0 time units, agents tick but simulation batched
#             env.schedule_simulation_steps(interval=1.0, t_end=100.0)
#         """
#         if self.scheduler is None:
#             raise RuntimeError("Call setup_event_driven() first")

#         from heron.scheduling import Event, EventType

#         t = t_start + interval  # First simulation step after agents have ticked
#         while t <= t_end:
#             self.scheduler.schedule(Event(
#                 timestamp=t,
#                 priority=priority,
#                 event_type=EventType.ENV_UPDATE,
#                 agent_id=None,
#                 payload={"type": "simulation_step"},
#             ))
#             t += interval


#     @property
#     def simulation_time(self) -> float:
#         """Current simulation time (from scheduler or timestep). [Both Modes]"""
#         if self.scheduler:
#             return self.scheduler.current_time
#         return float(self._timestep)

#     # ============================================
#     # Distributed Mode (Message Broker)
#     # ============================================

#     def setup_broker_channels(self) -> None:
#         """Setup message broker channels for all registered agents. [Distributed Mode]

#         Creates action and info channels for each agent based on their hierarchy.
#         Should be called after all agents are registered.

#         Raises:
#             RuntimeError: If message broker is not configured
#         """
#         if self.message_broker is None:
#             raise RuntimeError("Message broker not configured.")

#         for agent_id, agent in self._registered_agents.items():
#             # Get agent's hierarchy info
#             upstream_id = getattr(agent, 'upstream_id', None)
#             subordinate_ids = list(getattr(agent, 'subordinates', {}).keys())

#             # Get channels for this agent
#             channels = ChannelManager.agent_channels(
#                 agent_id=agent_id,
#                 upstream_id=upstream_id,
#                 subordinate_ids=subordinate_ids,
#                 env_id=self.env_id,
#             )

#             # Create all channels
#             for channel in channels['subscribe'] + channels['publish']:
#                 self.message_broker.create_channel(channel)

#     def publish_action(
#         self,
#         sender_id: AgentID,
#         recipient_id: AgentID,
#         action: Any,
#     ) -> None:
#         """Publish an action from sender to recipient via message broker. [Distributed Mode]

#         Args:
#             sender_id: ID of the agent sending the action
#             recipient_id: ID of the agent receiving the action
#             action: Action data to send

#         Raises:
#             RuntimeError: If message broker is not configured
#         """
#         if self.message_broker is None:
#             raise RuntimeError("Message broker not configured.")

#         channel = ChannelManager.action_channel(sender_id, recipient_id, self.env_id)
#         msg = Message(
#             env_id=self.env_id,
#             sender_id=sender_id,
#             recipient_id=recipient_id,
#             timestamp=float(self._timestep),
#             message_type=MessageType.ACTION,
#             payload={"action": action},
#         )
#         self.message_broker.publish(channel, msg)

#     def publish_info(
#         self,
#         sender_id: AgentID,
#         recipient_id: AgentID,
#         info: Dict[str, Any],
#     ) -> None:
#         """Publish info from sender to recipient via message broker. [Distributed Mode]

#         Args:
#             sender_id: ID of the agent sending the info
#             recipient_id: ID of the agent receiving the info
#             info: Information data to send

#         Raises:
#             RuntimeError: If message broker is not configured
#         """
#         if self.message_broker is None:
#             raise RuntimeError("Message broker not configured.")

#         channel = ChannelManager.info_channel(sender_id, recipient_id, self.env_id)
#         msg = Message(
#             env_id=self.env_id,
#             sender_id=sender_id,
#             recipient_id=recipient_id,
#             timestamp=float(self._timestep),
#             message_type=MessageType.INFO,
#             payload=info,
#         )
#         self.message_broker.publish(channel, msg)



#     # ============================================
#     # SystemAgent Integration (Both Modes)
#     # ============================================

#     def set_system_agent(self, system_agent: "SystemAgent") -> None:
#         """Set the SystemAgent for this environment. [Both Modes]

#         The SystemAgent serves as the interface between the environment and
#         the agent hierarchy. When set, the environment can use:
#         - system_agent.update_from_environment() to push state
#         - system_agent.get_state_for_environment() to get actions

#         Args:
#             system_agent: SystemAgent instance to manage agent hierarchy
#         """
#         self._system_agent = system_agent

#         # Register all coordinators from the system agent
#         for coordinator in system_agent.coordinators.values():
#             self._register_agent(coordinator)

#         # Configure message broker for system agent
#         if self.message_broker:
#             system_agent.set_message_broker(self.message_broker)
#             system_agent.env_id = self.env_id

#     @property
#     def system_agent(self) -> Optional["SystemAgent"]:
#         """Get the SystemAgent for this environment. [Both Modes]"""
#         return getattr(self, '_system_agent', None)

#     def set_proxy_agent(self, proxy_agent: "ProxyAgent") -> None:
#         """Set the ProxyAgent for state distribution. [Both Modes]

#         The ProxyAgent manages state distribution to agents with visibility
#         filtering. When set, agents can request state through the proxy
#         instead of accessing the environment directly.

#         Args:
#             proxy_agent: ProxyAgent instance for state distribution
#         """
#         self._proxy_agent = proxy_agent
#         self._register_agent(proxy_agent)

#     @property
#     def proxy_agent(self) -> Optional["ProxyAgent"]:
#         """Get the ProxyAgent for this environment. [Both Modes]"""
#         if self._proxy_agent is None:
#             raise RuntimeError("No proxy agent configured. Call set_proxy_agent() first.")
#         return self._proxy_agent

#     def update_proxy_state(self, state: Dict[str, Any]) -> None:
#         """Update the ProxyAgent's cached state. [Both Modes]

#         Convenience method to update the proxy agent's state cache.
#         Should be called after physics/simulation updates.

#         Args:
#             state: Current environment state to cache

#         Raises:
#             RuntimeError: If no proxy agent is configured
#         """
#         if self._proxy_agent is None:
#             raise RuntimeError("No proxy agent configured. Call set_proxy_agent() first.")
#         self._proxy_agent.update_state(state)

#     def step(
#         self,
#         actions: Dict[AgentID, Any],
#     ) -> Tuple[
#         Dict[AgentID, Any],
#         Dict[AgentID, float],
#         Dict[str, bool],
#         Dict[str, bool],
#         Dict[AgentID, Dict],
#     ]:
#         """Execute one environment step. [Training - Option A]

#         Base implementation template:
#         1. Applies actions to agents
#         2. Collects observations from agents
#         3. Returns default rewards/terminated/truncated/infos

#         Subclasses should override this to add physics simulation, reward
#         computation, and termination conditions.

#         Args:
#             actions: Dictionary mapping agent IDs to actions

#         Returns:
#             Tuple of (observations, rewards, terminated, truncated, infos)
#             - observations: Dict mapping agent IDs to observation arrays
#             - rewards: Dict mapping agent IDs to reward floats
#             - terminated: Dict with agent IDs and "__all__" key
#             - truncated: Dict with agent IDs and "__all__" key
#             - infos: Dict mapping agent IDs to info dicts
#         """
#         pass

#     def compute_reward(self) -> Dict[AgentID, float]:
#         """Compute rewards for all agents. [Both Modes]

#         If a SystemAgent is configured, delegates to its compute_reward()
#         which hierarchically aggregates rewards from coordinators and field agents.

#         Otherwise, directly calls compute_reward() on each registered agent.

#         Returns:
#             Dictionary mapping agent IDs to their reward values

#         Example:
#             # With SystemAgent hierarchy
#             rewards = env.compute_reward()
#             # Returns: {"field_1": 0.5, "field_2": -0.2, ...}

#             # Without SystemAgent (flat agents)
#             rewards = env.compute_reward()
#             # Returns: {"agent_1": 0.0, "agent_2": 0.0, ...}
#         """
#         if self._system_agent is not None:
#             return self._system_agent.compute_reward()

#         # Fallback: aggregate from registered agents directly
#         rewards = {}
#         for agent_id, agent in self._registered_agents.items():
#             reward = agent.compute_reward()
#             if isinstance(reward, dict):
#                 rewards.update(reward)
#             elif isinstance(reward, (int, float)):
#                 rewards[agent_id] = reward
#             else:
#                 rewards[agent_id] = 0.0

#         return rewards

#     def step_with_system_agent(
#         self,
#         actions: Dict[AgentID, Any],
#         global_state: Optional[Dict[str, Any]] = None,
#     ) -> None:
#         """Execute step using SystemAgent pattern. [Training - Option A]

#         This convenience method implements the standard CTDE flow:
#         1. SystemAgent observes (aggregates from coordinators)
#         2. SystemAgent acts (distributes actions to coordinators)
#         3. SystemAgent simulates (updates internal state)

#         Note: This method only handles the agent hierarchy observation/action flow.
#         Subclasses should call this from their step() implementation and handle
#         physics/simulation separately.

#         Args:
#             actions: Dictionary mapping coordinator IDs to actions
#             global_state: Optional global state for observation

#         Raises:
#             RuntimeError: If no system agent is configured
#         """
#         if self._system_agent is None:
#             raise RuntimeError("No system agent configured. Call set_system_agent() first.")

#         observation = self._system_agent.observe(global_state, self.proxy_agent)
#         self._system_agent.act(observation, upstream_action=actions)
#         # TODO: Need to gather agent states after act() to update system agent's internal state before simulate()
#         state_updates = self._system_agent.get_state_for_environment()
#         self._system_agent.simulate(state_updates)
#         self._system_agent.update_from_environment()  # Pass env results

#     def run_event_driven_with_system_agent(
#         self,
#         t_end: float,
#         get_global_state: Optional[Callable[[], Dict[str, Any]]] = None,
#         on_action_effect: Optional[Callable[[AgentID, Any], None]] = None,
#         max_events: Optional[int] = None,
#     ) -> int:
#         """Run event-driven simulation with SystemAgent. [Testing - Option B]

#         This convenience method sets up and runs event-driven execution
#         with the SystemAgent hierarchy. It:
#         1. Sets up the scheduler and registers all agents
#         2. Configures default event handlers
#         3. Runs the simulation until t_end

#         Args:
#             t_end: Stop when simulation time exceeds this
#             get_global_state: Optional function returning current global state
#             on_action_effect: Optional callback(agent_id, action) for actions
#             max_events: Optional maximum number of events to process

#         Returns:
#             Number of events processed

#         Raises:
#             RuntimeError: If no system agent is configured
#         """
#         if self._system_agent is None:
#             raise RuntimeError("No system agent configured. Call set_system_agent() first.")

#         # Setup scheduler if not already done
#         if self.scheduler is None:
#             self.setup_event_driven()

#         # Register system agent with scheduler
#         tick_config = getattr(self._system_agent, '_tick_config', None)
#         self.scheduler.register_agent(
#             agent_id=self._system_agent.agent_id,
#             tick_interval=tick_config.tick_interval if tick_config else 1.0,
#             obs_delay=tick_config.obs_delay if tick_config else 0.0,
#             act_delay=tick_config.act_delay if tick_config else 0.0,
#         )

#         # Setup default handlers
#         self.setup_default_handlers(
#             global_state_fn=get_global_state,
#             on_action_effect=on_action_effect,
#         )

#         # Run simulation
#         return self.run_event_driven(t_end=t_end, max_events=max_events)


class MultiAgentEnv(EnvCore):
    def close(self) -> None:
        """Clean up environment resources."""
        self.close_core()

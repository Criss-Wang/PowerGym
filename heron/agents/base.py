

from abc import ABC, abstractmethod
from builtins import float
from typing import Any, Dict, List, Optional, Callable

import gymnasium as gym

from heron.messaging import MessageBroker, ChannelManager, Message as BrokerMessage, MessageType
from heron.utils.typing import AgentID
from heron.scheduling.tick_config import TickConfig, JitterType
from heron.scheduling.scheduler import EventScheduler
from heron.scheduling.event import Event, EventType, EVENT_TYPE_FROM_STRING
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.agents.proxy_agent import ProxyAgent


class Agent(ABC):
    # class-level handler function mapping
    _event_handler_funcs: Dict[EventType, Callable[[Event, "EventScheduler"], None]] = {}
    
    def __init__(
        self,
        agent_id: AgentID,
        level: int = 1,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        subordinates: Optional[Dict[AgentID, "Agent"]] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None
    ):
        self.agent_id = agent_id
        self.level = level
        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = policy
        self.protocol = protocol
        self.state = None

        # Execution state
        self._timestep: float = 0.0

        # Message broker reference (set by environment in distributed mode)
        self._message_broker: Optional[MessageBroker] = None

        # Timing configuration (via TickConfig)
        self._tick_config = tick_config or TickConfig.deterministic()

        # Hierarchy structure (used by coordinators)
        self.env_id = env_id
        self.upstream_id = upstream_id
        self.subordinates = self._build_subordinates(subordinates)

    def _build_subordinates(self, subordinates: Optional[Dict[AgentID, "Agent"]] = None,) -> Dict[AgentID, "Agent"]:
        if not subordinates:
            return {}
        for _, agent in subordinates.items():
            agent.upstream_id = self.agent_id
            agent.env_id = self.env_id
        return subordinates

    @abstractmethod
    def init_state(self) -> None:
        pass
    
    @abstractmethod
    def init_action(self) -> None:
        pass

    @abstractmethod
    def set_state(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def set_action(self, *args, **kwargs) -> None:
        pass

    # ============================================
    # Core Lifecycle Methods (Both Modes)
    # - initialize: set up initial action, states, action/obs spaces (if any) with help from proxy
    # - reset: reseting fields and states, potentially returning current states
    # - execute: Synchronous Execution (for Training phase)
    # - tick: Event-Driven Execution (for Testing phase)
    # ============================================
    def initialize(self, proxy: Optional[ProxyAgent] = None) -> None:
        self.init_action()
        self.init_state()
        proxy.register_agent(self.agent_id)
        for subordinate in self.subordinates.values():
            subordinate.initialize(proxy=proxy)

    def reset(self, *, seed: Optional[int] = None, proxy: Optional[ProxyAgent] = None, **kwargs) -> Any:
        self._timestep = 0.0
        for subordinate in self.subordinates.values():
            subordinate.reset(seed=seed, proxy=proxy, **kwargs)

    @abstractmethod
    def execute(self, actions: Dict[AgentID, Any], proxy: Optional[ProxyAgent] = None) -> None:
        pass

    @abstractmethod
    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        pass

    # ============================================
    # Observation related functions
    # ============================================
    def observe(self, global_state: Optional[Dict[str, Any]] = None, proxy: Optional[ProxyAgent] = None, *args, **kwargs) -> Dict[AgentID, Any]:
        """
        Sample format:
        {
            "aid1": np.ndarray1,
            "aid2": np.ndarray2,
            ...
        }
        """
        if not proxy:
            raise ValueError("System Agent requires a proxy agent to observe states")
        obs = {
            self.agent_id: proxy.get_observation(self.agent_id, self.protocol), # local observation
        }
        for subordinate in self.subordinates.values():
            obs.update(subordinate.observe(proxy))
        return obs

    # ============================================
    # Reward related functions
    # ============================================
    def compute_rewards(self, proxy: ProxyAgent) -> Dict[AgentID, float]:
        if not proxy:
            raise ValueError("System Agent requires a proxy agent to compute rewards")

        # Local Reward computation steps:
        # 1. get local states from proxy
        # 2. collect reward params (e.g. safety, cost)
        # 3. calculate reward
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        local_reward = self.compute_local_reward(local_state)
        rewards = {
            self.agent_id: local_reward,
        }
        for subordinate in self.subordinates.values():
            rewards.update(subordinate.compute_rewards(proxy))
        return rewards
    
    def compute_local_reward(self, local_state: dict) -> float:
        # Default implementation returns 0 reward. Override in subclasses for custom reward logic.
        return 0.0

    # ============================================
    # Info related functions
    # ============================================
    def get_info(self, proxy: ProxyAgent) -> Dict[AgentID, Dict]:
        if not proxy:
            raise ValueError("System Agent requires a proxy agent to get infos")
        # Local info derivation
        # May use proxy to retrieve local states via proxy.get_local_state(self.agent_id)
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        local_info = self.get_local_info(local_state)
        infos = {
            self.agent_id: local_info,
        }
        for subordinate in self.subordinates.values():
            infos.update(subordinate.get_info(proxy))
        return infos
    
    def get_local_info(self, local_state: dict) -> Dict[AgentID, Any]:
        return {}
    
    # ============================================
    # terminateds related functions
    # ============================================
    def get_terminateds(self, proxy: ProxyAgent) -> Dict[AgentID, bool]:
        if not proxy:
            raise ValueError("System Agent requires a proxy agent to derive termination state")
        # May need to use env fields to decide
        # TODO: pass in the env field elegantly
        # e.g. done = (self._t % self.max_episode_steps) == 0
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        terminated = self.is_terminated(local_state)
        terminateds = {
            self.agent_id: terminated,
        }
        for subordinate in self.subordinates.values():
            terminateds.update(subordinate.get_terminateds(proxy))
        return terminateds
    
    def is_terminated(self, local_state: dict) -> bool:
        return False
    
    # ============================================
    # truncateds related functions
    # ============================================
    def get_truncateds(self, proxy: ProxyAgent) -> Dict[AgentID, float]:
        if not proxy:
            raise ValueError("System Agent requires a proxy agent to derive truncated states")
        # May need to use env fields to decide
        # TODO: pass in the env field elegantly
        # e.g. done = (self._t % self.max_episode_steps) == 0
        local_state = proxy.get_local_state(self.agent_id, self.protocol)
        truncated = self.is_truncated(local_state)
        truncateds = {
            self.agent_id: truncated
        }
        for subordinate in self.subordinates.values():
            truncateds.update(subordinate.get_truncateds(proxy))
        return truncateds
    
    def is_truncated(self, local_state: dict) -> bool:
        return False

    # ============================================
    # Action taking related functions
    # ============================================
    def act(self, actions: Dict[AgentID, Any], proxy: Optional[ProxyAgent] = None) -> None:
        actions = self.layer_actions(actions)

        # run self action & store local state updates in proxy
        self.handle_self_action(actions['self'], proxy)
        # run subordinate actions & store local state updates in proxy
        self.handle_subordinate_actions(actions['subordinates'], proxy)

    def layer_actions(self, actions: Dict[AgentID, Any]) -> Dict[AgentID, Any]:
        """
        Format:
        {
            "self": action1,
            "subordinates: {
                "sub1": {
                    "self": subaction1,
                    "subordinates": {
                        ....
                    }
                },
                ....
            }
        }
        """
        return {
            "self": actions.get(self.agent_id),
            "suborindates": {
                subordinate_id: subordinate.layer_actions(actions)
                for subordinate_id, subordinate in self.subordinates.items()
            }
        }

    def handle_self_action(self, action: Any, proxy: Optional[ProxyAgent] = None):
        if action:
            self.set_action(action)
        elif self.policy:
            local_obs = proxy.get_obervation(self.agent_id)
            self.set_action(self.policy.forward(observation=local_obs)) # This is where policy is needed!

        else:
            print(f"No action built for ({self}) becase there's no upstream action and no action policy")

        self.apply_action()
        if not self.state:
            raise ValueError("We cannot find appropriate agent state, double check your state update logic")
        proxy.set_local_state(self.state) # update agent-specific state in proxy for parent/subordinate to retrieve

    def handle_subordinate_actions(self, actions: Dict[AgentID, Any], proxy: Optional[ProxyAgent] = None):
        # Note: Parent Agent doesn't build action for subordinates, i.e. self.policy.forward only produces local action
        # TODO: Support parent-controlled actions
        for subordinate_id, subordinate in self.subordinates.items():
            if subordinate_id in actions:
                subordinate.execute(actions, proxy)
            else:
                print(f"{subordinate} not executed in current execution cycle")

    def apply_action(self):
        """Update self.state in agent based on self.action"""
        pass

    # ============================================
    # Event-Driven Execution via scheduler
    # ============================================
    
    # Event tick related methods. Note: these methods only define the logic of what to do when receiving certain events
    # (e.g. message delivery), but not when to trigger these events. The latter is determined by the scheduler and 
    # the tick configuration (e.g. tick interval, message delay, etc.) that defines the timing of event scheduling.
    @property
    def tick_config(self) -> TickConfig:
        """Get the tick configuration for this agent."""
        return self._tick_config

    @tick_config.setter
    def tick_config(self, config: TickConfig) -> None:
        """Set the tick configuration for this agent."""
        self._tick_config = config


    def enable_jitter(
        self,
        jitter_type: Optional[JitterType] = None,
        jitter_ratio: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """Enable jitter for testing mode.

        Converts current tick_config to use jitter with same base values.

        Args:
            jitter_type: Distribution type for jitter (default: GAUSSIAN)
            jitter_ratio: Jitter magnitude as fraction of base
            seed: Optional RNG seed for reproducibility
        """
        if jitter_type is None:
            jitter_type = JitterType.GAUSSIAN

        self._tick_config = TickConfig.with_jitter(
            tick_interval=self._tick_config.tick_interval,
            obs_delay=self._tick_config.obs_delay,
            act_delay=self._tick_config.act_delay,
            msg_delay=self._tick_config.msg_delay,
            jitter_type=jitter_type,
            jitter_ratio=jitter_ratio,
            seed=seed,
        )

    def disable_jitter(self) -> None:
        """Disable jitter (switch to deterministic mode)."""
        self._tick_config = TickConfig.deterministic(
            tick_interval=self._tick_config.tick_interval,
            obs_delay=self._tick_config.obs_delay,
            act_delay=self._tick_config.act_delay,
            msg_delay=self._tick_config.msg_delay,
        )

    # Action-related functions for event-driven execution (testing mode)
    def compute_action(self, obs: Any, scheduler: EventScheduler):
        # Note: Parent Agent doesn't build action for subordinates
        # TODO: Support parent-controlled actions -> self.send_subordinate_action
        # TODO: Add protocol-based parent control logic
        self.set_action(self.policy.forward(observation=obs))
        scheduler.schedule_action_effect(
            agent_id=self.agent_id,
            delay=self._tick_config.act_delay,
        )
    
    def on_action_effect(self, action: Any) -> Any:
        pass

    # Default handlers for common event types. Can be overridden or extended in subclasses for custom handling logic.
    def handler(cls, type: str):
        """Decorator for registering custom handler"""
        event_type = EVENT_TYPE_FROM_STRING.get(type)
        if not event_type:
            raise KeyError(f"Event type cannot be handler, please select from {EVENT_TYPE_FROM_STRING.keys()}")
        def decorator(func):
            cls._event_handler_funcs[event_type] = func
            return func
        return decorator

    @handler("agent_tick")
    def agent_tick_handler(self, event: Event, scheduler: EventScheduler) -> None:
        self.tick(scheduler, event.timestamp)

    @handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        action = event.payload.get("action")
        if action is not None:
            self.on_action_effect(action)

    @handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        pass

    def get_handlers(self) -> Dict[EventType, Callable[[Event, "EventScheduler"], None]]:
        return self._event_handler_funcs
    
    # ============================================
    # Additional subordinate controls (Protocol Usage)
    # TODO: fill this section up
    # ============================================

    # ============================================
    # Messaging via message broker
    #
    # Key utils:
    # - publish
    # - consume
    # - send_action
    # - send_info
    # - receive_actions
    # - receive_info
    # ============================================
    def set_message_broker(self, broker: MessageBroker) -> None:
        """Set the message broker for this agent. [Both Modes]

        Called by the environment to configure distributed messaging.

        Args:
            broker: MessageBroker instance
        """
        self._message_broker = broker

    @property
    def message_broker(self) -> Optional[MessageBroker]:
        """Get the message broker for this agent. [Both Modes]"""
        return self._message_broker


    def receive_upstream_action(
        self,
        sender_id: Optional[str] = None,
        clear: bool = True,
    ) -> List[Any]:
        """Receive action messages from the message broker.

        Convenience method for receiving action messages from upstream.

        Args:
            sender_id: Optional sender ID (uses upstream_id if not provided)
            clear: If True, remove consumed messages

        Returns:
            List of actions received
        """
        if self._message_broker is None:
            return []

        return self.receive_actions(
            broker=self._message_broker,
            upstream_id=sender_id,
            clear=clear,
        )

    def send_subordinate_action(
        self,
        recipient_id: str,
        action: Any,
    ) -> None:
        """Send an action to a subordinate agent.

        Convenience method for sending actions to subordinates.

        Args:
            recipient_id: ID of the subordinate agent
            action: Action to send
        """
        if self._message_broker is None:
            raise RuntimeError(
                f"Agent {self.agent_id} has no message broker configured."
            )

        self.send_action(
            broker=self._message_broker,
            recipient_id=recipient_id,
            action=action,
        )

    def _publish(
        self,
        broker: MessageBroker,
        channel: str,
        payload: Dict[str, Any],
        recipient_id: str = "broadcast",
        message_type: str = "INFO",
    ) -> None:
        """Publish a message to a channel.

        Args:
            broker: MessageBroker instance
            channel: Channel name to publish to
            payload: Message payload
            recipient_id: Recipient agent ID (default: broadcast)
            message_type: Type of message (ACTION, INFO, BROADCAST, etc.)
        """
        msg = BrokerMessage(
            env_id=self.env_id or "default",
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            timestamp=self._timestep,
            message_type=MessageType[message_type],
            payload=payload,
        )
        broker.publish(channel, msg)

    def _consume(
        self,
        broker: MessageBroker,
        channel: str,
        clear: bool = True,
    ) -> List[BrokerMessage]:
        """Consume messages from a channel.

        Args:
            broker: MessageBroker instance
            channel: Channel name to consume from
            clear: If True, remove consumed messages

        Returns:
            List of messages for this agent
        """
        return broker.consume(
            channel=channel,
            recipient_id=self.agent_id,
            env_id=self.env_id or "default",
            clear=clear,
        )

    def send_action(
        self,
        broker: MessageBroker,
        recipient_id: str,
        action: Any,
    ) -> None:
        """Send an action to a subordinate.

        Args:
            broker: MessageBroker instance
            recipient_id: ID of the recipient agent
            action: Action to send
        """
        channel = ChannelManager.action_channel(
            self.agent_id, recipient_id, self.env_id or "default"
        )
        self._publish(
            broker=broker,
            channel=channel,
            payload={"action": action},
            recipient_id=recipient_id,
            message_type="ACTION",
        )

    def send_info(
        self,
        broker: MessageBroker,
        recipient_id: str,
        info: Dict[str, Any],
    ) -> None:
        """Send info to an upstream agent.

        Args:
            broker: MessageBroker instance
            recipient_id: ID of the recipient agent (typically upstream)
            info: Information payload
        """
        channel = ChannelManager.info_channel(
            self.agent_id, recipient_id, self.env_id or "default"
        )
        self._publish(
            broker=broker,
            channel=channel,
            payload=info,
            recipient_id=recipient_id,
            message_type="INFO",
        )

    def receive_actions(
        self,
        broker: MessageBroker,
        upstream_id: Optional[str] = None,
        clear: bool = True,
    ) -> List[Any]:
        """Receive actions from upstream.

        Args:
            broker: MessageBroker instance
            upstream_id: ID of the upstream agent (uses self.upstream_id if not provided)
            clear: If True, remove consumed messages

        Returns:
            List of actions received
        """
        if upstream_id is None:
            upstream_id = self.upstream_id
        if upstream_id is None:
            return []

        channel = ChannelManager.action_channel(
            upstream_id, self.agent_id, self.env_id or "default"
        )
        messages = self._consume(broker, channel, clear=clear)
        return [msg.payload.get("action") for msg in messages if "action" in msg.payload]

    def receive_info(
        self,
        broker: MessageBroker,
        subordinate_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Receive info from subordinates.

        Args:
            broker: MessageBroker instance
            subordinate_ids: IDs of subordinate agents (uses self.subordinates if not provided)

        Returns:
            Dict mapping subordinate IDs to their info payloads
        """
        if subordinate_ids is None:
            subordinate_ids = list(self.subordinates.keys())

        result = {}
        for sub_id in subordinate_ids:
            channel = ChannelManager.info_channel(
                sub_id, self.agent_id, self.env_id or "default"
            )
            messages = self._consume(broker, channel)
            if messages:
                result[sub_id] = [msg.payload for msg in messages]

        return result
    
    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, level={self.level})"

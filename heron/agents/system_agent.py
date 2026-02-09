from typing import Any, Callable, Dict, Optional, Tuple
from enum import Enum

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.proxy_agent import ProxyAgent, PROXY_AGENT_ID
from heron.core.action import Action
from heron.core.observation import (
    OBS_KEY_COORDINATOR_OBS,
    OBS_KEY_SYSTEM_STATE,
)
from heron.core.state import SystemAgentState
from heron.utils.typing import AgentID, MultiAgentDict
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.scheduler import Event, EventScheduler
from heron.scheduling.tick_config import TickConfig
from gymnasium.spaces import Box, Space


SYSTEM_LEVEL = 3  # Level identifier for system-level agents
SYSTEM_AGENT_ID = "system_agent"


class SystemAgent(Agent):
    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        # hierarchy params
        subordinates: Optional[Dict[AgentID, "Agent"]] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None
    ):
        # Store protocol and policy
        self.protocol = protocol
        self.policy = policy

        super().__init__(
            agent_id=agent_id or SYSTEM_AGENT_ID,
            level=SYSTEM_LEVEL,
            subordinates=subordinates,
            upstream_id=None,  # System agent has no upstream
            env_id=env_id,
            tick_config=tick_config or TickConfig.deterministic(tick_interval=300.0),
            policy=policy,
            protocol=protocol
        )

    def init_state(self) -> None:
        self.state = SystemAgentState(
            owner_id=self.agent_id,
            owner_level=SYSTEM_LEVEL
        )

    def init_action(self) -> None:
        self.action = Action()

    def set_state(self, *args, **kwargs) -> None:
        pass

    def set_action(self, *args, **kwargs) -> None:
        pass

    # ============================================
    # Core Lifecycle Methods Overrides (see heron/agents/base.py for more details)
    # ============================================
    def reset(self, *, seed: Optional[int] = None, proxy: Optional[ProxyAgent] = None, **kwargs) -> Any:
        """Reset system agent and all coordinators. [Both Modes]

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed, proxy=proxy, **kwargs)

        return self.observe(proxy=proxy), {}
    
    def execute(self, actions: Dict[AgentID, Any], proxy: Optional[ProxyAgent] = None) -> None:
        if not proxy:
            raise ValueError("We still require a valid proxy agent so far")
        
        self.act(actions, proxy)
        
        # get latest global state (after the above local state updates are accounted for)
        global_state = proxy.get_global_states(self.agent_id, self.protocol) 
        # run external environment simulation step (upon action -> agent state update)
        updated_global_state = self.simulate(global_state)
        
        # broadcast updated global state via proxy
        proxy.set_global_state(updated_global_state)

        # get current step statistics
        obs = self.observe(proxy=proxy)
        rewards = self.compute_rewards(proxy)
        infos = self.get_info(proxy)
        terminateds = self.get_terminateds(proxy)
        truncateds = self.get_truncateds(proxy)

        # set step results in proxy agent
        proxy.set_step_result(obs, rewards, terminateds, truncateds, infos)


    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        """
        Action phase - equivalent to `self.act`
        - Initiate tick for suborindates
        - Initiate self
        - Schedule for simulation

        Note on entire event-driven flow:
        - Schedule Actions
        - Schedule Simulation (tick handles until this part)
        (following steps are handled by individual event handlers)
        - Actions take effect (some may fail to take effect before next step)
        - Run simulation
        - Update states
        - Compute rewards
        - Schedule next tick
        """
        self._timestep = current_time

        # Schedule subordinate ticks
        for subordinate_id in self.subordinates:
            scheduler.schedule_agent_tick(subordinate_id)
        
        if self.policy:
            # Compute & execute self action
            # Ask proxy_agent for global state to compute local action
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={"get_info": "obs", "protocol": self.protocol},
                delay=self._tick_config.msg_delay,
            )
        else:
            print(f"{self} doesn't act iself, becase there's no action policy")
        
        # schedule simulation
        scheduler.schedule_simulation(self.agent_id, self._simulation_wait_interval)


    # ============================================
    # Custom Handlers for Event-Driven Execution
    # ============================================
    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Deliver message via message broker."""
        recipient_id = event.agent_id
        assert recipient_id == self.agent_id
        message_content = event.payload.get("message", {})

        # 4 cases:
        # a. getting observation -> computing actions
        # b. getting global state -> starting simulation
        # c. getting local state -> compute rewards
        # d. receiving "state set completion" message from proxy -> schedule reward computation
        assert isinstance(message_content, dict)
        if "get_obs_response" in message_content:
            obs = message_content['body']
            self.compute_action(obs, scheduler)
        elif "get_global_state_response" in message_content:
            global_state = message_content['body']
            updated_global_state = self.simulate(global_state)
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={"set_state": "global", "body": updated_global_state},
                delay=self._tick_config.msg_delay,
            )
        elif "get_local_state_response" in message_content:
            local_state = message_content['body']
            tick_result = {
                "reward": self.compute_local_reward(local_state),
                "terminated": self.is_terminated(local_state),
                "truncated": self.is_truncated(local_state),
                "info": self.get_local_info(local_state)
            }

            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={"set_tick_result": "local", "body": tick_result},
                delay=self._tick_config.msg_delay,
            )
        elif "set_state_completion" in message_content:
            if message_content["set_state_completion"] != "success":
                raise ValueError(f"State update failed in proxy, cannot proceed with reward computation")
            
            # Initiate reward computation after state update by retrieving local states from proxy agent

            # Note: Parent agent will help initiate reward computation for subordinates.
            # The alternative options is to let proxy agent directly send reward computation message to subordinates,
            # but we want to keep the logic of "when to compute rewards" in the agent itself for better modularity and 
            # flexibility (e.g. parent agent may choose to delay subordinate reward computation until certain 
            # conditions are met, instead of immediately after state update)
            for subordinate_id in self.subordinates:
                scheduler.schedule_message_delivery(
                    sender_id=subordinate_id,
                    recipient_id=PROXY_AGENT_ID,
                    message={"get_info": "local_state", "protocol": self.protocol},
                    delay=self._tick_config.msg_delay,
                )

            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={"get_info": "local_state", "protocol": self.protocol},
                delay=self._tick_config.msg_delay,
            )
        else:
            raise NotImplementedError

    @Agent.handler("simulation")
    def simulation_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """
        Simulation cycle:
        1. Ask for proxy for global states
        2. Proxy returns the global states via message delivery
        3. Run simulation
        4. Gather updated global state and broadcast via proxy
        """
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={"get_info": "global_state", "protocol": self.protocol},
            delay=self._tick_config.msg_delay,
        )
    

    # ============================================
    # Simulation related functions - SystemAgent-specific
    # ============================================
    def set_simulation(self, simulation_func: Callable, wait_interval: Optional[float] = None):
        """
        simulation_func: simulation function passed from environment
        wait_interval: waiting time between action kick-off and simulation starts
        """
        self._simulation_func = simulation_func
        self._simulation_wait_interval = wait_interval or self.tick_config.tick_interval

    def simulate(self, global_state: Dict[AgentID, Any]) -> Any:
        return self._simulation_func(global_state)


    # ============================================
    # Utility Methods
    # ============================================
    @property
    def coordinators(self) -> Dict[AgentID, CoordinatorAgent]:
        """Alias for subordinates - more descriptive for SystemAgent context."""
        return self.subordinates

    @coordinators.setter
    def coordinators(self, value: Dict[AgentID, CoordinatorAgent]) -> None:
        """Set coordinators (subordinates)."""
        self.subordinates = value

    def __repr__(self) -> str:
        num_coords = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"SystemAgent(id={self.agent_id}, coordinators={num_coords}, protocol={protocol_name})"

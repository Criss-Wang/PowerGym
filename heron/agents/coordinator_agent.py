

from typing import Any, Dict, List, Optional

from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.agents.proxy_agent import ProxyAgent, PROXY_AGENT_ID
from heron.core.action import Action
from heron.core.observation import (
    Observation,
    OBS_KEY_SUBORDINATE_OBS,
    OBS_KEY_COORDINATOR_STATE,
)
from heron.core.state import CoordinatorAgentState
from heron.core.policies import Policy
from heron.utils.typing import AgentID
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig
from heron.scheduling.scheduler import EventScheduler, Event


COORDINATOR_LEVEL = 2  # Level identifier for coordinator-level agents


class CoordinatorAgent(Agent):
    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        upstream_id: Optional[AgentID] = None,
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

        self.protocol = protocol
        self.policy = policy
       
        super().__init__(
            agent_id=agent_id,
            level=COORDINATOR_LEVEL,
            subordinates=subordinates,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config or TickConfig.deterministic(tick_interval=60.0),
        )

    def init_state(self) -> None:
        self.state = CoordinatorAgentState(
            owner_id=self.agent_id,
            owner_level=COORDINATOR_LEVEL
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
    def execute(self, actions: Dict[AgentID, Any], proxy: Optional[ProxyAgent] = None) -> None:
        self.act(actions, proxy)

    def tick(
        self,
        scheduler: EventScheduler,
        current_time: float,
    ) -> None:
        """
        Action phase - equivalent to `self.act`

        Note: CoordinatorAgent ticks only upon SystemAgent.tick (see heron/agents/system_agent.py)

        Currently, we assume NO upstream action passed down, coodinator computes its own action
        If we receive upstream actions in the future, do it via self.receive_upstream_action
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

    # ============================================
    # Custom Handlers for Event-Driven Execution
    # ============================================
    @Agent.handler("message_delivery")
    def message_delivery_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """Deliver message via message broker."""
        recipient_id = event.agent_id
        assert recipient_id == self.agent_id
        message_content = event.payload.get("message", {})

        # Publish message via broker
        if "get_obs_response" in message_content:
            assert isinstance(message_content, dict)
            obs = message_content['body']
            self.compute_action(obs, scheduler)
        elif "get_local_state_response" in message_content:
            local_state = message_content['body']
            local_reward = self.compute_local_reward(local_state)
            scheduler.schedule_message_delivery(
                sender_id=self.agent_id,
                recipient_id=PROXY_AGENT_ID,
                message={"set_reward": "local", "body": local_reward},
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
    

    # ============================================
    # Convenience Property
    # ============================================
    @property
    def field_agents(self) -> Dict[AgentID, FieldAgent]:
        """Alias for subordinates - more descriptive for CoordinatorAgent context."""
        return self.subordinates

    @field_agents.setter
    def field_agents(self, value: Dict[AgentID, FieldAgent]) -> None:
        """Set field_agents (subordinates)."""
        self.subordinates = value

    def __repr__(self) -> str:
        num_fields = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__ if self.protocol else "None"
        return f"CoordinatorAgent(id={self.agent_id}, field_agents={num_fields}, protocol={protocol_name})"

from typing import Any, Dict as DictType, List, Optional

from heron.agents.base import Agent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.feature import FeatureProvider
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig
from heron.utils.typing import AgentID

class GridAgent(CoordinatorAgent):
    def __init__(
        self,
        agent_id: Optional[AgentID] = None,
        features: List[FeatureProvider] = [],
        # hierarchy params
        upstream_id: Optional[AgentID] = None,
        subordinates: Optional[DictType[AgentID, "Agent"]] = None,
        env_id: Optional[str] = None,
        # timing config (for event-driven scheduling)
        tick_config: Optional[TickConfig] = None,
        # execution params
        policy: Optional[Policy] = None,
        # coordination params
        protocol: Optional[Protocol] = None
    ):

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            subordinates=subordinates,
            env_id=env_id,
            tick_config=tick_config,
            policy=policy,
            protocol=protocol,
        )


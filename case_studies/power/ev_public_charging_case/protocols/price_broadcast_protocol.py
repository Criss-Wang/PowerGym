"""Price broadcast protocol - distributes coordinator price to all subordinates."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# HERON imports
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.core.observation import Observation
from heron.core.feature import FeatureProvider
from heron.core.action import Action
from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.envs.base import MultiAgentEnv
from heron.protocols.base import ActionProtocol, Protocol
from heron.protocols.vertical import VerticalProtocol
from heron.scheduling import EventScheduler, TickConfig, JitterType
from heron.scheduling.analysis import EventAnalyzer
from heron.utils.typing import AgentID

class PriceBroadcastActionProtocol(ActionProtocol):
    """Broadcast a single price action to all subordinates."""

    def compute_action_coordination(
            self,
            coordinator_action: Optional[Any],
            info_for_subordinates: Optional[Dict[AgentID, Any]] = None,
            coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        """Broadcast the same price to all subordinates.

        Args:
            coordinator_action: np.array([price]) from coordinator policy
            info_for_subordinates: Dict mapping subordinate_id -> info dict

        Returns:
            Dict mapping subordinate_id -> action (same price for all)
        """
        sub_ids = list(info_for_subordinates.keys()) if info_for_subordinates else []

        if coordinator_action is None or not sub_ids:
            return {sub_id: None for sub_id in sub_ids}

        # Ensure action is a numpy array
        if isinstance(coordinator_action, np.ndarray):
            price_action = coordinator_action
        else:
            price_action = np.array([float(coordinator_action)])

        # Broadcast same action to all subordinates
        actions = {sub_id: price_action for sub_id in sub_ids}

        return actions


class PriceProtocol(Protocol):
    """Complete price protocol with no inter-agent communication."""

    def __init__(self):
        from heron.protocols.base import NoCommunication
        super().__init__(
            communication_protocol=NoCommunication(),
            action_protocol=PriceBroadcastActionProtocol()
        )

    def coordinate(self, coordinator_state, coordinator_action=None, info_for_subordinates=None, context=None):
        """Override to add debug output."""
        print(f"[ProportionalProtocol.coordinate] Called with action={coordinator_action}, subordinates={list(info_for_subordinates.keys()) if info_for_subordinates else []}")
        return super().coordinate(coordinator_state, coordinator_action, info_for_subordinates, context)


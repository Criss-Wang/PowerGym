"""Base protocol classes for coordination.

This module defines the core protocol abstractions:
- CommunicationProtocol: Defines WHAT to communicate
- ActionProtocol: Defines HOW to coordinate actions
- Protocol: Combines communication and action coordination
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from heron.utils.typing import AgentID


class CommunicationProtocol(ABC):
    """Handles message-passing coordination between agents.

    Communication protocols define WHAT information is shared and HOW
    agents exchange coordination signals (prices, bids, consensus values, etc.).
    This is the "coordination signal computation" layer.
    """
    neighbors: Set["Agent"]  # Neighboring agents that are reachable

    @abstractmethod
    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_states: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Compute coordination messages to send to receivers.

        Pure function - computes what messages should be sent based on
        current state, without side effects.

        Args:
            sender_state: State of the coordinating agent
            receiver_states: States of agents receiving coordination signals
            context: Additional context (coordinator_action, topology, etc.)

        Returns:
            Dict mapping receiver_id -> message content
        """
        pass

    def add_neighbor(self, agent: "Agent") -> None:
        """Add a neighbor agent."""
        self.neighbors.add(agent)

    def init_neighbors(self, neighbors: List["Agent"]) -> None:
        """Initialize neighbor set."""
        self.neighbors = set(neighbors)


class ActionProtocol(ABC):
    """Handles action coordination between agents.

    Action protocols define HOW actions are coordinated - whether through
    direct control (setpoints), indirect incentives (prices), or consensus.
    This is the "action execution" layer.
    """

    @abstractmethod
    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[AgentID, Any],
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None
    ) -> Dict[AgentID, Any]:
        """Compute coordinated actions for subordinates.

        Pure function - decomposes coordinator action or computes subordinate
        actions based on coordination strategy.

        Args:
            coordinator_action: Action computed by coordinator policy (if any)
            subordinate_states: Current states of subordinate agents
            coordination_messages: Messages computed by communication protocol

        Returns:
            Dict mapping subordinate_id -> action (or None for decentralized)
        """
        pass


class Protocol(ABC):
    """Base protocol combining communication and action coordination.

    A protocol consists of:
    1. CommunicationProtocol: Defines coordination signals/messages
    2. ActionProtocol: Defines action coordination strategy

    Protocols can be:
    - Vertical (agent-owned): Parent coordinates subordinates
    - Horizontal (env-owned): Peers coordinate with each other
    """

    def __init__(
        self,
        communication_protocol: Optional[CommunicationProtocol] = None,
        action_protocol: Optional[ActionProtocol] = None
    ):
        self.communication_protocol = communication_protocol or NoCommunication()
        self.action_protocol = action_protocol or NoActionCoordination()

    def no_op(self) -> bool:
        """Check if this is a no-operation protocol."""
        return (
            isinstance(self.communication_protocol, NoCommunication) and
            isinstance(self.action_protocol, NoActionCoordination)
        )

    def coordinate(
        self,
        coordinator_state: Any,
        subordinate_states: Dict[AgentID, Any],
        coordinator_action: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[AgentID, Dict[str, Any]], Dict[AgentID, Any]]:
        """Execute full coordination cycle.

        This is the main entry point that orchestrates:
        1. Communication: Compute and deliver messages
        2. Action: Compute and apply coordinated actions

        Args:
            coordinator_state: State of coordinating agent
            subordinate_states: States of subordinate agents
            coordinator_action: Action from coordinator policy (if any)
            context: Additional context (subordinates dict, timestamp, etc.)

        Returns:
            Tuple of (messages, actions)
        """
        context = context or {}

        # Enrich context with coordinator_action
        context_with_action = {**context, "coordinator_action": coordinator_action}

        # Step 1: Communication coordination
        messages = self.communication_protocol.compute_coordination_messages(
            sender_state=coordinator_state,
            receiver_states=subordinate_states,
            context=context_with_action
        )

        # Step 2: Action coordination
        actions = self.action_protocol.compute_action_coordination(
            coordinator_action=coordinator_action,
            subordinate_states=subordinate_states,
            coordination_messages=messages
        )
        return messages, actions


# =============================================================================
# NO-OP PROTOCOL COMPONENTS
# =============================================================================

class NoCommunication(CommunicationProtocol):
    """No message passing."""

    def __init__(self):
        self.neighbors = set()

    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_states: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        return {r_id: {} for r_id in receiver_states}


class NoActionCoordination(ActionProtocol):
    """No action coordination."""

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[AgentID, Any],
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None
    ) -> Dict[AgentID, Any]:
        return {sub_id: None for sub_id in subordinate_states}


class NoProtocol(Protocol):
    """No coordination protocol."""

    def __init__(self):
        super().__init__(
            communication_protocol=NoCommunication(),
            action_protocol=NoActionCoordination()
        )

"""Vertical protocols for hierarchical coordination.

Vertical protocols handle Parent -> Subordinate coordination.
Each agent owns its own vertical protocol to coordinate its subordinates.

Includes:
- SetpointProtocol: Centralized control via direct action assignment
- PriceSignalProtocol: Decentralized coordination via price signals
"""

from typing import Any, Dict, Optional

import numpy as np

from heron.protocols.base import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
    NoCommunication,
    NoActionCoordination,
)
from heron.utils.typing import AgentID


class VerticalProtocol(Protocol):
    """Vertical coordination protocol for hierarchical control.

    Each agent owns its own vertical protocol to coordinate its subordinates.
    This is decentralized - each agent independently manages its children.

    Example:
        CoordinatorAgent owns a PriceSignalProtocol to coordinate its FieldAgents.
    """
    pass


# =============================================================================
# CENTRALIZED VERTICAL PROTOCOLS (Direct action control)
# =============================================================================

class SetpointCommunicationProtocol(CommunicationProtocol):
    """Sends setpoint assignments as informational messages."""

    def __init__(self):
        self.neighbors = set()

    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_states: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Compute setpoint messages."""
        context = context or {}
        coordinator_action = context.get("coordinator_action")
        subordinates = context.get("subordinates", {})

        if coordinator_action is None:
            return {r_id: {} for r_id in receiver_states}

        # Decompose action into per-subordinate setpoints
        setpoints = self._decompose_action(coordinator_action, subordinates)

        return {
            receiver_id: {
                "type": "setpoint_command",
                "setpoint": setpoints.get(receiver_id)
            }
            for receiver_id in receiver_states
            if receiver_id in setpoints
        }

    def _decompose_action(
        self,
        action: Any,
        subordinates: Dict[AgentID, "Agent"]
    ) -> Dict[AgentID, Any]:
        """Split action vector into per-subordinate setpoints."""
        if isinstance(action, dict):
            return action  # Already per-subordinate

        # Split numpy array based on subordinate action sizes
        action = np.asarray(action)
        setpoints = {}
        offset = 0

        for sub_id, subordinate in subordinates.items():
            if not hasattr(subordinate, 'action'):
                continue
            action_size = subordinate.action.dim_c + subordinate.action.dim_d
            setpoints[sub_id] = action[offset:offset + action_size]
            offset += action_size

        return setpoints


class CentralizedActionProtocol(ActionProtocol):
    """Direct action control - coordinator sets subordinate actions."""

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[AgentID, Any],
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None
    ) -> Dict[AgentID, Any]:
        """Decompose coordinator action into subordinate actions."""
        # If coordinator_action is a dict, use it directly
        if isinstance(coordinator_action, dict):
            return coordinator_action  # Already per-subordinate

        # For array actions or None, extract from coordination messages (setpoints)
        if coordination_messages:
            actions = {}
            for sub_id, msg in coordination_messages.items():
                if "setpoint" in msg:
                    actions[sub_id] = msg["setpoint"]
                else:
                    actions[sub_id] = None
            return actions

        # No action available
        return {sub_id: None for sub_id in subordinate_states}


class SetpointProtocol(VerticalProtocol):
    """Setpoint-based coordination - parent assigns direct setpoints.

    Communication: Send setpoint assignments (informational)
    Action: Centralized (coordinator directly controls subordinates)

    This protocol works in both centralized and distributed modes:
    - Centralized: Direct action application via subordinate.act()
    - Distributed: Actions sent via message broker

    Example:
        Use with a coordinator for centralized control::

            from heron.agents import CoordinatorAgent
            from heron.protocols import SetpointProtocol

            # Coordinator with setpoint-based control
            coordinator = CoordinatorAgent(
                agent_id="grid_operator",
                protocol=SetpointProtocol()
            )

            # Coordinator computes joint action and assigns to subordinates
            joint_action = np.array([0.5, 0.3, 0.2])  # Power setpoints
            coordinator.act(obs, upstream_action=joint_action)
            # Each subordinate receives its portion of the joint action
    """

    def __init__(self):
        super().__init__(
            communication_protocol=SetpointCommunicationProtocol(),
            action_protocol=CentralizedActionProtocol()
        )


# =============================================================================
# DECENTRALIZED VERTICAL PROTOCOLS (Indirect coordination via signals)
# =============================================================================

class PriceCommunicationProtocol(CommunicationProtocol):
    """Broadcasts price signals to subordinates."""

    def __init__(self, initial_price: float = 50.0):
        self.price = initial_price
        self.neighbors = set()

    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_states: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Compute price messages to broadcast."""
        context = context or {}

        # Extract price from coordinator action if available
        coordinator_action = context.get("coordinator_action")
        if coordinator_action is not None:
            if isinstance(coordinator_action, dict):
                self.price = coordinator_action.get("price", self.price)
            else:
                try:
                    self.price = float(coordinator_action)
                except (TypeError, ValueError):
                    pass  # Keep current price

        # Broadcast same price to all subordinates
        return {
            receiver_id: {
                "type": "price_signal",
                "price": self.price
            }
            for receiver_id in receiver_states
        }


class DecentralizedActionProtocol(ActionProtocol):
    """No direct action control - subordinates act independently based on messages."""

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[AgentID, Any],
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None
    ) -> Dict[AgentID, Any]:
        """Return empty actions - subordinates decide independently."""
        return {sub_id: None for sub_id in subordinate_states}


class PriceSignalProtocol(VerticalProtocol):
    """Price-based coordination via marginal price signals.

    Communication: Broadcast price signals
    Action: Decentralized (subordinates respond to prices independently)

    Attributes:
        price: Current price signal value

    Example:
        Use for decentralized coordination via price signals::

            from heron.protocols import PriceSignalProtocol

            # Initialize with starting price
            protocol = PriceSignalProtocol(initial_price=50.0)

            # Update price based on supply/demand
            protocol.price = 75.0  # High demand -> higher price

            # Subordinates receive price and decide independently
            # e.g., generators increase output, storage discharges
    """

    def __init__(self, initial_price: float = 50.0):
        """Initialize price signal protocol.

        Args:
            initial_price: Initial price value
        """
        super().__init__(
            communication_protocol=PriceCommunicationProtocol(initial_price),
            action_protocol=DecentralizedActionProtocol()
        )

    @property
    def price(self):
        """Get current price."""
        return self.communication_protocol.price

    @price.setter
    def price(self, value):
        """Set current price."""
        self.communication_protocol.price = value

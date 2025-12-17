"""Coordination protocols for hierarchical multi-agent control.

This module defines vertical and horizontal coordination protocols:
- Vertical protocols: Parent → subordinate coordination (agent-owned)
- Horizontal protocols: Peer ↔ peer coordination (environment-owned)

Protocol Architecture:
    Each protocol is composed of two components:
    1. CommunicationProtocol: Defines coordination messages (WHAT to communicate)
    2. ActionProtocol: Defines action coordination (HOW to coordinate actions)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from powergrid.agents.base import Agent, AgentID
from powergrid.messaging.base import Message, MessageType


# =============================================================================
# BASE PROTOCOL COMPONENTS
# =============================================================================

class CommunicationProtocol(ABC):
    """Handles message-passing coordination between agents.

    Communication protocols define WHAT information is shared and HOW
    agents exchange coordination signals (prices, bids, consensus values, etc.).
    This is the "coordination signal computation" layer.
    """

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

    def deliver_messages(
        self,
        messages: Dict[AgentID, Dict[str, Any]],
        receivers: Dict[AgentID, Agent],
        sender_id: AgentID,
        timestamp: float,
        mode: str = "centralized"
    ) -> None:
        """Deliver computed messages to receivers.

        Handles mode-specific delivery:
        - Centralized: Direct mailbox delivery
        - Distributed: Via message broker (handled by agent's async methods)

        Args:
            messages: Messages computed by compute_coordination_messages()
            receivers: Target agents
            sender_id: ID of coordinating agent
            timestamp: Current timestamp
            mode: "centralized" or "distributed"
        """
        for receiver_id, content in messages.items():
            if receiver_id not in receivers or not content:
                continue

            receiver = receivers[receiver_id]
            message = Message(
                env_id="",  # Environment-agnostic for protocol messages
                sender_id=sender_id,
                recipient_id=receiver_id,
                timestamp=timestamp,
                message_type=MessageType.INFO,
                payload=content
            )

            # Both modes use mailbox - distributed mode agents check mailbox async
            receiver.receive_message(message)


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

    def apply_actions(
        self,
        actions: Dict[AgentID, Any],
        subordinates: Dict[AgentID, Agent],
        mode: str = "centralized"
    ) -> None:
        """Apply coordinated actions to subordinates.

        Handles mode-specific application:
        - Centralized: Direct action setting via subordinate.act()
        - Distributed: Actions sent via message broker (environment handles this)

        Args:
            actions: Actions computed by compute_action_coordination()
            subordinates: Target agents
            mode: "centralized" or "distributed"
        """
        if mode == "centralized":
            # Centralized: directly call subordinate.act()
            for sub_id, action in actions.items():
                if sub_id not in subordinates or action is None:
                    continue
                subordinate = subordinates[sub_id]
                subordinate.act(subordinate.observation, upstream_action=action)
        # In distributed mode, environment handles action delivery via messages


# =============================================================================
# PROTOCOL: Composition of Communication + Action
# =============================================================================

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
        mode: str = "centralized",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute full coordination cycle.

        This is the main entry point that orchestrates:
        1. Communication: Compute and deliver messages
        2. Action: Compute and apply coordinated actions

        Args:
            coordinator_state: State of coordinating agent
            subordinate_states: States of subordinate agents
            coordinator_action: Action from coordinator policy (if any)
            mode: "centralized" or "distributed"
            context: Additional context (subordinates dict, timestamp, etc.)
        """
        context = context or {}
        subordinates = context.get("subordinates", {})
        timestamp = context.get("timestamp", 0.0)
        coordinator_id = context.get("coordinator_id", "coordinator")

        # Enrich context with coordinator_action
        context_with_action = {**context, "coordinator_action": coordinator_action}

        # Step 1: Communication coordination
        messages = self.communication_protocol.compute_coordination_messages(
            sender_state=coordinator_state,
            receiver_states=subordinate_states,
            context=context_with_action
        )

        self.communication_protocol.deliver_messages(
            messages=messages,
            receivers=subordinates,
            sender_id=coordinator_id,
            timestamp=timestamp,
            mode=mode
        )

        # Step 2: Action coordination
        actions = self.action_protocol.compute_action_coordination(
            coordinator_action=coordinator_action,
            subordinate_states=subordinate_states,
            coordination_messages=messages
        )

        self.action_protocol.apply_actions(
            actions=actions,
            subordinates=subordinates,
            mode=mode
        )


# =============================================================================
# VERTICAL PROTOCOLS (Agent-owned: Parent → Subordinate)
# =============================================================================

class VerticalProtocol(Protocol):
    """Vertical coordination protocol for hierarchical control.

    Each agent owns its own vertical protocol to coordinate its subordinates.
    This is decentralized - each agent independently manages its children.

    Example:
        GridAgent owns a PriceSignalProtocol to coordinate its DeviceAgents.
    """
    pass


# =============================================================================
# HORIZONTAL PROTOCOLS (Environment-owned: Peer ↔ Peer)
# =============================================================================

class HorizontalProtocol(Protocol):
    """Horizontal coordination protocol for peer-to-peer coordination.

    The environment owns and runs horizontal protocols, as they require
    global view of all agents. Agents participate but don't run the protocol.

    Example:
        Environment runs PeerToPeerTradingProtocol to enable trading between
        GridAgents MG1, MG2, and MG3.
    """
    pass


# =============================================================================
# NO-OP PROTOCOL COMPONENTS
# =============================================================================

class NoCommunication(CommunicationProtocol):
    """No message passing."""

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


class NoProtocol(VerticalProtocol):
    """No coordination protocol."""

    def __init__(self):
        super().__init__(
            communication_protocol=NoCommunication(),
            action_protocol=NoActionCoordination()
        )


# =============================================================================
# PRICE SIGNAL PROTOCOL (Decentralized coordination via price messages)
# =============================================================================

class PriceCommunicationProtocol(CommunicationProtocol):
    """Broadcasts price signals to subordinates."""

    def __init__(self, initial_price: float = 50.0):
        self.price = initial_price

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
    """No direct action control - devices act independently based on messages."""

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[AgentID, Any],
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None
    ) -> Dict[AgentID, Any]:
        """Return empty actions - devices decide independently."""
        return {sub_id: None for sub_id in subordinate_states}


class PriceSignalProtocol(VerticalProtocol):
    """Price-based coordination via marginal price signals.

    Communication: Broadcast price signals
    Action: Decentralized (devices respond to prices independently)

    Attributes:
        price: Current electricity price ($/MWh)
    """

    def __init__(self, initial_price: float = 50.0):
        """Initialize price signal protocol.

        Args:
            initial_price: Initial electricity price ($/MWh)
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


# =============================================================================
# SETPOINT PROTOCOL (Centralized control via direct action assignment)
# =============================================================================

class SetpointCommunicationProtocol(CommunicationProtocol):
    """Sends setpoint assignments as informational messages."""

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

        # Decompose action into per-device setpoints
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
        subordinates: Dict[AgentID, Agent]
    ) -> Dict[AgentID, Any]:
        """Split action vector into per-device setpoints."""
        if isinstance(action, dict):
            return action  # Already per-device

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
            return coordinator_action  # Already per-device

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
    """Setpoint-based coordination - parent assigns power setpoints.

    Communication: Send setpoint assignments (informational)
    Action: Centralized (coordinator directly controls devices)

    This protocol works in both centralized and distributed modes:
    - Centralized: Direct action application via subordinate.act()
    - Distributed: Actions sent via message broker
    """

    def __init__(self):
        super().__init__(
            communication_protocol=SetpointCommunicationProtocol(),
            action_protocol=CentralizedActionProtocol()
        )


class CentralizedSetpointProtocol(SetpointProtocol):
    """Alias for SetpointProtocol for backward compatibility.

    DEPRECATED: Use SetpointProtocol instead.
    The unified SetpointProtocol handles both centralized and distributed modes.
    """

    def __init__(self):
        import warnings
        warnings.warn(
            "CentralizedSetpointProtocol is deprecated. Use SetpointProtocol instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()


# =============================================================================
# P2P TRADING PROTOCOL (Horizontal peer coordination)
# =============================================================================

class TradingCommunicationProtocol(CommunicationProtocol):
    """P2P market coordination messages."""

    def __init__(self, trading_fee: float = 0.01):
        self.trading_fee = trading_fee

    def compute_coordination_messages(
        self,
        sender_state: Any,  # Not used in horizontal
        receiver_states: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Run market clearing and generate trade confirmations."""
        # Step 1: Collect bids/offers from receiver_states
        bids = {}
        offers = {}

        for agent_id, state in receiver_states.items():
            net_demand = state.get('net_demand', 0)
            marginal_cost = state.get('marginal_cost', 50)

            if net_demand > 0:  # Need to buy
                bids[agent_id] = {
                    'quantity': net_demand,
                    'max_price': marginal_cost * 1.2
                }
            elif net_demand < 0:  # Can sell
                offers[agent_id] = {
                    'quantity': -net_demand,
                    'min_price': marginal_cost * 0.8
                }

        # Step 2: Clear market
        trades = self._clear_market(bids, offers)

        # Step 3: Generate trade confirmation messages
        messages = {}
        for buyer_id, seller_id, quantity, price in trades:
            if buyer_id not in messages:
                messages[buyer_id] = {"type": "trade_confirmations", "trades": []}
            if seller_id not in messages:
                messages[seller_id] = {"type": "trade_confirmations", "trades": []}

            messages[buyer_id]["trades"].append({
                "counterparty": seller_id,
                "quantity": quantity,
                "price": price
            })
            messages[seller_id]["trades"].append({
                "counterparty": buyer_id,
                "quantity": -quantity,
                "price": price
            })

        return messages

    def _clear_market(
        self,
        bids: Dict[AgentID, Dict],
        offers: Dict[AgentID, Dict]
    ) -> List[Tuple[AgentID, AgentID, float, float]]:
        """Simple market clearing algorithm."""
        trades = []

        # Sort bids (descending by price) and offers (ascending by price)
        sorted_bids = sorted(
            bids.items(),
            key=lambda x: x[1]['max_price'],
            reverse=True
        )
        sorted_offers = sorted(
            offers.items(),
            key=lambda x: x[1]['min_price']
        )

        bid_idx = 0
        offer_idx = 0

        while bid_idx < len(sorted_bids) and offer_idx < len(sorted_offers):
            buyer_id, bid = sorted_bids[bid_idx]
            seller_id, offer = sorted_offers[offer_idx]

            if bid['max_price'] >= offer['min_price']:
                trade_price = (bid['max_price'] + offer['min_price']) / 2
                trade_qty = min(bid['quantity'], offer['quantity'])

                trades.append((buyer_id, seller_id, trade_qty, trade_price))

                bid['quantity'] -= trade_qty
                offer['quantity'] -= trade_qty

                if bid['quantity'] == 0:
                    bid_idx += 1
                if offer['quantity'] == 0:
                    offer_idx += 1
            else:
                break

        return trades


class TradingActionProtocol(ActionProtocol):
    """Adjust power setpoints based on trades."""

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[AgentID, Any],
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None
    ) -> Dict[AgentID, Any]:
        """Compute power adjustments based on cleared trades."""
        if not coordination_messages:
            return {sub_id: None for sub_id in subordinate_states}

        # Compute net power change from trades
        actions = {}
        for agent_id, message in coordination_messages.items():
            if "trades" in message:
                net_trade = sum(t["quantity"] for t in message["trades"])
                # Action = adjust power setpoint by net trade amount
                actions[agent_id] = {"power_adjustment": net_trade}
            else:
                actions[agent_id] = None

        return actions


class PeerToPeerTradingProtocol(HorizontalProtocol):
    """Peer-to-peer energy trading marketplace.

    Communication: Market clearing and trade confirmations
    Action: Adjust power setpoints based on trades

    Agents submit bids/offers based on their net demand and marginal cost.
    The environment (acting as market auctioneer) clears the market and
    sends trade confirmations back to agents.

    Attributes:
        trading_fee: Transaction fee as fraction of trade price
    """

    def __init__(self, trading_fee: float = 0.01):
        """Initialize P2P trading protocol.

        Args:
            trading_fee: Transaction fee as fraction of trade price
        """
        super().__init__(
            communication_protocol=TradingCommunicationProtocol(trading_fee),
            action_protocol=TradingActionProtocol()
        )

    @property
    def trading_fee(self):
        """Get trading fee."""
        return self.communication_protocol.trading_fee

    @trading_fee.setter
    def trading_fee(self, value):
        """Set trading fee."""
        self.communication_protocol.trading_fee = value


# =============================================================================
# CONSENSUS PROTOCOL (Horizontal peer coordination)
# =============================================================================

class ConsensusCommunicationProtocol(CommunicationProtocol):
    """Distributed consensus via gossip algorithm."""

    def __init__(self, max_iterations: int = 10, tolerance: float = 0.01):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def compute_coordination_messages(
        self,
        sender_state: Any,
        receiver_states: Dict[AgentID, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Run consensus algorithm (gossip) and return consensus values."""
        context = context or {}
        topology = context.get("topology")

        # Initialize with local values
        values = {
            agent_id: state.get('control_value', 0)
            for agent_id, state in receiver_states.items()
        }

        # Build adjacency from topology or use fully connected
        if topology and 'adjacency' in topology:
            adjacency = topology['adjacency']
        else:
            # Fully connected graph
            adjacency = {
                aid: [other for other in receiver_states if other != aid]
                for aid in receiver_states
            }

        # Iterative consensus
        for iteration in range(self.max_iterations):
            new_values = {}

            for agent_id in receiver_states:
                # Average with neighbors
                neighbors = adjacency.get(agent_id, [])
                neighbor_vals = [values[nid] for nid in neighbors if nid in values]

                if neighbor_vals:
                    new_values[agent_id] = (
                        values[agent_id] + sum(neighbor_vals)
                    ) / (len(neighbor_vals) + 1)
                else:
                    new_values[agent_id] = values[agent_id]

            # Check convergence
            max_change = max(
                abs(new_values[aid] - values[aid])
                for aid in receiver_states
            )

            values = new_values

            if max_change < self.tolerance:
                break

        # Return consensus values as messages
        return {
            agent_id: {
                "type": "consensus_value",
                "consensus_value": values[agent_id]
            }
            for agent_id in receiver_states
        }


class ConsensusActionProtocol(ActionProtocol):
    """No direct action control for consensus - agents use consensus values."""

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[AgentID, Any],
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None
    ) -> Dict[AgentID, Any]:
        """Return None - agents use consensus values in their own policies."""
        return {sub_id: None for sub_id in subordinate_states}


class ConsensusProtocol(HorizontalProtocol):
    """Distributed consensus via gossip algorithm.

    Communication: Iterative averaging until convergence
    Action: None (agents use consensus values independently)

    Agents iteratively average their values with neighbors until convergence.
    Useful for coordinated frequency regulation or voltage control.

    Attributes:
        max_iterations: Maximum gossip iterations
        tolerance: Convergence threshold
    """

    def __init__(self, max_iterations: int = 10, tolerance: float = 0.01):
        """Initialize consensus protocol.

        Args:
            max_iterations: Maximum gossip iterations
            tolerance: Convergence threshold
        """
        super().__init__(
            communication_protocol=ConsensusCommunicationProtocol(max_iterations, tolerance),
            action_protocol=ConsensusActionProtocol()
        )

    @property
    def max_iterations(self):
        """Get max iterations."""
        return self.communication_protocol.max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        """Set max iterations."""
        self.communication_protocol.max_iterations = value

    @property
    def tolerance(self):
        """Get tolerance."""
        return self.communication_protocol.tolerance

    @tolerance.setter
    def tolerance(self, value):
        """Set tolerance."""
        self.communication_protocol.tolerance = value


class NoHorizontalProtocol(HorizontalProtocol):
    """No peer coordination - agents act independently."""

    def __init__(self):
        super().__init__(
            communication_protocol=NoCommunication(),
            action_protocol=NoActionCoordination()
        )

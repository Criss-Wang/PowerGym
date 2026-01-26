"""Horizontal protocols for peer-to-peer coordination.

Horizontal protocols handle Peer <-> Peer coordination.
The environment owns and runs horizontal protocols, as they require
global view of all agents.

Includes:
- PeerToPeerTradingProtocol: P2P marketplace for resource trading
- ConsensusProtocol: Distributed consensus via gossip algorithm
"""

from typing import Any, Dict, List, Optional, Tuple

from heron.protocols.base import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
    NoCommunication,
    NoActionCoordination,
)
from heron.utils.typing import AgentID


class HorizontalProtocol(Protocol):
    """Horizontal coordination protocol for peer-to-peer coordination.

    The environment owns and runs horizontal protocols, as they require
    global view of all agents. Agents participate but don't run the protocol.

    Example:
        Environment runs PeerToPeerTradingProtocol to enable trading between
        coordinators A, B, and C.
    """
    pass


# =============================================================================
# P2P TRADING PROTOCOL
# =============================================================================

class TradingCommunicationProtocol(CommunicationProtocol):
    """P2P market coordination messages."""

    def __init__(self, trading_fee: float = 0.01):
        self.trading_fee = trading_fee
        self.neighbors = set()

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
    """Adjust setpoints based on trades."""

    def compute_action_coordination(
        self,
        coordinator_action: Optional[Any],
        subordinate_states: Dict[AgentID, Any],
        coordination_messages: Optional[Dict[AgentID, Dict[str, Any]]] = None
    ) -> Dict[AgentID, Any]:
        """Compute adjustments based on cleared trades."""
        if not coordination_messages:
            return {sub_id: None for sub_id in subordinate_states}

        # Compute net change from trades
        actions = {}
        for agent_id, message in coordination_messages.items():
            if "trades" in message:
                net_trade = sum(t["quantity"] for t in message["trades"])
                # Action = adjust setpoint by net trade amount
                actions[agent_id] = {"adjustment": net_trade}
            else:
                actions[agent_id] = None

        return actions


class PeerToPeerTradingProtocol(HorizontalProtocol):
    """Peer-to-peer trading marketplace.

    Communication: Market clearing and trade confirmations
    Action: Adjust setpoints based on trades

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
# CONSENSUS PROTOCOL
# =============================================================================

class ConsensusCommunicationProtocol(CommunicationProtocol):
    """Distributed consensus via gossip algorithm."""

    def __init__(self, max_iterations: int = 10, tolerance: float = 0.01):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.neighbors = set()

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
    Useful for coordinated control across distributed systems.

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

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
    """P2P market coordination messages.

    Attributes:
        trading_fee: Transaction fee as fraction of trade price
        demand_field: State field name for net demand (positive=buy, negative=sell)
        cost_field: State field name for marginal cost
        default_cost: Default marginal cost when field is missing
        buy_price_multiplier: Multiplier applied to cost for max buy price
        sell_price_multiplier: Multiplier applied to cost for min sell price
    """

    def __init__(
        self,
        trading_fee: float = 0.01,
        demand_field: str = "net_demand",
        cost_field: str = "marginal_cost",
        default_cost: float = 50.0,
        buy_price_multiplier: float = 1.2,
        sell_price_multiplier: float = 0.8,
    ):
        self.trading_fee = trading_fee
        self.demand_field = demand_field
        self.cost_field = cost_field
        self.default_cost = default_cost
        self.buy_price_multiplier = buy_price_multiplier
        self.sell_price_multiplier = sell_price_multiplier
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
            # Handle both dict-like objects and Observation objects
            if hasattr(state, 'local'):
                # Observation object - access local dict
                local = state.local
            elif hasattr(state, 'get'):
                # dict-like object
                local = state
            else:
                local = {}
            net_demand = local.get(self.demand_field, 0)
            marginal_cost = local.get(self.cost_field, self.default_cost)

            if net_demand > 0:  # Need to buy
                bids[agent_id] = {
                    'quantity': net_demand,
                    'max_price': marginal_cost * self.buy_price_multiplier
                }
            elif net_demand < 0:  # Can sell
                offers[agent_id] = {
                    'quantity': -net_demand,
                    'min_price': marginal_cost * self.sell_price_multiplier
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
        demand_field: State field name for net demand
        cost_field: State field name for marginal cost
        default_cost: Default marginal cost when field is missing
        buy_price_multiplier: Multiplier for max buy price
        sell_price_multiplier: Multiplier for min sell price
    """

    def __init__(
        self,
        trading_fee: float = 0.01,
        demand_field: str = "net_demand",
        cost_field: str = "marginal_cost",
        default_cost: float = 50.0,
        buy_price_multiplier: float = 1.2,
        sell_price_multiplier: float = 0.8,
    ):
        """Initialize P2P trading protocol.

        Args:
            trading_fee: Transaction fee as fraction of trade price
            demand_field: State field name for net demand (positive=buy, negative=sell)
            cost_field: State field name for marginal cost
            default_cost: Default marginal cost when field is missing
            buy_price_multiplier: Multiplier applied to cost for max buy price
            sell_price_multiplier: Multiplier applied to cost for min sell price
        """
        super().__init__(
            communication_protocol=TradingCommunicationProtocol(
                trading_fee=trading_fee,
                demand_field=demand_field,
                cost_field=cost_field,
                default_cost=default_cost,
                buy_price_multiplier=buy_price_multiplier,
                sell_price_multiplier=sell_price_multiplier,
            ),
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

    @property
    def demand_field(self):
        """Get demand field name."""
        return self.communication_protocol.demand_field

    @demand_field.setter
    def demand_field(self, value):
        """Set demand field name."""
        self.communication_protocol.demand_field = value

    @property
    def cost_field(self):
        """Get cost field name."""
        return self.communication_protocol.cost_field

    @cost_field.setter
    def cost_field(self, value):
        """Set cost field name."""
        self.communication_protocol.cost_field = value


# =============================================================================
# CONSENSUS PROTOCOL
# =============================================================================

class ConsensusCommunicationProtocol(CommunicationProtocol):
    """Distributed consensus via gossip algorithm.

    Attributes:
        max_iterations: Maximum gossip iterations
        tolerance: Convergence threshold
        value_field: State field name for the value to reach consensus on
        default_value: Default value when field is missing
    """

    def __init__(
        self,
        max_iterations: int = 10,
        tolerance: float = 0.01,
        value_field: str = "control_value",
        default_value: float = 0.0,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.value_field = value_field
        self.default_value = default_value
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
        def get_value(state):
            if hasattr(state, 'local'):
                return state.local.get(self.value_field, self.default_value)
            elif hasattr(state, 'get'):
                return state.get(self.value_field, self.default_value)
            return self.default_value

        values = {
            agent_id: get_value(state)
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
        value_field: State field name for the value to reach consensus on
        default_value: Default value when field is missing
    """

    def __init__(
        self,
        max_iterations: int = 10,
        tolerance: float = 0.01,
        value_field: str = "control_value",
        default_value: float = 0.0,
    ):
        """Initialize consensus protocol.

        Args:
            max_iterations: Maximum gossip iterations
            tolerance: Convergence threshold
            value_field: State field name for the value to reach consensus on
            default_value: Default value when field is missing
        """
        super().__init__(
            communication_protocol=ConsensusCommunicationProtocol(
                max_iterations=max_iterations,
                tolerance=tolerance,
                value_field=value_field,
                default_value=default_value,
            ),
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

    @property
    def value_field(self):
        """Get value field name."""
        return self.communication_protocol.value_field

    @value_field.setter
    def value_field(self, value):
        """Set value field name."""
        self.communication_protocol.value_field = value


class NoHorizontalProtocol(HorizontalProtocol):
    """No peer coordination - agents act independently."""

    def __init__(self):
        super().__init__(
            communication_protocol=NoCommunication(),
            action_protocol=NoActionCoordination()
        )

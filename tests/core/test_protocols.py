"""Tests for core.protocols module."""

import pytest
import numpy as np
from powergrid.core.protocols import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
    VerticalProtocol,
    NoProtocol,
    NoCommunication,
    NoActionCoordination,
    PriceSignalProtocol,
    PriceCommunicationProtocol,
    DecentralizedActionProtocol,
    SetpointProtocol,
    SetpointCommunicationProtocol,
    CentralizedActionProtocol,
    HorizontalProtocol,
    NoHorizontalProtocol,
    PeerToPeerTradingProtocol,
    TradingCommunicationProtocol,
    TradingActionProtocol,
    ConsensusProtocol,
    ConsensusCommunicationProtocol,
    ConsensusActionProtocol,
)
from powergrid.agents.base import Agent, Observation
from powergrid.core.action import Action


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, agent_id, action_dim_c=1, action_dim_d=0):
        super().__init__(agent_id=agent_id, level=1)
        self.observation = Observation(local={"value": 1.0})
        self.action = Action()
        self.action.dim_c = action_dim_c
        self.action.dim_d = action_dim_d
        self.mailbox = []
        self.last_action = None

    def observe(self, global_state=None, *args, **kwargs):
        return Observation(local={"value": 1.0}, timestamp=0.0)

    def act(self, observation, upstream_action=None, *args, **kwargs):
        self.last_action = upstream_action
        return np.array([1.0])

    def receive_message(self, message):
        """Receive a message and add to mailbox."""
        self.mailbox.append(message)


class TestProtocol:
    """Test Protocol base class."""

    def test_protocol_composition(self):
        """Test Protocol can be composed of communication + action protocols."""
        comm_protocol = NoCommunication()
        action_protocol = NoActionCoordination()
        protocol = Protocol(
            communication_protocol=comm_protocol,
            action_protocol=action_protocol
        )

        assert protocol.communication_protocol == comm_protocol
        assert protocol.action_protocol == action_protocol
        assert protocol.no_op()

    def test_protocol_coordinate(self):
        """Test Protocol coordinate orchestrates communication and action."""
        protocol = NoProtocol()

        device1 = MockAgent("device1")
        device2 = MockAgent("device2")
        devices = {"device1": device1, "device2": device2}

        coordinator_state = {"value": 1.0}
        subordinate_states = {
            "device1": {"value": 1.0},
            "device2": {"value": 2.0}
        }
        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        # Should not raise
        protocol.coordinate(
            coordinator_state=coordinator_state,
            subordinate_states=subordinate_states,
            coordinator_action=None,
            mode="centralized",
            context=context
        )


class TestCommunicationProtocol:
    """Test CommunicationProtocol base class."""

    def test_no_communication(self):
        """Test NoCommunication returns empty messages."""
        comm_protocol = NoCommunication()

        messages = comm_protocol.compute_coordination_messages(
            sender_state={},
            receiver_states={"device1": {}, "device2": {}},
            context={}
        )

        assert messages == {"device1": {}, "device2": {}}

    def test_message_delivery(self):
        """Test message delivery to receivers."""
        comm_protocol = NoCommunication()

        device1 = MockAgent("device1")
        device2 = MockAgent("device2")
        devices = {"device1": device1, "device2": device2}

        messages = {
            "device1": {"type": "test", "value": 1},
            "device2": {"type": "test", "value": 2}
        }

        comm_protocol.deliver_messages(
            messages=messages,
            receivers=devices,
            sender_id="coordinator",
            timestamp=1.0,
            mode="centralized"
        )

        assert len(device1.mailbox) == 1
        assert device1.mailbox[0].payload["value"] == 1
        assert len(device2.mailbox) == 1
        assert device2.mailbox[0].payload["value"] == 2


class TestActionProtocol:
    """Test ActionProtocol base class."""

    def test_no_action_coordination(self):
        """Test NoActionCoordination returns None actions."""
        action_protocol = NoActionCoordination()

        actions = action_protocol.compute_action_coordination(
            coordinator_action=np.array([1.0, 2.0]),
            subordinate_states={"device1": {}, "device2": {}},
            coordination_messages={}
        )

        assert actions == {"device1": None, "device2": None}

    def test_action_application_centralized(self):
        """Test action application in centralized mode."""
        action_protocol = NoActionCoordination()

        device1 = MockAgent("device1")
        device2 = MockAgent("device2")
        devices = {"device1": device1, "device2": device2}

        actions = {"device1": np.array([1.0]), "device2": np.array([2.0])}

        action_protocol.apply_actions(
            actions=actions,
            subordinates=devices,
            mode="centralized"
        )

        # In centralized mode, actions are applied directly
        assert device1.last_action is not None
        assert device2.last_action is not None


class TestVerticalProtocol:
    """Test VerticalProtocol base class."""

    def test_vertical_protocol_inherits_from_protocol(self):
        """Test VerticalProtocol is a Protocol."""
        protocol = NoProtocol()
        assert isinstance(protocol, VerticalProtocol)
        assert isinstance(protocol, Protocol)


class TestNoProtocol:
    """Test NoProtocol implementation."""

    def test_no_protocol_is_no_op(self):
        """Test NoProtocol is a no-op."""
        protocol = NoProtocol()
        assert protocol.no_op()

    def test_no_protocol_coordinate(self):
        """Test NoProtocol coordinate does nothing."""
        protocol = NoProtocol()

        device1 = MockAgent("device1")
        device2 = MockAgent("device2")
        devices = {"device1": device1, "device2": device2}

        coordinator_state = {}
        subordinate_states = {"device1": {}, "device2": {}}
        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        protocol.coordinate(
            coordinator_state=coordinator_state,
            subordinate_states=subordinate_states,
            coordinator_action=None,
            mode="centralized",
            context=context
        )

        # No messages should be sent
        assert len(device1.mailbox) == 0
        assert len(device2.mailbox) == 0
        # No actions should be applied
        assert device1.last_action is None
        assert device2.last_action is None


class TestPriceSignalProtocol:
    """Test PriceSignalProtocol implementation."""

    def test_price_protocol_initialization(self):
        """Test price protocol initialization."""
        protocol = PriceSignalProtocol(initial_price=60.0)
        assert protocol.price == 60.0

    def test_price_protocol_default_price(self):
        """Test default price."""
        protocol = PriceSignalProtocol()
        assert protocol.price == 50.0

    def test_price_protocol_composition(self):
        """Test PriceSignalProtocol is composed correctly."""
        protocol = PriceSignalProtocol(initial_price=55.0)

        assert isinstance(protocol.communication_protocol, PriceCommunicationProtocol)
        assert isinstance(protocol.action_protocol, DecentralizedActionProtocol)
        assert not protocol.no_op()

    def test_price_communication_protocol(self):
        """Test price communication protocol computes correct messages."""
        comm_protocol = PriceCommunicationProtocol(initial_price=55.0)

        messages = comm_protocol.compute_coordination_messages(
            sender_state={},
            receiver_states={"device1": {}, "device2": {}},
            context={}
        )

        assert len(messages) == 2
        assert messages["device1"]["price"] == 55.0
        assert messages["device1"]["type"] == "price_signal"
        assert messages["device2"]["price"] == 55.0

    def test_price_protocol_broadcast(self):
        """Test price broadcast to all subordinates via coordinate."""
        protocol = PriceSignalProtocol(initial_price=55.0)

        device1 = MockAgent("device1")
        device2 = MockAgent("device2")
        devices = {"device1": device1, "device2": device2}

        coordinator_state = {}
        subordinate_states = {"device1": {}, "device2": {}}
        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        protocol.coordinate(
            coordinator_state=coordinator_state,
            subordinate_states=subordinate_states,
            coordinator_action=None,
            mode="centralized",
            context=context
        )

        # Check messages were delivered
        assert len(device1.mailbox) == 1
        assert device1.mailbox[0].payload["price"] == 55.0
        assert device1.mailbox[0].payload["type"] == "price_signal"

        assert len(device2.mailbox) == 1
        assert device2.mailbox[0].payload["price"] == 55.0

    def test_price_protocol_update_from_action(self):
        """Test price update from coordinator action (scalar)."""
        protocol = PriceSignalProtocol(initial_price=50.0)

        device1 = MockAgent("device1")
        devices = {"device1": device1}

        coordinator_state = {}
        subordinate_states = {"device1": {}}
        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "coordinator_action": 75.0,
            "timestamp": 0.0
        }

        protocol.coordinate(
            coordinator_state=coordinator_state,
            subordinate_states=subordinate_states,
            coordinator_action=75.0,
            mode="centralized",
            context=context
        )

        # Price should be updated
        assert protocol.price == 75.0

        # Message should contain new price
        assert device1.mailbox[0].payload["price"] == 75.0

    def test_price_protocol_update_from_dict_action(self):
        """Test price update from dict action."""
        protocol = PriceSignalProtocol(initial_price=50.0)

        device1 = MockAgent("device1")
        devices = {"device1": device1}

        coordinator_state = {}
        subordinate_states = {"device1": {}}
        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        protocol.coordinate(
            coordinator_state=coordinator_state,
            subordinate_states=subordinate_states,
            coordinator_action={"price": 80.0},
            mode="centralized",
            context=context
        )

        assert protocol.price == 80.0
        assert device1.mailbox[0].payload["price"] == 80.0

    def test_decentralized_action_protocol(self):
        """Test DecentralizedActionProtocol returns no actions."""
        action_protocol = DecentralizedActionProtocol()

        actions = action_protocol.compute_action_coordination(
            coordinator_action=np.array([1.0, 2.0]),
            subordinate_states={"device1": {}, "device2": {}},
            coordination_messages={}
        )

        assert actions["device1"] is None
        assert actions["device2"] is None


class TestSetpointProtocol:
    """Test SetpointProtocol implementation."""

    def test_setpoint_protocol_composition(self):
        """Test SetpointProtocol composition."""
        protocol = SetpointProtocol()

        assert isinstance(protocol.communication_protocol, SetpointCommunicationProtocol)
        assert isinstance(protocol.action_protocol, CentralizedActionProtocol)

    def test_setpoint_communication_protocol_with_dict(self):
        """Test setpoint communication with dict action."""
        comm_protocol = SetpointCommunicationProtocol()

        device1 = MockAgent("device1", action_dim_c=2)
        device2 = MockAgent("device2", action_dim_c=1)
        devices = {"device1": device1, "device2": device2}

        action_dict = {
            "device1": np.array([1.0, 2.0]),
            "device2": np.array([3.0])
        }

        messages = comm_protocol.compute_coordination_messages(
            sender_state={},
            receiver_states={"device1": {}, "device2": {}},
            context={"coordinator_action": action_dict, "subordinates": devices}
        )

        assert "device1" in messages
        assert "device2" in messages
        np.testing.assert_array_equal(messages["device1"]["setpoint"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(messages["device2"]["setpoint"], np.array([3.0]))

    def test_setpoint_communication_protocol_with_array(self):
        """Test setpoint communication with numpy array."""
        comm_protocol = SetpointCommunicationProtocol()

        device1 = MockAgent("device1", action_dim_c=2)
        device2 = MockAgent("device2", action_dim_c=1)
        devices = {"device1": device1, "device2": device2}

        action_array = np.array([1.0, 2.0, 3.0])

        messages = comm_protocol.compute_coordination_messages(
            sender_state={},
            receiver_states={"device1": {}, "device2": {}},
            context={"coordinator_action": action_array, "subordinates": devices}
        )

        assert "device1" in messages
        assert "device2" in messages
        np.testing.assert_array_equal(messages["device1"]["setpoint"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(messages["device2"]["setpoint"], np.array([3.0]))

    def test_setpoint_protocol_coordinate_with_dict(self):
        """Test SetpointProtocol coordinate with dict action."""
        protocol = SetpointProtocol()

        device1 = MockAgent("device1", action_dim_c=2)
        device2 = MockAgent("device2", action_dim_c=1)
        devices = {"device1": device1, "device2": device2}

        action_dict = {
            "device1": np.array([1.5, 2.5]),
            "device2": np.array([3.5])
        }

        coordinator_state = {}
        subordinate_states = {"device1": {}, "device2": {}}
        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        protocol.coordinate(
            coordinator_state=coordinator_state,
            subordinate_states=subordinate_states,
            coordinator_action=action_dict,
            mode="centralized",
            context=context
        )

        # Check messages were sent
        assert len(device1.mailbox) == 1
        assert len(device2.mailbox) == 1

        # Check actions were applied
        np.testing.assert_array_equal(device1.last_action, np.array([1.5, 2.5]))
        np.testing.assert_array_equal(device2.last_action, np.array([3.5]))

    def test_setpoint_protocol_coordinate_with_array(self):
        """Test SetpointProtocol coordinate with numpy array."""
        protocol = SetpointProtocol()

        device1 = MockAgent("device1", action_dim_c=2)
        device2 = MockAgent("device2", action_dim_c=1)
        devices = {"device1": device1, "device2": device2}

        action_array = np.array([10.0, 20.0, 30.0])

        coordinator_state = {}
        subordinate_states = {"device1": {}, "device2": {}}
        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        protocol.coordinate(
            coordinator_state=coordinator_state,
            subordinate_states=subordinate_states,
            coordinator_action=action_array,
            mode="centralized",
            context=context
        )

        # Check messages contain decomposed setpoints
        assert len(device1.mailbox) == 1
        assert len(device2.mailbox) == 1

        # Check actions were applied
        np.testing.assert_array_equal(device1.last_action, np.array([10.0, 20.0]))
        np.testing.assert_array_equal(device2.last_action, np.array([30.0]))


class TestHorizontalProtocol:
    """Test HorizontalProtocol base class."""

    def test_horizontal_protocol_inherits_from_protocol(self):
        """Test HorizontalProtocol is a Protocol."""
        protocol = NoHorizontalProtocol()
        assert isinstance(protocol, HorizontalProtocol)
        assert isinstance(protocol, Protocol)


class TestNoHorizontalProtocol:
    """Test NoHorizontalProtocol implementation."""

    def test_no_horizontal_protocol_is_no_op(self):
        """Test no horizontal coordination."""
        protocol = NoHorizontalProtocol()
        assert protocol.no_op()


class TestPeerToPeerTradingProtocol:
    """Test P2P trading protocol."""

    def test_p2p_protocol_initialization(self):
        """Test P2P protocol initialization."""
        protocol = PeerToPeerTradingProtocol(trading_fee=0.02)
        assert protocol.trading_fee == 0.02

    def test_p2p_protocol_composition(self):
        """Test P2P protocol composition."""
        protocol = PeerToPeerTradingProtocol()

        assert isinstance(protocol.communication_protocol, TradingCommunicationProtocol)
        assert isinstance(protocol.action_protocol, TradingActionProtocol)

    def test_p2p_communication_no_trades(self):
        """Test P2P communication with no feasible trades."""
        comm_protocol = TradingCommunicationProtocol()

        receiver_states = {
            "agent1": {"net_demand": 0, "marginal_cost": 50},
            "agent2": {"net_demand": 0, "marginal_cost": 50}
        }

        messages = comm_protocol.compute_coordination_messages(
            sender_state={},
            receiver_states=receiver_states,
            context={}
        )

        # No trades, so no messages
        assert len(messages) == 0

    def test_p2p_communication_simple_trade(self):
        """Test simple P2P trade between buyer and seller."""
        comm_protocol = TradingCommunicationProtocol()

        receiver_states = {
            "buyer": {"net_demand": 10, "marginal_cost": 60},
            "seller": {"net_demand": -10, "marginal_cost": 40}
        }

        messages = comm_protocol.compute_coordination_messages(
            sender_state={},
            receiver_states=receiver_states,
            context={}
        )

        # Should have trade messages for both
        assert "buyer" in messages
        assert "seller" in messages
        assert len(messages["buyer"]["trades"]) == 1
        assert len(messages["seller"]["trades"]) == 1

        # Check trade details
        buyer_trade = messages["buyer"]["trades"][0]
        assert buyer_trade["counterparty"] == "seller"
        assert buyer_trade["quantity"] == 10
        assert 40 < buyer_trade["price"] < 60  # Between marginal costs

        seller_trade = messages["seller"]["trades"][0]
        assert seller_trade["counterparty"] == "buyer"
        assert seller_trade["quantity"] == -10  # Negative for seller


class TestConsensusProtocol:
    """Test consensus protocol."""

    def test_consensus_protocol_initialization(self):
        """Test consensus protocol initialization."""
        protocol = ConsensusProtocol(max_iterations=20, tolerance=0.001)

        assert protocol.max_iterations == 20
        assert protocol.tolerance == 0.001

    def test_consensus_protocol_composition(self):
        """Test consensus protocol composition."""
        protocol = ConsensusProtocol()

        assert isinstance(protocol.communication_protocol, ConsensusCommunicationProtocol)
        assert isinstance(protocol.action_protocol, ConsensusActionProtocol)

    def test_consensus_communication_convergence(self):
        """Test consensus communication reaches agreement."""
        comm_protocol = ConsensusCommunicationProtocol(max_iterations=100, tolerance=0.01)

        receiver_states = {
            "agent1": {"control_value": 10.0},
            "agent2": {"control_value": 20.0},
            "agent3": {"control_value": 30.0}
        }

        messages = comm_protocol.compute_coordination_messages(
            sender_state={},
            receiver_states=receiver_states,
            context={}
        )

        # All agents should get consensus values
        assert "agent1" in messages
        assert "agent2" in messages
        assert "agent3" in messages

        # Values should be close (converged to average ~20)
        values = [messages[aid]["consensus_value"] for aid in receiver_states]
        assert all(15 < v < 25 for v in values)

        # Values should be close to each other
        max_diff = max(values) - min(values)
        assert max_diff < 1.0

    def test_consensus_with_topology(self):
        """Test consensus with specific network topology."""
        comm_protocol = ConsensusCommunicationProtocol(max_iterations=100)

        receiver_states = {
            "agent1": {"control_value": 10.0},
            "agent2": {"control_value": 20.0},
            "agent3": {"control_value": 30.0}
        }

        # Line topology: agent1 - agent2 - agent3
        topology = {
            "adjacency": {
                "agent1": ["agent2"],
                "agent2": ["agent1", "agent3"],
                "agent3": ["agent2"]
            }
        }

        messages = comm_protocol.compute_coordination_messages(
            sender_state={},
            receiver_states=receiver_states,
            context={"topology": topology}
        )

        # Should still converge
        values = [messages[aid]["consensus_value"] for aid in receiver_states]
        max_diff = max(values) - min(values)
        assert max_diff < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for protocol components - communication and action protocols.

This test suite focuses on the core execution paths:
1. CommunicationProtocol: Message computation and delivery
2. ActionProtocol: Action coordination and application
3. Protocol: Full coordination cycle (communication + action)
"""

import pytest
import numpy as np
from heron.protocols.base import (
    Protocol,
    CommunicationProtocol,
    ActionProtocol,
    NoProtocol,
    NoCommunication,
    NoActionCoordination,
)
from heron.protocols.vertical import (
    VerticalProtocol,
    SetpointProtocol,
    PriceSignalProtocol,
    PriceCommunicationProtocol,
    DecentralizedActionProtocol,
    SetpointCommunicationProtocol,
    CentralizedActionProtocol,
)
from heron.protocols.horizontal import (
    PeerToPeerTradingProtocol,
    ConsensusProtocol,
)
from heron.agents.base import Agent, Observation
from heron.core.action import Action


# =============================================================================
# Test Fixtures
# =============================================================================

class SimpleAgent(Agent):
    """Simple mock agent for testing."""

    def __init__(self, agent_id, action_dim=2):
        super().__init__(agent_id=agent_id, level=1)
        self.action = Action()
        self.action.dim_c = action_dim
        self.action.dim_d = 0
        self.mailbox = []
        self.last_action = None
        self.observation = Observation(local={"value": 1.0}, timestamp=0.0)

    def observe(self, global_state=None, *args, **kwargs):
        return self.observation

    def act(self, observation, upstream_action=None, *args, **kwargs):
        self.last_action = upstream_action
        return np.array([1.0] * self.action.dim_c)

    def receive_message(self, message):
        """Receive a message and add to mailbox."""
        self.mailbox.append(message)


# =============================================================================
# CommunicationProtocol Tests
# =============================================================================

class TestCommunicationProtocol:
    """Test CommunicationProtocol base functionality."""

    def test_no_communication_returns_empty_messages(self):
        """Test NoCommunication returns empty dict for all receivers."""
        comm = NoCommunication()

        messages = comm.compute_coordination_messages(
            sender_state={"value": 1.0},
            receiver_states={"dev1": {}, "dev2": {}},
            context={}
        )

        assert messages == {"dev1": {}, "dev2": {}}

    def test_message_computation_returns_dict_per_receiver(self):
        """Test that message computation returns a dict per receiver."""
        comm = NoCommunication()

        receiver_states = {
            "dev1": {"value": 1.0},
            "dev2": {"value": 2.0}
        }

        messages = comm.compute_coordination_messages(
            sender_state={},
            receiver_states=receiver_states,
            context={}
        )

        # NoCommunication returns empty dict for each receiver
        assert "dev1" in messages
        assert "dev2" in messages
        assert messages["dev1"] == {}
        assert messages["dev2"] == {}

    def test_price_communication_broadcasts_price(self):
        """Test PriceCommunicationProtocol broadcasts price to all devices."""
        comm = PriceCommunicationProtocol(initial_price=55.0)

        messages = comm.compute_coordination_messages(
            sender_state={},
            receiver_states={"dev1": {}, "dev2": {}, "dev3": {}},
            context={}
        )

        assert len(messages) == 3
        assert all(msg["price"] == 55.0 for msg in messages.values())
        assert all(msg["type"] == "price_signal" for msg in messages.values())

    def test_price_communication_updates_from_action(self):
        """Test price updates from coordinator action."""
        comm = PriceCommunicationProtocol(initial_price=50.0)

        # Scalar action
        messages = comm.compute_coordination_messages(
            sender_state={},
            receiver_states={"dev1": {}},
            context={"coordinator_action": 75.0}
        )

        assert comm.price == 75.0
        assert messages["dev1"]["price"] == 75.0

        # Dict action
        messages = comm.compute_coordination_messages(
            sender_state={},
            receiver_states={"dev1": {}},
            context={"coordinator_action": {"price": 80.0}}
        )

        assert comm.price == 80.0
        assert messages["dev1"]["price"] == 80.0

    def test_setpoint_communication_with_dict_action(self):
        """Test SetpointCommunicationProtocol with dict action."""
        comm = SetpointCommunicationProtocol()

        dev1 = SimpleAgent("dev1", action_dim=2)
        dev2 = SimpleAgent("dev2", action_dim=1)
        devices = {"dev1": dev1, "dev2": dev2}

        action_dict = {
            "dev1": np.array([1.0, 2.0]),
            "dev2": np.array([3.0])
        }

        messages = comm.compute_coordination_messages(
            sender_state={},
            receiver_states={"dev1": {}, "dev2": {}},
            context={"coordinator_action": action_dict, "subordinates": devices}
        )

        assert "dev1" in messages
        assert "dev2" in messages
        assert messages["dev1"]["type"] == "setpoint_command"
        np.testing.assert_array_equal(messages["dev1"]["setpoint"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(messages["dev2"]["setpoint"], np.array([3.0]))

    def test_setpoint_communication_with_array_action(self):
        """Test SetpointCommunicationProtocol decomposes array action."""
        comm = SetpointCommunicationProtocol()

        dev1 = SimpleAgent("dev1", action_dim=2)
        dev2 = SimpleAgent("dev2", action_dim=1)
        devices = {"dev1": dev1, "dev2": dev2}

        action_array = np.array([10.0, 20.0, 30.0])

        messages = comm.compute_coordination_messages(
            sender_state={},
            receiver_states={"dev1": {}, "dev2": {}},
            context={"coordinator_action": action_array, "subordinates": devices}
        )

        np.testing.assert_array_equal(messages["dev1"]["setpoint"], np.array([10.0, 20.0]))
        np.testing.assert_array_equal(messages["dev2"]["setpoint"], np.array([30.0]))


# =============================================================================
# ActionProtocol Tests
# =============================================================================

class TestActionProtocol:
    """Test ActionProtocol base functionality."""

    def test_no_action_coordination_returns_none(self):
        """Test NoActionCoordination returns None for all subordinates."""
        action_proto = NoActionCoordination()

        actions = action_proto.compute_action_coordination(
            coordinator_action=np.array([1.0, 2.0]),
            subordinate_states={"dev1": {}, "dev2": {}},
            coordination_messages={}
        )

        assert actions == {"dev1": None, "dev2": None}

    def test_action_coordination_returns_dict_per_subordinate(self):
        """Test that action coordination returns a dict per subordinate."""
        action_proto = NoActionCoordination()

        subordinate_states = {
            "dev1": {"value": 1.0},
            "dev2": {"value": 2.0}
        }

        actions = action_proto.compute_action_coordination(
            coordinator_action=np.array([1.0, 2.0]),
            subordinate_states=subordinate_states,
            coordination_messages={}
        )

        # NoActionCoordination returns None for each subordinate
        assert "dev1" in actions
        assert "dev2" in actions
        assert actions["dev1"] is None
        assert actions["dev2"] is None

    def test_decentralized_action_returns_none(self):
        """Test DecentralizedActionProtocol returns None (no direct control)."""
        action_proto = DecentralizedActionProtocol()

        actions = action_proto.compute_action_coordination(
            coordinator_action=np.array([1.0, 2.0]),
            subordinate_states={"dev1": {}, "dev2": {}},
            coordination_messages={}
        )

        assert actions["dev1"] is None
        assert actions["dev2"] is None

    def test_centralized_action_with_dict(self):
        """Test CentralizedActionProtocol with dict action."""
        action_proto = CentralizedActionProtocol()

        action_dict = {
            "dev1": np.array([1.0, 2.0]),
            "dev2": np.array([3.0])
        }

        actions = action_proto.compute_action_coordination(
            coordinator_action=action_dict,
            subordinate_states={"dev1": {}, "dev2": {}},
            coordination_messages={}
        )

        np.testing.assert_array_equal(actions["dev1"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(actions["dev2"], np.array([3.0]))

    def test_centralized_action_extracts_from_messages(self):
        """Test CentralizedActionProtocol extracts actions from messages."""
        action_proto = CentralizedActionProtocol()

        messages = {
            "dev1": {"setpoint": np.array([10.0, 20.0])},
            "dev2": {"setpoint": np.array([30.0])}
        }

        actions = action_proto.compute_action_coordination(
            coordinator_action=None,
            subordinate_states={"dev1": {}, "dev2": {}},
            coordination_messages=messages
        )

        np.testing.assert_array_equal(actions["dev1"], np.array([10.0, 20.0]))
        np.testing.assert_array_equal(actions["dev2"], np.array([30.0]))


# =============================================================================
# Protocol Coordination Tests (Full Cycle)
# =============================================================================

class TestProtocolCoordination:
    """Test full Protocol coordination cycle."""

    def test_no_protocol_does_nothing(self):
        """Test NoProtocol performs no coordination."""
        protocol = NoProtocol()

        dev1 = SimpleAgent("dev1")
        dev2 = SimpleAgent("dev2")
        devices = {"dev1": dev1, "dev2": dev2}

        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 1.0
        }

        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}, "dev2": {}},
            coordinator_action=None,
            context=context
        )

        # Empty messages and None actions
        assert messages == {"dev1": {}, "dev2": {}}
        assert actions == {"dev1": None, "dev2": None}

    def test_price_protocol_full_cycle(self):
        """Test PriceSignalProtocol full coordination cycle."""
        protocol = PriceSignalProtocol(initial_price=50.0)

        dev1 = SimpleAgent("dev1")
        dev2 = SimpleAgent("dev2")
        devices = {"dev1": dev1, "dev2": dev2}

        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 1.0
        }

        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}, "dev2": {}},
            coordinator_action=65.0,  # New price
            context=context
        )

        # Verify messages contain price
        assert messages["dev1"]["price"] == 65.0
        assert messages["dev1"]["type"] == "price_signal"
        assert messages["dev2"]["price"] == 65.0

        # Verify no direct action (decentralized)
        assert actions["dev1"] is None
        assert actions["dev2"] is None

        # Verify price updated
        assert protocol.price == 65.0

    def test_setpoint_protocol_full_cycle_with_dict(self):
        """Test SetpointProtocol full cycle with dict action."""
        protocol = SetpointProtocol()

        dev1 = SimpleAgent("dev1", action_dim=2)
        dev2 = SimpleAgent("dev2", action_dim=1)
        devices = {"dev1": dev1, "dev2": dev2}

        action_dict = {
            "dev1": np.array([1.5, 2.5]),
            "dev2": np.array([3.5])
        }

        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 2.0
        }

        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}, "dev2": {}},
            coordinator_action=action_dict,
            context=context
        )

        # Verify messages contain setpoints
        assert messages["dev1"]["type"] == "setpoint_command"
        np.testing.assert_array_equal(messages["dev1"]["setpoint"], np.array([1.5, 2.5]))
        np.testing.assert_array_equal(messages["dev2"]["setpoint"], np.array([3.5]))

        # Verify actions returned (centralized)
        np.testing.assert_array_equal(actions["dev1"], np.array([1.5, 2.5]))
        np.testing.assert_array_equal(actions["dev2"], np.array([3.5]))

    def test_setpoint_protocol_full_cycle_with_array(self):
        """Test SetpointProtocol full cycle with array action."""
        protocol = SetpointProtocol()

        dev1 = SimpleAgent("dev1", action_dim=2)
        dev2 = SimpleAgent("dev2", action_dim=1)
        devices = {"dev1": dev1, "dev2": dev2}

        action_array = np.array([10.0, 20.0, 30.0])

        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 3.0
        }

        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}, "dev2": {}},
            coordinator_action=action_array,
            context=context
        )

        # Verify messages contain decomposed setpoints
        np.testing.assert_array_equal(messages["dev1"]["setpoint"], np.array([10.0, 20.0]))
        np.testing.assert_array_equal(messages["dev2"]["setpoint"], np.array([30.0]))

        # Verify actions decomposed and returned
        np.testing.assert_array_equal(actions["dev1"], np.array([10.0, 20.0]))
        np.testing.assert_array_equal(actions["dev2"], np.array([30.0]))

    def test_protocol_composition(self):
        """Test Protocol composition with custom components."""
        comm_protocol = PriceCommunicationProtocol(initial_price=60.0)
        action_protocol = DecentralizedActionProtocol()

        protocol = Protocol(
            communication_protocol=comm_protocol,
            action_protocol=action_protocol
        )

        assert protocol.communication_protocol == comm_protocol
        assert protocol.action_protocol == action_protocol
        assert not protocol.no_op()

    def test_protocol_no_op_detection(self):
        """Test Protocol correctly detects no-op configuration."""
        no_op_protocol = Protocol(
            communication_protocol=NoCommunication(),
            action_protocol=NoActionCoordination()
        )

        assert no_op_protocol.no_op()

        active_protocol = Protocol(
            communication_protocol=PriceCommunicationProtocol(),
            action_protocol=NoActionCoordination()
        )

        assert not active_protocol.no_op()


# =============================================================================
# Protocol Functional Tests
# =============================================================================

class TestProtocolFunctional:
    """Test protocol functional behavior (returns values, no side effects)."""

    def test_setpoint_returns_actions(self):
        """Test SetpointProtocol returns actions."""
        protocol = SetpointProtocol()

        dev1 = SimpleAgent("dev1", action_dim=2)
        devices = {"dev1": dev1}

        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}},
            coordinator_action={"dev1": np.array([5.0, 6.0])},
            context=context
        )

        # Actions returned
        np.testing.assert_array_equal(actions["dev1"], np.array([5.0, 6.0]))

    def test_price_protocol_returns_none_actions(self):
        """Test PriceSignalProtocol returns None actions (decentralized)."""
        protocol = PriceSignalProtocol(initial_price=50.0)

        dev1 = SimpleAgent("dev1")
        devices = {"dev1": dev1}
        context = {"subordinates": devices, "coordinator_id": "grid", "timestamp": 0.0}

        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}},
            coordinator_action=70.0,
            context=context
        )

        assert messages["dev1"]["price"] == 70.0
        assert actions["dev1"] is None  # Decentralized action


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestProtocolEdgeCases:
    """Test edge cases and error handling."""

    def test_protocol_with_no_subordinates(self):
        """Test protocol handles empty subordinate dict."""
        protocol = PriceSignalProtocol()

        context = {
            "subordinates": {},
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        # Should not raise
        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states={},
            coordinator_action=50.0,
            context=context
        )

        assert messages == {}
        assert actions == {}

    def test_protocol_with_none_action(self):
        """Test protocol handles None coordinator action."""
        protocol = SetpointProtocol()

        dev1 = SimpleAgent("dev1")
        devices = {"dev1": dev1}
        context = {"subordinates": devices, "coordinator_id": "grid", "timestamp": 0.0}

        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}},
            coordinator_action=None,
            context=context
        )

        # Should handle gracefully (no action)
        assert actions["dev1"] is None

    def test_setpoint_protocol_with_subordinate_mismatch(self):
        """Test setpoint protocol handles subordinate mismatch."""
        protocol = SetpointProtocol()

        dev1 = SimpleAgent("dev1")
        devices = {"dev1": dev1}

        # Action includes non-existent dev2
        action = {"dev1": np.array([1.0]), "dev2": np.array([2.0])}

        context = {"subordinates": devices, "coordinator_id": "grid", "timestamp": 0.0}

        # Should handle gracefully (only dev1 in results)
        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}},
            coordinator_action=action,
            context=context
        )

        # dev1 action returned
        np.testing.assert_array_equal(actions["dev1"], np.array([1.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for protocol components - communication and action protocols.

This test suite focuses on the core execution paths:
1. CommunicationProtocol: Message computation and delivery
2. ActionProtocol: Action coordination and application
3. Protocol: Full coordination cycle (communication + action)
"""

import pytest
import numpy as np
from powergrid.core.protocols import (
    CommunicationProtocol,
    ActionProtocol,
    Protocol,
    NoCommunication,
    NoActionCoordination,
    PriceCommunicationProtocol,
    DecentralizedActionProtocol,
    SetpointCommunicationProtocol,
    CentralizedActionProtocol,
    PriceSignalProtocol,
    SetpointProtocol,
    NoProtocol,
)
from powergrid.agents.base import Agent, Observation
from powergrid.core.action import Action


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

    def test_message_delivery_centralized_mode(self):
        """Test message delivery in centralized mode uses direct mailbox."""
        comm = NoCommunication()

        dev1 = SimpleAgent("dev1")
        dev2 = SimpleAgent("dev2")
        devices = {"dev1": dev1, "dev2": dev2}

        messages = {
            "dev1": {"type": "test", "value": 10},
            "dev2": {"type": "test", "value": 20}
        }

        comm.deliver_messages(
            messages=messages,
            receivers=devices,
            sender_id="coordinator",
            timestamp=1.0,
            mode="centralized"
        )

        # Verify messages delivered to mailbox
        assert len(dev1.mailbox) == 1
        assert dev1.mailbox[0].sender_id == "coordinator"
        assert dev1.mailbox[0].payload["value"] == 10
        assert dev1.mailbox[0].timestamp == 1.0

        assert len(dev2.mailbox) == 1
        assert dev2.mailbox[0].payload["value"] == 20

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

    def test_action_application_centralized(self):
        """Test action application in centralized mode calls act() directly."""
        action_proto = NoActionCoordination()

        dev1 = SimpleAgent("dev1")
        dev2 = SimpleAgent("dev2")
        devices = {"dev1": dev1, "dev2": dev2}

        actions = {
            "dev1": np.array([1.5, 2.5]),
            "dev2": np.array([3.5, 4.5])
        }

        action_proto.apply_actions(
            actions=actions,
            subordinates=devices,
            mode="centralized"
        )

        # Verify actions were applied via act()
        np.testing.assert_array_equal(dev1.last_action, np.array([1.5, 2.5]))
        np.testing.assert_array_equal(dev2.last_action, np.array([3.5, 4.5]))

    def test_action_application_distributed(self):
        """Test action application in distributed mode (delegated to env)."""
        action_proto = NoActionCoordination()

        dev1 = SimpleAgent("dev1")
        devices = {"dev1": dev1}

        actions = {"dev1": np.array([1.0, 2.0])}

        # Should not raise, but also doesn't apply actions (env handles this)
        action_proto.apply_actions(
            actions=actions,
            subordinates=devices,
            mode="distributed"
        )

        # In distributed mode, actions not applied directly
        assert dev1.last_action is None

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

        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}, "dev2": {}},
            coordinator_action=None,
            mode="centralized",
            context=context
        )

        # No messages, no actions
        assert len(dev1.mailbox) == 0
        assert len(dev2.mailbox) == 0
        assert dev1.last_action is None
        assert dev2.last_action is None

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

        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}, "dev2": {}},
            coordinator_action=65.0,  # New price
            mode="centralized",
            context=context
        )

        # Verify messages sent
        assert len(dev1.mailbox) == 1
        assert dev1.mailbox[0].payload["price"] == 65.0
        assert dev1.mailbox[0].payload["type"] == "price_signal"

        assert len(dev2.mailbox) == 1
        assert dev2.mailbox[0].payload["price"] == 65.0

        # Verify no direct action (decentralized)
        assert dev1.last_action is None
        assert dev2.last_action is None

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

        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}, "dev2": {}},
            coordinator_action=action_dict,
            mode="centralized",
            context=context
        )

        # Verify messages sent
        assert len(dev1.mailbox) == 1
        assert dev1.mailbox[0].payload["type"] == "setpoint_command"

        # Verify actions applied directly (centralized)
        np.testing.assert_array_equal(dev1.last_action, np.array([1.5, 2.5]))
        np.testing.assert_array_equal(dev2.last_action, np.array([3.5]))

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

        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}, "dev2": {}},
            coordinator_action=action_array,
            mode="centralized",
            context=context
        )

        # Verify messages contain decomposed setpoints
        assert len(dev1.mailbox) == 1
        assert len(dev2.mailbox) == 1

        # Verify actions decomposed and applied
        np.testing.assert_array_equal(dev1.last_action, np.array([10.0, 20.0]))
        np.testing.assert_array_equal(dev2.last_action, np.array([30.0]))

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
# Mode-Specific Tests
# =============================================================================

class TestProtocolModes:
    """Test protocol behavior in different modes."""

    def test_centralized_mode_applies_actions_directly(self):
        """Test centralized mode applies actions via act()."""
        protocol = SetpointProtocol()

        dev1 = SimpleAgent("dev1", action_dim=2)
        devices = {"dev1": dev1}

        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}},
            coordinator_action={"dev1": np.array([5.0, 6.0])},
            mode="centralized",
            context=context
        )

        # Action applied directly
        assert dev1.last_action is not None
        np.testing.assert_array_equal(dev1.last_action, np.array([5.0, 6.0]))

    def test_distributed_mode_sends_messages_only(self):
        """Test distributed mode sends messages but doesn't apply actions."""
        protocol = SetpointProtocol()

        dev1 = SimpleAgent("dev1", action_dim=2)
        devices = {"dev1": dev1}

        context = {
            "subordinates": devices,
            "coordinator_id": "grid",
            "timestamp": 0.0
        }

        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}},
            coordinator_action={"dev1": np.array([5.0, 6.0])},
            mode="distributed",
            context=context
        )

        # Message sent
        assert len(dev1.mailbox) == 1

        # Action NOT applied directly (env handles this in distributed mode)
        assert dev1.last_action is None

    def test_price_protocol_works_in_both_modes(self):
        """Test PriceSignalProtocol works identically in both modes."""
        protocol = PriceSignalProtocol(initial_price=50.0)

        # Test centralized
        dev1 = SimpleAgent("dev1")
        devices = {"dev1": dev1}
        context = {"subordinates": devices, "coordinator_id": "grid", "timestamp": 0.0}

        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}},
            coordinator_action=70.0,
            mode="centralized",
            context=context
        )

        assert len(dev1.mailbox) == 1
        assert dev1.mailbox[0].payload["price"] == 70.0
        assert dev1.last_action is None  # Decentralized action

        # Test distributed
        dev2 = SimpleAgent("dev2")
        devices2 = {"dev2": dev2}
        context2 = {"subordinates": devices2, "coordinator_id": "grid", "timestamp": 1.0}

        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev2": {}},
            coordinator_action=80.0,
            mode="distributed",
            context=context2
        )

        assert len(dev2.mailbox) == 1
        assert dev2.mailbox[0].payload["price"] == 80.0
        assert dev2.last_action is None  # Decentralized action


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
        protocol.coordinate(
            coordinator_state={},
            subordinate_states={},
            coordinator_action=50.0,
            mode="centralized",
            context=context
        )

    def test_protocol_with_none_action(self):
        """Test protocol handles None coordinator action."""
        protocol = SetpointProtocol()

        dev1 = SimpleAgent("dev1")
        devices = {"dev1": dev1}
        context = {"subordinates": devices, "coordinator_id": "grid", "timestamp": 0.0}

        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}},
            coordinator_action=None,
            mode="centralized",
            context=context
        )

        # Should handle gracefully (no action applied)
        assert dev1.last_action is None

    def test_setpoint_protocol_with_missing_subordinate(self):
        """Test setpoint protocol handles subordinate mismatch."""
        protocol = SetpointProtocol()

        dev1 = SimpleAgent("dev1")
        devices = {"dev1": dev1}

        # Action includes non-existent dev2
        action = {"dev1": np.array([1.0]), "dev2": np.array([2.0])}

        context = {"subordinates": devices, "coordinator_id": "grid", "timestamp": 0.0}

        # Should handle gracefully (skip dev2)
        protocol.coordinate(
            coordinator_state={},
            subordinate_states={"dev1": {}},
            coordinator_action=action,
            mode="centralized",
            context=context
        )

        # dev1 action applied
        np.testing.assert_array_equal(dev1.last_action, np.array([1.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

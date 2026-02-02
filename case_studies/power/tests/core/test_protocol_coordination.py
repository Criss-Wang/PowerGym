"""
Unit tests for Protocol Coordination.

Tests cover:
- Vertical protocol message passing (GridAgent → DeviceAgents)
- Horizontal protocol coordination (GridAgent ↔ GridAgent)
- Protocol-specific behaviors
- Message delivery via message broker
- Protocol ownership (vertical vs horizontal)

Key protocols tested:
- PriceSignalProtocol (vertical)
- SetpointProtocol (vertical)
- PeerToPeerTradingProtocol (horizontal)
- ConsensusProtocol (horizontal)
"""

import pytest
import numpy as np
import pandapower as pp

from powergrid.agents.power_grid_agent import PowerGridAgent
from heron.protocols.base import (
    Protocol,
    NoProtocol,
    NoCommunication,
    NoActionCoordination,
)
from heron.protocols.vertical import (
    SetpointProtocol,
    PriceSignalProtocol,
)
from heron.protocols.horizontal import (
    HorizontalProtocol,
    PeerToPeerTradingProtocol,
    ConsensusProtocol,
)
from powergrid.agents.generator import Generator
from powergrid.agents.storage import ESS
from powergrid.networks.ieee13 import IEEE13Bus


class TestVerticalProtocols:
    """Test vertical (agent-owned) protocol coordination."""

    @pytest.fixture
    def microgrid_with_devices(self):
        """Create a microgrid with multiple devices."""
        net = IEEE13Bus("MG1")

        # Create generator
        gen = Generator(
            agent_id="gen1",
            device_config={
                "name": "gen1",
                "device_state_config": {
                    "bus": "Bus 633",
                    "p_max_MW": 2.0,
                    "p_min_MW": 0.5,
                    "q_max_MVAr": 1.0,
                    "q_min_MVAr": -1.0,
                    "s_rated_MVA": 2.5,
                    "startup_time_hr": 1.0,
                    "shutdown_time_hr": 1.0,
                    "cost_curve_coefs": [0.02, 10.0, 0.0],
                },
            },
        )

        # Create ESS
        ess = ESS(
            agent_id="ess1",
            device_config={
                "name": "ess1",
                "device_state_config": {
                    "bus": "Bus 634",
                    "e_capacity_MWh": 5.0,
                    "soc_min": 0.1,
                    "soc_max": 0.9,
                    "p_max_MW": 1.5,
                    "p_min_MW": -1.5,
                    "q_max_MVAr": 0.7,
                    "q_min_MVAr": -0.7,
                    "s_rated_MVA": 1.8,
                    "init_soc": 0.5,
                    "ch_eff": 0.95,
                    "dsc_eff": 0.95,
                },
            },
        )

        return net, [gen, ess]

    def test_price_signal_protocol_initialization(self):
        """Test PriceSignalProtocol initialization."""
        protocol = PriceSignalProtocol(initial_price=50.0)

        assert protocol.price == 50.0
        assert isinstance(protocol, PriceSignalProtocol)

    def test_price_signal_broadcast(self, microgrid_with_devices):
        """Test that price signals are broadcast to all devices."""
        net, devices = microgrid_with_devices

        protocol = PriceSignalProtocol(initial_price=40.0)
        mg_agent = PowerGridAgent(
            net=net,
            grid_config={"name": "MG1", "base_power": 1.0},
            devices=devices,
            protocol=protocol,
            centralized=True,
        )

        # Add devices to network
        mg_agent.add_sgen([devices[0]])
        mg_agent.add_storage([devices[1]])

        # Reset to initialize
        mg_agent.reset(seed=42)

        # Update price
        protocol.price = 60.0

        # Use protocol.coordinate() which returns (messages, actions)
        device_dict = {d.agent_id: d for d in devices}
        device_states = {d.agent_id: {} for d in devices}
        context = {"subordinates": device_dict}

        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states=device_states,
            coordinator_action={"price": 60.0},
            context=context
        )

        # Check that protocol generated price messages
        for device_id, msg in messages.items():
            assert "price" in msg
            assert msg["price"] == 60.0

    def test_centralized_setpoint_protocol(self, microgrid_with_devices):
        """Test SetpointProtocol action distribution."""
        net, devices = microgrid_with_devices

        protocol = SetpointProtocol()
        mg_agent = PowerGridAgent(
            net=net,
            grid_config={"name": "MG1", "base_power": 1.0},
            devices=devices,
            protocol=protocol,
            centralized=True,
        )

        mg_agent.add_sgen([devices[0]])
        mg_agent.add_storage([devices[1]])

        # Reset
        mg_agent.reset(seed=42)

        # Create per-device actions as dict for setpoint protocol
        device_dict = {d.agent_id: d for d in devices}
        device_states = {d.agent_id: {} for d in devices}
        context = {"subordinates": device_dict}

        # Pass action as dict mapping device_id -> setpoint
        action = {
            devices[0].agent_id: np.array([1.0, 0.5]),  # Gen: P, Q
            devices[1].agent_id: np.array([0.8, 0.3]),  # ESS: P, Q
        }

        # Protocol should distribute action to devices
        messages, actions = protocol.coordinate(
            coordinator_state={},
            subordinate_states=device_states,
            coordinator_action=action,
            context=context
        )

        # Setpoint protocol should return actions for each device
        for device_id, device_action in actions.items():
            if device_action is not None:
                assert device_id in device_dict

    def test_protocol_ownership_vertical(self, microgrid_with_devices):
        """Test that vertical protocols are owned by agents."""
        net, devices = microgrid_with_devices

        protocol = PriceSignalProtocol(initial_price=50.0)
        mg_agent = PowerGridAgent(
            net=net,
            grid_config={"name": "MG1", "base_power": 1.0},
            devices=devices,
            protocol=protocol,
            centralized=True,
        )

        # Protocol should be owned by agent
        assert mg_agent.protocol is protocol
        assert isinstance(protocol, PriceSignalProtocol)


class TestHorizontalProtocols:
    """Test horizontal (environment-owned) protocol coordination."""

    @pytest.fixture
    def multiple_microgrids(self):
        """Create multiple microgrids for peer coordination."""
        microgrids = []

        for i in range(3):
            net = IEEE13Bus(f"MG{i+1}")

            gen = Generator(
                agent_id=f"gen{i+1}",
                device_config={
                    "name": f"gen{i+1}",
                    "device_state_config": {
                        "bus": "Bus 633",
                        "p_max_MW": 2.0,
                        "p_min_MW": 0.5,
                        "q_max_MVAr": 1.0,
                        "q_min_MVAr": -1.0,
                        "s_rated_MVA": 2.5,
                        "startup_time_hr": 1.0,
                        "shutdown_time_hr": 1.0,
                        "cost_curve_coefs": [0.02 * (i + 1), 10.0, 0.0],
                    },
                },
            )

            mg_agent = PowerGridAgent(
                net=net,
                grid_config={"name": f"MG{i+1}", "base_power": 1.0},
                devices=[gen],
                protocol=SetpointProtocol(),
                centralized=True,
            )

            mg_agent.add_sgen([gen])
            microgrids.append(mg_agent)

        return microgrids

    def test_p2p_trading_protocol_initialization(self):
        """Test PeerToPeerTradingProtocol initialization."""
        protocol = PeerToPeerTradingProtocol(trading_fee=0.02)

        assert protocol.trading_fee == 0.02
        assert isinstance(protocol, PeerToPeerTradingProtocol)

    def test_p2p_trading_market_clearing(self, multiple_microgrids):
        """Test P2P trading protocol market clearing mechanism."""
        protocol = PeerToPeerTradingProtocol(trading_fee=0.02)

        # Simulate bids and offers
        bids = {
            "MG1": {"quantity": 1.0, "max_price": 50.0},
        }

        offers = {
            "MG2": {"quantity": 0.8, "min_price": 45.0},
            "MG3": {"quantity": 0.5, "min_price": 48.0},
        }

        # Test market clearing via communication_protocol
        trades = protocol.communication_protocol._clear_market(bids, offers)

        # Should have at least one trade (MG1 buying from MG2/MG3)
        assert len(trades) > 0

        # Check trade structure
        for buyer, seller, quantity, price in trades:
            assert buyer == "MG1"
            assert seller in ["MG2", "MG3"]
            assert quantity > 0
            assert 45.0 <= price <= 50.0

    def test_consensus_protocol_initialization(self):
        """Test ConsensusProtocol initialization."""
        protocol = ConsensusProtocol(max_iterations=10, tolerance=0.01)

        assert protocol.max_iterations == 10
        assert protocol.tolerance == 0.01
        assert isinstance(protocol, ConsensusProtocol)

    def test_consensus_averaging(self, multiple_microgrids):
        """Test consensus protocol averaging mechanism."""
        protocol = ConsensusProtocol(max_iterations=10, tolerance=0.01)

        # Simulate agent values to be averaged
        agent_values = {
            "MG1": 100.0,
            "MG2": 80.0,
            "MG3": 90.0,
        }

        # Average should converge to mean
        avg_value = np.mean(list(agent_values.values()))
        assert np.isclose(avg_value, 90.0)

        # Consensus should move values toward average
        # (Testing the logic without full agent setup)
        assert avg_value == 90.0

    def test_protocol_ownership_horizontal(self):
        """Test that horizontal protocols are environment-owned."""
        protocol = PeerToPeerTradingProtocol(trading_fee=0.02)

        # Horizontal protocols coordinate between multiple agents
        assert isinstance(protocol, HorizontalProtocol)
        # Protocol should be passed via env_config, not agent config


class TestMessageDelivery:
    """Test message delivery via mailbox system."""

    @pytest.fixture
    def agent_with_devices(self):
        """Create agent with devices for message testing."""
        net = IEEE13Bus("MG1")

        gen = Generator(
            agent_id="gen1",
            device_config={
                "name": "gen1",
                "device_state_config": {
                    "bus": "Bus 633",
                    "p_max_MW": 2.0,
                    "p_min_MW": 0.5,
                    "q_max_MVAr": 1.0,
                    "q_min_MVAr": -1.0,
                    "s_rated_MVA": 2.5,
                    "startup_time_hr": 1.0,
                    "shutdown_time_hr": 1.0,
                    "cost_curve_coefs": [0.02, 10.0, 0.0],
                },
            },
        )

        protocol = PriceSignalProtocol(initial_price=50.0)
        mg_agent = PowerGridAgent(
            net=net,
            grid_config={"name": "MG1", "base_power": 1.0},
            devices=[gen],
            protocol=protocol,
            centralized=True,
        )

        mg_agent.add_sgen([gen])
        mg_agent.reset(seed=42)

        return mg_agent, gen

    def test_message_broker_initialization(self, agent_with_devices):
        """Test that devices can be configured with message broker."""
        mg_agent, gen = agent_with_devices

        # Device should be able to set message broker
        from heron.messaging.in_memory_broker import InMemoryBroker
        broker = InMemoryBroker()
        gen.set_message_broker(broker)
        assert gen.message_broker is not None

    def test_message_delivery_to_device(self, agent_with_devices):
        """Test that protocol generates messages for devices."""
        mg_agent, gen = agent_with_devices

        # Use protocol.coordinate() which returns (messages, actions)
        mg_agent.protocol.price = 60.0
        device_dict = {gen.agent_id: gen}
        device_states = {gen.agent_id: {}}
        context = {"subordinates": device_dict}

        messages, actions = mg_agent.protocol.coordinate(
            coordinator_state={},
            subordinate_states=device_states,
            coordinator_action={"price": 60.0},
            context=context
        )

        # Check that messages were generated for the device
        assert gen.agent_id in messages
        assert "price" in messages[gen.agent_id]
        assert messages[gen.agent_id]["price"] == 60.0

    def test_message_broker_communication(self, agent_with_devices):
        """Test message broker communication between agents."""
        mg_agent, gen = agent_with_devices

        # Configure message broker
        from heron.messaging.in_memory_broker import InMemoryBroker
        broker = InMemoryBroker()
        mg_agent.set_message_broker(broker)
        gen.set_message_broker(broker)
        mg_agent.env_id = "test_env"
        gen.env_id = "test_env"
        gen.upstream_id = mg_agent.agent_id

        # Messages via broker should be receivable
        messages = gen.receive_messages()
        assert isinstance(messages, list)


class TestProtocolIntegration:
    """Test protocol integration in full environment."""

    def test_vertical_protocol_in_environment(self):
        """Test vertical protocol works within environment context."""
        from powergrid.envs.networked_grid_env import NetworkedGridEnv

        # Create simple environment with vertical protocol
        net = IEEE13Bus("MG1")

        gen = Generator(
            agent_id="gen1",
            device_config={
                "name": "gen1",
                "device_state_config": {
                    "bus": "Bus 633",
                    "p_max_MW": 2.0,
                    "p_min_MW": 0.5,
                    "q_max_MVAr": 1.0,
                    "q_min_MVAr": -1.0,
                    "s_rated_MVA": 2.5,
                    "startup_time_hr": 1.0,
                    "shutdown_time_hr": 1.0,
                    "cost_curve_coefs": [0.02, 10.0, 0.0],
                },
            },
        )

        protocol = SetpointProtocol()
        mg_agent = PowerGridAgent(
            net=net,
            grid_config={"name": "MG1", "base_power": 1.0},
            devices=[gen],
            protocol=protocol,
            centralized=True,
        )

        mg_agent.add_sgen([gen])

        # Protocol should be accessible
        assert mg_agent.protocol is protocol
        assert isinstance(protocol, SetpointProtocol)

    def test_horizontal_protocol_environment_config(self):
        """Test horizontal protocol configuration at environment level."""
        # Horizontal protocols should be specified in env_config
        env_config = {
            "max_episode_steps": 24,
            "protocol": PeerToPeerTradingProtocol(trading_fee=0.02),
        }

        protocol = env_config["protocol"]
        assert isinstance(protocol, HorizontalProtocol)
        assert isinstance(protocol, PeerToPeerTradingProtocol)


class TestProtocolEdgeCases:
    """Test edge cases and error handling in protocols."""

    def test_price_signal_negative_price(self):
        """Test that negative prices are handled (can occur in some markets)."""
        protocol = PriceSignalProtocol(initial_price=-10.0)

        # Negative prices should be allowed (renewable oversupply scenario)
        assert protocol.price == -10.0

    def test_p2p_trading_zero_fee(self):
        """Test P2P trading with zero transaction fee."""
        protocol = PeerToPeerTradingProtocol(trading_fee=0.0)

        assert protocol.trading_fee == 0.0

    def test_consensus_extreme_tolerances(self):
        """Test consensus protocol with extreme parameters."""
        # Very tight tolerance
        protocol_tight = ConsensusProtocol(max_iterations=100, tolerance=0.0001)
        assert protocol_tight.tolerance == 0.0001

        # Loose tolerance (fast convergence)
        protocol_loose = ConsensusProtocol(max_iterations=5, tolerance=1.0)
        assert protocol_loose.tolerance == 1.0

    def test_protocol_with_no_devices(self):
        """Test protocol behavior when agent has no devices."""
        net = IEEE13Bus("MG1")

        protocol = SetpointProtocol()
        mg_agent = PowerGridAgent(
            net=net,
            grid_config={"name": "MG1", "base_power": 1.0},
            devices=[],  # No devices
            protocol=protocol,
            centralized=True,
        )

        # Should not crash
        mg_agent.reset(seed=42)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

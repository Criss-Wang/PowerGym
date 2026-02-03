"""Coordination protocols for HERON multi-agent systems.

This module provides protocol implementations for agent coordination:

Base Protocols:
- Protocol: Abstract base for all protocols
- NoProtocol: No-op protocol (agents act independently)

Vertical Protocols (hierarchical coordination):
- VerticalProtocol: Base for top-down coordination
- SetpointProtocol: Setpoint-based control
- PriceSignalProtocol: Price signal-based coordination
- SystemProtocol: System-level coordination (L3 -> L2)

Horizontal Protocols (peer coordination):
- HorizontalProtocol: Base for peer-to-peer coordination
- PeerToPeerTradingProtocol: P2P trading between agents
- ConsensusProtocol: Distributed consensus
- NoHorizontalProtocol: No horizontal coordination
"""

from heron.protocols.base import (
    Protocol,
    NoProtocol,
    CommunicationProtocol,
    ActionProtocol,
    NoCommunication,
    NoActionCoordination,
)
from heron.protocols.vertical import (
    VerticalProtocol,
    SetpointProtocol,
    PriceSignalProtocol,
    SetpointCommunicationProtocol,
    CentralizedActionProtocol,
    PriceCommunicationProtocol,
    DecentralizedActionProtocol,
    SystemProtocol,
    SystemCommunicationProtocol,
)
from heron.protocols.horizontal import (
    HorizontalProtocol,
    PeerToPeerTradingProtocol,
    ConsensusProtocol,
    NoHorizontalProtocol,
    TradingCommunicationProtocol,
    TradingActionProtocol,
    ConsensusCommunicationProtocol,
    ConsensusActionProtocol,
)

__all__ = [
    # Base
    "Protocol",
    "NoProtocol",
    "CommunicationProtocol",
    "ActionProtocol",
    "NoCommunication",
    "NoActionCoordination",
    # Vertical
    "VerticalProtocol",
    "SetpointProtocol",
    "PriceSignalProtocol",
    "SetpointCommunicationProtocol",
    "CentralizedActionProtocol",
    "PriceCommunicationProtocol",
    "DecentralizedActionProtocol",
    "SystemProtocol",
    "SystemCommunicationProtocol",
    # Horizontal
    "HorizontalProtocol",
    "PeerToPeerTradingProtocol",
    "ConsensusProtocol",
    "NoHorizontalProtocol",
    "TradingCommunicationProtocol",
    "TradingActionProtocol",
    "ConsensusCommunicationProtocol",
    "ConsensusActionProtocol",
]

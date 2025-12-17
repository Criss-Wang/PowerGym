Protocols
=========

.. toctree::
   :hidden:

   Vertical Protocols <vertical>
   Horizontal Protocols <horizontal>
   Custom Protocols <custom>

Coordination protocols define how agents communicate and coordinate.

Protocol Types
--------------

- :doc:`vertical` - Parent-subordinate coordination
- :doc:`horizontal` - Peer-to-peer coordination
- :doc:`custom` - Creating custom protocols

Overview
--------

PowerGrid protocols compose two coordination components:

**Protocol Architecture**

Each protocol consists of:

- **CommunicationProtocol**: Defines coordination messages (WHAT to communicate)
- **ActionProtocol**: Defines action coordination (HOW to coordinate actions)

This separation allows flexible mixing of communication strategies with action strategies.

**Vertical Protocols**

Parent → subordinate coordination (GridAgent → DeviceAgent)

- Price signals (PriceSignalProtocol)
- Power setpoints (SetpointProtocol)
- Custom vertical coordination

**Horizontal Protocols**

Peer ↔ peer coordination (GridAgent ↔ GridAgent)

- P2P energy trading (PeerToPeerTradingProtocol)
- Consensus algorithms (ConsensusProtocol)
- Custom horizontal coordination

See :doc:`/api/core/protocols` for full protocol guide and API reference.

Quick Example
-------------

.. code-block:: python

   from powergrid.core.protocols import PriceSignalProtocol, SetpointProtocol
   from powergrid.agents.grid_agent import PowerGridAgent

   # Price-based coordination (decentralized action)
   price_agent = PowerGridAgent(
       agent_id='MG1',
       protocol=PriceSignalProtocol(initial_price=50.0),
       ...
   )

   # Setpoint-based coordination (centralized action)
   setpoint_agent = PowerGridAgent(
       agent_id='MG2',
       protocol=SetpointProtocol(),
       ...
   )

   # Protocols automatically handle both centralized and distributed modes

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

PowerGrid uses two types of coordination protocols:

**Vertical Protocols**

Parent → subordinate coordination (GridAgent → DeviceAgent)

- Price signals
- Power setpoints
- ADMM optimization

**Horizontal Protocols**

Peer ↔ peer coordination (GridAgent ↔ GridAgent)

- P2P energy trading
- Consensus algorithms
- Auction mechanisms

See :doc:`/api/core/protocols` for full protocol guide and API reference.

Quick Example
-------------

.. code-block:: python

   from powergrid.core.protocols import PriceSignalProtocol
   from powergrid.agents import GridAgent

   # Create GridAgent with price signal protocol
   agent = GridAgent(
       agent_id='MG1',
       subordinates=devices,
       vertical_protocol=PriceSignalProtocol(initial_price=50.0)
   )

   # Protocol automatically coordinates devices
   # Devices respond by optimizing to the price signal

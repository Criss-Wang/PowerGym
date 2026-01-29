PowerGrid Case Study
====================

.. toctree::
   :hidden:

   api/index

A multi-agent reinforcement learning implementation for power grid control built on HERON.

Overview
--------

PowerGrid provides:

- **Hierarchical Control**: GridAgents coordinate DeviceAgents (batteries, generators, solar)
- **IEEE Test Networks**: Standard 13, 34, 123-bus distribution networks
- **Coordination Protocols**: Price signals, setpoints, P2P trading
- **MARL Training**: Integration with RLlib for MAPPO training

Quick Start
-----------

.. code-block:: python

   from powergrid.envs import MultiAgentMicrogrids
   from powergrid.agents import PowerGridAgent

   # Create multi-agent environment
   env = MultiAgentMicrogrids(config={
       'network': 'ieee13',
       'num_microgrids': 2,
       'mode': 'centralized'
   })

   obs, info = env.reset()
   for step in range(96):
       actions = {agent: policy(o) for agent, o in obs.items()}
       obs, rewards, dones, truncs, info = env.step(actions)

Use Cases
---------

**Microgrid Energy Management**
   Optimize distributed energy resources (DER) in microgrids including batteries, solar panels, and generators.

**Peer-to-Peer Energy Trading**
   Enable local energy markets where microgrids trade surplus energy.

**Voltage Regulation**
   Maintain voltage levels across the distribution network within safe limits.

**Frequency Control**
   Coordinate distributed generation to maintain grid frequency at 60 Hz.

Documentation
-------------

- :doc:`api/index` - Complete API reference
- :doc:`api/agents` - PowerGridAgent, DeviceAgent, and device types
- :doc:`api/envs` - NetworkedGridEnv, MultiAgentMicrogrids
- :doc:`api/features` - Electrical, Storage, Thermal features
- :doc:`api/networks` - IEEE test networks

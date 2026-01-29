API Reference
=============

.. toctree::
   :hidden:

   HERON Core <heron/index>
   Environments <envs>
   Agents <agents>
   Features <features>
   Networks <networks>
   Optimization <optimization>
   Setups <setups>
   Utils <utils>
   Messaging <messaging>
   Core <core/index>

Complete API documentation for HERON framework and PowerGrid case study.

HERON Core Framework
--------------------

The domain-agnostic multi-agent reinforcement learning framework.

- :doc:`heron/index` - Core HERON modules (agents, protocols, messaging, state/action)

PowerGrid Case Study
--------------------

Power systems implementation built on HERON.

**Agents & Environments**

- :doc:`envs` - Environment API (NetworkedGridEnv, MultiAgentMicrogrids)
- :doc:`agents` - Agent API (PowerGridAgent, DeviceAgent, Generator, Storage, Transformer)

**State & Features**

- :doc:`core/state` - Power grid state classes
- :doc:`features` - Feature extraction API (electrical, storage, thermal, etc.)

**Networks & Optimization**

- :doc:`networks` - IEEE test feeders (13, 34, 123-bus) and CIGRE networks
- :doc:`optimization` - MISOCP power flow solver

**Utilities**

- :doc:`setups` - Environment setup and configuration loading
- :doc:`utils` - Cost, safety, phase utilities
- :doc:`messaging` - Message broker (uses HERON messaging)
- :doc:`core/index` - Protocols guide and state reference

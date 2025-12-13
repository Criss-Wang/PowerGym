Architecture
============

Deep dive into PowerGrid 2.0's system design and implementation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   message_broker
   agents
   devices

**Architecture Components**

- :doc:`overview` - System architecture and design principles
- :doc:`message_broker` - Message-based communication system
- :doc:`agents` - Agent hierarchy and coordination
- :doc:`devices` - Device models and physics

**Key Concepts**

The architecture is built around several core abstractions:

- **Hierarchical Agents**: GridAgent (level 2) coordinates DeviceAgents (level 1)
- **Feature-Based State**: Composable FeatureProviders with visibility control
- **Flexible Actions**: Mixed continuous/discrete with normalization
- **Message Passing**: Realistic distributed execution with message broker
- **Dual Execution Modes**: Centralized (fast prototyping) and Distributed (realistic deployment)

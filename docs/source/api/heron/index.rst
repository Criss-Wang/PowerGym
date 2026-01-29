HERON Core Framework
====================

.. toctree::
   :hidden:

   agents
   core
   protocols
   messaging
   envs

**HERON** (Hierarchical Environments for Realistic Observability in Networks) is a domain-agnostic
multi-agent reinforcement learning framework for building hierarchical control systems.

Key Concepts
------------

**Hierarchical Agents**
   Three-level agent hierarchy: System (global) → Coordinator (mid-level) → Field (leaf).
   Each level has distinct responsibilities and observability.

**Feature-Based State**
   Composable ``FeatureProvider`` classes with visibility rules (public, owner, system, upper_level).
   Enables flexible state representation across different domains.

**Coordination Protocols**
   - **Vertical**: Top-down coordination (SetpointProtocol, PriceSignalProtocol)
   - **Horizontal**: Peer-to-peer coordination (P2PTradingProtocol, ConsensusProtocol)

**Dual Execution Modes**
   Same API for centralized (fast training) and distributed (realistic deployment) execution.

Modules
-------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`agents`
     - Hierarchical agent abstractions (FieldAgent, CoordinatorAgent, SystemAgent, ProxyAgent)
   * - :doc:`core`
     - Core data structures (Action, Observation, State, Feature, Policies)
   * - :doc:`protocols`
     - Vertical and horizontal coordination protocols
   * - :doc:`messaging`
     - Message broker interface (InMemoryBroker, extensible to Kafka/Redis)
   * - :doc:`envs`
     - Base environment interfaces for MARL

Quick Start
-----------

.. code-block:: python

   from heron.agents import CoordinatorAgent, FieldAgent
   from heron.protocols.vertical import SetpointProtocol
   from heron.messaging.memory import InMemoryBroker

   # Create message broker
   broker = InMemoryBroker()

   # Create field agents (leaf level)
   device1 = FieldAgent(agent_id='device_1', level=1, broker=broker)
   device2 = FieldAgent(agent_id='device_2', level=1, broker=broker)

   # Create coordinator with protocol
   coordinator = CoordinatorAgent(
       agent_id='coordinator_1',
       level=2,
       subordinates=[device1, device2],
       protocol=SetpointProtocol(),
       broker=broker
   )

   # Execution loop
   obs = coordinator.observe(global_state)
   action = coordinator.act(obs)

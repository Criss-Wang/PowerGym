Core
====

Core abstractions for HERON's modular state/action/observation system and coordination protocols.

.. note::

   Core abstractions (Action, Observation, Feature, Policies) are part of the HERON framework.
   See :doc:`../heron/core` for the domain-agnostic base classes.

   PowerGrid-specific state classes are documented below.

**Key Components:**

- **State**: Power grid state representation (devices and network)
- **Action**: Flexible continuous/discrete actions (see :doc:`../heron/core`)
- **Observation**: Structured observations (see :doc:`../heron/core`)
- **Protocols**: Coordination mechanisms (see :doc:`../heron/protocols`)
- **Policies**: Decision-making interfaces (see :doc:`../heron/core`)

.. toctree::
   :maxdepth: 1

   state
   protocols

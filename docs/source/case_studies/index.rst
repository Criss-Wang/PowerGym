Case Studies
============

.. toctree::
   :hidden:

   power/index

HERON includes production-ready case study implementations demonstrating hierarchical multi-agent control.

Available Case Studies
----------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Power Grid Control
      :link: power/index
      :link-type: doc

      Multi-agent microgrid control with IEEE test networks, DER coordination,
      and MAPPO training. Includes 70+ trained model checkpoints.

      - GridAgents coordinate DeviceAgents
      - IEEE 13, 34, 123-bus networks
      - Price signal and setpoint protocols
      - P2P energy trading

Building Your Own Case Study
----------------------------

HERON provides the building blocks to create new domain-specific implementations:

1. **Define Agents**: Extend ``FieldAgent`` and ``CoordinatorAgent`` for your domain
2. **Create Features**: Implement ``FeatureProvider`` classes for domain state
3. **Choose Protocols**: Use built-in or create custom coordination protocols
4. **Build Environment**: Create a ``ParallelEnv`` using HERON's base classes

See the :doc:`/developer/index` guide for detailed instructions.

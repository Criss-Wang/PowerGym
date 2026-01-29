Features
========

The features module provides modular feature providers for composing agent state representations.

Each feature encapsulates a cohesive set of observable/controllable attributes with:

- **Vectorization**: Convert to numpy arrays for ML
- **Visibility Rules**: Control who can observe (public, owner, system, upper_level)
- **Serialization**: Support for communication and logging
- **Update Methods**: Flexible state updates via keyword arguments

Connection Features
-------------------

Bus connectivity features.

.. automodule:: powergrid.core.features.connection
   :members:
   :undoc-members:
   :show-inheritance:

Electrical Features
-------------------

Active power (P), reactive power (Q), and voltage features.

.. automodule:: powergrid.core.features.electrical
   :members:
   :undoc-members:
   :show-inheritance:

Power Limits
------------

Generation and absorption power limit features.

.. automodule:: powergrid.core.features.power_limits
   :members:
   :undoc-members:
   :show-inheritance:

Status Features
---------------

Device status and operational state features.

.. automodule:: powergrid.core.features.status
   :members:
   :undoc-members:
   :show-inheritance:

Storage Features
----------------

State of charge (SOC) and energy capacity features for energy storage.

.. automodule:: powergrid.core.features.storage
   :members:
   :undoc-members:
   :show-inheritance:

Tap Changer Features
--------------------

Transformer tap position features.

.. automodule:: powergrid.core.features.tap_changer
   :members:
   :undoc-members:
   :show-inheritance:

Thermal Features
----------------

Line thermal loading constraint features.

.. automodule:: powergrid.core.features.thermal
   :members:
   :undoc-members:
   :show-inheritance:

VAR Features
------------

Reactive power (VAR) control features.

.. automodule:: powergrid.core.features.var
   :members:
   :undoc-members:
   :show-inheritance:

Step State Features
-------------------

Timestep and episode state features.

.. automodule:: powergrid.core.features.step_state
   :members:
   :undoc-members:
   :show-inheritance:

Inverter Features
-----------------

Renewable inverter constraint features.

.. automodule:: powergrid.core.features.inverter
   :members:
   :undoc-members:
   :show-inheritance:

Network Features
----------------

Network-level features for GridAgent state representation (bus voltages, line flows).

.. automodule:: powergrid.core.features.network
   :members:
   :undoc-members:
   :show-inheritance:

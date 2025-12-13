Features
========

The features module provides modular feature providers for composing agent state representations.

Each feature encapsulates a cohesive set of observable/controllable attributes with:

- **Vectorization**: Convert to numpy arrays for ML
- **Visibility Rules**: Control who can observe (public, owner, system, upper_level)
- **Serialization**: Support for communication and logging
- **Update Methods**: Flexible state updates via keyword arguments

Base Feature Provider
----------------------

.. automodule:: powergrid.features.base
   :members:
   :undoc-members:
   :show-inheritance:

Connection Features
-------------------

.. automodule:: powergrid.features.connection
   :members:
   :undoc-members:
   :show-inheritance:

Electrical Features
-------------------

.. automodule:: powergrid.features.electrical
   :members:
   :undoc-members:
   :show-inheritance:

Power Limits
------------

.. automodule:: powergrid.features.power_limits
   :members:
   :undoc-members:
   :show-inheritance:

Status Features
---------------

.. automodule:: powergrid.features.status
   :members:
   :undoc-members:
   :show-inheritance:

Storage Features
----------------

.. automodule:: powergrid.features.storage
   :members:
   :undoc-members:
   :show-inheritance:

Tap Changer Features
--------------------

.. automodule:: powergrid.features.tap_changer
   :members:
   :undoc-members:
   :show-inheritance:

Thermal Features
----------------

.. automodule:: powergrid.features.thermal
   :members:
   :undoc-members:
   :show-inheritance:

VAR Features
------------

.. automodule:: powergrid.features.var
   :members:
   :undoc-members:
   :show-inheritance:

Step State Features
-------------------

.. automodule:: powergrid.features.step_state
   :members:
   :undoc-members:
   :show-inheritance:

Inverter Features
-----------------

.. automodule:: powergrid.features.inverter
   :members:
   :undoc-members:
   :show-inheritance:

Network Features
----------------

Network-level features for GridAgent state representation.

.. automodule:: powergrid.features.network
   :members:
   :undoc-members:
   :show-inheritance:

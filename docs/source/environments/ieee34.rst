IEEE 34-Bus Test Feeder
=======================

A larger distribution system with more complex topology.

Network Characteristics
-----------------------

- **Buses**: 34
- **Voltage**: 24.9 kV primary, 4.16 kV secondary
- **Loads**: Mix of spot and distributed loads
- **Regulators**: Multiple voltage regulators
- **Transformers**: Step-down transformers
- **Total Load**: ~1.8 MW

Microgrid Configuration
-----------------------

Default setup with three microgrids for more complex coordination scenarios.

Usage
-----

.. code-block:: python

   from powergrid.envs.multi_agent import IEEE34MGEnv

   env = IEEE34MGEnv(
       mode='distributed',
       num_microgrids=3
   )

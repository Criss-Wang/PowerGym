CIGRE Medium Voltage Network
============================

The CIGRE MV benchmark network represents a European medium-voltage distribution system.

Network Characteristics
-----------------------

- **Buses**: 15
- **Voltage**: 20 kV
- **Lines**: Mix of overhead and cable
- **Loads**: Residential, commercial, and industrial
- **DER**: High renewable penetration

This network is particularly suitable for studying high renewable integration scenarios.

Usage
-----

.. code-block:: python

   from powergrid.envs.multi_agent import CIGREMVEnv

   env = CIGREMVEnv(
       mode='distributed',
       renewable_penetration=0.6  # 60% renewables
   )

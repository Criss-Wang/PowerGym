PowerGrid API Reference
=======================

.. toctree::
   :hidden:

   envs
   agents
   features
   networks
   optimization
   setups
   utils
   core/index

API documentation for the PowerGrid case study. Located in ``case_studies/power/``.

Agents & Environments
---------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :doc:`envs`
     - NetworkedGridEnv, MultiAgentMicrogrids
   * - :doc:`agents`
     - PowerGridAgent, DeviceAgent, Generator, Storage, Transformer

State & Features
----------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :doc:`core/index`
     - Power grid state classes
   * - :doc:`features`
     - Electrical, Storage, Thermal, Network features

Networks & Optimization
-----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :doc:`networks`
     - IEEE 13, 34, 123-bus and CIGRE test networks
   * - :doc:`optimization`
     - MISOCP power flow solver

Utilities
---------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :doc:`setups`
     - Environment configuration loading
   * - :doc:`utils`
     - Cost, safety, phase utilities

Agents
======

HERON provides a hierarchical agent architecture with three levels: Field (leaf), Coordinator (mid-level), and System (top-level).

Base Agent
----------

.. automodule:: heron.agents.base
   :members:
   :undoc-members:
   :show-inheritance:

Field Agent
-----------

Leaf-level agents that manage individual units/devices.

.. automodule:: heron.agents.field_agent
   :members:
   :undoc-members:
   :show-inheritance:

Coordinator Agent
-----------------

Mid-level agents that coordinate subordinate agents.

.. automodule:: heron.agents.coordinator_agent
   :members:
   :undoc-members:
   :show-inheritance:

System Agent
------------

Top-level global coordinator.

.. automodule:: heron.agents.system_agent
   :members:
   :undoc-members:
   :show-inheritance:

Proxy Agent
-----------

Agent for distributed execution support.

.. automodule:: heron.agents.proxy_agent
   :members:
   :undoc-members:
   :show-inheritance:

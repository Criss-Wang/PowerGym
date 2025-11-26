Custom Protocols
================

Create custom coordination protocols for your use case.

See :doc:`/api/core/protocols` for implementation templates and examples.

Template
--------

.. code-block:: python

   from powergrid.core.protocols import VerticalProtocol

   class MyProtocol(VerticalProtocol):
       def coordinate(self, subordinate_observations, parent_action=None):
           # Your coordination logic
           return signals
